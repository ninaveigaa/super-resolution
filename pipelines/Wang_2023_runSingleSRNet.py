'''
2D super resolution method (single-stage EDSR solver, sem discriminator)
'''

import sys
from pathlib import Path

import tensorflow as tf
from keras import mixed_precision
import tifffile
from sys import stdout
import numpy as np
import os
from glob import glob
import time
import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import Wang_2023_dualSRNetArgs
from src import metrics


AUTOTUNE = tf.data.experimental.AUTOTUNE
print(tf.__version__)

args = Wang_2023_dualSRNetArgs.args()

if args.metricsTracker:
    run_id = metrics.save_args(args, model_name=args.modelName)
    tracker = metrics.MetricsTracker(model_name=args.modelName, run_id=run_id)
    tracker.start_training()
    print(f"[metrics] Tracking enabled. run_id = {run_id}")

gpuList = ','.join([g.strip() for g in args.gpuIDs.split(',') if g.strip()])
args.numGPUs = len(gpuList.split(',')) if gpuList else 0
if 0 < args.numGPUs <= 4:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpuList
elif args.numGPUs == 0:
    print('No GPUs specified; using CPU.')

if args.mixedPrecision:
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)
else:
    policy = mixed_precision.Policy('float32')
    mixed_precision.set_global_policy(policy)
print('Compute dtype: %s' % policy.compute_dtype)
print('Variable dtype: %s' % policy.variable_dtype)

# detect hardware
if args.numGPUs <= 1:
    device = "/gpu:0" if args.numGPUs == 1 else "/cpu:0"
    strategy = tf.distribute.OneDeviceStrategy(device=device)
else:
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))


with strategy.scope():

    # ------------------------------------------------------------------
    # Data / patch extraction helpers
    # ------------------------------------------------------------------

    def createTrainingCubes2(args, HR, LRxy, batchsize, cropsize, scale, n_batches=None):
        # Extracts 2D patches for the (single) 2D generator, cycling between
        # the xy, xz and yz orientations of the volume purely as a data
        # augmentation strategy -- every patch produced is still a 2D image
        # (batch, cropsize, cropsize, 1) fed to a 2D conv network.
        # n_batches: how many cubes to generate. Defaults to args.itersPerEpoch
        # (the original training behavior); pass a smaller explicit value
        # (e.g. args.valNum) to generate validation cubes instead.
        if n_batches is None:
            n_batches = args.itersPerEpoch
        batchLR = np.zeros([batchsize * n_batches, cropsize, cropsize, 1], 'float32')
        batchHR = np.zeros([batchsize * n_batches * scale, cropsize * scale, cropsize * scale, 1], 'float32')
        n = 0
        n2 = 0
        for i in range(n_batches):
            if np.mod(i, 3) == 0:
                x = int(np.floor(np.random.rand() * (LRxy.shape[0] - batchsize)))
                y = int(np.floor(np.random.rand() * (LRxy.shape[1] - cropsize)))
                z = int(np.floor(np.random.rand() * (LRxy.shape[2] - cropsize)))

                block = np.expand_dims(LRxy[x:x + batchsize, y:y + cropsize, z:z + cropsize], 3)
                blockHR = np.expand_dims(
                    HR[x * scale:x * scale + batchsize * scale,
                       y * scale:y * scale + cropsize * scale,
                       z * scale:z * scale + cropsize * scale], 3)

            elif np.mod(i, 3) == 1:
                x = int(np.floor(np.random.rand() * (LRxy.shape[0] - cropsize)))
                y = int(np.floor(np.random.rand() * (LRxy.shape[1] - cropsize)))
                z = int(np.floor(np.random.rand() * (LRxy.shape[2] - batchsize)))

                block = np.expand_dims(LRxy[x:x + cropsize, y:y + cropsize, z:z + batchsize], 3)
                blockHR = np.expand_dims(
                    HR[x * scale:x * scale + cropsize * scale,
                       y * scale:y * scale + cropsize * scale,
                       z * scale:z * scale + batchsize * scale], 3)
                block = np.transpose(block, [2, 0, 1, 3])
                blockHR = np.transpose(blockHR, [2, 0, 1, 3])

            elif np.mod(i, 3) == 2:
                x = int(np.floor(np.random.rand() * (LRxy.shape[0] - cropsize)))
                y = int(np.floor(np.random.rand() * (LRxy.shape[1] - batchsize)))
                z = int(np.floor(np.random.rand() * (LRxy.shape[2] - cropsize)))

                block = np.expand_dims(LRxy[x:x + cropsize, y:y + batchsize, z:z + cropsize], 3)
                blockHR = np.expand_dims(
                    HR[x * scale:x * scale + cropsize * scale,
                       y * scale:y * scale + batchsize * scale,
                       z * scale:z * scale + cropsize * scale], 3)

                block = np.transpose(block, [1, 0, 2, 3])
                blockHR = np.transpose(blockHR, [1, 0, 2, 3])

            batchLR[n:n + batchsize] = block / 127.5 - 1
            batchHR[n2:n2 + batchsize * scale] = blockHR / 127.5 - 1
            n = n + batchsize
            n2 = n2 + batchsize * scale

            stdout.write("\rHR Cube: %d of %d" % (i + 1, n_batches))
            stdout.flush()
        stdout.write("\n")
        return batchHR, batchLR

    def augmentData(image):
        contFactor = (np.random.rand() * 2 - 1) * 0.2 + 1
        brightFactor = (np.random.rand() * 2 - 1) * 0.2 + 1

        image = image * brightFactor
        image = (image - tf.math.reduce_mean(image)) * contFactor + tf.math.reduce_mean(image)
        image = tf.clip_by_value(image, -1, 1)
        return image

    # ------------------------------------------------------------------
    # Architecture: single 2D EDSR generator
    # ------------------------------------------------------------------

    def conv2d(*args, **kwargs):
        return tf.keras.layers.Conv2D(*args, **kwargs)

    class InstanceNormalization(tf.keras.layers.Layer):
        """Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

        def __init__(self, epsilon=1e-5):
            super(InstanceNormalization, self).__init__()
            self.epsilon = epsilon

        def build(self, input_shape):
            self.scale = self.add_weight(
                name='scale', shape=input_shape[-1:],
                initializer=tf.random_normal_initializer(1., 0.02), trainable=True)
            self.offset = self.add_weight(
                name='offset', shape=input_shape[-1:], initializer='zeros', trainable=True)

        def call(self, x):
            mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
            inv = tf.math.rsqrt(variance + self.epsilon)
            normalized = (x - mean) * inv
            return self.scale * normalized + self.offset

    def res_block_EDSR(x_in, filters, kernel, norm_type='instancenorm', apply_norm=False):
        x = conv2d(filters, kernel, padding='same')(x_in)
        x = tf.keras.layers.Activation('relu')(x)
        if apply_norm:
            if norm_type.lower() == 'batchnorm':
                x = tf.keras.layers.BatchNormalization()(x)
            elif norm_type.lower() == 'instancenorm':
                x = InstanceNormalization()(x)
        x = conv2d(filters, kernel, padding='same')(x)
        x = tf.keras.layers.Add()([x_in, x])
        return x

    def upsampleEDSR(x, scale, num_filters, norm_type='instancenorm', apply_norm=False, nameIn=''):
        def upsample_edsr(x, factor, **kwargs):
            x = conv2d(num_filters, 3, padding='same', **kwargs)(x)
            x = tf.keras.layers.Activation('relu')(x)
            if apply_norm:
                if norm_type.lower() == 'batchnorm':
                    x = tf.keras.layers.BatchNormalization()(x)
                elif norm_type.lower() == 'instancenorm':
                    x = InstanceNormalization()(x)
            x = tf.keras.layers.UpSampling2D(size=factor)(x)
            return x

        if scale == 2:
            x = upsample_edsr(x, 2, name='conv2d_1_scale_2_up' + nameIn)
        elif scale == 3:
            x = upsample_edsr(x, 3, name='conv2d_1_scale_3_up' + nameIn)
        elif scale == 4:
            x = upsample_edsr(x, 2, name='conv2d_1_scale_2_up' + nameIn)
            x = upsample_edsr(x, 2, name='conv2d_2_scale_2_up' + nameIn)
        elif scale == 8:
            x = upsample_edsr(x, 2, name='conv2d_1_scale_2_up' + nameIn)
            x = upsample_edsr(x, 2, name='conv2d_2_scale_2_up' + nameIn)
            x = upsample_edsr(x, 2, name='conv2d_3_scale_2_up' + nameIn)
        return x

    def edsr(scale, num_filters=64, num_res_blocks=8):
        x_in = tf.keras.layers.Input(shape=(None, None, 1))
        x = x_in
        x = b = conv2d(num_filters, 3, padding='same')(x)
        for i in range(num_res_blocks):
            b = res_block_EDSR(b, num_filters, 3, norm_type='instancenorm', apply_norm=False)
        b = conv2d(num_filters, 3, padding='same')(b)
        x = tf.keras.layers.Add()([x, b])

        x = upsampleEDSR(x, scale, num_filters, norm_type='instancenorm', apply_norm=False)

        x = conv2d(1, 3, padding='same')(x)
        x = tf.keras.layers.Activation('tanh', dtype='float32')(x)

        return tf.keras.models.Model(x_in, x, name="EDSR")

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def meanAbsoluteError(labels, predictions):
        per_example_loss = tf.reduce_mean(tf.abs(labels - predictions), axis=[1, 2, 3])
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=labels.shape[0])

    # ------------------------------------------------------------------
    # Model factory
    # ------------------------------------------------------------------

    def createSRGenerator(args):
        generator = edsr(scale=args.scale, num_filters=args.ngsrf, num_res_blocks=args.numResBlocks)
        generator.summary(200)
        optimizerGenerator = tf.keras.optimizers.Adam(learning_rate=args.lr)
        optimizerGenerator = mixed_precision.LossScaleOptimizer(optimizerGenerator)
        return generator, optimizerGenerator

    # ------------------------------------------------------------------
    # Train step (single stage, no cascade, no transpose, no discriminator)
    # ------------------------------------------------------------------

    def train_step(HRBatch, BCBatch):
        Cxyz, Bxy = HRBatch, BCBatch
        if args.augFlag:
            Bxy = augmentData(Bxy)

        with tf.GradientTape() as tape:
            # downsample the depth/batch axis of the HR cube so it matches
            # the number of LR slices actually fed to the generator (this
            # is only needed because createTrainingCubes2 pads that axis by
            # `scale`, not because of any cross-axis cascade)
            Cxyd = tf.image.resize(
                tf.squeeze(Cxyz), [Cxyz.shape[0] // args.scale, Cxyz.shape[2]], method='bicubic')
            Cxyd = tf.expand_dims(Cxyd, 3)

            SRxy = generatorSR(Bxy, training=True)
            totalGsrLoss = meanAbsoluteError(Cxyd, SRxy)

            totalGsrLossScal = optimizerGeneratorSR.get_scaled_loss(totalGsrLoss)

        gradGsr = tape.gradient(totalGsrLossScal, generatorSR.trainable_variables)
        unsc_gradGsr = optimizerGeneratorSR.get_unscaled_gradients(gradGsr)
        optimizerGeneratorSR.apply_gradients(zip(unsc_gradGsr, generatorSR.trainable_variables))

        return totalGsrLoss

    @tf.function
    def distributed_train_step(HRBatch, BCBatch):
        PRGABL = strategy.run(train_step, args=(HRBatch, BCBatch))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, PRGABL, axis=None)

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------

    generatorSR, optimizerGeneratorSR = createSRGenerator(args)

    trainingDir = f"./{args.checkpoint_dir}/{args.modelName}/"
    if args.continue_train:
        print(f'Loading checkpoints from {trainingDir} for epoch {args.continueEpoch}')
        try:
            generatorSR.load_weights(f'{trainingDir}/GSR-{args.continueEpoch}.weights.h5')
        except Exception:
            print('Could not load SR related weights')

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------

    if args.phase == 'train':
        EPOCHS = args.epoch
        valoutDir = args.dataset_dir.split('/')[-2]
        rightNow = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        trainOutputDir = f'./training_outputs/{rightNow}-2Dsolver-{valoutDir}-{args.modelName}/'
        if not os.path.exists(trainingDir):
            os.makedirs(trainingDir, exist_ok=True)
        os.makedirs(trainOutputDir, exist_ok=True)

        print('2D training specified, dataset will be randomly mini-batched per epoch')
        print('Dataset and training -> data will be fully preloaded into RAM')

        BCLoc = glob(args.dataset_dir + 'LR/LR.npy')
        LRxy = np.load(BCLoc[0])

        HRLoc = glob(args.dataset_dir + 'HR/HR.npy')
        HR = np.load(HRLoc[0])

        # Held-out validation data, loaded ONCE from a SEPARATE directory --
        # not the same data used for training.
        valLRLoc = glob(args.val_dataset_dir + 'LR/LR.npy')
        valLRxy = np.load(valLRLoc[0])
        valHRLoc = glob(args.val_dataset_dir + 'HR/HR.npy')
        valHR = np.load(valHRLoc[0])
        print(f'Loaded validation data from {args.val_dataset_dir} '
              f'(LR shape: {valLRxy.shape}, HR shape: {valHR.shape})')

        if args.valTest:
            LRTestLoc = glob(args.dataset_dir + 'test/*')
            LRTest = np.load(LRTestLoc[0])
            LRTest = tf.cast(LRTest, tf.float32)
            LRTest = tf.expand_dims(LRTest, 3)

        start_time = time.time()
        for epoch in range(EPOCHS):
            # Without a discriminator there's no need for a fixed
            # GAN-style batch/patch size -- always use variable-size patch
            # sampling driven by a fixed voxel budget per batch.
            totalPerBatchVoxels = args.fine_size * args.fine_size * args.batch_size
            minPerDimSize = args.scale * 2
            maxPerDimSize = args.fine_size
            batchSizeThisEpoch = int(np.floor(np.random.rand() * (maxPerDimSize - minPerDimSize)) + minPerDimSize)
            fineSizeThisEpoch = int(np.floor(np.sqrt(totalPerBatchVoxels / batchSizeThisEpoch)))

            print(f'Reading and Distributing Dataset into GPUs, block size this epoch: '
                  f'{batchSizeThisEpoch} x {fineSizeThisEpoch} x {fineSizeThisEpoch} -> {args.scale}x')
            realHRBatches, realBCBatches = createTrainingCubes2(
                args, HR, LRxy, batchSizeThisEpoch, fineSizeThisEpoch, args.scale)

            HR_dataset = tf.data.Dataset.from_tensor_slices(realHRBatches).batch(batchSizeThisEpoch * args.scale)
            HR_dataset_dist = strategy.experimental_distribute_dataset(HR_dataset)

            # Validation patches come from the SEPARATE held-out validation
            # set. Patch/batch size is capped to fit inside the (smaller)
            # validation volume.
            valBatchSize = min(batchSizeThisEpoch, valLRxy.shape[0] - 1, valLRxy.shape[1] - 1, valLRxy.shape[2] - 1)
            valCropSize = min(fineSizeThisEpoch, valLRxy.shape[0] - 1, valLRxy.shape[1] - 1, valLRxy.shape[2] - 1)
            valHRBatches, valBCBatches = createTrainingCubes2(
                args, valHR, valLRxy, valBatchSize, valCropSize, args.scale, n_batches=args.valNum)
            HR_dataset_test = tf.data.Dataset.from_tensor_slices(valHRBatches).batch(valBatchSize * args.scale)
            LR_dataset_test = tf.data.Dataset.from_tensor_slices(valBCBatches).batch(valBatchSize)

            LR_dataset = tf.data.Dataset.from_tensor_slices(realBCBatches).batch(batchSizeThisEpoch)
            LR_dataset_dist = strategy.experimental_distribute_dataset(LR_dataset)

            lastTime = time.time()
            lr = args.lr * 0.5 ** (epoch / args.epoch_step)
            optimizerGeneratorSR.learning_rate = lr

            totGABL = 0
            num_batches = 0
            print(f'Learning Rate: {lr:.4e}')
            while num_batches < args.itersPerEpoch * args.iterCyclesPerEpoch:
                for x, y in zip(HR_dataset, LR_dataset):
                    num_batches += 1

                    GABL = distributed_train_step(x, y)
                    totGABL += GABL
                    currentTime = time.time()

                    stdout.write(
                        "\rEpoch: %4d, Iter: %4d, Time: %4.4f, Speed: %4.4f its/s, "
                        "GSRxyL: %4.4f"
                        % (epoch + 1, num_batches, currentTime - start_time,
                           1 / (currentTime - lastTime), GABL))
                    stdout.flush()
                    lastTime = currentTime

            stdout.write("\n")
            totGABL /= num_batches
            print('Mean Epoch Performance: GSRxyL: %4.4f' % (totGABL))

            # Metrics validation (loss, PSNR, SSIM) runs EVERY epoch.
            # Saving example .tif images stays on the print_freq schedule.
            saveImagesThisEpoch = np.mod(epoch + 1, args.print_freq) == 0 or epoch == 0

            valPSNRC = 0.0
            valSSIMC = 0.0
            valLossC = 0.0
            numTestBatches = 0
            if saveImagesThisEpoch:
                os.mkdir(f'./{trainOutputDir}/epoch-{epoch+1}/')

            for C, B in zip(HR_dataset_test, LR_dataset_test):
                Cd = tf.image.resize(tf.squeeze(C), [C.shape[0] // args.scale, C.shape[2]], method='bicubic')
                Cd = tf.expand_dims(Cd, 3)
                Co = np.asarray(Cd)
                fakeC = generatorSR(B, training=False)
                fakeCo = np.asarray(fakeC)

                psnrC = tf.image.psnr(fakeC, Cd, 2)
                ssimC = tf.image.ssim(fakeC, Cd, 2)
                lossC = tf.reduce_mean(tf.abs(fakeC - Cd))

                B = np.asarray(B)

                valPSNRC += np.mean(psnrC)
                valSSIMC += np.mean(ssimC)
                valLossC += float(lossC)
                numTestBatches += 1

                if saveImagesThisEpoch:
                    image_path = f'./{trainOutputDir}/epoch-{epoch+1}/{numTestBatches}-Bxy.tif'
                    B = (B + 1) * 127.5
                    tifffile.imwrite(image_path, np.array(np.squeeze(B.astype('uint8')), dtype='uint8'))

                    image_path = f'./{trainOutputDir}/epoch-{epoch+1}/{numTestBatches}-Cxyd.tif'
                    Co = (Co + 1) * 127.5
                    tifffile.imwrite(image_path, np.array(np.squeeze(Co.astype('uint8')), dtype='uint8'))

                    image_path = f'./{trainOutputDir}/epoch-{epoch+1}/{numTestBatches}-BSRxy.tif'
                    fakeCo = (fakeCo + 1) * 127.5
                    tifffile.imwrite(image_path, np.array(np.squeeze(fakeCo.astype('uint8')), dtype='uint8'))

                stdout.write("\rIter: %4d, Test: PSNR-SR: %4.4f" % (numTestBatches, np.mean(psnrC)))
                stdout.flush()
                if numTestBatches == args.valNum:
                    break

            valPSNRC /= numTestBatches
            valSSIMC /= numTestBatches
            valLossC /= numTestBatches

            if args.metricsTracker:
                tracker.log_epoch(
                    epoch,
                    train_loss_xy=float(totGABL),
                    val_loss_xy=float(valLossC),
                    psnr_xy=float(valPSNRC),
                    ssim_xy=float(valSSIMC),
                )

            stdout.write("\n")
            print(f'Mean Validation PSNR-SR: {valPSNRC}, SSIM-SR: {valSSIMC}, Loss-SR: {valLossC}')

            if args.valTest and saveImagesThisEpoch:
                print(f'Generating some test slices')
                testSRxy = generatorSR(LRTest)
                testSRxy = np.asarray(testSRxy)
                image_path = f'./{trainOutputDir}/epoch-{epoch+1}/testSRxy.tif'
                testSRxy = (testSRxy + 1) * 127.5
                tifffile.imwrite(image_path, np.array(np.squeeze(testSRxy.astype('uint8')), dtype='uint8'))

            if (epoch) % args.save_freq == 0:
                print('Saving network weights (archive)')
                generatorSR.save_weights(f'{trainingDir}/GSR-{epoch}.weights.h5')

                print('Saving network weights (rewritable checkpoint)')
                generatorSR.save_weights(f'{trainingDir}/GSR.weights.h5')

                print('Saving model (rewritable checkpoint)')
                generatorSR.save(f'{trainingDir}/GSR-{epoch}.keras')
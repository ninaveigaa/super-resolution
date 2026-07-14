# ---------------------------------------------------------------------------
# Import statements & Envoronment setup
# ---------------------------------------------------------------------------
import tensorflow as tf
import tensorflow.keras.backend as K
import tifffile
# Helper libraries
from sys import stdout
import numpy as np
import os
from glob import glob
import time
import datetime
from configs import Wang_2023_clean_args
from src import metrics


AUTOTUNE = tf.data.experimental.AUTOTUNE
print(tf.__version__)

args = Wang_2023_clean_args.args()  # args is global

gpuList = args.gpuIDs
args.numGPUs = len(gpuList.split(','))
if args.numGPUs <= 4:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpuList

if args.mixedPrecision:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
else:
    policy = tf.keras.mixed_precision.Policy('float32')
tf.keras.mixed_precision.set_global_policy(policy)
print('Compute dtype: %s' % policy.compute_dtype)
print('Variable dtype: %s' % policy.variable_dtype)

# detect hardware
if len(args.gpuIDs.split(',')) <= 1:
    strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
else:
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------
def _gaussian_kernel(kernel_size, sigma, n_channels, dtype):
    x = tf.range(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=dtype)
    g = tf.math.exp(-(tf.pow(x, 2) / (2 * tf.pow(tf.cast(sigma, dtype), 2))))
    g_norm2d = tf.pow(tf.reduce_sum(g), 2)
    g_kernel = tf.tensordot(g, g, axes=0) / g_norm2d
    g_kernel = tf.expand_dims(g_kernel, axis=-1)
    return tf.expand_dims(tf.tile(g_kernel, (1, 1, n_channels)), axis=-1)


def apply_blur(img, kernel_size, sigma, n_channel):
    blur = _gaussian_kernel(kernel_size, sigma, n_channel, img.dtype)
    img = tf.nn.depthwise_conv2d(img, blur, [1, 1, 1, 1], 'SAME')
    return img


def createTrainingCubes2(args, HR, LRxy, batchsize, cropsize, scale, n_batches=None):
    # read an HR block and extract the LRxy,LRyz, and LRxz blocks of size itersperepoch*batch,x,y,1
    # permute the block so the lrbc dimension is in the batch dimension
    if n_batches is None:
        n_batches = args.itersPerEpoch
    batchLR = np.zeros([batchsize * n_batches, cropsize, cropsize, 1], 'float32')
    batchHR = np.zeros([batchsize * n_batches * scale, cropsize * scale, cropsize * scale, 1], 'float32')
    n = 0
    n2 = 0
    for i in range(n_batches):
        # cycle between xy,yz, and xz for extra data
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


# ---------------------------------------------------------------------------
# Load & prepare datasets
# ---------------------------------------------------------------------------
totalPerBatchVoxels = args.fine_size * args.fine_size * args.batch_size
minPerDimSize = args.scale * 2
maxPerDimSize = args.fine_size

def Dataset():
    # training data
    BCLoc_train = glob(args.dataset_dir + 'training/LR/LR.npy')
    LRxy_train = np.load(BCLoc_train[0])

    HRLoc_train = glob(args.dataset_dir + 'training/HR/HR.npy')
    HR_train = np.load(HRLoc_train[0])

    realHRBatches, realBCBatches = createTrainingCubes2(
        args, HR_train, LRxy_train, trainingBatchSize, trainingFineSize, args.scale)

    HR_dataset = tf.data.Dataset.from_tensor_slices((realHRBatches)).batch(trainingBatchSize * args.scale)
    LR_dataset = tf.data.Dataset.from_tensor_slices((realBCBatches)).batch(trainingBatchSize)

    # validation data
    BCLoc_val = glob(args.dataset_dir + 'validation/LR/LR.npy')
    LRxy_val = np.load(BCLoc_val[0])

    HRLoc_val = glob(args.dataset_dir + 'validation/HR/HR.npy')
    HR_val = np.load(HRLoc_val[0])

    valBatchSize = min(trainingBatchSize, LRxy_val.shape[0] - 1, LRxy_val.shape[1] - 1, LRxy_val.shape[2] - 1)
    valCropSize = min(trainingFineSize, LRxy_val.shape[0] - 1, LRxy_val.shape[1] - 1, LRxy_val.shape[2] - 1)

    valHRBatches, valBCBatches = createTrainingCubes2(
        args, HR_val, LRxy_val, valBatchSize, valCropSize, args.scale, n_batches=args.valNum)

    HR_dataset_test = tf.data.Dataset.from_tensor_slices(valHRBatches).batch(valBatchSize * args.scale)
    LR_dataset_test = tf.data.Dataset.from_tensor_slices(valBCBatches).batch(valBatchSize)

    return HR_dataset, LR_dataset, HR_dataset_test, LR_dataset_test


# ---------------------------------------------------------------------------
# Build the model
# ---------------------------------------------------------------------------
def conv(ndims, *args, **kwargs):
    if ndims == 2:
        return tf.keras.layers.Conv2D(*args, **kwargs)
    elif ndims == 3:
        return tf.keras.layers.Conv3D(*args, **kwargs)


class InstanceNormalization(tf.keras.layers.Layer):
    """Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(1., 0.02),
            trainable=True)

        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)

    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset


class InstanceNormalization3D(tf.keras.layers.Layer):
    """Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization3D, self).__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(1., 0.02),
            trainable=True)

        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)

    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2, 3], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset


def instanceNorm(x, ndims):
    if ndims == 2:
        x = InstanceNormalization()(x)
    elif ndims == 3:
        x = InstanceNormalization3D()(x)
    return x


def res_block_EDSR(x_in, filters, kernel, norm_type='instancenorm', apply_norm=False, ndims=2):
    x = conv(ndims, filters, kernel, padding='same')(x_in)
    x = tf.keras.layers.Activation('relu')(x)
    if apply_norm:
        if norm_type.lower() == 'batchnorm':
            x = tf.keras.layers.BatchNormalization()(x)
        elif norm_type.lower() == 'instancenorm':
            x = instanceNorm(x, ndims)
    x = conv(ndims, filters, kernel, padding='same')(x)
    x = tf.keras.layers.Add()([x_in, x])
    return x


def upsampleEDSR(x, scale, num_filters, norm_type='instancenorm', apply_norm=False, ndims=2, nameIn=''):
    def upsample_edsr(x, factor, ndims, **kwargs):
        x = conv(ndims, num_filters, 3, padding='same', **kwargs)(x)
        x = tf.keras.layers.Activation('relu')(x)
        if apply_norm:
            if norm_type.lower() == 'batchnorm':
                x = tf.keras.layers.BatchNormalization()(x)
            elif norm_type.lower() == 'instancenorm':
                x = instanceNorm(x, ndims)
        if ndims == 2:
            x = tf.keras.layers.UpSampling2D(size=factor)(x)
            return x
        elif ndims == 3:
            x = tf.keras.layers.UpSampling3D(size=factor)(x)
            return x

    if scale == 2:
        x = upsample_edsr(x, 2, ndims=ndims, name='conv2d_1_scale_2_up' + nameIn)
    elif scale == 3:
        x = upsample_edsr(x, 3, ndims=ndims, name='conv2d_1_scale_3_up' + nameIn)
    elif scale == 4:
        x = upsample_edsr(x, 2, ndims=ndims, name='conv2d_1_scale_2_up' + nameIn)
        x = upsample_edsr(x, 2, ndims=ndims, name='conv2d_2_scale_2_up' + nameIn)
    elif scale == 8:
        x = upsample_edsr(x, 2, ndims=ndims, name='conv2d_1_scale_2_up' + nameIn)
        x = upsample_edsr(x, 2, ndims=ndims, name='conv2d_2_scale_2_up' + nameIn)
        x = upsample_edsr(x, 2, ndims=ndims, name='conv2d_3_scale_2_up' + nameIn)
    return x


def upsampleEDSR1D(x, scale, num_filters, norm_type='instancenorm', apply_norm=False, ndims=2, nameIn=''):
    def upsample_edsr(x, factor, ndims, **kwargs):
        x = conv(ndims, num_filters, 3, padding='same', **kwargs)(x)
        x = tf.keras.layers.Activation('relu')(x)
        if apply_norm:
            if norm_type.lower() == 'batchnorm':
                x = tf.keras.layers.BatchNormalization()(x)
            elif norm_type.lower() == 'instancenorm':
                x = instanceNorm(x, ndims)
        if ndims == 2:
            x = tf.keras.layers.UpSampling2D(size=factor)(x)
            return x
        elif ndims == 3:
            x = tf.keras.layers.UpSampling3D(size=factor)(x)
            return x

    if scale == 2:
        x = upsample_edsr(x, (2, 1), ndims=ndims, name='conv2d_1_scale_2_up' + nameIn)
    elif scale == 3:
        x = upsample_edsr(x, (3, 1), ndims=ndims, name='conv2d_1_scale_3_up' + nameIn)
    elif scale == 4:
        x = upsample_edsr(x, (2, 1), ndims=ndims, name='conv2d_1_scale_2_up' + nameIn)
        x = upsample_edsr(x, (2, 1), ndims=ndims, name='conv2d_2_scale_2_up' + nameIn)
    elif scale == 8:
        x = upsample_edsr(x, (2, 1), ndims=ndims, name='conv2d_1_scale_2_up' + nameIn)
        x = upsample_edsr(x, (2, 1), ndims=ndims, name='conv2d_2_scale_2_up' + nameIn)
        x = upsample_edsr(x, (2, 1), ndims=ndims, name='conv2d_3_scale_2_up' + nameIn)
    return x


def SubpixelConv2D(scale, **kwargs):
    return tf.keras.layers.Lambda(lambda x: tf.nn.depth_to_space(x, scale), **kwargs)


def edsr(scale, num_filters=64, num_res_blocks=8, ndims=2):
    if ndims == 2:
        x_in = tf.keras.layers.Input(shape=(None, None, 1))
    elif ndims == 3:
        x_in = tf.keras.layers.Input(shape=(None, None, None, 1))
    x = x_in
    x = b = conv(ndims, num_filters, 3, padding='same')(x)
    for i in range(num_res_blocks):
        b = res_block_EDSR(b, num_filters, 3, norm_type='instancenorm', apply_norm=False, ndims=ndims)
    b = conv(ndims, num_filters, 3, padding='same')(b)
    x = tf.keras.layers.Add()([x, b])

    x = upsampleEDSR(x, scale, num_filters, norm_type='instancenorm', apply_norm=False, ndims=ndims)
    x = conv(ndims, 1, 3, padding='same')(x)
    x = tf.keras.layers.Activation('tanh', dtype='float32')(x)

    return tf.keras.models.Model(x_in, x, name="EDSR")


def edsr1D(scale, num_filters=64, num_res_blocks=8, ndims=2):
    if ndims == 2:
        x_in = tf.keras.layers.Input(shape=(None, None, 1))
    elif ndims == 3:
        x_in = tf.keras.layers.Input(shape=(None, None, None, 1))
    x = x_in
    x = b = conv(ndims, num_filters, 3, padding='same')(x)
    for i in range(num_res_blocks):
        b = res_block_EDSR(b, num_filters, 3, norm_type='instancenorm', apply_norm=False, ndims=ndims)
    b = conv(ndims, num_filters, 3, padding='same')(b)
    x = tf.keras.layers.Add()([x, b])

    x = upsampleEDSR1D(x, scale, num_filters, norm_type='instancenorm', apply_norm=False, ndims=ndims)
    x = conv(ndims, 1, 3, padding='same')(x)
    x = tf.keras.layers.Activation('tanh', dtype='float32')(x)

    return tf.keras.models.Model(x_in, x, name="EDSR")


# ---------------------------------------------------------------------------
# Losses, model builder and optimizers
# ---------------------------------------------------------------------------
def meanAbsoluteError(labels, predictions):
    per_example_loss = tf.reduce_mean(tf.abs(labels - predictions), axis=[1, 2, 3])
    return tf.nn.compute_average_loss(per_example_loss, global_batch_size=labels.shape[0])


def createSRGenerator(args):
    generator = edsr(scale=args.scale, num_filters=args.ngsrf, num_res_blocks=args.numResBlocks, ndims=2)
    generator.summary(200)
    optimizerGenerator = tf.keras.optimizers.Adam(learning_rate=args.lr)
    if args.mixedPrecision:
        optimizerGenerator = tf.keras.mixed_precision.LossScaleOptimizer(optimizerGenerator)
    return generator, optimizerGenerator


def createSRCGenerator(args):
    generator = edsr1D(scale=args.scale, num_filters=args.ngsrf // 2, num_res_blocks=args.numResBlocks // 2, ndims=2)
    generator.summary(200)
    optimizerGenerator = tf.keras.optimizers.Adam(learning_rate=args.lr)
    if args.mixedPrecision:
        optimizerGenerator = tf.keras.mixed_precision.LossScaleOptimizer(optimizerGenerator)
    return generator, optimizerGenerator


with strategy.scope():
    generatorSR, optimizerGeneratorSR = createSRGenerator(args)
    generatorSRC, optimizerGeneratorSRC = createSRCGenerator(args)

# ---------------------------------------------------------------------------
# Train Step
# ---------------------------------------------------------------------------
def train_step(HRBatch, BCBatch):
    Cxyz, Bxy = HRBatch, BCBatch  # make sure the dims are correct
    print(f"Cxyz shape: {Cxyz.shape}, Bxy shape: {Bxy.shape}")
    # train
    with tf.GradientTape(persistent=True) as tape:
        Cxyd = tf.image.resize(tf.squeeze(Cxyz), [Cxyz.shape[0] // args.scale, Cxyz.shape[2]], method='bicubic')
        Cxyd = tf.expand_dims(Cxyd, 3)
        print(f"Cxyd shape: {Cxyd.shape}")
        SRxy = generatorSR(Bxy, training=True)
        print(f"SRxy shape: {SRxy.shape}")
        totalGsrXYLoss = meanAbsoluteError(Cxyd, SRxy)

        # set bit depth to 8 for SRxy
        SRxy = (SRxy + 1) * 127.5
        SRxy = tf.math.round(SRxy)
        SRxy = SRxy / 127.5 - 1

        # transpose the volume
        SRxy = tf.transpose(SRxy, perm=[1, 0, 2, 3])
        Cxyz = tf.transpose(Cxyz, perm=[1, 0, 2, 3])
        print(f"SRxy shape after transpose: {SRxy.shape}, Cxyz shape after transpose: {Cxyz.shape}")

        SRxyz = generatorSRC(SRxy, training=True)
        print(f"SRxyz shape: {SRxyz.shape}")
        totalGsrYZLoss = meanAbsoluteError(Cxyz, SRxyz)

        totalGsrXYZLoss = totalGsrYZLoss + totalGsrXYLoss

        if args.mixedPrecision:
            totalGsrLossScal = optimizerGeneratorSR.scale_loss(totalGsrXYZLoss)
            totalGsrcLossScal = optimizerGeneratorSRC.scale_loss(totalGsrXYZLoss)
        else:
            totalGsrLossScal = totalGsrXYZLoss
            totalGsrcLossScal = totalGsrXYZLoss

    # calculate gradients
    gradGsr = tape.gradient(totalGsrLossScal, generatorSR.trainable_variables)
    gradGsrc = tape.gradient(totalGsrcLossScal, generatorSRC.trainable_variables)

    # apply gradients
    optimizerGeneratorSR.apply_gradients(zip(gradGsr, generatorSR.trainable_variables))
    optimizerGeneratorSRC.apply_gradients(zip(gradGsrc, generatorSRC.trainable_variables))

    return totalGsrXYLoss, totalGsrYZLoss


@tf.function
def distributed_train_step(HRBatch, BCBatch):
    PRGABL, PRGBAL = strategy.run(train_step, args=(HRBatch, BCBatch))
    return (strategy.reduce(tf.distribute.ReduceOp.SUM, PRGABL, axis=None),
            strategy.reduce(tf.distribute.ReduceOp.SUM, PRGBAL, axis=None))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
trainingDir = os.path.join(args.checkpoint_dir, args.modelName) + "/"

if args.phase == 'train':
    EPOCHS = args.epoch
    valoutDir = args.dataset_dir.split('/')[-2]
    # Create a checkpoint directory to store the checkpoints.
    rightNow = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    trainOutputDir = f'./training_outputs/{rightNow}-distNN-{valoutDir}-{args.modelName}/'
    os.makedirs(trainingDir, exist_ok=True)
    os.makedirs(trainOutputDir, exist_ok=True)

    # -----------------------------------------------------------------
    # METRICS
    # -----------------------------------------------------------------
    run_id = metrics.save_args(args, args.modelName)
    metricsTracker = metrics.MetricsTracker(args.modelName, run_id)
    metricsTracker.start_training()
    print(f'Metrics tracking started, run_id: {run_id}')

    print('2D/3D training specified, datasets will be randomly mini-batched per epoch')
    print('2D/3D dataset and training -> data will be fully preloaded into RAM')

    if args.valTest:
        LRTestLoc = glob(args.dataset_dir + 'validation/*')
        LRTest = np.load(LRTestLoc[0])
        LRTest = tf.cast(LRTest, tf.float32)
        LRTest = tf.expand_dims(LRTest, 3)
    start_time = time.time()
    for epoch in range(EPOCHS):
        totalPerBatchVoxels = args.fine_size * args.fine_size * args.batch_size
        minPerDimSize = args.scale * 2
        maxPerDimSize = args.fine_size
        trainingBatchSize = int(np.floor(np.random.rand() * (maxPerDimSize - minPerDimSize)) + minPerDimSize)
        trainingFineSize = int(np.floor(np.sqrt(totalPerBatchVoxels / trainingBatchSize)))
        print(f'Reading and Distributing Dataset into GPUs, block size this epoch: '
              f'{trainingBatchSize} x {trainingFineSize} x {trainingFineSize} -> {args.scale}x')
        HR_dataset, LR_dataset, HR_dataset_test, LR_dataset_test = Dataset()

        # TRAIN LOOP
        lastTime = time.time()

        lr = args.lr * 0.5 ** (epoch / args.epoch_step)  # add cosine annealing later

        optimizerGeneratorSR.learning_rate = lr
        optimizerGeneratorSRC.learning_rate = lr
        totGABL = 0
        totGBAL = 0
        num_batches = 0
        print(f'Learning Rate: {lr:.4e}')
        while num_batches < args.itersPerEpoch * args.iterCyclesPerEpoch:
            for x, y in zip(HR_dataset, LR_dataset):
                num_batches += 1
                GABL, GBAL = distributed_train_step(x, y)
                totGABL += GABL
                totGBAL += GBAL
                currentTime = time.time()
                stdout.write("\rEpoch: %4d, Iter: %4d, Time: %4.4f, Speed: %4.4f its/s, GSRxyL: %4.4f, GSRyzL: %4.4f" %
                              (epoch + 1, num_batches, currentTime - start_time,
                               1 / (currentTime - lastTime), GABL, GBAL))
                stdout.flush()
                lastTime = currentTime
                if num_batches >= args.itersPerEpoch * args.iterCyclesPerEpoch:
                    break

        stdout.write("\n")
        totGABL /= num_batches
        totGBAL /= num_batches
        print('Mean Epoch Performance: GSRxyL: %4.4f, GSRyzL: %4.4f' % (totGABL, totGBAL))

        # -------------------------------------------------------------
        # METRICS
        # -------------------------------------------------------------
        epoch_val_loss_xy = None
        epoch_val_loss_z = None
        epoch_psnr_xy = None
        epoch_psnr_final = None
        epoch_ssim_xy = None
        epoch_ssim_final = None

        if np.mod(epoch + 1, args.print_freq) == 0 or epoch == 0:
            # validation LOOP
            valPSNRC = 0.0
            valPSNRCC = 0.0
            valSSIMC = 0.0
            valSSIMCC = 0.0
            valLossC = 0.0
            valLossCC = 0.0

            numTestBatches = 0
            os.makedirs(f'./{trainOutputDir}/epoch-{epoch + 1}/', exist_ok=True)

            for C, B in zip(HR_dataset_test, LR_dataset_test):

                Cd = tf.image.resize(tf.squeeze(C), [C.shape[0] // args.scale, C.shape[2]], method='bicubic')
                Cd = tf.expand_dims(Cd, 3)
                Co = np.asarray(Cd)
                fakeC = generatorSR(B, training=False)
                fakeCo = np.asarray(fakeC)

                psnrC = tf.image.psnr(fakeC, Cd, 2)
                ssimC = tf.image.ssim(fakeC, Cd, 2)
                lossC = meanAbsoluteError(Cd, fakeC)
                # set bit depth to 8 for SRxy
                fakeC = (fakeC + 1) * 127.5
                fakeC = tf.math.round(fakeC)
                fakeC = fakeC / 127.5 - 1
                # transpose and downsample here
                fakeC = tf.transpose(fakeC, [1, 0, 2, 3])
                B = tf.transpose(B, [1, 0, 2, 3])
                C = tf.transpose(C, [1, 0, 2, 3])
                fakeC_clean = generatorSRC(fakeC, training=False)
                psnrCC = tf.image.psnr(fakeC_clean, C, 2)
                ssimCC = tf.image.ssim(fakeC_clean, C, 2)
                lossCC = meanAbsoluteError(C, fakeC_clean)

                B = np.asarray(B)
                C = np.asarray(C)
                fakeC = np.asarray(fakeC)
                fakeC_clean = np.asarray(fakeC_clean)

                valPSNRC += np.mean(psnrC)
                valPSNRCC += np.mean(psnrCC)
                valSSIMC += np.mean(ssimC)
                valSSIMCC += np.mean(ssimCC)
                valLossC += float(lossC)
                valLossCC += float(lossCC)
                numTestBatches += 1

                image_path = f'./{trainOutputDir}/epoch-{epoch + 1}/{numTestBatches}-Bxy.tif'
                B = (B + 1) * 127.5
                tifffile.imwrite(image_path, np.array(np.squeeze(B.astype('uint8')), dtype='uint8'))

                image_path = f'./{trainOutputDir}/epoch-{epoch + 1}/{numTestBatches}-Cxyz.tif'
                Co = (Co + 1) * 127.5
                tifffile.imwrite(image_path, np.array(np.squeeze(Co.astype('uint8')), dtype='uint8'))

                image_path = f'./{trainOutputDir}/epoch-{epoch + 1}/{numTestBatches}-Ctxyz.tif'
                C = (C + 1) * 127.5
                tifffile.imwrite(image_path, np.array(np.squeeze(C.astype('uint8')), dtype='uint8'))

                image_path = f'./{trainOutputDir}/epoch-{epoch + 1}/{numTestBatches}-BSRxy.tif'
                fakeCo = (fakeCo + 1) * 127.5
                tifffile.imwrite(image_path, np.array(np.squeeze(fakeCo.astype('uint8')), dtype='uint8'))

                image_path = f'./{trainOutputDir}/epoch-{epoch + 1}/{numTestBatches}-BSRxytd.tif'
                fakeC = (fakeC + 1) * 127.5
                tifffile.imwrite(image_path, np.array(np.squeeze(fakeC.astype('uint8')), dtype='uint8'))

                image_path = f'./{trainOutputDir}/epoch-{epoch + 1}/{numTestBatches}-BSRxyz.tif'
                fakeC_clean = (fakeC_clean + 1) * 127.5
                tifffile.imwrite(image_path, np.array(np.squeeze(fakeC_clean.astype('uint8')), dtype='uint8'))

                stdout.write("\rIter: %4d, Test: PSNR-SR: %4.4f, PSNR-SRC: %4.4f" %
                              (numTestBatches, np.mean(psnrC), np.mean(psnrCC)))
                stdout.flush()
                if numTestBatches == args.valNum:
                    break

            valPSNRC /= numTestBatches
            valPSNRCC /= numTestBatches
            valSSIMC /= numTestBatches
            valSSIMCC /= numTestBatches
            valLossC /= numTestBatches
            valLossCC /= numTestBatches

            stdout.write("\n")
            print(f'Mean Validation PSNR-SR: {valPSNRC}, PSNR-SRC: {valPSNRCC}')

            epoch_val_loss_xy = valLossC
            epoch_val_loss_z = valLossCC
            epoch_psnr_xy = valPSNRC
            epoch_psnr_final = valPSNRCC
            epoch_ssim_xy = valSSIMC
            epoch_ssim_final = valSSIMCC

            if args.valTest:
                print(f'Generating some test cubes')
                testSRxy = generatorSR(LRTest)
                testSRxy = np.asarray(testSRxy)
                image_path = f'./{trainOutputDir}/epoch-{epoch + 1}/testSRxy.tif'
                testSRxy = (testSRxy + 1) * 127.5
                tifffile.imwrite(image_path, np.array(np.squeeze(testSRxy.astype('uint8')), dtype='uint8'))

        # -------------------------------------------------------------
        # METRICS
        # -------------------------------------------------------------
        metricsTracker.log_epoch(
            epoch=epoch + 1,
            train_loss_xy=float(totGABL),
            train_loss_z=float(totGBAL),
            val_loss_xy=epoch_val_loss_xy,
            val_loss_z=epoch_val_loss_z,
            psnr_xy=epoch_psnr_xy,
            psnr_final=epoch_psnr_final,
            ssim_xy=epoch_ssim_xy,
            ssim_final=epoch_ssim_final,
        )

        if (epoch) % args.save_freq == 0:
            print('Saving network weights (archive)')
            os.makedirs(f'{trainingDir}/GSR-{epoch}', exist_ok=True)
            os.makedirs(f'{trainingDir}/GSRC-{epoch}', exist_ok=True)
            generatorSR.save_weights(f'{trainingDir}/GSR-{epoch}/GSR.weights.h5')
            generatorSRC.save_weights(f'{trainingDir}/GSRC-{epoch}/GSRC.weights.h5')

            print('Saving network weights (rewritable checkpoint)')
            os.makedirs(f'{trainingDir}/GSR', exist_ok=True)
            os.makedirs(f'{trainingDir}/GSRC', exist_ok=True)
            generatorSR.save_weights(f'{trainingDir}/GSR/GSR.weights.h5')
            generatorSRC.save_weights(f'{trainingDir}/GSRC/GSRC.weights.h5')
            print('Saving model (rewritable checkpoint)')
            generatorSR.save(f'{trainingDir}/GSR-{epoch}.keras')
            generatorSRC.save(f'{trainingDir}/GSRC-{epoch}.keras')

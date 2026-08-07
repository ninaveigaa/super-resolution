"""
PyTorch training loop, equivalent to the 'train_step' / 'phase == train' part
of the original script, WITHOUT GAN (only the L1 loss of the two coupled
generators).

The arguments here mirror the original dualSRNetArgs.py exactly (same
--flags, dest and defaults). One point that needed a judgment call: the
original training script (train_step / phase=='train') validated by
recycling the first `valNum` chunks of the training dataset itself, but the
argparser requires `--val_dataset_dir` (held-out). Here I chose to actually
use a separate dataset for validation, loaded from `--val_dataset_dir` (same
folder convention: LR/LR.npy and HR/HR.npy).

Usage:
    python train.py --dataset_dir ./data/train/ --val_dataset_dir ./data/val/ \\
        --modelName myModel --scale 4 --gpuIDs 0
"""

import argparse
import datetime
import os
import sys
import time
from glob import glob

import numpy as np
import tifffile
import torch
import torch.nn.functional as F

from src.dataset import augment_data, make_epoch_cubes
from models.dual_edsr import create_sr_generator, create_src_generator


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str2int(v):
    if v == 'M':
        return v
    try:
        v = int(v)
    except ValueError:
        raise argparse.ArgumentTypeError('int value expected.')
    return v


def str2float(v):
    if v == 'M':
        return v
    try:
        v = float(v)
    except ValueError:
        raise argparse.ArgumentTypeError('float value expected.')
    return v


def parse_args():
    """Identical to the original dualSRNetArgs.py (same flags, dest and defaults)."""
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--metricsTracker', dest='metricsTracker', type=str2bool, default=True, help='if metrics are tracked')

    parser.add_argument('--mixedPrecision', dest='mixedPrecision', type=str2bool, default=False, help='16bit computes')
    parser.add_argument('--gpuIDs', dest='gpuIDs', type=str, default='2', help='IDs for the GPUs. Empty for CPU. Nospaces')
    parser.add_argument('--dataset_dir', dest='dataset_dir', default='/media/user/SSD2/fuelCellDataset2Dxy/', help='dataset path - include last slash')
    parser.add_argument('--augFlag', dest='augFlag', type=str2bool, default=False, help='augmentation')

    parser.add_argument('--scale', dest='scale', type=str2int, default=4, help='sr scale factor')
    parser.add_argument('--batch_size', dest='batch_size', type=str2int, default=64, help='# 2D images in subbatch')
    parser.add_argument('--subBlocks', dest='subBlocks', type=str2int, default=4, help='# 3D images in batch')
    parser.add_argument('--fine_size', dest='fine_size', type=str2int, default=64, help='then crop LR to this size')

    parser.add_argument('--epoch', dest='epoch', type=str2int, default=500, help='# of epoch')
    parser.add_argument('--itersPerEpoch', dest='itersPerEpoch', type=str2int, default=300, help='# iterations per epoch')
    parser.add_argument('--iterCyclesPerEpoch', dest='iterCyclesPerEpoch', type=str2int, default=3, help='# iteration cycles per epoch')

    parser.add_argument('--valNum', dest='valNum', type=str2int, default=10, help='# max val images')
    parser.add_argument('--valTest', dest='valTest', type=str2bool, default=False, help='# max val images')

    # base model uses dualEDSR
    parser.add_argument('--ngsrf', dest='ngsrf', type=str2int, default=64, help='# of gen SR filters in first conv layer')
    parser.add_argument('--numResBlocks', dest='numResBlocks', type=str2int, default=16, help='# of resBlocks in SR')

    # base model uses SCGAN (not implemented in this PyTorch version, see notes)
    parser.add_argument('--ganFlag', dest='ganFlag', type=str2bool, default=False, help='if gan is active')
    parser.add_argument('--ndsrf', dest='ndsrf', type=str2int, default=64, help='# of disc SR filters in first conv layer')
    parser.add_argument('--srAdv_lambda', dest='srAdv_lambda', type=str2float, default=1e-2, help='weight on Adv term for normal sr')
    parser.add_argument('--disc_size_max', dest='disc_size', type=str2int, default=64, help='then crop HR to this size during disc')

    parser.add_argument('--lr', dest='lr', type=float, default=1e-4, help='initial learning rate for adam')
    parser.add_argument('--epoch_step', dest='epoch_step', type=str2int, default=50, help='# of epoch to decay lr')

    parser.add_argument('--phase', dest='phase', type=str, default='train', help='train, test')

    # Model IO
    parser.add_argument('--save_freq', dest='save_freq', type=str2int, default=10, help='save a model every save_freq epochs')
    parser.add_argument('--print_freq', dest='print_freq', type=str2int, default=10, help='print the validation images every X epochs')
    parser.add_argument('--continue_train', dest='continue_train', type=str2bool, default=False, help='if continue training, load the latest model: 1: true, 0: false')
    parser.add_argument('--continueEpoch', dest='continueEpoch', type=str2int, default=0, help='')
    parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoints', help='models are saved here')
    parser.add_argument('--modelName', dest='modelName', default='dual2DSRTest', help='models are loaded here')

    # testing arguments
    parser.add_argument('--test_dir', dest='test_dir', default='/media/user/SSD2/testLR/', help='test sample slices are saved here as png slices')
    parser.add_argument('--test_temp_save_dir', dest='test_temp_save_dir', default='/media/user/SSD2/', help='test sample are saved here')
    parser.add_argument('--test_save_dir', dest='test_save_dir', default='/media/user/SSD2/', help='test sample are saved here')
    parser.add_argument('--val_dataset_dir', dest='val_dataset_dir', required=True, help='separate held-out validation set path -- include trailing slash')

    args = parser.parse_args()
    return args


def resolve_device(args):
    """Equivalent to the CUDA_VISIBLE_DEVICES section of the original TF script."""
    gpu_list = ','.join([g.strip() for g in args.gpuIDs.split(',') if g.strip()])
    if gpu_list:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
        if not torch.cuda.is_available():
            print('GPUs requested via --gpuIDs but CUDA is unavailable; falling back to CPU.')
            return torch.device('cpu')
        # after setting CUDA_VISIBLE_DEVICES, indices visible to torch start at 0
        return torch.device('cuda:0')
    print('No GPUs specified; using CPU.')
    return torch.device('cpu')


def mean_absolute_error(pred, target):
    """Equivalent to meanAbsoluteError (mean absolute error, without the distributed reduce_average_loss)."""
    return torch.mean(torch.abs(pred - target))


def downsample_batch_axis_bicubic(x, new_n):
    """
    Equivalent to the trick used in the original script:
        Cxyd = tf.image.resize(tf.squeeze(Cxyz), [Cxyz.shape[0]//scale, Cxyz.shape[2]], 'bicubic')
    There, the squeezed tensor (N, H, W) is treated by tf.image.resize as
    (height=N, width=H, channels=W): i.e. it bicubically resamples the BATCH
    axis (N), which in this pipeline represents the "z" thickness of that
    pass, from N to `new_n`, keeping H and W as they are (the target "W" in
    the resize equals the original one, so that axis is essentially unchanged).

    x: NCHW tensor (N, 1, H, W). Returns (new_n, 1, H, W).
    """
    n, c, h, w = x.shape
    assert c == 1
    # -> (1, W, N, H): W becomes the "channel" (not resampled), N and H are
    # the 2 spatial axes given to interpolate (target H = new_n, target W = h,
    # i.e. same as the original -> effectively unchanged).
    xr = x.squeeze(1).permute(2, 0, 1).unsqueeze(0)  # (1, W, N, H)
    xr = F.interpolate(xr, size=(new_n, h), mode='bicubic', align_corners=False)
    xr = xr.squeeze(0).permute(1, 2, 0).unsqueeze(1)  # (new_n, 1, H, W)
    return xr


def train_step(args, Bxy, Cxyz, generatorSR, generatorSRC, optSR, optSRC):
    """
    Equivalent to the original train_step, without the GAN part.
    Bxy:  (N, 1, h, w)       -> "xy" LR block
    Cxyz: (N*scale, 1, H, W) -> corresponding HR block
    """
    if args.augFlag:
        Bxy = augment_data(Bxy)

    optSR.zero_grad(set_to_none=True)
    optSRC.zero_grad(set_to_none=True)

    # bicubic downsample of the batch/z axis, target for the XY branch
    Cxyd = downsample_batch_axis_bicubic(Cxyz, Cxyz.shape[0] // args.scale)

    SRxy = generatorSR(Bxy)
    loss_xy = mean_absolute_error(SRxy, Cxyd)

    # quantize to 8 bits, same as the original (this also cuts the gradient
    # flow between generatorSR and the YZ loss, since round() has zero gradient)
    SRxy_q = torch.round((SRxy + 1) * 127.5)
    SRxy_q = SRxy_q / 127.5 - 1

    # "cube trick": swap the batch axis with the height (H) axis
    SRxy_t = SRxy_q.permute(2, 1, 0, 3).contiguous()
    Cxyz_t = Cxyz.permute(2, 1, 0, 3).contiguous()

    SRxyz = generatorSRC(SRxy_t)
    loss_yz = mean_absolute_error(SRxyz, Cxyz_t)

    total_loss = loss_xy + loss_yz
    total_loss.backward()

    optSR.step()
    optSRC.step()

    return loss_xy.item(), loss_yz.item()


@torch.no_grad()
def validate(args, epoch, out_dir, epoch_cubes, generatorSR, generatorSRC, device):
    generatorSR.eval()
    generatorSRC.eval()

    os.makedirs(f'{out_dir}/epoch-{epoch + 1}/', exist_ok=True)
    val_psnr_sr, val_psnr_src, n = 0.0, 0.0, 0

    for C, B in epoch_cubes.val_batches(args.valNum):
        Cd = downsample_batch_axis_bicubic(C, C.shape[0] // args.scale)
        fakeC = generatorSR(B)

        mse_sr = torch.mean((fakeC - Cd) ** 2)
        psnr_sr = 10 * torch.log10(4.0 / mse_sr)  # data_range=2 (values in [-1,1]) -> equivalent to tf.image.psnr(...,2)

        fakeC_q = torch.round((fakeC + 1) * 127.5)
        fakeC_q = fakeC_q / 127.5 - 1

        fakeC_t = fakeC_q.permute(2, 1, 0, 3).contiguous()
        B_t = B.permute(2, 1, 0, 3).contiguous()
        C_t = C.permute(2, 1, 0, 3).contiguous()

        fakeC_clean = generatorSRC(fakeC_t)
        mse_src = torch.mean((fakeC_clean - C_t) ** 2)
        psnr_src = 10 * torch.log10(4.0 / mse_src)

        val_psnr_sr += psnr_sr.item()
        val_psnr_src += psnr_src.item()
        n += 1

        def to_uint8(t):
            return np.squeeze(((t.cpu().numpy() + 1) * 127.5).round().astype('uint8'))

        tifffile.imwrite(f'{out_dir}/epoch-{epoch + 1}/{n}-Bxy.tif', to_uint8(B_t))
        tifffile.imwrite(f'{out_dir}/epoch-{epoch + 1}/{n}-Cxyz.tif', to_uint8(Cd))
        tifffile.imwrite(f'{out_dir}/epoch-{epoch + 1}/{n}-Ctxyz.tif', to_uint8(C_t))
        tifffile.imwrite(f'{out_dir}/epoch-{epoch + 1}/{n}-BSRxy.tif', to_uint8(fakeC))
        tifffile.imwrite(f'{out_dir}/epoch-{epoch + 1}/{n}-BSRxytd.tif', to_uint8(fakeC_t))
        tifffile.imwrite(f'{out_dir}/epoch-{epoch + 1}/{n}-BSRxyz.tif', to_uint8(fakeC_clean))

        sys.stdout.write("\rIter: %4d, Test: PSNR-SR: %4.4f, PSNR-SRC: %4.4f" % (n, psnr_sr.item(), psnr_src.item()))
        sys.stdout.flush()

    sys.stdout.write("\n")
    generatorSR.train()
    generatorSRC.train()
    if n > 0:
        print(f'Mean Validation PSNR-SR: {val_psnr_sr / n}, PSNR-SRC: {val_psnr_src / n}')


def main():
    args = parse_args()
    device = resolve_device(args)
    print(f'Using device: {device}')

    if args.ganFlag:
        print('WARNING: --ganFlag is True, but this PyTorch conversion does not yet implement '
              'the discriminators/adversarial losses (only the L1 part). Running without GAN.')

    if args.mixedPrecision:
        print('WARNING: --mixedPrecision is True, but PyTorch AMP is not wired up in this script '
              'yet; running in float32.')

    generatorSR = create_sr_generator(args, device)
    generatorSRC = create_src_generator(args, device)
    optSR = torch.optim.Adam(generatorSR.parameters(), lr=args.lr)
    optSRC = torch.optim.Adam(generatorSRC.parameters(), lr=args.lr)

    trainingDir = f"./{args.checkpoint_dir}/{args.modelName}/"
    os.makedirs(trainingDir, exist_ok=True)

    if args.continue_train:
        print(f'Loading checkpoints from {trainingDir} for epoch {args.continueEpoch}')
        try:
            generatorSR.load_state_dict(torch.load(f'{trainingDir}/GSR-{args.continueEpoch}.pt', map_location=device))
            generatorSRC.load_state_dict(torch.load(f'{trainingDir}/GSRC-{args.continueEpoch}.pt', map_location=device))
        except FileNotFoundError:
            print('Could not load SR related weights')

    valoutDir = [p for p in args.dataset_dir.split('/') if p][-1]
    rightNow = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    trainOutputDir = f'./training_outputs/{rightNow}-torchNN-{valoutDir}-{args.modelName}/'
    os.makedirs(trainOutputDir, exist_ok=True)

    print('Dataset will be fully preloaded into RAM (same as the original script)')
    LRxy = np.load(glob(args.dataset_dir + 'LR/LR.npy')[0])
    HR = np.load(glob(args.dataset_dir + 'HR/HR.npy')[0])

    print(f'Loading held-out validation dataset from {args.val_dataset_dir}')
    LRxy_val = np.load(glob(args.val_dataset_dir + 'LR/LR.npy')[0])
    HR_val = np.load(glob(args.val_dataset_dir + 'HR/HR.npy')[0])

    start_time = time.time()
    for epoch in range(args.epoch):
        # crop/batch size for this epoch: same as the "no GAN" branch of the original
        totalPerBatchVoxels = args.fine_size * args.fine_size * args.batch_size
        minPerDimSize = args.scale * 2
        maxPerDimSize = args.fine_size
        batchSizeThisEpoch = int(np.floor(np.random.rand() * (maxPerDimSize - minPerDimSize)) + minPerDimSize)
        fineSizeThisEpoch = int(np.floor(np.sqrt(totalPerBatchVoxels / batchSizeThisEpoch)))

        print(f'Generating training cubes, block size this epoch: '
              f'{batchSizeThisEpoch} x {fineSizeThisEpoch} x {fineSizeThisEpoch} -> {args.scale}x')
        epoch_cubes = make_epoch_cubes(args, HR, LRxy, batchSizeThisEpoch, fineSizeThisEpoch, args.scale, device)

        lr = args.lr * 0.5 ** (epoch / args.epoch_step)
        for g in optSR.param_groups:
            g['lr'] = lr
        for g in optSRC.param_groups:
            g['lr'] = lr
        print(f'Learning Rate: {lr:.4e}')

        tot_xy, tot_yz, num_batches = 0.0, 0.0, 0
        target_batches = args.itersPerEpoch * args.iterCyclesPerEpoch
        last_time = time.time()
        while num_batches < target_batches:
            for hr, lr_ in epoch_cubes.iter_batches():
                if num_batches >= target_batches:
                    break
                num_batches += 1
                loss_xy, loss_yz = train_step(args, lr_, hr, generatorSR, generatorSRC, optSR, optSRC)
                tot_xy += loss_xy
                tot_yz += loss_yz

                current_time = time.time()
                speed = 1.0 / max(current_time - last_time, 1e-8)
                last_time = current_time
                sys.stdout.write("\rEpoch: %4d, Iter: %4d, Time: %4.4f, Speed: %4.4f its/s, "
                                  "GSRxyL: %4.4f, GSRyzL: %4.4f" %
                                  (epoch + 1, num_batches, current_time - start_time, speed, loss_xy, loss_yz))
                sys.stdout.flush()

        sys.stdout.write("\n")
        print('Mean Epoch Performance: GSRxyL: %4.4f, GSRyzL: %4.4f' %
              (tot_xy / num_batches, tot_yz / num_batches))

        if np.mod(epoch + 1, args.print_freq) == 0 or epoch == 0:
            # fresh validation cubes, generated from the held-out dataset
            # (args.val_dataset_dir), using the same block size as this epoch
            val_cubes = make_epoch_cubes(args, HR_val, LRxy_val, batchSizeThisEpoch,
                                          fineSizeThisEpoch, args.scale, device)
            validate(args, epoch, trainOutputDir, val_cubes, generatorSR, generatorSRC, device)

        if epoch % args.save_freq == 0:
            print('Saving network weights (archive)')
            torch.save(generatorSR.state_dict(), f'{trainingDir}/GSR-{epoch}.pt')
            torch.save(generatorSRC.state_dict(), f'{trainingDir}/GSRC-{epoch}.pt')
            print('Saving network weights (rewritable checkpoint)')
            torch.save(generatorSR.state_dict(), f'{trainingDir}/GSR.pt')
            torch.save(generatorSRC.state_dict(), f'{trainingDir}/GSRC.pt')


if __name__ == '__main__':
    main()

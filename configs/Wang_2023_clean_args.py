"""
configs/Wang_2023_args_clean.py
"""

import argparse

def args():
    parser = argparse.ArgumentParser(description='Wang 2023 - EDSR SR training (xy -> xyz)')

    # ---------------------------------------------------------------
    # Run identity / phase
    # ---------------------------------------------------------------
    parser.add_argument('--modelName', dest='modelName', type=str, default='Wang2023_EDSR',
                         help='model/experiment name, used to name folders and metrics CSVs')
    parser.add_argument('--phase', dest='phase', type=str, default='train',
                         choices=['train', 'test'],
                         help="'train' runs the training loop; any other value skips the training block")

    # ---------------------------------------------------------------
    # Hardware
    # ---------------------------------------------------------------
    parser.add_argument('--gpuIDs', dest='gpuIDs', type=str, default='0',
                         help="comma-separated GPU IDs to use, e.g. '0,1,2'")
    parser.add_argument('--mixedPrecision', dest='mixedPrecision', action='store_true',
                         default=False,
                         help='use tf.keras.mixed_precision (float16) if set')

    # ---------------------------------------------------------------
    # Data
    # ---------------------------------------------------------------
    parser.add_argument('--dataset_dir', dest='dataset_dir', type=str,
                         default='./datasets/Wang2023/',
                         help="dataset root folder; expects subfolders "
                              "'training/LR/LR.npy', 'training/HR/HR.npy', "
                              "'validation/LR/LR.npy', 'validation/HR/HR.npy'")

    # ---------------------------------------------------------------
    # Model / architecture
    # ---------------------------------------------------------------
    parser.add_argument('--scale', dest='scale', type=int, default=4,
                         choices=[2, 3, 4, 8],
                         help='super-resolution scale factor')
    parser.add_argument('--ngsrf', dest='ngsrf', type=int, default=64,
                         help='number of filters in the SR generator (xy)')
    parser.add_argument('--numResBlocks', dest='numResBlocks', type=int, default=8,
                         help='number of residual blocks in the SR generator (xy)')

    # ---------------------------------------------------------------
    # Optimization
    # ---------------------------------------------------------------
    parser.add_argument('--lr', dest='lr', type=float, default=1e-4,
                         help='initial learning rate')
    parser.add_argument('--epoch', dest='epoch', type=int, default=200,
                         help='total number of training epochs')
    parser.add_argument('--epoch_step', dest='epoch_step', type=int, default=50,
                         help='epochs for lr to halve (exponential decay)')

    # ---------------------------------------------------------------
    # Batches / training cubes
    # ---------------------------------------------------------------
    parser.add_argument('--fine_size', dest='fine_size', type=int, default=64,
                         help='max crop size per dimension (used to compute '
                              'trainingBatchSize/trainingFineSize each epoch)')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=4,
                         help='used together with fine_size to compute total voxels per batch')
    parser.add_argument('--itersPerEpoch', dest='itersPerEpoch', type=int, default=100,
                         help='number of training cubes generated per createTrainingCubes2 call')
    parser.add_argument('--iterCyclesPerEpoch', dest='iterCyclesPerEpoch', type=int, default=1,
                         help='how many times the training dataset is iterated over within one epoch')

    # ---------------------------------------------------------------
    # Validation / logging / checkpoints
    # ---------------------------------------------------------------
    parser.add_argument('--valNum', dest='valNum', type=int, default=8,
                         help='number of validation cubes/batches per validation epoch')
    parser.add_argument('--valTest', dest='valTest', action='store_true', default=False,
                         help='if set, also generates an extra test .tif on each validation epoch')
    parser.add_argument('--print_freq', dest='print_freq', type=int, default=5,
                         help='run validation/print PSNR-SSIM every N epochs')
    parser.add_argument('--save_freq', dest='save_freq', type=int, default=10,
                         help='save model weights/checkpoints every N epochs')
    parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', type=str, default='checkpoints',
                         help='root (relative) folder where checkpoints are saved')

    parsed_args, _ = parser.parse_known_args()
    return parsed_args
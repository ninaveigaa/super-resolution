"""
Generic dual-solver training script.

Trains a pair of models jointly -- a 2D solver (handling the XY plane) and
a 1D solver (handling the Z axis) -- following a DualEDSR-style two-stage
architecture. This script is NOT specific to any one model: any 2D/1D
solver pair can be plugged in via --model_module/--model_class/--model_kwargs,
so no solver-specific hyperparameters (e.g. feature counts, block counts)
are hardcoded here.

Data pipeline: standard PyTorch Dataset/DataLoader, ported from a numpy
reference implementation (createTrainingCubes2). Each dataset item is a
single random block of shape (crop_size, crop_size, bc_depth), matching
the model's expected [Nx, Ny, Nz] input exactly -- bc_depth always ends
up as the last axis (Nz), which is the axis the model treats as a batch
of 2D slices internally. The model has no separate batch or channel
dimension of its own (it adds the channel dim itself via unsqueeze), so
--batch_size here controls gradient accumulation across samples rather
than a literal batched forward pass.
"""

import argparse
import csv
import importlib
import json
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset


def build_model(model_module, model_class, model_kwargs, scale, device):
    """Dynamically import and instantiate the solver-pair model.

    Keeps this script agnostic to any one solver: the caller decides which
    model class to use and what architecture-specific kwargs (feature
    counts, block counts, etc.) it needs, passed in as a JSON string via
    --model_kwargs, rather than those kwargs being declared as named
    arguments in this file.
    """
    module = importlib.import_module(model_module)
    cls = getattr(module, model_class)
    kwargs = dict(model_kwargs)
    kwargs.setdefault('scale', scale)
    return cls(**kwargs).to(device)


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(description='DualEDSR Training')

    # Data
    parser.add_argument('--dataset_dir',    type=str, required=True,
                         help='Directory containing the training low_res/high_res volumes')
    parser.add_argument('--val_dataset_dir', type=str, default=None,
                         help='Directory containing a SEPARATE held-out low_res/high_res volume '
                              'for validation (same filenames as --low_res/--high_res, but in this '
                              'directory). If omitted, falls back to sampling validation cubes from '
                              'the training volume itself -- fine for a smoke test, but not a real '
                              'held-out validation set.')
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--model_name',     type=str, required=True)
    parser.add_argument('--low_res',    type=str, required=True,
                         help='Filename (within dataset_dir / val_dataset_dir) of the low-res volume, e.g. lr.npy')
    parser.add_argument('--high_res',   type=str, required=True,
                         help='Filename (within dataset_dir / val_dataset_dir) of the high-res volume, e.g. hr.npy')

    # Model -- kept generic on purpose. Any 2D/1D solver pair can be
    # plugged in without editing this script: point --model_module /
    # --model_class at the class to instantiate, and pass whatever
    # architecture-specific hyperparameters it needs (feature counts,
    # block counts, etc.) as a JSON object via --model_kwargs.
    parser.add_argument('--model_module', type=str, default='edsr',
                         help='Python module to import the model class from')
    parser.add_argument('--model_class',  type=str, default='DualEDSR',
                         help='Name of the model class within model_module')
    parser.add_argument('--model_kwargs', type=json.loads, default='{}',
                         help='JSON object of kwargs to pass to the model constructor, '
                              'e.g. \'{"n_feats": 64, "n_resblocks": 16}\'')

    # Training
    parser.add_argument('--epochs',          type=int,   default=500)
    parser.add_argument('--iters_per_epoch', type=int,   default=300)
    parser.add_argument('--iter_cycles',     type=int,   default=3)
    parser.add_argument('--batch_size',      type=int,   default=4,
                         help='Number of sampled volumes to average gradients over per '
                              'optimizer step (the model takes one volume per forward call, '
                              'so this is gradient accumulation, not a literal batched forward)')
    parser.add_argument('--bc_depth',        type=int,   default=16,
                         help='Depth along the axis that gets folded into the model\'s '
                              'batch/channel dimension (was "batchsize" in the numpy version)')
    parser.add_argument('--crop_size',       type=int,   default=64,
                         help='Side length of the 2D spatial crop (was "cropsize" in the numpy version)')
    parser.add_argument('--lr',              type=float, default=1e-4)
    parser.add_argument('--epoch_step',      type=int,   default=50)
    parser.add_argument('--scale',           type=int,   default=4)
    parser.add_argument('--num_workers',     type=int,   default=4)

    # Logging
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--val_num',   type=int, default=5)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(dataset_dir, low_res_name, high_res_name):
    """Load the LR/HR volumes as numpy arrays.

    Expects .npy files. Swap this out if your volumes live in another
    format (e.g. raw binary, HDF5, DICOM series).
    """
    lr_path = os.path.join(dataset_dir, low_res_name)
    hr_path = os.path.join(dataset_dir, high_res_name)
    lr = np.load(lr_path).astype('float32')
    hr = np.load(hr_path).astype('float32')
    return lr, hr


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class VolumeDataset(Dataset):
    """Yields one random (LR, HR) block per __getitem__, ported from the
    numpy `createTrainingCubes2` reference implementation -- reshaped to
    match DualEDSR.forward's expected [Nx, Ny, Nz] input exactly.

    Each item is a block with two distinct axis roles:
      - `crop_size`  -- the true 2D spatial crop (the first two returned
                        axes, Nx and Ny).
      - `bc_depth`   -- depth along whichever physical axis is currently
                        playing the "batch of Z-slices" role (always
                        returned as the last axis, Nz, since that's the
                        axis DualEDSR.forward slices along internally via
                        `x_lr.permute(2, 0, 1)`).
    No channel dimension is added here -- the model adds it itself via
    `.unsqueeze(1)`.

    Orientation cycles through which physical axis plays the bc_depth
    role across successive indices (idx % 3), giving the 2D+1D solver
    pair exposure to the volume from all three directions.

    `__len__` is `iters_per_epoch` (or `val_num` for the validation
    dataset) -- since sampling is random, this is a "virtual" epoch
    length rather than a real dataset size.
    """

    def __init__(self, low_res, high_res, bc_depth, crop_size, iters_per_epoch, scale):
        self.lr = low_res
        self.hr = high_res
        self.bc_depth = bc_depth
        self.crop_size = crop_size
        self.iters_per_epoch = iters_per_epoch
        self.scale = scale

    def __len__(self):
        return self.iters_per_epoch

    def __getitem__(self, idx):
        d = self.bc_depth
        c = self.crop_size
        s = self.scale
        orientation = idx % 3

        if orientation == 0:  # depth along physical X
            x = np.random.randint(0, self.lr.shape[0] - d)
            y = np.random.randint(0, self.lr.shape[1] - c)
            z = np.random.randint(0, self.lr.shape[2] - c)

            block_lr = self.lr[x:x + d, y:y + c, z:z + c]
            block_hr = self.hr[x * s:x * s + d * s,
                                y * s:y * s + c * s,
                                z * s:z * s + c * s]

            # (X_depth, Y, Z) -> (Y, Z, X_depth): depth axis moves to last
            block_lr = np.transpose(block_lr, [1, 2, 0])
            block_hr = np.transpose(block_hr, [1, 2, 0])

        elif orientation == 1:  # depth along physical Z
            x = np.random.randint(0, self.lr.shape[0] - c)
            y = np.random.randint(0, self.lr.shape[1] - c)
            z = np.random.randint(0, self.lr.shape[2] - d)

            block_lr = self.lr[x:x + c, y:y + c, z:z + d]
            block_hr = self.hr[x * s:x * s + c * s,
                                y * s:y * s + c * s,
                                z * s:z * s + d * s]
            # (X, Y, Z_depth): depth is already last, no transpose needed

        else:  # orientation == 2, depth along physical Y
            x = np.random.randint(0, self.lr.shape[0] - c)
            y = np.random.randint(0, self.lr.shape[1] - d)
            z = np.random.randint(0, self.lr.shape[2] - c)

            block_lr = self.lr[x:x + c, y:y + d, z:z + c]
            block_hr = self.hr[x * s:x * s + c * s,
                                y * s:y * s + d * s,
                                z * s:z * s + c * s]

            # (X, Y_depth, Z) -> (X, Z, Y_depth): depth axis moves to last
            block_lr = np.transpose(block_lr, [0, 2, 1])
            block_hr = np.transpose(block_hr, [0, 2, 1])

        # np.ascontiguousarray avoids issues with the negative-stride /
        # non-contiguous views produced by np.transpose. Normalize to
        # [-1, 1] as in the reference.
        block_lr = np.ascontiguousarray(block_lr) / 127.5 - 1.0
        block_hr = np.ascontiguousarray(block_hr) / 127.5 - 1.0

        return (
            torch.from_numpy(block_lr.astype('float32')),
            torch.from_numpy(block_hr.astype('float32')),
        )


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def train_step(model, optimizer, lr_batch, hr_batch, device):
    """
    lr_batch : [B, Nx, Ny, Nz]        torch tensor (from DataLoader)
    hr_batch : [B, Nx*S, Ny*S, Nz*S]  torch tensor (from DataLoader)

    DualEDSR.forward takes exactly one volume [Nx, Ny, Nz] with no batch
    dimension of its own, so we can't fold the DataLoader's batch axis
    into a single forward call. Instead we loop over the B samples,
    backpropagating each one's (loss / B) so gradients accumulate across
    the batch before a single optimizer.step() -- gradient accumulation,
    which gives --batch_size a similar effect to a real batch size (more
    samples averaged per update) without requiring the model to support
    batching.
    """
    b = lr_batch.shape[0]
    optimizer.zero_grad()

    total_loss = total_lxy = total_lxyz = 0.0
    for i in range(b):
        x_lr = lr_batch[i].to(device)
        i_hr = hr_batch[i].to(device)

        sr_xy, sr_xyz = model(x_lr)
        loss, l_xy, l_xyz = model.compute_losses(sr_xy, sr_xyz, i_hr)
        (loss / b).backward()

        total_loss += loss.item()
        total_lxy  += l_xy.item()
        total_lxyz += l_xyz.item()

    optimizer.step()
    return total_loss / b, total_lxy / b, total_lxyz / b


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(model, val_loader, val_num, device):
    model.eval()
    psnr_xy_total  = 0.0
    psnr_xyz_total = 0.0
    n = 0

    with torch.no_grad():
        for i, (lr_batch, hr_batch) in enumerate(val_loader):
            if i >= val_num:
                break

            # val_loader uses batch_size=1, so index [0] to get the single
            # [Nx,Ny,Nz] volume DualEDSR.forward expects.
            x_lr = lr_batch[0].to(device)
            i_hr = hr_batch[0].to(device)

            sr_xy, sr_xyz = model(x_lr)

            # Intermediate target: HR downsampled in Z
            Nz  = sr_xy.shape[0]
            NxS = sr_xy.shape[2]
            NyS = sr_xy.shape[3]
            NzS = i_hr.shape[2]
            tmp = (
                i_hr.permute(2, 0, 1)
                .reshape(NzS, NxS * NyS)
                .permute(1, 0).unsqueeze(0)
            )
            tmp = F.interpolate(tmp, size=Nz, mode='linear', align_corners=False)
            hr_d = tmp.squeeze(0).permute(1, 0).reshape(Nz, NxS, NyS)

            # PSNR (data range = 2.0 for [-1, 1])
            mse_xy  = F.mse_loss(sr_xy.squeeze(1), hr_d).item()
            mse_xyz = F.mse_loss(sr_xyz, i_hr).item()
            psnr_xy  = 10 * np.log10(4.0 / (mse_xy  + 1e-8))
            psnr_xyz = 10 * np.log10(4.0 / (mse_xyz + 1e-8))

            psnr_xy_total  += psnr_xy
            psnr_xyz_total += psnr_xyz
            n += 1

    model.train()
    n = max(n, 1)
    return psnr_xy_total / n, psnr_xyz_total / n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Checkpoint directory
    save_dir = os.path.join(args.checkpoint_dir, args.model_name)
    os.makedirs(save_dir, exist_ok=True)

    # CSV log file (loss + PSNR per epoch)
    log_path = os.path.join(save_dir, 'training_log.csv')
    log_exists = os.path.isfile(log_path)
    log_file = open(log_path, 'a', newline='')
    log_writer = csv.writer(log_file)
    if not log_exists:
        log_writer.writerow([
            'epoch', 'lr', 'loss', 'loss_xy', 'loss_xyz',
            'psnr_xy', 'psnr_xyz',
        ])

    # Model -- dynamically loaded so this script isn't tied to one solver.
    model = build_model(
        args.model_module, args.model_class, args.model_kwargs, args.scale, device,
    )
    print(f'Model: {args.model_module}.{args.model_class} | '
          f'Parameters: {sum(p.numel() for p in model.parameters()):,}')

    optimizer = Adam(model.parameters(), lr=args.lr)

    # Data
    lr_data, hr_data = load_data(args.dataset_dir, args.low_res, args.high_res)

    if args.val_dataset_dir:
        val_lr_data, val_hr_data = load_data(args.val_dataset_dir, args.low_res, args.high_res)
    else:
        print('WARNING: --val_dataset_dir not given -- validation cubes will be sampled '
              'from the TRAINING volume (no real held-out set). Pass --val_dataset_dir '
              'to validate on a separate volume.')
        val_lr_data, val_hr_data = lr_data, hr_data

    train_dataset = VolumeDataset(
        lr_data, hr_data, args.bc_depth, args.crop_size, args.iters_per_epoch, args.scale,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    # A small, separate dataset/loader for validation so val patches don't
    # collide with the training generator's state.
    val_dataset = VolumeDataset(
        val_lr_data, val_hr_data, args.bc_depth, args.crop_size, args.val_num, args.scale,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    # Training loop
    start_time = time.time()

    for epoch in range(args.epochs):

        # Learning rate decay: lr * 0.5^(epoch / epoch_step)
        current_lr = args.lr * (0.5 ** (epoch / args.epoch_step))
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        total_loss = 0.0
        total_lxy  = 0.0
        total_lxyz = 0.0
        n_iters = 0

        for cycle in range(args.iter_cycles):
            for lr_batch, hr_batch in train_loader:
                loss, l_xy, l_xyz = train_step(model, optimizer, lr_batch, hr_batch, device)

                total_loss += loss
                total_lxy  += l_xy
                total_lxyz += l_xyz
                n_iters    += 1

                elapsed = time.time() - start_time
                print(
                    f'\rEpoch {epoch+1:4d}/{args.epochs} | '
                    f'Iter {n_iters:5d} | '
                    f'Time {elapsed:7.1f}s | '
                    f'LR {current_lr:.2e} | '
                    f'Loss {loss:.4f} (xy {l_xy:.4f} xyz {l_xyz:.4f})',
                    end='', flush=True,
                )

        print()
        avg_loss = total_loss / n_iters
        avg_lxy  = total_lxy  / n_iters
        avg_lxyz = total_lxyz / n_iters
        print(
            f'Epoch {epoch+1:4d} avg — '
            f'Loss {avg_loss:.4f} | Lxy {avg_lxy:.4f} | Lxyz {avg_lxyz:.4f}'
        )

        # Validation
        psnr_xy, psnr_xyz = None, None
        if (epoch + 1) % args.save_freq == 0 or epoch == 0:
            psnr_xy, psnr_xyz = validate(model, val_loader, args.val_num, device)
            print(f'Validation — PSNR-xy {psnr_xy:.2f} dB | PSNR-xyz {psnr_xyz:.2f} dB')

            # Save checkpoint
            ckpt_path = os.path.join(save_dir, f'epoch_{epoch+1:04d}.pt')
            torch.save({
                'epoch':       epoch + 1,
                'model':       model.state_dict(),
                'optimizer':   optimizer.state_dict(),
                'loss':        avg_loss,
            }, ckpt_path)
            print(f'Checkpoint saved: {ckpt_path}')

        # Log this epoch to CSV (PSNR columns left blank on non-validation epochs)
        log_writer.writerow([
            epoch + 1,
            current_lr,
            avg_loss,
            avg_lxy,
            avg_lxyz,
            psnr_xy  if psnr_xy  is not None else '',
            psnr_xyz if psnr_xyz is not None else '',
        ])
        log_file.flush()

    log_file.close()
    print('Training complete.')
    print(f'Training log saved to: {log_path}')


if __name__ == '__main__':
    main()
"""
Generic dual-solver training script.

Trains a pair of models jointly -- a 2D solver (handling the XY plane) and
a 1D solver (handling the Z axis) -- following the DualEDSR-style two-stage
architecture. This script is NOT specific to any one model: any 2D/1D
solver pair can be plugged in (e.g. EDSR + EDSR1D). The actual model
classes are imported and instantiated by the job submission script that
calls this training harness, not hardcoded here.

Usage:
    python DualTraining.py \
        --dataset_dir /path/to/data/ \
        --checkpoint_dir /path/to/checkpoints/ \
        --model_name dual_edsr \
        --epochs 500 \
        --iters_per_epoch 300 \
        --iter_cycles 3 \
        --batch_size 16 \
        --patch_size 64 \
        --lr 1e-4 \
        --epoch_step 50 \
        --scale 4 \
        --n_feats 64 \
        --n_resblocks 16 \
        --save_freq 10 \
        --val_num 5
"""

import argparse
import csv
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from edsr import EDSR, EDSR1D, DualEDSR


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(description='DualEDSR Training')

    # Data
    parser.add_argument('--dataset_dir',    type=str, required=True)
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--model_name',     type=str, default='dual_edsr')

    # Training
    parser.add_argument('--epochs',          type=int,   default=500)
    parser.add_argument('--iters_per_epoch', type=int,   default=300)
    parser.add_argument('--iter_cycles',     type=int,   default=3)
    parser.add_argument('--batch_size',      type=int,   default=16)
    parser.add_argument('--patch_size',      type=int,   default=64)
    parser.add_argument('--lr',              type=float, default=1e-4)
    parser.add_argument('--epoch_step',      type=int,   default=50)
    parser.add_argument('--scale',           type=int,   default=4)

    # Model
    parser.add_argument('--n_feats',     type=int, default=64)
    parser.add_argument('--n_resblocks', type=int, default=16)

    # Logging
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--val_num',   type=int, default=5)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_data(dataset_dir):
    lr = np.load(os.path.join(dataset_dir, 'LR', 'LR.npy'))
    hr = np.load(os.path.join(dataset_dir, 'HR', 'HR.npy'))
    print(f'LR shape: {lr.shape}')
    print(f'HR shape: {hr.shape}')
    return lr, hr


def create_training_batches(lr, hr, batch_size, patch_size, iters_per_epoch, scale):
    """
    Extracts random patches from the LR/HR volumes, cycling through
    XY, YZ and XZ orientations (as in the original TF implementation).

    Returns:
        batch_lr : [iters_per_epoch * batch_size, patch_size, patch_size]
        batch_hr : [iters_per_epoch * batch_size * scale, patch_size * scale, patch_size * scale]
    """
    total_lr = iters_per_epoch * batch_size
    total_hr = iters_per_epoch * batch_size * scale

    batch_lr = np.zeros([total_lr, patch_size, patch_size], dtype='float32')
    batch_hr = np.zeros([total_hr, patch_size * scale, patch_size * scale], dtype='float32')

    n_lr = 0
    n_hr = 0

    for i in range(iters_per_epoch):
        orientation = i % 3

        if orientation == 0:  # XY plane
            x = np.random.randint(0, lr.shape[0] - batch_size)
            y = np.random.randint(0, lr.shape[1] - patch_size)
            z = np.random.randint(0, lr.shape[2] - patch_size)
            block_lr = lr[x:x+batch_size,        y:y+patch_size,        z:z+patch_size]
            block_hr = hr[x*scale:(x+batch_size)*scale,
                          y*scale:(y+patch_size)*scale,
                          z*scale:(z+patch_size)*scale]

        elif orientation == 1:  # YZ plane
            x = np.random.randint(0, lr.shape[0] - patch_size)
            y = np.random.randint(0, lr.shape[1] - patch_size)
            z = np.random.randint(0, lr.shape[2] - batch_size)
            block_lr = lr[x:x+patch_size, y:y+patch_size, z:z+batch_size]
            block_hr = hr[x*scale:(x+patch_size)*scale,
                          y*scale:(y+patch_size)*scale,
                          z*scale:(z+batch_size)*scale]
            block_lr = np.transpose(block_lr, [2, 0, 1])
            block_hr = np.transpose(block_hr, [2, 0, 1])

        else:  # XZ plane
            x = np.random.randint(0, lr.shape[0] - patch_size)
            y = np.random.randint(0, lr.shape[1] - batch_size)
            z = np.random.randint(0, lr.shape[2] - patch_size)
            block_lr = lr[x:x+patch_size, y:y+batch_size, z:z+patch_size]
            block_hr = hr[x*scale:(x+patch_size)*scale,
                          y*scale:(y+batch_size)*scale,
                          z*scale:(z+patch_size)*scale]
            block_lr = np.transpose(block_lr, [1, 0, 2])
            block_hr = np.transpose(block_hr, [1, 0, 2])

        # Normalize to [-1, 1]
        batch_lr[n_lr:n_lr+batch_size]            = block_lr / 127.5 - 1.0
        batch_hr[n_hr:n_hr+batch_size*scale]      = block_hr / 127.5 - 1.0

        n_lr += batch_size
        n_hr += batch_size * scale

    return batch_lr, batch_hr


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def train_step(model, optimizer, lr_batch, hr_batch, device):
    """
    lr_batch : [batch_size, patch_size, patch_size]          numpy
    hr_batch : [batch_size*scale, patch_size*scale, patch_size*scale]  numpy
    """
    lr_vol = torch.from_numpy(lr_batch).to(device)  # [Nx, Ny, Nz]
    hr_vol = torch.from_numpy(hr_batch).to(device)  # [Nx*S, Ny*S, Nz*S]

    optimizer.zero_grad()
    sr_xy, sr_xyz = model(lr_vol)
    loss, l_xy, l_xyz = model.compute_losses(sr_xy, sr_xyz, hr_vol)
    loss.backward()
    optimizer.step()

    return loss.item(), l_xy.item(), l_xyz.item()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(model, lr_batches, hr_batches, val_num, device):
    model.eval()
    psnr_xy_total  = 0.0
    psnr_xyz_total = 0.0

    with torch.no_grad():
        for i in range(min(val_num, len(lr_batches))):
            lr_vol = torch.from_numpy(lr_batches[i]).to(device)
            hr_vol = torch.from_numpy(hr_batches[i]).to(device)

            sr_xy, sr_xyz = model(lr_vol)

            # Intermediate target: HR downsampled in Z
            Nz  = sr_xy.shape[0]
            NxS = sr_xy.shape[2]
            NyS = sr_xy.shape[3]
            NzS = hr_vol.shape[2]
            tmp = (
                hr_vol.permute(2, 0, 1)
                .reshape(NzS, NxS * NyS)
                .permute(1, 0).unsqueeze(0)
            )
            tmp = F.interpolate(tmp, size=Nz, mode='linear', align_corners=False)
            hr_d = tmp.squeeze(0).permute(1, 0).reshape(Nz, NxS, NyS)

            # PSNR (data range = 2.0 for [-1, 1])
            mse_xy  = F.mse_loss(sr_xy.squeeze(1), hr_d).item()
            mse_xyz = F.mse_loss(sr_xyz, hr_vol).item()
            psnr_xy  = 10 * np.log10(4.0 / (mse_xy  + 1e-8))
            psnr_xyz = 10 * np.log10(4.0 / (mse_xyz + 1e-8))

            psnr_xy_total  += psnr_xy
            psnr_xyz_total += psnr_xyz

    model.train()
    n = min(val_num, len(lr_batches))
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

    # Model
    model = DualEDSR(
        n_feats=args.n_feats,
        n_resblocks=args.n_resblocks,
        scale=args.scale,
    ).to(device)
    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')

    optimizer = Adam(model.parameters(), lr=args.lr)

    # Data
    lr_data, hr_data = load_data(args.dataset_dir)

    # Training loop
    start_time = time.time()

    for epoch in range(args.epochs):

        # Learning rate decay: lr * 0.5^(epoch / epoch_step)
        current_lr = args.lr * (0.5 ** (epoch / args.epoch_step))
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        # Build batches for this epoch
        batch_lr, batch_hr = create_training_batches(
            lr_data, hr_data,
            args.batch_size, args.patch_size,
            args.iters_per_epoch, args.scale,
        )

        # Training iterations
        total_loss = 0.0
        total_lxy  = 0.0
        total_lxyz = 0.0
        n_iters = 0

        for cycle in range(args.iter_cycles):
            for i in range(args.iters_per_epoch):
                lr_batch = batch_lr[i*args.batch_size : (i+1)*args.batch_size]
                hr_batch = batch_hr[i*args.batch_size*args.scale : (i+1)*args.batch_size*args.scale]

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
            # Build small validation set from the same batch
            val_lr = [batch_lr[i*args.batch_size:(i+1)*args.batch_size]
                      for i in range(min(args.val_num, args.iters_per_epoch))]
            val_hr = [batch_hr[i*args.batch_size*args.scale:(i+1)*args.batch_size*args.scale]
                      for i in range(min(args.val_num, args.iters_per_epoch))]

            psnr_xy, psnr_xyz = validate(model, val_lr, val_hr, args.val_num, device)
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
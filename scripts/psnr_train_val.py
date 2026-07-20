"""
scripts/psnr_train_val.py

Plots train PSNR (derived from the train loss) together with validation
PSNR (already present in the metrics csv), per epoch, up to a given epoch
limit. Saved as a static .png.

Usage:
    python scripts/psnr_train_val.py --csv_path measurements/metrics/Wang_2023/Wang_2023_260716_1917_metrics.csv
    python scripts/psnr_train_val.py --csv_path path/to/metrics.csv --max_epoch 200 --output psnr.png
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, required=True,
                         help='Path to the {run_id}_metrics.csv file to plot.')
    parser.add_argument('--max_epoch', type=int, default=200,
                         help='Only plot epochs up to (and including) this value.')
    parser.add_argument('--train_loss_col', type=str, default='train_loss_z',
                         help='Column used to derive the train PSNR (e.g. train_loss_xy or train_loss_z).')
    parser.add_argument('--val_psnr_col', type=str, default='psnr_final',
                         help='Column used as the validation PSNR (e.g. psnr_xy or psnr_final).')
    parser.add_argument('--max_signal', type=float, default=1.0,
                         help='Max signal/pixel value used in the PSNR formula (1.0 for images normalized to [0, 1]).')
    parser.add_argument('--val_marker_step', type=int, default=10,
                         help='Draw a validation marker every N epochs (cosmetic only; does not affect the data used).')
    parser.add_argument('--output', type=str, default=None,
                         help='Output .png path. Defaults to {csv_dir}/{run_id}_psnr_train_val.png')
    return parser.parse_args()


def loss_to_psnr(loss, max_signal=1.0):
    """Converts an MSE loss into PSNR (dB): PSNR = 10 * log10(MAX^2 / MSE)."""
    loss = np.asarray(loss, dtype=float)
    return 10.0 * np.log10((max_signal ** 2) / loss)


def main():
    args = get_args()
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f'{csv_path} not found.')
    print(f'Reading: {csv_path}')

    df = pd.read_csv(csv_path)
    run_id = df['run_id'].iloc[0]

    df = df[df['epoch'] <= args.max_epoch].sort_values('epoch').reset_index(drop=True)
    if df.empty:
        raise ValueError(f'No rows with epoch <= {args.max_epoch} in {csv_path}.')

    df['train_psnr'] = loss_to_psnr(df[args.train_loss_col], args.max_signal)
    df['val_psnr'] = df[args.val_psnr_col]

    val_markers = df.iloc[::args.val_marker_step]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(df['epoch'], df['train_psnr'], color='tab:blue', linewidth=1,
            label='DualEDSR (train)')

    ax.scatter(val_markers['epoch'], val_markers['val_psnr'],
               facecolors='none', edgecolors='tab:orange', linewidths=1.2,
               label='DualEDSR Validation')

    ax.set_title('Training and Validation per Epoch')
    ax.set_xlabel('Epochs (1,000 iterations)')
    ax.set_ylabel('PSNR (dB)')
    ax.set_xlim(0, args.max_epoch)
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    fig.suptitle(f'{run_id}')
    fig.tight_layout()

    output = Path(args.output) if args.output else csv_path.parent / f'{run_id}_psnr_train_val.png'
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    print(f'Saved: {output}')


if __name__ == '__main__':
    main()

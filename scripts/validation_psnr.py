"""
scripts/validation_psnr.py

Quick script to plot validation PSNR (per stage: xy and final) over the
first N epochs, from a {run_id}_metrics.csv file, saved as a static .png
in measurements/graphs/. Markers are drawn every `--marker_step` epochs.

Usage:
    python scripts/validation_psnr.py --model_name Wang_2023
    python scripts/validation_psnr.py --model_name Wang_2023 --run_id Wang_2023_260716_1917
    python scripts/validation_psnr.py --model_name Wang_2023 --run_id Wang_2023_260716_1917 --max_epoch 200 --marker_step 10
"""

import argparse
import glob
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--run_id', type=str, default=None,
                         help='Specific run to plot. Defaults to the most recent run for this model.')
    parser.add_argument('--base_dir', type=str, default='measurements')
    parser.add_argument('--max_epoch', type=int, default=200,
                         help='Only plot epochs up to (and including) this value.')
    parser.add_argument('--marker_step', type=int, default=10,
                         help='Draw a validation marker every N epochs.')
    parser.add_argument('--output', type=str, default=None,
                         help='Output .png path. Defaults to {base_dir}/graphs/{run_id}_val_psnr.png')
    return parser.parse_args()


def find_csv(model_name, run_id, base_dir):
    model_dir = Path(base_dir) / 'metrics' / model_name
    if run_id is not None:
        csv_path = model_dir / f'{run_id}_metrics.csv'
        if not csv_path.exists():
            raise FileNotFoundError(f'{csv_path} not found.')
        return csv_path

    candidates = sorted(glob.glob(str(model_dir / '*.csv')))
    if not candidates:
        raise FileNotFoundError(f'No *.csv files found in {model_dir}.')
    return Path(candidates[-1])  # most recent, by sort order (timestamp in run_id)


def main():
    args = get_args()
    csv_path = find_csv(args.model_name, args.run_id, args.base_dir)
    print(f'Reading: {csv_path}')

    df = pd.read_csv(csv_path)
    run_id = df['run_id'].iloc[0]

    df = df[df['epoch'] <= args.max_epoch].sort_values('epoch')
    if df.empty:
        raise ValueError(f'No rows with epoch <= {args.max_epoch} in {csv_path}.')

    # psnr_xy / psnr_final are only filled in on validation epochs -- drop
    # the epochs where they're NaN before plotting.
    psnr_df = df[df['psnr_final'].notna() | df['psnr_xy'].notna()]
    if psnr_df.empty:
        raise ValueError(f'No rows with psnr values in {csv_path} -- was validation ever run?')

    markers = psnr_df.iloc[::args.marker_step]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(markers['epoch'], markers['psnr_xy'], marker='o',
            markerfacecolor='none', color='tab:orange', label='Validation PSNR (xy)')

    ax.set_title('Validation PSNR per Epoch')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('PSNR (dB)')
    ax.set_xlim(0, args.max_epoch)
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    fig.suptitle(f'{run_id}')
    fig.tight_layout()

    graphs_dir = Path(args.base_dir) / 'graphs'
    graphs_dir.mkdir(parents=True, exist_ok=True)
    output = args.output or graphs_dir / f'{run_id}_val_psnr.png'
    fig.savefig(output, dpi=150)
    print(f'Saved: {output}')


if __name__ == '__main__':
    main()

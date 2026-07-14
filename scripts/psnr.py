"""
scripts/psnr_curves.py

Quick script to plot validation PSNR (per stage: xy and final) over epochs,
from the same {run_id}_time.csv file used by learning_curves.py. Saved as a
static .png in measurements/graphs/.

Usage:
    python scripts/psnr_curves.py --model_name Wang_2023
    python scripts/psnr_curves.py --model_name Wang_2023 --run_id Wang_2023_260713_1530
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
    parser.add_argument('--output', type=str, default=None,
                         help='Output .png path. Defaults to {base_dir}/graphs/{run_id}_psnr.png')
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

    # psnr_xy / psnr_final are only filled in on validation epochs -- drop
    # the epochs where they're NaN (pure training-only rows) before plotting.
    psnr_df = df[df['psnr_final'].notna()].sort_values('epoch')
    if psnr_df.empty:
        raise ValueError(f'No rows with psnr_final in {csv_path} -- was validation ever run?')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(psnr_df['epoch'], psnr_df['psnr_xy'], marker='o', color='#2a78d6')
    axes[0].set_title('PSNR -- step XY')
    axes[0].set_xlabel('epoch')
    axes[0].set_ylabel('PSNR (dB)')
    axes[0].grid(alpha=0.3)

    axes[1].plot(psnr_df['epoch'], psnr_df['psnr_final'], marker='o', color='#2a78d6')
    axes[1].set_title('PSNR -- step Z (final)')
    axes[1].set_xlabel('epoch')
    axes[1].set_ylabel('PSNR (dB)')
    axes[1].grid(alpha=0.3)

    fig.suptitle(f'{run_id}')
    fig.tight_layout()

    graphs_dir = Path(args.base_dir) / 'graphs'
    graphs_dir.mkdir(parents=True, exist_ok=True)
    output = args.output or graphs_dir / f'{run_id}_psnr.png'
    fig.savefig(output, dpi=150)
    print(f'Saved: {output}')


if __name__ == '__main__':
    main()

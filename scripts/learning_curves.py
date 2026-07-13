"""
scripts/learning_curves.py

Quick script to plot training + validation loss (per stage: xy and z) from
a {run_id}_metrics.csv file, saved as a static .png in measurements/graphs/.

Usage:
    python scripts/learning_curves.py --model_name Wang_2023
    python scripts/learning_curves.py --model_name Wang_2023 --run_id Wang_2023_260713_1530
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
                         help='Output .png path. Defaults to {base_dir}/graphs/{run_id}_loss.png')
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

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(df['epoch'], df['train_loss_xy'], label='train_loss_xy', color='#2a78d6')
    axes[0].plot(df['epoch'], df['val_loss_xy'], label='val_loss_xy', color='#e34948')
    axes[0].set_title('Loss -- step XY')
    axes[0].set_xlabel('epoch')
    axes[0].set_ylabel('loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(df['epoch'], df['train_loss_z'], label='train_loss_z', color='#2a78d6')
    axes[1].plot(df['epoch'], df['val_loss_z'], label='val_loss_z', color='#e34948')
    axes[1].set_title('Loss -- step Z (final)')
    axes[1].set_xlabel('epoch')
    axes[1].set_ylabel('loss')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle(f'{run_id}')
    fig.tight_layout()

    graphs_dir = Path(args.base_dir) / 'graphs'
    graphs_dir.mkdir(parents=True, exist_ok=True)
    output = args.output or graphs_dir / f'{run_id}_loss.png'
    fig.savefig(output, dpi=150)
    print(f'Saved: {output}')


if __name__ == '__main__':
    main()
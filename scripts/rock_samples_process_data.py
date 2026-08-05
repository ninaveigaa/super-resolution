"""
scripts/rock_samples_process_data.py

Converts the raw A.tiff / B.tiff volumes into numpy arrays (uint8),
saving them into the training/validation structure expected by the
pipeline:

    data/raw/rock_samples/A.tiff -> data/processed/rock_samples/training/HR/A.npy
    data/raw/rock_samples/B.tiff -> data/processed/rock_samples/validation/HR/B.npy

Usage:
    python scripts/rock_samples_process_data.py
"""

import sys
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import preprocessing

# LR volumes are 4x smaller than HR along every axis (400 -> 100).
LR_SCALE_FACTOR = 0.25

# (raw .tiff path, HR output .npy path, LR output .npy path)
PAIRS = [
    (
        "data/raw/rock_samples/A.tiff",
        "data/processed/rock_samples/training/HR/HR.npy",
        "data/processed/rock_samples/training/LR/LR.npy",
    ),
    (
        "data/raw/rock_samples/B.tiff",
        "data/processed/rock_samples/validation/HR/HR.npy",
        "data/processed/rock_samples/validation/LR/LR.npy",
    ),
]


def main():
    for tiff_path, hr_path, lr_path in PAIRS:
        hr = preprocessing.convert_ftype(tiff_path, "npy", output_path=hr_path)
        # min_val/max_val fixed at 0-255: the data is already uint8 across
        # the full range, so this just enforces the dtype without
        # rescaling the values (avoids the default contrast stretching
        # based on the array's own min/max).
        hr = preprocessing.convert_dtype(hr, np.uint8, output_path=hr_path, min_val=0, max_val=255)
        print(f"{hr_path}: shape={hr.shape}, dtype={hr.dtype}")

        lr = preprocessing.cubic_interpolation(hr, scale_factor=LR_SCALE_FACTOR, output_path=lr_path)
        # Cubic spline interpolation (order=3) can overshoot slightly
        # beyond the original 0-255 range, which would silently wrap
        # around when cast back to uint8. Clipping here guards against that.
        lr = preprocessing.convert_dtype(lr, np.uint8, output_path=lr_path, min_val=0, max_val=255)
        print(f"{lr_path}: shape={lr.shape}, dtype={lr.dtype}")


if __name__ == "__main__":
    main()

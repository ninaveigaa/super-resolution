"""
scripts/visual_check_dataset_dualedsr_tf.py

Quick utility: converts the final prepared .npy files (training/validation,
LR/HR) back to .tif, for visual inspection in an image viewer (ImageJ/Fiji,
napari, etc.).

Usage (from the repo root):
    python scripts/visual_check_dataset_dualedsr_tf.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import preprocessing

OUT_DIR = Path("data/processed/dualedsr_tf")

FILES_TO_EXPORT = [
    OUT_DIR / "training" / "LR" / "LR.npy",
    OUT_DIR / "training" / "HR" / "HR.npy",
    OUT_DIR / "validation" / "LR" / "LR.npy",
    OUT_DIR / "validation" / "HR" / "HR.npy",
]

EXPORT_DIR = OUT_DIR / "visual_check"

for npy_path in FILES_TO_EXPORT:
    if not npy_path.exists():
        print(f"[SKIPPED] {npy_path} not found.")
        continue

    # e.g. training/LR/LR.npy -> visual_check/training_LR.tif
    tag = f"{npy_path.parent.parent.name}_{npy_path.parent.name}"
    output_path = EXPORT_DIR / f"{tag}.tif"

    array = preprocessing.convert_ftype(npy_path, "tiff", output_path=output_path)
    print(f"{npy_path}  ->  {output_path}  (shape={array.shape}, dtype={array.dtype})")

print(f"\nDone. Open the files under {EXPORT_DIR}/ in ImageJ/Fiji or napari to inspect visually.")
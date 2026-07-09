"""
scripts/Wang_2023_process_data.py

Full data preparation pipeline for DualEDSR (TensorFlow) training, using the
chainable functions in src/preprocessing.py, followed by an automatic export
of the final files back to .tif for visual inspection.

Pipeline (per file, LR and HR independently):

    raw .tiff (uint16), loaded as [Z, Y, X] by tifffile
        --> transpose: [Z, Y, X] -> [Y, X, Z] (Z last, matching the article's
            [Nx, Ny, Nz] convention -- see Milestone 2)
        --> convert_ftype: .tiff -> .npy
        --> convert_dtype: uint16 -> uint8
        --> cubic_interpolation: LR ONLY, voxel-size correction (4.2um -> 2.8um)
        --> center_crop: to the article's target training dimensions
        --> splitting: 80% train / 20% validation (LAST step)

Final output, matching the layout expected by the training script:

    data/processed/dualedsr_tf/training/LR/LR.npy
    data/processed/dualedsr_tf/training/HR/HR.npy
    data/processed/dualedsr_tf/validation/LR/LR.npy
    data/processed/dualedsr_tf/validation/HR/HR.npy

...plus a visual-inspection export (always run, right after the pipeline):

    data/processed/dualedsr_tf/visual_check/training_LR.tif
    data/processed/dualedsr_tf/visual_check/training_HR.tif
    data/processed/dualedsr_tf/visual_check/validation_LR.tif
    data/processed/dualedsr_tf/visual_check/validation_HR.tif

Usage (from the repo root):
    python scripts/Wang_2023_process_data.py
"""

import sys
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import preprocessing

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RAW_DIR = Path("data/raw/Wang_2023")
OUT_DIR = Path("data/processed/Wang_2023")

LR_RAW = RAW_DIR / "ffov_crop_origsize.tiff"
HR_RAW = RAW_DIR / "PEFC_hres_0p7um.tiff"

VOXEL_SIZE_LR_ORIGINAL_UM = 4.2
VOXEL_SIZE_LR_TARGET_UM = 2.8
LR_SCALE_FACTOR = VOXEL_SIZE_LR_ORIGINAL_UM / VOXEL_SIZE_LR_TARGET_UM  # 1.5x

LR_TARGET_SHAPE = (150, 150, 225)
HR_TARGET_SHAPE = (600, 600, 900)

TRAIN_FRACTION = 0.8

VISUAL_CHECK_DIR = OUT_DIR / "visual_check"


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def prepare_data():
    """Runs the full LR/HR preparation pipeline, saving the final
    train/validation .npy files in the layout the training script expects.
    Returns the dict of final {path: array} pairs, for reuse by export_for_visual_check."""
    assert LR_RAW.exists(), f"LR raw file not found: {LR_RAW} (run scripts/Wang_2023_load_raw_data.sh first)"
    assert HR_RAW.exists(), f"HR raw file not found: {HR_RAW} (run scripts/Wang_2023_load_raw_data.sh first)"

    # -----------------------------------------------------------------
    # LR: full chain, including the voxel-size correction (cubic_interpolation)
    # -----------------------------------------------------------------
    section("Processing LR (ffov_crop_origsize.tiff)")
    lr_npy = preprocessing.convert_ftype(LR_RAW, "npy", output_path=OUT_DIR / "_steps" / "LR_01_npy.npy")
    print(f"  1. .tiff -> .npy:        {lr_npy.shape}, {lr_npy.dtype}")

    # tifffile loads TIFF stacks as [Z, Y, X] (Z/slices always first).
    # Our pipeline assumes Z LAST (matching the article's [Nx, Ny, Nz]
    # convention), so we transpose right after loading, before any
    # crop/upsample step -- otherwise crops target the wrong axis entirely.
    lr_npy = preprocessing.TrackedArray(lr_npy.transpose(1, 2, 0), source_path=lr_npy.source_path)
    print(f"     (transposed [Z,Y,X] -> [Y,X,Z]): {lr_npy.shape}")

    lr_uint8 = preprocessing.convert_dtype(lr_npy, np.uint8, output_path=OUT_DIR / "_steps" / "LR_02_uint8.npy")
    print(f"  2. uint16 -> uint8:      {lr_uint8.shape}, {lr_uint8.dtype}")

    lr_upsampled = preprocessing.cubic_interpolation(
        lr_uint8, scale_factor=LR_SCALE_FACTOR,
        output_path=OUT_DIR / "_steps" / "LR_03_upsampled.npy",
    )
    print(f"  3. upsample ({LR_SCALE_FACTOR:.2f}x): {lr_upsampled.shape}")

    lr_cropped = preprocessing.center_crop(
        lr_upsampled, size=LR_TARGET_SHAPE,
        output_path=OUT_DIR / "_steps" / "LR_04_cropped.npy",
    )
    print(f"  4. center_crop:          {lr_cropped.shape}  (target: {LR_TARGET_SHAPE})")

    # -----------------------------------------------------------------
    # HR: same chain, but WITHOUT the interpolation step (already at
    # the correct resolution, per Milestone 2)
    # -----------------------------------------------------------------
    section("Processing HR (PEFC_hres_0p7um.tiff)")
    hr_npy = preprocessing.convert_ftype(HR_RAW, "npy", output_path=OUT_DIR / "_steps" / "HR_01_npy.npy")
    print(f"  1. .tiff -> .npy:        {hr_npy.shape}, {hr_npy.dtype}")

    hr_npy = preprocessing.TrackedArray(hr_npy.transpose(1, 2, 0), source_path=hr_npy.source_path)
    print(f"     (transposed [Z,Y,X] -> [Y,X,Z]): {hr_npy.shape}")

    hr_uint8 = preprocessing.convert_dtype(hr_npy, np.uint8, output_path=OUT_DIR / "_steps" / "HR_02_uint8.npy")
    print(f"  2. uint16 -> uint8:      {hr_uint8.shape}, {hr_uint8.dtype}")

    hr_cropped = preprocessing.center_crop(
        hr_uint8, size=HR_TARGET_SHAPE,
        output_path=OUT_DIR / "_steps" / "HR_03_cropped.npy",
    )
    print(f"  3. center_crop:          {hr_cropped.shape}  (target: {HR_TARGET_SHAPE})")

    # -----------------------------------------------------------------
    # Splitting -- LAST step, for both LR and HR
    # -----------------------------------------------------------------
    section(f"Splitting: {int(TRAIN_FRACTION*100)}% train / "
            f"{round((1-TRAIN_FRACTION)*100)}% validation")
    lr_split = preprocessing.splitting(lr_cropped, size=TRAIN_FRACTION,
                                        output_dir=OUT_DIR / "_steps", base_name="LR_05_split")
    hr_split = preprocessing.splitting(hr_cropped, size=TRAIN_FRACTION,
                                        output_dir=OUT_DIR / "_steps", base_name="HR_05_split")
    print(f"  LR train: {lr_split['train'].shape}  |  LR validation: {lr_split['validation'].shape}")
    print(f"  HR train: {hr_split['train'].shape}  |  HR validation: {hr_split['validation'].shape}")

    # -----------------------------------------------------------------
    # Final save -- exact filenames/layout expected by the training script
    # -----------------------------------------------------------------
    section("Saving final files (layout expected by the training script)")
    final_paths = {
        OUT_DIR / "training" / "LR" / "LR.npy": lr_split["train"],
        OUT_DIR / "training" / "HR" / "HR.npy": hr_split["train"],
        OUT_DIR / "validation" / "LR" / "LR.npy": lr_split["validation"],
        OUT_DIR / "validation" / "HR" / "HR.npy": hr_split["validation"],
    }
    for path, array in final_paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, array)
        print(f"  {path}  (shape={array.shape}, dtype={array.dtype})")

    print(f"\nData preparation done. Final files ready under {OUT_DIR}/training/ and {OUT_DIR}/validation/")
    print(f"(Intermediate step files kept under {OUT_DIR}/_steps/ for inspection/traceability.)")

    return final_paths


def export_for_visual_check(final_paths):
    """Converts the final prepared .npy files back to .tif, for visual
    inspection in an image viewer (ImageJ/Fiji, napari, etc.)."""
    section("Exporting final files to .tif for visual inspection")

    for npy_path in final_paths:
        if not npy_path.exists():
            print(f"[SKIPPED] {npy_path} not found.")
            continue

        # e.g. training/LR/LR.npy -> visual_check/training_LR.tif
        tag = f"{npy_path.parent.parent.name}_{npy_path.parent.name}"
        output_path = VISUAL_CHECK_DIR / f"{tag}.tif"

        # Our pipeline stores volumes as [Y, X, Z] (Z last). tifffile treats
        # the FIRST axis as the page/slice axis when writing, so we transpose
        # back to [Z, Y, X] here -- otherwise Preview/Fiji would scroll
        # through the wrong axis (Y instead of the physically meaningful
        # Z/depth axis).
        volume = np.load(npy_path)
        volume_zyx = volume.transpose(2, 0, 1)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(output_path, volume_zyx)
        print(f"  {npy_path}  ->  {output_path}  (saved as [Z,Y,X]={volume_zyx.shape})")

    print(f"\nOpen the files under {VISUAL_CHECK_DIR}/ in ImageJ/Fiji or napari to inspect visually.")


def main():
    final_paths = prepare_data()
    export_for_visual_check(final_paths)


if __name__ == "__main__":
    main()
"""
scripts/Wang_2023_process_data.py

Full data preparation pipeline for DualEDSR testing.

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

TEST_RAW = RAW_DIR / "LRTest.tif"

# LRTest.tif contains two volumes with DIFFERENT XY resolutions, stored as
# two separate TIFF series in the same file (not a single Z-stack):
#   series 0: (2128, 1150, 345)  -> Volume A
#   series 1: (2000,  275, 1000) -> Volume B
# So we read each series by index instead of slicing a single array by Z.
TEST_A_SERIES_INDEX = 0
TEST_B_SERIES_INDEX = 1

TEST_TARGET_SHAPE = (100, 100, 100)

VISUAL_CHECK_DIR = OUT_DIR / "visual_check"


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def split_test_volume():
    """Reads the two TIFF series stored in LRTest.tif (Volume A and Volume B
    have different XY resolutions, so they are separate series rather than
    a single Z-stack to slice) and saves each as its own intermediate .tif
    so the existing preprocessing.convert_ftype step can consume it like
    before."""
    section("Splitting combined raw file into Test A / Test B")

    with tifffile.TiffFile(TEST_RAW) as tf:
        n_series = len(tf.series)
        assert n_series > max(TEST_A_SERIES_INDEX, TEST_B_SERIES_INDEX), (
            f"Expected at least {max(TEST_A_SERIES_INDEX, TEST_B_SERIES_INDEX) + 1} "
            f"series in {TEST_RAW}, found {n_series}"
        )
        volume_a = tf.series[TEST_A_SERIES_INDEX].asarray()
        volume_b = tf.series[TEST_B_SERIES_INDEX].asarray()

    steps_dir = OUT_DIR / "_steps"
    steps_dir.mkdir(parents=True, exist_ok=True)

    raw_a_path = steps_dir / "Test_A_00_raw.tif"
    raw_b_path = steps_dir / "Test_B_00_raw.tif"

    tifffile.imwrite(raw_a_path, volume_a)
    tifffile.imwrite(raw_b_path, volume_b)

    print(f"  Volume A (series {TEST_A_SERIES_INDEX}): {volume_a.shape}  ->  {raw_a_path}")
    print(f"  Volume B (series {TEST_B_SERIES_INDEX}): {volume_b.shape}  ->  {raw_b_path}")

    return raw_a_path, raw_b_path


def process_block(raw_path, tag):
    """Runs the same conversion -> dtype -> crop pipeline used for a single
    test block, parameterized by an input path and a tag used for naming
    intermediate files (e.g. 'Test_A', 'Test_B')."""
    section(f"Processing Test Block ({tag}) from {raw_path.name}")

    npy = preprocessing.convert_ftype(raw_path, "npy", output_path=OUT_DIR / "_steps" / f"{tag}_01_npy.npy")
    print(f"  1. .tiff -> .npy:        {npy.shape}, {npy.dtype}")

    uint8 = preprocessing.convert_dtype(npy, np.uint8, output_path=OUT_DIR / "_steps" / f"{tag}_02_uint8.npy")
    print(f"  2. uint16 -> uint8:      {uint8.shape}, {uint8.dtype}")

    cropped = preprocessing.center_crop(
        uint8, size=TEST_TARGET_SHAPE,
        output_path=OUT_DIR / "_steps" / f"{tag}_03_cropped.npy",
    )
    print(f"  3. center_crop:          {cropped.shape}  (target: {TEST_TARGET_SHAPE})")

    return cropped


def prepare_data():
    assert TEST_RAW.exists(), f"Test raw file not found: {TEST_RAW} (run scripts/Wang_2023_load_raw_data.sh first)"

    # -----------------------------------------------------------------
    # Split combined raw volume into Test Block A and Test Block B
    # -----------------------------------------------------------------
    raw_a_path, raw_b_path = split_test_volume()

    # -----------------------------------------------------------------
    # Process each block through the same pipeline
    # -----------------------------------------------------------------
    testA_cropped = process_block(raw_a_path, "Test_A")
    testB_cropped = process_block(raw_b_path, "Test_B")

    # -----------------------------------------------------------------
    # Final save
    # -----------------------------------------------------------------
    section("Saving final files")
    final_paths = {
        OUT_DIR / "test" / "Test_A.npy": testA_cropped,
        OUT_DIR / "test" / "Test_B.npy": testB_cropped,
    }
    for path, array in final_paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, array)
        print(f"  {path}  (shape={array.shape}, dtype={array.dtype})")

    return final_paths


def export_for_visual_check(final_paths):
    """Converts the final prepared .npy files back to .tif, for visual
    inspection"""
    section("Exporting final files to .tif for visual inspection")

    for npy_path in final_paths:
        if not npy_path.exists():
            print(f"[SKIPPED] {npy_path} not found.")
            continue

        # NOTE: includes npy_path.stem (Test_A / Test_B) so the two files
        # don't collide -- they share the same parent dirs (.../test/).
        tag = f"{npy_path.parent.parent.name}_{npy_path.parent.name}_{npy_path.stem}"
        output_path = VISUAL_CHECK_DIR / f"{tag}.tif"

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
import numpy as np
from pathlib import Path
from scipy.ndimage import zoom


def bicubic_downsample(volume: np.ndarray, factor: float = 0.25) -> np.ndarray:
    return zoom(volume, zoom=factor, order=3)


def create_pairs(hr_path: str, output_dir: str):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hr = np.load(hr_path)
    lr = bicubic_downsample(hr)

    n_pairs = lr.shape[0]

    for i in range(n_pairs):
        pair = {"lr": lr[i], "hr": hr[i]}
        np.save(output_dir / f"pair_{i:04d}.npy", pair)


if __name__ == "__main__":
    create_pairs(
        hr_path="data/raw/HR.npy",
        output_dir="data/processed/pairs"
    )

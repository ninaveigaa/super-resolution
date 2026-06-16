import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch


class SRDataset(Dataset):
    def __init__(self, pairs_dir: str):
        self.pairs_dir = Path(pairs_dir)
        self.pairs = sorted(self.pairs_dir.glob("pair_*.npy"))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = np.load(self.pairs[idx], allow_pickle=True).item()
        lr = torch.from_numpy(pair["lr"]).float().unsqueeze(0) / 255.0
        hr = torch.from_numpy(pair["hr"]).float().unsqueeze(0) / 255.0
        return lr, hr


def get_dataloader(pairs_dir: str, batch_size: int = 8, shuffle: bool = True):
    dataset = SRDataset(pairs_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

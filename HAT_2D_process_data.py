import random
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from basicsr.utils.registry import DATASET_REGISTRY
except ImportError:
    class DATASET_REGISTRY:
        @staticmethod
        def register():
            return lambda cls: cls

# orientation -> (axis to slice along, the two in-plane axes)
ORIENTATIONS = {
    'xy': (2, (0, 1)),
    'xz': (1, (0, 2)),
    'yz': (0, (1, 2)),
}


def _slice(vol, axis, i):
    idx = [slice(None)] * 3
    idx[axis] = i
    return np.array(vol[tuple(idx)])


def _pad(patch, size, mode='reflect'):
    h, w = patch.shape
    return np.pad(patch, ((0, size - h), (0, size - w)), mode=mode)


def _augment(lr, hr, rng):
    if rng.random() < 0.5:
        lr, hr = lr[:, ::-1], hr[:, ::-1]
    if rng.random() < 0.5:
        lr, hr = lr[::-1, :], hr[::-1, :]
    if rng.random() < 0.5:
        lr, hr = lr.T, hr.T
    return np.ascontiguousarray(lr), np.ascontiguousarray(hr)


def _to_tensor(patch, channels):
    t = torch.from_numpy(patch.astype(np.float32) / 255.0).unsqueeze(0)
    return t.repeat(channels, 1, 1) if channels == 3 else t


@DATASET_REGISTRY.register()
class NpyVolumeSRDataset(Dataset):
    """Random 2D patch dataset (XY/YZ/XZ) for HAT SR training, sampled
    directly from 3D .npy volumes (single-channel uint8).

    Options (opt):
        dataroot_lq, dataroot_gt: paths to the LR/HR .npy volumes
        scale: LR->HR scale factor (same for all 3 axes)
        gt_size: fixed HR patch size (lr patch size = gt_size / scale)
        min_lr_size: smallest random crop size before padding (default lr_size//4)
        length: virtual dataset length (train) or number of fixed
            validation samples (val)
        channels: 1 or 3 (replicate channel for pretrained-weight reuse)
        augment: apply flips/rotation (default: True for train, False for val,
            based on opt['phase'] set automatically by BasicSR)
        seed: base seed used for the deterministic (val) sampling
    """

    def __init__(self, opt):
        self.opt = opt
        self.scale = opt['scale']
        self.gt_size = opt['gt_size']
        self.lr_size = self.gt_size // self.scale
        self.min_lr_size = opt.get('min_lr_size', max(8, self.lr_size // 4))
        self.channels = opt.get('channels', 1)
        self.is_train = opt.get('phase', 'train') == 'train'
        self.augment = opt.get('augment', self.is_train)
        self.length = opt.get('length', 10000 if self.is_train else 100)
        self.seed = opt.get('seed', 0)

        self.lr_vol = np.load(opt['dataroot_lq'], mmap_mode='r')
        self.gt_vol = np.load(opt['dataroot_gt'], mmap_mode='r')
        for a in range(3):
            assert self.gt_vol.shape[a] == self.lr_vol.shape[a] * self.scale, \
                f'axis {a}: HR/LR shape mismatch for scale={self.scale}'

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # deterministic RNG for validation (reproducible), random for train
        rng = random.Random(self.seed + index) if not self.is_train else random

        axis, (a0, a1) = ORIENTATIONS[rng.choice(list(ORIENTATIONS))]
        i_lr = rng.randint(0, self.lr_vol.shape[axis] - 1)
        i_hr = min(i_lr * self.scale + self.scale // 2, self.gt_vol.shape[axis] - 1)

        lr_plane = _slice(self.lr_vol, axis, i_lr)
        gt_plane = _slice(self.gt_vol, axis, i_hr)

        max_crop = min(self.lr_size, *lr_plane.shape)
        crop = rng.randint(min(self.min_lr_size, max_crop), max_crop)
        top = rng.randint(0, lr_plane.shape[0] - crop)
        left = rng.randint(0, lr_plane.shape[1] - crop)

        lr_patch = lr_plane[top:top + crop, left:left + crop]
        gt_patch = gt_plane[top * self.scale:(top + crop) * self.scale,
                             left * self.scale:(left + crop) * self.scale]

        lr_patch = _pad(lr_patch, self.lr_size)
        gt_patch = _pad(gt_patch, self.gt_size)

        if self.augment:
            lr_patch, gt_patch = _augment(lr_patch, gt_patch, rng)

        return {
            'lq': _to_tensor(lr_patch, self.channels),
            'gt': _to_tensor(gt_patch, self.channels),
            'lq_path': f"{self.opt['dataroot_lq']}#{axis}_{i_lr}",
            'gt_path': f"{self.opt['dataroot_gt']}#{axis}_{i_hr}",
        }


if __name__ == '__main__':
    np.save('/tmp/lr.npy', (np.random.rand(150, 150, 180) * 255).astype(np.uint8))
    np.save('/tmp/gt.npy', (np.random.rand(600, 600, 720) * 255).astype(np.uint8))
    ds = NpyVolumeSRDataset(dict(
        dataroot_lq='/tmp/lr.npy', dataroot_gt='/tmp/gt.npy',
        scale=4, gt_size=64, length=8, phase='train',
    ))
    s = ds[0]
    print(s['lq'].shape, s['gt'].shape)

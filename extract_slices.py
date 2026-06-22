"""
Extract 2D slices from 3D LR/HR numpy volumes for HAT fine-tuning.

Volume shapes:
    LR: (225, 150, 150)  ->  (Z, X, Y)  uint8
    HR: (900, 600, 600)  ->  (Z, X, Y)  uint8  (4x scale)

Orientations extracted:
    XY: iterate over Z axis (225 slices)
    YZ: iterate over X axis (150 slices)

Split: 80% train / 20% val per axis
Output: RGB PNG pairs saved to HAT/datasets/PEMFC/
"""

import numpy as np
from PIL import Image
from pathlib import Path

LR_PATH = 'data/raw/LR.npy'
HR_PATH = 'data/raw/HR.npy'
OUT_DIR = Path('HAT/datasets/PEMFC')

def to_rgb(array_2d):
    rgb = np.stack([array_2d, array_2d, array_2d], axis=-1)
    return Image.fromarray(rgb, mode='RGB')

def save_pair(lr_slice, hr_slice, split, name):
    to_rgb(lr_slice).save(OUT_DIR / split / 'LR' / f'{name}.png')
    to_rgb(hr_slice).save(OUT_DIR / split / 'HR' / f'{name}.png')

for split in ('train', 'val'):
    for modality in ('LR', 'HR'):
        (OUT_DIR / split / modality).mkdir(parents=True, exist_ok=True)

print('Loading volumes...')
LR = np.load(LR_PATH)
HR = np.load(HR_PATH)
print(f'  LR: {LR.shape} {LR.dtype}  min={LR.min()} max={LR.max()}')
print(f'  HR: {HR.shape} {HR.dtype}  min={HR.min()} max={HR.max()}')

scale = 4
assert HR.shape == tuple(s * scale for s in LR.shape), \
    f"HR {HR.shape} must be exactly {scale}x LR {LR.shape}"

nz      = LR.shape[0]
train_z = int(nz * 0.8)
print(f'\nExtracting XY slices ({nz} total, {train_z} train / {nz-train_z} val)...')
for z in range(nz):
    split    = 'train' if z < train_z else 'val'
    lr_slice = LR[z, :, :]
    hr_slice = HR[z * scale, :, :]
    save_pair(lr_slice, hr_slice, split, f'xy_{z:04d}')
    if (z + 1) % 50 == 0 or z == nz - 1:
        print(f'  XY: {z+1}/{nz}')

nx      = LR.shape[1]
train_x = int(nx * 0.8)
print(f'\nExtracting YZ slices ({nx} total, {train_x} train / {nx-train_x} val)...')
for x in range(nx):
    split    = 'train' if x < train_x else 'val'
    lr_slice = LR[:, x, :]
    hr_slice = HR[:, x * scale, :]
    save_pair(lr_slice, hr_slice, split, f'yz_{x:04d}')
    if (x + 1) % 30 == 0 or x == nx - 1:
        print(f'  YZ: {x+1}/{nx}')

train_count = len(list((OUT_DIR / 'train' / 'LR').glob('*.png')))
val_count   = len(list((OUT_DIR / 'val'   / 'LR').glob('*.png')))
print(f'\nDone!')
print(f'  train: {train_count} slices  (XY: {train_z}, YZ: {train_x})')
print(f'  val:   {val_count} slices   (XY: {nz-train_z}, YZ: {nx-train_x})')
print(f'  saved to: {OUT_DIR.resolve()}')

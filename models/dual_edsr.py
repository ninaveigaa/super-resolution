"""
DualSRNet architecture in PyTorch.

Equivalent to the network-definition part of the original TF/Keras script:
- InstanceNormalization (2D)      -> nn.InstanceNorm2d(affine=True)
- res_block_EDSR                  -> ResBlockEDSR
- upsampleEDSR / upsampleEDSR1D   -> UpsampleEDSR (with a configurable (h, w) factor)
- edsr / edsr1D                   -> EDSR (`symmetric=False` reproduces the 1D version)

Tensor convention: NCHW (PyTorch default), while the original TF script uses
NHWC. This mostly matters when building batches (see dataset.py) and in the
"cube trick" (batch<->height permute) done in the training loop.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv2d_same(in_ch, out_ch, kernel_size=3, stride=1):
    """Conv2D with 'same' padding (equivalent to Keras' padding='same' for stride=1)."""
    return nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=kernel_size // 2)


class ResBlockEDSR(nn.Module):
    """Equivalent to res_block_EDSR (conv -> relu -> [norm] -> conv -> add skip)."""

    def __init__(self, filters, kernel_size=3, apply_norm=False):
        super().__init__()
        self.conv1 = conv2d_same(filters, filters, kernel_size)
        self.relu = nn.ReLU(inplace=True)
        self.norm = nn.InstanceNorm2d(filters, affine=True, eps=1e-5) if apply_norm else nn.Identity()
        self.conv2 = conv2d_same(filters, filters, kernel_size)

    def forward(self, x):
        skip = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.norm(x)
        x = self.conv2(x)
        return skip + x


class UpsampleEDSR(nn.Module):
    """
    Equivalent to upsampleEDSR / upsampleEDSR1D.

    factor_hw: per-step upsampling factor, e.g. (2, 2) for the regular EDSR,
    or (2, 1) for EDSR1D (upsamples only the "z"/height dimension, keeping the
    width unchanged). Uses nearest-neighbor upsampling, same as Keras' UpSampling2D.
    """

    def __init__(self, scale, num_filters, factor_hw=(2, 2), apply_norm=False):
        super().__init__()
        assert scale in (2, 3, 4, 8), "scale must be 2, 3, 4 or 8"
        if scale in (2, 3):
            steps = [scale]
        elif scale == 4:
            steps = [2, 2]
        elif scale == 8:
            steps = [2, 2, 2]

        # factor_hw tells us, per dimension, whether it should scale with `step`
        # (value > 1, e.g. the "xy" axis) or stay fixed at 1 (the axis EDSR1D
        # doesn't touch).
        h_scales = factor_hw[0] > 1
        w_scales = factor_hw[1] > 1

        blocks = []
        for step in steps:
            step_hw = (step if h_scales else 1, step if w_scales else 1)
            blocks.append(nn.Sequential(
                conv2d_same(num_filters, num_filters, 3),
                nn.ReLU(inplace=True),
                nn.InstanceNorm2d(num_filters, affine=True, eps=1e-5) if apply_norm else nn.Identity(),
                nn.Upsample(scale_factor=step_hw, mode='nearest'),
            ))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class EDSR(nn.Module):
    """
    Equivalent to edsr(...) / edsr1D(...).

    symmetric=True  -> equivalent to edsr()   (upsamples (h, w), the "XY" generatorSR)
    symmetric=False -> equivalent to edsr1D() (upsamples only h, the "YZ" generatorSRC)

    Input/output: 1 channel, NCHW layout, values in [-1, 1].
    """

    def __init__(self, scale, num_filters=64, num_res_blocks=8, symmetric=True, apply_norm=False):
        super().__init__()
        self.head = conv2d_same(1, num_filters, 3)
        self.res_blocks = nn.ModuleList([
            ResBlockEDSR(num_filters, 3, apply_norm=apply_norm) for _ in range(num_res_blocks)
        ])
        self.body_tail = conv2d_same(num_filters, num_filters, 3)

        factor_hw = (2, 2) if symmetric else (2, 1)
        self.upsample = UpsampleEDSR(scale, num_filters, factor_hw=factor_hw, apply_norm=apply_norm)

        self.tail = conv2d_same(num_filters, 1, 3)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.head(x)
        b = x
        for block in self.res_blocks:
            b = block(b)
        b = self.body_tail(b)
        x = x + b
        x = self.upsample(x)
        x = self.tail(x)
        x = self.tanh(x)
        return x


def create_sr_generator(args, device):
    """Equivalent to createSRGenerator(args) (no GAN: returns just the model)."""
    model = EDSR(scale=args.scale, num_filters=args.ngsrf,
                 num_res_blocks=args.numResBlocks, symmetric=True).to(device)
    return model


def create_src_generator(args, device):
    """Equivalent to createSRCGenerator(args) (no GAN: returns just the model)."""
    model = EDSR(scale=args.scale, num_filters=args.ngsrf // 2,
                 num_res_blocks=args.numResBlocks // 2, symmetric=False).to(device)
    return model



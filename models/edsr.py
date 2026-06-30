import torch
import torch.nn as nn
import torch.nn.functional as F

n_feats = 64
n_resblocks = 16

class ResBlock(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return x + self.body(x)
    

class EDSR(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_feats=n_feats, n_resblocks=n_resblocks):
        super().__init__()
        self.head = nn.Conv2d(in_channels, n_feats, kernel_size=3, padding=1)

        body = [ResBlock(n_feats) for _ in range(n_resblocks)]
        body.append(nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1))
        self.body = nn.Sequential(*body)

        self.tail = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(n_feats, n_feats * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(n_feats, out_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        x = x + res
        x = self.tail(x)
        return x
    

class EDSR1D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_feats1d=n_feats//2, n_resblocks1d=n_resblocks//2):
        super().__init__()
        self.head = nn.Conv2d(in_channels, n_feats1d, kernel_size=3, padding=1)

        body = [ResBlock(n_feats1d) for _ in range(n_resblocks1d)]
        body.append(nn.Conv2d(n_feats1d, n_feats1d, kernel_size=3, padding=1))
        self.body = nn.Sequential(*body)

        self.tail = nn.Sequential(
            nn.Conv2d(n_feats1d, n_feats1d, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=(2, 1), mode='nearest'),
            nn.Conv2d(n_feats1d, n_feats1d, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=(2, 1), mode='nearest'),
            nn.Conv2d(n_feats1d, out_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        x = x + res
        x = self.tail(x)
        return x
    
class DualEDSR(nn.Module):
    def __init__(self, n_feats=64, n_resblocks=16, scale=4):
        super().__init__()
        self.scale = scale

        self.sr_xy = EDSR(n_feats=n_feats, n_resblocks=n_resblocks)
        self.sr_z  = EDSR1D(n_feats1d=n_feats//2, n_resblocks1d=n_resblocks//2)

    @staticmethod
    def _quantize(x):
        return torch.round((x + 1.0) * 127.5) / 127.5 - 1.0

    def forward(self, x_lr, quantize=True):
        # x_lr: [Nx, Ny, Nz]
        Nx, Ny, Nz = x_lr.shape

        # SRxy: treat Z slices as a batch of 2D XY images
        x_xy  = x_lr.permute(2, 0, 1).unsqueeze(1)   # [Nz, 1, Nx, Ny]
        sr_xy = self.sr_xy(x_xy)                       # [Nz, 1, Nx*S, Ny*S]

        # Quantize to uint8 range between the two branches (as in original TF code)
        sr_xy_q = self._quantize(sr_xy) if quantize else sr_xy

        # SRz: permute so that Z becomes the H dimension (to be upsampled)
        sr_xy_p = (
            sr_xy_q.squeeze(1)      # [Nz, Nx*S, Ny*S]
            .permute(1, 0, 2)       # [Nx*S, Nz, Ny*S]
            .unsqueeze(1)           # [Nx*S, 1, Nz, Ny*S]
        )
        sr_xyz_raw = self.sr_z(sr_xy_p)   # [Nx*S, 1, Nz*S, Ny*S]

        # Final SR volume
        sr_xyz = (
            sr_xyz_raw.squeeze(1)   # [Nx*S, Nz*S, Ny*S]
            .permute(0, 2, 1)       # [Nx*S, Ny*S, Nz*S]
        )

        return sr_xy, sr_xyz

    def compute_losses(self, sr_xy, sr_xyz, i_hr):
        Nz  = sr_xy.shape[0]
        NxS = sr_xy.shape[2]
        NyS = sr_xy.shape[3]
        NzS = i_hr.shape[2]

        # Downsample HR in Z: Nz*S -> Nz (intermediate target for Lxy)
        tmp = (
            i_hr.permute(2, 0, 1)          # [NzS, NxS, NyS]
            .reshape(NzS, NxS * NyS)
            .permute(1, 0).unsqueeze(0)    # [1, NxS*NyS, NzS]
        )
        tmp = F.interpolate(tmp, size=Nz, mode='linear', align_corners=False)
        i_hr_d = (
            tmp.squeeze(0).permute(1, 0)   # [Nz, NxS*NyS]
            .reshape(Nz, NxS, NyS)
            .unsqueeze(1)                  # [Nz, 1, NxS, NyS]
        )

        l_xy  = F.l1_loss(sr_xy,  i_hr_d)
        l_xyz = F.l1_loss(sr_xyz, i_hr)
        return l_xy + l_xyz, l_xy, l_xyz

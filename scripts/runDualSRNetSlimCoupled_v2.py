import torch
import torch.nn as nn
import torch.nn.functional as F

from edsr import EDSR, EDSR1D


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


if __name__ == '__main__':
    import torch
    torch.manual_seed(0)

    model = DualEDSR(n_feats=8, n_resblocks=2, scale=4)
    print(f'Total parameters : {sum(p.numel() for p in model.parameters()):,}')
    print(f'  sr_xy : {sum(p.numel() for p in model.sr_xy.parameters()):,}')
    print(f'  sr_z  : {sum(p.numel() for p in model.sr_z.parameters()):,}\n')

    lr = torch.randn(10, 10, 15)
    hr = torch.randn(40, 40, 60)

    with torch.no_grad():
        sr_xy, sr_xyz = model(lr)
        loss, lxy, lxyz = model.compute_losses(sr_xy, sr_xyz, hr)

    print(f'sr_xy  : {tuple(sr_xy.shape)}   <- expected [15, 1, 40, 40]')
    print(f'sr_xyz : {tuple(sr_xyz.shape)}     <- expected [40, 40, 60]')
    print(f'loss={loss.item():.4f}  Lxy={lxy.item():.4f}  Lxyz={lxyz.item():.4f}')

    assert sr_xy.shape  == (15, 1, 40, 40)
    assert sr_xyz.shape == (40, 40, 60)
    print('\nOK')
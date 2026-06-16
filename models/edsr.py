import torch.nn as nn

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
    def __init__(self, in_channels=1, out_channels=1, n_feats=64, n_resblocks=16):
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
            nn.Conv2d(n_feats, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        x = x + res
        x = self.tail(x)
        return x
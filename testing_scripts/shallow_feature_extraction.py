import torch
import torch.nn as nn

class Shallow_Feature_Extractor(nn.Module):
    def __init__(self, in_channels=3, out_channels=180):
        super(Shallow_Feature_Extractor, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                                  nn.Conv2d(out_channels,64, kernel_size=3, stride=1, padding=1),
                                  nn.PixelShuffle(upscale_factor=2),
                                  nn.Conv2d(16,64, kernel_size=3, stride=1, padding=1),
                                  nn.PixelShuffle(upscale_factor=2),
                                  nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        x = self.body(x)
        return x
    
model = Shallow_Feature_Extractor()
x = torch.randn(1, 3, 32, 32)
y = model(x)
print(y.shape)
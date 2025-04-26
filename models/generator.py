import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attention import SelfAttention, CBAM

class UNetResidualBlock(nn.Module):
    """Residual block with optional CBAM and Self-Attention."""
    def __init__(self, channels, use_cbam=True, use_self_att=False):
        super(UNetResidualBlock, self).__init__()
        self.use_cbam = use_cbam
        self.use_sa = use_self_att
        layers = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0),
            nn.InstanceNorm2d(channels),
            nn.ReLU(True)
        ]
        if use_self_att:
            layers.append(SelfAttention(channels))
        layers += [
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0),
            nn.InstanceNorm2d(channels)
        ]
        self.block = nn.Sequential(*layers)
        if use_cbam:
            self.cbam = CBAM(channels)

    def forward(self, x):
        out = x + self.block(x)
        if self.use_cbam:
            out = self.cbam(out)
        return out

class UNetGenerator(nn.Module):
    """U-Net generator with skip connections, spectral norm and adaptive attention."""
    def __init__(self, input_nc=1, output_nc=3, ngf=64, n_blocks=8):
        super(UNetGenerator, self).__init__()
        # Initial conv with spectral norm
        self.initial = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.utils.spectral_norm(nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0)),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True)
        )
        # Encoder
        self.enc1 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(ngf, ngf*2, kernel_size=3, stride=2, padding=1)),
            nn.InstanceNorm2d(ngf*2), nn.ReLU(True)
        )
        self.enc2 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(ngf*2, ngf*4, kernel_size=3, stride=2, padding=1)),
            nn.InstanceNorm2d(ngf*4), nn.ReLU(True)
        )
        # Residual blocks
        self.resblocks = nn.ModuleList()
        for i in range(n_blocks):
            use_sa = (i >= n_blocks - 2)
            self.resblocks.append(UNetResidualBlock(ngf*4, use_cbam=True, use_self_att=use_sa))
        # Decoder (upsampling)
        self.dec2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.utils.spectral_norm(nn.Conv2d(ngf*4, ngf*2, kernel_size=3, padding=1)),
            nn.InstanceNorm2d(ngf*2), nn.ReLU(True)
        )
        self.dec1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.utils.spectral_norm(nn.Conv2d(ngf*2, ngf, kernel_size=3, padding=1)),
            nn.InstanceNorm2d(ngf), nn.ReLU(True)
        )
        # Final conv
        self.final = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.utils.spectral_norm(nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)),
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder
        x1 = self.initial(x)      # (ngf)
        x2 = self.enc1(x1)        # (ngf*2)
        x3 = self.enc2(x2)        # (ngf*4)
        # Residual blocks
        x4 = x3
        for block in self.resblocks:
            x4 = block(x4)
        # Decoder + skip connections
        y2 = self.dec2(x4)        # (ngf*2)
        y2 = y2 + x2              # skip add
        y1 = self.dec1(y2)        # (ngf)
        y1 = y1 + x1              # skip add
        # Final
        out = self.final(y1)
        return out

# Проверка
if __name__ == "__main__":
    batch, c, h, w = 1, 1, 600, 600
    inp = torch.randn(batch, c, h, w)
    net = UNetGenerator(input_nc=1, output_nc=3, ngf=64, n_blocks=8)
    out = net(inp)
    print(f"Input: {inp.shape} -> Output: {out.shape}")

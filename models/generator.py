# models/generator.py

import torch
import torch.nn as nn
from models.attention import SelfAttention, CBAM

class ResnetBlockWithCBAM(nn.Module):
    """Residual Block с CBAM Attention"""
    def __init__(self, dim):
        super(ResnetBlockWithCBAM, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            nn.InstanceNorm2d(dim),
        )
        self.cbam = CBAM(dim)

    def forward(self, x):
        out = x + self.conv_block(x)
        out = self.cbam(out)
        return out

class ResnetBlockWithSelfAttention(nn.Module):
    """Residual Block с Self-Attention"""
    def __init__(self, dim):
        super(ResnetBlockWithSelfAttention, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            SelfAttention(dim),

            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            nn.InstanceNorm2d(dim),
        )

    def forward(self, x):
        return x + self.conv_block(x)

class GlobalGenerator(nn.Module):
    """Генератор с чередованием CBAM и Self-Attention в Residual-блоках"""
    def __init__(self, input_nc=1, output_nc=3, ngf=64, n_blocks=9):
        super(GlobalGenerator, self).__init__()
        
        model = []

        # Первый слой
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True)
        ]

        # Downsampling (2 раза)
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(ngf * mult * 2),
                nn.ReLU(True)
            ]

        # Residual-блоки: чередуем CBAM и Self-Attention
        mult = 2**n_downsampling
        for i in range(n_blocks):
            if i % 2 == 0:
                model += [ResnetBlockWithCBAM(ngf * mult)]
            else:
                model += [ResnetBlockWithSelfAttention(ngf * mult)]

        # Upsampling (2 раза)
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(ngf * mult, ngf * mult // 2, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(ngf * mult // 2),
                nn.ReLU(True)
            ]

        # Финальный слой
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

# Проверка работоспособности
if __name__ == "__main__":
    batch_size = 1
    input_nc = 1
    output_nc = 3
    image_size = 600

    x = torch.randn((batch_size, input_nc, image_size, image_size))

    model = GlobalGenerator(input_nc=input_nc, output_nc=output_nc, ngf=64, n_blocks=9)
    preds = model(x)

    print(f'Input shape:  {x.shape}')
    print(f'Output shape: {preds.shape}')
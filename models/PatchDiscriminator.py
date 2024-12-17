import torch
import torch.nn as nn

class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels):
        super(PatchDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),  # in_channels = 1 (SAR) + 3 (Optical)
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),  # Карта патчей
            # nn.Sigmoid()  # Вероятность для патчей
        )

    def forward(self, sar, optical):
        x = torch.cat([sar, optical], dim=1)  # Конкатенация по каналам
        return self.model(x)

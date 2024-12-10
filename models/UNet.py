import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, num_groups=8):  # Добавляем параметр для количества групп в GroupNorm
        super(UNet, self).__init__()

        def down_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(num_groups, out_channels),  # Добавляем GroupNorm
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(num_groups, out_channels),  # Добавляем GroupNorm
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2)
            )

        def up_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(num_groups, out_channels),  # Добавляем GroupNorm
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(num_groups, out_channels),  # Добавляем GroupNorm
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
            )

        # Энкодер
        self.down1 = down_block(3, 64)
        self.down2 = down_block(64, 128)
        self.down3 = down_block(128, 256)
        self.down4 = down_block(256, 512)

        # Средний блок
        self.middle = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups, 1024),  # Добавляем GroupNorm
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups, 1024),  # Добавляем GroupNorm
            nn.ReLU(inplace=True),
        )

        # Декодер
        self.up4 = up_block(1024 + 512, 512)
        self.up3 = up_block(512 + 256, 256)
        self.up2 = up_block(256 + 128, 128)
        self.up1 = up_block(128 + 64, 64)

        # Финальный слой
        self.final = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):
        # Энкодер
        d1 = self.down1(x)  # [N, 64, H/2, W/2]
        d2 = self.down2(d1)  # [N, 128, H/4, W/4]
        d3 = self.down3(d2)  # [N, 256, H/8, W/8]
        d4 = self.down4(d3)  # [N, 512, H/16, W/16]

        # Средний блок
        m = self.middle(d4)  # [N, 1024, H/16, W/16]

        # Декодер
        u4 = self.up4(torch.cat([m, d4], dim=1))  # [N, 512, H/8, W/8]
        u3 = self.up3(torch.cat([u4, d3], dim=1))  # [N, 256, H/4, W/4]
        u2 = self.up2(torch.cat([u3, d2], dim=1))  # [N, 128, H/2, W/2]
        u1 = self.up1(torch.cat([u2, d1], dim=1))  # [N, 64, H, W]

        # Финальный слой
        return self.final(u1)  # [N, 3, H, W]

import torch
import torch.nn as nn
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import datetime
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import math
import torch.nn.functional as F

import math
import torch
from torch.optim.lr_scheduler import _LRScheduler

class Discriminator(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels, 64, 4, 2, 1, padding_mode="reflect")),
            nn.LeakyReLU(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        self.conv3 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        self.conv4 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(256, 512, 4, 1, 1)),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2)
        )
        self.final = nn.Conv2d(512, 1, 4, 1, 1)

    def forward(self, x, y, return_features=False):
        x = torch.cat([x, y], dim=1)

        # Возвращаем признаки из всех слоев
        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        f4 = self.conv4(f3)
        out = self.final(f4)
        if not return_features:
            return out
        else:
            return out, [f1, f2, f3, f4]  # Возвращаем выход и признаки

if __name__ == "__main__":
    x = torch.randn((1, 1, 256, 256))
    y = torch.randn((1, 3, 256, 256))
    model = Discriminator()
    preds = model(x, y)
    print(f'Discriminator:\n\t{x.shape} + {y.shape} ===> {preds.shape}')



class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv   = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)

    def forward(self, x):
        batch, C, width, height = x.size()
        proj_query = self.query_conv(x).view(batch, -1, width * height).permute(0, 2, 1)
        proj_key   = self.key_conv(x).view(batch, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(batch, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1)).view(batch, C, width, height)
        out = self.gamma * out + x
        return out

class Generator(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        # Encoder (downsampling)
        self.down1 = self._block(in_channels, 64, norm=False)       # Выход: 64 каналов, размер: 128x128
        self.down2 = self._block(64, 128)                             # Выход: 128 каналов, размер: 64x64
        self.down3 = self._block(128, 256)                            # Выход: 256 каналов, размер: 32x32
        # Добавляем self attention для слоя с 256 каналами
        self.sa = SelfAttention(256)
        self.down4 = self._block(256, 512, dropout=0.5)               # Выход: 512 каналов, размер: 16x16
        self.down5 = self._block(512, 512, dropout=0.5)               # Выход: 512 каналов, размер: 8x8

        # Дополнительный слой для апсемплинга d5 до разрешения d4 (16x16)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        # Decoder (upsampling) с учетом skip connections:
        # up1 принимает конкатенацию: d5 (после апсемплинга, 512 каналов) + d4 (512 каналов) = 1024 каналов
        self.up1 = self._block(1024, 512, dropout=0.5, up=True)
        # up2: объединение u1 (512) и d3 (256) → 768 каналов
        self.up2 = self._block(768, 256, dropout=0.5, up=True)
        # up3: объединение u2 (256) и d2 (128) → 384 каналов
        self.up3 = self._block(384, 128, up=True)
        # up4: объединение u3 (128) и d1 (64) → 192 канала
        self.up4 = self._block(192, 64, up=True)
        
        self.final = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    
    def _block(self, in_ch, out_ch, norm=True, dropout=0.0, up=False):
        layers = []
        if up:
            layers.append(nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False))
        else:
            layers.append(nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False, padding_mode="reflect"))
        if norm:
            layers.append(nn.InstanceNorm2d(out_ch))
        if dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Encoder
        d1 = self.down1(x)    # (B, 64, 128, 128)
        d2 = self.down2(d1)   # (B, 128, 64, 64)
        d3 = self.down3(d2)   # (B, 256, 32, 32)
        # Применяем self attention к d3
        d3 = self.sa(d3)
        d4 = self.down4(d3)   # (B, 512, 16, 16)
        d5 = self.down5(d4)   # (B, 512, 8, 8)
        
        # Апсемплинг d5 до разрешения 16x16 для корректного объединения с d4
        d5_up = self.upsample(d5)  # (B, 512, 16, 16)
        
        # Decoder с использованием skip connections
        u1 = self.up1(torch.cat([d5_up, d4], dim=1))     # Вход: 1024 каналов → (B, 512, 32, 32)
        u2 = self.up2(torch.cat([u1, d3], dim=1))        # Вход: 512+256 = 768 каналов → (B, 256, 64, 64)
        u3 = self.up3(torch.cat([u2, d2], dim=1))        # Вход: 256+128 = 384 каналов → (B, 128, 128, 128)
        u4 = self.up4(torch.cat([u3, d1], dim=1))        # Вход: 128+64 = 192 каналов → (B, 64, 256, 256)
        
        return self.final(u4)  # (B, 3, 256, 256)

if __name__ == "__main__":
    x = torch.randn((1, 1, 256, 256))
    model = Generator()
    preds = model(x)
    print(f'Generator:\n\t{x.shape} ===> {preds.shape}')


class Pix2PixGAN(nn.Module):
    def __init__(self, device):
        super(Pix2PixGAN, self).__init__()
        self.device = device
        self.generator = Generator().to(self.device)
        self.discriminator = Discriminator().to(self.device)
        self.scaler = torch.amp.GradScaler(enabled=(device.type=='cuda'))

        self.optimizer_G = torch.optim.AdamW(
            self.generator.parameters(),
            lr=2e-4,
            betas=(0.5, 0.999),
            eps=1e-6
        )
        self.optimizer_D = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=2e-4,
            betas=(0.5, 0.999),
            eps=1e-6
        )

        self.criterion_GAN = nn.MSELoss()
        self.criterion_L1 = nn.L1Loss()

        self.psnr = PeakSignalNoiseRatio().to(self.device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=2.0).to(self.device)

        self.l1_lambda = 100    # Коэффициент для L1 loss
        self.lambda_gp = 0.1     # Коэффициент для gradient penalty
        self.lambda_fm = 1     # Коэффициент для feature matching

        self.scheduler_G = CosineAnnealingWarmRestarts(
            self.optimizer_G,
            T_0=50,  # Каждые 50 эпох перезапуск
            T_mult=2, 
            eta_min=1e-6
        )

        self.scheduler_D = CosineAnnealingWarmRestarts(
            self.optimizer_D,
            T_0=50,  # Каждые 50 эпох перезапуск
            T_mult=2, 
            eta_min=1e-6
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')

    def compute_gradient_penalty(self, real_A, real_B, fake_B):
        alpha = torch.rand(real_B.size(0), 1, 1, 1, device=self.device)
        interpolates = (alpha * real_B + (1 - alpha) * fake_B).requires_grad_(True)
        
        d_interpolates = self.discriminator(real_A, interpolates)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True
        )[0]
        
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def train_step(self, real_A, real_B):
        real_A, real_B = real_A.to(self.device), real_B.to(self.device)

        # ---------------------------
        # Обучение дискриминатора
        # ---------------------------
        self.optimizer_D.zero_grad()
        with torch.set_grad_enabled(True):
            fake_B_for_D = self.generator(real_A).detach()
            output_real = self.discriminator(real_A, real_B)
            output_fake = self.discriminator(real_A, fake_B_for_D)
            target_real = torch.ones_like(output_real)
            target_fake = torch.zeros_like(output_fake)

            loss_D_real = self.criterion_GAN(output_real, target_real)
            loss_D_fake = self.criterion_GAN(output_fake, target_fake)
            loss_D_gp = self.compute_gradient_penalty(real_A, real_B, fake_B_for_D)
            loss_D = (loss_D_real + loss_D_fake) * 0.5 + self.lambda_gp * loss_D_gp
            
        loss_D.backward()
        self.optimizer_D.step()

        # ---------------------------
        # Обучение генератора
        # ---------------------------
        self.optimizer_G.zero_grad()
        with torch.set_grad_enabled(True):
            fake_B_for_G = self.generator(real_A)
            pred_fake, fake_features = self.discriminator(real_A, fake_B_for_G, return_features=True)
            _, real_features = self.discriminator(real_A, real_B, return_features=True)

            loss_G_GAN = self.criterion_GAN(pred_fake, torch.ones_like(pred_fake))
            loss_G_L1 = self.criterion_L1(fake_B_for_G, real_B) * self.l1_lambda
            loss_fm = 0
            for real_feat, fake_feat in zip(real_features[:-1], fake_features[:-1]):
                loss_fm += self.criterion_L1(fake_feat, real_feat.detach())
            loss_G_FM = loss_fm * self.lambda_fm
            loss_G = loss_G_GAN + loss_G_L1 + loss_G_FM

            # Вычисляем PSNR и SSIM
            psnr_value = self.psnr(fake_B_for_G, real_B)
            ssim_value = self.ssim(fake_B_for_G, real_B)
        
            loss_G.backward()
            self.optimizer_G.step()

        return loss_G.item(), loss_D.item(), ssim_value.item(), psnr_value.item()

    def val_step(self, real_A, real_B):
        real_A, real_B = real_A.to(self.device), real_B.to(self.device)
        fake_B = self.generator(real_A)
        loss_G_L1 = self.criterion_L1(fake_B, real_B)
        output_real = self.discriminator(real_A, real_B)
        output_fake = self.discriminator(real_A, fake_B)
        target_real = torch.ones_like(output_real)
        target_fake = torch.zeros_like(output_fake)
        loss_D_real = self.criterion_GAN(output_real, target_real)
        loss_D_fake = self.criterion_GAN(output_fake, target_fake)
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # Вычисляем PSNR и SSIM
        psnr_value = self.psnr(fake_B, real_B)
        ssim_value = self.ssim(fake_B, real_B)

        return loss_G_L1.item(), loss_D.item(), ssim_value.item(), psnr_value.item()


    def step_schedulers(self, loss_D, loss_G):
        self.scheduler_D.step()
        self.scheduler_G.step()

    def save_state(self, epoch, save_dir=os.path.join(os.getcwd(), 'checkpoints')):
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'scheduler_G_state_dict': self.scheduler_G.state_dict(),
            'scheduler_D_state_dict': self.scheduler_D.state_dict(),
            'date': datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        }
        os.makedirs(save_dir, exist_ok=True)
        checkpoint_file = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save(checkpoint, checkpoint_file)

    def load_state(self, checkpoint_name, device):
        save_dir = os.path.join(os.getcwd(), f'checkpoints/{checkpoint_name}.pth')
        if not os.path.isfile(save_dir):
            print(f"Чекпоинт не найден по пути: {save_dir}")
            print("Начинаем обучение с нуля.")
            return 0

        checkpoint = torch.load(save_dir, map_location=device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        self.scheduler_G.load_state_dict(checkpoint['scheduler_G_state_dict'])
        self.scheduler_D.load_state_dict(checkpoint['scheduler_D_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        date_saved = checkpoint.get('date', "Неизвестно")
        print(f"Чекпоинт успешно загружен: {save_dir}")
        print(f"Дата сохранения: {date_saved}, эпоха {start_epoch+1}")
        return start_epoch

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gan = Pix2PixGAN(device)
    x = torch.randn((4, 1, 256, 256))
    y = torch.randn((4, 3, 256, 256))

    train_loss_D, train_loss_G, train_ssim, train_psnr = gan.train_step(x, y)
    val_loss_D, val_loss_G, val_ssim, val_psnr = gan.val_step(x, y)
    
    print('Pix2PixGAN:' +
        f'\n\ttrain_loss_D = {train_loss_D:.2f}, train_loss_G = {train_loss_G:.2f}, train_ssim = {train_ssim:.2f}, train_psnr = {train_psnr:.2f}' +
        f'\n\tval_loss_D = {val_loss_D:.2f}, val_loss_G = {val_loss_G:.2f}, val_ssim = {val_ssim:.2f}, val_psnr = {val_psnr:.2f}')
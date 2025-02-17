import torch
import torch.nn as nn
import os
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import datetime
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from lpips import LPIPS

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
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(in_channels),
        )
    
    def forward(self, x):
        return x + self.block(x)
    
def compute_gradient_penalty(discriminator, real_A, real_B, fake_B, device):
    alpha = torch.rand(real_B.size(0), 1, 1, 1, device=device)
    interpolates = (alpha * real_B + (1 - alpha) * fake_B).requires_grad_(True)
    d_interpolates = discriminator(real_A, interpolates)
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

class Discriminator(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels, 64, 4, 2, 1, padding_mode="reflect")),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.attn = SelfAttention(128)
        self.final = nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=1)

    def forward(self, x, y, return_features=False):
        x = torch.cat([x, y], dim=1)
        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        f3 = self.attn(f2)
        out = self.final(f3)
        if not return_features:
            return out
        return out, [f1, f2, f3]

if __name__ == "__main__":
    x = torch.randn((1, 1, 256, 256))
    y = torch.randn((1, 3, 256, 256))
    model = Discriminator()
    preds = model(x, y)
    print(f'Discriminator:\n\t{x.shape} + {y.shape} ===> {preds.shape}')

class Generator(nn.Module):
    def __init__(self, in_channels=1, num_residuals=6):
        super(Generator, self).__init__()
        self.initial = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels, 64, kernel_size=7, stride=1, padding=3)),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(num_residuals)])
        self.final = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.initial(x)
        x = self.res_blocks(x)
        return self.final(x)

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

        self.criterion_GAN = nn.MSELoss()
        self.criterion_L1 = nn.L1Loss()
        self.criterion_perceptual = LPIPS(pretrained=True).to(self.device)

        self.optimizer_G = torch.optim.AdamW(
            self.generator.parameters(),
            lr=1e-4,
            betas=(0.5, 0.999)
        )
        self.optimizer_D = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=4e-4,
            betas=(0.5, 0.999)
        )

        self.psnr = PeakSignalNoiseRatio().to(self.device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=2.0).to(self.device)

        self.l1_lambda = 100    # Коэффициент для L1 loss
        self.lambda_gp = 10     # Коэффициент для gradient penalty
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

    def train_step(self, real_A, real_B):
        real_A, real_B = real_A.to(self.device), real_B.to(self.device)

        # ---------------------------
        # Обучение дискриминатора
        # ---------------------------
        self.optimizer_D.zero_grad()
        fake_B_for_D = self.generator(real_A).detach()
        output_real = self.discriminator(real_A, real_B)
        output_fake = self.discriminator(real_A, fake_B_for_D)
        target_real = torch.ones_like(output_real) * 0.9
        target_fake = torch.ones_like(output_fake) * 0.1

        loss_D_real = self.criterion_GAN(output_real, target_real)
        loss_D_fake = self.criterion_GAN(output_fake, target_fake)
        loss_D_gp = compute_gradient_penalty(self.discriminator, real_A, real_B, fake_B_for_D, self.device)
        loss_D = (loss_D_real + loss_D_fake) * 0.5 + self.lambda_gp * loss_D_gp

        loss_D.backward()
        self.optimizer_D.step()

        # ---------------------------
        # Обучение генератора
        # ---------------------------
        self.optimizer_G.zero_grad()
        fake_B_for_G = self.generator(real_A)
        pred_fake, fake_features = self.discriminator(real_A, fake_B_for_G, return_features=True)
        _, real_features = self.discriminator(real_A, real_B, return_features=True)

        loss_G_GAN = self.criterion_GAN(pred_fake, torch.ones_like(pred_fake))
        loss_G_L1 = self.criterion_L1(fake_B_for_G, real_B) * self.l1_lambda
        loss_fm = 0
        for real_feat, fake_feat in zip(real_features[:-1], fake_features[:-1]):
            loss_fm += self.criterion_L1(fake_feat, real_feat.detach())
        loss_G_FM = loss_fm * self.lambda_fm
        loss_G_perceptual = self.criterion_perceptual(fake_B_for_G, real_B).mean()
        loss_G = loss_G_GAN + loss_G_L1 + loss_G_FM + loss_G_perceptual

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
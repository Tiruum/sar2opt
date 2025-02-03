import torch
import torch.nn as nn
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
import datetime
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import math
import torch.nn.functional as F

import math
import torch
from torch.optim.lr_scheduler import _LRScheduler

class WarmupCosineAnnealingWarmRestarts(_LRScheduler):
    """
    Расширенный планировщик (scheduler) для оптимизатора PyTorch,
    сочетающий warm-up и Cosine Annealing Warm Restarts.
    
    Аргументы:
        optimizer (Optimizer): Optimizer, к которому привязан scheduler.
        warmup_epochs (int): Количество эпох для "прогрева" (warm-up).
        T_0 (int): Количество эпох в первом цикле косинусного затухания.
        T_mult (int, float): Во сколько раз увеличивается период T_i после каждого рестарта.
        eta_min (float): Минимальный learning rate во время косинусного затухания.
        warmup_start_lr (float): Начальное значение LR на старте (0 по умолчанию).
        last_epoch (int): Номер последней обученной эпохи (по умолчанию -1).
    """
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        T_0: int,
        T_mult: float = 1,
        eta_min: float = 0.0,
        warmup_start_lr: float = 2e-4,
        last_epoch: int = -1
    ):
        # Параметры warm-up
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        
        # Параметры Cosine Annealing Warm Restarts
        self.T_0 = T_0       # длина первого цикла
        self.T_mult = T_mult # во сколько раз увеличиваем T_i после рестартов
        self.eta_min = eta_min
        
        # Текущее значение T_i и счётчик "какой цикл"
        self.T_i = T_0       # длина текущего цикла
        self.cycle = 0       # номер цикла
        self.cycle_epoch = 0 # сколько эпох прошло с начала текущего цикла
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """
        Возвращает список LR для каждой param_group оптимизатора на текущей эпохе (self.last_epoch).
        """
        # 1) Фаза warm-up (линейный рост от warmup_start_lr до base_lr)
        if self.last_epoch < self.warmup_epochs:
            # Пропорция прогрева от 0 до 1
            warmup_progress = float(self.last_epoch) / float(self.warmup_epochs)
            
            # Линейная интерполяция: LR = warmup_start_lr + (base_lr - warmup_start_lr)*warmup_progress
            return [
                self.warmup_start_lr + (base_lr - self.warmup_start_lr) * warmup_progress
                for base_lr in self.base_lrs
            ]
        
        # 2) Фаза Cosine Annealing с рестартами
        else:
            # Сколько эпох прошло с конца warm-up
            epochs_since_warmup = self.last_epoch - self.warmup_epochs
            
            # Определяем, не пора ли рестартнуть новый цикл
            # (когда epochs_since_warmup >= T_i, значит заканчивается текущий цикл)
            if epochs_since_warmup // self.T_i > self.cycle:
                # Заходим в новый цикл
                self.cycle += 1
                # Обновляем T_i (умножаем на T_mult)
                self.T_i = int(self.T_i * self.T_mult)
            
            # Теперь выясняем, сколько эпох прошло в рамках "текущего" цикла
            # Начало цикла = сумма всех предыдущих T_i; проще всего смотреть разницу
            cycle_start_epoch = sum(self.T_0 * (self.T_mult**i) for i in range(self.cycle))
            self.cycle_epoch = epochs_since_warmup - (cycle_start_epoch - self.T_0*(self.T_mult**(self.cycle-1)) if self.cycle > 0 else 0)
            # Вариант попроще: self.cycle_epoch = epochs_since_warmup - (cycle_start_epoch - self.T_i) если аккуратно считать.
            # Ниже упрощённый расчёт:
            # self.cycle_epoch = epochs_since_warmup - (cycle_start_epoch - self.T_i)
            
            # Нормируем на длину цикла, чтобы получить прогресс от 0 до 1
            # (self.cycle_epoch / self.T_i) 
            cosine_progress = float(self.cycle_epoch) / float(self.T_i)
            
            # Косинусная формула:
            # LR = eta_min + (base_lr - eta_min) * (1 + cos(pi * progress)) / 2
            return [
                self.eta_min + (base_lr - self.eta_min) * 
                (1.0 + math.cos(math.pi * cosine_progress)) / 2.0
                for base_lr in self.base_lrs
            ]

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.utils.spectral_norm(  # Добавляем Spectral Normalization
                nn.Conv2d(
                    in_channels, 
                    out_channels, 
                    4, stride, 1, 
                    padding_mode="reflect", 
                    bias=False
                )
            ),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.conv(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect"
            ),
            nn.LeakyReLU(0.2)
        )
        self.layers = nn.Sequential(
            CNNBlock(64, 128, stride=1),
            CNNBlock(128, 256),
            CNNBlock(256, 512),
            nn.Conv2d(512, 1, 4, 1, 1, padding_mode="reflect")
        )

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.initial(x)
        x = self.layers(x)
        return x

if __name__ == "__main__":
    x = torch.randn((1, 1, 256, 256))
    y = torch.randn((1, 3, 256, 256))
    model = Discriminator()
    preds = model(x, y)
    print(f'Discriminator:\n\t{x.shape} + {y.shape} ===> {preds.shape}')


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect", **kwargs)
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if act == "relu" else nn.LeakyReLU(0.2, inplace=True)
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.conv(x)

class Generator(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(inplace=True)
        )

        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1, bias=False, padding_mode="reflect"),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(inplace=True),
            ResBlock(128),
            ResBlock(128)
        )
        self.down2 = Block(128, 256, down=True, act="leaky", use_dropout=False)
        self.down3 = Block(256, 512, down=True, act="leaky", use_dropout=False)
        self.down4 = Block(512, 512, down=True, act="leaky", use_dropout=False)
        self.down5 = Block(512, 512, down=True, act="leaky", use_dropout=False)
        self.down6 = Block(512, 512, down=True, act="leaky", use_dropout=False)
        self.down7 = Block(512, 512, down=True, act="leaky", use_dropout=False)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 4, 2, 1, padding_mode="reflect"),
            nn.ReLU(inplace=True)
        )
        self.up1 = Block(512, 512, down=False, act="relu", use_dropout=True)
        self.up2 = Block(1024, 512, down=False, act="relu", use_dropout=True)
        self.up3 = Block(1024, 512, down=False, act="relu", use_dropout=True)
        self.up4 = Block(1024, 512, down=False, act="relu", use_dropout=False)
        self.up5 = Block(1024, 256, down=False, act="relu", use_dropout=False)
        self.up6 = Block(512, 128, down=False, act="relu", use_dropout=False)
        self.up7 = Block(256, 64, down=False, act="relu", use_dropout=False)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.initial(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        bottleneck = self.bottleneck(d7)
        u1 = self.up1(bottleneck)
        u2 = self.up2(torch.cat([u1, d7], 1))
        u3 = self.up3(torch.cat([u2, d6], 1))
        u4 = self.up4(torch.cat([u3, d5], 1))
        u5 = self.up5(torch.cat([u4, d4], 1))
        u6 = self.up6(torch.cat([u5, d3], 1))
        u7 = self.up7(torch.cat([u6, d2], 1))
        return self.final(torch.cat([u7, d1], 1))

if __name__ == "__main__":
    x = torch.randn((1, 1, 256, 256))
    model = Generator()
    preds = model(x)
    print(f'Generator:\n\t{x.shape} ===> {preds.shape}')


class Pix2PixGAN(nn.Module):
    def __init__(self, device, l1_lambda=15):
        super(Pix2PixGAN, self).__init__()
        self.device = device
        self.generator = Generator().to(self.device)
        self.discriminator = Discriminator().to(self.device)
        self.scaler = torch.amp.GradScaler('cuda')

        self.optimizer_G = torch.optim.AdamW(self.generator.parameters(), lr=2e-3, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.AdamW(self.discriminator.parameters(), lr=2e-3, betas=(0.5, 0.999))

        self.criterion_GAN = nn.MSELoss()
        self.criterion_L1 = nn.L1Loss()

        self.psnr = PeakSignalNoiseRatio().to(self.device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=2.0).to(self.device)

        self.l1_lambda = l1_lambda

        # self.scheduler_G = ReduceLROnPlateau(
        #     self.optimizer_G, 
        #     mode='min', 
        #     factor=0.75, 
        #     patience=100,
        #     min_lr=1e-6
        # )
        # self.scheduler_D = ReduceLROnPlateau(
        #     self.optimizer_D, 
        #     mode='min', 
        #     factor=0.75, 
        #     patience=100,
        #     min_lr=1e-6
        # )

        self.scheduler_G = WarmupCosineAnnealingWarmRestarts(
            self.optimizer_G,
            warmup_epochs=5,
            T_0=10,        # первый Cosine-цикл = 10 эпох
            T_mult=2,      # каждый следующий цикл в 2 раза длиннее предыдущего
            eta_min=1e-5,
            warmup_start_lr=1e-4
        )

        self.scheduler_D = WarmupCosineAnnealingWarmRestarts(
            self.optimizer_D,
            warmup_epochs=5,
            T_0=10,        # первый Cosine-цикл = 10 эпох
            T_mult=2,      # каждый следующий цикл в 2 раза длиннее предыдущего
            eta_min=1e-5,
            warmup_start_lr=1e-4
        )

    def train_step(self, real_A, real_B):
        real_A, real_B = real_A.to(self.device), real_B.to(self.device)
        self.optimizer_D.zero_grad()
        with torch.amp.autocast('cuda'):
            # Обучение дискриминатора
            fake_B = self.generator(real_A)

            # Получаем выходные данные дискриминатора
            output_real = self.discriminator(real_A, real_B)
            output_fake = self.discriminator(real_A, fake_B.detach())

            # Создаем целевые метки
            target_real = torch.ones_like(output_real)
            target_fake = torch.zeros_like(output_fake)

            # Вычисляем потери
            loss_D_real = self.criterion_GAN(output_real, target_real)
            loss_D_fake = self.criterion_GAN(output_fake, target_fake)
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            
        self.scaler.scale(loss_D).backward() # Масштабируем градиенты
        self.scaler.step(self.optimizer_D)

        # Обучение генератора
        self.optimizer_G.zero_grad()

        with torch.amp.autocast('cuda'):
            output_fake_for_G = self.discriminator(real_A, fake_B)
            loss_G_GAN = self.criterion_GAN(output_fake_for_G, torch.ones_like(output_fake_for_G))
            loss_G_L1 = self.criterion_L1(fake_B, real_B) * self.l1_lambda
            loss_G = loss_G_GAN + loss_G_L1

            # Вычисляем PSNR и SSIM
            psnr_value = self.psnr(fake_B, real_B)  # Оцениваем на первом изображении в батче
            ssim_value = self.ssim(fake_B, real_B)
        
        self.scaler.scale(loss_G).backward()
        self.scaler.step(self.optimizer_G)
        self.scaler.update()  # Обновляем масштаб

        return loss_G.item(), loss_D.item(), ssim_value.item(), psnr_value.item()

    def val_step(self, real_A, real_B):
        real_A, real_B = real_A.to(self.device), real_B.to(self.device)
        fake_B = self.generator(real_A)

        # Потери для генератора
        loss_G_L1 = self.criterion_L1(fake_B, real_B)

        # Потери для дискриминатора
        output_real = self.discriminator(real_A, real_B)
        output_fake = self.discriminator(real_A, fake_B)
        
        target_real = torch.ones_like(output_real)
        target_fake = torch.zeros_like(output_fake)

        loss_D_real = self.criterion_GAN(output_real, target_real)
        loss_D_fake = self.criterion_GAN(output_fake, target_fake)
        loss_D = (loss_D_real + loss_D_fake) * 0.5

        # Вычисляем PSNR и SSIM
        psnr_value = self.psnr(fake_B, real_B)  # Оцениваем на первом изображении в батче
        ssim_value = self.ssim(fake_B, real_B)

        return loss_G_L1.item(), loss_D.item(), ssim_value.item(), psnr_value.item()


    def step_schedulers(self, loss_D, loss_G):
        """
        Шаг для ReduceLROnPlateau. Передаем потери дискриминатора и генератора.
        """
        self.scheduler_D.step()
        self.scheduler_G.step()

    # Метод для сохранения состояния модели
    def save_state(self, epoch, save_dir=os.path.join(os.getcwd(), 'checkpoints')):
        """
        Сохраняет состояние модели, включая параметры генератора, дискриминатора, оптимизаторов и шедулеров.

        Аргументы:
            epoch (int): Номер текущей эпохи.
            save_dir (str): Путь для сохранения контрольной точки.

        Возвращает:
            None
        """
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

    # Метод для загрузки состояния модели
    def load_state(self, checkpoint_name, device):
        """
        Загружает состояние модели, включая параметры генератора, дискриминатора, оптимизаторов и шедулеров.

        Аргументы:
            checkpoint_name (str): Имя файла чекпоинта (без расширения).
            device (torch.device): Устройство для загрузки модели (CPU или GPU).

        Возвращает:
            start_epoch (int): Эпоха, с которой можно продолжить обучение.
        """
        save_dir = os.path.join(os.getcwd(), f'checkpoints/{checkpoint_name}.pth')
        if not os.path.isfile(save_dir):
            print(f"Чекпоинт не найден по пути: {save_dir}")
            print("Начинаем обучение с нуля.")
            return 0

        checkpoint = torch.load(save_dir, map_location=device)

        # Восстанавливаем состояния генератора и дискриминатора
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

        # Восстанавливаем состояния оптимизаторов
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])

        # Восстанавливаем состояния шедулеров
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
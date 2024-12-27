import torch
import torch.nn as nn
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
import datetime

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, 1, bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(out_channels),
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
            # CNNBlock(512, 1024),
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
    print(f'Discriminator: {x.shape} + {y.shape} ===> {preds.shape}')


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
        return self.conv(x)
        return self.dropout(self.conv(x)) if self.use_dropout else self.conv(x)

class Generator(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(inplace=True)
        )

        self.down1 = Block(64, 128, down=True, act="leaky", use_dropout=False)
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
    print(f'Generator: {x.shape} ===> {preds.shape}')


class Pix2PixGAN(nn.Module):
    def __init__(self, device, l1_lambda=15):
        super(Pix2PixGAN, self).__init__()
        self.device = device
        self.generator = Generator().to(self.device)
        self.discriminator = Discriminator().to(self.device)

        self.optimizer_G = torch.optim.AdamW(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.AdamW(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

        self.criterion_GAN = nn.MSELoss()
        self.criterion_L1 = nn.L1Loss()

        self.l1_lambda = l1_lambda

        self.scheduler_G = ReduceLROnPlateau(
            self.optimizer_G, 
            mode='min', 
            factor=0.75, 
            patience=25,
            min_lr=1e-6
        )
        self.scheduler_D = ReduceLROnPlateau(
            self.optimizer_D, 
            mode='min', 
            factor=0.75, 
            patience=25,
            min_lr=1e-6
        )

    def train_step(self, real_A, real_B):
        real_A, real_B = real_A.to(self.device), real_B.to(self.device)

        # Train Discriminator
        self.optimizer_D.zero_grad()

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
        loss_D.backward()
        self.optimizer_D.step()

        # Train Generator
        self.optimizer_G.zero_grad()

        output_fake_for_G = self.discriminator(real_A, fake_B)
        loss_G_GAN = self.criterion_GAN(output_fake_for_G, torch.ones_like(output_fake_for_G))
        loss_G_L1 = self.criterion_L1(fake_B, real_B) * self.l1_lambda
        loss_G = loss_G_GAN + loss_G_L1
        loss_G.backward()
        self.optimizer_G.step()

        return loss_D.item(), loss_G.item()

    def step_schedulers(self, loss_D, loss_G):
        """
        Шаг для ReduceLROnPlateau. Передаем потери дискриминатора и генератора.
        """
        self.scheduler_D.step(loss_D)
        self.scheduler_G.step(loss_G)

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
    x = torch.randn((1, 1, 256, 256))
    y = torch.randn((1, 3, 256, 256))
    loss_D, loss_G = gan.train_step(x, y)
    print(f'Pix2PixGAN: {x.shape} + {y.shape} ===> Loss_D={loss_D}, Loss_G={loss_G}')
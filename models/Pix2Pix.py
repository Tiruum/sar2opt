import torch
import torch.nn as nn
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
import datetime





#   _____  _               _           _             _             
#  |  __ \(_)             (_)         (_)           | |            
#  | |  | |_ ___  ___ _ __ _ _ __ ___  _ _ __   __ _| |_ ___  _ __ 
#  | |  | | / __|/ __| '__| | '_ ` _ \| | '_ \ / _` | __/ _ \| '__|
#  | |__| | \__ \ (__| |  | | | | | | | | | | | (_| | || (_) | |   
#  |_____/|_|___/\___|_|  |_|_| |_| |_|_|_| |_|\__,_|\__\___/|_|   

class Discriminator(nn.Module):
    """
    Класс Discriminator представляет собой сверточную нейронную сеть, используемую для различения изображений.
    
    Аргументы:
    in_channels (int): Количество входных каналов. По умолчанию 4 (1 канал для источника и 3 канала для цели).
    
    Методы:
    __init__(self, in_channels=4): Инициализирует слои дискриминатора.
    forward(self, src, target): Выполняет прямое распространение через сеть. Объединяет входные изображения (src и target) по каналу и пропускает их через модель.
    
    Пример использования:
    discriminator = Discriminator()
    output = discriminator(src_image, target_image)
    """
    def __init__(self, in_channels=4):  # 2 channel source + 3 channels target
        super(Discriminator, self).__init__()
        def discriminator_block(in_channels, out_channels, stride):
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1),
                nn.BatchNorm2d(out_channels),  # No BN in first layer
                nn.LeakyReLU(0.2, inplace=True)
            ]
            return layers

        self.model = nn.Sequential(
            # C64: 4x4 kernel, stride 2, padding 1
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # C128: 4x4 kernel, stride 2, padding 1
            *discriminator_block(64, 128, stride=2),

            # C256: 4x4 kernel, stride 2, padding 1
            *discriminator_block(128, 256, stride=2),

            # C512: 4x4 kernel, stride 1, padding 1
            *discriminator_block(256, 512, stride=1),

            # C1: 4x4 kernel, stride 1, padding 1
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),

            # Sigmoid activation
            nn.Sigmoid()
        )


    def forward(self, src, target):
        x = torch.cat((src, target), dim=1)
        return self.model(x)

if __name__ == '__main__':
    D = Discriminator()
    x = torch.randn(1, 1, 256, 256)
    target = torch.randn(1, 3, 256, 256)
    print(f'{x.shape} + {target.shape} -> {D(x, target).shape}')  # torch.Size([1, 1, 30, 30])





#    _____                           _             
#   / ____|                         | |            
#  | |  __  ___ _ __   ___ _ __ __ _| |_ ___  _ __ 
#  | | |_ |/ _ \ '_ \ / _ \ '__/ _` | __/ _ \| '__|
#  | |__| |  __/ | | |  __/ | | (_| | || (_) | |   
#   \_____|\___|_| |_|\___|_|  \__,_|\__\___/|_|   

class UNetGenerator(nn.Module):
    def __init__(self, input_channels=1, output_channels=3):
        super(UNetGenerator, self).__init__()
        
        def encoder_block(in_channels, out_channels, use_bn=True):
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            if use_bn:
                layers.append(nn.BatchNorm2d(out_channels))
            return nn.Sequential(*layers)

        def decoder_block(in_channels, out_channels, dropout=0):
            layers = [
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels)
            ]
            if dropout != 0:
                layers.append(nn.Dropout(dropout))
            return nn.Sequential(*layers)

        def bottleneck_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.ReLU(inplace=True),
            )

        # Encoder
        self.enc1 = encoder_block(input_channels, 64, use_bn=False)
        self.enc2 = encoder_block(64, 128)
        self.enc3 = encoder_block(128, 256)
        self.enc4 = encoder_block(256, 512)
        self.enc5 = encoder_block(512, 512)
        self.enc6 = encoder_block(512, 512)
        self.enc7 = encoder_block(512, 512)

        self.bottleneck = bottleneck_block(512, 512)

        # Decoder
        self.dec1 = decoder_block(512, 512, dropout=0.5)
        self.dec2 = decoder_block(1024, 512, dropout=0.5)
        self.dec3 = decoder_block(1024, 512, dropout=0.5)
        self.dec4 = decoder_block(1024, 512)
        self.dec5 = decoder_block(1024, 256)
        self.dec6 = decoder_block(512, 128)
        self.dec7 = decoder_block(256, 64)
        self.final = nn.ConvTranspose2d(128, output_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)

        b = self.bottleneck(e7)

        # Decoder + Skip connections
        d1 = self.dec1(b)
        d2 = self.dec2(torch.cat([d1, e7], dim=1))
        d3 = self.dec3(torch.cat([d2, e6], dim=1))
        d4 = self.dec4(torch.cat([d3, e5], dim=1))
        d5 = self.dec5(torch.cat([d4, e4], dim=1))
        d6 = self.dec6(torch.cat([d5, e3], dim=1))
        d7 = self.dec7(torch.cat([d6, e2], dim=1))
        return torch.tanh(self.final(torch.cat([d7, e1], dim=1)))

if __name__ == '__main__':
    G = UNetGenerator()
    x = torch.randn(1, 1, 256, 256)
    print(f'{x.shape} -> {G(x).shape}')  # torch.Size([1, 3, 256, 256])





#   _____ _      ___  _____ _      
#  |  __ (_)    |__ \|  __ (_)     
#  | |__) |__  __  ) | |__) |__  __
#  |  ___/ \ \/ / / /|  ___/ \ \/ /
#  | |   | |>  < / /_| |   | |>  < 
#  |_|   |_/_/\_\____|_|   |_/_/\_\

class Pix2PixGAN(nn.Module):
    def __init__(self, device):
        super(Pix2PixGAN, self).__init__()
        self.device = device
        self.generator = UNetGenerator().to(self.device)
        self.discriminator = Discriminator().to(self.device)

        self.optimizer_G = torch.optim.AdamW(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.AdamW(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

        self.criterion_GAN = nn.MSELoss()
        self.criterion_L1 = nn.L1Loss()

        self.l1_lambda = 50

        self.scheduler_G = ReduceLROnPlateau(
            self.optimizer_G, 
            mode='min', 
            factor=0.5, 
            patience=5,
            min_lr=1e-6
        )
        self.scheduler_D = ReduceLROnPlateau(
            self.optimizer_D, 
            mode='min', 
            factor=0.5, 
            patience=5,
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

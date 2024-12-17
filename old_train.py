import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import os
from torchvision.models import vgg16, VGG16_Weights

def perceptual_loss(generated, real, model):
    gen_features = model(generated)
    real_features = model(real)
    return nn.L1Loss()(gen_features, real_features)

project_root = os.path.abspath(os.path.join(os.getcwd()))

# Импорт модели UNet из файла models/UNet.py
from models.UNet import UNet
from models.PatchDiscriminator import PatchDiscriminator
from utils.helpers import *
from utils.loss_tracker import save_losses
from utils.ImageBuffer import ImageBuffer
from utils.GetTemp import GetGpuTemp

# Функция обучения GAN
def train_gan(train_loader, generator, discriminator, g_optimizer, d_optimizer, criterion_gan, criterion_l1, epochs, device, patience=25):
    '''
    Функция для обучения GAN.
    
    Parameters:
    - train_loader (DataLoader): DataLoader для тренировочных данных.
    - generator (nn.Module): Генератор.
    - discriminator (nn.Module): Дискриминатор.
    - g_optimizer (Optimizer): Оптимизатор для генератора.
    - d_optimizer (Optimizer): Оптимизатор для дискриминатора.
    - criterion_gan (Loss): Функция потерь для GAN.
    - criterion_l1 (Loss): L1 Loss.
    - epochs (int): Количество эпох.
    - device (str): Устройство (cuda/cpu).
    - patience (int): Количество эпох без улучшения для ранней остановки.
    '''

    best_g_loss = float("inf")
    best_d_loss = float("inf")
    g_losses, d_losses = [], []

    # Инициализация планировщика для генератора и дискриминатора
    g_scheduler = ReduceLROnPlateau(g_optimizer, mode='min', factor=0.5, patience=10)
    d_scheduler = ReduceLROnPlateau(d_optimizer, mode='min', factor=0.5, patience=10)

    # Инициализация буфера
    fake_image_buffer = ImageBuffer(buffer_size=100)

    for epoch in range(epochs):
        g_loss_total, d_loss_total = 0, 0

        for batch_idx, (sar, optical) in enumerate(tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{epochs}]", postfix="Training...", leave=False, ncols=100, unit="batch", colour="#22c55e")):
            sar, optical = sar.to(device), optical.to(device)

            # --- Обучение дискриминатора ---
            discriminator.train()
            g_output = generator(sar)

            buffered_images = fake_image_buffer.push_and_pop(g_output.detach())
            # Добавление небольшого шума для избежания чрезмерного доверия дискриминатора к буферу
            buffered_images += torch.rand_like(buffered_images) * 0.05

            # Реальные метки и поддельные метки
            real_labels = torch.full_like(discriminator(optical), 0.9, device=device)
            fake_labels = torch.full_like(discriminator(buffered_images), 0.1, device=device)

            real_labels += torch.randn_like(real_labels) * 0.05
            fake_labels += torch.randn_like(fake_labels) * 0.05

            # Потери дискриминатора для реальных и поддельных данных
            d_real = discriminator(optical)
            d_real_loss = criterion_gan(d_real, real_labels)

            d_fake = discriminator(buffered_images)
            d_fake_loss = criterion_gan(d_fake, fake_labels)

            d_loss = (d_real_loss + d_fake_loss) / 2
            if batch_idx % 2 == 0:
                d_optimizer.zero_grad()
                d_loss.backward()
                d_optimizer.step()

            # --- Обучение генератора ---
            generator.train()

            # Потери генератора: Adversarial Loss + L1 Loss
            d_fake, fake_features = discriminator(g_output, return_features=True)
            _, real_features = discriminator(optical, return_features=True)

            g_adv_loss = criterion_gan(d_fake, real_labels)
            g_l1_loss = criterion_l1(g_output, optical)
            g_perceptual_loss = perceptual_loss(g_output, optical, perceptual_model)

            # Feature Matching Loss (L1 на промежуточные признаки)
            fm_loss = sum([criterion_l1(fake_feat, real_feat.detach()) for fake_feat, real_feat in zip(fake_features, real_features)]) / len(fake_features)

            g_loss = 5 * g_adv_loss + 2 * g_l1_loss + 5 * g_perceptual_loss + 0.1 * fm_loss

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            g_loss_total += g_loss.item()
            d_loss_total += d_loss.item()

        # Замедляем обучение дискриминатора
        if (epoch + 1) % 20 == 0:
            for param_group in d_optimizer.param_groups:
                param_group['lr'] *= 0.5

        # Логирование потерь
        g_losses.append(g_loss_total / len(train_loader))
        d_losses.append(d_loss_total / len(train_loader))
        save_losses(g_losses, d_losses, f"{project_root}/logs")

        tqdm.write(f"Epoch: [{epoch+1}/{epochs}] Gen Loss: {g_loss_total/len(train_loader):.4f}, Dis Loss: {d_loss_total/len(train_loader):.4f}, GPU Temp: {GetGpuTemp()}C")
        tqdm.write(f"Curr Gen LR: {g_scheduler.get_last_lr()[0]:.6f}, Curr Dis LR: {d_scheduler.get_last_lr()[0]:.6f}, Buffer size: {len(buffered_images)}")

        # Сохранение лучших моделей
        if epoch % 2 == 0:
            torch.save(generator.state_dict(), os.path.join(SAVE_MODEL_PATH, "best_generator_patchgan.pth"))
            torch.save(discriminator.state_dict(), os.path.join(SAVE_MODEL_PATH, "best_discriminator_patchgan.pth"))

        # Обновление планировщика
        g_scheduler.step(g_loss_total / len(train_loader))
        d_scheduler.step(d_loss_total / len(train_loader))

    print("Training completed!")

# Запуск тренировки
if __name__ == "__main__":
    # Настройки
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    EPOCHS = 200  # Количество эпох
    LEARNING_RATE = 1e-5  # Скорость обучения
    SAVE_MODEL_PATH = f"{project_root}/saved_models"  # Папка для сохранения модели
    os.makedirs(SAVE_MODEL_PATH, exist_ok=True)

    perceptual_model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:9].to(DEVICE).eval()

    # Инициализация модели, функции потерь и оптимизатора
    generator = UNet(in_channels=1, output_channels=3).to(DEVICE)
    # generator.load_state_dict(torch.load("./saved_models/best_model_UNet.pth", map_location=DEVICE))
    discriminator = PatchDiscriminator(in_channels=3).to(DEVICE)

    # Инициализация оптимизаторов
    g_optimizer = AdamW(generator.parameters(), lr=1e-4)
    d_optimizer = AdamW(discriminator.parameters(), lr=LEARNING_RATE)

    criterion_gan = nn.MSELoss()
    criterion_l1 = nn.L1Loss()

    train_gan(
        train_loader,
        generator,
        discriminator,
        g_optimizer,
        d_optimizer,
        criterion_gan,
        criterion_l1,
        EPOCHS,
        DEVICE
    )

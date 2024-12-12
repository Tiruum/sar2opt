import torch
from torch import nn
from torch.optim import Adam
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

    for epoch in range(epochs):
        g_loss_total, d_loss_total = 0, 0

        for sar, optical in tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{epochs}]", postfix="Training...", leave=False, ncols=100, unit="batch"):
            sar, optical = sar.to(device), optical.to(device)

            # --- Обучение дискриминатора ---
            discriminator.train()
            g_output = generator(sar)

            # Реальные метки и поддельные метки
            real_labels = torch.ones_like(discriminator(optical), device=device) * 0.8
            fake_labels = torch.zeros_like(discriminator(g_output.detach()), device=device) + 0.2

            # Потери дискриминатора для реальных и поддельных данных
            d_real = discriminator(optical)
            d_real_loss = criterion_gan(d_real, real_labels)

            d_fake = discriminator(g_output.detach())
            d_fake_loss = criterion_gan(d_fake, fake_labels)

            d_loss = (d_real_loss + d_fake_loss) / 2
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # --- Обучение генератора ---
            generator.train()

            # Потери генератора: Adversarial Loss + L1 Loss
            d_fake = discriminator(g_output)
            g_adv_loss = criterion_gan(d_fake, real_labels)
            g_l1_loss = criterion_l1(g_output, optical)
            # g_perceptual_loss = perceptual_loss(g_output, optical, perceptual_model)
            g_loss = g_adv_loss + 25 * g_l1_loss

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            g_loss_total += g_loss.item()
            d_loss_total += d_loss.item()

        # Логирование потерь
        g_losses.append(g_loss_total / len(train_loader))
        d_losses.append(d_loss_total / len(train_loader))
        save_losses(g_losses, d_losses, f"{project_root}/logs")

        tqdm.write(f"Epoch: [{epoch+1}/{epochs}] Generator Loss: {g_loss_total/len(train_loader):.4f}, Discriminator Loss: {d_loss_total/len(train_loader):.4f}")

        # Сохранение лучших моделей
        if epoch % 5 == 0:
            torch.save(generator.state_dict(), os.path.join(SAVE_MODEL_PATH, "best_generator_patchgan.pth"))
            torch.save(discriminator.state_dict(), os.path.join(SAVE_MODEL_PATH, "best_discriminator_patchgan.pth"))

    print("Training completed!")

# Запуск тренировки
if __name__ == "__main__":
    # Настройки
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPOCHS = 200  # Количество эпох
    LEARNING_RATE = 1e-5  # Скорость обучения
    SAVE_MODEL_PATH = f"{project_root}/saved_models"  # Папка для сохранения модели
    os.makedirs(SAVE_MODEL_PATH, exist_ok=True)

    # perceptual_model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:9].to(DEVICE).eval()

    # Инициализация модели, функции потерь и оптимизатора
    generator = UNet(in_channels=1, output_channels=3).to(DEVICE)
    discriminator = PatchDiscriminator(in_channels=3).to(DEVICE)

    # Инициализация оптимизаторов
    g_optimizer = Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    d_optimizer = Adam(discriminator.parameters(), lr=LEARNING_RATE)

    criterion_gan = nn.BCELoss()
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

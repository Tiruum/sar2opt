import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import os

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
from utils.GetTemp import get_gpu_temperature

def train_gan(train_loader, generator, discriminator, g_optimizer, d_optimizer, criterion_gan, criterion_l1, epochs, device, patience=25):
    torch.cuda.empty_cache()
    best_g_loss = float("inf")
    g_losses, d_losses = [], []

    # Инициализация планировщика для генератора и дискриминатора
    g_scheduler = ReduceLROnPlateau(g_optimizer, mode='min', factor=0.5, patience=10)
    d_scheduler = ReduceLROnPlateau(d_optimizer, mode='min', factor=0.5, patience=10)

    for epoch in range(epochs):
        g_loss_total, d_loss_total = 0, 0

        for batch_idx, (sar, optical) in enumerate(tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{epochs}]", leave=False)):
            sar, optical = sar.to(device), optical.to(device)

            # === Обучение дискриминатора ===
            g_output = generator(sar).detach()  # Генерация фейкового изображения
            real_labels = torch.ones_like(discriminator(sar, optical), device=device)
            fake_labels = torch.zeros_like(discriminator(sar, g_output), device=device)

            d_real = discriminator(sar, optical)
            d_real_loss = criterion_gan(d_real, real_labels)

            d_fake = discriminator(sar, g_output)
            d_fake_loss = criterion_gan(d_fake, fake_labels)

            d_loss = (d_real_loss + d_fake_loss) / 2

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # === Обучение генератора ===
            g_output = generator(sar)
            d_fake = discriminator(sar, g_output)

            g_adv_loss = criterion_gan(d_fake, real_labels)
            g_l1_loss = criterion_l1(g_output, optical)

            g_loss = g_adv_loss + 100 * g_l1_loss  # GAN Loss + L1 Loss

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            g_loss_total += g_loss.item()
            d_loss_total += d_loss.item()

        # Логирование потерь
        g_losses.append(g_loss_total / len(train_loader))
        d_losses.append(d_loss_total / len(train_loader))

        tqdm.write(f"Epoch: [{epoch+1}/{epochs}] Gen Loss: {g_loss_total/len(train_loader):.4f}, Dis Loss: {d_loss_total/len(train_loader):.4f}")

        # Сохранение лучших моделей
        if epoch % 1 == 0:
            torch.save(generator.state_dict(), os.path.join(SAVE_MODEL_PATH, "best_generator_pix2pix.pth"))
            torch.save(discriminator.state_dict(), os.path.join(SAVE_MODEL_PATH, "best_discriminator_pix2pix.pth"))
            tqdm.write(f"\tModels saved! Curr Gen LR: {g_scheduler.get_last_lr()[0]:.6f}, Curr Dis LR: {d_scheduler.get_last_lr()[0]:.6f}")
        
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
    LEARNING_RATE = 0.0002  # Скорость обучения
    SAVE_MODEL_PATH = f"{project_root}/saved_models"  # Папка для сохранения модели
    os.makedirs(SAVE_MODEL_PATH, exist_ok=True)

    # Инициализация модели, функции потерь и оптимизатора
    generator = UNet(in_channels=1, output_channels=3).to(DEVICE)
    discriminator = PatchDiscriminator(in_channels=4).to(DEVICE)

    # Инициализация оптимизаторов
    g_optimizer = AdamW(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    d_optimizer = AdamW(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

    criterion_gan = nn.BCEWithLogitsLoss()
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

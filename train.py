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
from utils.EarlyStopping import EarlyStopping

# Функция для тренировки одной эпохи
def train_one_epoch(loader, model, criterion, optimizer, device):
    model.train()
    epoch_loss = 0

    for sar, optical in tqdm(loader, desc="Training", leave=False):
        sar, optical = sar.to(device), optical.to(device)

        # Прямой проход
        output = model(sar)

        # Вычисление функции потерь
        loss = criterion(output, optical)
        epoch_loss += loss.item()

        # Обратный проход
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return epoch_loss / len(loader)

# Функция для валидации
def validate(loader, model, criterion, device):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for sar, optical in tqdm(loader, desc="Validation", leave=False):
            sar, optical = sar.to(device), optical.to(device)

            # Прямой проход
            output = model(sar)

            # Вычисление функции потерь
            loss = criterion(output, optical)
            epoch_loss += loss.item()

    return epoch_loss / len(loader)

def train_model_with_early_stopping(train_loader, test_loader, model, criterion, optimizer, epochs, device, patience=25):
    best_val_loss = float("inf")  # Изначально лучшее значение — бесконечность
    model_save_path = os.path.join(SAVE_MODEL_PATH, "best_model.pth")  # Путь для сохранения лучшей модели
    no_improvement_epochs = 0  # Счётчик эпох без улучшения
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        print(f"Epoch [{epoch + 1}/{epochs}]")

        # Тренировка
        train_loss = train_one_epoch(train_loader, model, criterion, optimizer, device)
        # Валидация
        val_loss = validate(test_loader, model, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        save_losses(train_losses, val_losses, f'{project_root}/logs')

        print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        # Проверяем, улучшилась ли валидационная потеря
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            no_improvement_epochs = 0
            print(f"Validation loss improved. Model saved to {model_save_path}")
        else:
            no_improvement_epochs += 1
            print(f"No improvement for {no_improvement_epochs} epoch(s).")

        # Проверяем условие ранней остановки
        if no_improvement_epochs >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement. Last epoch: {epoch + 1}")
            break

    print(f"Training complete. Best model saved at: {model_save_path}")

def validate_gan(loader, generator, criterion_l1, device, epoch, epochs):
    generator.eval()
    val_loss = 0

    with torch.no_grad():
        for sar, optical in tqdm(loader, desc=f"Epoch [{epoch + 1}/{epochs}]", postfix="Validation...", leave=False, ncols=100, unit="batch"):
            sar, optical = sar.to(device), optical.to(device)
            g_output = generator(sar)
            loss = criterion_l1(g_output, optical)
            val_loss += loss.item()

    return val_loss / len(loader)  # Средняя потеря на валидационной выборке


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

    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(epochs):
        g_loss_total, d_loss_total = 0, 0

        for sar, optical in tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{epochs}]", postfix="Training...", leave=False, ncols=100, unit="batch"):
            sar, optical = sar.to(device), optical.to(device)

            # --- Обучение дискриминатора ---
            discriminator.train()
            g_output = generator(sar)

            # Реальные метки и поддельные метки
            real_labels = torch.ones_like(discriminator(optical), device=device) * 0.9
            fake_labels = torch.zeros_like(discriminator(g_output.detach()), device=device) + 0.1

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
            g_perceptual_loss = perceptual_loss(g_output, optical, perceptual_model)
            g_loss = g_adv_loss + 10 * g_l1_loss + 5 * g_perceptual_loss

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            g_loss_total += g_loss.item()
            d_loss_total += d_loss.item()

        # Логирование потерь
        g_losses.append(g_loss_total / len(train_loader))
        d_losses.append(d_loss_total / len(train_loader))
        save_losses(g_losses, d_losses, f"{project_root}/logs")

        val_loss = validate_gan(test_loader, generator, criterion_l1, DEVICE, epoch, epochs)

        tqdm.write(f"Epoch: [{epoch+1}/{epochs}] Generator Loss: [train: {g_loss_total/len(train_loader):.4f}, val: {val_loss:.4f}], Discriminator Loss: {d_loss_total/len(train_loader):.4f}")

        # Сохранение лучших моделей
        if g_loss_total < best_g_loss:
            torch.save(generator.state_dict(), os.path.join(SAVE_MODEL_PATH, "best_generator_patchgan.pth"))
            best_g_loss = g_loss_total
        if d_loss_total < best_d_loss:
            torch.save(discriminator.state_dict(), os.path.join(SAVE_MODEL_PATH, "best_discriminator_patchgan.pth"))
            best_d_loss = d_loss_total

        # Проверка ранней остановки
        early_stopping(val_loss)  # Передаём валидационную потерю генератора
        if early_stopping.early_stop:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    print("Training completed!")

# Запуск тренировки
if __name__ == "__main__":
    # Настройки
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPOCHS = 200  # Количество эпох
    LEARNING_RATE = 1e-5  # Скорость обучения
    SAVE_MODEL_PATH = f"{project_root}/saved_models"  # Папка для сохранения модели
    os.makedirs(SAVE_MODEL_PATH, exist_ok=True)

    perceptual_model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:9].to(DEVICE).eval()

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

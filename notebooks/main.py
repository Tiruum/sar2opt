import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
import os
import sys
import numpy as np

project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Импорт модели UNet из файла models/UNet.py
from models.UNet import UNet
from utils.helpers import *

# Настройки
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 200  # Количество эпох
LEARNING_RATE = 1e-4  # Скорость обучения
SAVE_MODEL_PATH = f"{project_root}/saved_models"  # Папка для сохранения модели

# Убедимся, что папка для сохранения модели существует
os.makedirs(SAVE_MODEL_PATH, exist_ok=True)

# Инициализация модели, функции потерь и оптимизатора
model = UNet(in_channels=1, output_channels=3).to(DEVICE)
criterion = nn.L1Loss()  # L1 Loss
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
model.load_state_dict(torch.load('../saved_models/best_model.pth', map_location=DEVICE))

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

# Основной цикл обучения
def train_model(train_loader, test_loader, model, criterion, optimizer, epochs, device):
    for epoch in range(epochs):
        print(f"Epoch [{epoch + 1}/{epochs}]")

        train_loss = train_one_epoch(train_loader, model, criterion, optimizer, device)
        val_loss = validate(test_loader, model, criterion, device)

        print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        # Сохраняем модель после каждой эпохи
        model_save_path = os.path.join(SAVE_MODEL_PATH, f"unet_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

def train_model_with_early_stopping(train_loader, test_loader, model, criterion, optimizer, epochs, device, patience=10):
    best_val_loss = float("inf")  # Изначально лучшее значение — бесконечность
    save_path = "../saved_models/best_model.pth"  # Путь для сохранения лучшей модели
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
        np.savez('losses.npz', train_losses=train_losses, val_losses=val_losses)

        print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        # Проверяем, улучшилась ли валидационная потеря
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            no_improvement_epochs = 0
            print(f"Validation loss improved. Model saved to {save_path}")
        else:
            no_improvement_epochs += 1
            print(f"No improvement for {no_improvement_epochs} epoch(s).")

        # Проверяем условие ранней остановки
        if no_improvement_epochs >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement. Last epoch: {epoch + 1}")
            break

    print(f"Training complete. Best model saved at: {save_path}")


# Запуск тренировки
if __name__ == "__main__":
    train_model_with_early_stopping(train_loader, test_loader, model, criterion, optimizer, EPOCHS, DEVICE)

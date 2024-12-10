import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images_A = sorted(os.listdir(os.path.join(root_dir, 'trainA')))
        self.images_B = sorted(os.listdir(os.path.join(root_dir, 'trainB')))

    def __len__(self):
        return len(self.images_A)

    def __getitem__(self, idx):
        imgA_path = os.path.join(self.root_dir, 'trainA', self.images_A[idx])
        imgB_path = os.path.join(self.root_dir, 'trainB', self.images_B[idx])
        imgA = Image.open(imgA_path).convert("RGB")
        imgB = Image.open(imgB_path).convert("RGB")

        if self.transform:
            imgA = self.transform(imgA)
            imgB = self.transform(imgB)

        return imgA, imgB

def visualize_results(input_image, generated_image, target_image):
    # Обратная нормализация
    unnormalize = T.Normalize((-1, -1, -1), (2, 2, 2))
    input_image = unnormalize(input_image)
    generated_image = unnormalize(generated_image)
    target_image = unnormalize(target_image)

    # Преобразование из тензоров в изображения
    input_image = input_image.permute(1, 2, 0).cpu().numpy()
    generated_image = generated_image.permute(1, 2, 0).detach().cpu().numpy()
    target_image = target_image.permute(1, 2, 0).cpu().numpy()

    # Отображение
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(input_image)
    plt.title("Radar Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(generated_image)
    plt.title("Generated Optical Image")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(target_image)
    plt.title("Target Optical Image")
    plt.axis("off")

    plt.show()

def save_best_model(epoch, model, optimizer, loss, filepath):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filepath)
    print(f"Best model saved at epoch {epoch} with loss {loss:.4f}")

def load_checkpoint(filepath, model, optimizer=None):
    checkpoint = torch.load(filepath)  # Загрузка данных чекпоинта
    model.load_state_dict(checkpoint['model_state_dict'])  # Восстановление весов модели
    if optimizer:  # Если требуется восстановить оптимизатор
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']  # Последняя сохранённая эпоха
    loss = checkpoint['loss']  # Последняя сохранённая функция потерь
    print(f"Модель загружена: epoch {epoch}, loss {loss}")
    return epoch

def save_loss_history(train_loss_history, test_loss_history, filepath):
    np.savez(filepath, train_loss=train_loss_history, test_loss=test_loss_history)
    print(f"Loss history saved to {filepath}")

def load_loss_history(filepath):
    try:
        data = np.load(filepath)
        train_loss = data['train_loss'].tolist()  # Преобразуем обратно в список
        test_loss = data['test_loss'].tolist()  # Преобразуем обратно в список
        print(f"Loss history loaded from {filepath}")
        return train_loss, test_loss
    except FileNotFoundError:
        print(f"Loss history ({filepath}) не найдена, начинаем сначала.")
        return [], []
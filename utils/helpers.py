import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.functional import to_tensor, to_pil_image
import torch.nn.functional as F
from PIL import Image, ImageFilter
import cv2
import numpy as np
from scipy.ndimage import median_filter
import random

# Настройки
IMAGE_SIZE = 256  # Размер для ресайза изображений
DATA_DIR = "./dataset"  # Путь к папке с данными
BATCH_SIZE = 8  # Размер батча

def crop_black_borders(image, rotation_angle):
    width, height = image.size

    if rotation_angle == 0:
        # При угле 0° никакой обрезки не делаем
        return image
    
    angle_rad = np.radians(rotation_angle)
    # Безопасный размер без чёрных полей после поворота
    safe_size = int(min(width, height) / (abs(np.cos(angle_rad)) + abs(np.sin(angle_rad))))

    center_x, center_y = width // 2, height // 2
    left = max(center_x - safe_size // 2, 0)
    upper = max(center_y - safe_size // 2, 0)
    right = min(center_x + safe_size // 2, width)
    lower = min(center_y + safe_size // 2, height)

    cropped_image = image.crop((left, upper, right, lower))
    return cropped_image

class InMemoryDataset(Dataset):
    def __init__(self, dataset):
        self.data = [dataset[i] for i in range(len(dataset))]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class Augmentation:
    """
    Класс для совместного применения аугментаций к двум изображениям.
    """
    def __init__(self, rotation_degree=15, horizontal_flip=True, vertical_flip=True, crop_black_borders=True):
        self.rotation_degree = rotation_degree
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.crop_black_borders = crop_black_borders

    def __call__(self, sar, optical):
        # Случайное вращение
        if self.rotation_degree != 0:
            angle = transforms.RandomRotation.get_params([-self.rotation_degree, self.rotation_degree])
            sar = transforms.functional.rotate(sar, angle)
            optical = transforms.functional.rotate(optical, angle)
            # Обрезка черных границ
            if self.crop_black_borders:
                sar = crop_black_borders(sar, angle)
                optical = crop_black_borders(optical, angle)
        
        # Случайное горизонтальное отражение
        if self.horizontal_flip and random.random() < 0.5:
            sar = transforms.functional.hflip(sar)
            optical = transforms.functional.hflip(optical)
        
        # Случайное вертикальное отражение
        if self.vertical_flip and random.random() < 0.5:
            sar = transforms.functional.vflip(sar)
            optical = transforms.functional.vflip(optical)
        
        return sar, optical

# Трансформации
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),                          # Преобразование [0, 255] -> [0, 1]
    transforms.Normalize(mean=(0.5,), std=(0.5,))   # Преобразование [0, 1] -> [-1, 1]
])

# Датасет
class SARToOpticalDataset(Dataset):
    def __init__(self, sar_dir, optical_dir, transform=None, augment=None):
        self.sar_dir = sar_dir
        self.optical_dir = optical_dir
        self.transform = transform
        self.augment = augment

        # Фильтрация только файлов, исключая директории и скрытые файлы
        self.sar_images = sorted(
            [f for f in os.listdir(sar_dir) if os.path.isfile(os.path.join(sar_dir, f)) and not f.startswith(".")]
        )
        self.optical_images = sorted(
            [f for f in os.listdir(optical_dir) if os.path.isfile(os.path.join(optical_dir, f)) and not f.startswith(".")]
        )

    def __len__(self):
        return len(self.sar_images)

    def __getitem__(self, idx):
        # Загружаем SAR-изображение
        sar_path = os.path.join(self.sar_dir, self.sar_images[idx])
        sar_image = Image.open(sar_path).convert("L")  # Ч/б изображение

        # Загружаем оптическое изображение
        optical_path = os.path.join(self.optical_dir, self.optical_images[idx])
        optical_image = Image.open(optical_path).convert("RGB")  # Цветное изображение

        # Применяем совместимые аугментации, если таковые есть
        if self.augment:
            sar_image, optical_image = self.augment(sar_image, optical_image)

        # Применяем трансформации, если таковые есть
        if self.transform:
            sar_image = self.transform(sar_image)
            optical_image = self.transform(optical_image)

        return sar_image, optical_image


# Создаём экземпляры датасетов
train_augment = Augmentation(rotation_degree=15, horizontal_flip=True, vertical_flip=True)

train_dataset = SARToOpticalDataset(
    sar_dir=os.path.join(DATA_DIR, "trainA"),
    optical_dir=os.path.join(DATA_DIR, "trainB"),
    transform=transform,
    augment=train_augment
)

test_dataset = SARToOpticalDataset(
    sar_dir=os.path.join(DATA_DIR, "testA"),
    optical_dir=os.path.join(DATA_DIR, "testB"),
    transform=transform
)

# train_dataset_in_memory = InMemoryDataset(train_dataset)

# DataLoader для батчей
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)

# Проверка работы
if __name__ == "__main__":
    for sar, optical in train_loader:
        print(f"SAR batch shape: {sar.shape}")
        print(f"Optical batch shape: {optical.shape}")
        break
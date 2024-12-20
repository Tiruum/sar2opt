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
BATCH_SIZE = 16  # Размер батча

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

# Функция для применения медианного фильтра и Лапласа
def preprocess_with_laplace(image):
    image = image.filter(ImageFilter.MedianFilter(size=5))      # Применяем медианный фильтр
    
    img_np = np.array(image, dtype=np.uint8)                    # Конвертируем в numpy для фильтра Лапласа
    
    laplace_filtered = cv2.Laplacian(img_np, cv2.CV_64F)        # Фильтр Лапласа
    laplace_filtered = np.uint8(np.absolute(laplace_filtered))  # Абсолютные значения
    
    return img_np, laplace_filtered                             # Возвращаем исходное и фильтрованное изображение

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
        self.sar_images = sorted(os.listdir(sar_dir))
        self.optical_images = sorted(os.listdir(optical_dir))

    def __len__(self):
        return len(self.sar_images)

    def __getitem__(self, idx):
        # Загружаем SAR-изображение
        sar_path = os.path.join(self.sar_dir, self.sar_images[idx])
        sar_image = Image.open(sar_path).convert("L")  # Ч/б изображение
        
        # Получаем исходное и Laplace-фильтрованное изображение
        sar_np, laplace_filtered = preprocess_with_laplace(sar_image)
        
        # Объединяем SAR и Laplace по каналу
        combined_sar = np.stack([sar_np, laplace_filtered], axis=-1)            # (H, W, 2)
        combined_sar = Image.fromarray(combined_sar)                            # Конвертируем в PIL

        # Загружаем оптическое изображение
        optical_path = os.path.join(self.optical_dir, self.optical_images[idx])
        optical_image = Image.open(optical_path).convert("RGB")                 # Цветное изображение

        # Применяем совместимые аугментации, если таковые есть
        if self.augment:
            combined_sar, optical_image = self.augment(combined_sar, optical_image)

        # Применяем трансформации, если таковые есть
        if self.transform:
            combined_sar = self.transform(combined_sar)
            optical_image = self.transform(optical_image)

        return combined_sar, optical_image

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

# DataLoader для батчей
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# Проверка работы
if __name__ == "__main__":
    for sar, optical in train_loader:
        print(f"SAR batch shape: {sar.shape}")
        print(f"Optical batch shape: {optical.shape}")
        break

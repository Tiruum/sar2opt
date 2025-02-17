import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils.ConfigLoader import ConfigLoader
config = ConfigLoader()

# Настройки
IMAGE_SIZE = config.get('dataset', 'image_size')  # Размер для ресайза изображений
DATA_DIR = "./dataset"  # Путь к папке с данными
BATCH_SIZE = config.get('dataset', 'batch_size') # Размер батча

# Трансформации
common_transform = A.Compose([
    A.Affine(
        scale=(0.9, 1.1),
        translate_percent=(-0.1, 0.1),
        rotate=(-15, 15),
        border_mode=cv2.BORDER_REFLECT_101,
        p=0.9
    ),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
], additional_targets={'optical': 'image'})

# Отдельные пайплайны для специфичных аугментаций
sar_specific = A.Compose([
    A.GaussianBlur(blur_limit=3, p=0.3),
    A.GaussNoise(std_range=(0.01, 0.05), p=0.7),
    A.Normalize(mean=(0.5,), std=(0.5,)),
    ToTensorV2()
])

optical_specific = A.Compose([
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

resize_transform = A.Resize(IMAGE_SIZE, IMAGE_SIZE)

# Датасет
class SARToOpticalDataset(Dataset):
    def __init__(self, sar_dir, optical_dir,
                 common_transform=None,
                 sar_specific=None,
                 optical_specific=None,
                 resize_transform=None):
        self.sar_dir = sar_dir
        self.optical_dir = optical_dir
        self.common_transform = common_transform
        self.sar_specific = sar_specific
        self.optical_specific = optical_specific
        self.resize_transform = resize_transform

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
        sar_img = cv2.imread(sar_path, cv2.IMREAD_GRAYSCALE)  # Ч/б изображение

        # Загружаем оптическое изображение
        optical_path = os.path.join(self.optical_dir, self.optical_images[idx])
        optical_img = cv2.imread(optical_path, cv2.IMREAD_COLOR)  # Цветное изображение
        optical_img = cv2.cvtColor(optical_img, cv2.COLOR_BGR2RGB)

        sar_np = np.array(sar_img)  # shape: (H, W)
        optical_np = np.array(optical_img)  # shape: (H, W, 3)

        # Если необходимо изменить размер до IMAGE_SIZE x IMAGE_SIZE
        if self.resize_transform:
            # Albumentations ожидает ключ 'image'
            sar_np = self.resize_transform(image=sar_np)['image']
            optical_np = self.resize_transform(image=optical_np)['image']

        # Приводим SAR к формату (H, W, 1)
        if len(sar_np.shape) == 2:
            sar_np = np.expand_dims(sar_np, axis=-1)

        # Дублируем канал SAR, чтобы получить трёхканальное изображение для общего пайплайна
        sar_3ch = np.concatenate([sar_np, sar_np, sar_np], axis=-1)

        # Применяем общий пайплайн синхронно к обоим изображениям
        if self.common_transform:
            transformed = self.common_transform(image=sar_3ch, optical=optical_np)
            sar_common = transformed['image']        # по-прежнему 3 канала
            optical_common = transformed['optical']
        else:
            sar_common = sar_3ch
            optical_common = optical_np

        # Преобразуем SAR обратно в одноканальное – берём, например, первый канал
        sar_1ch = sar_common[..., 0]
        sar_1ch = np.expand_dims(sar_1ch, axis=-1)  # shape: (H, W, 1)

        # Применяем специфичные трансформации (нормализация, ToTensorV2)
        if self.sar_specific:
            sar_final = self.sar_specific(image=sar_1ch)['image']
        else:
            sar_final = sar_1ch

        if self.optical_specific:
            optical_final = self.optical_specific(image=optical_common)['image']
        else:
            optical_final = optical_common

        return sar_final, optical_final

train_dataset = SARToOpticalDataset(
    sar_dir=os.path.join(DATA_DIR, "trainA"),
    optical_dir=os.path.join(DATA_DIR, "trainB"),
    common_transform=common_transform,
    sar_specific=sar_specific,
    optical_specific=optical_specific,
    resize_transform=resize_transform
)

test_dataset = SARToOpticalDataset(
    sar_dir=os.path.join(DATA_DIR, "testA"),
    optical_dir=os.path.join(DATA_DIR, "testB"),
    common_transform=None,  # или можно оставить None для тестовых данных
    sar_specific=sar_specific,
    optical_specific=optical_specific,
    resize_transform=resize_transform
)

# DataLoader для батчей
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=os.cpu_count() // 2,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=os.cpu_count() // 2,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2
)

# Проверка работы
if __name__ == "__main__":
    for sar, optical in train_loader:
        print(f"SAR batch shape: {sar.shape}")
        print(f"Optical batch shape: {optical.shape}")
        break
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

# Настройки
IMAGE_SIZE = 256  # Размер для ресайза изображений
DATA_DIR = "./dataset"  # Путь к папке с данными
BATCH_SIZE = 16  # Размер батча

# Аугментации
augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),  # Случайное отражение по горизонтали
    transforms.RandomVerticalFlip(p=0.5),  # Случайное отражение по вертикали
    transforms.RandomRotation(degrees=15),  # Случайное вращение на ±15 градусов\
])

# Трансформации
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),  # Преобразование [0, 255] -> [0, 1]
    transforms.Normalize(mean=(0.5,), std=(0.5,))  # Преобразование [0, 1] -> [-1, 1]
])

# Функция для применения медианного фильтра и Лапласа
def preprocess_with_laplace(image):
    # Применяем медианный фильтр
    image = image.filter(ImageFilter.MedianFilter(size=5))
    
    # Конвертируем в numpy для фильтра Лапласа
    img_np = np.array(image, dtype=np.uint8)
    
    # Фильтр Лапласа
    laplace_filtered = cv2.Laplacian(img_np, cv2.CV_64F)
    laplace_filtered = np.uint8(np.absolute(laplace_filtered))  # Абсолютные значения
    
    return img_np, laplace_filtered  # Возвращаем исходное и фильтрованное изображение


# Датасет
class SARToOpticalDataset(Dataset):
    def __init__(self, sar_dir, optical_dir, transform=None, augment=None):
        self.sar_dir = sar_dir
        self.optical_dir = optical_dir
        self.transform = transform
        self.augment = augment
        self.sar_images = os.listdir(sar_dir)
        self.optical_images = os.listdir(optical_dir)

    def __len__(self):
        return len(self.sar_images)

    def __getitem__(self, idx):
        # Загружаем SAR-изображение
        sar_path = os.path.join(self.sar_dir, self.sar_images[idx])
        sar_image = Image.open(sar_path).convert("L")  # Ч/б изображение
        
        # Получаем исходное и Laplace-фильтрованное изображение
        sar_np, laplace_filtered = preprocess_with_laplace(sar_image)
        
        # Объединяем SAR и Laplace по каналу
        combined_sar = np.stack([sar_np, laplace_filtered], axis=-1)  # (H, W, 2)
        combined_sar = Image.fromarray(combined_sar)  # Конвертируем в PIL

        # Загружаем оптическое изображение
        optical_path = os.path.join(self.optical_dir, self.optical_images[idx])
        optical_image = Image.open(optical_path).convert("RGB")  # Цветное изображение

        # Применяем аугментации, если таковые есть
        if self.augment:
            augmented = self.augment(transforms.ToTensor()(combined_sar))
            combined_sar = to_pil_image(augmented)
            optical_image = self.augment(optical_image)
        
        # Применяем трансформации, если таковые есть
        if self.transform:
            combined_sar = self.transform(combined_sar)
            optical_image = self.transform(optical_image)
        
        return combined_sar, optical_image

# Создаём экземпляры датасетов
train_dataset = SARToOpticalDataset(
    sar_dir=os.path.join(DATA_DIR, "trainA"),
    optical_dir=os.path.join(DATA_DIR, "trainB"),
    transform=transform,
    augment=None
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


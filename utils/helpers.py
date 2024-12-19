import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import median_filter

# Настройки
IMAGE_SIZE = 256  # Размер для ресайза изображений
DATA_DIR = "./dataset"  # Путь к папке с данными
BATCH_SIZE = 8  # Размер батча

# --- Фильтр Собеля ---
def sobel_filter(image):
    """Функция для применения фильтра Собеля."""
    sobel_x = torch.tensor([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # 1x1x3x3
    sobel_y = torch.tensor([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # 1x1x3x3
    
    image = image.unsqueeze(0)  # Добавляем batch dimension
    
    # Применяем свертку с ядрами Собеля
    edge_x = F.conv2d(image, sobel_x, padding=1)
    edge_y = F.conv2d(image, sobel_y, padding=1)
    
    # Суммируем градиенты
    edges = torch.sqrt(edge_x**2 + edge_y**2).squeeze(0)  # Возвращаем размер [1, H, W]
    return edges

# --- Медианный фильтр ---
def apply_median_filter(image_tensor, kernel_size=3):
    """Функция для применения медианного фильтра."""
    image_np = image_tensor.squeeze(0).numpy()  # Переводим из Tensor в numpy
    filtered = median_filter(image_np, size=kernel_size)
    return torch.tensor(filtered, dtype=torch.float32).unsqueeze(0)  # Возвращаем Tensor с размером [1, H, W]

# Трансформации
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),  # Преобразование [0, 255] -> [0, 1]
    transforms.Normalize(mean=(0.5,), std=(0.5,))  # Преобразование [0, 1] -> [-1, 1]
])

# Датасет
class SARToOpticalDataset(Dataset):
    def __init__(self, sar_dir, optical_dir, transform=None):
        self.sar_dir = sar_dir
        self.optical_dir = optical_dir
        self.transform = transform
        self.sar_images = os.listdir(sar_dir)
        self.optical_images = os.listdir(optical_dir)

    def __len__(self):
        return len(self.sar_images)
        return len(self.sar_images)

    def __getitem__(self, idx):
        # Загружаем SAR-изображение
        sar_path = os.path.join(self.sar_dir, self.sar_images[idx])
        sar_image = Image.open(sar_path).convert("L")  # Ч/б изображение
        
        # Загружаем оптическое изображение
        optical_path = os.path.join(self.optical_dir, self.optical_images[idx])
        optical_image = Image.open(optical_path).convert("RGB")  # Цветное изображение
        
        if self.transform:
            sar_image = self.transform(sar_image)
            optical_image = self.transform(optical_image)
            
            # Применяем фильтры к SAR-изображению
            sar_image = apply_median_filter(sar_image)  # Медианный фильтр
            sar_image = sobel_filter(sar_image)  # Фильтр Собеля
        
        return sar_image, optical_image

# Создаём экземпляры датасетов
train_dataset = SARToOpticalDataset(
    sar_dir=os.path.join(DATA_DIR, "trainA"),
    optical_dir=os.path.join(DATA_DIR, "trainB"),
    transform=transform
)

test_dataset = SARToOpticalDataset(
    sar_dir=os.path.join(DATA_DIR, "testA"),
    optical_dir=os.path.join(DATA_DIR, "testB"),
    transform=transform
)

# DataLoader для батчей
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Проверка работы
if __name__ == "__main__":
    for sar, optical in train_loader:
        print(f"SAR batch shape: {sar.shape}")
        print(f"Optical batch shape: {optical.shape}")
        break

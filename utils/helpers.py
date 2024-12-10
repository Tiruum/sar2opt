import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Настройки
IMAGE_SIZE = 256  # Размер для ресайза изображений
DATA_DIR = "../dataset"  # Путь к папке с данными
BATCH_SIZE = 8  # Размер батча

# Нормализация в диапазоне [-1, 1]
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

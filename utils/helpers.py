import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Настройки
IMAGE_SIZE = 256  # Размер для ресайза изображений
DATA_DIR = os.path.abspath(os.path.join(os.getcwd(), 'dataset'))  # Путь к папке с данными
BATCH_SIZE = 8  # Размер батча

# Нормализация в диапазоне [-1, 1]
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),  # Преобразование [0, 255] -> [0, 1]
    transforms.Normalize(mean=(0.5,), std=(0.5,))  # Преобразование [0, 1] -> [-1, 1]
])

# Датасет
class SARToOpticalDataset(Dataset):
    def __init__(self, sar_dir, optical_dir, sar_transform=None, optical_transform=None):
        self.sar_dir = sar_dir
        self.optical_dir = optical_dir
        self.sar_transform = sar_transform
        self.optical_transform = optical_transform
        self.sar_images = os.listdir(sar_dir)
        self.optical_images = os.listdir(optical_dir)

    def __len__(self):
        return len(self.sar_images)

    def __getitem__(self, idx):
        # Загружаем SAR-изображение
        sar_path = os.path.join(self.sar_dir, self.sar_images[idx])
        sar_image = Image.open(sar_path).convert("L")  # SAR: Ч/б изображение
        
        # Загружаем оптическое изображение
        optical_path = os.path.join(self.optical_dir, self.optical_images[idx])
        optical_image = Image.open(optical_path).convert("RGB")  # Оптическое: Цветное изображение
        
        # Применяем аугментации
        if self.sar_transform:
            sar_image = self.sar_transform(sar_image)
        if self.optical_transform:
            optical_image = self.optical_transform(optical_image)
        
        return sar_image, optical_image

# Аугментации для SAR-изображений
sar_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Случайное горизонтальное отражение
    transforms.RandomVerticalFlip(),    # Случайное вертикальное отражение
    transforms.RandomRotation(30),      # Поворот на угол до 30 градусов
    transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),  # Масштабирование с обрезкой
    transforms.ToTensor(),              # Преобразование в тензор
    transforms.Normalize(mean=(0.5,), std=(0.5,))  # Нормализация в диапазоне [-1, 1]
])

# Аугментации для оптических изображений
optical_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Цветовые аугментации
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # Нормализация [-1, 1]
])

# Создаём экземпляры датасетов
train_dataset = SARToOpticalDataset(
    sar_dir=os.path.join(DATA_DIR, "trainA"),
    optical_dir=os.path.join(DATA_DIR, "trainB"),
    sar_transform=sar_transform,
    optical_transform=optical_transform
)

test_dataset = SARToOpticalDataset(
    sar_dir=os.path.join(DATA_DIR, "testA"),
    optical_dir=os.path.join(DATA_DIR, "testB"),
    sar_transform=transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ]),
    optical_transform=transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
)


# DataLoader для батчей
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Проверка работы
if __name__ == "__main__":
    for sar, optical in train_loader:
        print(f"SAR batch shape: {sar.shape}")
        print(f"Optical batch shape: {optical.shape}")

        print(f"SAR min: {sar.min().item()}, SAR max: {sar.max().item()}")
        print(f"Optical min: {optical.min().item()}, Optical max: {optical.max().item()}")
        break

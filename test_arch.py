import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image

# Твои импорты генератора
from models.generator import UNetGenerator
from utils.Dataset import SARToOpticalDataset
from utils.Config import Config

# Фиксация сидов для детерминизма
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Загружаем одну пару без common_transform (отключаем аугментацию)
one_image_dataset = SARToOpticalDataset(
    sar_dir=os.path.join(Config.DATA_DIR, "trainA"),
    optical_dir=os.path.join(Config.DATA_DIR, "trainB"),
    common_transform=None,  # ВАЖНО: убрать аугментацию
    sar_specific=None,      # Тоже убираем, чтобы ничего не трогало SAR
    optical_specific=None,  # И оптику тоже
    resize_transform=None   # И ресайз уберём, работаем в оригинале
)

sar, optical = one_image_dataset[0]  # Берём первую пару

# Убедимся что тензоры в правильном формате
sar = torch.from_numpy(sar).unsqueeze(0).permute(0, 3, 1, 2).float() / 255.0 * 2.0 - 1.0
optical = torch.from_numpy(optical).unsqueeze(0).permute(0, 3, 1, 2).float() / 255.0 * 2.0 - 1.0

device = Config.DEVICE
sar = sar.to(device)
optical = optical.to(device)

# Инициализация генератора
generator = UNetGenerator(input_nc=1, output_nc=3, ngf=Config.NGF, n_blocks=8).to(device)

# Лосс и оптимайзер
criterion = nn.L1Loss()
optimizer = optim.Adam(generator.parameters(), lr=1e-4)

# Папка для сохранения прогрессии
os.makedirs("./debug_outputs", exist_ok=True)

# Обучение
epochs = 2000
for epoch in range(epochs):
    generator.train()

    optimizer.zero_grad()
    output = generator(sar)
    loss = criterion(output, optical)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0 or epoch == epochs - 1:
        print(f"Epoch [{epoch}/{epochs}] | Loss: {loss.item():.6f}")
        save_image((output + 1) * 0.5, f"./debug_outputs/fake_epoch{epoch}.png")
        save_image((optical + 1) * 0.5, f"./debug_outputs/real.png")  # Сохраним таргет один раз
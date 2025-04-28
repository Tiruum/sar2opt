import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image

# Твои импорты генератора
from models.generator import UNetGenerator
from models.discriminator import NLayerDiscriminator
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
discriminator = NLayerDiscriminator(input_nc=1 + 3, ndf=Config.NDF, n_layers=3).to(device)

bce_loss = nn.BCEWithLogitsLoss()
l1_loss = nn.L1Loss()

optimizer_G = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

# Папка для сохранения прогрессии
os.makedirs("./debug_outputs_gan", exist_ok=True)

lambda_gan = 1.0
lambda_l1 = 100.0

# Обучение
epochs = 2000
for epoch in range(epochs):
    # Генератор
    generator.train()
    discriminator.train()

    optimizer_G.zero_grad()
    optimizer_D.zero_grad()

    fake_optical = generator(sar)

    # --- Дискриминатор ---
    real_pair = torch.cat([sar, optical], dim=1)
    fake_pair = torch.cat([sar, fake_optical.detach()], dim=1)

    pred_real = discriminator(real_pair)
    pred_fake = discriminator(fake_pair)

    valid = torch.ones_like(pred_real)
    fake = torch.zeros_like(pred_fake)

    loss_D_real = bce_loss(pred_real, valid)
    loss_D_fake = bce_loss(pred_fake, fake)
    loss_D = (loss_D_real + loss_D_fake) * 0.5

    loss_D.backward()
    optimizer_D.step()

    # --- Генератор ---
    fake_pair = torch.cat([sar, fake_optical], dim=1)
    pred_fake = discriminator(fake_pair)

    loss_G_gan = bce_loss(pred_fake, valid)
    loss_G_l1 = l1_loss(fake_optical, optical)

    loss_G = lambda_gan * loss_G_gan + lambda_l1 * loss_G_l1

    loss_G.backward()
    optimizer_G.step()

    if epoch % 100 == 0 or epoch == epochs - 1:
        print(f"Epoch [{epoch}/{epochs}] | Loss_G: {loss_G.item():.6f} | Loss_D: {loss_D.item():.6f}")
        save_image((fake_optical + 1) * 0.5, f"./debug_outputs_gan/fake_epoch{epoch}.png")
        save_image((optical + 1) * 0.5, f"./debug_outputs_gan/real.png")
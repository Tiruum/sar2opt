# test.py

import os
import torch
from torchvision.utils import save_image
from tqdm import tqdm

from models.generator import UNetGenerator
from utils import visualize_batch
from utils.Dataset import test_loader
from utils.Config import Config
from typing import Literal

def load_checkpoint(model, checkpoint_path, device=Config.DEVICE):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def test():
    device = torch.device(Config.DEVICE)

    # Инициализируем генератор
    netG = UNetGenerator(
        input_nc=Config.INPUT_NC,
        output_nc=Config.OUTPUT_NC,
        ngf=Config.NGF,
        n_blocks=8
    ).to(device)

    # Путь к чекпоинту генератора
    checkpoint_path = os.path.join(Config.CHECKPOINTS_DIR, "netG_epoch_300.pth")  # укажи актуальный чекпоинт!
    netG = load_checkpoint(netG, checkpoint_path, device)
    netG.eval()

    # Создаем директории для сохранения результатов
    save_dir = os.path.join(Config.RESULTS_DIR, "test")
    os.makedirs(save_dir, exist_ok=True)

    # Прогоняем тест
    with torch.no_grad():
        for idx, (sar, real_optical) in enumerate(tqdm(test_loader, desc="Testing")):
            sar = sar.to(device)
            real_optical = real_optical.to(device)

            fake_optical = netG(sar)

            visualize_batch(sar, fake_optical, real_optical,
                            save_path=os.path.join(f'{save_dir}', f"{idx:04d}.png"),
                            max_rows=6, mode='quality')
if __name__ == "__main__":
    test()
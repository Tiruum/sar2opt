# test.py

import os
import torch
from torchvision.utils import save_image
from tqdm import tqdm

from models.generator import GlobalGenerator
from utils.Dataset import test_loader
from utils.Config import Config

def load_checkpoint(model, checkpoint_path, device='cuda'):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def test():
    device = torch.device(Config.DEVICE)

    # Инициализируем генератор
    netG = GlobalGenerator(
        input_nc=Config.INPUT_NC,
        output_nc=Config.OUTPUT_NC,
        ngf=Config.NGF,
        n_blocks=9
    ).to(device)

    # Путь к чекпоинту генератора
    checkpoint_path = os.path.join(Config.CHECKPOINTS_DIR, "netG_epoch_200.pth")  # укажи актуальный чекпоинт!
    netG = load_checkpoint(netG, checkpoint_path, device)
    netG.eval()

    # Создаем директории для сохранения результатов
    # save_dir_fake = os.path.join(Config.RESULTS_DIR, "test", "fake")
    # save_dir_real = os.path.join(Config.RESULTS_DIR, "test", "real")
    # save_dir_sar = os.path.join(Config.RESULTS_DIR, "test", "sar")
    save_dir_concatenated = os.path.join(Config.RESULTS_DIR, "test", "concatenated")
    # os.makedirs(save_dir_fake, exist_ok=True)
    # os.makedirs(save_dir_real, exist_ok=True)
    # os.makedirs(save_dir_sar, exist_ok=True)
    os.makedirs(save_dir_concatenated, exist_ok=True)

    # Прогоняем тест
    with torch.no_grad():
        for idx, (sar, real_optical) in enumerate(tqdm(test_loader, desc="Testing")):
            sar = sar.to(device)
            real_optical = real_optical.to(device)

            fake_optical = netG(sar)

            # Нормализуем в [0,1] для сохранения
            fake_optical_vis = (fake_optical + 1) / 2.0
            real_optical_vis = (real_optical + 1) / 2.0
            sar_vis = (sar + 1) / 2.0
            sar_vis = sar_vis.repeat(1, 3, 1, 1)

            # Склеиваем изображения по вертикали
            concatenated_output = torch.cat((fake_optical_vis, real_optical_vis, sar_vis), dim=2)

            # save_image(fake_optical_vis, os.path.join(save_dir_fake, f"{idx:04d}_fake.png"))
            # save_image(real_optical_vis, os.path.join(save_dir_real, f"{idx:04d}_real.png"))
            # save_image(sar_vis, os.path.join(save_dir_sar, f"{idx:04d}_sar.png"))
            save_image(concatenated_output, os.path.join(save_dir_concatenated, f"{idx:04d}_concatenated.png"))

if __name__ == "__main__":
    test()
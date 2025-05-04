# test.py

import os
import torch
from tqdm import tqdm

from utils import visualize_batch
from utils.Dataset import test_loader
from utils.Config import Config
from utils.Factory import build_models, build_optimizers
from utils.checkpoints import load_checkpoint

def test():
    device = torch.device(Config.DEVICE)

    # Инициализируем генератор
    netG, netD = build_models(device)
    optG, optD = build_optimizers(netG, netD)

    # Путь к чекпоинту генератора
    checkpoint_path = os.path.join(Config.CHECKPOINTS_DIR, "netG_epoch_100.pth")  # укажи актуальный чекпоинт!
    netG, _, _ = load_checkpoint(netG, optG, checkpoint_path, device)
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
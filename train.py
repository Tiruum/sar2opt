# train.py

import os
from typing import Dict, Optional
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from time import time
from torch.amp import autocast, GradScaler

from utils import sec2hhmmss, visualize_batch
from utils.Dataset import mini_train_loader, mini_test_loader
from utils.Dataset import train_loader, test_loader
from utils.Config import Config
from utils.Factory import build_criterions, build_models, build_optimizers
from utils.Logger import Logger
from utils.checkpoints import load_checkpoint, save_checkpoint

torch.backends.cudnn.benchmark = Config.CUDNN_BENCHMARK

# train_loader = mini_train_loader
# test_loader = mini_test_loader

logger = Logger(name="SAR2OPT")

def check_loss_validity(loss_value: torch.Tensor, name: str, epoch: int, iter_idx: int) -> torch.Tensor:
    """Проверяет значение функции потерь на NaN и Inf."""
    if torch.isnan(loss_value) or torch.isinf(loss_value):
        logger.warning(f"Обнаружен NaN или Inf в {name} на эпохе {epoch}, итерации {iter_idx}")
        return torch.tensor(0.0, device=loss_value.device)
    return loss_value

def train_epoch(
    netG: nn.Module,
    netD: nn.Module,
    optimizer_G: torch.optim.Optimizer,
    optimizer_D: torch.optim.Optimizer,
    crits: Dict[str, nn.Module],
    epoch: int,
    device: torch.device,
    amp_device_type: str
) -> Dict[str, float]:
    netG.train()
    netD.train()

    total_g_loss = 0
    total_d_loss = 0

    scaler = GradScaler(enabled=Config.USE_AMP)

    progress_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.NUM_EPOCHS} Train", ascii=" ▏▎▍▌▋▊▉█", smoothing=0.5)

    for i, (real_sar, real_optical) in enumerate(progress_train):
        real_sar = real_sar.to(device, memory_format=torch.channels_last)
        real_optical = real_optical.to(device, memory_format=torch.channels_last)

        # --------- Обновление дискриминатора ---------
        netD.requires_grad_(True)
        optimizer_D.zero_grad(set_to_none=True)

        # Генерируем фейковое изображение
        with torch.no_grad(), autocast(device_type=amp_device_type, enabled=Config.USE_AMP):
            fake_optical = netG(real_sar)

        with autocast(device_type=amp_device_type, enabled=Config.USE_AMP):
            # Конкатенируем SAR + Optical
            fake_pair = torch.cat((real_sar, fake_optical), dim=1)
            real_pair = torch.cat((real_sar, real_optical), dim=1)

            pred_fake = netD(fake_pair)
            pred_real = netD(real_pair)

            # Считаем Loss дискриминатора
            d_loss_fake = sum(crits['GAN'](fake, False) for fake in pred_fake)
            d_loss_real = sum(crits['GAN'](real, True, real_label_smooth=0.9) for real in pred_real)

            d_loss = (d_loss_fake + d_loss_real) * 0.5
            d_loss = check_loss_validity(d_loss, "d_loss", epoch, i)

        if Config.USE_AMP:
            scaler.scale(d_loss).backward()
            scaler.step(optimizer_D)
            scaler.update()
        else:
            d_loss.backward()
            optimizer_D.step()

        # --------- Обновление генератора ---------
        netD.requires_grad_(False)
        optimizer_G.zero_grad(set_to_none=True)

        with autocast(device_type=amp_device_type, enabled=Config.USE_AMP):
            # Снова прогоняем (чтобы получить свежие данные после обновления дискриминатора)
            fake_optical = netG(real_sar)
            fake_pair = torch.cat((real_sar, fake_optical), dim=1)
            pred_fake = netD(fake_pair)

            # Вычисление всех компонентов функции потерь
            g_gan_loss = sum(crits['GAN'](fake, True) for fake in pred_fake)                                    # GAN Loss генератора
            l1_loss = crits['L1'](fake_optical, real_optical)                                                   # L1 Loss
            fm_loss = sum(crits['FM']([fake], [real.detach()]) for fake, real in zip(pred_fake, pred_real))     # Feature Matching Loss
            perceptual_loss = crits['Perceptual'](fake_optical, real_optical.detach())                          # Perceptual Loss
            tv_loss = crits['TV'](fake_optical)                                                        # Total Variation Loss
            lpips_loss = crits['LPIPS'](fake_optical, real_optical.detach())                                    # LPIPS Loss
            lab_l, lab_ab = crits['Lab'](fake_optical, real_optical)                                            # LabColor (раздельно L и ab)
            g_ssim = crits['SSIM'](fake_optical, real_optical)                                                  # SSIM (DSSIM)
            edge_loss = crits['Edge'](fake_optical, real_optical)                                               # Edge Loss

            # Общий Loss генератора с весами
            g_loss = (
                g_gan_loss * Config.GAN_LOSS_WEIGHT +
                l1_loss * Config.L1_LOSS_WEIGHT +
                fm_loss * Config.FM_LOSS_WEIGHT +
                perceptual_loss * Config.PERCEPTUAL_LOSS_WEIGHT +
                tv_loss * Config.TV_LOSS_WEIGHT +
                lpips_loss * Config.LPIPS_LOSS_WEIGHT +
                lab_l * Config.LAB_L_LOSS_WEIGHT +
                lab_ab * Config.LAB_AB_LOSS_WEIGHT +
                g_ssim * Config.SSIM_LOSS_WEIGHT +
                edge_loss * Config.EDGE_LOSS_WEIGHT
            )
            
            # Проверка на NaN/Inf
            g_loss = check_loss_validity(g_loss, "g_loss", epoch, i)

        if Config.USE_AMP:
            scaler.scale(g_loss).backward()
            scaler.step(optimizer_G)
            scaler.update()
        else:
            g_loss.backward()
            optimizer_G.step()

        total_g_loss += g_loss.item()
        total_d_loss += d_loss.item()

        progress_train.set_postfix({
            "G_loss": f"{g_loss.item():.4f}",
            "D_loss": f"{d_loss.item():.4f}"
        })

    return {
        'G_loss': total_g_loss / len(train_loader),
        'D_loss': total_d_loss / len(train_loader),
        'L1': l1_loss.item(),
        'FM': fm_loss.item(),
        'Perceptual': perceptual_loss.item(),
        'LPIPS': lpips_loss.item(),
        'TV': tv_loss.item(),
        'GAN': g_gan_loss.item(),
        'Lab_L': lab_l.item(), 
        'Lab_ab': lab_ab.item(),
        'SSIM': g_ssim.item(),
        'Edge': edge_loss.item()
    }

def val_epoch(
    netG: nn.Module, 
    crits: Dict[str, nn.Module], 
    epoch: int, 
    device: torch.device, 
    amp_device_type: str
) -> Dict[str, float]:
    netG.eval()
    val_metrics = {
        'FM': 0.0,
        'L1': 0.0,
        'Perceptual': 0.0,
        'LPIPS': 0.0,
        'Lab_L': 0.0,
        'Lab_ab': 0.0,
        'SSIM': 0.0,
        'Edge': 0.0,
        'TV': 0.0,
    }

    with torch.no_grad():
        progress_test = tqdm(test_loader, desc=f"Epoch {epoch+1}/{Config.NUM_EPOCHS} Val", ascii=" ▏▎▍▌▋▊▉█", smoothing=0.5)
        for real_sar, real_optical in progress_test:
            real_sar = real_sar.to(device, memory_format=torch.channels_last)
            real_optical = real_optical.to(device, memory_format=torch.channels_last)
            with autocast(device_type=amp_device_type, enabled=Config.USE_AMP):
                fake_optical = netG(real_sar)

                # Accumulate metrics
                val_metrics['L1'] += crits['L1'](fake_optical, real_optical).item()
                val_metrics['Perceptual'] += crits['Perceptual'](fake_optical, real_optical).item()
                val_metrics['LPIPS'] += crits['LPIPS'](fake_optical, real_optical).item()
                val_metrics['TV'] += crits['TV'](fake_optical).item()
                l_l, l_ab = crits['Lab'](fake_optical, real_optical)
                val_metrics['Lab_L'] += l_l.item()
                val_metrics['Lab_ab'] += l_ab.item()
                val_metrics['SSIM'] += crits['SSIM'](fake_optical, real_optical).item()
                val_metrics['Edge'] += crits['Edge'](fake_optical, real_optical).item()

    return val_metrics


def train(
    run_name: Optional[str] = None,
    resume_g_path: Optional[str] = None,
    resume_d_path: Optional[str] = None
) -> None:
    # Настройка устройства и режима AMP
    device = torch.device(Config.DEVICE)
    amp_device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Настройка имени запуска и директории для логов
    if run_name is None:
        run_name = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    log_dir = os.path.join(Config.RESULTS_DIR, 'logs', run_name)
    writer = SummaryWriter(log_dir=log_dir)

    # Инициализация моделей, оптимизаторов и функций потерь
    netG, netD = build_models(device)
    optimizer_G, optimizer_D = build_optimizers(netG, netD)
    crits = build_criterions(device)

    # Подготовка фиксированного набора данных для визуализации
    fixed_real_sar, fixed_real_optical = next(iter(train_loader))
    fixed_real_sar = fixed_real_sar.to(device, memory_format=torch.channels_last)
    fixed_real_optical = fixed_real_optical.to(device, memory_format=torch.channels_last)

    # Загрузка чекпоинтов при необходимости
    start_epoch = 0
    if resume_g_path and resume_d_path:
        logger.info(f"Продолжение обучения с чекпоинтов:\nG: {resume_g_path}\nD: {resume_d_path}")
        netG, optimizer_G, start_epoch = load_checkpoint(netG, optimizer_G, resume_g_path, device)
        netD, optimizer_D, _ = load_checkpoint(netD, optimizer_D, resume_d_path, device)

    # Создание директорий для чекпоинтов
    os.makedirs(Config.CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)

    # Основной цикл обучения
    for epoch in range(start_epoch, Config.NUM_EPOCHS):
        # --- TRAIN LOOP ---
        train_metrics = train_epoch(netG, netD, optimizer_G, optimizer_D, crits, epoch, device, amp_device_type)
        for name, value in train_metrics.items():
            writer.add_scalar(f'Train/{name}', value, epoch)

        # --- VALIDATION LOOP ---
        val_metrics = val_epoch(netG, crits, epoch, device, amp_device_type)
        for name, total in val_metrics.items():
            avg = total / len(test_loader)
            writer.add_scalar(f'Val/{name}', avg, epoch)

        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()

        # Сохранение чекпоинта
        if (epoch + 1) % 10 == 0:
            save_checkpoint(netG, optimizer_G, epoch, os.path.join(Config.CHECKPOINTS_DIR, f"netG_epoch_{epoch+1}.pth"))
            save_checkpoint(netD, optimizer_D, epoch, os.path.join(Config.CHECKPOINTS_DIR, f"netD_epoch_{epoch+1}.pth"))

        os.makedirs(f'{Config.RESULTS_DIR}/train', exist_ok=True)
        # Сохраняем одну сгенерированную картинку каждые 10 эпох
        if (epoch + 1) % 10 == 0:
            netG.eval()
            with torch.no_grad():
                with autocast(device_type=amp_device_type, enabled=Config.USE_AMP):
                    fake_optical = netG(fixed_real_sar)

                if Config.USE_AMP:
                    fake_optical = fake_optical.float()

                visualize_batch(fixed_real_sar, fake_optical, fixed_real_optical,
                                save_path=os.path.join(f'{Config.RESULTS_DIR}/train', f"epoch_{epoch+1}.png"),
                                max_rows=6, mode='quality', title=f"Epoch {epoch+1}")
    writer.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default=None, help='Имя текущего прогона (для TensorBoard)')
    parser.add_argument('--resume_g', type=str, default=None, help='Путь до чекпоинта генератора')
    parser.add_argument('--resume_d', type=str, default=None, help='Путь до чекпоинта дискриминатора')
    args = parser.parse_args()
    start_time = time()
    logger.info(f"Начало обучения")
    try:
        train(run_name=args.run_name, resume_g_path=args.resume_g, resume_d_path=args.resume_d)
        logger.success(f"Обучение  завершено ({sec2hhmmss(time() - start_time)})")
    except Exception as e:
        logger.error(f"Ошибка при обучении: {str(e)}")
        raise
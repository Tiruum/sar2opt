# train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from datetime import datetime

from models.generator import UNetGenerator
from models.multiscale_discriminator import MultiscaleDiscriminator
from models.losses import (
    EdgeLoss, GANLoss, L1Loss, FeatureMatchingLoss,
    PerceptualLoss, LPIPSLoss,
    LabColorLoss, SSIMLoss
)

from utils.Dataset import train_loader
from utils.Config import Config

import pandas as pd

def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)

def total_variation_loss(img):
    tv_h = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]))
    tv_w = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]))
    return tv_h + tv_w

def train(run_name: str = None):
    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(Config.RESULTS_DIR, 'logs', run_name)
    writer = SummaryWriter(log_dir=log_dir)
    device = torch.device(Config.DEVICE)
    losses_logs = {
        "epoch": [],
        "G_loss": [],
        "D_loss": [],
        "L1": [],
        "FeatureMatching": [],
        "Perceptual": [],
        "LPIPS": [],
        "TotalVariation": [],
        "GAN": []
    }

    # Инициализация моделей
    netG = UNetGenerator(
        input_nc=Config.INPUT_NC,
        output_nc=Config.OUTPUT_NC,
        ngf=Config.NGF,
        n_blocks=8
    ).to(device)

    netD = MultiscaleDiscriminator(
        input_nc=Config.INPUT_NC + Config.OUTPUT_NC,
        ndf=Config.NDF,
        n_layers=3,
        num_D=3
    ).to(device)

    # Лосс функции
    criterionGAN = GANLoss(use_lsgan=True).to(device)
    criterionL1 = L1Loss().to(device)
    criterionFM = FeatureMatchingLoss().to(device)
    criterionPerceptual = PerceptualLoss().to(device)
    criterionLPIPS = LPIPSLoss().to(device)
    criterionLab = LabColorLoss().to(device)
    criterionSSIM = SSIMLoss().to(device)
    criterionEdge = EdgeLoss(mode='sobel').to(device)

    # Оптимизаторы
    optimizer_G = optim.Adam(netG.parameters(), lr=Config.LEARNING_RATE_G, betas=(Config.BETA1, Config.BETA2))
    optimizer_D = optim.Adam(netD.parameters(), lr=Config.LEARNING_RATE_D, betas=(Config.BETA1, Config.BETA2))

    # Создание директорий для чекпоинтов
    os.makedirs(Config.CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)

    # Основной цикл обучения
    for epoch in range(Config.NUM_EPOCHS):
        netG.train()
        netD.train()

        total_g_loss = 0
        total_d_loss = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.NUM_EPOCHS}")

        for i, (real_sar, real_optical) in enumerate(progress_bar):
            real_sar = real_sar.to(device)
            real_optical = real_optical.to(device)

            # --------- Обновление дискриминатора ---------
            # train_discriminator = (epoch % 2 == 0)
            # if train_discriminator:
            netD.requires_grad_(True)
            optimizer_D.zero_grad()

            # Генерируем фейковое изображение
            with torch.no_grad():
                fake_optical = netG(real_sar)

            # Конкатенируем SAR + Optical
            fake_pair = torch.cat((real_sar, fake_optical), dim=1)
            real_pair = torch.cat((real_sar, real_optical), dim=1)

            pred_fake = netD(fake_pair)
            pred_real = netD(real_pair)

            # Считаем Loss дискриминатора
            d_loss_fake = sum(criterionGAN(fake, False) for fake in pred_fake)
            d_loss_real = sum(criterionGAN(real, True, real_label_smooth=0.9) for real in pred_real)

            d_loss = (d_loss_fake + d_loss_real) * 0.5
            d_loss.backward()
            optimizer_D.step()

            # --------- Обновление генератора ---------
            netD.requires_grad_(False)
            optimizer_G.zero_grad()

            # Снова прогоняем (чтобы получить свежие данные после обновления дискриминатора)
            fake_optical = netG(real_sar)
            fake_pair = torch.cat((real_sar, fake_optical), dim=1)
            pred_fake = netD(fake_pair)

            g_gan_loss = sum(criterionGAN(fake, True) for fake in pred_fake)                                    # GAN Loss генератора
            l1_loss = criterionL1(fake_optical, real_optical)                                                   # L1 Loss
            fm_loss = sum(criterionFM([fake], [real.detach()]) for fake, real in zip(pred_fake, pred_real))     # Feature Matching Loss
            perceptual_loss = criterionPerceptual(fake_optical, real_optical.detach())                          # Perceptual Loss
            tv_loss = total_variation_loss(fake_optical)                                                        # Total Variation Loss
            lpips_loss = criterionLPIPS(fake_optical, real_optical.detach())                                    # LPIPS Loss
            lab_l, lab_ab = criterionLab(fake_optical, real_optical)                                            # LabColor (раздельно L и ab)
            g_ssim = criterionSSIM(fake_optical, real_optical)                                                  # SSIM (DSSIM)
            edge_loss = criterionEdge(fake_optical, real_optical)                                               # Edge Loss

            # Общий Loss генератора
            g_loss = g_gan_loss * Config.GAN_LOSS_WEIGHT + \
                    l1_loss * Config.L1_LOSS_WEIGHT + \
                    fm_loss * Config.FM_LOSS_WEIGHT + \
                    perceptual_loss * Config.PERCEPTUAL_LOSS_WEIGHT + \
                    tv_loss * Config.TV_LOSS_WEIGHT + \
                    lpips_loss * Config.LPIPS_LOSS_WEIGHT + \
                    lab_l * Config.LAB_L_LOSS_WEIGHT + \
                    lab_ab * Config.LAB_AB_LOSS_WEIGHT + \
                    g_ssim * Config.SSIM_LOSS_WEIGHT + \
                    edge_loss + Config.EDGE_LOSS_WEIGHT

            g_loss.backward()
            optimizer_G.step()

            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()

            progress_bar.set_postfix({
                "G_loss": f"{g_loss.item():.4f}",
                "D_loss": f"{d_loss.item():.4f}"
            })

        # Сохранение чекпоинта
        if (epoch + 1) % 10 == 0:
            save_checkpoint(netG, optimizer_G, epoch, os.path.join(Config.CHECKPOINTS_DIR, f"netG_epoch_{epoch+1}.pth"))
            save_checkpoint(netD, optimizer_D, epoch, os.path.join(Config.CHECKPOINTS_DIR, f"netD_epoch_{epoch+1}.pth"))

        # Запись в TensorBoard
        writer.add_scalar('Loss/Generator', total_g_loss / len(train_loader), epoch)
        writer.add_scalar('Loss/Discriminator', total_d_loss / len(train_loader), epoch)
        writer.add_scalar('Loss/L1', l1_loss.item(), epoch)
        writer.add_scalar('Loss/FeatureMatching', fm_loss.item(), epoch)
        writer.add_scalar('Loss/Perceptual', perceptual_loss.item(), epoch)
        writer.add_scalar('Loss/LPIPS', lpips_loss.item(), epoch)
        writer.add_scalar('Loss/TotalVariation', tv_loss.item(), epoch)
        writer.add_scalar('Loss/GAN', g_gan_loss.item(), epoch)
        writer.add_scalar('Loss/Lab_L', lab_l.item(), epoch)
        writer.add_scalar('Loss/Lab_ab', lab_ab.item(), epoch)
        writer.add_scalar('Loss/SSIM', g_ssim.item(), epoch)
        writer.add_scalar('Loss/TV', tv_loss.item(), epoch)
        writer.close()

        # Логируем потери
        losses_logs["epoch"].append(epoch)
        losses_logs["G_loss"].append(total_g_loss / len(train_loader))
        losses_logs["D_loss"].append(total_d_loss / len(train_loader))
        losses_logs["L1"].append(l1_loss.item())
        losses_logs["FeatureMatching"].append(fm_loss.item())
        losses_logs["Perceptual"].append(perceptual_loss.item())
        losses_logs["LPIPS"].append(lpips_loss.item())
        losses_logs["TotalVariation"].append(tv_loss.item())
        losses_logs["GAN"].append(g_gan_loss.item())

        df = pd.DataFrame(losses_logs)
        df.to_csv(os.path.join(Config.RESULTS_DIR, 'losses_logs.csv'), index=False)

        os.makedirs(f'{Config.RESULTS_DIR}/train', exist_ok=True)
        # Сохраняем одну сгенерированную картинку каждые 5 эпох
        if (epoch + 1) % 5 == 0:
            netG.eval()
            with torch.no_grad():
                real_sar, real_optical = next(iter(train_loader))
                real_sar = real_sar.to(device)
                fake_optical = netG(real_sar)

                save_image((fake_optical + 1) / 2.0, os.path.join(f'{Config.RESULTS_DIR}/train', f"epoch_{epoch+1}_fake.png"))
                save_image((real_optical + 1) / 2.0, os.path.join(f'{Config.RESULTS_DIR}/train', f"epoch_{epoch+1}_real.png"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default=None,
                        help='Имя текущего прогона (для TensorBoard)')
    args = parser.parse_args()
    train(run_name=args.run_name)
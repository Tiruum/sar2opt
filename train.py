# train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from models.generator import GlobalGenerator
from models.multiscale_discriminator import MultiscaleDiscriminator
from models.losses import GANLoss, L1Loss, FeatureMatchingLoss, PerceptualLoss

from utils.Dataset import train_loader
from utils.Config import Config

def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)

def train():
    writer = SummaryWriter(log_dir=os.path.join(Config.RESULTS_DIR, 'logs'))
    device = torch.device(Config.DEVICE)

    # Инициализация моделей
    netG = GlobalGenerator(
        input_nc=Config.INPUT_NC,
        output_nc=Config.OUTPUT_NC,
        ngf=Config.NGF,
        n_blocks=9
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

    # Оптимизаторы
    optimizer_G = optim.Adam(netG.parameters(), lr=Config.LEARNING_RATE, betas=(Config.BETA1, Config.BETA2))
    optimizer_D = optim.Adam(netD.parameters(), lr=Config.LEARNING_RATE, betas=(Config.BETA1, Config.BETA2))

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
            optimizer_D.zero_grad()

            # Генерируем фейковое изображение
            fake_optical = netG(real_sar)

            # Конкатенируем SAR + Optical
            fake_pair = torch.cat((real_sar, fake_optical), dim=1)
            real_pair = torch.cat((real_sar, real_optical), dim=1)

            pred_fake = netD(fake_pair)
            pred_real = netD(real_pair)

            # Считаем Loss дискриминатора
            d_loss_fake = 0
            d_loss_real = 0

            for fake_out, real_out in zip(pred_fake, pred_real):
                d_loss_fake += criterionGAN(fake_out, target_is_real=False)
                d_loss_real += criterionGAN(real_out, target_is_real=True)

            d_loss = (d_loss_fake + d_loss_real) * 0.5
            d_loss.backward()
            optimizer_D.step()

            # --------- Обновление генератора ---------
            optimizer_G.zero_grad()

            # Снова прогоняем (чтобы получить свежие данные после обновления дискриминатора)
            fake_optical = netG(real_sar)
            fake_pair = torch.cat((real_sar, fake_optical), dim=1)
            pred_fake = netD(fake_pair)

            # GAN Loss генератора
            g_gan_loss = 0
            for fake_out in pred_fake:
                g_gan_loss += criterionGAN(fake_out, target_is_real=True)

            # L1 Loss
            l1_loss = criterionL1(fake_optical, real_optical)

            # Feature Matching Loss
            fm_loss = 0
            for fake_out, real_out in zip(pred_fake, pred_real):
                fm_loss += criterionFM([fake_out], [real_out])

            # Perceptual Loss
            perceptual_loss = criterionPerceptual(fake_optical, real_optical)

            # Общий Loss генератора
            g_loss = g_gan_loss + l1_loss * 10.0 + fm_loss * 10.0 + perceptual_loss * 1.0

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

if __name__ == "__main__":
    train()
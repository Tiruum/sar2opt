from typing import Literal
import torch.optim as optim
from models.generator import UNetGenerator
from models.multiscale_discriminator import MultiscaleDiscriminator
from models.losses import (
    EdgeLoss, GANLoss, L1Loss, FeatureMatchingLoss,
    PerceptualLoss, LPIPSLoss,
    LabColorLoss, SSIMLoss, TVLoss
)
import torch.nn as nn
from .Config import Config


def build_models(device):
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
        num_D=4
    ).to(device)
    return netG, netD

def build_optimizers(netG, netD):
    optG = optim.Adam(netG.parameters(), lr=Config.LEARNING_RATE_G, betas=(Config.BETA1, Config.BETA2))
    optD = optim.Adam(netD.parameters(), lr=Config.LEARNING_RATE_D, betas=(Config.BETA1, Config.BETA2))
    return optG, optD

def build_criterions(device) -> dict[Literal['GAN', 'L1', 'FM', 'Perceptual', 'LPIPS', 'Lab', 'SSIM', 'Edge', 'TV'], nn.Module]:
    crits = {}
    crits['GAN'] = GANLoss(use_lsgan=True).to(device)
    crits['L1'] = L1Loss().to(device)
    crits['FM'] = FeatureMatchingLoss().to(device)
    crits['Perceptual'] = PerceptualLoss().to(device)
    crits['LPIPS'] = LPIPSLoss().to(device)
    crits['Lab'] = LabColorLoss().to(device)
    crits['SSIM'] = SSIMLoss().to(device)
    crits['Edge'] = EdgeLoss().to(device)
    crits['TV'] = TVLoss().to(device)
    return crits

def build_lr_schedulers(optG, optD):
    scheduler_G = optim.lr_scheduler.CosineAnnealingLR(optG, T_max=Config.NUM_EPOCHS, eta_min=1e-6)
    scheduler_D = optim.lr_scheduler.CosineAnnealingLR(optD, T_max=Config.NUM_EPOCHS, eta_min=1e-6)
    return scheduler_G, scheduler_D
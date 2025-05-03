# models/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from utils.Config import Config
import lpips
from kornia.filters import spatial_gradient
import kornia.color as Kc
from kornia.losses import ssim_loss

class GANLoss(nn.Module):
    """
    GAN Loss для генератора и дискриминатора
    с поддержкой сглаживания как реальных, так и фейковых меток.
    """
    def __init__(self, use_lsgan=True):
        """
        Args:
            use_lsgan (bool): если True — используем MSELoss (LSGAN),
                              иначе BCEWithLogitsLoss (обычный GAN).
        """
        super(GANLoss, self).__init__()
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCEWithLogitsLoss()

    def get_target_tensor(self, prediction, target_is_real, real_label_smooth=1.0, fake_label_smooth=0.0):
        """
        Генерация целевых тензоров для лосса.
        
        Args:
            prediction: выход дискриминатора
            target_is_real (bool): True — реальные данные, False — фейковые
            real_label_smooth (float): значение метки для реальных данных (по умолчанию 1.0)
            fake_label_smooth (float): значение метки для фейковых данных (по умолчанию 0.0)
        """
        if target_is_real:
            return torch.full_like(prediction, real_label_smooth)
        else:
            return torch.full_like(prediction, fake_label_smooth)

    def forward(self, prediction, target_is_real, real_label_smooth=1.0, fake_label_smooth=0.0):
        """
        Вычисление лосса.
        
        Args:
            prediction: выход дискриминатора
            target_is_real (bool): реальные или фейковые данные
            real_label_smooth (float): сглаживание для реальных
            fake_label_smooth (float): сглаживание для фейковых
        """
        target_tensor = self.get_target_tensor(
            prediction, 
            target_is_real, 
            real_label_smooth=real_label_smooth, 
            fake_label_smooth=fake_label_smooth
        )
        return self.loss(prediction, target_tensor)

class L1Loss(nn.Module):
    """Pixel-level L1 Loss"""
    def __init__(self):
        super(L1Loss, self).__init__()
        self.loss = nn.L1Loss()

    def forward(self, pred, target):
        return self.loss(pred, target)

class FeatureMatchingLoss(nn.Module):
    """Feature Matching Loss"""
    def __init__(self):
        super(FeatureMatchingLoss, self).__init__()
        self.loss = nn.L1Loss()

    def forward(self, fake_features, real_features):
        """
        fake_features, real_features — списки фичей с каждого дискриминатора
        """
        total_loss = 0
        num_D = len(fake_features)
        for i in range(num_D):
            fake_feat = fake_features[i]
            real_feat = real_features[i]
            for j in range(len(fake_feat)):
                total_loss += self.loss(fake_feat[j], real_feat[j].detach())
        return total_loss / num_D

class PerceptualLoss(nn.Module):
    """Perceptual Loss на базе VGG16"""
    def __init__(self, layers=['relu3_3'], device=Config.DEVICE):
        super(PerceptualLoss, self).__init__()
        self.device = device
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features.to(device).eval()
        
        self.layers = layers
        self.blocks = nn.ModuleList()
        
        # Разбиваем VGG по уровням
        block = nn.Sequential()
        counter = 0
        mapping = {
            'relu1_2': 3,
            'relu2_2': 8,
            'relu3_3': 15,
            'relu4_3': 22
        }
        
        last_layer = max(mapping[layer] for layer in layers)
        
        for i in range(last_layer + 1):
            block.add_module(str(i), vgg[i])
            if i in mapping.values():
                self.blocks.append(block)
                block = nn.Sequential()

        for param in self.blocks.parameters():
            param.requires_grad = False

        self.loss = nn.L1Loss()

    def forward(self, fake_img, real_img):
        loss = 0
        x = fake_img
        y = real_img

        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += self.loss(x, y.detach())

        return loss
    
class LPIPSLoss(nn.Module):
    """LPIPS Perceptual Loss для генератора"""
    def __init__(self, net='vgg', device=Config.DEVICE):
        super(LPIPSLoss, self).__init__()
        # net: 'vgg' or 'alex'
        self.lpips = lpips.LPIPS(net=net).to(device)

    def forward(self, fake, real):
        # expects inputs in [-1,1]
        return self.lpips(fake, real).mean()
    
class LabColorLoss(nn.Module):
    """Loss в Lab-пространстве: L и ab каналы"""
    def __init__(self):
        super(LabColorLoss, self).__init__()

    def forward(self, fake_rgb, real_rgb):
        # Проверяем, что вход 3-канальный
        assert fake_rgb.size(1) == 3 and real_rgb.size(1) == 3, \
            "LabColorLoss требует 3-канальное RGB изображение"
        fake_lab = Kc.rgb_to_lab((fake_rgb + 1) * 0.5)
        real_lab = Kc.rgb_to_lab((real_rgb + 1) * 0.5)
        l_loss  = F.l1_loss(fake_lab[:, :1], real_lab[:, :1])
        ab_loss = F.l1_loss(fake_lab[:, 1:], real_lab[:, 1:])
        return l_loss, ab_loss


class SSIMLoss(nn.Module):
    """SSIM Loss using kornia.losses.ssim_loss (DSSIM)"""
    def __init__(self, window_size=11, max_val=1.0, eps=1e-12, reduction='mean', padding='same'):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.max_val = max_val
        self.eps = eps
        self.reduction = reduction
        self.padding = padding

    def forward(self, fake, real):
        fake01 = (fake + 1) * 0.5
        real01 = (real + 1) * 0.5
        return ssim_loss(fake01, real01,
                         window_size=self.window_size,
                         max_val=self.max_val,
                         eps=self.eps,
                         reduction=self.reduction,
                         padding=self.padding)
    
class EdgeLoss(nn.Module):
    """
    L1 Loss по картам градиентов (sobel/spatial_gradient)
    Регуляризует чёткость контуров.
    """
    def __init__(self, mode: str = 'sobel'):
        """
        mode: 'sobel' или 'scharr' (настраивается в spatial_gradient)
        """
        super(EdgeLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.mode = mode

    def forward(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        # Предположим входной fake/real ∈[-1,1], приводим в [0,1]
        fake01 = (fake + 1) * 0.5
        real01 = (real + 1) * 0.5
        # spatial_gradient возвращает тензор shape=(B,C,2,H,W):
        # два канала — dx и dy
        fake_grad = spatial_gradient(fake01, mode=self.mode)
        real_grad = spatial_gradient(real01, mode=self.mode)
        # считаем L1 по всем каналам и направлениям
        return self.l1(fake_grad, real_grad.detach())

class TVLoss(nn.Module):
    """Total Variation Loss для сглаживания изображений"""
    def __init__(self):
        super(TVLoss, self).__init__()
        
    def forward(self, img):
        """
        Вычисляет Total Variation Loss для изображения.
        Это помогает сглаживать изображения, уменьшая разницу между соседними пикселями.
        
        Args:
            img (torch.Tensor): Входное изображение формата (B, C, H, W)
            
        Returns:
            torch.Tensor: Вычисленный TV Loss
        """
        tv_h = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]))
        tv_w = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]))
        return tv_h + tv_w

class ColorHistogramLoss(nn.Module):
    """Лосс для согласования цветовых гистограмм"""
    def __init__(self, bins=64):
        super(ColorHistogramLoss, self).__init__()
        self.bins = bins
        
    def forward(self, fake, real):
        # Нормализуем к [0,1]
        fake_norm = (fake + 1) / 2.0
        real_norm = (real + 1) / 2.0
        
        loss = 0
        # Для каждого RGB-канала
        for c in range(3):
            fake_hist = torch.histc(fake_norm[:,c], bins=self.bins, min=0, max=1)
            real_hist = torch.histc(real_norm[:,c], bins=self.bins, min=0, max=1)
            
            # Нормализуем гистограммы
            fake_hist = fake_hist / (torch.sum(fake_hist) + 1e-8)
            real_hist = real_hist / (torch.sum(real_hist) + 1e-8)
            
            # Earth Mover's Distance (приближение)
            loss += torch.mean(torch.abs(torch.cumsum(fake_hist, 0) - torch.cumsum(real_hist, 0)))
        
        return loss / 3.0
    
if __name__ == "__main__":
    import torch

    batch_size = 1
    pred_fake = torch.randn((batch_size, 1, 30, 30))
    pred_real = torch.randn((batch_size, 1, 30, 30))

    fake_features = [[torch.randn(batch_size, 64, 30, 30)] for _ in range(3)]
    real_features = [[torch.randn(batch_size, 64, 30, 30)] for _ in range(3)]

    fake_image = torch.randn((batch_size, 3, 600, 600))
    real_image = torch.randn((batch_size, 3, 600, 600))

    gan_loss = GANLoss()
    l1_loss = L1Loss()
    fm_loss = FeatureMatchingLoss()
    perceptual_loss = PerceptualLoss(layers=['relu3_3'], device='cpu')
    lpips_loss = LPIPSLoss(net='vgg', device='cpu')
    print(f"GAN Loss Fake: {gan_loss(pred_fake, target_is_real=False).item()}")
    print(f"GAN Loss Real: {gan_loss(pred_real, target_is_real=True).item()}")
    print(f"L1 Pixel Loss: {l1_loss(fake_image, real_image).item()}")
    print(f"Feature Matching Loss: {fm_loss(fake_features, real_features).item()}")
    print(f"Perceptual Loss: {perceptual_loss(fake_image, real_image).item()}")
    print(f"LPIPS Loss: {lpips_loss(fake_image, real_image).item()}")
    print('LabColorLoss (l, ab):', LabColorLoss()(fake_image, real_image))
    print('SSIMLoss:', SSIMLoss()(fake_image, real_image).item())
    print("Edge:", EdgeLoss()(fake_image, real_image).item())
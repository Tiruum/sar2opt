# models/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from utils.Config import Config
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
        vgg = models.vgg16(pretrained=True).features.to(device).eval()
        
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

    print(f"GAN Loss Fake: {gan_loss(pred_fake, target_is_real=False).item()}")
    print(f"GAN Loss Real: {gan_loss(pred_real, target_is_real=True).item()}")
    print(f"L1 Pixel Loss: {l1_loss(fake_image, real_image).item()}")
    print(f"Feature Matching Loss: {fm_loss(fake_features, real_features).item()}")
    print(f"Perceptual Loss: {perceptual_loss(fake_image, real_image).item()}")
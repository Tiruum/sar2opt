import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from utils.helpers import normalize_input

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        weights = MobileNet_V2_Weights.IMAGENET1K_V1
        mobilenet = mobilenet_v2(weights=weights).features[:4]
        for param in mobilenet.parameters():
            param.requires_grad = False
        self.mobilenet = mobilenet.eval()

    def forward(self, input, target):
        # Нормализуем входы для MobileNet
        input = normalize_input(input)
        target = normalize_input(target)

        input_features = self.mobilenet(input)
        target_features = self.mobilenet(target)

        return torch.nn.functional.mse_loss(input_features, target_features)



from torchmetrics.functional import structural_similarity_index_measure as ssim

def ssim_loss(input, target):
    # Нормализуем входы в диапазон [0, 1]
    input = normalize_input(input)
    target = normalize_input(target)
    return 1 - ssim(input, target)

def total_variation_loss(img, epsilon=1e-6):
    diff_h = torch.abs(img[:, :, :-1] - img[:, :, 1:]) + epsilon
    diff_w = torch.abs(img[:, :-1, :] - img[:, 1:, :]) + epsilon
    return torch.sum(diff_h) + torch.sum(diff_w)

class CombinedLoss(nn.Module):
    def __init__(self, perceptual_weight=0.1, ssim_weight=0.1, mse_weight=1.0, tv_weight=0.01):
        super(CombinedLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.perceptual_loss = PerceptualLoss()
        self.ssim_weight = ssim_weight
        self.tv_weight = tv_weight
        self.perceptual_weight = perceptual_weight
        self.mse_weight = mse_weight

    def forward(self, output, target):
        mse = self.mse_loss(output, target)
        perceptual = self.perceptual_loss(output, target)
        ssim_l = ssim_loss(output, target)
        tv = total_variation_loss(output)

        # Проверка на NaN
        if torch.isnan(mse) or torch.isnan(perceptual) or torch.isnan(ssim_l) or torch.isnan(tv):
            print(f"MSE: {mse}, Perceptual: {perceptual}, SSIM: {ssim_l}, TV: {tv}")
            raise ValueError("NaN detected in loss components")

        return (
            self.mse_weight * mse +
            self.perceptual_weight * perceptual +
            self.ssim_weight * ssim_l +
            self.tv_weight * tv
        )
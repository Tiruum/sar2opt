import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        weights = MobileNet_V2_Weights.IMAGENET1K_V1  # Предобученные веса VGG16 на ImageNet
        vgg = mobilenet_v2(weights=weights).features[:4]  # Берём первые 16 слоёв
        for param in vgg.parameters():
            param.requires_grad = False  # Замораживаем веса
        self.vgg = vgg.eval()  # Устанавливаем режим оценки

    def forward(self, input, target):
        input_features = self.vgg(input)
        target_features = self.vgg(target)
        return torch.nn.functional.mse_loss(input_features, target_features)


from torchmetrics.functional import structural_similarity_index_measure as ssim

def ssim_loss(input, target):
    return 1 - ssim(input, target)

def total_variation_loss(img):
    return torch.sum(torch.abs(img[:, :, :-1] - img[:, :, 1:])) + \
           torch.sum(torch.abs(img[:, :-1, :] - img[:, 1:, :]))

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

        return self.mse_weight * mse + \
               self.perceptual_weight * perceptual + \
               self.ssim_weight * ssim_l + \
               self.tv_weight * tv

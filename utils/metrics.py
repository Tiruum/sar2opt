# metrics.py

import os
import argparse
import torch
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from pytorch_fid import fid_score
import lpips

def get_all_images(folder, image_size):
    """Загружает все изображения из папки"""
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor()
    ])
    dataset = ImageFolder(folder, transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    return loader

def calculate_lpips(loader_real, loader_fake, device):
    loss_fn = lpips.LPIPS(net='vgg').to(device)
    lpips_score = 0.0
    n = min(len(loader_real), len(loader_fake))
    
    real_iter = iter(loader_real)
    fake_iter = iter(loader_fake)

    for _ in range(n):
        real_img, _ = next(real_iter)
        fake_img, _ = next(fake_iter)

        real_img = real_img.to(device)
        fake_img = fake_img.to(device)

        lpips_score += loss_fn(real_img, fake_img).item()

    return lpips_score / n

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # FID
    fid_value = fid_score.calculate_fid_given_paths(
        [args.real_dir, args.fake_dir],
        batch_size=50,
        device=device,
        dims=2048
    )

    print(f"FID Score: {fid_value:.4f}")

    # LPIPS
    real_loader = get_all_images(args.real_dir, args.image_size)
    fake_loader = get_all_images(args.fake_dir, args.image_size)

    lpips_value = calculate_lpips(real_loader, fake_loader, device)
    print(f"LPIPS Score: {lpips_value:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--real_dir', type=str, required=True, help='Path to real images folder')
    parser.add_argument('--fake_dir', type=str, required=True, help='Path to fake images folder')
    parser.add_argument('--image_size', type=int, default=256, help='Image size (resize for LPIPS)')
    args = parser.parse_args()

    main(args)
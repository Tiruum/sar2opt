# config.py
import os

class Config:
    # Параметры обучения
    IMAGE_SIZE = 256
    BATCH_SIZE = 8
    NUM_EPOCHS = 300
    LEARNING_RATE_G = 1e-4
    LEARNING_RATE_D = 2e-4
    BETA1 = 0.5  # для Adam оптимизатора
    BETA2 = 0.999

    # Пути
    DATA_DIR = './dataset'
    CHECKPOINTS_DIR = './checkpoints'
    RESULTS_DIR = './results'

    # Архитектура
    INPUT_NC = 1  # Число каналов у SAR изображения
    OUTPUT_NC = 3  # Число каналов у оптического изображения
    NGF = 64  # Количество фичей в первом слое генератора
    NDF = 64  # Количество фичей в первом слое дискриминатора

    # Датасет
    NUM_WORKERS = 10 #  os.cpu_count() // 2,
    PERSISTENT_WORKERS = True,
    PREFETCH_FACTOR = 2 # 2

    GAN_LOSS_WEIGHT        = 2.0
    L1_LOSS_WEIGHT         = 80.0
    FM_LOSS_WEIGHT         = 3.0
    PERCEPTUAL_LOSS_WEIGHT = 0.7
    LPIPS_LOSS_WEIGHT      = 0.5
    TV_LOSS_WEIGHT         = 0.5
    LAB_L_LOSS_WEIGHT      = 0.5
    LAB_AB_LOSS_WEIGHT     = 0.5
    SSIM_LOSS_WEIGHT       = 1.0
    EDGE_LOSS_WEIGHT       = 1.0

    DEVICE = 'cuda'

config = Config()
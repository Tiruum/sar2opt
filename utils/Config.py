# config.py
import os

class Config:
    # Параметры обучения
    IMAGE_SIZE = 256
    BATCH_SIZE = 8
    NUM_EPOCHS = 100
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
    NDF = 32  # Количество фичей в первом слое дискриминатора

    # Датасет
    NUM_WORKERS = 10 #  os.cpu_count() // 2,
    PERSISTENT_WORKERS = True,
    PREFETCH_FACTOR = 2 # 2

    # Losses weights
    GAN_LOSS_WEIGHT = 1
    L1_LOSS_WEIGHT = 80.0
    FM_LOSS_WEIGHT = 10.0
    PERCEPTUAL_LOSS_WEIGHT = 1.0
    TV_LOSS_WEIGHT = 0.1
    LPIPS_LOSS_WEIGHT = 0.1
    # Прочее
    DEVICE = 'cuda'  # 'cuda' или 'cpu'

config = Config()
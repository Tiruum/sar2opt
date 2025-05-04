# config.py
import os

class Config:
    # Параметры обучения
    IMAGE_SIZE = 256
    BATCH_SIZE = 8
    NUM_EPOCHS = 500
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

    GAN_LOSS_WEIGHT        = 1.0    # оставить  
    L1_LOSS_WEIGHT         = 10.0   # ↓ сильно, чтобы убрать размытие  
    FM_LOSS_WEIGHT         = 5.0    # ↑ чуть, для текстур  
    PERCEPTUAL_LOSS_WEIGHT = 3.0    # оставить/немного ↓, локальные фичи  
    LPIPS_LOSS_WEIGHT      = 8.0    # ↑ для глобальных патчей и текстур  
    SSIM_LOSS_WEIGHT       = 12.0   # ↑ сильно, структурная точность  
    EDGE_LOSS_WEIGHT       = 30.0   # ↑, четкость контуров  
    TV_LOSS_WEIGHT         = 1.0    # ↑, убрать мелкий шум  
    LAB_L_LOSS_WEIGHT      = 2.0    # ↓, чтобы не «гладило» слишком яркость  
    LAB_AB_LOSS_WEIGHT     = 6.0    # ↑, цвета покрасивее  
    COLOR_HIST_LOSS_WEIGHT = 5.0    # ↑, для лучшей цветопередачи  
    FREQUENCY_LOSS_WEIGHT  = 6.0    # ↓ немного, высокие частоты уже «сильные»  
    
    USE_AMP = False
    CUDNN_BENCHMARK = False
    DEVICE = 'cuda'

config = Config()
# print({k: v for k, v in vars(Config).items() if isinstance(v, (int, float, str, list, dict))})
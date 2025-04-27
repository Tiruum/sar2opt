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
    GAN_LOSS_WEIGHT        = 1.0    # Adversarial Loss (GAN): отвечает за реалистичность текстур и детализацию, но слишком большой вес → артефакты. Обычно λ_GAN ∈ [0.1, 10].
    L1_LOSS_WEIGHT         = 90.0   # L1 Loss: стабилизирует обучение, отвечает за грубую цветовую/яркостную точность, смягчает шум, λ_L1 ∈ [10, 200].
    FM_LOSS_WEIGHT         = 5.0    # Feature Matching: выравнивает фичи разных масштабов, улучшает стабильность, λ_FM ∈ [1, 10].
    PERCEPTUAL_LOSS_WEIGHT = 1.0    # Perceptual Loss (VGG): сохраняет высокоуровневый контент и стиль, избавляет от размытости, λ_perc ∈ [0.001, 1].
    LPIPS_LOSS_WEIGHT      = 0.1    # LPIPS: оптимизирует глубокое восприятие сходства, коррелирует с человеческим восприятием, λ_LPIPS ∈ [0.01, 0.1].
    TV_LOSS_WEIGHT         = 0.1    # Total Variation: сглаживает шум и мелкие артефакты, λ_TV ∈ [1e-8, 0.1].
    LAB_L_LOSS_WEIGHT      = 1.0    # Lab-Color Loss: усиливает цветовую точность по L- и ab-каналам, λ_L ∈ [0.1, 2], λ_ab ∈ [0.1, 2].
    LAB_AB_LOSS_WEIGHT     = 1.0    # Lab-Color Loss: усиливает цветовую точность по L- и ab-каналам, λ_L ∈ [0.1, 2], λ_ab ∈ [0.1, 2].
    SSIM_LOSS_WEIGHT       = 1.0    # SSIM: повышает структурную консистентность и чёткость контуров, λ_SSIM ∈ [0.1, 5].


    # Прочее
    DEVICE = 'cuda' # 'cuda' | 'mps' | 'cpu'

config = Config()
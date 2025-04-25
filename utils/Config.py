# config.py

class Config:
    # Параметры обучения
    IMAGE_SIZE = 600
    BATCH_SIZE = 1
    NUM_EPOCHS = 1
    LEARNING_RATE = 0.0002
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
    NUM_WORKERS = 0 # os.cpu_count() // 2,
    PERSISTENT_WORKERS = False # True,
    PREFETCH_FACTOR = None # 2

    # Прочее
    DEVICE = 'mps'  # 'cuda' или 'cpu'

config = Config()
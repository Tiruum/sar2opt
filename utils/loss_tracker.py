import numpy as np
import os

def save_losses(train_losses, val_losses, log_path):
    """
    Сохраняет потери в формате .npz для последующего анализа.
    
    Parameters:
    - train_losses (list): Список потерь на тренировке.
    - val_losses (list): Список потерь на валидации.
    - log_path (str): Путь к директории, где сохраняются логи.
    """
    os.makedirs(log_path, exist_ok=True)
    file_path = os.path.join(log_path, "losses.npz")
    np.savez(file_path, train_losses=train_losses, val_losses=val_losses)

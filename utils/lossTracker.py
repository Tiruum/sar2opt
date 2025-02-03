import numpy as np
import os

def save_losses(log_path=os.path.join(os.getcwd(), 'logs'), **losses):
    """
    Сохраняет потери в формате .npz для последующего анализа.
    
    Parameters:
    - log_path (str): Путь к директории, где сохраняются логи.
    - losses (dict): Произвольное количество переменных для сохранения.
    """
    os.makedirs(log_path, exist_ok=True)
    file_path = f'{log_path}/losses.npz'
    np.savez(file_path, **losses)

def load_losses(log_path=f'{os.getcwd()}/logs/losses.npz'):
    """
    Загружает потери из файла .npz.
    
    Parameters:
    - log_path (str): Путь к директории, где сохраняются логи.
    
    Returns:
    - losses (dict): Словарь с загруженными переменными.
    """
    if not os.path.isfile(log_path):
        print(f"Файл с потерями не найден по пути: {log_path}")
        return {}
    return dict(np.load(log_path))
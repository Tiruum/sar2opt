import os
import datetime
import torch

def save_checkpoint(epoch, model, checkpoint_path=os.path.join(os.getcwd(), 'checkpoints')):
    """
    Сохраняет контрольную точку состояния модели, включая параметры генератора, дискриминатора, оптимизаторов и шедулеров.

    Аргументы:
        epoch (int): Номер текущей эпохи.
        model (Pix2PixGAN): Модель, содержащая генератор, дискриминатор, оптимизаторы и шедулеры.
        checkpoint_path (str): Путь для сохранения контрольной точки.

    Возвращает:
        None
    """
    checkpoint = {
        'epoch': epoch,
        'generator_state_dict': model.generator.state_dict(),
        'discriminator_state_dict': model.discriminator.state_dict(),
        'optimizer_G_state_dict': model.optimizer_G.state_dict(),
        'optimizer_D_state_dict': model.optimizer_D.state_dict(),
        'scheduler_G_state_dict': model.scheduler_G.state_dict(),
        'scheduler_D_state_dict': model.scheduler_D.state_dict(),
        'date': datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    }
    os.makedirs(checkpoint_path, exist_ok=True)
    checkpoint_file = os.path.join(checkpoint_path, f"checkpoint_epoch_{epoch+1}.pth")
    torch.save(checkpoint, checkpoint_file)
    print(f"Сохранён чекпоинт: {checkpoint_file}")


def load_checkpoint(checkpoint_name, model, device):
    """
    Загружает состояние модели, включая параметры генератора, дискриминатора, оптимизаторов и шедулеров.

    Аргументы:
        checkpoint_name (str): Имя файла чекпоинта (без расширения).
        model (Pix2PixGAN): Модель, содержащая генератор, дискриминатор, оптимизаторы и шедулеры.
        device (torch.device): Устройство для загрузки модели (CPU или GPU).

    Возвращает:
        start_epoch (int): Эпоха, с которой можно продолжить обучение.
    """
    checkpoint_path = os.path.join(os.getcwd(), f'checkpoints/{checkpoint_name}.pth')
    if not os.path.isfile(checkpoint_path):
        print(f"Чекпоинт не найден по пути: {checkpoint_path}")
        print("Начинаем обучение с нуля.")
        return 0

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Восстанавливаем состояния генератора и дискриминатора
    model.generator.load_state_dict(checkpoint['generator_state_dict'])
    model.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

    # Восстанавливаем состояния оптимизаторов
    model.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
    model.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])

    # Восстанавливаем состояния шедулеров
    model.scheduler_G.load_state_dict(checkpoint['scheduler_G_state_dict'])
    model.scheduler_D.load_state_dict(checkpoint['scheduler_D_state_dict'])

    start_epoch = checkpoint.get('epoch', 0)
    date_saved = checkpoint.get('date', "Неизвестно")

    print(f"Чекпоинт успешно загружен: {checkpoint_path}")
    print(f"Дата сохранения: {date_saved}, эпоха {start_epoch+1}")

    return start_epoch

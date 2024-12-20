import os
import datetime
import torch

def save_checkpoint(epoch, model, g_loss, d_loss, checkpoint_path=os.path.join(os.getcwd(), 'checkpoints')):
    '''
    Сохраняет контрольную точку состояния модели.

    Аргументы:
        epoch (int): Номер текущей эпохи.
        model (torch.nn.Module): Модель, содержащая генератор и дискриминатор.
        g_loss (float): Потери генератора на текущей эпохе.
        d_loss (float): Потери дискриминатора на текущей эпохе.
        checkpoint_path (str): Путь, по которому будет сохранена контрольная точка.

    Возвращает:
        None
    Эта функция сохраняет словари состояний генератора, дискриминатора и их 
    соответствующих оптимизаторов, а также номер текущей эпохи, потери генератора 
    и дискриминатора в файл, указанный в checkpoint_path.
    '''

    checkpoint = {
        'epoch': epoch,
        'generator_state_dict': model.generator.state_dict(),
        'discriminator_state_dict': model.discriminator.state_dict(),
        'optimizer_G_state_dict': model.optimizer_G.state_dict(),
        'optimizer_D_state_dict': model.optimizer_D.state_dict(),
        'g_loss': g_loss,
        'd_loss': d_loss,
        'date': datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    }
    os.makedirs(checkpoint_path, exist_ok=True)
    torch.save(checkpoint, f"{checkpoint_path}/checkpoint_epoch_{epoch+1}.pth")

def load_checkpoint(checkpoint_name, model, device):
    """
    Загрузка состояния модели и оптимизаторов из чекпоинта с проверкой существования файла.
    
    Аргументы:
        checkpoint_path (str): Путь к файлу чекпоинта (.pth).
        model (Pix2PixGAN): Экземпляр класса, содержащий generator, discriminator, optimizers.
        device (torch.device): Устройство (cpu или cuda), на котором будут храниться модели.
    
    Возвращает:
        start_epoch (int): Эпоха, с которой можно продолжить обучение.
        g_loss (list): Список сохранённых значений G Loss.
        d_loss (list): Список сохранённых значений D Loss.
        hyperparams (dict): Сохранённые гиперпараметры.
    """
    checkpoint_path = os.path.join(os.getcwd(), f'checkpoints/{checkpoint_name}.pth')
    if not os.path.isfile(checkpoint_path):
        print(f"Чекпоинт не найден по пути: {checkpoint_path}")
        print("Начинаем обучение с нуля.")
        return 0, [], [], {}
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    # Восстанавливаем веса генератора и дискриминатора
    model.generator.load_state_dict(checkpoint['generator_state_dict'])
    model.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    
    # Восстанавливаем состояния оптимизаторов
    model.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
    model.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
    
    start_epoch = checkpoint.get('epoch', 0)
    loss_G = checkpoint.get('loss_G', [])
    loss_D = checkpoint.get('loss_D', [])
    date_saved = checkpoint.get('date', "Неизвестно")

    print(f"Чекпоинт успешно загружен: {checkpoint_path}")
    print(f"Дата сохранения: {date_saved}, эпоха {start_epoch+1}") # Хотя в tqdm 400, тут 399 из-за +1, тут все норм
    
    return start_epoch, loss_G, loss_D
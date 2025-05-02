import pandas as pd
import matplotlib.pyplot as plt
from utils.Config import Config

df = pd.read_csv(f'tensorboard_losses.csv')

# Список всех потерь, которые нужно отобразить
losses = ['Loss/Generator', 'Loss/Discriminator', 'Loss/L1', 'Loss/FeatureMatching', 'Loss/Perceptual', 'Loss/LPIPS', 'Loss/TotalVariation', 'Loss/GAN', 'Loss/Lab_L', 'Loss/Lab_ab', 'Loss/SSIM', 'Loss/TV']

# Рассчитываем количество строк для 3 столбцов
n_cols = 3
n_rows = (len(losses) + n_cols - 1) // n_cols  # Округление вверх

# Создаем подграфики в сетке 3 столбца
fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(18, 15), sharex=True)
fig.suptitle('Losses during training', fontsize=16)
axes = axes.flatten()  # Преобразуем двумерный массив axes в одномерный для удобства

# Отображаем каждую потерю на отдельном графике
for i, loss_name in enumerate(losses):
    df.plot(y=loss_name, ax=axes[i], legend=False, color='blue')
    axes[i].set_title(f'{loss_name}')
    axes[i].set_ylabel('Loss value')
    axes[i].set_xlabel('Epoch')
    axes[i].grid(True)

# Скрываем пустые подграфики если есть
for i in range(len(losses), n_rows * n_cols):
    axes[i].set_visible(False)

# Настраиваем общий макет
plt.tight_layout()
plt.subplots_adjust(top=0.92)  # Освобождаем место для общего заголовка
plt.show()


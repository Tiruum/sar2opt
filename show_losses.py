import pandas as pd
import matplotlib.pyplot as plt
from utils.Config import Config
from collections import defaultdict

df = pd.read_csv(f'tensorboard_losses.csv')

# Получаем все названия лоссов
loss_columns = df.columns[1:]
print(f'Losses: {loss_columns}')

# Группируем лоссы по их базовому имени (без Train/Val префикса)
loss_groups = defaultdict(list)
for col in loss_columns:
    # Разделяем на префикс (Train/Val) и базовое имя
    parts = col.split('/')
    if len(parts) == 2:
        prefix, base_name = parts
        loss_groups[base_name].append(col)

# Создаем список уникальных базовых имен лоссов
unique_losses = list(loss_groups.keys())
print(f'Unique losses: {unique_losses}')

# Рассчитываем количество строк для 3 столбцов
n_cols = 3
n_rows = (len(unique_losses) + n_cols - 1) // n_cols  # Округление вверх

# Создаем подграфики в сетке 3 столбца
fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(18, 15), sharex=True)
fig.suptitle('Losses during training', fontsize=16)
axes = axes.flatten()  # Преобразуем двумерный массив axes в одномерный для удобства

# Отображаем каждую группу лоссов на отдельном графике
for i, base_name in enumerate(unique_losses):
    group_losses = loss_groups[base_name]
    
    # Цвета для Train и Val
    colors = ['blue', 'red']
    
    for j, loss_col in enumerate(group_losses):
        prefix = loss_col.split('/')[0]  # Train или Val
        color = colors[0] if prefix == 'Train' else colors[1]
        df.plot(y=loss_col, ax=axes[i], legend=True, color=color, label=prefix)
    
    axes[i].set_title(f'{base_name}')
    axes[i].set_ylabel('Loss value')
    axes[i].set_xlabel('Epoch')
    axes[i].grid(True)
    axes[i].legend()

# Скрываем пустые подграфики если есть
for i in range(len(unique_losses), n_rows * n_cols):
    axes[i].set_visible(False)

# Настраиваем общий макет
plt.tight_layout()
plt.subplots_adjust(top=0.92)  # Освобождаем место для общего заголовка
plt.savefig('losses_plot.png', dpi=300, bbox_inches='tight')
# plt.show()


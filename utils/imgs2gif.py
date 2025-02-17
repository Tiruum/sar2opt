import os
from PIL import Image

# Задаем путь к папке с изображениями
folder_path = '../saved_images'

# Задаем список названий файлов
filenames = [f'epoch_{i}_images.png' for i in range(20, 801, 20)] # 20, 40, 60, 80, 100

# Создаем список изображений
images = []
for filename in filenames:
    img_path = os.path.join(folder_path, filename)
    if os.path.exists(img_path):
        img = Image.open(img_path)
        images.append(img)
    else:
        print(f'Файл {img_path} не найден.')

# Задаем путь и имя для нового GIF-файла
gif_path = os.path.join(folder_path, 'combined_images.gif')

# Создаем и сохраняем GIF-изображение
if images:
    images[0].save(gif_path, save_all=True, append_images=images[1:], loop=0, duration=500)
    print(f'GIF-изображение сохранено по пути: {gif_path}')
else:
    print('Не найдено ни одного изображения для создания GIF.')


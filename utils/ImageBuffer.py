import random
import torch

class ImageBuffer:
    def __init__(self, buffer_size=50):
        """
        Буфер для хранения ранее сгенерированных изображений.
        """
        self.buffer_size = buffer_size
        self.buffer = []

    def push_and_pop(self, images):
        """
        Добавляет новые изображения в буфер и возвращает старые или новые изображения.
        """
        return_images = []
        for img in images:
            img = img.unsqueeze(0)  # Добавляем измерение batch
            if len(self.buffer) < self.buffer_size:
                # Если буфер не заполнен, добавляем изображение и возвращаем его
                self.buffer.append(img)
                return_images.append(img)
            else:
                # С вероятностью 50% возвращаем старое изображение
                if random.uniform(0, 1) > 0.5:
                    idx = random.randint(0, len(self.buffer) - 1)
                    return_images.append(self.buffer[idx])
                    self.buffer[idx] = img
                else:
                    # Или добавляем новое изображение
                    return_images.append(img)
        return torch.cat(return_images)  # Возвращаем батч
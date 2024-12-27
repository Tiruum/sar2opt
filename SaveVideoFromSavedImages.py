import os
from moviepy import *

image_files = []
for i in range(10, 401, 10):
    filename = f"saved_images/epoch_{i}_images.png"
    if os.path.exists(filename):
        image_files.append(filename)
clip = ImageSequenceClip(image_files, fps=2)  # 2 кадра в секунду
clip.write_videofile("sequence.mp4")

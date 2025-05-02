from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--run_name', type=str, default=None, help='Имя текущего прогона (для TensorBoard)')
args = parser.parse_args()

# путь до логов TensorBoard (внутри папки logs/run_xxx)
event_path = f"results/logs/{args.run_name}"  # подставь свою

# загружаем эвенты
ea = event_accumulator.EventAccumulator(event_path)
ea.Reload()

# выводим список доступных тэгов
print("Доступные теги:", ea.Tags()["scalars"])

# собираем все скаляры
df = pd.DataFrame()

for tag in ea.Tags()["scalars"]:
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    df[tag] = pd.Series(values, index=steps)

# сохраняем всё в один CSV
df.to_csv("tensorboard_losses.csv")

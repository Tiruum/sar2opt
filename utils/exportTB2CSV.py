import os
import numpy as np
import pandas as pd
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def tabulate_events(dpath):
    summary_iterators = [
        EventAccumulator(os.path.join(dpath, dname)).Reload() 
        for dname in os.listdir(dpath)
    ]
    tags = summary_iterators[0].Tags()['scalars']

    # Убеждаемся, что все итераторы имеют одинаковые теги
    for it in summary_iterators:
        assert it.Tags()['scalars'] == tags

    out = defaultdict(list)
    steps = []

    for tag in tags:
        steps = [e.step for e in summary_iterators[0].Scalars(tag)]
        for events in zip(*[acc.Scalars(tag) for acc in summary_iterators]):
            # Убеждаемся, что все события соответствуют одному шагу
            assert len(set(e.step for e in events)) == 1
            out[tag].append([e.value for e in events])
    return out, steps

def save_merged_csv(dpath):
    # Получаем данные (словарь тегов с их значениями) и список шагов (эпох)
    data, steps = tabulate_events(dpath)
    
    # Собираем объединённый DataFrame:
    # для каждого тега (метрики) выбираем нужный столбец значений
    merged_data = {}
    for tag, values in data.items():
        arr = np.array(values)  # размерность: (число_эпох, число_запусков)
        # Если по метрике значений более одного, используем второй столбец, иначе первый
        col_index = 1 if arr.shape[1] > 1 else 0
        # Заменяем "/" на "_" в имени тега для удобства
        series = pd.Series(arr[:, col_index], index=steps, name=tag.replace("/", "_"))
        merged_data[tag.replace("/", "_")] = series

    merged_df = pd.DataFrame(merged_data)
    merged_df.index.name = 'epoch'
    
    # Сохраняем объединённый DataFrame в файл merged.csv непосредственно в директории dpath
    merged_file = os.path.join(dpath, f'{dpath.split("/")[-1]}.csv')
    merged_df.to_csv(merged_file, index_label='epoch')
    print("Сохранен файл:", merged_file)

if __name__ == '__main__':
    path = "./runs/ID-2"
    save_merged_csv(path)

import json
import os

class ConfigLoader:
    """
    Класс для загрузки конфигурации из JSON-файла и предоставления параметров по запросу.
    """
    def __init__(self, config_path=os.path.join(os.getcwd(), "config.json")):
        """
        Инициализация загрузчика конфигурации.
        
        :param config_path: Путь к JSON-файлу с конфигурацией.
        """
        self.config_path = config_path
        self._config = None

        # Загружаем конфигурацию при инициализации
        self._load_config()

    def _load_config(self):
        """
        Загружает конфигурацию из JSON-файла.
        """
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Конфигурационный файл не найден: {self.config_path}")
        with open(self.config_path, "r") as f:
            self._config = json.load(f)

    def get(self, *keys, default=None):
        """
        Получить параметр по ключам, с возможностью указать значение по умолчанию.
        
        :param keys: Последовательность ключей для доступа к параметру.
        :param default: Значение по умолчанию, если ключ не найден.
        :return: Значение параметра или default, если ключ не найден.
        """
        result = self._config
        for key in keys:
            if isinstance(result, dict) and key in result:
                result = result[key]
            else:
                return default
        return result

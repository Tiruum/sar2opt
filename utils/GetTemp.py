import pynvml

def get_gpu_temperature():
    try:
        pynvml.nvmlInit()  # Инициализация NVML
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # GPU 0
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        pynvml.nvmlShutdown()  # Завершаем работу NVML
        return temp
    except Exception as e:
        return None
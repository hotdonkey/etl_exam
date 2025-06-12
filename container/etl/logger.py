# logger.py

import logging
import os

def setup_logger(log_file='../logs/app.log'):
    # Создаем папку logs, если её нет
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Формат записи лога
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Настройка логгера
    logger = logging.getLogger("etl_pipeline")
    logger.setLevel(logging.INFO)

    # Обработчик для файла
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Обработчик для консоли (опционально)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger
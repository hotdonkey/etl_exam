import os
import yaml
import requests
from logger import setup_logger

# Инициализируем логгер
logger = setup_logger('../logs/etl_pipeline.log')

def download_data(url, output_path):
    """Загружает данные по URL и сохраняет в файл"""
    logger.info("Начало загрузки данных...")
    try:
        response = requests.get(url)
        with open(output_path, 'wb') as f:
            f.write(response.content)
        logger.info(f"Данные успешно загружены в {output_path}")
    except Exception as e:
        logger.error(f"Ошибка при загрузке данных: {e}")
        raise


if __name__ == "__main__":
    with open(os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml'), 'r') as f:
        config = yaml.safe_load(f)
    
    DATA_URL = config['DATA_URL']
    RAW_DATA_PATH = config['RAW_DATA_PATH']

    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
    download_data(DATA_URL, RAW_DATA_PATH)


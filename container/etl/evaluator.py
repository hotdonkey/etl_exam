import os
import yaml
import json
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from logger import setup_logger
import pickle

# Инициализируем логгер
logger = setup_logger('../logs/etl_pipeline.log')

def load_data(test_data_path):
    """Загружает тестовые данные"""
    logger.info(f"Загрузка тестовых данных из {test_data_path}...")
    X_test = pd.read_csv(os.path.join(test_data_path, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(test_data_path, "y_test.csv")).values.ravel()
    logger.info(f"Данные загружены. X_test: {X_test.shape}, y_test: {y_test.shape}")
    return X_test, y_test

def load_model(model_path):
    """Загружает обученную модель из файла"""
    logger.info(f"Загрузка модели из {model_path}...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    logger.info("Модель успешно загружена.")
    return model

def evaluate_model(y_true, y_pred):
    """Вычисляет основные метрики качества модели"""
    logger.info("Расчёт метрик качества...")

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=1),
        "recall": recall_score(y_true, y_pred, pos_label=1),
        "f1": f1_score(y_true, y_pred, pos_label=1)
    }

    logger.info(f"Метрики: {metrics}")
    return metrics

def save_results(metrics, output_path):
    """Сохраняет результаты оценки в JSON"""
    logger.info(f"Сохранение результатов в {output_path}...")

    os.makedirs(os.path.dirname(f"{output_path}"), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump({k: round(v, 4) for k, v in metrics.items()}, f, indent=4)

    logger.info("Результаты успешно сохранены.")

if __name__ == "__main__":
    logger.info("=== Запуск скрипта evaluator.py ===")

    # Загрузка конфига
    with open(os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml'), 'r') as f:
        config = yaml.safe_load(f)

    TEST_DATA_PATH = config['OUTPUT_DIR']
    MODEL_PATH = config['MODEL_OUTPUT_PATH']
    METRICS_OUTPUT_PATH = config['METRICS_OUTPUT_PATH']

    # Загрузка данных
    X_test, y_test = load_data(TEST_DATA_PATH)

    # Загрузка модели
    model = load_model(f"{MODEL_PATH}logistic_regression.pkl")

    # Предсказание
    logger.info("Предсказание на тестовой выборке...")
    y_pred = model.predict(X_test)

    # Оценка
    metrics = evaluate_model(y_true=y_test, y_pred=y_pred)

    # Сохранение результатов
    save_results(metrics, f"{METRICS_OUTPUT_PATH}metrics.json")

    logger.info("=== Скрипт evaluator.py успешно выполнен ===")
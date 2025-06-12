import os
import yaml
import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from logger import setup_logger

import warnings

# Инициализируем логгер
logger = setup_logger('../logs/etl_pipeline.log')

def train_test_opener(open_path):
    """Загружает обучающие данные из CSV"""
    logger.info(f"Загрузка данных из {open_path}...")
    X_train = pd.read_csv(f"{open_path}/X_train.csv")
    y_train = pd.read_csv(f"{open_path}/y_train.csv").values.ravel()  # Преобразуем в 1D массив
    logger.info(f"Данные загружены. X_train: {X_train.shape}, y_train: {y_train.shape}")
    return X_train, y_train


def build_pipeline():
    """Создаёт пайплайн с предобработкой и моделью"""
    logger.info("Создание пайплайна...")
    pipeline = Pipeline([
        ('power_transform', PowerTransformer()),
        ('scaler', StandardScaler()),
        ('pca', PCA()),
        ('model', LogisticRegression(random_state=42))
    ])
    return pipeline


def train_model(X_train, y_train):
    """Обучает модель с подбором гиперпараметров"""
    logger.info("Начало обучения модели...")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        param_grid = {
            'pca__n_components': [0.9, 0.95],
            'model__C': [0.01, 0.1, 1, 10, 100],
            'model__penalty': ['l1', 'l2', "elasticnet"]
        }

        grid_search = GridSearchCV(
            build_pipeline(), param_grid, cv=5,
            scoring='accuracy', n_jobs=-1, verbose=1)

        grid_search.fit(X_train, y_train)

        logger.info(f"Обучение завершено. Лучшие параметры: {grid_search.best_params_}")
        logger.info(f"Лучшая точность на трейне: {grid_search.best_score_:.4f}")

        return grid_search


def save_model(model, output_dir="../models/"):
    """Сохраняет обученную модель в формате pickle"""
    logger.info(f"Сохранение модели в {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, "logistic_regression.pkl")
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    logger.info(f"Модель успешно сохранена в {model_path}")


if __name__ == "__main__":
    logger.info("=== Запуск скрипта trainer.py ===")

    # Загрузка конфига
    with open(os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml'), 'r') as f:
        config = yaml.safe_load(f)
        
    OUTPUT_DIR = config['OUTPUT_DIR']
    MODEL_OUTPUT_PATH = config['MODEL_OUTPUT_PATH']

    # Загрузка данных
    X_train, y_train = train_test_opener(OUTPUT_DIR)

    # Обучение модели
    model = train_model(X_train, y_train)

    # Сохранение модели
    save_model(model, MODEL_OUTPUT_PATH)

    logger.info("=== Скрипт trainer.py успешно выполнен ===")
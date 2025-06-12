import os
import pandas as pd
import yaml
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from logger import setup_logger

# Инициализируем логгер
logger = setup_logger('../logs/etl_pipeline.log')

def load_data(file_path):
    """Загружает данные в DataFrame"""
    logger.info(f"Чтение данных из {file_path}...")
    columns = [
        "id", "diagnosis", "radius_mean", "texture_mean",
        "perimeter_mean", "area_mean", "smoothness_mean",
        "compactness_mean", "concavity_mean", "concave points_mean",
        "symmetry_mean", "fractal_dimension_mean", "radius_se", "texture_se",
        "perimeter_se", "area_se", "smoothness_se", "compactness_se", "concavity_se",
        "concave points_se", "symmetry_se", "fractal_dimension_se", "radius_worst",
        "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
        "compactness_worst", "concavity_worst", "concave points_worst",
        "symmetry_worst", "fractal_dimension_worst"
    ]
    df = pd.read_csv(file_path, names=columns, header=None)
    logger.info(f"Файл прочитан. Размер: {df.shape}")
    return df

def preprocess_data(df):
    """Удаляет ID, кодирует целевую переменную"""
    X = df.drop(columns=["id", "diagnosis"])
    y = df["diagnosis"]

    le = LabelEncoder()
    y = le.fit_transform(y)

    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """Разделение на обучающую и тестовую выборки"""
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def save_artifacts(X_train, X_test, y_train, y_test, output_dir):
    """Сохраняет обучающие и тестовые данные в CSV"""
    logger.info(f"Сохранение артефактов в {output_dir}...")

    os.makedirs(output_dir, exist_ok=True)

    # Сохраняем признаки
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)

    # Сохраняем таргет
    pd.DataFrame(y_train, columns=["diagnosis"]).to_csv(
        os.path.join(output_dir, "y_train.csv"), index=False
    )
    pd.DataFrame(y_test, columns=["diagnosis"]).to_csv(
        os.path.join(output_dir, "y_test.csv"), index=False
    )

    logger.info("Артефакты успешно сохранены.")


if __name__ == "__main__":
    
    with open(os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml'), 'r') as f:
        config = yaml.safe_load(f)
    
    RAW_DATA_PATH = config['RAW_DATA_PATH']
    OUTPUT_DIR = config['OUTPUT_DIR']
    
    df = load_data(RAW_DATA_PATH)
    print("Первые строки:")
    print(df.head())
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    print("Размеры выборок:", len(X_train), len(X_test))
    
    save_artifacts(X_train, X_test, y_train, y_test, OUTPUT_DIR)
    
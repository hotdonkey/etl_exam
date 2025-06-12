
import os
import yaml
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

# Путь к корню проекта
with open(os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml'), 'r') as f:
    config = yaml.safe_load(f)
    
PROJECT_DIR =  config['PROJECT_DIR']


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='etl_exam_pipeline',
    default_args=default_args,
    description='ETL pipeline for Breast Cancer Diagnosis with Logistic Regression',
    schedule_interval='@daily',
    start_date=datetime(2025, 5, 11),
    catchup=False,
) as dag:

    # Шаг 1: Загрузка данных
    extractor_task = BashOperator(
        task_id='download_data_task',
        bash_command=f'cd {PROJECT_DIR}/etl/ && python extractor.py'
    )

    # Шаг 2: Предобработка данных
    transformer_task = BashOperator(
        task_id='preprocess_data_task',
        bash_command=f'cd {PROJECT_DIR}/etl/ && python transformer.py'
    )

    # Шаг 3: Обучение модели
    trainer_task = BashOperator(
        task_id='train_model_task',
        bash_command=f'cd {PROJECT_DIR}/etl/ && python trainer.py'
    )

    # Шаг 4: Оценка модели
    evaluator_task = BashOperator(
        task_id='evaluate_model_task',
        bash_command=f'cd {PROJECT_DIR}/etl/ && python evaluator.py'
    )

    # Зависимости
    extractor_task >> transformer_task >> trainer_task >> evaluator_task
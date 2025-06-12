FROM apache/airflow:2.9.0

# Переключаемся на root для установки системных зависимостей
USER root

# Установка необходимых системных библиотек + gosu
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libgomp1 \
        postgresql-client \
        gosu && \  
        apt-get clean && \
        rm -rf /var/lib/apt/lists/*

USER airflow


# Копируем DAG-и скрипты
COPY dags/ /opt/airflow/dags/
COPY etl/ /opt/airflow/etl/
COPY config/ /opt/airflow/config/
COPY logs/ /opt/airflow/logs/
COPY models/ /opt/airflow/models/
COPY results/ /opt/airflow/results/

# Устанавливаем дополнительные Python-зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Проверяем версию Airflow
RUN airflow version
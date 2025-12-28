FROM ghcr.io/mlflow/mlflow:v2.9.2

RUN pip install --no-cache-dir psycopg2-binary

RUN apt-get update && apt-get install -y \
    default-mysql-client \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    pandas \
    numpy \
    sqlalchemy \
    pymysql

WORKDIR /app
COPY scripts/ scripts/
COPY requirements.txt .

CMD ["python", "scripts/run.py"]
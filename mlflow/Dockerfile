FROM ghcr.io/mlflow/mlflow:v2.16.2

# MLflow image does not include optional DB/S3 drivers by default.
# PostgreSQL backend and MinIO artifact storage need these packages.
RUN pip install --no-cache-dir psycopg2-binary boto3

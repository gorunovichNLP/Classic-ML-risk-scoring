FROM python:3.11-slim

# The base mlflow image ships without DB/S3 drivers, so we add them:
#   psycopg2-binary -> connect to the Postgres backend store
#   boto3           -> connect to MinIO (S3 API) for the artifact store
RUN pip install --no-cache-dir "mlflow>=2.14,<3" psycopg2-binary boto3

EXPOSE 5000
FROM python:3.12-slim

RUN pip install --no-cache-dir mlflow==2.16.2

EXPOSE 5000

WORKDIR /mlflow

ENTRYPOINT ["mlflow", "server", \
  "--host", "0.0.0.0", \
  "--port", "5000", \
  "--backend-store-uri", "sqlite:////mlflow/store/mlflow.db", \
  "--artifacts-destination", "/mlflow/artifacts", \
  "--serve-artifacts"]

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install the hotels package + api extras. Copy only what the install needs
# first so layers stay cache-friendly.
COPY pyproject.toml ./
COPY src ./src

RUN pip install --no-cache-dir ".[api,mlflow]"

EXPOSE 8000

CMD ["uvicorn", "hotels.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

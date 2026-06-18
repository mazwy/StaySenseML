"""CLI wrapper for the experiment runner. Expects the DVC split outputs
(data/train.parquet, data/test.parquet) and a running MLflow server.

    docker compose up -d mlflow
    .venv/bin/dvc repro split    # if the split parquets aren't fresh
    .venv/bin/python scripts/run_experiments.py
"""
from hotels.experiments import run

if __name__ == "__main__":
    run()

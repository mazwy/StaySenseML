"""Sanity check that the MLflow tracking server in docker-compose is reachable
and can accept a run. Logs one dummy run with a parameter, a metric, and a
small artifact, then prints the run URL.

Run after `docker compose up -d mlflow`:

    .venv/bin/python scripts/mlflow_smoke.py
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import mlflow

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("smoke-test")

    with mlflow.start_run(run_name="smoke") as run:
        mlflow.log_param("hello", "world")
        mlflow.log_metric("answer", 42)

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "note.txt"
            p.write_text("mlflow server is reachable.\n")
            mlflow.log_artifact(str(p))

        print("ok. run id:", run.info.run_id)
        print(f"open: {TRACKING_URI}/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}")


if __name__ == "__main__":
    main()

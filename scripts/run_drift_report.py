"""Generate a data drift report and log it to MLflow.

    .venv/bin/python scripts/run_drift_report.py                 # time split (default)
    .venv/bin/python scripts/run_drift_report.py --mode split    # train vs test
    .venv/bin/python scripts/run_drift_report.py --no-mlflow     # skip mlflow logging
"""
from hotels.monitoring import run

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["time", "split"], default="time")
    parser.add_argument("--output-dir", default="reports/drift")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()
    run(mode=args.mode, output_dir=args.output_dir, log_to_mlflow=not args.no_mlflow)

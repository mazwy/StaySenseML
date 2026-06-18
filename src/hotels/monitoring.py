"""Data drift monitoring with Evidently. Compares a reference distribution
against a current one and writes an HTML report plus a JSON summary.

Two slicing modes:
  - 'time': sort the cleaned bookings by arrival date, first 80% becomes the
    reference set, last 20% becomes the current set. This is the realistic
    monitoring scenario (later data vs earlier data).
  - 'split': use the DVC-produced train.parquet as reference and test.parquet
    as current. Random stratified split, so drift should be near zero. Useful
    as a sanity check.

Both modes engineer features inside the function so the report runs on the
columns the model actually consumes.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from hotels import config, features
from hotels.stages._io import load_params, resolve

MONTH_TO_NUM = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}


def _add_sort_key(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_sort_key"] = pd.to_datetime(
        df["arrival_date_year"].astype(str)
        + "-"
        + df["arrival_date_month"].map(MONTH_TO_NUM).astype(str)
        + "-"
        + df["arrival_date_day_of_month"].astype(str),
        errors="coerce",
    )
    return df


def load_time_split(cleaned_path: Path, ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_parquet(cleaned_path)
    df = _add_sort_key(df).sort_values("_sort_key").reset_index(drop=True)
    cutoff = int(len(df) * ratio)
    reference = df.iloc[:cutoff].drop(columns=["_sort_key"])
    current = df.iloc[cutoff:].drop(columns=["_sort_key"])
    return features.engineer(reference), features.engineer(current)


def load_split_mode() -> tuple[pd.DataFrame, pd.DataFrame]:
    p = load_params()
    reference = pd.read_parquet(resolve(p["paths"]["train"]))
    current = pd.read_parquet(resolve(p["paths"]["test"]))
    return reference, current


def build_report(reference: pd.DataFrame, current: pd.DataFrame) -> evidently.core.report.Snapshot:  # noqa: F821
    cols = list(config.NUMERIC_COLUMNS) + list(config.CATEGORICAL_COLUMNS)
    reference = reference[cols]
    current = current[cols]
    report = Report(metrics=[DataDriftPreset(columns=cols)])
    return report.run(current_data=current, reference_data=reference)


def _summary(snapshot, reference: pd.DataFrame, current: pd.DataFrame, mode: str) -> dict:
    return {
        "mode": mode,
        "reference_rows": int(len(reference)),
        "current_rows": int(len(current)),
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "result_dict": snapshot.dict(),
    }


def run(mode: str = "time", output_dir: str | Path = "reports/drift",
        log_to_mlflow: bool = True) -> dict:
    if mode == "time":
        p = load_params()
        cleaned = resolve(p["paths"]["cleaned"])
        reference, current = load_time_split(cleaned)
    elif mode == "split":
        reference, current = load_split_mode()
    else:
        raise ValueError(f"unknown mode {mode!r}, use 'time' or 'split'")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / f"drift_{mode}.html"
    json_path = out_dir / f"drift_{mode}.json"

    snapshot = build_report(reference, current)
    snapshot.save_html(str(html_path))
    summary = _summary(snapshot, reference, current, mode)
    json_path.write_text(json.dumps(summary, indent=2, default=str))

    print(f"drift({mode}): reference n={len(reference)}, current n={len(current)}")
    print(f"wrote {html_path}")
    print(f"wrote {json_path}")

    if log_to_mlflow:
        _log_to_mlflow(mode, html_path, json_path, len(reference), len(current))

    return {"html": str(html_path), "json": str(json_path), "summary": summary}


def _log_to_mlflow(mode: str, html: Path, json_path: Path, n_ref: int, n_cur: int) -> None:
    import mlflow

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001"))
    mlflow.set_experiment("hotels-drift-monitoring")
    with mlflow.start_run(run_name=f"drift-{mode}"):
        mlflow.log_param("mode", mode)
        mlflow.log_param("reference_rows", n_ref)
        mlflow.log_param("current_rows", n_cur)
        mlflow.log_artifact(str(html))
        mlflow.log_artifact(str(json_path))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["time", "split"], default="time")
    parser.add_argument("--output-dir", default="reports/drift")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()
    run(mode=args.mode, output_dir=args.output_dir, log_to_mlflow=not args.no_mlflow)

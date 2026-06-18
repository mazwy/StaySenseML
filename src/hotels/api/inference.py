"""Model loading + inference plumbing for the FastAPI app.

The production model is fetched from the MLflow registry via the alias defined
in `hotels.experiments`. The cleaning chain run here matches what the model
saw during training, minus `drop_junk_rows` (the API should not silently drop
records: schema validation does that job upstream)."""
from __future__ import annotations

import os
from dataclasses import dataclass

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient

from hotels import data, features
from hotels.experiments import PRODUCTION_ALIAS, REGISTERED_MODEL_NAME

DEFAULT_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")
DEFAULT_THRESHOLD = float(os.environ.get("HOTELS_DECISION_THRESHOLD", "0.5"))


@dataclass
class LoadedModel:
    pipeline: object
    version: str
    model_kind: str | None
    auc: str | None
    tracking_uri: str

    @property
    def model_uri(self) -> str:
        return f"models:/{REGISTERED_MODEL_NAME}@{PRODUCTION_ALIAS}"


def load_production_model(tracking_uri: str = DEFAULT_TRACKING_URI) -> LoadedModel:
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, PRODUCTION_ALIAS)
    pipeline = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}@{PRODUCTION_ALIAS}")
    return LoadedModel(
        pipeline=pipeline,
        version=mv.version,
        model_kind=mv.tags.get("model_kind"),
        auc=mv.tags.get("auc"),
        tracking_uri=tracking_uri,
    )


def bookings_to_frame(records: list[dict]) -> pd.DataFrame:
    """Apply the inference-time preprocessing chain: fill_missing + engineer.
    Does *not* drop_leakage (the API never receives those columns) or
    drop_junk_rows (validation should reject bad rows upstream)."""
    df = pd.DataFrame(records)
    df = data.fill_missing(df)
    df = features.engineer(df)
    return df


def predict(loaded: LoadedModel, records: list[dict], threshold: float) -> list[dict]:
    X = bookings_to_frame(records)
    proba = loaded.pipeline.predict_proba(X)[:, 1]
    labels = (proba >= threshold).astype(int)
    return [
        {"probability_cancelled": float(p), "label": int(lbl), "threshold": threshold}
        for p, lbl in zip(proba, labels, strict=True)
    ]

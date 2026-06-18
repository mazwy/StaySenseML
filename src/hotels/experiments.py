"""Multi-model experiment runner with MLflow tracking + registry.

Trains LogReg, RandomForest, and HistGradientBoosting on the same train/test
split (produced by the DVC split stage). Each model is a separate MLflow run
inside the same experiment. The run with the highest held-out ROC AUC is
registered as a new model version under the registered model name, and the
`production` alias is moved onto it.
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

from hotels import config, preprocess
from hotels.stages._io import load_params, resolve

warnings.filterwarnings("ignore")

EXPERIMENT_NAME = "hotels-cancellation"
REGISTERED_MODEL_NAME = "hotels-cancellation"
PRODUCTION_ALIAS = "production"
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")


@dataclass
class Candidate:
    name: str
    estimator: BaseEstimator
    fit_kwargs_factory: Any = None  # callable(y_train) -> dict, or None


def _build_candidates(model_params: dict) -> list[Candidate]:
    return [
        Candidate(
            name="logreg",
            estimator=LogisticRegression(
                C=model_params["C"],
                penalty=model_params["penalty"],
                solver=model_params["solver"],
                class_weight=model_params["class_weight"],
                max_iter=model_params["max_iter"],
                random_state=config.RANDOM_STATE,
            ),
        ),
        Candidate(
            name="random_forest",
            estimator=RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=-1,
                random_state=config.RANDOM_STATE,
            ),
        ),
        Candidate(
            name="hist_gbm",
            estimator=HistGradientBoostingClassifier(
                max_iter=400,
                learning_rate=0.05,
                max_leaf_nodes=63,
                random_state=config.RANDOM_STATE,
            ),
            # HistGB doesn't take class_weight; pass balanced sample_weight at fit.
            fit_kwargs_factory=lambda y: {
                "clf__sample_weight": compute_sample_weight("balanced", y)
            },
        ),
    ]


def _score(name: str, pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    return {
        "auc": float(roc_auc_score(y_test, y_proba)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_cancelled": float(precision_score(y_test, y_pred)),
        "recall_cancelled": float(recall_score(y_test, y_pred)),
        "f1_cancelled": float(f1_score(y_test, y_pred)),
    }


def _load_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    p = load_params()
    train = pd.read_parquet(resolve(p["paths"]["train"]))
    test = pd.read_parquet(resolve(p["paths"]["test"]))
    y_train = train[config.TARGET_COLUMN]
    X_train = train.drop(columns=[config.TARGET_COLUMN])
    y_test = test[config.TARGET_COLUMN]
    X_test = test.drop(columns=[config.TARGET_COLUMN])
    return X_train, X_test, y_train, y_test


def _set_production_alias(client: MlflowClient, name: str, version: str) -> None:
    """Move the production alias onto the given version. Safe across re-runs."""
    try:
        client.delete_registered_model_alias(name, PRODUCTION_ALIAS)
    except Exception:
        pass
    client.set_registered_model_alias(name, PRODUCTION_ALIAS, version)


def run() -> dict[str, Any]:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient()

    X_train, X_test, y_train, y_test = _load_split()
    params = load_params()
    candidates = _build_candidates(params["model"])

    results = []
    for cand in candidates:
        with mlflow.start_run(run_name=cand.name) as run:
            pipe = Pipeline(
                [
                    ("preproc", preprocess.build_preprocessor()),
                    ("clf", cand.estimator),
                ]
            )
            fit_kwargs = cand.fit_kwargs_factory(y_train) if cand.fit_kwargs_factory else {}
            pipe.fit(X_train, y_train, **fit_kwargs)

            metrics = _score(cand.name, pipe, X_test, y_test)
            mlflow.log_param("model_kind", cand.name)
            mlflow.log_param("split_random_state", config.RANDOM_STATE)
            mlflow.log_param("test_size", params["split"]["test_size"])
            for k, v in cand.estimator.get_params().items():
                if isinstance(v, (int, float, str, bool)) or v is None:
                    mlflow.log_param(f"clf__{k}", v)
            for k, v in metrics.items():
                mlflow.log_metric(k, v)

            sample = X_test.head(5)
            signature = infer_signature(sample, pipe.predict(sample))
            mlflow.sklearn.log_model(
                sk_model=pipe,
                name="model",
                signature=signature,
                input_example=sample,
            )
            print(f"{cand.name:>14}  auc={metrics['auc']:.4f}  run_id={run.info.run_id}")
            results.append((cand.name, metrics["auc"], run.info.run_id))

    winner_name, winner_auc, winner_run_id = max(results, key=lambda r: r[1])
    print(f"\nwinner: {winner_name} with AUC {winner_auc:.4f}")

    model_uri = f"runs:/{winner_run_id}/model"
    mv = mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)
    client.set_model_version_tag(REGISTERED_MODEL_NAME, mv.version, "model_kind", winner_name)
    client.set_model_version_tag(REGISTERED_MODEL_NAME, mv.version, "auc", f"{winner_auc:.4f}")
    _set_production_alias(client, REGISTERED_MODEL_NAME, mv.version)
    print(
        f"registered as {REGISTERED_MODEL_NAME} v{mv.version} with alias "
        f"'{PRODUCTION_ALIAS}'"
    )
    return {
        "winner": winner_name,
        "auc": winner_auc,
        "version": mv.version,
        "all": results,
    }


if __name__ == "__main__":
    run()

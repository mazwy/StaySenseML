"""Stage 4: fit a model on the train split, evaluate on test, write model and
metrics. Phase 4 will replace the model construction here with MLflow-tracked
experiments across multiple model families. For now this just trains one
LogisticRegression with the params from params.yaml.

Usage: python -m hotels.stages.train
"""
from __future__ import annotations

import json
import warnings

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from hotels import config, preprocess
from hotels.stages._io import ensure_parent, load_params, resolve

warnings.filterwarnings("ignore")


def _build_model(model_params: dict) -> LogisticRegression:
    return LogisticRegression(
        C=model_params["C"],
        penalty=model_params["penalty"],
        solver=model_params["solver"],
        class_weight=model_params["class_weight"],
        max_iter=model_params["max_iter"],
        random_state=config.RANDOM_STATE,
    )


def main() -> None:
    p = load_params()
    train_path = resolve(p["paths"]["train"])
    test_path = resolve(p["paths"]["test"])
    model_out = ensure_parent(resolve(p["paths"]["model"]))
    metrics_out = ensure_parent(resolve(p["paths"]["metrics"]))

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    y_train = train_df[config.TARGET_COLUMN]
    X_train = train_df.drop(columns=[config.TARGET_COLUMN])
    y_test = test_df[config.TARGET_COLUMN]
    X_test = test_df.drop(columns=[config.TARGET_COLUMN])

    pipe = Pipeline(
        [
            ("preproc", preprocess.build_preprocessor()),
            ("clf", _build_model(p["model"])),
        ]
    )
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    metrics = {
        "auc": float(roc_auc_score(y_test, y_proba)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_cancelled": float(precision_score(y_test, y_pred)),
        "recall_cancelled": float(recall_score(y_test, y_pred)),
        "f1_cancelled": float(f1_score(y_test, y_pred)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }

    joblib.dump(pipe, model_out)
    metrics_out.write_text(json.dumps(metrics, indent=2) + "\n")
    print("train metrics:", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

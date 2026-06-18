"""Stage 3: stratified train/test split, write train.parquet and test.parquet.

Usage: python -m hotels.stages.split
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from hotels import config
from hotels.stages._io import ensure_parent, load_params, resolve


def main() -> None:
    p = load_params()
    in_path = resolve(p["paths"]["featured"])
    train_out = ensure_parent(resolve(p["paths"]["train"]))
    test_out = ensure_parent(resolve(p["paths"]["test"]))

    df = pd.read_parquet(in_path)
    y = df[config.TARGET_COLUMN]
    X = df.drop(columns=[config.TARGET_COLUMN])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=p["split"]["test_size"],
        stratify=y,
        random_state=p["split"]["random_state"],
    )

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_df.to_parquet(train_out, index=False)
    test_df.to_parquet(test_out, index=False)

    print(
        f"split: train {train_df.shape} (cancel rate {y_train.mean():.4f}), "
        f"test {test_df.shape} (cancel rate {y_test.mean():.4f})"
    )


if __name__ == "__main__":
    main()

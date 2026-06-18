"""Train/test split helpers. Stratified 80/20 by default. Kept in its own
module so the same split is reproducible from any entry-point (training script,
DVC stage, test)."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df.drop(columns=[config.TARGET_COLUMN]), df[config.TARGET_COLUMN]


def make_train_test(
    df: pd.DataFrame,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X, y = split_xy(df)
    return train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

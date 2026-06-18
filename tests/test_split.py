"""Stratified split: rate is preserved on both sides, RNG is reproducible."""
from __future__ import annotations

import numpy as np
import pandas as pd

from hotels import config, split


def _balanced_frame(n: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    y = (rng.random(n) < 0.37).astype(int)
    return pd.DataFrame({
        "is_canceled": y,
        "x1": rng.normal(size=n),
        "x2": rng.normal(size=n),
    })


def test_split_xy_splits_target_off():
    df = _balanced_frame()
    X, y = split.split_xy(df)
    assert "is_canceled" not in X.columns
    assert y.name == "is_canceled"
    assert len(X) == len(y) == len(df)


def test_train_test_preserves_cancel_rate():
    df = _balanced_frame(n=5000)
    Xtr, Xte, ytr, yte = split.make_train_test(df)
    base = df["is_canceled"].mean()
    assert abs(ytr.mean() - base) < 0.01
    assert abs(yte.mean() - base) < 0.01


def test_train_test_is_deterministic():
    df = _balanced_frame()
    a = split.make_train_test(df, random_state=config.RANDOM_STATE)
    b = split.make_train_test(df, random_state=config.RANDOM_STATE)
    assert a[3].equals(b[3])  # y_test identical

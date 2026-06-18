"""Step 4 of the notebook: engineered features and dropping the columns they
replace. Kept idempotent so the FastAPI service can call this on a single
incoming row with the same code path."""

from __future__ import annotations

import pandas as pd

from . import config


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df["total_guests"] = df["adults"] + df["children"] + df["babies"]
    df["is_family"] = ((df["children"] > 0) | (df["babies"] > 0)).astype(int)
    df["room_changed"] = (df["reserved_room_type"] != df["assigned_room_type"]).astype(int)
    return df


def drop_redundant_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=config.REDUNDANT_COLUMNS, errors="ignore")


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Full Step 4 pipeline in one call."""
    return drop_redundant_columns(add_engineered_features(df))

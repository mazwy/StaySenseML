"""Raw data loading and the cleaning steps from Step 3 of the notebook.

The order matters. drop_leakage must run before anything else (or any model
trained on the result picks up the post-cancellation status column and scores
fake accuracy). Each function returns a new dataframe rather than mutating in
place."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    return pd.read_csv(path if path is not None else config.DEFAULT_RAW_PATH)


def drop_leakage(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=config.LEAKAGE_COLUMNS, errors="ignore")


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """children to 0, country to 'Unknown', agent/company collapsed to bool flags."""
    df = df.copy()
    df["children"] = df["children"].fillna(0).astype(int)
    df["country"] = df["country"].fillna("Unknown")
    df["has_agent"] = df["agent"].notna().astype(int)
    df["has_company"] = df["company"].notna().astype(int)
    return df.drop(columns=["agent", "company"])


def drop_junk_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Zero-guest records (no actual person) and the one adr=5400 outlier."""
    guests = df["adults"] + df["children"] + df["babies"]
    df = df[guests > 0]
    df = df[(df["adr"] >= 0) & (df["adr"] < config.ADR_MAX)]
    return df.copy()


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Full Step 3 pipeline in one call."""
    df = drop_leakage(df)
    df = fill_missing(df)
    df = drop_junk_rows(df)
    return df

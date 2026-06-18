"""ColumnTransformer factory. Centralised so that training, the FastAPI service,
and tests all build the exact same preprocessor."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), list(config.NUMERIC_COLUMNS)),
            (
                "cat",
                OneHotEncoder(
                    min_frequency=config.COUNTRY_MIN_FREQUENCY,
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                list(config.CATEGORICAL_COLUMNS),
            ),
        ]
    )

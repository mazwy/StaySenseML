"""Preprocessor smoke test. Checks the ColumnTransformer fits + transforms
and that `country` rare-bucketing keeps the encoded feature count bounded."""
from __future__ import annotations

import numpy as np
import pandas as pd

from hotels import config, data, features, preprocess


def _synthetic_engineered_frame(n: int = 200) -> pd.DataFrame:
    """A larger frame than the conftest fixture, so the encoder has enough
    rows to fit. Uses configured column lists so the test stays in sync."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        rows.append(dict(
            hotel="City Hotel" if i % 2 else "Resort Hotel",
            is_canceled=int(i % 3 == 0),
            lead_time=int(rng.integers(0, 400)),
            arrival_date_year=int(rng.choice([2015, 2016, 2017])),
            arrival_date_month=rng.choice(["January", "July", "October"]),
            arrival_date_week_number=int(rng.integers(1, 53)),
            arrival_date_day_of_month=int(rng.integers(1, 28)),
            stays_in_weekend_nights=int(rng.integers(0, 4)),
            stays_in_week_nights=int(rng.integers(0, 6)),
            adults=int(rng.integers(1, 4)),
            children=int(rng.integers(0, 2)),
            babies=0,
            meal=rng.choice(["BB", "HB", "FB"]),
            country=rng.choice(["PRT", "GBR", "FRA", "ESP", "DEU", "RAREXYZ"]),
            market_segment=rng.choice(["Direct", "Online TA", "Groups"]),
            distribution_channel=rng.choice(["Direct", "TA/TO"]),
            is_repeated_guest=int(rng.integers(0, 2)),
            previous_cancellations=int(rng.integers(0, 3)),
            previous_bookings_not_canceled=int(rng.integers(0, 3)),
            reserved_room_type=rng.choice(["A", "D"]),
            assigned_room_type=rng.choice(["A", "D"]),
            booking_changes=int(rng.integers(0, 3)),
            deposit_type=rng.choice(["No Deposit", "Non Refund"]),
            agent=240.0 if i % 4 else np.nan,
            company=np.nan,
            days_in_waiting_list=0,
            customer_type=rng.choice(["Transient", "Contract"]),
            adr=float(rng.uniform(40, 300)),
            required_car_parking_spaces=int(rng.integers(0, 2)),
            total_of_special_requests=int(rng.integers(0, 4)),
        ))
    df = pd.DataFrame(rows)
    return features.engineer(data.fill_missing(df))


def test_preprocessor_fits_and_produces_dense_array():
    df = _synthetic_engineered_frame()
    X = df.drop(columns=[config.TARGET_COLUMN])
    pre = preprocess.build_preprocessor()
    Xt = pre.fit_transform(X)
    assert Xt.ndim == 2
    assert Xt.shape[0] == len(df)
    # Numeric columns survive, categoricals expand via one-hot.
    assert Xt.shape[1] > len(config.NUMERIC_COLUMNS)


def test_country_rare_bucketing_collapses_long_tail():
    df = _synthetic_engineered_frame()
    X = df.drop(columns=[config.TARGET_COLUMN])
    pre = preprocess.build_preprocessor()
    pre.fit(X)
    names = list(pre.get_feature_names_out())
    # "RAREXYZ" appears once or twice; min_frequency=50 must bucket it out.
    assert not any("country_RAREXYZ" in n for n in names)
    assert any("country_infrequent_sklearn" in n for n in names)

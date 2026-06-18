"""API smoke tests. Patches the MLflow model loader so CI does not need a
running tracking server. The fake pipeline is a real sklearn Pipeline (the
project's own preprocessor + a tiny LogReg fit on synthetic data), so we
exercise the full bookings_to_frame -> preprocess -> predict path."""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from hotels import config, data, features, preprocess
from hotels.api.inference import LoadedModel


def _fake_pipeline() -> Pipeline:
    """Build a real pipeline fit on synthetic engineered data."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(200):
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
            country=rng.choice(["PRT", "GBR", "FRA", "ESP", "DEU"]),
            market_segment=rng.choice(["Direct", "Online TA", "Groups"]),
            distribution_channel=rng.choice(["Direct", "TA/TO"]),
            is_repeated_guest=int(rng.integers(0, 2)),
            previous_cancellations=int(rng.integers(0, 3)),
            previous_bookings_not_canceled=int(rng.integers(0, 3)),
            reserved_room_type=rng.choice(["A", "D"]),
            assigned_room_type=rng.choice(["A", "D"]),
            booking_changes=0,
            deposit_type=rng.choice(["No Deposit", "Non Refund"]),
            agent=240.0 if i % 4 else np.nan,
            company=np.nan,
            days_in_waiting_list=0,
            customer_type=rng.choice(["Transient", "Contract"]),
            adr=float(rng.uniform(40, 300)),
            required_car_parking_spaces=int(rng.integers(0, 2)),
            total_of_special_requests=int(rng.integers(0, 4)),
        ))
    df = features.engineer(data.fill_missing(pd.DataFrame(rows)))
    y = df[config.TARGET_COLUMN]
    X = df.drop(columns=[config.TARGET_COLUMN])
    pipe = Pipeline([
        ("preproc", preprocess.build_preprocessor()),
        ("clf", LogisticRegression(max_iter=500, random_state=0)),
    ])
    pipe.fit(X, y)
    return pipe


@pytest.fixture
def client():
    fake = LoadedModel(
        pipeline=_fake_pipeline(),
        version="test-1",
        model_kind="logreg",
        auc="0.5000",
        tracking_uri="http://test",
    )
    with patch("hotels.api.inference.load_production_model", return_value=fake):
        from hotels.api.main import app
        with TestClient(app) as c:
            yield c


SAMPLE_BOOKING = dict(
    hotel="Resort Hotel", lead_time=10, arrival_date_year=2016,
    arrival_date_month="July", arrival_date_week_number=27,
    arrival_date_day_of_month=2, stays_in_weekend_nights=2,
    stays_in_week_nights=3, adults=2, children=0, babies=0, meal="BB",
    country="PRT", market_segment="Direct", distribution_channel="Direct",
    is_repeated_guest=0, previous_cancellations=0,
    previous_bookings_not_canceled=0, reserved_room_type="A",
    assigned_room_type="A", booking_changes=0, deposit_type="No Deposit",
    agent=240, company=None, days_in_waiting_list=0,
    customer_type="Transient", adr=85.0, required_car_parking_spaces=1,
    total_of_special_requests=2,
)


def test_healthz_returns_ok_with_loaded_model(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"] == "test-1"


def test_info_includes_kind_and_alias(client):
    r = client.get("/info")
    assert r.status_code == 200
    body = r.json()
    assert body["model_kind"] == "logreg"
    assert body["alias"] == "production"


def test_predict_returns_probability_and_label(client):
    r = client.post("/predict", json=SAMPLE_BOOKING)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["probability_cancelled"] <= 1.0
    assert body["label"] in (0, 1)
    assert body["threshold"] == 0.5


def test_predict_batch_handles_empty_list(client):
    r = client.post("/predict/batch", json={"bookings": []})
    assert r.status_code == 200
    assert r.json() == {"predictions": []}


def test_predict_batch_returns_same_length(client):
    r = client.post(
        "/predict/batch",
        json={"bookings": [SAMPLE_BOOKING, SAMPLE_BOOKING, SAMPLE_BOOKING]},
    )
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 3


def test_predict_rejects_invalid_hotel(client):
    bad = dict(SAMPLE_BOOKING, hotel="Motel 6")
    r = client.post("/predict", json=bad)
    assert r.status_code == 422

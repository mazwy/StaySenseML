"""Step 4 feature engineering. Locks the formulas and the drop-list."""
from __future__ import annotations

from hotels import data, features


def test_add_engineered_features_produces_the_four_new_columns(raw_df):
    cleaned = data.clean(raw_df)
    out = features.add_engineered_features(cleaned)
    for col in ["total_nights", "total_guests", "is_family", "room_changed"]:
        assert col in out.columns


def test_engineered_features_compute_correctly(raw_df):
    cleaned = data.clean(raw_df)
    out = features.add_engineered_features(cleaned)

    # Pick the family-stay row: 2+2+1 guests, 2 weekend + 2 week nights,
    # has children/babies, reserved == assigned room type.
    family = out[out["country"] == "GBR"].iloc[0]
    assert family["total_nights"] == 4
    assert family["total_guests"] == 5
    assert family["is_family"] == 1
    assert family["room_changed"] == 0


def test_drop_redundant_columns_removes_all_planned_drops(raw_df):
    cleaned = data.clean(raw_df)
    engineered = features.add_engineered_features(cleaned)
    out = features.drop_redundant_columns(engineered)
    must_be_gone = [
        "stays_in_weekend_nights", "stays_in_week_nights",
        "adults", "children", "babies",
        "reserved_room_type", "assigned_room_type",
        "arrival_date_year", "arrival_date_week_number",
        "arrival_date_day_of_month",
    ]
    for col in must_be_gone:
        assert col not in out.columns
    # arrival_date_month survives for seasonality.
    assert "arrival_date_month" in out.columns


def test_engineer_full_pipeline_shape(raw_df):
    out = features.engineer(data.clean(raw_df))
    assert out.shape[1] == 24

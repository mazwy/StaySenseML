"""Locks down the Step 3 cleaning behaviour. These are the README constraints
that, if quietly broken, would corrupt the model in a way that's hard to spot
later (a fake-good AUC from leakage, or a silent data shift)."""
from __future__ import annotations

from hotels import data


def test_drop_leakage_removes_both_columns(raw_df):
    out = data.drop_leakage(raw_df)
    assert "reservation_status" not in out.columns
    assert "reservation_status_date" not in out.columns
    # Other columns are untouched.
    assert "is_canceled" in out.columns
    assert len(out) == len(raw_df)


def test_drop_leakage_is_idempotent(raw_df):
    out = data.drop_leakage(data.drop_leakage(raw_df))
    assert "reservation_status" not in out.columns


def test_fill_missing_yields_no_nans_in_relevant_cols(raw_df):
    out = data.fill_missing(raw_df)
    assert out["children"].isna().sum() == 0
    assert out["country"].isna().sum() == 0
    assert out["children"].dtype.kind in {"i", "u"}


def test_fill_missing_creates_flag_columns_and_drops_ids(raw_df):
    out = data.fill_missing(raw_df)
    assert "has_agent" in out.columns
    assert "has_company" in out.columns
    assert "agent" not in out.columns
    assert "company" not in out.columns
    # Flags are 0/1 ints.
    assert set(out["has_agent"].unique()).issubset({0, 1})
    assert set(out["has_company"].unique()).issubset({0, 1})


def test_drop_junk_rows_removes_zero_guests_and_adr_outlier(raw_df):
    cleaned = data.fill_missing(raw_df)
    out = data.drop_junk_rows(cleaned)
    guests = out["adults"] + out["children"] + out["babies"]
    assert (guests > 0).all()
    assert (out["adr"] >= 0).all()
    assert (out["adr"] < 1000).all()


def test_clean_runs_all_three_in_order(raw_df):
    out = data.clean(raw_df)
    assert "reservation_status" not in out.columns
    assert "agent" not in out.columns
    assert out["children"].isna().sum() == 0
    # The zero-guest row and the adr=5400 row are gone (2 out of 5).
    assert len(out) == 3

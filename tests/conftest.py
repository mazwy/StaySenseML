"""Shared fixtures. The `raw_df` fixture is a small synthetic booking frame
with the exact columns the cleaning pipeline expects, so tests do not depend
on `hotels.csv` being available."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """Mirrors the 32-column raw CSV layout, with a handful of rows that
    exercise the cleaning rules: a leakage row, a NaN-in-`children` row, a
    zero-guest row, and an adr outlier."""
    rows = [
        # Normal kept booking
        dict(hotel="Resort Hotel", is_canceled=0, lead_time=10,
             arrival_date_year=2016, arrival_date_month="July",
             arrival_date_week_number=27, arrival_date_day_of_month=2,
             stays_in_weekend_nights=2, stays_in_week_nights=3,
             adults=2, children=0, babies=0, meal="BB", country="PRT",
             market_segment="Direct", distribution_channel="Direct",
             is_repeated_guest=0, previous_cancellations=0,
             previous_bookings_not_canceled=0, reserved_room_type="A",
             assigned_room_type="A", booking_changes=0,
             deposit_type="No Deposit", agent=240.0, company=np.nan,
             days_in_waiting_list=0, customer_type="Transient",
             adr=85.0, required_car_parking_spaces=1,
             total_of_special_requests=2,
             reservation_status="Check-Out",
             reservation_status_date="2016-07-07"),
        # Cancelled booking with leakage columns
        dict(hotel="City Hotel", is_canceled=1, lead_time=300,
             arrival_date_year=2016, arrival_date_month="August",
             arrival_date_week_number=32, arrival_date_day_of_month=10,
             stays_in_weekend_nights=2, stays_in_week_nights=5,
             adults=2, children=np.nan, babies=0, meal="HB",
             country=np.nan, market_segment="Online TA",
             distribution_channel="TA/TO", is_repeated_guest=0,
             previous_cancellations=1, previous_bookings_not_canceled=0,
             reserved_room_type="D", assigned_room_type="A",
             booking_changes=0, deposit_type="Non Refund",
             agent=np.nan, company=140.0, days_in_waiting_list=0,
             customer_type="Transient", adr=120.0,
             required_car_parking_spaces=0,
             total_of_special_requests=0,
             reservation_status="Canceled",
             reservation_status_date="2016-05-01"),
        # Family stay
        dict(hotel="Resort Hotel", is_canceled=0, lead_time=5,
             arrival_date_year=2017, arrival_date_month="June",
             arrival_date_week_number=23, arrival_date_day_of_month=5,
             stays_in_weekend_nights=2, stays_in_week_nights=2,
             adults=2, children=2, babies=1, meal="FB", country="GBR",
             market_segment="Direct", distribution_channel="Direct",
             is_repeated_guest=1, previous_cancellations=0,
             previous_bookings_not_canceled=3, reserved_room_type="E",
             assigned_room_type="E", booking_changes=1,
             deposit_type="No Deposit", agent=np.nan, company=np.nan,
             days_in_waiting_list=0, customer_type="Transient",
             adr=200.0, required_car_parking_spaces=1,
             total_of_special_requests=3,
             reservation_status="Check-Out",
             reservation_status_date="2017-06-09"),
        # Zero-guest junk row
        dict(hotel="City Hotel", is_canceled=0, lead_time=20,
             arrival_date_year=2016, arrival_date_month="May",
             arrival_date_week_number=19, arrival_date_day_of_month=12,
             stays_in_weekend_nights=0, stays_in_week_nights=1,
             adults=0, children=0, babies=0, meal="BB", country="ESP",
             market_segment="Online TA", distribution_channel="TA/TO",
             is_repeated_guest=0, previous_cancellations=0,
             previous_bookings_not_canceled=0, reserved_room_type="A",
             assigned_room_type="A", booking_changes=0,
             deposit_type="No Deposit", agent=14.0, company=np.nan,
             days_in_waiting_list=0, customer_type="Transient",
             adr=80.0, required_car_parking_spaces=0,
             total_of_special_requests=0,
             reservation_status="Check-Out",
             reservation_status_date="2016-05-13"),
        # adr outlier
        dict(hotel="City Hotel", is_canceled=1, lead_time=15,
             arrival_date_year=2016, arrival_date_month="December",
             arrival_date_week_number=52, arrival_date_day_of_month=22,
             stays_in_weekend_nights=1, stays_in_week_nights=1,
             adults=1, children=0, babies=0, meal="BB", country="USA",
             market_segment="Direct", distribution_channel="Direct",
             is_repeated_guest=0, previous_cancellations=0,
             previous_bookings_not_canceled=0, reserved_room_type="A",
             assigned_room_type="A", booking_changes=0,
             deposit_type="No Deposit", agent=np.nan, company=np.nan,
             days_in_waiting_list=0, customer_type="Transient",
             adr=5400.0, required_car_parking_spaces=0,
             total_of_special_requests=0,
             reservation_status="Canceled",
             reservation_status_date="2016-11-01"),
    ]
    return pd.DataFrame(rows)

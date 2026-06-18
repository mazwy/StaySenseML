"""Project-wide constants. Column lists and the small set of hyperparameters
that have to stay in sync between the training pipeline, tests, and the
prediction service."""

from pathlib import Path

RANDOM_STATE = 42
TEST_SIZE = 0.2

DEFAULT_RAW_PATH = Path(__file__).resolve().parents[2] / "hotels.csv"

LEAKAGE_COLUMNS = ["reservation_status", "reservation_status_date"]

REDUNDANT_COLUMNS = [
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "reserved_room_type",
    "assigned_room_type",
    "arrival_date_year",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
]

CATEGORICAL_COLUMNS = [
    "hotel",
    "arrival_date_month",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "deposit_type",
    "customer_type",
]

NUMERIC_COLUMNS = [
    "lead_time",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "booking_changes",
    "days_in_waiting_list",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
    "has_agent",
    "has_company",
    "total_nights",
    "total_guests",
    "is_family",
    "room_changed",
]

TARGET_COLUMN = "is_canceled"

COUNTRY_MIN_FREQUENCY = 50
ADR_MAX = 1000.0

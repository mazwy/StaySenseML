"""Request and response schemas for the prediction API.

The input mirrors a raw booking record: same columns and dtypes the cleaning
pipeline expects, minus the two leakage columns. The model is trained on the
post-cleaning feature set, so the API runs `fill_missing` and `engineer`
internally before calling `predict_proba`.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Booking(BaseModel):
    """A single booking. Mirrors the source csv columns the cleaning pipeline
    consumes."""

    hotel: Literal["City Hotel", "Resort Hotel"]
    lead_time: int = Field(ge=0)
    arrival_date_year: int
    arrival_date_month: str
    arrival_date_week_number: int = Field(ge=1, le=53)
    arrival_date_day_of_month: int = Field(ge=1, le=31)
    stays_in_weekend_nights: int = Field(ge=0)
    stays_in_week_nights: int = Field(ge=0)
    adults: int = Field(ge=0)
    children: float | None = None
    babies: int = Field(ge=0)
    meal: str
    country: str | None = None
    market_segment: str
    distribution_channel: str
    is_repeated_guest: int = Field(ge=0, le=1)
    previous_cancellations: int = Field(ge=0)
    previous_bookings_not_canceled: int = Field(ge=0)
    reserved_room_type: str
    assigned_room_type: str
    booking_changes: int = Field(ge=0)
    deposit_type: Literal["No Deposit", "Non Refund", "Refundable"]
    agent: float | None = None
    company: float | None = None
    days_in_waiting_list: int = Field(ge=0)
    customer_type: str
    adr: float
    required_car_parking_spaces: int = Field(ge=0)
    total_of_special_requests: int = Field(ge=0)


class Prediction(BaseModel):
    probability_cancelled: float
    label: int
    threshold: float


class BatchRequest(BaseModel):
    bookings: list[Booking]


class BatchResponse(BaseModel):
    predictions: list[Prediction]


class HealthResponse(BaseModel):
    status: Literal["ok", "loading", "error"]
    model_loaded: bool
    model_version: str | None = None
    model_kind: str | None = None


class ModelInfo(BaseModel):
    registered_name: str
    alias: str
    version: str
    model_kind: str | None
    auc: str | None
    tracking_uri: str

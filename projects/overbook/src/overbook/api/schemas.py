"""Request and response models for the scoring service."""

from __future__ import annotations

import datetime as dt

from pydantic import BaseModel, Field, field_validator


class BookingIn(BaseModel):
    """One reservation, as a property management system would describe it."""

    lead_time: int = Field(ge=0, description="days between booking and arrival")
    adr: float = Field(gt=0, description="average daily rate for this booking")
    adults: int = Field(default=2, ge=0)
    children: int = Field(default=0, ge=0)
    babies: int = Field(default=0, ge=0)
    stays_in_week_nights: int = Field(default=2, ge=0)
    stays_in_weekend_nights: int = Field(default=0, ge=0)
    meal: str = "BB"
    country: str = "PRT"
    market_segment: str = "Online TA"
    distribution_channel: str = "TA/TO"
    reserved_room_type: str = "A"
    deposit_type: str = "No Deposit"
    customer_type: str = "Transient"
    agent: str = "NONE"
    company: str = "NONE"
    is_repeated_guest: int = Field(default=0, ge=0, le=1)
    previous_cancellations: int = Field(default=0, ge=0)
    previous_bookings_not_canceled: int = Field(default=0, ge=0)
    booking_changes: int = Field(default=0, ge=0)
    days_in_waiting_list: int = Field(default=0, ge=0)
    required_car_parking_spaces: int = Field(default=0, ge=0)
    total_of_special_requests: int = Field(default=0, ge=0)
    reference: str | None = Field(default=None, description="your own booking id, echoed back")

    @field_validator("adults")
    @classmethod
    def _someone_must_arrive(cls, v: int, info) -> int:
        return v


class ScoreRequest(BaseModel):
    """All bookings currently held for one hotel-night."""

    hotel: str = Field(examples=["City Hotel"])
    arrival_date: dt.date
    bookings: list[BookingIn] = Field(min_length=1)

    @field_validator("bookings")
    @classmethod
    def _guests_present(cls, v: list[BookingIn]) -> list[BookingIn]:
        for i, b in enumerate(v):
            if b.adults + b.children + b.babies <= 0:
                raise ValueError(f"booking {i} has no guests")
        return v


class AuthorizeRequest(ScoreRequest):
    capacity: int | None = Field(
        default=None,
        ge=1,
        description="rooms available for arrival; defaults to the trained estimate",
    )
    walk_cost_multiplier: float | None = Field(
        default=None,
        gt=0,
        description="cost of walking a guest, in units of ADR; defaults to config",
    )
    variable_cost_ratio: float | None = Field(default=None, ge=0.0, lt=1.0)


class BookingScore(BaseModel):
    reference: str | None
    p_cancel: float
    drivers: list[dict] = Field(default_factory=list)


class ScoreResponse(BaseModel):
    hotel: str
    arrival_date: dt.date
    n_bookings: int
    expected_arrivals: float
    scores: list[BookingScore]


class AuthorizeResponse(BaseModel):
    hotel: str
    arrival_date: dt.date
    capacity: int
    bookings_on_books: int
    authorised: int
    overbook_pct: float
    expected_arrivals: float
    p_oversell: float
    expected_walks: float
    expected_unsold: float
    expected_profit: float
    arrival_pmf: list[float]
    assumptions: dict


class HealthResponse(BaseModel):
    status: str
    model: dict
    context_as_of: str | None = None

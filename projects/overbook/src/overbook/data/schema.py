"""Schema contract for the raw hotel-bookings extract.

Two jobs:

1. Catch a bad or changed upstream file before it silently poisons a model.
2. Name the leaky columns in exactly one place. ``tests/test_leakage.py``
   asserts that nothing in :data:`LEAKY_COLUMNS` ever reaches a feature matrix,
   so this list is the project's single definition of "must not be learned from".
"""

from __future__ import annotations

import pandas as pd

#: Columns that encode the outcome or are only known after the guest arrives.
#: Every one of these is a plausible-looking column that will hand you a
#: near-perfect model and a worthless one.
LEAKY_COLUMNS: dict[str, str] = {
    "reservation_status": (
        "Literally the target: 'Canceled'/'No-Show'/'Check-Out' is is_canceled re-spelled."
    ),
    "reservation_status_date": (
        "The date the booking reached its final status. For cancellations this is the "
        "cancellation date, which does not exist at prediction time."
    ),
    "assigned_room_type": (
        "The room actually assigned, which happens at check-in. A booking that never "
        "arrived cannot be assigned a room, so disagreement with reserved_room_type "
        "leaks the outcome."
    ),
}

#: Columns that accrue between booking creation and arrival. They are legitimate
#: under the ``arrival_eve`` view and must be dropped under ``booking_time``.
ACCRUAL_COLUMNS: tuple[str, ...] = ("booking_changes", "days_in_waiting_list")

EXPECTED_COLUMNS: tuple[str, ...] = (
    "hotel",
    "is_canceled",
    "lead_time",
    "arrival_date_year",
    "arrival_date_month",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "reserved_room_type",
    "assigned_room_type",
    "booking_changes",
    "deposit_type",
    "agent",
    "company",
    "days_in_waiting_list",
    "customer_type",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
    "reservation_status",
    "reservation_status_date",
)

MONTHS: dict[str, int] = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}


class SchemaError(ValueError):
    """Raised when the raw extract does not match the contract."""


def validate_raw(df: pd.DataFrame) -> None:
    """Validate the raw extract, collecting every problem before raising."""
    problems: list[str] = []

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        problems.append(f"missing columns: {missing}")
    unexpected = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    if unexpected:
        problems.append(f"unexpected columns: {unexpected}")

    if problems:  # nothing below can be trusted if the shape is wrong
        raise SchemaError("; ".join(problems))

    if df.empty:
        problems.append("extract is empty")
    if not df["is_canceled"].isin([0, 1]).all():
        problems.append("is_canceled contains values outside {0, 1}")
    if (df["lead_time"] < 0).any():
        problems.append("negative lead_time")
    bad_months = set(df["arrival_date_month"].unique()) - set(MONTHS)
    if bad_months:
        problems.append(f"unparseable arrival_date_month values: {sorted(bad_months)}")
    for col in ("adults", "children", "babies", "stays_in_week_nights", "stays_in_weekend_nights"):
        if (df[col].fillna(0) < 0).any():
            problems.append(f"negative {col}")

    if problems:
        raise SchemaError("; ".join(problems))


def assert_no_leakage(columns) -> None:
    """Raise if any leaky column appears in ``columns``.

    Called from the feature builder and from the test suite so the guarantee
    holds at runtime, not only under test.
    """
    found = [c for c in columns if c in LEAKY_COLUMNS]
    if found:
        detail = "\n".join(f"  - {c}: {LEAKY_COLUMNS[c]}" for c in found)
        raise SchemaError(f"leaky column(s) reached the feature matrix:\n{detail}")

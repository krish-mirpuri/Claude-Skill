"""Shared fixtures.

The suite runs without the dataset: a small synthetic extract stands in for
the real CSV, so CI needs no network and no 17 MB download. Tests that genuinely
need trained artifacts are marked ``slow`` and skip when they are absent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from overbook.config import load_config
from overbook.data.clean import clean
from overbook.data.schema import EXPECTED_COLUMNS

MONTH_NAMES = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]


def make_raw(n: int = 3000, seed: int = 11) -> pd.DataFrame:
    """A synthetic extract with the real schema and a plausible risk structure."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-07-01", "2017-08-31", freq="D")
    arrival = pd.to_datetime(rng.choice(dates, n))
    lead = rng.integers(0, 400, n)
    deposit = rng.choice(["No Deposit", "Non Refund", "Refundable"], n, p=[0.87, 0.12, 0.01])

    # Risk rises with lead time and is near-certain for non-refundable bookings,
    # mirroring the real data so tests exercise a realistic probability spread.
    logit = -1.4 + 0.004 * lead + 3.5 * (deposit == "Non Refund")
    p = 1 / (1 + np.exp(-logit))

    df = pd.DataFrame(
        {
            "hotel": rng.choice(["City Hotel", "Resort Hotel"], n),
            "is_canceled": (rng.random(n) < p).astype(int),
            "lead_time": lead,
            "arrival_date_year": arrival.year,
            "arrival_date_month": [MONTH_NAMES[m - 1] for m in arrival.month],
            "arrival_date_week_number": arrival.isocalendar().week.astype(int),
            "arrival_date_day_of_month": arrival.day,
            "stays_in_weekend_nights": rng.integers(0, 3, n),
            "stays_in_week_nights": rng.integers(0, 6, n),
            "adults": rng.integers(1, 4, n),
            "children": rng.integers(0, 2, n).astype(float),
            "babies": 0,
            "meal": rng.choice(["BB", "HB", "SC"], n),
            "country": rng.choice(["PRT", "GBR", "FRA", "ESP", "ZZZ"], n),
            "market_segment": rng.choice(["Online TA", "Groups", "Direct"], n),
            "distribution_channel": rng.choice(["TA/TO", "Direct"], n),
            "is_repeated_guest": rng.integers(0, 2, n),
            "previous_cancellations": rng.integers(0, 3, n),
            "previous_bookings_not_canceled": rng.integers(0, 3, n),
            "reserved_room_type": rng.choice(["A", "D", "E"], n),
            "assigned_room_type": rng.choice(["A", "D", "E"], n),
            "booking_changes": rng.integers(0, 3, n),
            "deposit_type": deposit,
            "agent": rng.choice(["9", "240", "NONE"], n),
            "company": "NONE",
            "days_in_waiting_list": 0,
            "customer_type": rng.choice(["Transient", "Contract"], n),
            "adr": np.round(rng.uniform(40, 220, n), 2),
            "required_car_parking_spaces": rng.integers(0, 2, n),
            "total_of_special_requests": rng.integers(0, 3, n),
            "reservation_status": "Check-Out",
            "reservation_status_date": arrival.astype(str),
        }
    )
    df.loc[df["is_canceled"] == 1, "reservation_status"] = "Canceled"
    return df[list(EXPECTED_COLUMNS)]


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    return make_raw()


@pytest.fixture(scope="session")
def clean_df(raw_df, cfg) -> pd.DataFrame:
    return clean(raw_df, cfg)


@pytest.fixture(scope="session")
def artifacts(cfg):
    """Trained artifacts, or skip — CI runs the unit suite without them."""
    from overbook.models.train import load_artifacts

    try:
        return load_artifacts(cfg)
    except FileNotFoundError:
        pytest.skip("no trained artifacts; run `make all` first")

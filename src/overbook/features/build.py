"""Leakage-safe feature construction.

Two views of the same booking:

``booking_time``
    Only what the hotel knew the moment the booking was created. This is the
    honest setting for "should we take this reservation?".

``arrival_eve``
    Everything known the night before arrival — it adds information that
    accrues while the booking sits on the books (amendments, waiting-list
    time, how full the date got). This is the setting the overbooking
    decision actually lives in, because the authorisation level is set
    against the bookings on hand.

Reporting both is the point: the gap between them is the value of waiting.

Category vocabularies are fitted on the training window only, so a category
that appears for the first time in 2017 cannot influence the encoding the
model was trained with.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from overbook.config import Config, load_config
from overbook.data.schema import ACCRUAL_COLUMNS, assert_no_leakage

log = logging.getLogger(__name__)

OTHER = "__OTHER__"

#: Columns carried through the pipeline for the decision layer. They are NOT
#: features and are never handed to a model.
KEY_COLUMNS: tuple[str, ...] = (
    "booking_id",
    "hotel",
    "arrival_date",
    "booking_date",
    "booking_seq",
    "adr",
    "split",
    "is_canceled",
)

BASE_NUMERIC: tuple[str, ...] = (
    "lead_time",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "total_guests",
    "total_nights",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
)

BASE_CATEGORICAL: tuple[str, ...] = (
    "hotel",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "deposit_type",
    "customer_type",
    "agent",
    "company",
)

#: Features only available once the booking has been sitting on the books.
ARRIVAL_EVE_NUMERIC: tuple[str, ...] = (
    *ACCRUAL_COLUMNS,  # booking_changes, days_in_waiting_list
    "bookings_on_books",  # how many bookings this arrival date attracted
    "booking_rank_frac",  # where this booking sits in that queue
    "hotel_cancel_rate_28d",
    "hotel_cancel_rate_90d",
)


@dataclass
class FeatureFrame:
    """Design matrix plus the keys needed to score a decision."""

    X: pd.DataFrame
    y: pd.Series
    keys: pd.DataFrame
    categorical: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.X)

    def subset(self, mask: pd.Series) -> FeatureFrame:
        mask = np.asarray(mask)
        return FeatureFrame(
            self.X.loc[mask].reset_index(drop=True),
            self.y.loc[mask].reset_index(drop=True),
            self.keys.loc[mask].reset_index(drop=True),
            list(self.categorical),
        )

    def split(self, name: str) -> FeatureFrame:
        return self.subset(self.keys["split"] == name)


def _calendar(df: pd.DataFrame) -> pd.DataFrame:
    a, b = df["arrival_date"].dt, df["booking_date"].dt
    out = pd.DataFrame(index=df.index)
    out["arrival_month"] = a.month
    out["arrival_dayofweek"] = a.dayofweek
    out["arrival_weekofyear"] = a.isocalendar().week.astype(int).to_numpy()
    out["arrival_dayofyear"] = a.dayofyear
    out["arrival_is_weekend"] = (a.dayofweek >= 5).astype(int)
    out["booking_dayofweek"] = b.dayofweek
    out["booking_month"] = b.month
    return out


def _ratios(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["log1p_lead_time"] = np.log1p(df["lead_time"])
    out["adr_per_guest"] = df["adr"] / df["total_guests"].clip(lower=1)
    out["adr_per_night"] = df["adr"]  # ADR is already nightly; kept for readability
    prior = df["previous_cancellations"] + df["previous_bookings_not_canceled"]
    out["prev_cancel_ratio"] = df["previous_cancellations"] / (prior + 1.0)
    out["has_agent"] = (df["agent"] != "NONE").astype(int)
    out["has_company"] = (df["company"] != "NONE").astype(int)
    out["room_type_is_standard"] = (df["reserved_room_type"] == "A").astype(int)
    return out


def _rolling_hotel_cancel_rate(df: pd.DataFrame, window: int) -> pd.Series:
    """Cancellation rate over the ``window`` days of arrivals BEFORE this one.

    Anchored on arrival date and shifted by a day, so every booking that feeds
    the average had already reached its final status by the evening before the
    arrival being scored. No outcome from the current date is included.
    """
    daily = (
        df.groupby(["hotel", "arrival_date"])["is_canceled"]
        .agg(["sum", "size"])
        .rename(columns={"sum": "cx", "size": "n"})
    )
    parts = []
    for hotel, grp in daily.groupby(level=0):
        g = grp.droplevel(0).sort_index()
        # Reindex to a dense daily calendar so "28 days" means days, not rows.
        full = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="D"), fill_value=0)
        rolled = full.shift(1).rolling(window, min_periods=1).sum()
        rate = (rolled["cx"] / rolled["n"].replace(0, np.nan)).rename("rate")
        parts.append(rate.to_frame().assign(hotel=hotel).set_index("hotel", append=True))
    lookup = pd.concat(parts).reorder_levels([1, 0])["rate"]
    idx = pd.MultiIndex.from_arrays([df["hotel"], df["arrival_date"]])
    return pd.Series(lookup.reindex(idx).to_numpy(), index=df.index)


def _on_books(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    n = df.groupby(["hotel", "arrival_date"])["booking_id"].transform("size")
    out["bookings_on_books"] = n
    out["booking_rank_frac"] = df["booking_seq"] / n.clip(lower=1)
    return out


class FeatureBuilder:
    """Fit category vocabularies on train, then transform any window."""

    def __init__(self, cfg: Config | None = None, view: str | None = None):
        self.cfg = cfg or load_config()
        self.view = view or self.cfg.features.view
        if self.view not in {"arrival_eve", "booking_time"}:
            raise ValueError(f"unknown view {self.view!r}")
        self.vocab_: dict[str, set[str]] = {}
        self.columns_: list[str] = []
        self.categorical_: list[str] = []

    # -- fitting ----------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> FeatureBuilder:
        """Learn category vocabularies from the given (training) window."""
        min_freq = self.cfg.features.min_country_freq
        for col in BASE_CATEGORICAL:
            counts = df[col].astype(str).value_counts()
            keep = (
                counts[counts >= min_freq].index
                if col in {"country", "agent", "company"}
                else counts.index
            )
            self.vocab_[col] = set(map(str, keep))
        return self

    # -- transforming -----------------------------------------------------
    def transform(
        self, df: pd.DataFrame, rolling_overrides: dict[str, float] | None = None
    ) -> FeatureFrame:
        """Build the design matrix.

        ``rolling_overrides`` exists for serving. The rolling cancellation
        features are computed from realised outcomes, which a live request does
        not carry, so the API supplies the last known value per hotel instead
        of silently computing one from the request's own (unknown) labels.
        """
        if not self.vocab_:
            raise RuntimeError("FeatureBuilder.transform called before fit")

        numeric = [*BASE_NUMERIC]
        blocks = [df[list(BASE_NUMERIC)], _calendar(df), _ratios(df)]

        if self.view == "arrival_eve":
            eve = _on_books(df)
            for w in self.cfg.features.rolling_cancel_windows:
                col = f"hotel_cancel_rate_{w}d"
                if rolling_overrides and col in rolling_overrides:
                    eve[col] = float(rolling_overrides[col])
                else:
                    eve[col] = _rolling_hotel_cancel_rate(df, w)
            eve[list(ACCRUAL_COLUMNS)] = df[list(ACCRUAL_COLUMNS)]
            blocks.append(eve)

        cats = pd.DataFrame(index=df.index)
        for col in BASE_CATEGORICAL:
            s = df[col].astype(str)
            s = s.where(s.isin(self.vocab_[col]), OTHER)
            cats[col] = pd.Categorical(s, categories=sorted(self.vocab_[col] | {OTHER}))
        blocks.append(cats)

        X = pd.concat(blocks, axis=1)
        # A column added twice would silently shadow itself downstream.
        dupes = X.columns[X.columns.duplicated()].tolist()
        if dupes:
            raise ValueError(f"duplicate feature columns: {dupes}")

        # Runtime guarantee, not just a unit test.
        assert_no_leakage(X.columns)

        self.columns_ = X.columns.tolist()
        self.categorical_ = list(BASE_CATEGORICAL)
        del numeric  # kept above only for readability of the block order

        keys = df[list(KEY_COLUMNS)].reset_index(drop=True)
        return FeatureFrame(
            X.reset_index(drop=True),
            df["is_canceled"].reset_index(drop=True).rename("is_canceled"),
            keys,
            list(BASE_CATEGORICAL),
        )

    def fit_transform_all(self, df: pd.DataFrame) -> FeatureFrame:
        """Fit on the train window, transform every row.

        Fitting on ``split == 'train'`` only is what keeps the later windows
        genuinely out of sample.
        """
        self.fit(df[df["split"] == "train"])
        ff = self.transform(df)
        log.info(
            "built %d features (view=%s) for %d bookings",
            ff.X.shape[1],
            self.view,
            len(ff),
        )
        return ff


def build_features(
    df: pd.DataFrame, cfg: Config | None = None, view: str | None = None
) -> tuple[FeatureFrame, FeatureBuilder]:
    fb = FeatureBuilder(cfg, view)
    return fb.fit_transform_all(df), fb

"""Turn the raw extract into one canonical, typed booking table.

Nothing here is modelling: this step only fixes types, repairs known data
defects, and derives the two timestamps the whole project hangs off —
``arrival_date`` and ``booking_date``.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from overbook.config import Config, load_config
from overbook.data.schema import MONTHS, validate_raw

log = logging.getLogger(__name__)

#: Rows whose booking covers nobody. 180 such rows exist; they cannot arrive
#: and cannot be walked, so they are not decisions.
_ZERO_GUEST_MSG = "dropped %d bookings with zero guests (adults+children+babies == 0)"


def load_raw(cfg: Config | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    path = cfg.resolve(cfg.data.raw_path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `make data` (or `overbook download`) first."
        )
    df = pd.read_csv(path, low_memory=False)
    validate_raw(df)
    return df


def clean(df: pd.DataFrame, cfg: Config | None = None) -> pd.DataFrame:
    """Clean the raw extract into the canonical booking table."""
    cfg = cfg or load_config()
    df = df.copy()

    if cfg.data.drop_exact_duplicates:
        before = len(df)
        df = df.drop_duplicates()
        log.info("dropped %d exact duplicate rows", before - len(df))

    # --- the two timestamps everything else depends on -------------------
    df["arrival_date"] = pd.to_datetime(
        dict(
            year=df["arrival_date_year"],
            month=df["arrival_date_month"].map(MONTHS),
            day=df["arrival_date_day_of_month"],
        )
    )
    # lead_time is days between booking and arrival, so this reconstructs the
    # booking timestamp the extract does not carry directly.
    df["booking_date"] = df["arrival_date"] - pd.to_timedelta(df["lead_time"], unit="D")

    # --- known data defects ----------------------------------------------
    df["children"] = df["children"].fillna(0).astype(int)
    df["country"] = df["country"].fillna("UNK").astype(str)
    # agent/company are identifiers, not magnitudes; NULL means "none involved".
    for col in ("agent", "company"):
        df[col] = df[col].fillna("NONE").astype(str).str.replace(r"\.0$", "", regex=True)

    lo, hi = cfg.data.adr_clip
    n_clipped = int(((df["adr"] < lo) | (df["adr"] > hi)).sum())
    df["adr"] = df["adr"].clip(lo, hi)
    log.info("clipped adr into [%s, %s] for %d rows", lo, hi, n_clipped)

    df["total_guests"] = df["adults"] + df["children"] + df["babies"]
    n_zero = int((df["total_guests"] == 0).sum())
    if n_zero:
        log.info(_ZERO_GUEST_MSG, n_zero)
        df = df[df["total_guests"] > 0]

    df["total_nights"] = df["stays_in_week_nights"] + df["stays_in_weekend_nights"]

    # --- deterministic arrival-queue order --------------------------------
    # The backtest asks "what if we had stopped accepting after the k-th
    # booking for this date?", so bookings need a stable order. Ties on
    # booking_date are common; a seeded jitter breaks them reproducibly
    # without favouring whatever order the CSV happened to arrive in.
    rng = np.random.default_rng(cfg.seed)
    df = df.assign(_tiebreak=rng.random(len(df)))
    df = df.sort_values(["hotel", "arrival_date", "booking_date", "_tiebreak"])
    df["booking_seq"] = df.groupby(["hotel", "arrival_date"]).cumcount()
    df = df.drop(columns="_tiebreak").reset_index(drop=True)
    df.insert(0, "booking_id", np.arange(len(df), dtype=np.int64))

    df["split"] = assign_split(df["arrival_date"], cfg)
    return df


def assign_split(arrival: pd.Series, cfg: Config) -> pd.Series:
    """Label each booking train/valid/test by ARRIVAL date."""
    out = pd.Series("unused", index=arrival.index, dtype="object")
    for name in ("train", "valid", "test"):
        lo, hi = getattr(cfg.split, name)
        mask = (arrival >= pd.Timestamp(lo)) & (arrival <= pd.Timestamp(hi))
        out[mask] = name
    return out


def build_processed(cfg: Config | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    df = clean(load_raw(cfg), cfg)
    dest = cfg.resolve(cfg.data.processed_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest, index=False)
    log.info("wrote %s  (%d rows x %d cols)", dest, *df.shape)
    log.info("split sizes: %s", df["split"].value_counts().to_dict())
    return df


def load_processed(cfg: Config | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    path = cfg.resolve(cfg.data.processed_path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run `overbook prepare` first.")
    return pd.read_parquet(path)


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build_processed()

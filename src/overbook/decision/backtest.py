"""Replay the authorisation decision over held-out arrival dates.

What makes this a fair test
---------------------------
Every policy is replayed against the bookings that genuinely existed for each
arrival date, and each booking's realised outcome is known, so "what if we had
stopped accepting after the k-th booking" is answerable without simulating
anything: dropping a booking simply removes a known outcome from the queue.

The standing assumption is **no demand substitution** — a guest turned away
does not rebook a different date at the same hotel, and no walk-in replaces
them. That biases the comparison *against* aggressive policies, so it does not
flatter the method being proposed. It is restated in the README limitations.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from overbook.config import Config
from overbook.decision.policy import (
    AcceptAll,
    DateState,
    Distributional,
    Economics,
    ExpectedArrivals,
    FixedRate,
    MeanRate,
    NoOverbook,
    Oracle,
    Policy,
    realised_profit,
)

log = logging.getLogger(__name__)


def estimate_capacity(train_keys: pd.DataFrame, quantile: float) -> dict[str, int]:
    """Infer each hotel's nightly capacity from realised arrivals.

    The extract has no room inventory, so capacity has to be inferred. We take
    a high quantile of realised arrivals per night in the TRAINING window: a
    hotel is assumed to have been roughly full on its busiest nights. The
    quantile is a config knob because the whole comparison is sensitive to it,
    and ``reports/sensitivity_capacity.csv`` sweeps it.
    """
    shows = (
        train_keys[train_keys["is_canceled"] == 0]
        .groupby(["hotel", "arrival_date"])
        .size()
        .rename("arrivals")
        .reset_index()
    )
    cap = shows.groupby("hotel")["arrivals"].quantile(quantile).round().astype(int)
    out = cap.to_dict()
    log.info("estimated capacity at q=%.2f: %s", quantile, out)
    return out


def build_states(
    keys: pd.DataFrame, probs: np.ndarray, capacity: dict[str, int]
) -> list[DateState]:
    """One :class:`DateState` per hotel-night, bookings in queue order."""
    df = keys.copy()
    df["p_cancel"] = probs
    df = df.sort_values(["hotel", "arrival_date", "booking_seq"])
    states = []
    for (hotel, date), g in df.groupby(["hotel", "arrival_date"], sort=True):
        states.append(
            DateState(
                hotel=str(hotel),
                arrival_date=pd.Timestamp(date),
                capacity=int(capacity[hotel]),
                cancel_probs=g["p_cancel"].to_numpy(dtype=float),
                showed=(1 - g["is_canceled"].to_numpy(dtype=int)),
                adr_mean=float(g["adr"].mean()),
            )
        )
    return states


def default_policies(cfg: Config, econ: Economics, pooled_cancel_rate: float) -> list[Policy]:
    policies: list[Policy] = [NoOverbook()]
    policies += [FixedRate(r) for r in cfg.backtest.fixed_overbook_rates if r > 0]
    policies += [
        MeanRate(pooled_cancel_rate),
        AcceptAll(),
        ExpectedArrivals(),
        Distributional(econ),
        Oracle(),
    ]
    return policies


def run_backtest(states: list[DateState], policies: list[Policy], econ: Economics) -> pd.DataFrame:
    """Long-format results: one row per (date, policy)."""
    rows = []
    for st in states:
        for pol in policies:
            a = pol.limit(st)
            arrivals = int(st.showed[:a].sum())
            rows.append(
                {
                    "hotel": st.hotel,
                    "arrival_date": st.arrival_date,
                    "policy": pol.name,
                    "capacity": st.capacity,
                    "bookings_on_books": st.n_bookings,
                    "adr_mean": st.adr_mean,
                    "authorised": a,
                    "arrivals": arrivals,
                    "served": min(arrivals, st.capacity),
                    "walked": max(arrivals - st.capacity, 0),
                    "unsold": max(st.capacity - arrivals, 0),
                    "profit": realised_profit(st, a, econ),
                }
            )
    return pd.DataFrame(rows)


def summarise(results: pd.DataFrame, baseline: str = "no_overbook") -> pd.DataFrame:
    """Per-policy totals, plus the headline per-room-night comparison."""
    g = results.groupby("policy")
    summary = g.agg(
        nights=("profit", "size"),
        total_profit=("profit", "sum"),
        mean_profit_per_night=("profit", "mean"),
        rooms=("capacity", "sum"),
        served=("served", "sum"),
        walked=("walked", "sum"),
        unsold=("unsold", "sum"),
        mean_authorised=("authorised", "mean"),
    )
    summary["occupancy"] = summary["served"] / summary["rooms"]
    summary["profit_per_room_night"] = summary["total_profit"] / summary["rooms"]
    summary["walks_per_100_nights"] = 100.0 * summary["walked"] / summary["nights"]
    summary["nights_oversold_pct"] = (
        100.0 * results.assign(o=results["walked"] > 0).groupby("policy")["o"].mean()
    )

    base_total = summary.loc[baseline, "total_profit"]
    summary["uplift_vs_baseline"] = summary["total_profit"] - base_total
    summary["uplift_pct"] = 100.0 * summary["uplift_vs_baseline"] / abs(base_total)

    if "oracle" in summary.index:
        gap = summary.loc["oracle", "total_profit"] - base_total
        summary["pct_of_oracle_gap_closed"] = (
            100.0 * summary["uplift_vs_baseline"] / gap if gap else np.nan
        )
    return summary.sort_values("total_profit", ascending=False)


def bootstrap_uplift(
    results: pd.DataFrame,
    policy: str,
    baseline: str,
    n_samples: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """Percentile CI for the per-night profit difference, resampling NIGHTS.

    Nights are the independent unit here; bookings within a night share a
    decision, so resampling bookings would understate the uncertainty.
    """
    wide = results.pivot_table(index=["hotel", "arrival_date"], columns="policy", values="profit")
    diff = (wide[policy] - wide[baseline]).to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_samples, diff.size))
    draws = diff[idx].mean(axis=1)
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return {
        "policy": policy,
        "baseline": baseline,
        "nights": int(diff.size),
        "mean_uplift_per_night": float(diff.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_positive": float((draws > 0).mean()),
        "total_uplift": float(diff.sum()),
    }

"""Drift diagnostics between the training window and a later window.

A model trained on 2015-2016 arrivals and deployed against 2017 arrivals is
already an out-of-distribution problem: the market segment mix moves, lead
times stretch, and the cancellation rate itself trends. These functions
quantify that rather than leaving it as an excuse for a disappointing test
score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

EPS = 1e-6

#: Rule of thumb from credit-risk practice, kept explicit so it can be argued with.
PSI_BANDS = ((0.10, "stable"), (0.25, "moderate shift"), (np.inf, "major shift"))


def psi_numeric(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """Population stability index using quantile bins from the reference sample."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if expected.size == 0 or actual.size == 0:
        return float("nan")
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)))
    if edges.size < 3:  # effectively constant
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, bins=edges)[0] / expected.size
    a = np.histogram(actual, bins=edges)[0] / actual.size
    e, a = np.clip(e, EPS, None), np.clip(a, EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_categorical(expected: pd.Series, actual: pd.Series) -> float:
    e = expected.astype(str).value_counts(normalize=True)
    a = actual.astype(str).value_counts(normalize=True)
    cats = e.index.union(a.index)
    ev = np.clip(e.reindex(cats).fillna(0.0).to_numpy(), EPS, None)
    av = np.clip(a.reindex(cats).fillna(0.0).to_numpy(), EPS, None)
    return float(np.sum((av - ev) * np.log(av / ev)))


def band(psi: float) -> str:
    for threshold, label in PSI_BANDS:
        if psi < threshold:
            return label
    return "major shift"


def drift_report(
    reference: pd.DataFrame, current: pd.DataFrame, categorical: list[str]
) -> pd.DataFrame:
    """PSI for every feature, plus a KS test for the numeric ones."""
    rows = []
    for col in reference.columns:
        if col not in current.columns:
            continue
        if col in categorical or str(reference[col].dtype) in {"object", "category"}:
            psi = psi_categorical(reference[col], current[col])
            ks_stat, ks_p = np.nan, np.nan
        else:
            psi = psi_numeric(reference[col].to_numpy(), current[col].to_numpy())
            ref = reference[col].dropna().to_numpy()
            cur = current[col].dropna().to_numpy()
            ks_stat, ks_p = ks_2samp(ref, cur) if ref.size and cur.size else (np.nan, np.nan)
        rows.append(
            {
                "feature": col,
                "kind": "categorical" if col in categorical else "numeric",
                "psi": psi,
                "band": band(psi) if np.isfinite(psi) else "n/a",
                "ks_stat": ks_stat,
                "ks_pvalue": ks_p,
            }
        )
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def monthly_stability(keys: pd.DataFrame, probs: np.ndarray) -> pd.DataFrame:
    """Observed vs predicted cancellation rate by arrival month.

    Calibration drifting apart over time is the failure mode that quietly
    breaks the decision layer, so it is tracked as a time series, not a single
    aggregate number.
    """
    df = keys[["arrival_date", "is_canceled"]].copy()
    df["p"] = probs
    df["month"] = df["arrival_date"].dt.to_period("M").astype(str)
    out = df.groupby("month").agg(
        n=("is_canceled", "size"),
        observed=("is_canceled", "mean"),
        predicted=("p", "mean"),
    )
    out["gap"] = out["predicted"] - out["observed"]
    return out.reset_index()

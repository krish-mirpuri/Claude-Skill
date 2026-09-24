"""Evaluation metrics, with calibration treated as a first-class result.

Ranking metrics (AUC) say whether the model orders bookings correctly.
The overbooking decision does not consume an ordering — it consumes
probabilities, and multiplies them by money. A model with excellent AUC and a
20% probability bias will happily recommend the wrong authorisation level, so
Brier score and expected calibration error are reported alongside AUC and are
the metrics the decision layer is actually judged on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

EPS = 1e-15


def expected_calibration_error(
    y: np.ndarray, p: np.ndarray, n_bins: int = 20
) -> tuple[float, float]:
    """Equal-frequency ECE and maximum calibration error.

    Equal-frequency (not equal-width) bins keep every bin populated, which
    matters here because predictions pile up at both ends.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    bins = np.array_split(order, n_bins)
    gaps, weights = [], []
    for b in bins:
        if len(b) == 0:
            continue
        gaps.append(abs(y[b].mean() - p[b].mean()))
        weights.append(len(b))
    gaps_arr = np.asarray(gaps)
    w = np.asarray(weights, dtype=float)
    return float((gaps_arr * w).sum() / w.sum()), float(gaps_arr.max())


def classification_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    ece, mce = expected_calibration_error(y, p)
    single_class = len(np.unique(y)) < 2
    return {
        "n": int(len(y)),
        "cancel_rate": float(y.mean()),
        "mean_pred": float(p.mean()),
        "roc_auc": float("nan") if single_class else float(roc_auc_score(y, p)),
        "pr_auc": float("nan") if single_class else float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "ece": ece,
        "mce": mce,
        # >1 means the model over-predicts cancellations in aggregate, which
        # translates directly into over-aggressive overbooking.
        "bias_ratio": float(p.mean() / max(y.mean(), EPS)),
    }


def reliability_table(y: np.ndarray, p: np.ndarray, n_bins: int = 20) -> pd.DataFrame:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    rows = []
    for i, b in enumerate(np.array_split(order, n_bins)):
        if len(b) == 0:
            continue
        rows.append(
            {
                "bin": i,
                "n": len(b),
                "mean_predicted": float(p[b].mean()),
                "observed_rate": float(y[b].mean()),
                "lo": float(p[b].min()),
                "hi": float(p[b].max()),
            }
        )
    return pd.DataFrame(rows)


def segment_metrics(
    y: np.ndarray, p: np.ndarray, groups: pd.Series, min_n: int = 200
) -> pd.DataFrame:
    """Per-segment metrics — an aggregate number can hide a broken segment."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p), "g": groups.to_numpy()})
    rows = []
    for g, sub in df.groupby("g", observed=True):
        if len(sub) < min_n:
            continue
        m = classification_metrics(sub["y"].to_numpy(), sub["p"].to_numpy())
        m["segment"] = str(g)
        rows.append(m)
    cols = ["segment", "n", "cancel_rate", "mean_pred", "roc_auc", "brier", "ece", "bias_ratio"]
    return pd.DataFrame(rows)[cols].sort_values("n", ascending=False).reset_index(drop=True)

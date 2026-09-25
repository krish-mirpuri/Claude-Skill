"""Probability calibration.

The decision layer multiplies probabilities by euros, so a probability that is
merely *well-ranked* is not good enough — it has to mean what it says. The
calibrator is fitted on the ``valid`` window, which no model has seen during
fitting or early stopping, and is then frozen.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

log = logging.getLogger(__name__)
EPS = 1e-6


class Calibrator:
    """Wraps a fitted 1-D probability->probability map."""

    def __init__(self, method: str = "isotonic"):
        if method not in {"isotonic", "sigmoid", "none"}:
            raise ValueError(f"unknown calibration method {method!r}")
        self.method = method
        self._model = None

    def fit(self, p: np.ndarray, y: np.ndarray) -> Calibrator:
        p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
        y = np.asarray(y, dtype=int)
        if self.method == "isotonic":
            self._model = IsotonicRegression(
                y_min=0.0, y_max=1.0, out_of_bounds="clip", increasing=True
            ).fit(p, y)
        elif self.method == "sigmoid":
            self._model = LogisticRegression(C=1e6, solver="lbfgs").fit(_logit(p).reshape(-1, 1), y)
        log.info("fitted %s calibrator on %d rows", self.method, len(p))
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
        if self.method == "none" or self._model is None:
            return p
        if self.method == "isotonic":
            out = self._model.predict(p)
        else:
            out = self._model.predict_proba(_logit(p).reshape(-1, 1))[:, 1]
        return np.clip(out, EPS, 1 - EPS)

    __call__ = predict


def _logit(p: np.ndarray) -> np.ndarray:
    return np.log(p / (1.0 - p))


def rolling_recalibrate(
    history: pd.DataFrame,
    target: pd.DataFrame,
    *,
    period: str = "M",
    method: str = "isotonic",
    p_col: str = "p_raw",
    y_col: str = "is_canceled",
    date_col: str = "arrival_date",
) -> pd.Series:
    """Refit the calibrator as outcomes arrive, using only the past.

    A calibrator fitted once on 2017-Q1 goes stale: the cancellation rate in
    this data climbs from 33% in the calibration window to 41% in the
    evaluation window, and a stale map under-predicts cancellations by about
    four points — which biases every overbooking decision downstream.

    A hotel does not have that problem in practice. By the time it authorises
    rooms for June, every booking that arrived in April and May has resolved.
    This function reproduces that: for each period it fits on outcomes from
    strictly earlier arrival dates only, scores the period, and then folds
    those outcomes into the history. Nothing from a period informs its own
    predictions.
    """
    target = target.sort_values(date_col)
    keys = target[date_col].dt.to_period(period)
    hist = history[[p_col, y_col]].copy()
    out = pd.Series(index=target.index, dtype=float)

    for key in keys.unique():
        mask = (keys == key).to_numpy()
        block = target.loc[mask]
        cal = Calibrator(method).fit(hist[p_col].to_numpy(), hist[y_col].to_numpy())
        out.loc[block.index] = cal.predict(block[p_col].to_numpy())
        hist = pd.concat([hist, block[[p_col, y_col]]], ignore_index=True)

    log.info(
        "rolling recalibration over %d %s-periods (history grew %d -> %d rows)",
        len(keys.unique()),
        period,
        len(history),
        len(hist),
    )
    return out.reindex(history.index.union(target.index)).loc[target.index]

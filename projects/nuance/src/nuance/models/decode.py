"""Turning per-label scores into predicted label sets.

Most multi-label papers report a number produced by thresholding at 0.5 and
say nothing further about it. 0.5 is not a neutral choice — it is a *fitted
parameter* that happens to have been fitted by convention — and for a label
with 77 training positives it is nowhere near optimal.

The natural fix is to tune a threshold per label. On this dataset that fix
**makes things worse**, because the tuning set has thirteen positives for
``grief`` and a threshold fitted on thirteen examples is noise. Cross-fitting
the tuning on out-of-fold training predictions repairs the overfit. All the
strategies are implemented here so the comparison can be measured rather than
argued about.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from nuance.evaluation.metrics import f1_from_counts

log = logging.getLogger(__name__)

STRATEGIES = ("flat", "global_tuned", "per_label_dev", "per_label_oof")


@dataclass
class Decoder:
    """A fitted decision rule: per-label thresholds plus a decoding policy."""

    thresholds: np.ndarray
    strategy: str = "flat"
    ensure_at_least_one: bool = False
    fitted_on: str = ""
    meta: dict = field(default_factory=dict)

    def decode(self, P: np.ndarray) -> np.ndarray:
        P = np.asarray(P, dtype=float)
        B = (P >= self.thresholds).astype(np.int8)
        if self.ensure_at_least_one:
            # Every comment in GoEmotions carries at least one label, so an
            # empty prediction is guaranteed wrong. Falling back to the
            # arg-max costs nothing and recovers those rows.
            empty = B.sum(axis=1) == 0
            if empty.any():
                B[empty, np.argmax(P[empty], axis=1)] = 1
        return B

    def __len__(self) -> int:
        return len(self.thresholds)


def _best_threshold(y: np.ndarray, p: np.ndarray, grid: np.ndarray) -> float:
    """Threshold maximising F1 for one label, ties broken toward 0.5.

    Breaking ties toward the middle matters for rare labels, where long
    stretches of the grid give identical F1 and picking the edge is an
    arbitrary commitment to an extreme.
    """
    B = p[None, :] >= grid[:, None]
    tp = (B & (y == 1)).sum(axis=1).astype(float)
    fp = (B & (y == 0)).sum(axis=1).astype(float)
    fn = ((~B) & (y == 1)).sum(axis=1).astype(float)
    scores = f1_from_counts(tp, fp, fn)
    best = scores.max()
    candidates = grid[scores >= best - 1e-12]
    return float(candidates[np.argmin(np.abs(candidates - 0.5))])


def tune_per_label(Y: np.ndarray, P: np.ndarray, grid: np.ndarray) -> np.ndarray:
    Y = np.asarray(Y, dtype=np.int8)
    return np.array([_best_threshold(Y[:, j], P[:, j], grid) for j in range(Y.shape[1])])


def tune_global(Y: np.ndarray, P: np.ndarray, grid: np.ndarray) -> float:
    """One threshold shared by every label, chosen on micro-F1."""
    Y = np.asarray(Y, dtype=np.int8)
    best, best_t = -1.0, 0.5
    for t in grid:
        B = (P >= t).astype(np.int8)
        tp = float((Y & B).sum())
        fp = float(((1 - Y) & B).sum())
        fn = float((Y & (1 - B)).sum())
        score = float(f1_from_counts(tp, fp, fn))
        if score > best:
            best, best_t = score, float(t)
    return best_t


def build_decoders(
    *,
    n_labels: int,
    grid: np.ndarray,
    Y_dev: np.ndarray,
    P_dev: np.ndarray,
    Y_oof: np.ndarray,
    P_oof: np.ndarray,
    ensure_at_least_one: bool = False,
) -> dict[str, Decoder]:
    """Every threshold strategy, fitted where it is entitled to look."""
    decoders = {
        "flat": Decoder(
            np.full(n_labels, 0.5), "flat", ensure_at_least_one, "nothing (the convention)"
        ),
        "global_tuned": Decoder(
            np.full(n_labels, tune_global(Y_dev, P_dev, grid)),
            "global_tuned",
            ensure_at_least_one,
            "dev split",
        ),
        "per_label_dev": Decoder(
            tune_per_label(Y_dev, P_dev, grid),
            "per_label_dev",
            ensure_at_least_one,
            "dev split",
        ),
        "per_label_oof": Decoder(
            tune_per_label(Y_oof, P_oof, grid),
            "per_label_oof",
            ensure_at_least_one,
            "out-of-fold training predictions",
        ),
    }
    for name, d in decoders.items():
        d.meta = {
            "min": float(d.thresholds.min()),
            "max": float(d.thresholds.max()),
            "median": float(np.median(d.thresholds)),
            "tuning_positives": int(np.asarray(Y_dev).sum())
            if "dev" in d.fitted_on
            else (int(np.asarray(Y_oof).sum()) if "out-of-fold" in d.fitted_on else 0),
        }
        log.debug("decoder %s: %s", name, d.meta)
    return decoders

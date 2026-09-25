"""Multi-label metrics, with the uncertainty attached.

The habit this module exists to break is reporting a single macro-F1 for a
28-label benchmark whose rarest label has six positive examples in the test
set. That number is not a measurement; it is a measurement plus a coin flip,
and the coin is large enough to cover the gap between most published methods.

So every metric here comes with a bootstrap interval, comparisons between
models are **paired** (same resample applied to both, which is far tighter and
far more honest than eyeballing two overlapping intervals), and labels without
enough support to be estimated are flagged rather than quietly averaged in.

Implementation note: F1 depends on the data only through per-label true
positive / false positive / false negative counts, and those counts are linear
in the rows. So a bootstrap resample is a *weighted sum* of per-row
contributions, and 2,000 resamples become one matrix multiply instead of 2,000
passes over the data. That is what makes it affordable to bootstrap everything
rather than only the headline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

EPS = 1e-12
_CHUNK = 500  # bootstrap draws per matrix multiply


# --------------------------------------------------------------------------
# counts
# --------------------------------------------------------------------------
def row_contributions(Y: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, ...]:
    """Per-row, per-label ``(tp, fp, fn)`` indicators."""
    Y = np.asarray(Y, dtype=np.int8)
    B = np.asarray(B, dtype=np.int8)
    if Y.shape != B.shape:
        raise ValueError(f"shape mismatch: truth {Y.shape} vs predictions {B.shape}")
    return (
        (Y & B).astype(np.float32),
        ((1 - Y) & B).astype(np.float32),
        (Y & (1 - B)).astype(np.float32),
    )


def f1_from_counts(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> np.ndarray:
    """Per-label F1. A label with no positives and no predictions scores 0."""
    return 2 * tp / np.maximum(2 * tp + fp + fn, EPS)


def macro_f1_from_counts(tp, fp, fn, mask: np.ndarray | None = None) -> np.ndarray:
    per = f1_from_counts(tp, fp, fn)
    if mask is not None:
        per = per[..., mask]
    return per.mean(axis=-1)


def micro_f1_from_counts(tp, fp, fn, mask: np.ndarray | None = None) -> np.ndarray:
    if mask is not None:
        tp, fp, fn = tp[..., mask], fp[..., mask], fn[..., mask]
    TP, FP, FN = tp.sum(-1), fp.sum(-1), fn.sum(-1)
    return 2 * TP / np.maximum(2 * TP + FP + FN, EPS)


# --------------------------------------------------------------------------
# bootstrap
# --------------------------------------------------------------------------
@dataclass
class BootstrapCounts:
    """Resampled ``(tp, fp, fn)`` of shape ``(n_boot, n_labels)``.

    Holding the resample *weights* rather than regenerating them lets two
    models be compared on identical resamples — the paired comparison.
    """

    tp: np.ndarray
    fp: np.ndarray
    fn: np.ndarray
    weights: np.ndarray  # (n_boot, n_rows)


def bootstrap_weights(n_rows: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Multinomial resample weights — the same thing as sampling rows with
    replacement, in a form a BLAS call can consume."""
    p = np.full(n_rows, 1.0 / n_rows)
    return rng.multinomial(n_rows, p, size=n_boot).astype(np.float32)


def resample_counts(Y, B, weights: np.ndarray) -> BootstrapCounts:
    TP, FP, FN = row_contributions(Y, B)
    out = []
    for M in (TP, FP, FN):
        parts = [weights[i : i + _CHUNK] @ M for i in range(0, len(weights), _CHUNK)]
        out.append(np.vstack(parts))
    return BootstrapCounts(*out, weights=weights)


def interval(draws: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2], axis=0)
    return float(lo), float(hi)


# --------------------------------------------------------------------------
# reports
# --------------------------------------------------------------------------
def measurable_mask(support: np.ndarray, min_support: int) -> np.ndarray:
    return np.asarray(support) >= min_support


def headline(
    Y: np.ndarray,
    B: np.ndarray,
    *,
    min_support: int = 50,
    n_boot: int = 2000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> dict:
    """Headline metrics, each with a bootstrap interval.

    ``macro_f1_measurable`` restricts the macro average to labels with enough
    test positives to estimate. The gap between it and plain ``macro_f1`` is
    the part of the usual headline that is carried by labels nobody can score.
    """
    rng = rng or np.random.default_rng(0)
    Y = np.asarray(Y, dtype=np.int8)
    B = np.asarray(B, dtype=np.int8)
    support = Y.sum(axis=0)
    mask = measurable_mask(support, min_support)

    TP, FP, FN = (m.sum(axis=0) for m in row_contributions(Y, B))
    boot = resample_counts(Y, B, bootstrap_weights(len(Y), n_boot, rng))

    def both(point, draws, name):
        lo, hi = interval(draws, alpha)
        return {
            name: float(point),
            f"{name}_ci_low": lo,
            f"{name}_ci_high": hi,
            f"{name}_ci_width": hi - lo,
        }

    out: dict = {
        "n": int(len(Y)),
        "n_labels": int(Y.shape[1]),
        "n_labels_measurable": int(mask.sum()),
        "min_support_measurable": int(min_support),
        "label_cardinality": float(Y.sum(axis=1).mean()),
    }
    out |= both(
        macro_f1_from_counts(TP, FP, FN),
        macro_f1_from_counts(boot.tp, boot.fp, boot.fn),
        "macro_f1",
    )
    out |= both(
        micro_f1_from_counts(TP, FP, FN),
        micro_f1_from_counts(boot.tp, boot.fp, boot.fn),
        "micro_f1",
    )
    if mask.any():
        out |= both(
            macro_f1_from_counts(TP, FP, FN, mask),
            macro_f1_from_counts(boot.tp, boot.fp, boot.fn, mask),
            "macro_f1_measurable",
        )
    out["subset_accuracy"] = float((Y == B).all(axis=1).mean())
    out["hamming_loss"] = float((Y != B).mean())
    out["mean_labels_predicted"] = float(B.sum(axis=1).mean())
    return out


def per_label_table(
    Y: np.ndarray,
    B: np.ndarray,
    names: list[str],
    *,
    min_support: int = 50,
    n_boot: int = 2000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """One row per label: support, precision, recall, F1 and its interval."""
    rng = rng or np.random.default_rng(0)
    Y = np.asarray(Y, dtype=np.int8)
    B = np.asarray(B, dtype=np.int8)
    TP, FP, FN = (m.sum(axis=0) for m in row_contributions(Y, B))
    boot = resample_counts(Y, B, bootstrap_weights(len(Y), n_boot, rng))
    f1_draws = f1_from_counts(boot.tp, boot.fp, boot.fn)
    lo, hi = np.quantile(f1_draws, [alpha / 2, 1 - alpha / 2], axis=0)

    support = Y.sum(axis=0)
    return (
        pd.DataFrame(
            {
                "label": names,
                "support": support,
                "predicted": B.sum(axis=0),
                "precision": TP / np.maximum(TP + FP, EPS),
                "recall": TP / np.maximum(TP + FN, EPS),
                "f1": f1_from_counts(TP, FP, FN),
                "f1_ci_low": lo,
                "f1_ci_high": hi,
                "f1_ci_width": hi - lo,
                "measurable": measurable_mask(support, min_support),
            }
        )
        .sort_values("support", ascending=False)
        .reset_index(drop=True)
    )


def paired_comparison(
    Y: np.ndarray,
    B_a: np.ndarray,
    B_b: np.ndarray,
    *,
    statistic: str = "macro_f1",
    min_support: int = 50,
    n_boot: int = 2000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> dict:
    """Paired bootstrap of ``a − b`` on the same resamples.

    Two systems evaluated on the same test set are *not* independent, so
    comparing their separate confidence intervals throws away most of the
    power. Resampling both on identical draws cancels the shared variance and
    routinely turns "the intervals overlap, who knows" into a clear answer.
    """
    rng = rng or np.random.default_rng(0)
    Y = np.asarray(Y, dtype=np.int8)
    mask = measurable_mask(Y.sum(axis=0), min_support) if statistic.endswith("measurable") else None
    fn_ = macro_f1_from_counts if statistic.startswith("macro") else micro_f1_from_counts

    weights = bootstrap_weights(len(Y), n_boot, rng)
    a, b = resample_counts(Y, B_a, weights), resample_counts(Y, B_b, weights)
    draws = fn_(a.tp, a.fp, a.fn, mask) - fn_(b.tp, b.fp, b.fn, mask)

    point_a = fn_(*[m.sum(axis=0) for m in row_contributions(Y, B_a)], mask)
    point_b = fn_(*[m.sum(axis=0) for m in row_contributions(Y, B_b)], mask)
    lo, hi = interval(draws, alpha)
    return {
        "statistic": statistic,
        "a": float(point_a),
        "b": float(point_b),
        "difference": float(point_a - point_b),
        "ci_low": lo,
        "ci_high": hi,
        "p_a_better": float((draws > 0).mean()),
        "significant": bool(lo > 0 or hi < 0),
    }

"""Where the errors actually land.

A macro-F1 of 0.45 sounds like a model that barely works. It is worth asking
what the mistakes *are* before believing that. If the model answers
``nervousness`` where the annotators said ``fear``, it has understood the
comment and disagreed about a word; if it answers ``gratitude``, it has not.

These functions split errors by whether they stay inside an Ekman family, and
build the label-vs-label confusion structure that multi-label problems do not
get for free (there is no single predicted class to cross-tabulate).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nuance.data.taxonomy import Taxonomy


def confusion_pairs(
    Y: np.ndarray, B: np.ndarray, names: list[str], top_k: int = 20
) -> pd.DataFrame:
    """Most frequent (true label, predicted-instead label) pairs.

    Restricted to rows where the true label was missed and something else was
    predicted, which is the multi-label analogue of an off-diagonal cell.
    """
    Y = np.asarray(Y, dtype=np.int8)
    B = np.asarray(B, dtype=np.int8)
    missed = Y & (1 - B)  # label was there, we did not say it
    added = (1 - Y) & B  # we said it, it was not there
    counts = missed.T @ added  # (true, predicted-instead)
    rows = []
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            if i != j and counts[i, j]:
                rows.append(
                    {"true": names[i], "predicted_instead": names[j], "count": int(counts[i, j])}
                )
    return (
        pd.DataFrame(rows).sort_values("count", ascending=False).head(top_k).reset_index(drop=True)
    )


def family_error_split(
    Y: np.ndarray,
    B: np.ndarray,
    names: list[str],
    taxonomy: Taxonomy,
    view: str = "ekman",
) -> dict:
    """Share of substitution errors that stay inside the same family."""
    Y = np.asarray(Y, dtype=np.int8)
    B = np.asarray(B, dtype=np.int8)
    families = np.array([taxonomy.family_of(n, view) for n in names])

    missed = Y & (1 - B)
    added = (1 - Y) & B
    counts = missed.T @ added
    same = np.equal.outer(families, families)
    np.fill_diagonal(counts, 0)

    total = int(counts.sum())
    inside = int(counts[same].sum())

    # The other hypothesis worth testing: that the hard boundary is not
    # "which emotion" but "is there an emotion here at all". Annotators
    # disagree constantly about whether a flat comment carries anything, and
    # if that is where the errors live, no amount of finer modelling of the
    # emotion vocabulary will help.
    neutral_idx = names.index("neutral") if "neutral" in names else None
    involving_neutral = 0
    if neutral_idx is not None:
        involving_neutral = int(counts[neutral_idx, :].sum() + counts[:, neutral_idx].sum())

    return {
        "view": view,
        "substitution_errors": total,
        "within_family": inside,
        "share_within_family": float(inside / max(total, 1)),
        "involving_neutral": involving_neutral,
        "share_involving_neutral": float(involving_neutral / max(total, 1)),
    }


def coarse_gain(
    Y_fine: np.ndarray, B_fine: np.ndarray, taxonomy: Taxonomy, view: str = "ekman"
) -> dict:
    """What the same predictions score once collapsed to a coarser taxonomy.

    Nothing about the model changes — only the question being asked. The jump
    is a measure of how much of the fine-grained difficulty is the taxonomy
    rather than the text.
    """
    from nuance.evaluation.metrics import (
        f1_from_counts,
        macro_f1_from_counts,
        micro_f1_from_counts,
        row_contributions,
    )

    Yc = taxonomy.collapse(Y_fine, view)
    Bc = taxonomy.collapse(B_fine, view)
    tp, fp, fn = (m.sum(axis=0) for m in row_contributions(Yc, Bc))
    return {
        "view": view,
        "n_labels": int(Yc.shape[1]),
        "min_support": int(Yc.sum(axis=0).min()),
        "macro_f1": float(macro_f1_from_counts(tp, fp, fn)),
        "micro_f1": float(micro_f1_from_counts(tp, fp, fn)),
        "per_label": dict(
            zip(taxonomy.names(view), f1_from_counts(tp, fp, fn).round(4).tolist(), strict=True)
        ),
    }

"""Finding labels that are probably wrong.

Implements confident learning (Northcutt, Jiang & Chuang, JAIR 2021) for the
multi-label case, treating each label as its own binary problem. The method is
simple and does not need a second set of annotators:

1. Get out-of-fold predictions, so no example's score comes from a model that
   trained on it.
2. For each label, take the *self-confidence* threshold — the mean predicted
   probability among the examples that carry that label. This adapts to how
   hard and how rare the label is instead of imposing a global cut-off.
3. Flag disagreements that clear the threshold confidently: examples the model
   is sure carry a label they were not given, and examples given a label the
   model is sure they do not carry.

The audit is written from scratch rather than pulled from a library because
the interesting part is step 2, and burying it behind an import would defeat
the purpose of the exercise.

A caveat this module cannot fix: with one model doing the flagging, "the
labels are wrong" and "the model is wrong in a consistent way" look identical.
That is why :func:`family_breakdown` exists — if the flags concentrate inside
emotion families, they are ambiguity rather than error.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from nuance.data.taxonomy import Taxonomy

log = logging.getLogger(__name__)

MISSING = "label probably missing"
SPURIOUS = "label probably spurious"


def self_confidence_thresholds(Y: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Per-label mean predicted probability among its positive examples."""
    Y = np.asarray(Y, dtype=bool)
    P = np.asarray(P, dtype=float)
    out = np.full(Y.shape[1], 0.5)
    for j in range(Y.shape[1]):
        if Y[:, j].any():
            out[j] = float(P[Y[:, j], j].mean())
    return out


def estimated_noise_rates(Y: np.ndarray, P: np.ndarray, names: list[str]) -> pd.DataFrame:
    """Per-label share of assignments the model confidently disputes."""
    Y = np.asarray(Y, dtype=np.int8)
    P = np.asarray(P, dtype=float)
    t = self_confidence_thresholds(Y, P)
    rows = []
    for j, name in enumerate(names):
        pos = Y[:, j] == 1
        confident = P[:, j] >= t[j]
        spurious = int((pos & ~confident).sum())
        missing = int((~pos & confident).sum())
        rows.append(
            {
                "label": name,
                "support": int(pos.sum()),
                "threshold": float(t[j]),
                "spurious": spurious,
                "missing": missing,
                "disputed_share_of_positives": spurious / max(int(pos.sum()), 1),
            }
        )
    return pd.DataFrame(rows).sort_values("disputed_share_of_positives", ascending=False)


def suspect_labels(
    Y: np.ndarray,
    P: np.ndarray,
    texts,
    names: list[str],
    *,
    top_k: int = 300,
) -> pd.DataFrame:
    """The most confidently disputed label assignments, ranked."""
    Y = np.asarray(Y, dtype=np.int8)
    P = np.asarray(P, dtype=float)
    texts = np.asarray(texts, dtype=object)
    t = self_confidence_thresholds(Y, P)

    margin = P - t  # positive => model is confident it applies
    given = Y == 1
    spurious = given & (margin < 0)
    missing = (~given) & (margin > 0)

    rows = []
    for kind, mask, score in (
        (SPURIOUS, spurious, -margin),
        (MISSING, missing, margin),
    ):
        idx, lab = np.nonzero(mask)
        for i, j in zip(idx, lab, strict=True):
            rows.append(
                {
                    "row": int(i),
                    "label": names[j],
                    "kind": kind,
                    "confidence": float(score[i, j]),
                    # Margins are not comparable across labels: a label whose
                    # self-confidence threshold sits at 0.89 produces far
                    # larger margins than one at 0.24, and ranking on the raw
                    # value hands every slot to a single label. Dividing by
                    # the headroom puts them on one scale.
                    "rank_score": float(
                        score[i, j] / max(t[j] if kind == SPURIOUS else 1 - t[j], 1e-6)
                    ),
                    "p": float(P[i, j]),
                    "threshold": float(t[j]),
                    "given_labels": ", ".join(names[k] for k in np.nonzero(Y[i])[0]),
                    "text": str(texts[i]),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = _balanced_head(out, top_k)
    log.info(
        "flagged %d suspected label errors (%d spurious, %d missing)",
        len(out),
        int((out["kind"] == SPURIOUS).sum()),
        int((out["kind"] == MISSING).sum()),
    )
    return out.reset_index(drop=True)


def _balanced_head(df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """Take the strongest flags while covering labels and both kinds.

    Straight top-k is dominated by whichever label has the most headroom, which
    makes the audit look like a single-label problem when it is not. Selecting
    round-robin over (kind, label) groups, strongest first within each group,
    keeps the ranking meaningful and the output representative.
    """
    df = df.sort_values("rank_score", ascending=False)
    df["within_group"] = df.groupby(["kind", "label"]).cumcount()
    return (
        df.sort_values(["within_group", "rank_score"], ascending=[True, False])
        .head(top_k)
        .sort_values(["within_group", "rank_score"], ascending=[True, False])
        .drop(columns="within_group")
    )


def family_breakdown(suspects: pd.DataFrame, taxonomy: Taxonomy) -> dict:
    """Do the flagged labels sit in the same emotion family as the given ones?

    A flag that swaps ``nervousness`` for ``fear`` is the taxonomy being
    fine-grained. A flag that swaps ``gratitude`` for ``anger`` is an error.
    Reporting the split is the difference between "the labels are noisy" and
    "the labels are noisy in a way that matters".
    """
    if suspects.empty:
        return {}
    missing = suspects[suspects["kind"] == MISSING]
    same_family = 0
    for _, row in missing.iterrows():
        family = taxonomy.family_of(row["label"])
        given = [g.strip() for g in row["given_labels"].split(",") if g.strip()]
        if any(taxonomy.family_of(g) == family for g in given):
            same_family += 1
    return {
        "flags_considered": int(len(missing)),
        "same_ekman_family": int(same_family),
        "share_same_family": float(same_family / max(len(missing), 1)),
    }


def rows_to_drop(Y: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Rows where the model confidently disagrees *and* offers a replacement.

    "Scored below its label's mean confidence" is far too weak on its own —
    roughly half of any label's positives sit below their own mean by
    construction, so that criterion flags half the corpus and means nothing.

    The confident-joint criterion is the one that carries information: a row
    qualifies only when a label it *was* given falls short of that label's
    threshold **and** some label it was *not* given clears its own. That is a
    substitution the model is confident about, not merely a low score.
    """
    Y = np.asarray(Y, dtype=np.int8)
    P = np.asarray(P, dtype=float)
    t = self_confidence_thresholds(Y, P)
    weak_given = ((Y == 1) & (P < t)).any(axis=1)
    confident_other = ((Y == 0) & (P >= t)).any(axis=1)
    return np.unique(np.nonzero(weak_given & confident_other)[0])

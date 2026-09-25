"""How much of the labelling budget actually bought anything.

43,410 comments were annotated to build this training set. The question a team
facing a new label taxonomy actually asks is not "what is the best possible
score" but "how many examples until it stops improving, and does choosing
which ones to label help?"

Two strategies, both starting from the same random seed set:

``random``       label a random batch each round — the honest baseline, and a
                 surprisingly hard one to beat.
``uncertainty``  label the comments the current model is least sure about.
                 For multi-label, a comment's uncertainty is summed across
                 labels, so a comment that is borderline on several labels
                 outranks one that is borderline on a single label.

Each budget is repeated over several seeds, because a single active-learning
curve is mostly noise and curves that cross once are not evidence of anything.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from nuance.config import Config
from nuance.evaluation.metrics import (
    macro_f1_from_counts,
    micro_f1_from_counts,
    row_contributions,
)
from nuance.models.train import build_model

log = logging.getLogger(__name__)


def uncertainty_scores(P: np.ndarray) -> np.ndarray:
    """Total binary uncertainty across labels: sum of ``min(p, 1-p)``."""
    P = np.asarray(P, dtype=float)
    return np.minimum(P, 1.0 - P).sum(axis=1)


def _score(model, texts, Y_true) -> tuple[float, float]:
    B = (model.predict_proba(texts) >= 0.5).astype(np.int8)
    tp, fp, fn = (m.sum(axis=0) for m in row_contributions(Y_true, B))
    return float(macro_f1_from_counts(tp, fp, fn)), float(micro_f1_from_counts(tp, fp, fn))


def run_curve(
    cfg: Config,
    train_texts: np.ndarray,
    Y_train: np.ndarray,
    test_texts: np.ndarray,
    Y_test: np.ndarray,
    *,
    strategy: str,
    seed: int,
    model_name: str = "word_logistic",
) -> list[dict]:
    """One active-learning run: grow the labelled pool through the budgets."""
    rng = np.random.default_rng(seed)
    n = len(train_texts)
    budgets = sorted(b for b in cfg.active.budgets if b <= n)

    labelled = rng.choice(n, size=budgets[0], replace=False)
    rows: list[dict] = []
    model = None
    for i, budget in enumerate(budgets):
        if i > 0:
            pool = np.setdiff1d(np.arange(n), labelled, assume_unique=False)
            extra = budget - len(labelled)
            if strategy == "random":
                pick = rng.choice(pool, size=min(extra, len(pool)), replace=False)
            elif strategy == "uncertainty":
                if model is None:  # pragma: no cover - the seed round always fits one
                    raise RuntimeError("uncertainty sampling needs a fitted model")
                scores = uncertainty_scores(model.predict_proba(train_texts[pool]))
                pick = pool[np.argsort(scores)[::-1][:extra]]
            else:
                raise ValueError(f"unknown strategy {strategy!r}")
            labelled = np.concatenate([labelled, pick])

        model = build_model(model_name, cfg).fit(train_texts[labelled], Y_train[labelled])
        macro, micro = _score(model, test_texts, Y_test)
        rows.append(
            {
                "strategy": strategy,
                "seed": seed,
                "budget": int(len(labelled)),
                "macro_f1": macro,
                "micro_f1": micro,
                "labels_seen": int(Y_train[labelled].sum()),
                "rarest_label_positives": int(Y_train[labelled].sum(axis=0).min()),
            }
        )
        log.info("%-11s seed=%d budget=%-6d macro=%.4f", strategy, seed, len(labelled), macro)
    return rows


def run_simulation(
    cfg: Config, train_texts, Y_train, test_texts, Y_test, model_name="word_logistic"
) -> pd.DataFrame:
    rows = []
    for strategy in cfg.active.strategies:
        for s in range(cfg.active.seeds):
            rows += run_curve(
                cfg,
                np.asarray(train_texts, dtype=object),
                np.asarray(Y_train),
                np.asarray(test_texts, dtype=object),
                np.asarray(Y_test),
                strategy=strategy,
                seed=cfg.seed + s,
                model_name=model_name,
            )
    return pd.DataFrame(rows)


def budget_to_reach(curve: pd.DataFrame, target_fraction: float = 0.95) -> pd.DataFrame:
    """Smallest budget reaching ``target_fraction`` of the full-data score."""
    out = []
    for strategy, g in curve.groupby("strategy"):
        mean = g.groupby("budget")["macro_f1"].mean()
        full = mean.loc[mean.index.max()]
        target = target_fraction * full
        reached = mean[mean >= target]
        out.append(
            {
                "strategy": strategy,
                "full_data_macro_f1": float(full),
                "target": float(target),
                "budget_reached": int(reached.index.min()) if len(reached) else None,
                "share_of_corpus": float(reached.index.min() / mean.index.max())
                if len(reached)
                else None,
            }
        )
    return pd.DataFrame(out)

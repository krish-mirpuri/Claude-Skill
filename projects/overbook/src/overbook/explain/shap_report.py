"""SHAP explanations, at two scales.

Global
    Mean |SHAP| per feature, to state what the model leans on rather than
    guess from a split-count importance (which rewards high-cardinality
    columns like ``agent`` regardless of usefulness).

Local
    The handful of drivers behind one booking's score, returned by the API so
    a revenue manager overriding a recommendation can see why it was made.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def _explainer(booster):
    import shap  # imported lazily: heavy, and optional for the core pipeline

    return shap.TreeExplainer(booster)


def shap_values(booster, X: pd.DataFrame, sample: int | None = 5000, seed: int = 0):
    """SHAP values in log-odds space for a sample of rows."""
    if sample is not None and len(X) > sample:
        X = X.sample(sample, random_state=seed)
    values = _explainer(booster).shap_values(X)
    if isinstance(values, list):  # older SHAP returns one array per class
        values = values[1]
    return np.asarray(values), X


def global_importance(booster, X: pd.DataFrame, sample: int | None = 5000) -> pd.Series:
    values, Xs = shap_values(booster, X, sample=sample)
    return (
        pd.Series(np.abs(values).mean(axis=0), index=Xs.columns)
        .sort_values(ascending=False)
        .rename("mean_abs_shap")
    )


def explain_one(booster, x: pd.DataFrame, top_n: int = 5) -> list[dict]:
    """Top drivers for a single row, signed toward cancellation."""
    values, _ = shap_values(booster, x, sample=None)
    row = values[0]
    order = np.argsort(np.abs(row))[::-1][:top_n]
    out = []
    for i in order:
        col = x.columns[i]
        val = x.iloc[0, i]
        out.append(
            {
                "feature": str(col),
                "value": (
                    None if pd.isna(val) else (val.item() if hasattr(val, "item") else str(val))
                ),
                "shap": float(row[i]),
                "direction": "increases cancellation risk"
                if row[i] > 0
                else "decreases cancellation risk",
            }
        )
    return out

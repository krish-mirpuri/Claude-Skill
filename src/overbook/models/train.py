"""Model training with forward-only validation.

Three models, in increasing order of ambition, so the gain from each step is
visible rather than asserted:

* ``prior``     — predict the training-window cancellation rate for everyone.
* ``logistic``  — one-hot + regularised logistic regression.
* ``lightgbm``  — gradient boosting with native categorical handling.

Validation discipline
---------------------
The ``valid`` window (2017-Q1) is reserved **entirely** for probability
calibration. Early stopping therefore cannot use it, or the calibrator would
be fitted on data the model had already peeked at. Instead the tail of the
training window is held out as an inner early-stopping set, the best iteration
count is recorded, and the final model is refitted on the whole training
window for that many rounds.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from overbook.config import Config, load_config
from overbook.features.build import FeatureFrame

log = logging.getLogger(__name__)

INNER_HOLDOUT_FRACTION = 0.15  # tail of the training window, by arrival date


class Predictor:
    """Picklable ``DataFrame -> P(cancel)`` callable.

    Deliberately a class rather than a closure: these objects are serialised
    with joblib and loaded by the API process, and a lambda cannot cross that
    boundary.
    """

    name = "predictor"

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def __call__(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_proba(X)


class PriorPredictor(Predictor):
    name = "prior"

    def __init__(self, rate: float):
        self.rate = float(rate)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.rate, dtype=float)


class SklearnPredictor(Predictor):
    name = "logistic"

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X)[:, 1]


class LightGBMPredictor(Predictor):
    name = "lightgbm"

    def __init__(self, booster: lgb.Booster):
        self.booster = booster

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.booster.predict(X), dtype=float)


@dataclass
class TrainedModel:
    name: str
    predictor: Predictor
    n_features: int
    metadata: dict

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.predictor.predict_proba(X)

    @property
    def booster(self) -> lgb.Booster | None:
        return getattr(self.predictor, "booster", None)


def rolling_origin_folds(arrival: pd.Series, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window folds cut on arrival date.

    Fold *k* trains on the first ``k+1`` blocks of dates and validates on the
    next one. Unlike ``KFold`` this never lets a later booking inform an
    earlier prediction, and unlike ``TimeSeriesSplit`` on row order it splits
    on real dates, so all bookings for one arrival date stay together.
    """
    dates = np.sort(arrival.unique())
    if n_splits < 1 or len(dates) < n_splits + 1:
        raise ValueError(f"cannot build {n_splits} folds from {len(dates)} dates")
    edges = np.array_split(dates, n_splits + 1)
    folds = []
    for k in range(n_splits):
        tr_dates = np.concatenate(edges[: k + 1])
        va_dates = edges[k + 1]
        folds.append(
            (
                np.flatnonzero(arrival.isin(tr_dates).to_numpy()),
                np.flatnonzero(arrival.isin(va_dates).to_numpy()),
            )
        )
    return folds


def _inner_holdout(ff: FeatureFrame) -> tuple[np.ndarray, np.ndarray]:
    """Split the training window into (fit, early-stopping) by arrival date."""
    dates = np.sort(ff.keys["arrival_date"].unique())
    cut = dates[int(len(dates) * (1 - INNER_HOLDOUT_FRACTION))]
    is_tail = (ff.keys["arrival_date"] >= cut).to_numpy()
    return np.flatnonzero(~is_tail), np.flatnonzero(is_tail)


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------
def train_prior(ff: FeatureFrame) -> TrainedModel:
    rate = float(ff.y.mean())
    return TrainedModel(
        name="prior",
        predictor=PriorPredictor(rate),
        n_features=0,
        metadata={"train_cancel_rate": rate},
    )


def train_logistic(ff: FeatureFrame, cfg: Config) -> TrainedModel:
    cats = ff.categorical
    nums = [c for c in ff.X.columns if c not in cats]
    pre = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
                ),
                nums,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", min_frequency=25, sparse_output=True),
                cats,
            ),
        ]
    )
    pipe = Pipeline(
        [
            ("pre", pre),
            (
                "clf",
                LogisticRegression(max_iter=2000, C=0.5, solver="lbfgs", random_state=cfg.seed),
            ),
        ]
    )
    pipe.fit(ff.X, ff.y)
    return TrainedModel(
        name="logistic",
        predictor=SklearnPredictor(pipe),
        n_features=ff.X.shape[1],
        metadata={"C": 0.5, "n_numeric": len(nums), "n_categorical": len(cats)},
    )


def train_lightgbm(ff: FeatureFrame, cfg: Config) -> TrainedModel:
    params = dict(cfg.model.lightgbm)
    n_rounds = int(params.pop("num_boost_round", 1000))
    stopping = int(params.pop("early_stopping_rounds", 50))
    params.setdefault("seed", cfg.seed)
    params.setdefault("deterministic", True)
    params.setdefault("num_threads", 4)

    fit_idx, stop_idx = _inner_holdout(ff)
    cats = ff.categorical

    dtrain = lgb.Dataset(ff.X.iloc[fit_idx], label=ff.y.iloc[fit_idx], categorical_feature=cats)
    dstop = lgb.Dataset(ff.X.iloc[stop_idx], label=ff.y.iloc[stop_idx], reference=dtrain)
    probe = lgb.train(
        params,
        dtrain,
        num_boost_round=n_rounds,
        valid_sets=[dstop],
        valid_names=["inner_holdout"],
        callbacks=[lgb.early_stopping(stopping, verbose=False), lgb.log_evaluation(0)],
    )
    best = probe.best_iteration or n_rounds
    log.info("lightgbm early stopping chose %d rounds", best)

    # Refit on the entire training window for the chosen number of rounds.
    dfull = lgb.Dataset(ff.X, label=ff.y, categorical_feature=cats)
    booster = lgb.train(params, dfull, num_boost_round=best, callbacks=[lgb.log_evaluation(0)])

    return TrainedModel(
        name="lightgbm",
        predictor=LightGBMPredictor(booster),
        n_features=ff.X.shape[1],
        metadata={"best_iteration": int(best), **{k: v for k, v in params.items()}},
    )


def cross_validate_lightgbm(ff: FeatureFrame, cfg: Config) -> pd.DataFrame:
    """Out-of-fold predictions over the training window (rolling origin)."""
    from overbook.models.evaluate import classification_metrics

    folds = rolling_origin_folds(ff.keys["arrival_date"], cfg.model.n_splits)
    params = dict(cfg.model.lightgbm)
    n_rounds = int(params.pop("num_boost_round", 1000))
    stopping = int(params.pop("early_stopping_rounds", 50))
    params.setdefault("seed", cfg.seed)
    params.setdefault("num_threads", 4)

    rows = []
    for k, (tr, va) in enumerate(folds, start=1):
        dtr = lgb.Dataset(ff.X.iloc[tr], label=ff.y.iloc[tr], categorical_feature=ff.categorical)
        dva = lgb.Dataset(ff.X.iloc[va], label=ff.y.iloc[va], reference=dtr)
        b = lgb.train(
            params,
            dtr,
            num_boost_round=n_rounds,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(stopping, verbose=False), lgb.log_evaluation(0)],
        )
        p = b.predict(ff.X.iloc[va], num_iteration=b.best_iteration)
        m = classification_metrics(ff.y.iloc[va].to_numpy(), p)
        m.update(
            fold=k,
            n_train=len(tr),
            n_valid=len(va),
            valid_from=str(ff.keys["arrival_date"].iloc[va].min().date()),
            valid_to=str(ff.keys["arrival_date"].iloc[va].max().date()),
        )
        rows.append(m)
        log.info("fold %d: auc=%.4f  brier=%.4f", k, m["roc_auc"], m["brier"])
    return pd.DataFrame(rows)


def save_artifacts(
    model: TrainedModel, feature_builder, cfg: Config, extra: dict | None = None
) -> Path:
    cfg.ensure_dirs()
    out = cfg.resolve(cfg.paths.models_dir)
    joblib.dump(model.predictor, out / "predictor.joblib")
    joblib.dump(feature_builder, out / "feature_builder.joblib")
    if model.booster is not None:
        # Plain-text dump as well: it survives library upgrades that can
        # invalidate a pickle, and it is readable during an incident.
        model.booster.save_model(str(out / "model_lgbm.txt"))
    card = {
        "model": model.name,
        "n_features": model.n_features,
        "feature_view": feature_builder.view,
        "metadata": model.metadata,
        **(extra or {}),
    }
    (out / "model_card.json").write_text(json.dumps(card, indent=2, default=str))
    log.info("saved model artifacts to %s", out)
    return out


def load_artifacts(cfg: Config | None = None):
    cfg = cfg or load_config()
    out = cfg.resolve(cfg.paths.models_dir)
    predictor = joblib.load(out / "predictor.joblib")
    feature_builder = joblib.load(out / "feature_builder.joblib")
    calibrator_path = out / "calibrator.joblib"
    calibrator = joblib.load(calibrator_path) if calibrator_path.exists() else None
    return predictor, feature_builder, calibrator

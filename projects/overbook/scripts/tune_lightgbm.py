"""Hyperparameter search, selected on rolling-origin CV inside the training window.

Run this to reproduce the LightGBM block in ``conf/config.yaml``. Selection
uses the training window only — the validation window is reserved for
calibration and the test window is never consulted, so the numbers in
``reports/RESULTS.md`` remain an honest out-of-sample estimate.

    python scripts/tune_lightgbm.py --out reports/tuning.csv
"""

from __future__ import annotations

import argparse
import itertools
import logging
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from overbook.config import load_config
from overbook.data.clean import load_processed
from overbook.features.build import build_features
from overbook.models.evaluate import classification_metrics
from overbook.models.train import rolling_origin_folds

log = logging.getLogger("tune")

GRID = {
    "num_leaves": [15, 31, 63],
    "min_data_in_leaf": [50, 200],
    "learning_rate": [0.03, 0.05],
    "lambda_l2": [1.0, 10.0],
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="reports/tuning.csv")
    ap.add_argument("--max-rounds", type=int, default=1500)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    warnings.filterwarnings("ignore", category=UserWarning)

    cfg = load_config()
    ff, _ = build_features(load_processed(cfg), cfg)
    train = ff.split("train")
    folds = rolling_origin_folds(train.keys["arrival_date"], cfg.model.n_splits)

    base = {
        "objective": "binary",
        "verbosity": -1,
        "seed": cfg.seed,
        "num_threads": 4,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
    }
    keys = list(GRID)
    rows = []
    for combo in itertools.product(*(GRID[k] for k in keys)):
        params = {**base, **dict(zip(keys, combo, strict=True))}
        losses, aucs, iters = [], [], []
        for tr_idx, va_idx in folds:
            dtr = lgb.Dataset(
                train.X.iloc[tr_idx],
                label=train.y.iloc[tr_idx],
                categorical_feature=train.categorical,
            )
            dva = lgb.Dataset(train.X.iloc[va_idx], label=train.y.iloc[va_idx], reference=dtr)
            booster = lgb.train(
                params,
                dtr,
                num_boost_round=args.max_rounds,
                valid_sets=[dva],
                callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
            )
            pred = booster.predict(train.X.iloc[va_idx], num_iteration=booster.best_iteration)
            m = classification_metrics(train.y.iloc[va_idx].to_numpy(), pred)
            losses.append(m["log_loss"])
            aucs.append(m["roc_auc"])
            iters.append(booster.best_iteration)
        rows.append(
            {
                **dict(zip(keys, combo, strict=True)),
                "cv_log_loss": float(np.mean(losses)),
                "cv_roc_auc": float(np.mean(aucs)),
                "mean_best_iteration": int(np.mean(iters)),
            }
        )
        log.info(
            "%s -> logloss %.5f auc %.5f",
            dict(zip(keys, combo, strict=True)),
            rows[-1]["cv_log_loss"],
            rows[-1]["cv_roc_auc"],
        )

    out = pd.DataFrame(rows).sort_values("cv_log_loss").reset_index(drop=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(out.head(10).to_string(index=False))
    print(f"\nbest: {out.iloc[0].to_dict()}\nwritten to {args.out}")


if __name__ == "__main__":
    main()

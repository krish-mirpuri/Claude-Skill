"""Serving context: the state the model needs that a request cannot carry.

Two of the ``arrival_eve`` features are rolling cancellation rates computed
from realised outcomes. A live scoring request has no outcomes, so the values
have to come from somewhere else: this module snapshots the most recent value
per hotel at training time and the API reads it back. Refreshing this snapshot
is what a nightly retrain would do in production.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from overbook.config import Config, load_config
from overbook.data.clean import load_processed
from overbook.features.build import _rolling_hotel_cancel_rate

log = logging.getLogger(__name__)
CONTEXT_FILE = "serving_context.json"


def build_serving_context(cfg: Config | None = None) -> dict:
    cfg = cfg or load_config()
    df = load_processed(cfg)
    ctx: dict = {"hotels": {}, "as_of": str(df["arrival_date"].max().date())}
    for w in cfg.features.rolling_cancel_windows:
        df[f"hotel_cancel_rate_{w}d"] = _rolling_hotel_cancel_rate(df, w)
    latest = df.sort_values("arrival_date").groupby("hotel").tail(1)
    for _, row in latest.iterrows():
        ctx["hotels"][str(row["hotel"])] = {
            f"hotel_cancel_rate_{w}d": float(row[f"hotel_cancel_rate_{w}d"])
            for w in cfg.features.rolling_cancel_windows
        }
    # A reasonable default nightly rate, used when a request names a hotel we
    # have no history for.
    ctx["fallback"] = {
        f"hotel_cancel_rate_{w}d": float(df[f"hotel_cancel_rate_{w}d"].mean())
        for w in cfg.features.rolling_cancel_windows
    }
    ctx["capacity"] = _capacity_snapshot(cfg, df)
    path = cfg.resolve(cfg.paths.models_dir) / CONTEXT_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ctx, indent=2))
    log.info("wrote serving context to %s", path)
    return ctx


def _capacity_snapshot(cfg: Config, df: pd.DataFrame) -> dict:
    from overbook.decision.backtest import estimate_capacity

    train = df[df["split"] == "train"]
    return {k: int(v) for k, v in estimate_capacity(train, cfg.economics.capacity_quantile).items()}


def load_serving_context(cfg: Config | None = None) -> dict:
    cfg = cfg or load_config()
    path = cfg.resolve(cfg.paths.models_dir) / CONTEXT_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `overbook train` (or `overbook serving-context`)."
        )
    return json.loads(Path(path).read_text())

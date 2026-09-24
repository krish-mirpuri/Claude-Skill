"""Typed configuration loaded from ``conf/config.yaml``.

Every tunable that can change a reported number lives in the YAML file so a
result can always be traced back to the exact settings that produced it.
"""

from __future__ import annotations

import datetime as dt
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "conf" / "config.yaml"


class DataConfig(BaseModel):
    source_url: str
    sha256: str
    raw_path: Path
    processed_path: Path
    drop_exact_duplicates: bool = False
    adr_clip: tuple[float, float] = (5.0, 1000.0)


class SplitConfig(BaseModel):
    train: tuple[dt.date, dt.date]
    valid: tuple[dt.date, dt.date]
    test: tuple[dt.date, dt.date]

    @field_validator("train", "valid", "test", mode="before")
    @classmethod
    def _parse(cls, v):
        return tuple(dt.date.fromisoformat(str(x)) for x in v)

    def check_ordering(self) -> None:
        """Fail loudly if the windows overlap or run backwards."""
        windows = [("train", self.train), ("valid", self.valid), ("test", self.test)]
        for name, (lo, hi) in windows:
            if lo > hi:
                raise ValueError(f"{name} window runs backwards: {lo} > {hi}")
        for (a_name, (_, a_hi)), (b_name, (b_lo, _)) in zip(windows, windows[1:], strict=False):
            if a_hi >= b_lo:
                raise ValueError(f"{a_name} window overlaps {b_name}: {a_hi} >= {b_lo}")


class FeatureConfig(BaseModel):
    view: str = "arrival_eve"
    min_country_freq: int = 50
    rolling_cancel_windows: list[int] = Field(default_factory=lambda: [28, 90])

    @field_validator("view")
    @classmethod
    def _known_view(cls, v: str) -> str:
        if v not in {"arrival_eve", "booking_time"}:
            raise ValueError(f"unknown feature view {v!r}")
        return v


class ModelConfig(BaseModel):
    n_splits: int = 4
    calibration: str = "isotonic"
    calibration_refresh: str | None = "M"
    lightgbm: dict = Field(default_factory=dict)


class DecisionConfig(BaseModel):
    overdispersion: bool = True
    quadrature_nodes: int = 7


class EconomicsConfig(BaseModel):
    """Cost assumptions. The dataset has no cost columns; these are inputs."""

    variable_cost_ratio: float = 0.25
    walk_cost_multiplier: float = 2.0
    capacity_quantile: float = 0.90
    sensitivity_walk_multipliers: list[float] = Field(default_factory=list)

    @field_validator("variable_cost_ratio")
    @classmethod
    def _unit_interval(cls, v: float) -> float:
        if not 0.0 <= v < 1.0:
            raise ValueError("variable_cost_ratio must be in [0, 1)")
        return v


class BacktestConfig(BaseModel):
    bootstrap_samples: int = 2000
    fixed_overbook_rates: list[float] = Field(default_factory=list)
    capacity_quantiles: list[float] = Field(default_factory=list)


class PathsConfig(BaseModel):
    models_dir: Path
    reports_dir: Path
    figures_dir: Path


class Config(BaseModel):
    seed: int
    data: DataConfig
    split: SplitConfig
    features: FeatureConfig
    model: ModelConfig
    decision: DecisionConfig = Field(default_factory=DecisionConfig)
    economics: EconomicsConfig
    backtest: BacktestConfig
    paths: PathsConfig

    def resolve(self, p: Path | str) -> Path:
        """Interpret a configured path relative to the project root."""
        p = Path(p)
        return p if p.is_absolute() else PROJECT_ROOT / p

    def ensure_dirs(self) -> None:
        for p in (self.paths.models_dir, self.paths.reports_dir, self.paths.figures_dir):
            self.resolve(p).mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=4)
def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> Config:
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    cfg = Config.model_validate(raw)
    cfg.split.check_ordering()
    return cfg

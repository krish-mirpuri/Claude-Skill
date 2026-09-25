"""Typed configuration loaded from ``conf/config.yaml``."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml
from pydantic import BaseModel, Field, field_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "conf" / "config.yaml"


class WordFeatures(BaseModel):
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 2
    sublinear_tf: bool = True


class CharFeatures(BaseModel):
    enabled: bool = True
    ngram_range: tuple[int, int] = (3, 5)
    min_df: int = 3
    max_features: int | None = 300_000


class FeatureConfig(BaseModel):
    word: WordFeatures = Field(default_factory=WordFeatures)
    char: CharFeatures = Field(default_factory=CharFeatures)
    lowercase: bool = True


class DataConfig(BaseModel):
    goemotions_base: str
    goemotions_files: dict[str, str]
    twitter_url: str
    twitter_sha256: str
    raw_dir: Path
    processed_dir: Path


class ModelConfig(BaseModel):
    C: float = 4.0
    class_weight: str | None = "balanced"
    max_iter: int = 1000
    n_folds: int = 5


class ThresholdConfig(BaseModel):
    grid_start: float = 0.05
    grid_stop: float = 0.96
    grid_step: float = 0.01
    report: str = "flat"

    @property
    def grid(self) -> np.ndarray:
        return np.arange(self.grid_start, self.grid_stop, self.grid_step)


class EvaluationConfig(BaseModel):
    bootstrap_samples: int = 2000
    min_support_measurable: int = 50
    alpha: float = 0.05


class NoiseConfig(BaseModel):
    top_k: int = 300


class TransferConfig(BaseModel):
    strip_emoticons: bool = True
    max_per_class: int = 5000


class ActiveConfig(BaseModel):
    budgets: list[int] = Field(default_factory=list)
    strategies: list[str] = Field(default_factory=list)
    seeds: int = 3


class PathsConfig(BaseModel):
    models_dir: Path
    reports_dir: Path
    figures_dir: Path


class Config(BaseModel):
    seed: int
    data: DataConfig
    features: FeatureConfig
    model: ModelConfig
    thresholds: ThresholdConfig
    evaluation: EvaluationConfig
    noise: NoiseConfig
    transfer: TransferConfig
    active: ActiveConfig
    paths: PathsConfig

    @field_validator("seed")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("seed must be positive")
        return v

    def resolve(self, p: Path | str) -> Path:
        p = Path(p)
        return p if p.is_absolute() else PROJECT_ROOT / p

    def ensure_dirs(self) -> None:
        for p in (
            self.paths.models_dir,
            self.paths.reports_dir,
            self.paths.figures_dir,
            self.data.raw_dir,
            self.data.processed_dir,
        ):
            self.resolve(p).mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=4)
def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> Config:
    with open(path) as fh:
        return Config.model_validate(yaml.safe_load(fh))

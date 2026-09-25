"""Load the GoEmotions curated splits into text plus a binary label matrix."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from nuance.config import Config, load_config
from nuance.data.taxonomy import Taxonomy, load_taxonomy
from nuance.features.text import normalise

log = logging.getLogger(__name__)

SPLITS = ("train", "dev", "test")


@dataclass
class Split:
    name: str
    text: pd.Series  # normalised
    raw_text: pd.Series
    Y: np.ndarray  # (n, 28) binary
    ids: pd.Series

    def __len__(self) -> int:
        return len(self.text)

    def labels(self, taxonomy: Taxonomy, view: str) -> np.ndarray:
        return taxonomy.collapse(self.Y, view)


@dataclass
class Corpus:
    train: Split
    dev: Split
    test: Split
    taxonomy: Taxonomy

    def __getitem__(self, name: str) -> Split:
        if name not in SPLITS:
            raise KeyError(f"unknown split {name!r}")
        return getattr(self, name)

    def support(self, view: str = "fine") -> pd.DataFrame:
        """Positives per label in each split — the table that decides what is
        measurable at all."""
        rows = {}
        for name in SPLITS:
            rows[name] = self[name].labels(self.taxonomy, view).sum(axis=0)
        return pd.DataFrame(rows, index=self.taxonomy.names(view))


def _read_split(path: Path, n_labels: int) -> tuple[pd.DataFrame, np.ndarray]:
    # QUOTE_NONE: the comments contain bare quote characters, and letting the
    # CSV reader treat them as quoting silently merges rows.
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["text", "label_ids", "comment_id"],
        quoting=3,
        dtype=str,
        keep_default_na=False,
    )
    Y = np.zeros((len(df), n_labels), dtype=np.int8)
    for row, ids in enumerate(df["label_ids"]):
        for i in ids.split(","):
            Y[row, int(i)] = 1
    if not Y.any(axis=1).all():
        raise ValueError(f"{path.name}: some rows carry no label")
    return df, Y


def load_corpus(cfg: Config | None = None) -> Corpus:
    cfg = cfg or load_config()
    raw = cfg.resolve(cfg.data.raw_dir)
    taxonomy = load_taxonomy(raw)
    splits = {}
    for name in SPLITS:
        path = raw / f"{name}.tsv"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing — run `nuance download` first.")
        df, Y = _read_split(path, len(taxonomy.fine))
        splits[name] = Split(
            name=name,
            text=df["text"].map(normalise),
            raw_text=df["text"],
            Y=Y,
            ids=df["comment_id"],
        )
        log.info(
            "%-5s %6d comments, %.2f labels each, %d labels present",
            name,
            len(df),
            Y.sum() / len(df),
            int((Y.sum(axis=0) > 0).sum()),
        )
    return Corpus(taxonomy=taxonomy, **splits)

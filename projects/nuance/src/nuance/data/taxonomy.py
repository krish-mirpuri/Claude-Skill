"""The GoEmotions label taxonomy, and the two coarser views of it.

Three nested views of the same annotations:

``fine``       27 emotions + neutral, as annotated.
``ekman``      the six Ekman families + neutral, using Google's own mapping.
``sentiment``  positive / negative / ambiguous / neutral.

The coarse views are not a convenience. A quarter of the fine labels have
fewer than 100 test examples, which makes their per-label scores unmeasurable
(see :mod:`nuance.evaluation.metrics`). Collapsing is what turns the benchmark
back into something a number can be reported about — and the gap between the
two tells you how much of the fine-grained difficulty is genuine signal and
how much is taxonomy ambiguity.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

NEUTRAL = "neutral"
VIEWS = ("fine", "ekman", "sentiment")


@dataclass(frozen=True)
class Taxonomy:
    """Label names for each view, plus the matrices that collapse between them."""

    fine: list[str]
    ekman: list[str]
    sentiment: list[str]
    _ekman_map: dict[str, list[str]]
    _sentiment_map: dict[str, list[str]]

    # -- names ------------------------------------------------------------
    def names(self, view: str = "fine") -> list[str]:
        if view not in VIEWS:
            raise ValueError(f"unknown view {view!r}; expected one of {VIEWS}")
        return getattr(self, view)

    def index(self, name: str, view: str = "fine") -> int:
        return self.names(view).index(name)

    # -- collapsing --------------------------------------------------------
    def projection(self, view: str) -> np.ndarray:
        """``(n_fine, n_view)`` 0/1 matrix mapping fine labels onto ``view``."""
        if view == "fine":
            return np.eye(len(self.fine), dtype=np.int8)
        mapping = self._ekman_map if view == "ekman" else self._sentiment_map
        target = self.names(view)
        P = np.zeros((len(self.fine), len(target)), dtype=np.int8)
        for group, members in mapping.items():
            g = target.index(group)
            for m in members:
                P[self.fine.index(m), g] = 1
        P[self.fine.index(NEUTRAL), target.index(NEUTRAL)] = 1
        return P

    def collapse(self, Y: np.ndarray, view: str) -> np.ndarray:
        """Project a fine-grained label matrix onto a coarser view."""
        if view == "fine":
            return np.asarray(Y, dtype=np.int8)
        return (np.asarray(Y) @ self.projection(view) > 0).astype(np.int8)

    def family_of(self, fine_label: str, view: str = "ekman") -> str:
        """Which coarse group a fine label belongs to."""
        if fine_label == NEUTRAL:
            return NEUTRAL
        mapping = self._ekman_map if view == "ekman" else self._sentiment_map
        for group, members in mapping.items():
            if fine_label in members:
                return group
        raise KeyError(f"{fine_label!r} is in no {view} group")


def load_taxonomy(raw_dir: Path) -> Taxonomy:
    raw_dir = Path(raw_dir)
    fine = (raw_dir / "emotions.txt").read_text().split()
    ekman_map = json.loads((raw_dir / "ekman_mapping.json").read_text())
    sentiment_map = json.loads((raw_dir / "sentiment_mapping.json").read_text())

    covered = {m for members in ekman_map.values() for m in members} | {NEUTRAL}
    missing = set(fine) - covered
    if missing:
        raise ValueError(f"emotions absent from the Ekman mapping: {sorted(missing)}")

    return Taxonomy(
        fine=fine,
        ekman=[*sorted(ekman_map), NEUTRAL],
        sentiment=[*sorted(sentiment_map), NEUTRAL],
        _ekman_map=ekman_map,
        _sentiment_map=sentiment_map,
    )

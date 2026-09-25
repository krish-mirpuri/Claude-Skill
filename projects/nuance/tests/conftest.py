"""Shared fixtures.

The suite builds its own miniature corpus with the real file formats, so CI
needs no network and no 20 MB of downloads. Tests that genuinely need trained
artifacts are marked ``slow`` and skip when they are absent.
"""

from __future__ import annotations

import json

import pytest

from nuance.config import load_config
from nuance.data.taxonomy import load_taxonomy

EMOTIONS = [
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
]
EKMAN = {
    "anger": ["anger", "annoyance", "disapproval"],
    "disgust": ["disgust"],
    "fear": ["fear", "nervousness"],
    "joy": [
        "joy",
        "amusement",
        "approval",
        "excitement",
        "gratitude",
        "love",
        "optimism",
        "relief",
        "pride",
        "admiration",
        "desire",
        "caring",
    ],
    "sadness": ["sadness", "disappointment", "embarrassment", "grief", "remorse"],
    "surprise": ["surprise", "realization", "confusion", "curiosity"],
}
SENTIMENT = {
    "positive": [
        "amusement",
        "excitement",
        "joy",
        "love",
        "desire",
        "optimism",
        "caring",
        "pride",
        "admiration",
        "gratitude",
        "relief",
        "approval",
    ],
    "negative": [
        "fear",
        "nervousness",
        "remorse",
        "embarrassment",
        "disappointment",
        "sadness",
        "grief",
        "disgust",
        "anger",
        "annoyance",
        "disapproval",
    ],
    "ambiguous": ["realization", "surprise", "curiosity", "confusion"],
}

#: Words that make a label predictable, so a tiny model can still learn something.
CUES = {
    "gratitude": "thanks",
    "anger": "furious",
    "joy": "delighted",
    "sadness": "heartbroken",
    "neutral": "the",
}


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def raw_dir(tmp_path_factory):
    """A miniature corpus written in the real on-disk formats."""
    d = tmp_path_factory.mktemp("raw")
    (d / "emotions.txt").write_text("\n".join(EMOTIONS) + "\n")
    (d / "ekman_mapping.json").write_text(json.dumps(EKMAN))
    (d / "sentiment_mapping.json").write_text(json.dumps(SENTIMENT))

    cued = list(CUES)
    for split, n in (("train", 900), ("dev", 200), ("test", 200)):
        lines = []
        for i in range(n):
            label = cued[i % len(cued)]
            j = EMOTIONS.index(label)
            text = f"{CUES[label]} this is comment number {i} about something"
            ids = str(j)
            if i % 17 == 0 and label != "neutral":  # a few genuine multi-label rows
                ids = f"{j},{EMOTIONS.index('neutral')}"
            lines.append(f"{text}\t{ids}\tid{split}{i}")
        (d / f"{split}.tsv").write_text("\n".join(lines) + "\n")
    return d


@pytest.fixture(scope="session")
def taxonomy(raw_dir):
    return load_taxonomy(raw_dir)


@pytest.fixture(scope="session")
def corpus(raw_dir, cfg, tmp_path_factory):
    from nuance.data.goemotions import load_corpus

    patched = cfg.model_copy(deep=True)
    patched.data.raw_dir = raw_dir
    return load_corpus(patched)


@pytest.fixture(scope="session")
def artifacts(cfg):
    """Trained artifacts, or skip."""
    import joblib

    d = cfg.resolve(cfg.paths.models_dir)
    if not (d / "model_card.json").exists():
        pytest.skip("no trained artifacts; run `make all` first")
    return (
        json.loads((d / "model_card.json").read_text()),
        joblib.load(d / "champion.joblib"),
        joblib.load(d / "decoders.joblib"),
    )

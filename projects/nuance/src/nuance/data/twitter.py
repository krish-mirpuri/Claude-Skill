"""The out-of-domain evaluation set, and the trap inside it.

NLTK's ``twitter_samples`` corpus is 5,000 "positive" and 5,000 "negative"
tweets. Those labels were not annotated by anyone — they come from *distant
supervision*: tweets containing ``:)`` went in one pile and tweets containing
``:(`` went in the other. The emoticon that produced the label is still sitting
in the text.

Evaluate a sentiment model on it as-is and you are largely measuring whether
the model can find a smiley, which every bag-of-words model can do perfectly.
The number looks excellent and means nothing. This loader strips emoticons by
default, and ``nuance.transfer`` reports the score both ways so the size of the
illusion is on the record rather than in a footnote.

Two further honesties about this set: the labels are noisy (sarcasm, mixed
sentiment), and it is drawn from July 2015, which is a different era as well as
a different platform from the Reddit comments the model is trained on.
"""

from __future__ import annotations

import json
import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from nuance.config import Config, load_config
from nuance.features.text import count_emoticons, normalise

log = logging.getLogger(__name__)

_FILES = {
    "positive": "twitter_samples/positive_tweets.json",
    "negative": "twitter_samples/negative_tweets.json",
}


@dataclass
class TransferSet:
    """Tweets with a binary sentiment label, in both stripped and raw form."""

    text: pd.Series  # normalised, emoticons removed
    text_with_emoticons: pd.Series
    sentiment: pd.Series  # "positive" / "negative"
    created_at: pd.Series
    n_emoticons: pd.Series

    def __len__(self) -> int:
        return len(self.text)

    def y(self, positive: str = "positive") -> np.ndarray:
        return (self.sentiment == positive).to_numpy().astype(np.int8)


def load_transfer_set(cfg: Config | None = None) -> TransferSet:
    cfg = cfg or load_config()
    path = cfg.resolve(cfg.data.raw_dir) / "twitter_samples.zip"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run `nuance download` first.")

    rows = []
    with zipfile.ZipFile(path) as z:
        for sentiment, member in _FILES.items():
            lines = z.read(member).decode("utf-8").splitlines()
            for line in lines[: cfg.transfer.max_per_class]:
                obj = json.loads(line)
                rows.append((obj["text"], sentiment, obj.get("created_at")))

    raw = pd.Series([r[0] for r in rows], dtype="string")
    out = TransferSet(
        text=raw.map(lambda t: normalise(t, strip_emoticons=True)),
        text_with_emoticons=raw.map(lambda t: normalise(t, strip_emoticons=False)),
        sentiment=pd.Series([r[1] for r in rows], dtype="string"),
        created_at=pd.Series([r[2] for r in rows], dtype="string"),
        n_emoticons=raw.map(count_emoticons),
    )
    empty = int((out.text.str.len() == 0).sum())
    log.info(
        "transfer set: %d tweets (%d positive), %.2f emoticons each, %d left empty after stripping",
        len(out),
        int((out.sentiment == "positive").sum()),
        float(out.n_emoticons.mean()),
        empty,
    )
    return out


def leakage_summary(ts: TransferSet) -> dict:
    """How separable are the classes using emoticons alone?

    This is the size of the shortcut a careless evaluation would be measuring.
    """
    pos = ts.sentiment == "positive"
    happy = ts.text_with_emoticons.str.contains(r":\)|:-\)|:D|=\)", regex=True)
    sad = ts.text_with_emoticons.str.contains(r":\(|:-\(|=\(", regex=True)
    guess = np.where(happy & ~sad, 1, np.where(sad & ~happy, 0, -1))
    decided = guess >= 0
    correct = (guess[decided] == pos.to_numpy()[decided].astype(int)).mean()
    return {
        "tweets": len(ts),
        "mean_emoticons_per_tweet": float(ts.n_emoticons.mean()),
        "share_decidable_by_emoticon_alone": float(decided.mean()),
        "accuracy_of_emoticon_rule_where_it_fires": float(correct),
        "accuracy_of_emoticon_rule_overall": float((guess == pos.to_numpy().astype(int)).mean()),
    }


def save_processed(ts: TransferSet, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "text": ts.text,
            "text_with_emoticons": ts.text_with_emoticons,
            "sentiment": ts.sentiment,
            "created_at": ts.created_at,
            "n_emoticons": ts.n_emoticons,
        }
    ).to_parquet(path, index=False)
    return path

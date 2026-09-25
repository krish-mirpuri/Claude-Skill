"""Does it survive leaving Reddit?

In-domain test scores answer a narrow question: how well does this model do on
more of exactly the data it was trained on? The moment anyone deploys it, the
text comes from somewhere else. So the model is trained on Reddit comments,
collapsed to positive/negative, and evaluated on tweets from July 2015 —
different platform, different length, different conventions, three years
apart.

Two reference points make the number interpretable:

* **VADER**, a rule-based sentiment lexicon tuned for social media. It never
  saw either training set, so it has no domain to lose.
* **The emoticon ablation.** NLTK's tweet labels came from emoticons still
  present in the text, so the same evaluation with emoticons left in shows
  exactly how good a leaky evaluation can be made to look.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from nuance.config import Config
from nuance.data.goemotions import Corpus
from nuance.data.twitter import TransferSet
from nuance.models.train import LinearModel

log = logging.getLogger(__name__)


def binary_sentiment_rows(corpus: Corpus, split: str) -> tuple[np.ndarray, np.ndarray]:
    """GoEmotions rows that are unambiguously positive or negative.

    Rows that are neutral, ambiguous, or *both* positive and negative are
    dropped: the target corpus has no such category, so scoring them would
    measure a mismatch in the label space rather than a mismatch in domain.
    """
    s = corpus[split]
    S = s.labels(corpus.taxonomy, "sentiment")
    names = corpus.taxonomy.names("sentiment")
    pos = S[:, names.index("positive")] == 1
    neg = S[:, names.index("negative")] == 1
    keep = pos ^ neg
    return s.text.to_numpy()[keep], pos[keep].astype(np.int8)


def _accuracy_and_f1(y: np.ndarray, pred: np.ndarray) -> dict:
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "n": int(len(y)),
        "accuracy": float((y == pred).mean()),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
    }


def vader_predictions(texts) -> np.ndarray:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyser = SentimentIntensityAnalyzer()
    return np.array(
        [1 if analyser.polarity_scores(str(t))["compound"] >= 0 else 0 for t in texts],
        dtype=np.int8,
    )


def run_transfer(cfg: Config, corpus: Corpus, tweets: TransferSet) -> pd.DataFrame:
    """The 2x2 that matters: {our model, VADER} x {in-domain, out-of-domain}."""
    X_train, y_train = binary_sentiment_rows(corpus, "train")
    X_test, y_test = binary_sentiment_rows(corpus, "test")
    log.info("sentiment training rows: %d (%.1f%% positive)", len(y_train), 100 * y_train.mean())

    model = LinearModel(cfg, use_char=True, name="sentiment_wordchar_logistic")
    model.fit(X_train, y_train.reshape(-1, 1))

    def predict(texts) -> np.ndarray:
        return (model.predict_proba(texts)[:, 0] >= 0.5).astype(np.int8)

    y_tweets = tweets.y()
    rows = [
        {
            "model": "wordchar logistic (trained on Reddit)",
            "evaluated_on": "Reddit (in-domain)",
            "emoticons": "n/a",
            **_accuracy_and_f1(y_test, predict(X_test)),
        },
        {
            "model": "wordchar logistic (trained on Reddit)",
            "evaluated_on": "Twitter (out-of-domain)",
            "emoticons": "stripped",
            **_accuracy_and_f1(y_tweets, predict(tweets.text)),
        },
        {
            "model": "wordchar logistic (trained on Reddit)",
            "evaluated_on": "Twitter (out-of-domain)",
            "emoticons": "left in (leaky)",
            **_accuracy_and_f1(y_tweets, predict(tweets.text_with_emoticons)),
        },
        {
            "model": "VADER lexicon (no training)",
            "evaluated_on": "Reddit (in-domain)",
            "emoticons": "n/a",
            **_accuracy_and_f1(y_test, vader_predictions(X_test)),
        },
        {
            "model": "VADER lexicon (no training)",
            "evaluated_on": "Twitter (out-of-domain)",
            "emoticons": "stripped",
            **_accuracy_and_f1(y_tweets, vader_predictions(tweets.text)),
        },
        {
            "model": "VADER lexicon (no training)",
            "evaluated_on": "Twitter (out-of-domain)",
            "emoticons": "left in (leaky)",
            **_accuracy_and_f1(y_tweets, vader_predictions(tweets.text_with_emoticons)),
        },
        {
            "model": "majority class",
            "evaluated_on": "Reddit (in-domain)",
            "emoticons": "n/a",
            **_accuracy_and_f1(y_test, np.full_like(y_test, int(y_train.mean() >= 0.5))),
        },
        {
            "model": "majority class",
            "evaluated_on": "Twitter (out-of-domain)",
            "emoticons": "n/a",
            **_accuracy_and_f1(y_tweets, np.ones_like(y_tweets)),
        },
    ]
    return pd.DataFrame(rows)

"""Normalisation, and the emoticon leak in the transfer corpus."""

from __future__ import annotations

import numpy as np
import pytest

from nuance.features.text import (
    build_vectoriser,
    count_emoticons,
    feature_names,
    normalise,
)


def test_urls_and_handles_are_masked():
    out = normalise("check https://example.com/x and ask @someone about /r/python")
    assert "[URL]" in out and "[NAME]" in out and "[SUBREDDIT]" in out
    assert "example.com" not in out


def test_whitespace_is_collapsed():
    assert normalise("  too    many\n\nspaces ") == "too many spaces"


def test_emoticons_survive_by_default():
    assert ":)" in normalise("nice one :)")


@pytest.mark.parametrize(
    "text", ["so happy :)", "gutted :-(", "wow :D", "love it <3", "hmm :/", "yay =)"]
)
def test_emoticons_are_removed_when_asked(text):
    assert count_emoticons(text) >= 1
    assert count_emoticons(normalise(text, strip_emoticons=True)) == 0


def test_stripping_leaves_the_words_alone():
    assert normalise("this is great :)", strip_emoticons=True) == "this is great"


def test_stripping_does_not_eat_ordinary_punctuation():
    assert normalise("wait, what? (really)", strip_emoticons=True) == "wait, what? (really)"


def test_emoji_are_removed_too():
    assert count_emoticons("amazing 🎉🎉") == 2
    assert normalise("amazing 🎉🎉", strip_emoticons=True) == "amazing"


def test_the_leak_is_real_and_stripping_closes_it():
    """The transfer trap, in miniature.

    Labels derived from emoticons are perfectly recoverable from the text
    while the emoticons remain, and not recoverable at all once they are gone.
    """
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    words = ["the meeting", "that film", "this weather", "my commute", "the food"]
    texts, y = [], []
    for i in range(400):
        label = int(rng.random() < 0.5)
        # The words carry no signal at all; only the emoticon does.
        texts.append(f"{words[i % len(words)]} was something {':)' if label else ':('}")
        y.append(label)
    y = np.array(y)

    def accuracy(strip: bool) -> float:
        cleaned = [normalise(t, strip_emoticons=strip) for t in texts]
        from sklearn.feature_extraction.text import TfidfVectorizer

        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        X = vec.fit_transform(cleaned)
        model = LogisticRegression(max_iter=1000).fit(X[:300], y[:300])
        return float((model.predict(X[300:]) == y[300:]).mean())

    assert accuracy(strip=False) > 0.95, "with emoticons the task is trivial"
    assert accuracy(strip=True) < 0.65, "without them there is nothing left to learn"


DOCS = [f"comment number {i} about the same sort of thing entirely" for i in range(20)]


def test_vectoriser_unions_word_and_char_features(cfg):
    features = cfg.features.model_copy(deep=True)
    features.char.enabled = True
    vec = build_vectoriser(features)
    vec.fit(DOCS)
    names = feature_names(vec)
    assert any(n.startswith("word:") for n in names)
    assert any(n.startswith("char:") for n in names)


def test_char_features_can_be_turned_off(cfg):
    features = cfg.features.model_copy(deep=True)
    features.char.enabled = False
    vec = build_vectoriser(features)
    vec.fit(DOCS)
    assert not any(":" in n for n in feature_names(vec))

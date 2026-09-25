"""Text normalisation and the vectoriser.

Both corpora go through the *same* normalisation. That matters for the
transfer experiment: if Reddit comments arrive with names masked as ``[NAME]``
and tweets arrive with raw ``@handles``, part of the measured domain gap is
just preprocessing, and the experiment stops being about domain at all.
"""

from __future__ import annotations

import re

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

from nuance.config import FeatureConfig

_URL = re.compile(r"https?://\S+|www\.\S+")
_MENTION = re.compile(r"(?<![\w/])@\w{2,}")
_SUBREDDIT = re.compile(r"(?<![\w/])/?r/\w+")
_WHITESPACE = re.compile(r"\s+")

#: Emoticons carry the label in NLTK's tweet sample, so the transfer loader
#: needs to be able to remove them. Ordered longest-first so ``:-)`` is matched
#: before ``:-``.
EMOTICON = re.compile(
    r"(?:[<>]?[:;=8xX][\-o\*'^]?[\)\]\(\[dDpPoO/\\\|\}\{@3]+)"
    r"|(?:[\)\]\(\[dDpP/\\\|\}\{@3]+[\-o\*'^]?[:;=8xX][<>]?)"
    r"|(?:<3)|(?:\^_?\^)|(?:[oO0]_[oO0])|(?:T_T)|(?:;_;)"
)
#: Broad unicode emoji ranges — same reasoning.
EMOJI = re.compile("[\U0001f300-\U0001faff\U00002600-\U000027bf\U0001f1e6-\U0001f1ff⬀-⯿️]")


def normalise(text: str, *, strip_emoticons: bool = False) -> str:
    """Canonical form shared by every corpus.

    GoEmotions already masks people as ``[NAME]`` and subreddits as
    ``[RELIGION]``-style placeholders; this brings other corpora into line.
    """
    t = str(text)
    t = _URL.sub(" [URL] ", t)
    t = _MENTION.sub(" [NAME] ", t)
    t = _SUBREDDIT.sub(" [SUBREDDIT] ", t)
    if strip_emoticons:
        t = EMOTICON.sub(" ", t)
        t = EMOJI.sub(" ", t)
    return _WHITESPACE.sub(" ", t).strip()


def count_emoticons(text: str) -> int:
    return len(EMOTICON.findall(str(text))) + len(EMOJI.findall(str(text)))


def build_vectoriser(cfg: FeatureConfig) -> FeatureUnion | TfidfVectorizer:
    """Word n-grams, optionally unioned with character n-grams.

    Character n-grams are not decoration here: the text is short, misspelled
    and emphatic (``sooo``, ``AAAAH``), and word features throw all of that
    away.
    """
    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=tuple(cfg.word.ngram_range),
        min_df=cfg.word.min_df,
        sublinear_tf=cfg.word.sublinear_tf,
        lowercase=cfg.lowercase,
        strip_accents="unicode",
    )
    if not cfg.char.enabled:
        return word
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=tuple(cfg.char.ngram_range),
        min_df=cfg.char.min_df,
        max_features=cfg.char.max_features,
        sublinear_tf=True,
        lowercase=cfg.lowercase,
    )
    return FeatureUnion([("word", word), ("char", char)])


def feature_names(vectoriser) -> list[str]:
    if isinstance(vectoriser, FeatureUnion):
        names: list[str] = []
        for prefix, vec in vectoriser.transformer_list:
            names += [f"{prefix}:{f}" for f in vec.get_feature_names_out()]
        return names
    return list(vectoriser.get_feature_names_out())


def ensure_csr(X) -> csr_matrix:
    return X if isinstance(X, csr_matrix) else csr_matrix(X)

"""The model ladder, and honest out-of-fold predictions.

Deliberately linear. On 43k short comments a TF-IDF linear model lands within
a couple of points of the transformer baseline the dataset authors published,
which is the point the project is making: on this benchmark the reported
differences between architectures are smaller than the uncertainty in the
measurement, so the interesting engineering is in the evaluation, not the
encoder.

Everything here is picklable — the API loads these objects directly.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

from nuance.config import Config
from nuance.features.text import build_vectoriser, feature_names

log = logging.getLogger(__name__)


class Model:
    """Common interface: fit on raw text, emit per-label probabilities."""

    name = "model"

    def fit(self, texts, Y):  # pragma: no cover - interface
        raise NotImplementedError

    def predict_proba(self, texts) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


class PriorModel(Model):
    """Predict each label's training base rate. The floor any model must clear."""

    name = "prior"

    def fit(self, texts, Y):
        self.rates_ = np.asarray(Y, dtype=float).mean(axis=0)
        return self

    def predict_proba(self, texts) -> np.ndarray:
        return np.tile(self.rates_, (len(texts), 1))


class LinearModel(Model):
    """TF-IDF features into a one-vs-rest linear classifier."""

    def __init__(
        self,
        cfg: Config,
        *,
        use_char: bool,
        estimator: str = "logistic",
        class_weight: str | None = "balanced",
        name: str | None = None,
    ):
        self.cfg = cfg
        self.use_char = use_char
        self.estimator = estimator
        self.class_weight = class_weight
        self.name = name or f"{'wordchar' if use_char else 'word'}_{estimator}"

    def _make_vectoriser(self):
        features = self.cfg.features.model_copy(deep=True)
        features.char.enabled = self.use_char
        return build_vectoriser(features)

    def _make_estimator(self):
        if self.estimator == "logistic":
            base = LogisticRegression(
                C=self.cfg.model.C,
                max_iter=self.cfg.model.max_iter,
                class_weight=self.class_weight,
            )
        elif self.estimator == "svm":
            # Squared-hinge SVM: a genuinely different inductive bias from
            # logistic regression, which is why it is in the ladder. Its
            # decision function is mapped to [0, 1] for decoding.
            base = LinearSVC(C=0.5, class_weight=self.class_weight, max_iter=5000)
        else:
            raise ValueError(f"unknown estimator {self.estimator!r}")
        return OneVsRestClassifier(base, n_jobs=-1)

    def fit(self, texts, Y):
        Y = np.asarray(Y)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        self.n_labels_ = int(Y.shape[1])
        self.vectoriser_ = self._make_vectoriser()
        X = self.vectoriser_.fit_transform(texts)
        self.clf_ = self._make_estimator().fit(X, Y)
        log.info("%s: %d features, %d labels", self.name, X.shape[1], self.n_labels_)
        return self

    def _as_label_columns(self, P: np.ndarray) -> np.ndarray:
        """Force the output to one column per label.

        Handed a single-column target, scikit-learn's one-vs-rest wrapper
        decides the problem is ordinary binary classification and returns
        ``[P(negative), P(positive)]`` instead of one column. Reading column 0
        then silently inverts every prediction — an accuracy of 0.13 where
        0.87 was expected, which looks like a broken model rather than a
        broken index. Normalising the shape here keeps the multi-label
        contract true for every label count.
        """
        P = np.asarray(P, dtype=float)
        if P.ndim == 1:
            # A 1-D result is one score per row for a single label, never one
            # row of many labels. Deciding by shape alone gets this backwards
            # for a batch of one, which is exactly what serving sends.
            P = P.reshape(-1, 1)
        if P.shape[1] == self.n_labels_:
            return P
        if self.n_labels_ == 1 and P.shape[1] == 2:
            positive = int(np.flatnonzero(np.asarray(self.clf_.classes_) == 1)[0])
            return P[:, [positive]]
        raise ValueError(f"{self.name}: got {P.shape[1]} score columns for {self.n_labels_} labels")

    def predict_proba(self, texts) -> np.ndarray:
        X = self.vectoriser_.transform(texts)
        if self.estimator == "logistic":
            return self._as_label_columns(self.clf_.predict_proba(X))
        scores = np.asarray(self.clf_.decision_function(X), dtype=float)
        if scores.ndim == 1:
            scores = scores.reshape(-1, 1)
        return self._as_label_columns(1.0 / (1.0 + np.exp(-scores)))

    # -- interpretability --------------------------------------------------
    def top_terms(self, label_index: int, k: int = 12) -> list[tuple[str, float]]:
        """The n-grams pushing a label up. A linear model can just be read."""
        coef = self.clf_.estimators_[label_index].coef_.ravel()
        names = feature_names(self.vectoriser_)
        idx = np.argsort(coef)[::-1][:k]
        return [(names[i], float(coef[i])) for i in idx]

    def explain(self, text: str, label_index: int, k: int = 6) -> list[dict]:
        """Per-term contributions for one prediction: coefficient x tf-idf.

        For a linear model this is the exact decomposition of the score, not
        an approximation of it.
        """
        X = self.vectoriser_.transform([text])
        coef = self.clf_.estimators_[label_index].coef_.ravel()
        names = feature_names(self.vectoriser_)
        _, cols = X.nonzero()
        contrib = [(names[c], float(X[0, c] * coef[c])) for c in cols]
        contrib.sort(key=lambda t: abs(t[1]), reverse=True)
        return [{"term": t, "contribution": v} for t, v in contrib[:k]]


class MostFrequentModel(Model):
    """Always predict the single commonest label.

    A base-rate predictor scores exactly zero once thresholded at 0.5, because
    no label's base rate reaches 0.5 — which is itself a useful reminder that
    0.5 is a choice. This baseline gives the floor a non-degenerate value.
    """

    name = "most_frequent"

    def fit(self, texts, Y):
        self.index_ = int(np.asarray(Y).sum(axis=0).argmax())
        self.n_labels_ = np.asarray(Y).shape[1]
        return self

    def predict_proba(self, texts) -> np.ndarray:
        P = np.zeros((len(texts), self.n_labels_))
        P[:, self.index_] = 1.0
        return P


def build_model(name: str, cfg: Config) -> Model:
    if name == "prior":
        return PriorModel()
    if name == "most_frequent":
        return MostFrequentModel()
    if name == "word_logistic":
        return LinearModel(cfg, use_char=False)
    if name == "wordchar_logistic":
        return LinearModel(cfg, use_char=True)
    if name == "wordchar_logistic_unweighted":
        return LinearModel(
            cfg, use_char=True, class_weight=None, name="wordchar_logistic_unweighted"
        )
    if name == "wordchar_svm":
        return LinearModel(cfg, use_char=True, estimator="svm")
    raise ValueError(f"unknown model {name!r}")


def out_of_fold_predictions(
    texts, Y, cfg: Config, model_name: str = "wordchar_logistic"
) -> np.ndarray:
    """K-fold out-of-fold probabilities over the training set.

    The vectoriser is refitted inside every fold. Fitting it once on all the
    training text and then splitting would let each fold's vocabulary and IDF
    weights carry information from the rows it is about to predict — a small
    leak, but this array is what thresholds and the label-noise audit are built
    on, so it has to be clean.
    """
    texts = np.asarray(texts, dtype=object)
    Y = np.asarray(Y)
    oof = np.zeros(Y.shape, dtype=float)
    folds = KFold(cfg.model.n_folds, shuffle=True, random_state=cfg.seed)
    for k, (fit_idx, out_idx) in enumerate(folds.split(texts), start=1):
        model = build_model(model_name, cfg).fit(texts[fit_idx], Y[fit_idx])
        oof[out_idx] = model.predict_proba(texts[out_idx])
        log.info("out-of-fold %d/%d done", k, cfg.model.n_folds)
    return oof

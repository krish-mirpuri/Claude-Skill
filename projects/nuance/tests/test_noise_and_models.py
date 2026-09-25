"""The label-noise audit, the model ladder, and the corpus loader."""

from __future__ import annotations

import numpy as np
import pytest

from nuance.data.goemotions import load_corpus
from nuance.models.train import LinearModel, build_model, out_of_fold_predictions
from nuance.noise.confident_learning import (
    MISSING,
    SPURIOUS,
    estimated_noise_rates,
    family_breakdown,
    rows_to_drop,
    self_confidence_thresholds,
    suspect_labels,
)


# -- corpus ----------------------------------------------------------------
def test_corpus_splits_load(corpus):
    for split in ("train", "dev", "test"):
        s = corpus[split]
        assert len(s) > 0
        assert s.Y.shape[1] == len(corpus.taxonomy.fine)
        assert s.Y.any(axis=1).all(), "every comment carries at least one label"


def test_unknown_split_is_rejected(corpus):
    with pytest.raises(KeyError):
        corpus["holdout"]


def test_support_table_matches_the_label_matrix(corpus):
    support = corpus.support("fine")
    np.testing.assert_array_equal(support["train"].to_numpy(), corpus.train.Y.sum(axis=0))


def test_text_is_normalised_on_load(corpus):
    assert not corpus.train.text.str.contains("  ").any()


def test_missing_corpus_raises_a_useful_error(cfg, tmp_path):
    patched = cfg.model_copy(deep=True)
    patched.data.raw_dir = tmp_path
    with pytest.raises(FileNotFoundError):
        load_corpus(patched)


# -- the ladder -------------------------------------------------------------
def test_prior_predicts_base_rates(corpus):
    model = build_model("prior", None).fit(corpus.train.text, corpus.train.Y)
    P = model.predict_proba(corpus.dev.text)
    np.testing.assert_allclose(P[0], corpus.train.Y.mean(axis=0))
    assert (P == P[0]).all()


def test_most_frequent_predicts_exactly_one_label(corpus):
    model = build_model("most_frequent", None).fit(corpus.train.text, corpus.train.Y)
    P = model.predict_proba(corpus.dev.text)
    assert (P.sum(axis=1) == 1).all()
    assert P.argmax(axis=1)[0] == corpus.train.Y.sum(axis=0).argmax()


def test_unknown_model_is_rejected(cfg):
    with pytest.raises(ValueError):
        build_model("transformer", cfg)


@pytest.mark.parametrize("use_char", [False, True])
def test_linear_model_learns_the_planted_cues(corpus, cfg, use_char):
    model = LinearModel(cfg, use_char=use_char).fit(corpus.train.text, corpus.train.Y)
    P = model.predict_proba(corpus.test.text)
    assert P.shape == corpus.test.Y.shape
    assert ((P >= 0) & (P <= 1)).all()
    B = (P >= 0.5).astype(np.int8)
    from nuance.evaluation.metrics import macro_f1_from_counts, row_contributions

    tp, fp, fn = (m.sum(axis=0) for m in row_contributions(corpus.test.Y, B))
    assert macro_f1_from_counts(tp, fp, fn) > 0.1


def test_binary_target_is_not_silently_inverted(corpus, cfg):
    """Regression test for a bug that inverted every prediction.

    Given a single-column target, scikit-learn's one-vs-rest wrapper treats
    the problem as ordinary binary classification and emits
    ``[P(negative), P(positive)]``. Reading column 0 produced an accuracy of
    0.13 where 0.87 was correct — a number that looks like a broken model
    rather than a transposed index.
    """
    texts = corpus.train.text.to_numpy()
    y = corpus.train.Y[:, corpus.taxonomy.index("gratitude")]
    model = LinearModel(cfg, use_char=False).fit(texts, y.reshape(-1, 1))
    P = model.predict_proba(texts)
    assert P.shape == (len(texts), 1)
    accuracy = float(((P[:, 0] >= 0.5).astype(int) == y).mean())
    assert accuracy > 0.6, f"predictions look inverted (accuracy {accuracy:.3f})"


def test_scoring_a_single_text_keeps_its_shape(corpus, cfg):
    """Regression test: a batch of one must not be transposed.

    A single comment scored against 28 labels and 28 comments scored against
    one label both arrive as a 28-element array. Resolving that by shape gets
    it wrong for whichever case it did not expect — and serving only ever
    sends the first one.
    """
    model = LinearModel(cfg, use_char=False).fit(corpus.train.text, corpus.train.Y)
    n_labels = corpus.train.Y.shape[1]
    assert model.predict_proba(["one comment"]).shape == (1, n_labels)
    assert model.predict_proba(["a", "b", "c"]).shape == (3, n_labels)


def test_out_of_fold_predictions_cover_every_row(corpus, cfg):
    small = cfg.model_copy(deep=True)
    small.model.n_folds = 3
    small.features.char.enabled = False
    oof = out_of_fold_predictions(corpus.train.text, corpus.train.Y, small, "word_logistic")
    assert oof.shape == corpus.train.Y.shape
    assert ((oof >= 0) & (oof <= 1)).all()
    assert oof.sum() > 0


def test_linear_model_can_be_read(corpus, cfg):
    model = LinearModel(cfg, use_char=False).fit(corpus.train.text, corpus.train.Y)
    j = corpus.taxonomy.index("gratitude")
    terms = model.top_terms(j, k=5)
    assert len(terms) == 5
    assert any("thanks" in t for t, _ in terms), "the planted cue should surface"
    drivers = model.explain("thanks so much for this", j, k=3)
    assert drivers and {"term", "contribution"} <= set(drivers[0])


# -- the audit --------------------------------------------------------------
def test_self_confidence_threshold_is_the_mean_over_positives():
    Y = np.array([[1], [1], [0]], dtype=np.int8)
    P = np.array([[0.8], [0.6], [0.9]])
    assert self_confidence_thresholds(Y, P)[0] == pytest.approx(0.7)


def test_audit_recovers_planted_label_errors():
    """Corrupt known rows and check the audit points at them."""
    rng = np.random.default_rng(0)
    n, labels = 600, 6
    truth = np.zeros((n, labels), dtype=np.int8)
    truth[np.arange(n), rng.integers(0, labels, n)] = 1

    # A well-behaved model: confident and correct about the true labels.
    P = np.clip(truth * 0.75 + rng.random((n, labels)) * 0.2, 0.01, 0.99)

    corrupted = truth.copy()
    planted = rng.choice(n, size=40, replace=False)
    for i in planted:
        corrupted[i] = 0
        corrupted[i, (truth[i].argmax() + 3) % labels] = 1

    suspects = suspect_labels(
        corrupted,
        P,
        [f"t{i}" for i in range(n)],
        [f"l{j}" for j in range(labels)],
        top_k=200,
    )
    flagged = set(suspects["row"])
    recall = len(flagged & set(planted.tolist())) / len(planted)
    assert recall > 0.8, f"only recovered {recall:.0%} of the planted errors"
    assert {SPURIOUS, MISSING} <= set(suspects["kind"])


def test_flags_are_spread_across_labels():
    """Per-label thresholds make raw margins incomparable; the ranking must
    not hand every slot to whichever label has the most headroom."""
    rng = np.random.default_rng(1)
    n, labels = 800, 8
    Y = (rng.random((n, labels)) < 0.2).astype(np.int8)
    Y[Y.sum(axis=1) == 0, 0] = 1
    P = rng.random((n, labels))
    suspects = suspect_labels(
        Y, P, [f"t{i}" for i in range(n)], [f"l{j}" for j in range(labels)], top_k=120
    )
    assert suspects["label"].nunique() >= labels - 1


def test_noise_rates_cover_every_label():
    rng = np.random.default_rng(2)
    Y = (rng.random((300, 5)) < 0.3).astype(np.int8)
    rates = estimated_noise_rates(Y, rng.random((300, 5)), [f"l{j}" for j in range(5)])
    assert len(rates) == 5
    assert rates["disputed_share_of_positives"].between(0, 1).all()


def test_rows_to_drop_needs_a_confident_replacement():
    """A low score alone is not evidence; a confident alternative is."""
    Y = np.array([[1, 0], [1, 0], [1, 0]], dtype=np.int8)
    # Label 0's self-confidence threshold is the mean of 0.10, 0.80, 0.15 = 0.35.
    # Label 1 has no positives, so it falls back to 0.5.
    #   row 0: scores below its own label AND label 1 is confident  -> drop
    #   row 1: scores above its own label                           -> keep
    #   row 2: scores below its own label, nothing else is confident -> keep
    P = np.array([[0.10, 0.95], [0.80, 0.05], [0.15, 0.05]])
    np.testing.assert_array_equal(rows_to_drop(Y, P), np.array([0]))


def test_family_breakdown_separates_shades_from_errors(taxonomy):
    import pandas as pd

    suspects = pd.DataFrame(
        [
            {"kind": MISSING, "label": "nervousness", "given_labels": "fear"},
            {"kind": MISSING, "label": "anger", "given_labels": "gratitude"},
        ]
    )
    out = family_breakdown(suspects, taxonomy)
    assert out["flags_considered"] == 2
    assert out["same_ekman_family"] == 1
    assert out["share_same_family"] == pytest.approx(0.5)

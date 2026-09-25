"""Threshold selection and decoding."""

from __future__ import annotations

import numpy as np
import pytest

from nuance.models.decode import (
    Decoder,
    _best_threshold,
    build_decoders,
    tune_global,
    tune_per_label,
)

GRID = np.arange(0.05, 0.96, 0.01)


def _scores(Y, rng, separation=1.8):
    logits = separation * (np.asarray(Y, dtype=float) - 0.5) + rng.normal(0, 1, Y.shape)
    return 1.0 / (1.0 + np.exp(-logits))


def test_threshold_lands_inside_the_grid():
    rng = np.random.default_rng(0)
    y = (rng.random(500) < 0.3).astype(np.int8)
    p = _scores(y.reshape(-1, 1), rng).ravel()
    t = _best_threshold(y, p, GRID)
    assert GRID.min() <= t <= GRID.max()


def test_a_separable_label_gets_a_threshold_that_separates_it():
    y = np.array([0] * 50 + [1] * 50, dtype=np.int8)
    p = np.concatenate([np.full(50, 0.1), np.full(50, 0.9)])
    t = _best_threshold(y, p, GRID)
    assert 0.1 < t <= 0.9
    assert ((p >= t).astype(np.int8) == y).all()


def test_ties_break_toward_the_middle():
    """With a wide plateau of equally good thresholds, pick the least extreme.

    For a rare label most of the grid gives identical F1, and taking the first
    or last index is an arbitrary commitment to an extreme value that will not
    survive to the next sample.
    """
    y = np.array([0] * 90 + [1] * 10, dtype=np.int8)
    p = np.concatenate([np.full(90, 0.05), np.full(10, 0.95)])
    assert _best_threshold(y, p, GRID) == pytest.approx(0.5, abs=0.02)


def test_tuning_returns_one_threshold_per_label():
    rng = np.random.default_rng(1)
    Y = (rng.random((400, 6)) < 0.25).astype(np.int8)
    assert tune_per_label(Y, _scores(Y, rng), GRID).shape == (6,)


def test_global_tuning_returns_a_single_value():
    rng = np.random.default_rng(2)
    Y = (rng.random((400, 6)) < 0.25).astype(np.int8)
    t = tune_global(Y, _scores(Y, rng), GRID)
    assert isinstance(t, float) and GRID.min() <= t <= GRID.max()


def test_decoder_applies_thresholds_per_label():
    P = np.array([[0.4, 0.6], [0.9, 0.1]])
    B = Decoder(np.array([0.5, 0.7])).decode(P)
    np.testing.assert_array_equal(B, np.array([[0, 0], [1, 0]], dtype=np.int8))


def test_fallback_guarantees_a_prediction():
    """Every comment in this corpus carries a label, so predicting none is
    always wrong."""
    P = np.array([[0.1, 0.2, 0.3], [0.05, 0.9, 0.05]])
    strict = Decoder(np.full(3, 0.8), ensure_at_least_one=False).decode(P)
    lenient = Decoder(np.full(3, 0.8), ensure_at_least_one=True).decode(P)
    assert strict[0].sum() == 0
    assert lenient[0].sum() == 1
    assert lenient[0].argmax() == 2
    np.testing.assert_array_equal(strict[1], lenient[1])


def test_fallback_never_removes_a_prediction():
    rng = np.random.default_rng(3)
    P = rng.random((200, 8))
    strict = Decoder(np.full(8, 0.6), ensure_at_least_one=False).decode(P)
    lenient = Decoder(np.full(8, 0.6), ensure_at_least_one=True).decode(P)
    assert (lenient >= strict).all()


def test_build_decoders_records_where_each_was_fitted():
    rng = np.random.default_rng(4)
    Y_dev = (rng.random((300, 5)) < 0.3).astype(np.int8)
    Y_oof = (rng.random((900, 5)) < 0.3).astype(np.int8)
    decoders = build_decoders(
        n_labels=5,
        grid=GRID,
        Y_dev=Y_dev,
        P_dev=_scores(Y_dev, rng),
        Y_oof=Y_oof,
        P_oof=_scores(Y_oof, rng),
    )
    assert set(decoders) == {"flat", "global_tuned", "per_label_dev", "per_label_oof"}
    assert (decoders["flat"].thresholds == 0.5).all()
    assert len(set(decoders["global_tuned"].thresholds)) == 1
    assert "dev" in decoders["per_label_dev"].fitted_on
    assert "out-of-fold" in decoders["per_label_oof"].fitted_on
    for d in decoders.values():
        assert len(d) == 5

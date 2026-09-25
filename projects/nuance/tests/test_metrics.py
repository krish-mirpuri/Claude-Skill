"""Metrics, and the uncertainty machinery around them."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import f1_score

from nuance.evaluation.metrics import (
    bootstrap_weights,
    f1_from_counts,
    headline,
    macro_f1_from_counts,
    measurable_mask,
    micro_f1_from_counts,
    paired_comparison,
    per_label_table,
    resample_counts,
    row_contributions,
)


def noisy_scores(Y: np.ndarray, rng, separation: float = 1.6) -> np.ndarray:
    """Scores whose classes genuinely overlap.

    Writing ``positives get 0.5 + noise, negatives get noise/2`` produces a
    *perfect* classifier and therefore zero-width intervals, which makes every
    test about uncertainty vacuously pass. Pushing the two classes apart in
    log-odds and letting them overlap is the realistic case.
    """
    logits = separation * (np.asarray(Y, dtype=float) - 0.5) + rng.normal(0, 1, Y.shape)
    return 1.0 / (1.0 + np.exp(-logits))


@pytest.fixture
def prediction_pair():
    rng = np.random.default_rng(7)
    Y = (rng.random((800, 12)) < 0.15).astype(np.int8)
    return Y, (noisy_scores(Y, rng) >= 0.5).astype(np.int8)


def _counts(Y, B):
    return tuple(m.sum(axis=0) for m in row_contributions(Y, B))


def test_macro_f1_matches_sklearn(prediction_pair):
    Y, B = prediction_pair
    assert macro_f1_from_counts(*_counts(Y, B)) == pytest.approx(
        f1_score(Y, B, average="macro", zero_division=0)
    )


def test_micro_f1_matches_sklearn(prediction_pair):
    Y, B = prediction_pair
    assert micro_f1_from_counts(*_counts(Y, B)) == pytest.approx(
        f1_score(Y, B, average="micro", zero_division=0)
    )


def test_per_label_f1_matches_sklearn(prediction_pair):
    Y, B = prediction_pair
    np.testing.assert_allclose(
        f1_from_counts(*_counts(Y, B)),
        f1_score(Y, B, average=None, zero_division=0),
        atol=1e-12,
    )


def test_perfect_and_empty_predictions():
    Y = np.array([[1, 0], [0, 1]], dtype=np.int8)
    assert macro_f1_from_counts(*_counts(Y, Y)) == pytest.approx(1.0)
    assert macro_f1_from_counts(*_counts(Y, np.zeros_like(Y))) == pytest.approx(0.0)


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError, match="shape mismatch"):
        row_contributions(np.zeros((3, 2), np.int8), np.zeros((3, 4), np.int8))


def test_resampling_preserves_the_total_weight(prediction_pair):
    Y, B = prediction_pair
    w = bootstrap_weights(len(Y), 50, np.random.default_rng(0))
    assert w.shape == (50, len(Y))
    np.testing.assert_allclose(w.sum(axis=1), len(Y))


def test_bootstrap_brackets_the_point_estimate(prediction_pair):
    Y, B = prediction_pair
    h = headline(Y, B, n_boot=500, rng=np.random.default_rng(1))
    assert h["macro_f1_ci_low"] <= h["macro_f1"] <= h["macro_f1_ci_high"]
    assert h["micro_f1_ci_low"] <= h["micro_f1"] <= h["micro_f1_ci_high"]


def test_intervals_shrink_as_the_test_set_grows():
    """The core claim: a wide interval is a small-sample problem."""
    widths = []
    for n in (400, 6400):
        rng = np.random.default_rng(2)
        Y = (rng.random((n, 8)) < 0.2).astype(np.int8)
        P = noisy_scores(Y, rng)
        widths.append(
            headline(Y, (P >= 0.5).astype(np.int8), n_boot=600, rng=np.random.default_rng(3))[
                "macro_f1_ci_width"
            ]
        )
    # 16x the data should narrow the interval by roughly 4x (1/sqrt(n)).
    assert widths[1] < widths[0] / 2.5


def test_rare_labels_get_wider_intervals(prediction_pair):
    """A rare label is not scored badly — it is barely scored at all."""
    Y, B = prediction_pair
    Y, B = Y.copy(), B.copy()
    # Label 0 becomes rare: six positives, of which the model finds four, plus
    # two false positives. F1 is a respectable 0.57 and almost meaningless.
    Y[:, 0] = 0
    B[:, 0] = 0
    Y[:6, 0] = 1
    B[:4, 0] = 1
    B[6:8, 0] = 1

    table = per_label_table(
        Y,
        B,
        [f"l{i}" for i in range(Y.shape[1])],
        min_support=50,
        n_boot=800,
        rng=np.random.default_rng(4),
    )
    rare = table[table["label"] == "l0"].iloc[0]
    common = table[table["measurable"]].iloc[0]

    assert rare["support"] == 6
    assert not rare["measurable"]
    assert common["measurable"]
    assert rare["f1_ci_width"] > 3 * common["f1_ci_width"]


def test_measurable_mask():
    np.testing.assert_array_equal(
        measurable_mask(np.array([6, 60, 600]), 50), np.array([False, True, True])
    )


def test_headline_counts_measurable_labels(prediction_pair):
    Y, B = prediction_pair
    h = headline(Y, B, min_support=10_000, n_boot=100, rng=np.random.default_rng(5))
    assert h["n_labels_measurable"] == 0
    assert "macro_f1_measurable" not in h


def test_identical_systems_compare_to_exactly_zero(prediction_pair):
    Y, B = prediction_pair
    cmp = paired_comparison(Y, B, B, n_boot=200, rng=np.random.default_rng(6))
    assert cmp["difference"] == pytest.approx(0.0)
    assert cmp["ci_low"] == pytest.approx(0.0)
    assert cmp["ci_high"] == pytest.approx(0.0)
    assert not cmp["significant"]


def test_a_clearly_better_system_is_called_significant(prediction_pair):
    Y, _ = prediction_pair
    good = Y.copy()
    bad = np.zeros_like(Y)
    cmp = paired_comparison(Y, good, bad, n_boot=400, rng=np.random.default_rng(7))
    assert cmp["difference"] > 0
    assert cmp["significant"] and cmp["p_a_better"] == pytest.approx(1.0)


def test_paired_comparison_is_tighter_than_independent_intervals():
    """Why the comparison is paired at all.

    Two similar systems scored on one test set share most of their noise.
    Resampling them together cancels it; comparing their separate intervals
    keeps it, and routinely reports "overlapping, inconclusive" for a
    difference that is in fact consistent.
    """
    rng = np.random.default_rng(8)
    Y = (rng.random((2000, 10)) < 0.2).astype(np.int8)
    P = noisy_scores(Y, rng)
    a = (P >= 0.50).astype(np.int8)
    b = (P >= 0.53).astype(np.int8)

    paired = paired_comparison(Y, a, b, n_boot=800, rng=np.random.default_rng(9))
    ha = headline(Y, a, n_boot=800, rng=np.random.default_rng(10))
    hb = headline(Y, b, n_boot=800, rng=np.random.default_rng(11))

    independent_overlap = (
        ha["macro_f1_ci_low"] <= hb["macro_f1_ci_high"]
        and hb["macro_f1_ci_low"] <= ha["macro_f1_ci_high"]
    )
    paired_width = paired["ci_high"] - paired["ci_low"]
    assert independent_overlap, "the two systems' own intervals overlap"
    assert paired["significant"], "yet the paired test resolves the difference"
    assert paired_width < ha["macro_f1_ci_width"] / 2


def test_same_weights_are_reused_across_models(prediction_pair):
    Y, B = prediction_pair
    w = bootstrap_weights(len(Y), 30, np.random.default_rng(12))
    first = resample_counts(Y, B, w)
    second = resample_counts(Y, B, w)
    np.testing.assert_array_equal(first.tp, second.tp)

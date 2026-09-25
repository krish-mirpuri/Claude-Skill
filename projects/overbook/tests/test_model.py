"""Metrics, calibration, and the guarantee that recalibration stays causal."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from overbook.models.calibrate import Calibrator, rolling_recalibrate
from overbook.models.evaluate import (
    classification_metrics,
    expected_calibration_error,
    reliability_table,
    segment_metrics,
)
from overbook.monitoring.drift import band, drift_report, psi_categorical, psi_numeric


def _wellcalibrated(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.02, 0.98, n)
    return (rng.random(n) < p).astype(int), p


def test_perfect_calibration_has_near_zero_error():
    y, p = _wellcalibrated()
    ece, mce = expected_calibration_error(y, p)
    assert ece < 0.01 and mce < 0.05


def test_a_biased_model_is_caught_by_ece_not_by_auc():
    """The point of tracking calibration: ranking metrics cannot see a shift.

    The rescaling here is *strictly* monotone, so AUC is identical to the last
    bit. That matters for the next test: isotonic regression is only weakly
    monotone, and the ties it creates do move AUC slightly.
    """
    y, p = _wellcalibrated()
    skewed = p**1.6  # strictly increasing on (0, 1), badly scaled
    honest, biased = classification_metrics(y, p), classification_metrics(y, skewed)
    assert honest["roc_auc"] == pytest.approx(biased["roc_auc"], abs=1e-12)
    assert biased["ece"] > 10 * honest["ece"]
    # bias_ratio is mean(prediction) / mean(outcome); 1.0 is unbiased in
    # aggregate, and the distorted scores pull it away from 1 in either
    # direction depending on which way the distortion leans.
    assert abs(biased["bias_ratio"] - 1) > abs(honest["bias_ratio"] - 1)


def test_metric_bundle_is_complete_and_finite():
    y, p = _wellcalibrated(2000, seed=1)
    m = classification_metrics(y, p)
    assert set(m) >= {
        "n",
        "cancel_rate",
        "mean_pred",
        "roc_auc",
        "pr_auc",
        "brier",
        "log_loss",
        "ece",
        "mce",
        "bias_ratio",
    }
    assert all(np.isfinite(v) for v in m.values())


def test_single_class_windows_do_not_crash():
    m = classification_metrics(np.zeros(50, dtype=int), np.full(50, 0.3))
    assert np.isnan(m["roc_auc"]) and np.isfinite(m["brier"])


def test_reliability_table_partitions_the_sample():
    y, p = _wellcalibrated(5000, seed=2)
    tbl = reliability_table(y, p, n_bins=10)
    assert len(tbl) == 10
    assert tbl["n"].sum() == 5000
    assert tbl["mean_predicted"].is_monotonic_increasing


def test_segment_metrics_skips_thin_segments():
    y, p = _wellcalibrated(3000, seed=3)
    groups = pd.Series(["big"] * 2900 + ["tiny"] * 100)
    out = segment_metrics(y, p, groups, min_n=200)
    assert out["segment"].tolist() == ["big"]


# -- calibration -----------------------------------------------------------
def test_isotonic_calibration_repairs_a_distorted_score():
    rng = np.random.default_rng(4)
    n = 30000
    true_p = rng.uniform(0.05, 0.95, n)
    y = (rng.random(n) < true_p).astype(int)
    distorted = np.clip(true_p**2, 1e-6, 1 - 1e-6)  # monotone but badly scaled

    half = n // 2
    cal = Calibrator("isotonic").fit(distorted[:half], y[:half])
    before = classification_metrics(y[half:], distorted[half:])
    after = classification_metrics(y[half:], cal.predict(distorted[half:]))
    assert after["ece"] < before["ece"] / 3
    assert after["brier"] < before["brier"]
    # Isotonic regression is a step function, so it maps distinct scores onto
    # equal ones. Those ties cost a sliver of AUC — small, but not zero, and
    # worth stating rather than claiming exact invariance.
    assert after["roc_auc"] == pytest.approx(before["roc_auc"], abs=5e-3)
    assert after["roc_auc"] <= before["roc_auc"] + 1e-12


@pytest.mark.parametrize("method", ["isotonic", "sigmoid", "none"])
def test_calibrators_return_valid_probabilities(method):
    y, p = _wellcalibrated(3000, seed=5)
    out = Calibrator(method).fit(p, y).predict(p)
    assert out.shape == p.shape and (out > 0).all() and (out < 1).all()


def test_identity_calibrator_changes_nothing():
    _, p = _wellcalibrated(500, seed=6)
    np.testing.assert_allclose(Calibrator("none").fit(p, np.zeros(500, int)).predict(p), p)


def test_unknown_calibration_method_is_rejected():
    with pytest.raises(ValueError):
        Calibrator("magic")


def test_rolling_recalibration_cannot_see_its_own_period():
    """Corrupt the final month; earlier months' predictions must not move."""
    rng = np.random.default_rng(7)
    n = 6000
    dates = pd.to_datetime(rng.choice(pd.date_range("2017-01-01", "2017-05-31"), n))
    p_raw = rng.uniform(0.05, 0.95, n)
    y = (rng.random(n) < p_raw).astype(int)
    frame = pd.DataFrame({"arrival_date": dates, "p_raw": p_raw, "is_canceled": y})

    history = frame.iloc[:1000]
    target = frame.iloc[1000:].copy()
    baseline = rolling_recalibrate(history, target)

    tampered = target.copy()
    last = tampered["arrival_date"] >= pd.Timestamp("2017-05-01")
    tampered.loc[last, "is_canceled"] = 1
    after = rolling_recalibrate(history, tampered)

    earlier = (target["arrival_date"] < pd.Timestamp("2017-05-01")).reindex(baseline.index)
    np.testing.assert_allclose(baseline[earlier.to_numpy()], after[earlier.to_numpy()])


def test_rolling_recalibration_tracks_a_drifting_base_rate():
    """A frozen map lags a rising cancellation rate; refitting keeps up."""
    rng = np.random.default_rng(8)
    days = pd.date_range("2017-01-01", "2017-06-30")
    rows = []
    for i, day in enumerate(days):
        rate = 0.25 + 0.30 * i / len(days)  # 25% climbing to 55%
        for _ in range(60):
            score = rng.uniform(0, 1)
            rows.append((day, score, int(rng.random() < np.clip(rate + 0.3 * (score - 0.5), 0, 1))))
    frame = pd.DataFrame(rows, columns=["arrival_date", "p_raw", "is_canceled"])

    history = frame[frame["arrival_date"] < "2017-03-01"]
    target = frame[frame["arrival_date"] >= "2017-03-01"].copy()

    frozen = Calibrator("isotonic").fit(history["p_raw"], history["is_canceled"])
    static_pred = frozen.predict(target["p_raw"].to_numpy())
    rolled = rolling_recalibrate(history, target).to_numpy()
    truth = target["is_canceled"].to_numpy()

    assert abs(rolled.mean() - truth.mean()) < abs(static_pred.mean() - truth.mean())
    assert (
        classification_metrics(truth, rolled)["ece"]
        < classification_metrics(truth, static_pred)["ece"]
    )


# -- drift -----------------------------------------------------------------
def test_psi_is_zero_for_identical_distributions():
    rng = np.random.default_rng(9)
    x = rng.normal(size=5000)
    assert psi_numeric(x, x) == pytest.approx(0.0, abs=1e-9)


def test_psi_grows_as_a_distribution_moves():
    rng = np.random.default_rng(10)
    ref = rng.normal(size=8000)
    small = psi_numeric(ref, rng.normal(0.1, 1, 8000))
    large = psi_numeric(ref, rng.normal(1.5, 1, 8000))
    assert small < large
    assert band(large) in {"moderate shift", "major shift"}
    assert band(0.01) == "stable"


def test_categorical_psi_detects_a_mix_shift():
    a = pd.Series(["x"] * 700 + ["y"] * 300)
    b = pd.Series(["x"] * 300 + ["y"] * 700)
    assert psi_categorical(a, a) == pytest.approx(0.0, abs=1e-9)
    assert psi_categorical(a, b) > 0.25


def test_drift_report_covers_every_shared_column(clean_df, cfg):
    from overbook.features.build import build_features

    ff, _ = build_features(clean_df, cfg)
    train, test = ff.split("train"), ff.split("test")
    report = drift_report(train.X, test.X, ff.categorical)
    assert set(report["feature"]) == set(train.X.columns)
    assert report["psi"].notna().all()

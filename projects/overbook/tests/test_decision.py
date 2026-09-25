"""The decision layer: exact arithmetic, then the economics on top of it."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest
from scipy.stats import binom

from overbook.decision.backtest import (
    bootstrap_uplift,
    build_states,
    estimate_capacity,
    run_backtest,
    summarise,
)
from overbook.decision.overdispersion import (
    arrival_pmf,
    estimate_shared_shock,
    shock_nodes,
)
from overbook.decision.poisson_binomial import (
    expected_shortfall_and_oversell,
    pmf_mean,
    pmf_quantile,
    pmf_var,
    poisson_binomial_pmf,
)
from overbook.decision.policy import (
    AcceptAll,
    DateState,
    Distributional,
    Economics,
    ExpectedArrivals,
    FixedRate,
    MeanRate,
    NoOverbook,
    Oracle,
    expected_profit_curve,
    realised_profit,
)


# -- exactness -------------------------------------------------------------
def test_matches_the_binomial_when_all_probabilities_are_equal():
    n, p = 40, 0.37
    np.testing.assert_allclose(
        poisson_binomial_pmf(np.full(n, p)), binom.pmf(np.arange(n + 1), n, p), atol=1e-12
    )


def test_matches_brute_force_enumeration():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 12)
    brute = np.zeros(13)
    for bits in itertools.product([0, 1], repeat=12):
        prob = np.prod([p[i] if b else 1 - p[i] for i, b in enumerate(bits)])
        brute[sum(bits)] += prob
    np.testing.assert_allclose(poisson_binomial_pmf(p), brute, atol=1e-12)


def test_moments_match_closed_form():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 60)
    pmf = poisson_binomial_pmf(p)
    assert pmf_mean(pmf) == pytest.approx(p.sum())
    assert pmf_var(pmf) == pytest.approx((p * (1 - p)).sum())


def test_pmf_is_a_distribution():
    rng = np.random.default_rng(2)
    pmf = poisson_binomial_pmf(rng.uniform(0, 1, 200))
    assert pmf.sum() == pytest.approx(1.0)
    assert (pmf >= 0).all()


@pytest.mark.parametrize("probs,expected", [([], 0), ([1.0, 1.0, 1.0], 3), ([0.0, 0.0], 0)])
def test_degenerate_inputs(probs, expected):
    pmf = poisson_binomial_pmf(probs)
    assert pmf_mean(pmf) == pytest.approx(expected)


def test_rejects_probabilities_outside_the_unit_interval():
    with pytest.raises(ValueError):
        poisson_binomial_pmf([0.5, 1.3])


def test_quantiles_and_tail_expectations():
    pmf = poisson_binomial_pmf(np.full(10, 0.5))
    assert pmf_quantile(pmf, 0.5) == 5
    unsold, walked = expected_shortfall_and_oversell(pmf, capacity=5)
    assert unsold == pytest.approx(walked)  # symmetric distribution about capacity


# -- overdispersion --------------------------------------------------------
def test_zero_shock_reproduces_the_independent_case():
    rng = np.random.default_rng(3)
    s = rng.uniform(0.2, 0.9, 50)
    np.testing.assert_allclose(arrival_pmf(s, tau=0.0), poisson_binomial_pmf(s))


def test_shock_widens_the_distribution_without_moving_the_mean_much():
    rng = np.random.default_rng(4)
    s = rng.uniform(0.3, 0.9, 120)
    narrow, wide = arrival_pmf(s, tau=0.0), arrival_pmf(s, tau=0.10)
    assert pmf_var(wide) > 2 * pmf_var(narrow)
    assert pmf_mean(wide) == pytest.approx(pmf_mean(narrow), rel=0.02)
    assert wide.sum() == pytest.approx(1.0)


def test_shock_nodes_have_unit_mean_and_the_requested_spread():
    nodes, weights = shock_nodes(0.08, 7)
    assert float(np.dot(nodes, weights)) == pytest.approx(1.0, abs=1e-9)
    var = float(np.dot((nodes - 1.0) ** 2, weights))
    assert np.sqrt(var) == pytest.approx(0.08, rel=0.05)


def test_shock_estimator_recovers_a_planted_value():
    rng = np.random.default_rng(5)
    tau_true, n_nights = 0.09, 4000
    mu = rng.uniform(40, 90, n_nights)
    pb_var = mu * 0.24  # p(1-p) summed, roughly
    shock = rng.normal(1.0, tau_true, n_nights)
    actual = mu * shock + rng.normal(0, np.sqrt(pb_var))
    assert estimate_shared_shock(mu, pb_var, actual) == pytest.approx(tau_true, rel=0.1)


def test_shock_estimator_returns_zero_when_there_is_no_excess_spread():
    rng = np.random.default_rng(6)
    mu = np.full(500, 50.0)
    assert estimate_shared_shock(mu, np.full(500, 12.0), mu + rng.normal(0, 1, 500)) == 0.0


# -- the profit curve ------------------------------------------------------
def _state(n=100, capacity=60, seed=7, adr=100.0):
    rng = np.random.default_rng(seed)
    p = rng.beta(2, 4, n)
    return DateState(
        "City Hotel", pd.Timestamp("2017-06-01"), capacity, p, (rng.random(n) > p).astype(int), adr
    )


def test_authorising_nobody_earns_nothing():
    curve = expected_profit_curve(np.full(10, 0.8), 5, margin=75.0, walk_cost=200.0)
    assert curve[0] == 0.0


def test_curve_length_matches_the_queue():
    s = np.full(25, 0.7)
    assert expected_profit_curve(s, 10, 75.0, 200.0).size == 26


def test_curve_matches_a_direct_computation_at_each_level():
    rng = np.random.default_rng(8)
    s = rng.uniform(0.3, 0.95, 18)
    curve = expected_profit_curve(s, 9, margin=80.0, walk_cost=160.0)
    for a in (0, 1, 7, 12, 18):
        pmf = poisson_binomial_pmf(s[:a])
        k = np.arange(pmf.size)
        payoff = 80.0 * np.minimum(k, 9) - 160.0 * np.maximum(k - 9, 0)
        assert curve[a] == pytest.approx(float(np.dot(pmf, payoff)))


def test_with_certain_arrivals_the_optimum_is_exactly_capacity():
    """No uncertainty, so there is nothing to overbook against."""
    curve = expected_profit_curve(np.ones(30), capacity=12, margin=75.0, walk_cost=200.0)
    assert int(np.argmax(curve)) == 12


def test_a_costlier_walk_never_raises_the_authorisation_level():
    state = _state()
    limits = [Distributional(Economics(0.25, m)).limit(state) for m in (0.5, 1, 2, 4, 8, 16)]
    assert limits == sorted(limits, reverse=True)


def test_the_shock_variant_is_never_more_aggressive():
    state = _state()
    econ = Economics(0.25, 2.0)
    assert Distributional(econ, tau=0.08).limit(state) <= Distributional(econ).limit(state)


# -- policies --------------------------------------------------------------
def test_every_policy_returns_a_feasible_limit():
    state = _state(n=40, capacity=200)  # capacity deliberately exceeds the queue
    econ = Economics()
    for pol in (
        NoOverbook(),
        AcceptAll(),
        FixedRate(0.15),
        MeanRate(0.37),
        ExpectedArrivals(),
        Distributional(econ),
        Distributional(econ, 0.08),
        Oracle(),
    ):
        assert 0 <= pol.limit(state) <= state.n_bookings


def test_mean_rate_inverts_the_cancellation_rate():
    state = _state(n=500, capacity=63)
    assert MeanRate(0.37).limit(state) == round(63 / 0.63)


def test_fixed_rate_scales_capacity():
    assert FixedRate(0.10).limit(_state(n=500, capacity=100)) == 110


def test_oracle_is_optimal_in_hindsight():
    state = _state(seed=9)
    econ = Economics()
    best = max(range(state.n_bookings + 1), key=lambda a: realised_profit(state, a, econ))
    assert realised_profit(state, Oracle().limit(state), econ) == pytest.approx(
        realised_profit(state, best, econ)
    )


def test_oracle_never_walks_a_guest():
    state = _state(seed=10)
    a = Oracle().limit(state)
    assert int(state.showed[:a].sum()) <= state.capacity


def test_realised_profit_prices_walks_and_empty_rooms():
    state = DateState(
        "H",
        pd.Timestamp("2017-01-01"),
        capacity=2,
        cancel_probs=np.zeros(4),
        showed=np.ones(4, dtype=int),
        adr_mean=100.0,
    )
    econ = Economics(variable_cost_ratio=0.25, walk_cost_multiplier=2.0)
    assert realised_profit(state, 2, econ) == pytest.approx(150.0)  # 2 x 75
    assert realised_profit(state, 3, econ) == pytest.approx(150.0 - 200.0)  # one walked
    assert realised_profit(state, 1, econ) == pytest.approx(75.0)


def test_economics_arithmetic():
    econ = Economics(variable_cost_ratio=0.3, walk_cost_multiplier=2.5)
    assert econ.margin(100.0) == pytest.approx(70.0)
    assert econ.walk_cost(100.0) == pytest.approx(250.0)


# -- backtest plumbing -----------------------------------------------------
def _keys(clean_df):
    df = clean_df[clean_df["split"] == "test"].copy()
    rng = np.random.default_rng(12)
    df["p"] = rng.uniform(0.05, 0.9, len(df))
    return df


def test_estimate_capacity_tracks_the_requested_quantile(clean_df):
    train = clean_df[clean_df["split"] == "train"]
    loose = estimate_capacity(train, 0.95)
    tight = estimate_capacity(train, 0.50)
    assert all(loose[h] >= tight[h] for h in loose)


def test_build_states_preserves_queue_order(clean_df):
    df = _keys(clean_df)
    cap = {h: 2 for h in df["hotel"].unique()}
    states = build_states(df, df["p"].to_numpy(), cap)
    assert states
    one = states[0]
    night = df[(df["hotel"] == one.hotel) & (df["arrival_date"] == one.arrival_date)]
    night = night.sort_values("booking_seq")
    np.testing.assert_allclose(one.cancel_probs, night["p"].to_numpy())
    np.testing.assert_array_equal(one.showed, 1 - night["is_canceled"].to_numpy())


def test_summarise_totals_are_consistent(clean_df):
    df = _keys(clean_df)
    # Capacity of one room per night: the synthetic fixture holds only a couple
    # of bookings per night, so anything larger never binds and every policy
    # collapses onto "accept everything".
    cap = {h: 1 for h in df["hotel"].unique()}
    states = build_states(df, df["p"].to_numpy(), cap)
    econ = Economics()
    results = run_backtest(states, [NoOverbook(), AcceptAll(), Oracle()], econ)
    summary = summarise(results)
    assert summary.loc["no_overbook", "uplift_vs_baseline"] == pytest.approx(0.0)
    assert summary.loc["oracle", "walked"] == 0
    assert summary["total_profit"].idxmax() == "oracle"
    for policy in summary.index:
        rows = results[results["policy"] == policy]
        assert summary.loc[policy, "total_profit"] == pytest.approx(rows["profit"].sum())


def test_bootstrap_interval_brackets_the_point_estimate(clean_df):
    df = _keys(clean_df)
    cap = {h: 1 for h in df["hotel"].unique()}
    states = build_states(df, df["p"].to_numpy(), cap)
    results = run_backtest(states, [NoOverbook(), Oracle()], Economics())
    out = bootstrap_uplift(results, "oracle", "no_overbook", n_samples=500, seed=3)
    assert out["ci_low"] <= out["mean_uplift_per_night"] <= out["ci_high"]
    assert out["p_positive"] == pytest.approx(1.0)

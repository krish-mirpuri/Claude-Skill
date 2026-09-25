"""Exact Poisson-binomial arithmetic.

If a hotel holds *n* bookings for one night and booking *i* shows up with
probability ``s_i``, the number of arrivals is a sum of **non-identical**
Bernoulli variables — a Poisson-binomial, not a binomial.

Why not just use the mean
-------------------------
The usual shortcut is to work with expected arrivals, ``sum(s_i)``. That is
only valid when the loss is symmetric, and overbooking is the textbook case
where it is not: an empty room costs one night's margin, while walking a guest
costs a refund, a room at another hotel and the relationship. Under an
asymmetric loss the optimal decision depends on the whole distribution, so the
distribution is what we compute.

The exact PMF comes from a convolution DP in O(n^2), which at n < 400 bookings
per night is microseconds — there is no reason to approximate.
"""

from __future__ import annotations

import numpy as np


def poisson_binomial_pmf(success_probs) -> np.ndarray:
    """PMF of the number of successes among independent Bernoulli trials.

    Parameters
    ----------
    success_probs:
        Per-trial success probabilities. For overbooking these are *show*
        probabilities, i.e. ``1 - P(cancel)``.

    Returns
    -------
    Array of length ``n + 1`` where element ``k`` is ``P(exactly k successes)``.
    """
    s = np.asarray(success_probs, dtype=float).ravel()
    if s.size and (s.min() < 0.0 or s.max() > 1.0):
        raise ValueError("success probabilities must lie in [0, 1]")
    pmf = np.zeros(s.size + 1, dtype=float)
    pmf[0] = 1.0
    for i, p in enumerate(s, start=1):
        # Convolve one more Bernoulli in. The right-hand side is fully
        # evaluated before assignment, so the in-place update is safe.
        pmf[1 : i + 1] = pmf[1 : i + 1] * (1.0 - p) + pmf[0:i] * p
        pmf[0] *= 1.0 - p
    return pmf


def pmf_mean(pmf: np.ndarray) -> float:
    return float(np.dot(np.arange(pmf.size), pmf))


def pmf_var(pmf: np.ndarray) -> float:
    k = np.arange(pmf.size)
    m = float(np.dot(k, pmf))
    return float(np.dot((k - m) ** 2, pmf))


def pmf_quantile(pmf: np.ndarray, q: float) -> int:
    """Smallest k with ``P(A <= k) >= q``."""
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must lie in [0, 1]")
    return int(np.searchsorted(np.cumsum(pmf), q, side="left"))


def expected_shortfall_and_oversell(pmf: np.ndarray, capacity: int) -> tuple[float, float]:
    """``(E[unsold rooms], E[walked guests])`` for a given capacity."""
    k = np.arange(pmf.size)
    unsold = float(np.dot(np.maximum(capacity - k, 0), pmf))
    walked = float(np.dot(np.maximum(k - capacity, 0), pmf))
    return unsold, walked

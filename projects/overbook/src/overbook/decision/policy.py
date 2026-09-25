"""Authorisation policies: how many bookings to accept for one arrival date.

The decision
------------
For a given hotel and arrival date the hotel holds a queue of bookings ordered
by when they were made. An *authorisation level* ``a`` means "accept the first
``a`` bookings for this date and turn the rest away". Arrivals are then a
Poisson-binomial over the accepted bookings' show probabilities, and:

    profit(A) = margin * min(A, C) - walk_cost * max(A - C, 0)

where ``C`` is capacity. Revenue saturates at capacity; the penalty does not.
That kink is the entire problem — it is why the optimal ``a`` depends on the
shape of the arrival distribution and not merely its mean.

The policies below are deliberately laid out as a ladder, so the backtest can
attribute the gain rather than just claim it:

``NoOverbook``          capacity as the limit — the do-nothing control.
``FixedRate``           C x (1 + r) — what most properties actually do.
``MeanRate``            C / (1 - p̄) using one pooled historical rate.
``ExpectedArrivals``    the model's probabilities, but collapsed to a mean.
``Distributional``      the model's probabilities, used as a distribution.
``Oracle``              perfect foresight; an upper bound, not a policy.

The ``ExpectedArrivals`` rung matters most: it shares a model with
``Distributional`` and differs only in using the mean instead of the full
distribution, which separates "better forecasts" from "better decisions".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from overbook.decision.overdispersion import DEFAULT_NODES, arrival_pmf, shock_nodes


@dataclass(frozen=True)
class Economics:
    """Money. None of this is in the dataset; all of it is an assumption."""

    variable_cost_ratio: float = 0.25
    walk_cost_multiplier: float = 2.0

    def margin(self, adr: float) -> float:
        """Contribution margin of one occupied room-night."""
        return adr * (1.0 - self.variable_cost_ratio)

    def walk_cost(self, adr: float) -> float:
        """Cost of turning away a guest who holds a confirmed booking."""
        return adr * self.walk_cost_multiplier


@dataclass(frozen=True)
class DateState:
    """Everything known about one hotel-night at decision time.

    ``showed`` holds realised outcomes. It is used to *score* a decision and
    by the oracle; no policy other than ``Oracle`` may read it.
    """

    hotel: str
    arrival_date: pd.Timestamp
    capacity: int
    cancel_probs: np.ndarray
    showed: np.ndarray
    adr_mean: float

    @property
    def show_probs(self) -> np.ndarray:
        return 1.0 - self.cancel_probs

    @property
    def n_bookings(self) -> int:
        return int(self.cancel_probs.size)


def expected_profit_curve(
    show_probs: np.ndarray,
    capacity: int,
    margin: float,
    walk_cost: float,
    tau: float = 0.0,
    n_nodes: int = DEFAULT_NODES,
) -> np.ndarray:
    """Expected profit for every authorisation level ``a = 0..n``.

    One shared O(n^2) pass: the Poisson-binomial DP is carried forward so that
    the PMF after ``a`` bookings is reused to compute the PMF after ``a + 1``.
    With ``tau > 0`` the same pass runs once per shared-shock quadrature node
    and the curves are mixed — see :mod:`overbook.decision.overdispersion`.
    """
    s = np.asarray(show_probs, dtype=float).ravel()
    if tau > 0.0:
        nodes, weights = shock_nodes(tau, n_nodes)
        total = np.zeros(s.size + 1, dtype=float)
        for u, w in zip(nodes, weights, strict=True):
            total += w * expected_profit_curve(
                np.clip(s * u, 0.0, 1.0), capacity, margin, walk_cost, tau=0.0
            )
        return total

    n = s.size
    k = np.arange(n + 1)
    payoff = margin * np.minimum(k, capacity) - walk_cost * np.maximum(k - capacity, 0)

    pmf = np.zeros(n + 1, dtype=float)
    pmf[0] = 1.0
    curve = np.empty(n + 1, dtype=float)
    curve[0] = 0.0
    for a in range(1, n + 1):
        p = s[a - 1]
        pmf[1 : a + 1] = pmf[1 : a + 1] * (1.0 - p) + pmf[0:a] * p
        pmf[0] *= 1.0 - p
        curve[a] = float(np.dot(pmf[: a + 1], payoff[: a + 1]))
    return curve


def realised_profit(state: DateState, limit: int, econ: Economics) -> float:
    """Profit actually earned had we authorised ``limit`` bookings."""
    limit = int(np.clip(limit, 0, state.n_bookings))
    arrivals = int(state.showed[:limit].sum())
    served = min(arrivals, state.capacity)
    walked = max(arrivals - state.capacity, 0)
    return econ.margin(state.adr_mean) * served - econ.walk_cost(state.adr_mean) * walked


# --------------------------------------------------------------------------
# policies
# --------------------------------------------------------------------------
class Policy:
    name = "policy"

    def limit(self, state: DateState) -> int:  # pragma: no cover - interface
        raise NotImplementedError

    def _clip(self, a: float, state: DateState) -> int:
        """You cannot authorise more bookings than were ever requested."""
        return int(np.clip(round(a), 0, state.n_bookings))


class NoOverbook(Policy):
    name = "no_overbook"

    def limit(self, state: DateState) -> int:
        return self._clip(state.capacity, state)


class AcceptAll(Policy):
    name = "accept_all"

    def limit(self, state: DateState) -> int:
        return state.n_bookings


class FixedRate(Policy):
    """Authorise a flat percentage above capacity, the common industry rule."""

    def __init__(self, rate: float):
        self.rate = float(rate)
        self.name = f"fixed_{int(round(rate * 100)):02d}pct"

    def limit(self, state: DateState) -> int:
        return self._clip(state.capacity * (1.0 + self.rate), state)


class MeanRate(Policy):
    """Authorise C / (1 - p̄) from one pooled historical cancellation rate.

    This is the strong classical baseline: it does account for cancellations,
    just with a single number for every date, guest and season.
    """

    def __init__(self, pooled_cancel_rate: float):
        self.p = float(np.clip(pooled_cancel_rate, 0.0, 0.95))
        self.name = "mean_rate"

    def limit(self, state: DateState) -> int:
        return self._clip(state.capacity / (1.0 - self.p), state)


class ExpectedArrivals(Policy):
    """Model probabilities, collapsed to a mean: fill until E[arrivals] = C."""

    name = "expected_arrivals"

    def limit(self, state: DateState) -> int:
        cum = np.cumsum(state.show_probs)
        a = int(np.searchsorted(cum, state.capacity, side="left") + 1)
        return self._clip(a, state)


class Distributional(Policy):
    """Maximise expected profit over the arrival distribution.

    With ``tau > 0`` the distribution is widened by the shared-shock mixture,
    which corrects the independence assumption the plain Poisson-binomial
    makes. That correction is what lets the policy respond to how expensive a
    walk actually is: raise the walk cost and it pulls the limit in, which no
    mean-based rule can do.
    """

    def __init__(self, econ: Economics, tau: float = 0.0):
        self.econ = econ
        self.tau = float(tau)
        self.name = "distributional_shock" if self.tau > 0 else "distributional"

    def limit(self, state: DateState) -> int:
        curve = expected_profit_curve(
            state.show_probs,
            state.capacity,
            self.econ.margin(state.adr_mean),
            self.econ.walk_cost(state.adr_mean),
            tau=self.tau,
        )
        return int(np.argmax(curve))


class Oracle(Policy):
    """Perfect foresight upper bound — reads the outcomes. Not achievable."""

    name = "oracle"

    def limit(self, state: DateState) -> int:
        shows = np.cumsum(state.showed)
        reached = np.flatnonzero(shows >= state.capacity)
        # Stop at the booking that fills the last room; otherwise take them all.
        return int(reached[0] + 1) if reached.size else state.n_bookings


def diagnose(state: DateState, econ: Economics, tau: float = 0.0) -> dict:
    """Explain one decision: the curve, the chosen level and its risk profile."""
    curve = expected_profit_curve(
        state.show_probs,
        state.capacity,
        econ.margin(state.adr_mean),
        econ.walk_cost(state.adr_mean),
        tau=tau,
    )
    a = int(np.argmax(curve))
    pmf = arrival_pmf(state.show_probs[:a], tau=tau)
    k = np.arange(pmf.size)
    return {
        "hotel": state.hotel,
        "arrival_date": str(pd.Timestamp(state.arrival_date).date()),
        "capacity": state.capacity,
        "bookings_on_books": state.n_bookings,
        "authorised": a,
        "overbook_pct": 100.0 * (a - state.capacity) / max(state.capacity, 1),
        "expected_arrivals": float(np.dot(k, pmf)),
        "p_oversell": float(pmf[k > state.capacity].sum()),
        "expected_walks": float(np.dot(np.maximum(k - state.capacity, 0), pmf)),
        "expected_unsold": float(np.dot(np.maximum(state.capacity - k, 0), pmf)),
        "expected_profit": float(curve[a]),
        "tau": float(tau),
        "profit_curve": curve.tolist(),
        "arrival_pmf": pmf.tolist(),
    }

"""Correcting the independence assumption behind the Poisson-binomial.

The exact PMF in :mod:`overbook.decision.poisson_binomial` is exact *given*
that bookings cancel independently. They do not. Measured on the validation
window, the spread of night-level arrivals around the model's forecast is
about twice what independence predicts:

    mean Poisson-binomial variance   ~  8
    observed squared forecast error  ~ 23

Two things drive the gap. Guests on one night share shocks the model cannot
see — a conference moving, a storm, a tour operator releasing an allotment —
and the model's own error is itself correlated within a night.

Rather than abandon the exact computation, we keep it and mix over a shared
multiplicative shock ``U`` with mean 1 and standard deviation ``tau``:

    A | U ~ PoissonBinomial(clip(s_i * U, 0, 1))

By the law of total variance this adds ``tau^2 * E[A]^2`` to the spread, which
is exactly the missing term. Conditional on ``U`` the arrivals are still
independent, so each component is still computed exactly; the mixture is a
weighted sum over a handful of Gauss-Hermite nodes.

``tau`` is estimated on the validation window and then frozen — it is never
fitted on the evaluation period.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.polynomial.hermite_e import hermegauss

from overbook.decision.poisson_binomial import poisson_binomial_pmf

log = logging.getLogger(__name__)

DEFAULT_NODES = 7


def shock_nodes(tau: float, n_nodes: int = DEFAULT_NODES) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Hermite quadrature for a mean-1 shock with standard deviation ``tau``."""
    x, w = hermegauss(n_nodes)
    w = w / w.sum()
    return np.clip(1.0 + tau * x, 1e-6, None), w


def estimate_shared_shock(
    predicted_means: np.ndarray, pb_variances: np.ndarray, actual: np.ndarray
) -> float:
    """Solve for the ``tau`` that makes the predictive spread match reality.

    ``E[(A - mu)^2] = E[sigma_pb^2] + tau^2 * E[mu^2]``, so

        tau = sqrt(max(0, (observed MSE - mean PB variance) / mean mu^2)).

    Returns 0 when arrivals are no more dispersed than independence implies.
    """
    mu = np.asarray(predicted_means, dtype=float)
    pbv = np.asarray(pb_variances, dtype=float)
    act = np.asarray(actual, dtype=float)
    if mu.size == 0:
        return 0.0
    mse = float(np.mean((act - mu) ** 2))
    excess = mse - float(pbv.mean())
    tau = float(np.sqrt(max(0.0, excess / float(np.mean(mu**2)))))
    log.info(
        "shared shock: observed MSE %.1f vs independence variance %.1f -> tau = %.4f",
        mse,
        pbv.mean(),
        tau,
    )
    return tau


def arrival_pmf(
    show_probs: np.ndarray, tau: float = 0.0, n_nodes: int = DEFAULT_NODES
) -> np.ndarray:
    """Arrival PMF, optionally widened by the shared shock."""
    s = np.asarray(show_probs, dtype=float)
    if tau <= 0.0:
        return poisson_binomial_pmf(s)
    nodes, weights = shock_nodes(tau, n_nodes)
    out = np.zeros(s.size + 1, dtype=float)
    for u, w in zip(nodes, weights, strict=True):
        out += w * poisson_binomial_pmf(np.clip(s * u, 0.0, 1.0))
    return out

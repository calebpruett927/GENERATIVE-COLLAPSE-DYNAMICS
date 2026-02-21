"""Information Geometry of the Bernoulli Embedding — RCFT.INTSTACK.v1

Derives Fisher geodesic distance, geodesic parametrization, Fano-Fisher
duality, and thermodynamic efficiency from the kernel's Bernoulli embedding
c_i ∈ (0, 1).  All results are Tier-2 RCFT extensions reading Tier-1
invariants.

Theorems
--------
T17  Fisher Geodesic Distance
     d_F(c₁, c₂) = 2|arcsin(√c₁) − arcsin(√c₂)|  per channel.
     Weighted:  d_F² = Σ wᵢ [2(arcsin(√c₁ᵢ) − arcsin(√c₂ᵢ))]².
     Maximum distance π (death to life).
     Čencov uniqueness: only Riemannian metric invariant under
     sufficient statistics (Čencov 1982).

T18  Geodesic Parametrization
     c(t) = sin²((1−t)θ₁ + t θ₂)  where  θ = arcsin(√c).
     This is the minimum-information-cost path between two states.

T19  Fano-Fisher Duality
     h″(c) = −g_F(c) = −1/(c(1−c))  exactly.
     The curvature of the entropy-fidelity envelope equals the
     negative Fisher metric.  Proof: h(c) = −c ln c − (1−c) ln(1−c),
     h′(c) = ln((1−c)/c), h″(c) = −1/(c(1−c)).

T22  Thermodynamic Efficiency Ratio
     η = d_F(start, end) / L(path)  ∈ (0, 1].
     η = 1 for geodesic (optimal recovery), η < 1 for suboptimal paths.

Cross-references
    Lemma 4   (heterogeneity gap = Fisher Information)
    Lemma 5   (S = ln 2 iff c = 1/2 — equator maximum entropy)
    Lemma 15  (Entropy-fidelity envelope S ≤ h(F))
    Lemma 41  (S + κ ≤ 0, equality at c = 1/2 — equator coupling)
    kernel_optimized.py  (Bernoulli embedding, Def 1-8)
    tau_r_star.py  (budget identity)
    frozen_contract.py   (equator_phi diagnostic — Φ_eq = 0 on equator)

Epistemic mapping (Three-Agent Model, DOI: 10.5281/zenodo.16526052)
    Agent 1 (Present/Measuring) → ω = drift — the act of observation
    Agent 2 (Retained/Archive)  → F = fidelity — what survives
    Agent 3 (Unknown/Horizon)   → Γ(ω) = cost of crossing into the unknown
    The geodesic (T18) is the optimal path for Agent 3 → Agent 1
    transitions; the efficiency (T22) measures deviation from optimal.
    The equator c = 1/2 is the boundary of maximum epistemic symmetry
    between Agent 1 and Agent 2.

Collapse Equator Fidelity Law (DOI: 10.5281/zenodo.16423283)
    Ψ(t) ∈ F ⟺ ∃ t* : Ψ(t*) ∈ E ∧ IC(t*) > θ
    The Fano-Fisher duality (T19) proves the equator E is the unique
    axis of self-duality, analogous to Re(s) = 1/2 in the zeta field.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

# --- Frozen constants (from contract) ---
EPSILON = 1e-8


# =====================================================================
# Data structures
# =====================================================================


class FisherDistanceResult(NamedTuple):
    """Result of a Fisher geodesic distance computation."""

    distance: float
    """Geodesic distance in radians (0 to π per channel)."""
    max_possible: float
    """Maximum possible distance (π for 1-D, π√n for n-channel)."""
    normalized: float
    """Distance / max_possible ∈ [0, 1]."""


class GeodesicPoint(NamedTuple):
    """A point on the Fisher geodesic path."""

    t: float
    """Path parameter t ∈ [0, 1]."""
    c: float
    """Coordinate value c(t)."""
    theta: float
    """Angle parameter θ(t) = arcsin(√c(t))."""


class FanoFisherResult(NamedTuple):
    """Verification of the Fano-Fisher duality h″ = −g_F."""

    c: float
    """Evaluation point."""
    h_double_prime: float
    """Numerical second derivative of binary entropy."""
    neg_g_fisher: float
    """−1/(c(1−c))."""
    relative_error: float
    """|h″ − (−g_F)| / |g_F|."""


class EfficiencyResult(NamedTuple):
    """Thermodynamic efficiency of a path."""

    geodesic_distance: float
    """Fisher geodesic distance (lower bound)."""
    path_length: float
    """Actual Fisher path length."""
    efficiency: float
    """η = geodesic_distance / path_length ∈ (0, 1]."""
    excess_fraction: float
    """(path_length / geodesic_distance) − 1  (wasted fraction)."""


# =====================================================================
# T17: Fisher Geodesic Distance
# =====================================================================


def fisher_distance_1d(
    c1: float,
    c2: float,
    *,
    epsilon: float = EPSILON,
) -> float:
    """Fisher geodesic distance between two Bernoulli parameters.

    Theorem T17 (per channel):
        d_F(c₁, c₂) = 2|arcsin(√c₁) − arcsin(√c₂)|

    This is the Bhattacharyya angle on the Bernoulli statistical
    manifold.  It equals the geodesic (shortest path) distance
    under the Fisher-Rao metric  g(c) = 1/(c(1−c)).

    Parameters
    ----------
    c1, c2 : float
        Bernoulli parameters in (0, 1).
    epsilon : float
        Numerical guard for clipping.

    Returns
    -------
    float
        Geodesic distance in radians, range [0, π].
    """
    c1 = float(np.clip(c1, epsilon, 1.0 - epsilon))
    c2 = float(np.clip(c2, epsilon, 1.0 - epsilon))
    return 2.0 * abs(math.asin(math.sqrt(c1)) - math.asin(math.sqrt(c2)))


def fisher_distance_weighted(
    c1: np.ndarray | list[float],
    c2: np.ndarray | list[float],
    w: np.ndarray | list[float],
    *,
    epsilon: float = EPSILON,
) -> FisherDistanceResult:
    """Weighted Fisher geodesic distance for n-channel system.

    Theorem T17 (weighted):
        d²_F = Σ wᵢ [2(arcsin(√c₁ᵢ) − arcsin(√c₂ᵢ))]²

    This is the unique (Čencov) Riemannian distance invariant
    under sufficient statistics on the product Bernoulli manifold.

    Parameters
    ----------
    c1, c2 : array-like
        Coordinate vectors in (0, 1)^n.
    w : array-like
        Weight vector (non-negative, sums to 1).
    epsilon : float
        Numerical guard.

    Returns
    -------
    FisherDistanceResult
        Named tuple with distance, max_possible, normalized.
    """
    c1_arr = np.clip(np.asarray(c1, dtype=float), epsilon, 1.0 - epsilon)
    c2_arr = np.clip(np.asarray(c2, dtype=float), epsilon, 1.0 - epsilon)
    w_arr = np.asarray(w, dtype=float)

    if c1_arr.shape != c2_arr.shape or c1_arr.shape != w_arr.shape:
        msg = "c1, c2, and w must have the same shape"
        raise ValueError(msg)

    diffs = 2.0 * (np.arcsin(np.sqrt(c1_arr)) - np.arcsin(np.sqrt(c2_arr)))
    distance = float(np.sqrt(np.sum(w_arr * diffs**2)))
    max_possible = math.pi * math.sqrt(float(np.sum(w_arr)))
    normalized = distance / max_possible if max_possible > 0 else 0.0

    return FisherDistanceResult(
        distance=distance,
        max_possible=max_possible,
        normalized=normalized,
    )


# =====================================================================
# T18: Geodesic Parametrization
# =====================================================================


def fisher_geodesic(
    c_start: float,
    c_end: float,
    t: float | np.ndarray,
    *,
    epsilon: float = EPSILON,
) -> float | np.ndarray:
    """Geodesic path c(t) on the Bernoulli manifold.

    Theorem T18:
        c(t) = sin²((1−t)θ₁ + t θ₂)    where  θ = arcsin(√c)

    This parametrizes the shortest Fisher-Rao path from c_start to
    c_end as t goes from 0 to 1.

    Parameters
    ----------
    c_start, c_end : float
        Endpoint coordinates in (0, 1).
    t : float or ndarray
        Path parameter(s) in [0, 1].
    epsilon : float
        Numerical guard.

    Returns
    -------
    float or ndarray
        Coordinate(s) along the geodesic.
    """
    c_s = float(np.clip(c_start, epsilon, 1.0 - epsilon))
    c_e = float(np.clip(c_end, epsilon, 1.0 - epsilon))
    theta_s = math.asin(math.sqrt(c_s))
    theta_e = math.asin(math.sqrt(c_e))

    t_arr = np.asarray(t, dtype=float)
    theta_t = (1.0 - t_arr) * theta_s + t_arr * theta_e
    result = np.sin(theta_t) ** 2

    if np.ndim(t) == 0:
        return float(result)
    return result  # type: ignore[return-value]


def compute_geodesic_path(
    c_start: float,
    c_end: float,
    n_points: int = 100,
    *,
    epsilon: float = EPSILON,
) -> list[GeodesicPoint]:
    """Return discrete points along the Fisher geodesic.

    Parameters
    ----------
    c_start, c_end : float
        Endpoint coordinates.
    n_points : int
        Number of sample points.

    Returns
    -------
    list[GeodesicPoint]
        Sequence of (t, c, θ) along the geodesic.
    """
    ts = np.linspace(0.0, 1.0, n_points)
    cs = fisher_geodesic(c_start, c_end, ts, epsilon=epsilon)
    cs_arr = np.atleast_1d(cs)
    thetas = np.arcsin(np.sqrt(np.clip(cs_arr, epsilon, 1.0 - epsilon)))
    return [GeodesicPoint(t=float(ts[i]), c=float(cs_arr[i]), theta=float(thetas[i])) for i in range(n_points)]


# =====================================================================
# T19: Fano-Fisher Duality
# =====================================================================


def binary_entropy(c: float | np.ndarray, *, epsilon: float = EPSILON) -> float | np.ndarray:
    """Binary entropy h(c) = −c ln c − (1−c) ln(1−c).

    Clamped to (ε, 1−ε) for log-safety.
    """
    c_safe = np.clip(c, epsilon, 1.0 - epsilon)
    return -(c_safe * np.log(c_safe) + (1.0 - c_safe) * np.log(1.0 - c_safe))


def fisher_metric_1d(c: float, *, epsilon: float = EPSILON) -> float:
    """Fisher-Rao metric g_F(c) = 1/(c(1−c)) on the Bernoulli manifold."""
    c_safe = float(np.clip(c, epsilon, 1.0 - epsilon))
    return 1.0 / (c_safe * (1.0 - c_safe))


def verify_fano_fisher_duality(
    c_values: list[float] | np.ndarray | None = None,
    *,
    epsilon: float = EPSILON,
    h: float = 1e-6,
) -> list[FanoFisherResult]:
    """Verify Theorem T19: h″(c) = −g_F(c) = −1/(c(1−c)).

    Uses centered finite differences for h″ and compares to the
    analytical Fisher metric.

    Parameters
    ----------
    c_values : array-like or None
        Points at which to verify (default: [0.1, 0.3, 0.5, 0.7, 0.9]).
    epsilon : float
        Guard band.
    h : float
        Step size for finite difference.

    Returns
    -------
    list[FanoFisherResult]
        Verification results at each point.
    """
    if c_values is None:
        c_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    results: list[FanoFisherResult] = []
    for c in c_values:
        h_pp = float((binary_entropy(c + h) - 2 * binary_entropy(c) + binary_entropy(c - h)) / h**2)
        neg_gf = -fisher_metric_1d(c, epsilon=epsilon)
        rel_err = abs(h_pp - neg_gf) / abs(neg_gf) if abs(neg_gf) > 0 else 0.0
        results.append(FanoFisherResult(c=c, h_double_prime=h_pp, neg_g_fisher=neg_gf, relative_error=rel_err))

    return results


# =====================================================================
# T22: Thermodynamic Efficiency Ratio
# =====================================================================


def compute_path_length(
    c_series: np.ndarray | list[float],
    *,
    epsilon: float = EPSILON,
) -> float:
    """Fisher-Rao path length of an observed 1-D trajectory.

    L = Σ d_F(c_{i}, c_{i+1})  over consecutive points.

    Parameters
    ----------
    c_series : array-like
        Sequence of observed coordinate values in (0, 1).

    Returns
    -------
    float
        Total Fisher path length.
    """
    c_arr = np.clip(np.asarray(c_series, dtype=float), epsilon, 1.0 - epsilon)
    if len(c_arr) < 2:
        return 0.0
    thetas = np.arcsin(np.sqrt(c_arr))
    return float(2.0 * np.sum(np.abs(np.diff(thetas))))


def compute_path_length_weighted(
    c_matrix: np.ndarray,
    w: np.ndarray | list[float],
    *,
    epsilon: float = EPSILON,
) -> float:
    """Fisher-Rao path length for n-channel trajectory.

    Parameters
    ----------
    c_matrix : ndarray, shape (T, n)
        T time points, n channels.
    w : array-like, shape (n,)
        Channel weights.

    Returns
    -------
    float
        Weighted Fisher path length.
    """
    c_arr = np.clip(np.asarray(c_matrix, dtype=float), epsilon, 1.0 - epsilon)
    w_arr = np.asarray(w, dtype=float)
    if c_arr.ndim == 1:
        c_arr = c_arr.reshape(-1, 1)
    thetas = np.arcsin(np.sqrt(c_arr))
    dthetas = np.diff(thetas, axis=0)  # (T-1, n)
    step_lengths = np.sqrt(np.sum(w_arr * (2.0 * dthetas) ** 2, axis=1))
    return float(np.sum(step_lengths))


def compute_efficiency(
    c_series: np.ndarray | list[float],
    *,
    epsilon: float = EPSILON,
) -> EfficiencyResult:
    """Thermodynamic efficiency of a 1-D trajectory.

    Theorem T22:  η = d_F(start, end) / L(path)  ∈ (0, 1]

    Parameters
    ----------
    c_series : array-like
        Observed coordinate trajectory.

    Returns
    -------
    EfficiencyResult
        Named tuple with geodesic_distance, path_length, efficiency,
        excess_fraction.
    """
    c_arr = np.asarray(c_series, dtype=float)
    if len(c_arr) < 2:
        return EfficiencyResult(0.0, 0.0, 1.0, 0.0)

    d_geo = fisher_distance_1d(float(c_arr[0]), float(c_arr[-1]), epsilon=epsilon)
    L = compute_path_length(c_arr, epsilon=epsilon)

    eta = 1.0 if epsilon > L else min(d_geo / L, 1.0)

    excess = (L / d_geo - 1.0) if d_geo > epsilon else 0.0

    return EfficiencyResult(
        geodesic_distance=d_geo,
        path_length=L,
        efficiency=eta,
        excess_fraction=max(excess, 0.0),
    )


# =====================================================================
# Convenience: budget-weighted geodesic cost
# =====================================================================


def compute_geodesic_budget_cost(
    c_start: float,
    c_end: float,
    R: float,
    *,
    p: int = 3,
    alpha: float = 1.0,
    C: float = 0.0,
    epsilon: float = EPSILON,
    n_points: int = 1000,
) -> float:
    """Integrated budget cost ∫ τ_R*(c(t)) dt along the Fisher geodesic.

    Computes the total thermodynamic cost of optimal (geodesic) recovery
    from c_start to c_end at measurement rate R.

    Parameters
    ----------
    c_start, c_end : float
        Start and end fidelity.
    R : float
        Measurement rate (positive).
    p : int
        Drift exponent.
    alpha : float
        Curvature coupling.
    C : float
        Curvature (constant along path approximation).
    epsilon : float
        Guard band.
    n_points : int
        Integration resolution.

    Returns
    -------
    float
        Total budget cost.
    """
    if R <= 0:
        msg = "R must be positive"
        raise ValueError(msg)

    ts = np.linspace(0.0, 1.0, n_points)
    c_t = fisher_geodesic(c_start, c_end, ts, epsilon=epsilon)
    omega_t = 1.0 - np.atleast_1d(c_t)
    gamma_t = omega_t**p / (1.0 - omega_t + epsilon) + alpha * C
    tau_r_star = gamma_t / R
    return float(np.trapezoid(tau_r_star, ts))

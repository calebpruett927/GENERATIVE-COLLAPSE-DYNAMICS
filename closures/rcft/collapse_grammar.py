"""Collapse Grammar and Transfer Matrix — RCFT.INTSTACK.v1

Formalizes the regime sequence {STABLE, WATCH, COLLAPSE} as a Markov
grammar with a transfer matrix derived from the Gibbs measure on
Γ(ω).  Extracts spectral gap, mixing time, and entropy rate as
universal complexity measures.

Theorems
--------
T23  Collapse Grammar Transfer Matrix
     The 3×3 transfer matrix  T_{ij}  encodes transition probabilities
     between regimes under Metropolis dynamics with potential Γ(ω).
     At inverse temperature β:
       - Low β (high T):   all regimes accessible, high entropy
       - High β (low T):   STABLE dominates, low entropy
     The spectral gap  Δ = 1 − |λ₂|  determines mixing time  τ_mix ≈ 1/Δ.
     The entropy rate  h = −Σ πⱼ T_{ij} log₂ T_{ij}  classifies
     system complexity independent of domain.

Grammar Classification
     h / log₂(3) ∈ [0, 1]:
       [0.0, 0.2)  FROZEN     — predictable single regime
       [0.2, 0.5)  ORDERED    — some transitions, mostly predictable
       [0.5, 0.8)  COMPLEX    — rich transition dynamics
       [0.8, 1.0]  CHAOTIC    — maximally unpredictable

Cross-references
    constants.py  (RegimeThresholds: ω_stable=0.038, ω_collapse=0.30)
    T13  (Gibbs measure)
    T2   (Γ(ω) drift potential)
    T19  (Fano-Fisher duality — equator symmetry)
    T20  (Central charge c_eff = 1/p — partition function)

Epistemic mapping (Three-Agent Model, DOI: 10.5281/zenodo.16526052)
    The three grammar states map to epistemic agents:
      STABLE   → Agent 2 dominates (archive is healthy, boundaries clear)
      WATCH    → boundary active (measurement consuming fidelity)
      COLLAPSE → Agent 1 overwhelms Agent 2 (measurement destroying archive)
    The grammar entropy rate quantifies whether the epistemic field is
    FROZEN (stuck), ORDERED (predictable), COMPLEX (edge of chaos),
    or CHAOTIC (random).  GCD lives at the ORDERED/COMPLEX boundary:
    always partially structured, never fully random — the "living,
    correctable system" of the Three-Agent framework.

Collapse Equator Fidelity Law (DOI: 10.5281/zenodo.16423283)
    The transfer matrix governs how the system moves through the
    equator over time.  High β (strong collapse penalty) concentrates
    the stationary distribution in STABLE — return through the equator
    with high integrity.  Low β permits all regimes — the equator
    loses its filtering power.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

# --- Frozen constants ---
EPSILON = 1e-8
P_EXPONENT = 3
OMEGA_STABLE = 0.038
OMEGA_COLLAPSE = 0.30
N_REGIMES = 3
REGIME_LABELS = ("STABLE", "WATCH", "COLLAPSE")


# =====================================================================
# Data structures
# =====================================================================


class TransferMatrixResult(NamedTuple):
    """Result of transfer matrix computation."""

    T: np.ndarray
    """3×3 column-stochastic transfer matrix."""
    eigenvalues: np.ndarray
    """Eigenvalues sorted by magnitude (descending)."""
    spectral_gap: float
    """Δ = 1 − |λ₂|."""
    mixing_time: float
    """τ_mix ≈ 1/Δ."""
    stationary: np.ndarray
    """Stationary distribution π (left eigenvector for λ=1)."""
    beta: float
    """Inverse temperature used."""


class GrammarEntropyResult(NamedTuple):
    """Entropy rate of the collapse grammar."""

    entropy_rate: float
    """Collapse grammar entropy rate  h  in bits/step (Bernoulli field entropy rate; classical Shannon rate is the degenerate limit)."""
    max_entropy: float
    """Maximum entropy  log₂(N_regimes)."""
    normalized_entropy: float
    """h / h_max ∈ [0, 1]."""
    complexity_class: str
    """FROZEN | ORDERED | COMPLEX | CHAOTIC."""


class GrammarDiagnostic(NamedTuple):
    """Combined grammar diagnostic."""

    transfer: TransferMatrixResult
    entropy: GrammarEntropyResult
    regime_labels: tuple[str, ...]


# =====================================================================
# Regime classifier
# =====================================================================


def classify_regime_index(
    omega: float,
    *,
    omega_stable: float = OMEGA_STABLE,
    omega_collapse: float = OMEGA_COLLAPSE,
) -> int:
    """Map ω to regime index: 0=STABLE, 1=WATCH, 2=COLLAPSE."""
    if omega < omega_stable:
        return 0
    if omega < omega_collapse:
        return 1
    return 2


# =====================================================================
# T23: Transfer Matrix
# =====================================================================


def compute_transfer_matrix(
    beta: float,
    *,
    p: int = P_EXPONENT,
    epsilon: float = EPSILON,
    n_samples: int = 200000,
    proposal_sigma: float = 0.05,
    seed: int | None = None,
) -> TransferMatrixResult:
    """Estimate the 3×3 regime transfer matrix via Metropolis MCMC.

    Simulates dynamics under the potential Γ(ω) = ω^p/(1−ω+ε) at
    inverse temperature β.  The transfer matrix T_{ij} = P(regime i | regime j).

    Parameters
    ----------
    beta : float
        Inverse temperature (positive).
    p : int
        Drift exponent.
    epsilon : float
        Guard band.
    n_samples : int
        Number of MCMC transitions.
    proposal_sigma : float
        Standard deviation of Gaussian proposal.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    TransferMatrixResult
    """
    if beta <= 0:
        msg = "beta must be positive"
        raise ValueError(msg)

    rng = np.random.default_rng(seed)
    T = np.zeros((N_REGIMES, N_REGIMES))

    # Start in the WATCH regime
    omega = 0.15

    for _ in range(n_samples):
        old_regime = classify_regime_index(omega)

        # Metropolis proposal
        omega_new = np.clip(omega + rng.normal(0, proposal_sigma), epsilon, 1.0 - epsilon)
        dG = (omega_new**p / (1.0 - omega_new + epsilon)) - (omega**p / (1.0 - omega + epsilon))

        if dG < 0 or rng.random() < math.exp(-beta * dG):
            omega = omega_new

        new_regime = classify_regime_index(omega)
        T[new_regime, old_regime] += 1

    # Normalize columns (column-stochastic)
    col_sums = T.sum(axis=0)
    col_sums[col_sums == 0] = 1.0
    T /= col_sums

    # Spectral analysis
    eigvals_raw = np.linalg.eigvals(T)
    eigvals_sorted = np.sort(np.abs(eigvals_raw))[::-1]
    spectral_gap = float(eigvals_sorted[0] - eigvals_sorted[1]) if len(eigvals_sorted) > 1 else 1.0
    mixing_time = 1.0 / spectral_gap if spectral_gap > 1e-12 else float("inf")

    # Stationary distribution (left eigenvector for largest eigenvalue)
    eigvals_full, eigvecs_full = np.linalg.eig(T)
    idx = int(np.argmax(np.abs(eigvals_full)))
    stationary = np.abs(np.asarray(eigvecs_full[:, idx], dtype=np.complex128).real)
    stat_sum = stationary.sum()
    if stat_sum > 0:
        stationary = stationary / stat_sum

    return TransferMatrixResult(
        T=T,
        eigenvalues=eigvals_sorted,
        spectral_gap=spectral_gap,
        mixing_time=mixing_time,
        stationary=stationary,
        beta=beta,
    )


# =====================================================================
# Entropy rate
# =====================================================================


def _classify_complexity(normalized: float) -> str:
    """Classify grammar complexity from normalized entropy."""
    if normalized < 0.2:
        return "FROZEN"
    if normalized < 0.5:
        return "ORDERED"
    if normalized < 0.8:
        return "COMPLEX"
    return "CHAOTIC"


def compute_grammar_entropy(
    T: np.ndarray,
    stationary: np.ndarray,
) -> GrammarEntropyResult:
    """Entropy rate of a Markov chain (collapse grammar).

    h = −Σ_{i,j}  π_j  T_{ij}  log₂(T_{ij})

    Parameters
    ----------
    T : ndarray, shape (n, n)
        Column-stochastic transfer matrix.
    stationary : ndarray, shape (n,)
        Stationary distribution.

    Returns
    -------
    GrammarEntropyResult
    """
    n = T.shape[0]
    h = 0.0
    for j in range(n):
        for i in range(n):
            if T[i, j] > 1e-15 and stationary[j] > 1e-15:
                h -= stationary[j] * T[i, j] * math.log2(T[i, j])

    max_h = math.log2(n)
    norm_h = h / max_h if max_h > 0 else 0.0
    complexity = _classify_complexity(norm_h)

    return GrammarEntropyResult(
        entropy_rate=h,
        max_entropy=max_h,
        normalized_entropy=norm_h,
        complexity_class=complexity,
    )


# =====================================================================
# Full grammar diagnostic
# =====================================================================


def diagnose_grammar(
    beta: float,
    *,
    p: int = P_EXPONENT,
    epsilon: float = EPSILON,
    n_samples: int = 200000,
    seed: int | None = 42,
) -> GrammarDiagnostic:
    """Full collapse grammar diagnostic at inverse temperature β.

    Combines transfer matrix estimation and entropy rate analysis.

    Parameters
    ----------
    beta : float
        Inverse temperature.
    p : int
        Drift exponent.
    seed : int or None
        Random seed.

    Returns
    -------
    GrammarDiagnostic
    """
    transfer = compute_transfer_matrix(beta, p=p, epsilon=epsilon, n_samples=n_samples, seed=seed)
    entropy = compute_grammar_entropy(transfer.T, transfer.stationary)

    return GrammarDiagnostic(
        transfer=transfer,
        entropy=entropy,
        regime_labels=REGIME_LABELS,
    )


def compute_grammar_phase_diagram(
    beta_values: list[float] | None = None,
    *,
    p: int = P_EXPONENT,
    seed: int | None = 42,
) -> list[GrammarDiagnostic]:
    """Compute grammar diagnostics across a range of β values.

    Parameters
    ----------
    beta_values : list[float] or None
        Inverse temperatures (default: logspace from 0.1 to 1000).

    Returns
    -------
    list[GrammarDiagnostic]
    """
    if beta_values is None:
        beta_values = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]

    return [diagnose_grammar(b, p=p, seed=seed + i if seed is not None else None) for i, b in enumerate(beta_values)]

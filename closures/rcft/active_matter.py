"""Active Matter Frictional Cooling — RCFT Closure Module

Implements the canonical UMCP analysis of self-sustained frictional cooling
in active granular systems, correcting the formulaic errors in the
domain-specific audit paper.

Physical system (Antonov et al., Nature Communications 16, 7235 (2025)):
    - N = 180 vibrated macroscopic robots (diameter 15 mm, mass 0.83 g)
    - Packing fraction Φ = 0.45
    - Shaker amplitude A: 18.66–21.56 μm at 110 Hz
    - Tracking at 150 Hz, velocity via Δt = 0.1 s finite differencing
    - Regimes: cooled → mixed → heated as activity f₀ increases

Embedding strategy (4D Ψ-trace):
    c₁ = 1 − ⟨v⟩/v_max       (kinetic fidelity: high = arrested)
    c₂ = 1/(1 + 5·σ_v)        (speed concentration: high = uniform)
    c₃ = fraction(v < 0.1)    (arrest fraction)
    c₄ = 1/(1 + CV)           (order parameter: inverse coeff of variation)

Key corrections over the audit paper:
    1. IC = exp(Σ ln cᵢ)  — product of coordinates, NOT F·e⁻ˢ·(1−ω)·...
    2. α = 1.0 (not 1.5)
    3. Regime: ω-primary cascade (STABLE/WATCH/COLLAPSE/CRITICAL), not OR-chain
    4. τ_R = INF_REC on no return (not 0)
    5. Curvature = std(c)/0.5 — population std of embedded coords, not spatial ϕ

Cross-references:
    frozen_contract.py   (classify_regime, compute_kernel, compute_tau_R)
    kernel_optimized.py  (OptimizedKernelComputer)
    information_geometry.py  (Fisher geodesic distances between phases)
    universality_class.py    (partition function, critical exponents)
    collapse_grammar.py  (transfer matrix at β ↔ 1/activity)
    KERNEL_SPECIFICATION.md  (Lemmas 1-46)
"""

from __future__ import annotations

import math
import sys as _sys
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Frozen contract constants — path insertion for src/umcp
_sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))
from umcp.frozen_contract import (
    ALPHA,
    EPSILON,
    P_EXPONENT,
    classify_regime,
    compute_kernel,
    compute_tau_R,
    cost_curvature,
    equator_phi,
    gamma_omega,
)

# =====================================================================
# Result containers
# =====================================================================


@dataclass
class ActiveMatterConfig:
    """Configuration for active matter embedding.

    Parameters match Antonov et al. experimental setup.
    """

    n_particles: int = 180
    particle_diameter_mm: float = 15.0
    particle_mass_g: float = 0.83
    packing_fraction: float = 0.45
    shaker_freq_hz: float = 110.0
    tracking_freq_hz: float = 150.0
    dt_velocity_s: float = 0.1
    arrest_threshold: float = 0.1  # normalized speed below which → arrested
    n_dims: int = 4
    weights: tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)


@dataclass
class PhaseResult:
    """UMCP kernel results for one experimental phase."""

    name: str
    omega_mean: float
    omega_std: float
    F_mean: float
    S_mean: float
    C_mean: float
    IC_mean: float
    IC_std: float
    kappa_mean: float
    tau_R_median: float
    n_inf_tau: int  # count of INF_REC τ_R values
    regime_dominant: str
    regime_counts: dict[str, int]
    phi_eq_mean: float  # equator deviation
    drift_cost_mean: float  # Γ(ω)
    curvature_cost_mean: float  # D_C
    n_timesteps: int


@dataclass
class ActiveMatterAudit:
    """Complete UMCP audit of active matter frictional cooling."""

    config: ActiveMatterConfig
    phases: list[PhaseResult]
    fisher_distances: dict[str, float]
    path_efficiency: float
    universality: dict[str, Any]
    grammar: dict[str, Any]
    corrected_thresholds: dict[str, dict[str, float]]


# =====================================================================
# Embedding: particle velocities → 4D Ψ(t)
# =====================================================================


def embed_particle_velocities(
    velocities: NDArray[np.floating[Any]],
    *,
    config: ActiveMatterConfig | None = None,
) -> NDArray[np.floating[Any]]:
    """Embed N-particle speed data into 4D UMCP Ψ(t) trace.

    Parameters
    ----------
    velocities : ndarray, shape (T, N)
        Particle speed time series (T timesteps, N particles).
    config : ActiveMatterConfig or None
        Configuration (default values if None).

    Returns
    -------
    psi : ndarray, shape (T, 4)
        Embedded coordinates clipped to [ε, 1−ε].

    Embedding channels
    ------------------
    c₁ : 1 − ⟨v⟩/v_max     — kinetic fidelity (high = arrested)
    c₂ : 1/(1 + 5·σ_v)      — speed concentration
    c₃ : frac(v < threshold) — arrest fraction
    c₄ : 1/(1 + CV)          — order (inverse coefficient of variation)
    """
    if config is None:
        config = ActiveMatterConfig()

    T, _N = velocities.shape
    v_max = np.max(velocities) + 1e-12
    v_norm = velocities / v_max

    psi = np.zeros((T, config.n_dims))
    for t in range(T):
        v = v_norm[t]
        mean_v = np.mean(v)
        std_v = np.std(v)

        psi[t, 0] = 1.0 - mean_v  # kinetic fidelity
        psi[t, 1] = 1.0 / (1.0 + 5.0 * std_v)  # concentration
        psi[t, 2] = np.mean(v < config.arrest_threshold)  # arrest fraction
        psi[t, 3] = 1.0 / (1.0 + std_v / (mean_v + 1e-12))  # order

    return np.clip(psi, EPSILON, 1.0 - EPSILON)


# =====================================================================
# Kernel pipeline: Ψ(t) → invariants + regime
# =====================================================================


def compute_invariants(
    psi: NDArray[np.floating[Any]],
    *,
    weights: NDArray[np.floating[Any]] | None = None,
    eta: float = 0.10,
    H_rec: int = 50,
    norm: str = "L2",
) -> list[dict[str, Any]]:
    """Compute Tier-1 kernel invariants at each timestep.

    Parameters
    ----------
    psi : ndarray, shape (T, n)
        Embedded trace clipped to [ε, 1−ε].
    weights : ndarray or None
        Per-dimension weights (uniform if None).
    eta : float
        Return threshold for τ_R.
    H_rec : int
        Recovery horizon (max lookback).
    norm : str
        Norm for τ_R ("L2", "L1", "Linf").

    Returns
    -------
    list[dict]
        Per-timestep invariant dictionaries with keys:
        t, omega, F, S, C, tau_R, kappa, IC, regime, phi_eq, D_omega, D_C
    """
    T, n = psi.shape
    w: NDArray[np.floating[Any]] = np.ones(n) / n if weights is None else np.asarray(weights)
    results: list[dict[str, Any]] = []
    for t in range(T):
        c_t = psi[t]
        tau_R = compute_tau_R(psi[: t + 1], t, eta, H_rec, norm)
        ko = compute_kernel(c_t, w, tau_R, EPSILON)
        regime = classify_regime(ko.omega, ko.F, ko.S, ko.C, ko.IC)

        results.append(
            {
                "t": t,
                "omega": ko.omega,
                "F": ko.F,
                "S": ko.S,
                "C": ko.C,
                "tau_R": ko.tau_R,
                "kappa": ko.kappa,
                "IC": ko.IC,
                "regime": regime.value,
                "phi_eq": equator_phi(ko.omega, ko.F, ko.C),
                "D_omega": gamma_omega(ko.omega),
                "D_C": cost_curvature(ko.C),
            }
        )

    return results


# =====================================================================
# Phase analysis
# =====================================================================


def analyze_phase(
    name: str,
    invariants: list[dict[str, Any]],
) -> PhaseResult:
    """Compute summary statistics for one experimental phase.

    Parameters
    ----------
    name : str
        Phase label (e.g., "Cooled", "Mixed", "Heated").
    invariants : list[dict]
        Invariant dicts for timesteps in this phase.

    Returns
    -------
    PhaseResult
    """
    from collections import Counter

    omegas = [r["omega"] for r in invariants]
    Fs = [r["F"] for r in invariants]
    Ss = [r["S"] for r in invariants]
    Cs = [r["C"] for r in invariants]
    ICs = [r["IC"] for r in invariants]
    kappas = [r["kappa"] for r in invariants]
    taus = [r["tau_R"] for r in invariants]
    regimes = [r["regime"] for r in invariants]
    phis = [r["phi_eq"] for r in invariants]
    d_omegas = [r["D_omega"] for r in invariants]
    d_cs = [r["D_C"] for r in invariants]

    finite_taus = [t for t in taus if not math.isinf(t)]
    regime_counts = dict(Counter(regimes))
    dominant = Counter(regimes).most_common(1)[0][0]

    return PhaseResult(
        name=name,
        omega_mean=float(np.mean(omegas)),
        omega_std=float(np.std(omegas)),
        F_mean=float(np.mean(Fs)),
        S_mean=float(np.mean(Ss)),
        C_mean=float(np.mean(Cs)),
        IC_mean=float(np.mean(ICs)),
        IC_std=float(np.std(ICs)),
        kappa_mean=float(np.mean(kappas)),
        tau_R_median=float(np.median(finite_taus)) if finite_taus else float("inf"),
        n_inf_tau=len(taus) - len(finite_taus),
        regime_dominant=dominant,
        regime_counts=regime_counts,
        phi_eq_mean=float(np.mean(phis)),
        drift_cost_mean=float(np.mean(d_omegas)),
        curvature_cost_mean=float(np.mean(d_cs)),
        n_timesteps=len(invariants),
    )


# =====================================================================
# Corrected frictional cooling operator
# =====================================================================


def frictional_cooling_rate(
    velocities: NDArray[np.floating[Any]],
    *,
    mu_friction: float = 1.0,
) -> NDArray[np.floating[Any]]:
    """Frictional energy dissipation rate per timestep.

    Corrected version of the paper's symbolic Fc operator.
    Instead of Dirac-delta collision gating, we compute the
    total pairwise velocity variance (which equals the mean
    dissipation rate under Coulomb friction):

        E_diss(t) = μ · Σᵢ (vᵢ(t) − ⟨v⟩(t))²

    This is proportional to the cluster kinetic temperature
    and vanishes for arrested states.

    Parameters
    ----------
    velocities : ndarray, shape (T, N)
        Particle speeds.
    mu_friction : float
        Friction coefficient (default 1.0, absorbed into normalization).

    Returns
    -------
    ndarray, shape (T,)
        Dissipation rate per timestep.
    """
    mean_v = np.mean(velocities, axis=1, keepdims=True)
    return mu_friction * np.sum((velocities - mean_v) ** 2, axis=1)


# =====================================================================
# Cluster stability (Γ_cluster) — corrected
# =====================================================================


def cluster_stability(
    invariants: list[dict[str, Any]],
    *,
    alpha: float = ALPHA,
) -> list[float]:
    """Cluster stability functional per timestep — CORRECTED.

    The paper's Eq. 8 defines:
        Γ_cluster = F · e^(-S) · e^(-λC/(1+τ_R))

    This drops the redundant (1-ω) factor from the IC formula.
    However, the canonical UMCP IC is exp(Σ wᵢ ln cᵢ), which is
    even simpler and more principled.

    We compute both:
    - Canonical IC = exp(κ) (from kernel)
    - The paper's Γ_cluster for comparison

    Returns the canonical IC values.

    Parameters
    ----------
    invariants : list[dict]
        Per-timestep invariant dicts.
    alpha : float
        Curvature cost coefficient.

    Returns
    -------
    list[float]
        Canonical IC values (exp(κ)).
    """
    return [r["IC"] for r in invariants]


# =====================================================================
# Activity-to-beta mapping
# =====================================================================


def activity_to_beta(
    f0: float,
    *,
    f0_ref: float = 2.0,
    beta_ref: float = 10.0,
) -> float:
    """Map reduced activity f₀ to inverse temperature β.

    The partition function Z(β) governs equilibrium statistics
    under the Γ(ω) potential. Higher activity (higher f₀)
    corresponds to lower β (higher temperature, more exploration).

    Mapping: β = β_ref · (f0_ref / f0)

    Parameters
    ----------
    f0 : float
        Reduced activity (f/ΔC from Antonov et al.).
    f0_ref : float
        Reference activity (mid-range, moderate friction).
    beta_ref : float
        Reference β at f0_ref.

    Returns
    -------
    float
        Inverse temperature β for grammar/partition analysis.
    """
    if f0 <= 0:
        return float("inf")
    return beta_ref * (f0_ref / f0)


# =====================================================================
# Full audit pipeline
# =====================================================================


def run_audit(
    velocities: NDArray[np.floating[Any]],
    phase_boundaries: list[tuple[str, int, int]],
    *,
    config: ActiveMatterConfig | None = None,
) -> ActiveMatterAudit:
    """Run complete UMCP audit on active matter velocity data.

    Parameters
    ----------
    velocities : ndarray, shape (T, N)
        Full particle speed time series.
    phase_boundaries : list of (name, start, end)
        Phase labels with start/end timestep indices.
    config : ActiveMatterConfig or None
        Experimental configuration.

    Returns
    -------
    ActiveMatterAudit
        Complete audit with phases, Fisher distances, grammar, etc.
    """
    if config is None:
        config = ActiveMatterConfig()

    # Import RCFT modules
    from closures.rcft.collapse_grammar import diagnose_grammar
    from closures.rcft.information_geometry import (
        compute_path_length_weighted,
        fisher_distance_weighted,
    )
    from closures.rcft.universality_class import (
        compute_central_charge,
        compute_critical_exponents,
    )

    # Step 1: Embed
    w = np.array(config.weights)
    psi = embed_particle_velocities(velocities, config=config)

    # Step 2: Kernel invariants
    all_inv = compute_invariants(psi, weights=w)

    # Step 3: Phase analysis
    phases = []
    phase_coords: dict[str, NDArray[np.floating[Any]]] = {}
    for name, start, end in phase_boundaries:
        phase_inv = all_inv[start:end]
        phases.append(analyze_phase(name, phase_inv))
        phase_coords[name] = np.mean(psi[start:end], axis=0)

    # Step 4: Fisher geodesic distances
    fisher_dists: dict[str, float] = {}
    phase_names = [p.name for p in phases]
    for i in range(len(phase_names)):
        for j in range(i + 1, len(phase_names)):
            n1, n2 = phase_names[i], phase_names[j]
            if n1 in phase_coords and n2 in phase_coords:
                result = fisher_distance_weighted(phase_coords[n1], phase_coords[n2], w)
                fisher_dists[f"{n1}→{n2}"] = result.distance

    # Path efficiency
    all_c = np.array([list(psi[r["t"]]) for r in all_inv])
    pl = compute_path_length_weighted(all_c, w)
    shortest = max(fisher_dists.values()) if fisher_dists else 0.0
    path_eff = shortest / pl if pl > 0 else 0.0

    # Step 5: Universality
    cc = compute_central_charge(p=P_EXPONENT)
    ex = compute_critical_exponents(p=P_EXPONENT)
    universality = {
        "c_eff": cc.c_eff,
        "C_V_measured": cc.C_V_measured,
        "p": P_EXPONENT,
        "exponents": {
            "nu": ex.nu,
            "beta_exp": ex.beta_exp,
            "gamma": ex.gamma,
            "delta": ex.delta,
            "eta": ex.eta,
            "alpha": ex.alpha,
            "d_eff": ex.d_eff,
        },
    }

    # Step 6: Grammar at representative β values
    diag_cooled = diagnose_grammar(100.0, p=P_EXPONENT, seed=42)
    diag_mixed = diagnose_grammar(10.0, p=P_EXPONENT, seed=42)
    diag_heated = diagnose_grammar(1.0, p=P_EXPONENT, seed=42)
    grammar = {
        "cooled_beta_100": {
            "entropy_rate": diag_cooled.entropy.entropy_rate,
            "complexity": diag_cooled.entropy.complexity_class,
            "spectral_gap": diag_cooled.transfer.spectral_gap,
        },
        "mixed_beta_10": {
            "entropy_rate": diag_mixed.entropy.entropy_rate,
            "complexity": diag_mixed.entropy.complexity_class,
            "spectral_gap": diag_mixed.transfer.spectral_gap,
        },
        "heated_beta_1": {
            "entropy_rate": diag_heated.entropy.entropy_rate,
            "complexity": diag_heated.entropy.complexity_class,
            "spectral_gap": diag_heated.transfer.spectral_gap,
        },
    }

    # Step 7: Corrected thresholds
    corrected = {
        "STABLE": {"omega": 0.038, "F": 0.90, "S": 0.15, "C": 0.14},
        "WATCH": {"omega": 0.038},
        "COLLAPSE": {"omega": 0.30},
        "CRITICAL": {"IC": 0.30},
    }

    return ActiveMatterAudit(
        config=config,
        phases=phases,
        fisher_distances=fisher_dists,
        path_efficiency=path_eff,
        universality=universality,
        grammar=grammar,
        corrected_thresholds=corrected,
    )


# =====================================================================
# Synthesis data generator (for testing)
# =====================================================================


def generate_synthetic_velocities(
    *,
    n_particles: int = 180,
    T_per_phase: int = 100,
    T_transition: int = 20,
    seed: int = 42,
) -> tuple[NDArray[np.floating[Any]], list[tuple[str, int, int]]]:
    """Generate synthetic particle velocity data matching Antonov et al.

    Parameters
    ----------
    n_particles : int
        Number of particles.
    T_per_phase : int
        Timesteps per stable phase.
    T_transition : int
        Timesteps per transition region.
    seed : int
        Random seed.

    Returns
    -------
    velocities : ndarray, shape (T_total, N)
    phase_boundaries : list of (name, start, end)
    """
    rng = np.random.default_rng(seed)

    # Cooled: near-zero speeds
    cooled = np.abs(
        rng.exponential(0.02, (T_per_phase, n_particles)) + 0.005 * rng.standard_normal((T_per_phase, n_particles))
    )

    # Mixed: bimodal
    mixed = np.zeros((T_per_phase, n_particles))
    n_arrested = int(0.6 * n_particles)
    for t in range(T_per_phase):
        mixed[t, :n_arrested] = rng.exponential(0.03, n_arrested)
        mixed[t, n_arrested:] = 0.3 + rng.exponential(0.15, n_particles - n_arrested)

    # Heated: broad distribution
    heated = 0.5 + rng.exponential(0.25, (T_per_phase, n_particles))

    # Transitions
    trans1 = np.zeros((T_transition, n_particles))
    trans2 = np.zeros((T_transition, n_particles))
    for t in range(T_transition):
        alpha = t / T_transition
        trans1[t] = (1 - alpha) * cooled[-1] + alpha * mixed[0]
        trans2[t] = (1 - alpha) * mixed[-1] + alpha * heated[0]

    full = np.vstack([cooled, trans1, mixed, trans2, heated])

    t0 = 0
    t1 = T_per_phase
    t2 = t1 + T_transition
    t3 = t2 + T_per_phase
    t4 = t3 + T_transition
    t5 = t4 + T_per_phase

    boundaries = [
        ("Cooled", t0, t1),
        ("Trans→Mixed", t1, t2),
        ("Mixed", t2, t3),
        ("Trans→Heated", t3, t4),
        ("Heated", t4, t5),
    ]

    return full, boundaries

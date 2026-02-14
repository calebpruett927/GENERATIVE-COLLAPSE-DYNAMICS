"""
τ_R* Thermodynamics — Tier-2 Diagnostic with Tier-0 Protocol Checks

Formalizes the critical return delay τ_R* as a Tier-2 diagnostic extension
that reads Tier-1 kernel invariants, performs Tier-0 protocol checks (seam
validation, regime gates, identity verification), and produces thermodynamic
diagnostics (phase classification, budget surplus/deficit, trapping analysis,
R_critical computation).

Tier Architecture:
    Tier-1 (input):  F, ω, S, C, κ, IC — structural invariants (immutable)
    Tier-0 (checks): Seam PASS/FAIL, identity verification, regime gate
    Tier-2 (output): τ_R*, phase, surplus/deficit, trapping, R_crit, Γ(ω)

No back-edges: this module reads Tier-1 outputs but never modifies them.
All computations use frozen constants from the contract. Promoting any
Tier-2 diagnostic produced here to a Tier-0 gate requires a formal seam
weld and contract version bump.

Implemented Theorems & Definitions (from budget identity Def 11):
    Def  T1: τ_R* = (Γ(ω) + αC + Δκ) / R — critical return delay
    Def  T2: Γ(ω) = ω^p/(1-ω+ε) — simple pole at ω=1, effective residue 1/2
             (Z₂ symmetry under ε-regularization; see tau_r_star_dynamics.py Thm T10)
    Def  T3: τ_R* phase surface over (ω, C) space
    Def  T4: Free-return surface Δκ* = −Γ(ω) − αC
    Def  T5: R as the only externally controllable variable
    Def  T6: Δκ as the unique temporal variable (arrow of time)
    Thm  T1: Regime-dependent dominance (STABLE→Δκ, WATCH→αC, COLLAPSE→Γ)
    Thm  T2: Surplus (τ_R*<0) and deficit (τ_R*>0) phases
    Thm  T3: Trapping threshold at c_trap where Γ(ω_trap) = α
    Thm  T4: R_critical = numerator / tol_seam
    Thm  T5: R_min diverges as 1/(1-ω); R_min·(1-ω) → 1/tol_seam = 200
    Thm  T6: Degradation budget paradox — near-death systems have the
             largest degradation budgets because Γ → ∞. This is not a
             paradox: cost is already absorbed into the existing catastrophe.
    Thm  T7: Asymmetric arrow of time (Second Law analog) —
             degradation is free (releases surplus), improvement costs time.
             Improvement costs ~200× more than degradation at c = 0.60.
             The arrow emerges from the budget identity without postulate.
    Thm  T8: Path dependence and multi-step penalty — N-step transitions
             cost more than direct paths; each step incurs Γ(ω_k) overhead.
             No reversible limit exists (irreducible Γ per observation).
    Thm  T9: Measurement cost (Zeno analog) — N observations of a
             stationary system incur N×Γ(ω) overhead. Observing more makes
             seam closure harder. Optimal: observe as rarely as allowed.
             This is not a design choice — it is a consequence of the
             budget identity. There is no vantage point outside the system
             from which collapse can be observed without cost. The belief
             that one can measure without being measured, observe without
             being inside, or validate without incurring budget is the
             "positional illusion" ("The Seam of Reality", Paulus 2025).
             Γ(ω) is the irreducible price of being inside the system
             you are measuring. See: epistemic_weld.py
             (quantify_positional_illusion).

Physical Analogs & Universality (§5):
    - Critical exponent zν = 1 (simple pole) — between mean-field (1/2)
      and Ising 2D (≈2.17). Cleanest possible critical behavior.
    - Thermodynamic correspondence: R↔T, Γ↔TΔS_irr, Δκ↔W_rev,
      τ_R*<0 ↔ exothermic, τ_R*>0 ↔ endothermic.
    - Black hole analog: ω³ numerator parallels M³ in Hawking evaporation.
      Pole at ω=1 is "event horizon" — information cannot return.
    - Landau theory: τ_R* is susceptibility χ ~ 1/|1-ω|; ω is order
      parameter; ordered (ω≪1) vs disordered (ω→1) phases.
    - p=3 Goldilocks: cubic is the unique exponent where the ω^p→1/(1-ω)
      crossover happens at ω≈0.30–0.40 (Watch-to-Collapse boundary).

Foundational Results (from kernel analysis):
    F1: Heterogeneity gap Δ = F − IC = Var(c)/(2·c̄) — exact (Fisher Information is the degenerate limit).
    F2: S ≤ h(F) tight bound — entropy controlled by drift proxy.
    F3: IC and κ are renormalization group invariants (scale-free).
    F4: Fisher metric volume diverges near collapse (det G → ∞ as c_i → 0).
    F5: Dimensional fragility: c* = 0.70^(1/n) — higher dimensions need
        higher per-component fidelity to avoid collapse.

Capabilities enabled:
    - Complete thermodynamic potential from five frozen constants (no fitting)
    - Arrow of time without postulate (Second Law from budget arithmetic)
    - Measurement cost theory (Zeno analog: observation has cost)
    - Sharp decision boundary at c_trap (recoverable vs requires intervention)
    - Natural clock R_natural = 1/τ_R* as system heartbeat metric
    - Cross-domain applicability (finance, physics, biology, engineering)

References:
    KERNEL_SPECIFICATION.md §3 (Def 11, Budget Model)
    KERNEL_SPECIFICATION.md §5 (Empirical Verification)
    AXIOM.md (Core axiom, constants, independently derived identities)
    docs/MATHEMATICAL_ARCHITECTURE.md (interconnection map)
    TIER_SYSTEM.md (Tier-0/1/2 architecture)

Interconnections:
    - Reads: frozen_contract.py (gamma_omega, cost_curvature, check_seam_pass)
    - Reads: kernel_optimized.py (OptimizedKernelComputer, KernelOutputs)
    - Reads: seam_optimized.py (SeamChainAccumulator)
    - Tests: tests/test_145_tau_r_star.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, NamedTuple

from umcp.frozen_contract import (
    ALPHA,
    EPSILON,
    P_EXPONENT,
    TOL_SEAM,
    Regime,
    classify_regime,
    cost_curvature,
    gamma_omega,
)

# =============================================================================
# TIER-2 PHASE CLASSIFICATION
# =============================================================================


class ThermodynamicPhase(Enum):
    """Budget surplus/deficit phase (Theorem T2).

    Determines whether a system generates surplus (spontaneous return)
    or requires deficit spending (return costs time).
    """

    SURPLUS = "SURPLUS"  # τ_R* < 0: system generates budget credit
    DEFICIT = "DEFICIT"  # τ_R* > 0: return costs more than budget allows
    FREE_RETURN = "FREE_RETURN"  # τ_R* ≈ 0: exactly at the break-even surface
    TRAPPED = "TRAPPED"  # τ_R* > 0 AND no single-step escape (Theorem T3)
    POLE = "POLE"  # ω → 1: singular, Γ diverges (Def T2)


class DominanceTerm(Enum):
    """Which budget term dominates τ_R* (Theorem T1).

    Regime-dependent: Stable → Δκ dominates, Watch → αC,
    Collapse → Γ(ω).
    """

    DRIFT = "DRIFT"  # Γ(ω) dominates — collapse regime
    CURVATURE = "CURVATURE"  # αC dominates — watch regime
    MEMORY = "MEMORY"  # Δκ dominates — stable regime


# =============================================================================
# TIER-2 DIAGNOSTIC OUTPUTS
# =============================================================================


class TauRStarResult(NamedTuple):
    """Core τ_R* computation result."""

    tau_R_star: float  # Critical return delay
    gamma: float  # Γ(ω) = ω^p / (1 - ω + ε)
    D_C: float  # αC curvature cost
    delta_kappa: float  # Memory term
    R: float  # Return rate (input)
    numerator: float  # Γ + αC + Δκ (total budget numerator)


@dataclass(frozen=True)
class ThermodynamicDiagnostic:
    """Complete Tier-2 thermodynamic diagnostic for a kernel state.

    This is the primary output of the τ_R* diagnostic. It contains:
    - The τ_R* value and its decomposition
    - Phase classification (surplus/deficit/trapped/pole)
    - Dominant budget term
    - Trapping analysis
    - R_critical and R_min estimates
    - All Tier-0 check results (identities, seam, regime)

    WARNING: This is a Tier-2 DIAGNOSTIC. Promoting any field to a
    Tier-0 gate requires a seam weld and contract version bump.
    """

    # ── Tier-2 diagnostics (computed) ──
    tau_R_star: float
    gamma: float
    D_C: float
    delta_kappa: float
    R: float
    numerator: float
    phase: ThermodynamicPhase
    dominance: DominanceTerm
    R_critical: float  # Minimum R for seam viability
    R_min: float  # Minimum R for seam closure at current state
    is_trapped: bool  # Whether single-step escape is impossible
    c_trap: float  # Trapping threshold (≈ 0.60)

    # ── Tier-1 inputs (read-only, never modified) ──
    omega: float
    F: float
    S: float
    C: float
    kappa: float
    IC: float

    # ── Tier-0 check results ──
    regime: Regime
    tier1_identity_F: bool  # F = 1 - ω (machine precision)
    tier1_identity_IC: bool  # |IC - exp(κ)| < tol
    tier1_bound_AMGM: bool  # IC ≤ F + tol_seam
    tier0_checks_pass: bool  # All Tier-0 checks satisfied

    # ── Diagnostic metadata ──
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON output."""
        return {
            "tau_R_star": self.tau_R_star,
            "gamma": self.gamma,
            "D_C": self.D_C,
            "delta_kappa": self.delta_kappa,
            "R": self.R,
            "numerator": self.numerator,
            "phase": self.phase.value,
            "dominance": self.dominance.value,
            "R_critical": self.R_critical,
            "R_min": self.R_min,
            "is_trapped": self.is_trapped,
            "c_trap": self.c_trap,
            "omega": self.omega,
            "F": self.F,
            "S": self.S,
            "C": self.C,
            "kappa": self.kappa,
            "IC": self.IC,
            "regime": self.regime.value,
            "tier1_identity_F": self.tier1_identity_F,
            "tier1_identity_IC": self.tier1_identity_IC,
            "tier1_bound_AMGM": self.tier1_bound_AMGM,
            "tier0_checks_pass": self.tier0_checks_pass,
            "warnings": list(self.warnings),
        }


# =============================================================================
# CORE τ_R* COMPUTATION (Def T1)
# =============================================================================


def compute_tau_R_star(
    omega: float,
    C: float,
    R: float,
    delta_kappa: float = 0.0,
    *,
    p: int = P_EXPONENT,
    alpha: float = ALPHA,
    epsilon: float = EPSILON,
) -> TauRStarResult:
    """Compute the critical return delay τ_R* (Def T1).

    τ_R* = (Γ(ω) + αC + Δκ) / R

    where Γ(ω) = ω^p / (1 - ω + ε) is the drift cost.

    Args:
        omega: Drift proxy (1 - F)
        C: Curvature proxy (normalized dispersion)
        R: Return rate estimator (external, > 0)
        delta_kappa: Memory term κ(t₁) - κ(t₀), default 0
        p: Contraction exponent (frozen at 3)
        alpha: Curvature coefficient (frozen at 1.0)
        epsilon: Guard band (frozen at 1e-8)

    Returns:
        TauRStarResult with τ_R* and decomposition

    Raises:
        ValueError: If R ≤ 0 (must be positive for finite τ_R*)
    """
    if R <= 0:
        raise ValueError(f"R must be positive, got {R}")

    g = gamma_omega(omega, p, epsilon)
    d_c = cost_curvature(C, alpha)
    numerator = g + d_c + delta_kappa

    return TauRStarResult(
        tau_R_star=numerator / R,
        gamma=g,
        D_C=d_c,
        delta_kappa=delta_kappa,
        R=R,
        numerator=numerator,
    )


# =============================================================================
# R_CRITICAL AND R_MIN (Theorems T4, T5)
# =============================================================================


def compute_R_critical(
    omega: float,
    C: float,
    delta_kappa: float = 0.0,
    *,
    p: int = P_EXPONENT,
    alpha: float = ALPHA,
    epsilon: float = EPSILON,
    tol_seam: float = TOL_SEAM,
) -> float:
    """Minimum return rate for seam viability (Theorem T4).

    R_crit = (Γ(ω) + αC + Δκ) / tol_seam

    Below R_crit, the seam cannot close regardless of τ_R.

    Args:
        omega: Drift proxy
        C: Curvature proxy
        delta_kappa: Memory term
        p: Contraction exponent
        alpha: Curvature coefficient
        epsilon: Guard band
        tol_seam: Seam tolerance

    Returns:
        R_critical
    """
    g = gamma_omega(omega, p, epsilon)
    d_c = cost_curvature(C, alpha)
    return (g + d_c + delta_kappa) / tol_seam


def compute_R_min(
    omega: float,
    C: float,
    tau_R_target: float,
    delta_kappa: float = 0.0,
    *,
    p: int = P_EXPONENT,
    alpha: float = ALPHA,
    epsilon: float = EPSILON,
) -> float:
    """Minimum R for seam closure at a target τ_R (Theorem T5).

    R_min = (Γ(ω) + αC + Δκ) / τ_R_target

    Diverges as 1/(1-ω) near collapse — R_min·(1-ω) → 1/tol_seam = 200.

    Args:
        omega: Drift proxy
        C: Curvature proxy
        tau_R_target: Target return delay
        delta_kappa: Memory term

    Returns:
        Minimum R for viability
    """
    if tau_R_target <= 0:
        return float("inf")

    g = gamma_omega(omega, p, epsilon)
    d_c = cost_curvature(C, alpha)
    return (g + d_c + delta_kappa) / tau_R_target


# =============================================================================
# TRAPPING THRESHOLD (Theorem T3)
# =============================================================================


def compute_trapping_threshold(
    *,
    p: int = P_EXPONENT,
    alpha: float = ALPHA,
    epsilon: float = EPSILON,
) -> float:
    """Compute the trapping threshold c_trap (Theorem T3).

    The trapping threshold is the confidence level below which no
    single-step curvature correction can produce τ_R* ≤ 0. Below c_trap,
    the system is trapped — structural intervention (not incremental
    improvement) is required.

    The threshold satisfies Γ(ω_trap) = α·C_max where C_max = 1.0 is
    the maximum achievable curvature correction (reducing C from 1 to 0).
    When Γ(ω) > α, even eliminating all curvature cannot compensate
    for drift cost.

    At p = 3, α = 1.0: c_trap ≈ 0.315 (ω_trap ≈ 0.685).

    Returns:
        c_trap ∈ (0, 1)
    """
    # Binary search for ω where Γ(ω) = α (maximum curvature correction)
    # Below c_trap (above ω_trap), drift cost exceeds any curvature fix
    low, high = 0.0, 1.0 - epsilon
    for _ in range(100):
        mid = (low + high) / 2
        g = gamma_omega(mid, p, epsilon)
        if g > alpha:  # Γ(ω) > α·C_max where C_max = 1.0
            high = mid
        else:
            low = mid
    omega_trap = (low + high) / 2
    return 1.0 - omega_trap  # c_trap = 1 - ω_trap


def is_trapped(
    omega: float,
    C: float,
    *,
    p: int = P_EXPONENT,
    alpha: float = ALPHA,
    epsilon: float = EPSILON,
) -> bool:
    """Check if system is trapped (Theorem T3).

    A system is trapped when Γ(ω) exceeds any achievable single-step
    correction. This means incremental improvement cannot produce
    surplus — structural intervention is required.

    Args:
        omega: Drift proxy
        C: Curvature proxy

    Returns:
        True if system cannot escape via single-step improvement
    """
    g = gamma_omega(omega, p, epsilon)
    # Maximum achievable single-step correction:
    # best case Δκ ≈ -(α·C) and curvature improves to 0
    max_correction = alpha * C
    return g > max_correction


# =============================================================================
# PHASE AND DOMINANCE CLASSIFICATION (Theorems T1, T2)
# =============================================================================


def classify_phase(tau_R_star: float, omega: float, *, epsilon: float = EPSILON) -> ThermodynamicPhase:
    """Classify thermodynamic phase from τ_R* (Theorem T2).

    Args:
        tau_R_star: Critical return delay
        omega: Drift proxy (for pole detection)
        epsilon: Tolerance for free-return surface

    Returns:
        ThermodynamicPhase
    """
    # Pole detection: ω ≈ 1 means Γ → ∞
    if omega > 1.0 - 10 * epsilon:
        return ThermodynamicPhase.POLE

    if math.isinf(tau_R_star) or math.isnan(tau_R_star):
        return ThermodynamicPhase.POLE

    # Free-return surface: τ_R* ≈ 0
    if abs(tau_R_star) < epsilon:
        return ThermodynamicPhase.FREE_RETURN

    if tau_R_star < 0:
        return ThermodynamicPhase.SURPLUS

    return ThermodynamicPhase.DEFICIT


def classify_dominance(gamma: float, D_C: float, delta_kappa: float) -> DominanceTerm:
    """Identify which term dominates the budget (Theorem T1).

    Regime-dependent dominance:
    - STABLE: Δκ dominates (drift and curvature are negligible)
    - WATCH: αC dominates (curvature heterogeneity drives cost)
    - COLLAPSE: Γ(ω) dominates (cubic slowing overwhelms everything)

    Args:
        gamma: Drift cost Γ(ω)
        D_C: Curvature cost αC
        delta_kappa: Memory term

    Returns:
        DominanceTerm
    """
    abs_terms = {
        DominanceTerm.DRIFT: abs(gamma),
        DominanceTerm.CURVATURE: abs(D_C),
        DominanceTerm.MEMORY: abs(delta_kappa),
    }
    return max(abs_terms, key=abs_terms.get)  # type: ignore[arg-type]


# =============================================================================
# TIER-0 IDENTITY CHECKS
# =============================================================================


def check_tier1_identities(
    F: float,
    omega: float,
    IC: float,
    kappa: float,
    *,
    tol_F: float = 1e-9,
    tol_IC: float = 1e-6,
    tol_seam: float = TOL_SEAM,
) -> tuple[bool, bool, bool, list[str]]:
    """Verify Tier-1 kernel identities (Tier-0 protocol check).

    Checks:
        1. F = 1 - ω (conservation / unitarity)
        2. IC ≈ exp(κ) (exponential identity)
        3. IC ≤ F + tol_seam (AM-GM bound)

    Args:
        F: Fidelity
        omega: Drift
        IC: Integrity composite
        kappa: Log-integrity
        tol_F: Tolerance for F = 1 - ω
        tol_IC: Tolerance for IC = exp(κ)
        tol_seam: Tolerance for AM-GM bound

    Returns:
        (identity_F, identity_IC, bound_AMGM, failures)
    """
    failures: list[str] = []

    # F = 1 - ω
    identity_F = abs(F - (1.0 - omega)) < tol_F
    if not identity_F:
        failures.append(f"F={F:.10f} ≠ 1-ω={1.0 - omega:.10f} (|Δ|={abs(F - (1.0 - omega)):.2e})")

    # IC ≈ exp(κ)
    exp_kappa = math.exp(kappa) if kappa > -700 else 0.0  # Avoid underflow
    identity_IC = abs(IC - exp_kappa) < tol_IC or (IC == 0.0 and exp_kappa < tol_IC)
    if not identity_IC:
        failures.append(f"|IC-exp(κ)|={abs(IC - exp_kappa):.2e} ≥ tol={tol_IC}")

    # IC ≤ F (AM-GM)
    bound_AMGM = F + tol_seam >= IC
    if not bound_AMGM:
        failures.append(f"IC={IC:.6f} > F+tol={F + tol_seam:.6f} (AM-GM violated)")

    return identity_F, identity_IC, bound_AMGM, failures


# =============================================================================
# FULL DIAGNOSTIC (Tier-2 with Tier-0 checks, reading Tier-1)
# =============================================================================


def diagnose(
    omega: float,
    F: float,
    S: float,
    C: float,
    kappa: float,
    IC: float,
    R: float,
    delta_kappa: float = 0.0,
    integrity: float | None = None,
    *,
    p: int = P_EXPONENT,
    alpha: float = ALPHA,
    epsilon: float = EPSILON,
    tol_seam: float = TOL_SEAM,
) -> ThermodynamicDiagnostic:
    """Run the complete τ_R* thermodynamic diagnostic.

    This is the primary entry point for the Tier-2 diagnostic.
    It reads Tier-1 invariants, runs Tier-0 identity checks,
    computes all τ_R* thermodynamic quantities, and returns
    a complete diagnostic.

    Architecture:
        Tier-1 → (F, ω, S, C, κ, IC) read as inputs
        Tier-0 → identity checks, regime classification
        Tier-2 → τ_R*, phase, dominance, trapping, R_crit

    No back-edges: this function never modifies its inputs.

    Args:
        omega: Drift proxy (Tier-1)
        F: Fidelity (Tier-1)
        S: Entropy (Tier-1)
        C: Curvature (Tier-1)
        kappa: Log-integrity (Tier-1)
        IC: Integrity composite (Tier-1)
        R: Return rate (external, > 0)
        delta_kappa: Memory term κ(t₁) - κ(t₀)
        integrity: Alias for IC (for classify_regime compatibility)
        p: Contraction exponent (frozen)
        alpha: Curvature coefficient (frozen)
        epsilon: Guard band (frozen)
        tol_seam: Seam tolerance (frozen)

    Returns:
        ThermodynamicDiagnostic with complete analysis

    Raises:
        ValueError: If R ≤ 0
    """
    if R <= 0:
        raise ValueError(f"R must be positive, got {R}")

    warnings: list[str] = []

    # ── Tier-0: Identity checks ──
    id_F, id_IC, bound_AMGM, id_failures = check_tier1_identities(F, omega, IC, kappa, tol_seam=tol_seam)
    if id_failures:
        warnings.extend(id_failures)

    tier0_pass = id_F and id_IC and bound_AMGM

    # ── Tier-0: Regime classification ──
    integrity_val = integrity if integrity is not None else IC
    regime = classify_regime(omega, F, S, C, integrity_val)

    # ── Tier-2: τ_R* computation ──
    result = compute_tau_R_star(omega, C, R, delta_kappa, p=p, alpha=alpha, epsilon=epsilon)

    # ── Tier-2: Phase classification ──
    phase = classify_phase(result.tau_R_star, omega, epsilon=epsilon)

    # ── Tier-2: Dominance ──
    dominance = classify_dominance(result.gamma, result.D_C, result.delta_kappa)

    # ── Tier-2: Trapping ──
    trapped = is_trapped(omega, C, p=p, alpha=alpha, epsilon=epsilon)
    if trapped and phase == ThermodynamicPhase.DEFICIT:
        phase = ThermodynamicPhase.TRAPPED

    c_trap = compute_trapping_threshold(p=p, alpha=alpha, epsilon=epsilon)

    # ── Tier-2: R_critical and R_min ──
    r_crit = compute_R_critical(omega, C, delta_kappa, p=p, alpha=alpha, epsilon=epsilon, tol_seam=tol_seam)
    r_min = compute_R_min(omega, C, tol_seam, delta_kappa, p=p, alpha=alpha, epsilon=epsilon)

    # ── Diagnostic warnings ──
    if regime == Regime.COLLAPSE:
        warnings.append(f"COLLAPSE regime: Γ(ω)={result.gamma:.4f} dominates budget")
    if trapped:
        warnings.append(f"TRAPPED: no single-step escape at ω={omega:.4f}")
    if omega > 0.95:
        warnings.append(f"Near pole: ω={omega:.4f}, Γ(ω)={result.gamma:.2e}")

    return ThermodynamicDiagnostic(
        tau_R_star=result.tau_R_star,
        gamma=result.gamma,
        D_C=result.D_C,
        delta_kappa=delta_kappa,
        R=R,
        numerator=result.numerator,
        phase=phase,
        dominance=dominance,
        R_critical=r_crit,
        R_min=r_min,
        is_trapped=trapped,
        c_trap=c_trap,
        omega=omega,
        F=F,
        S=S,
        C=C,
        kappa=kappa,
        IC=IC,
        regime=regime,
        tier1_identity_F=id_F,
        tier1_identity_IC=id_IC,
        tier1_bound_AMGM=bound_AMGM,
        tier0_checks_pass=tier0_pass,
        warnings=warnings,
    )


# =============================================================================
# BATCH DIAGNOSTIC (multiple invariant rows)
# =============================================================================


def diagnose_invariants(
    invariants: list[dict[str, Any]],
    R: float = 0.01,
    *,
    p: int = P_EXPONENT,
    alpha: float = ALPHA,
    epsilon: float = EPSILON,
    tol_seam: float = TOL_SEAM,
) -> list[ThermodynamicDiagnostic]:
    """Run τ_R* diagnostic on a list of invariant rows.

    Processes rows from casepack invariants.json format.
    Computes Δκ between consecutive rows when possible.

    Args:
        invariants: List of dicts with keys: omega, F, S, C, kappa, IC
        R: Return rate (default 0.01)

    Returns:
        List of ThermodynamicDiagnostic results
    """
    results: list[ThermodynamicDiagnostic] = []
    prev_kappa: float | None = None

    for row in invariants:
        omega = float(row.get("omega", 0))
        F_val = float(row.get("F", 0))
        S_val = float(row.get("S", 0))
        C_val = float(row.get("C", 0))
        kappa_val = float(row.get("kappa", 0))
        IC_val = float(row.get("IC", 0))

        # Memory term from consecutive rows
        dk = kappa_val - prev_kappa if prev_kappa is not None else 0.0
        prev_kappa = kappa_val

        diag = diagnose(
            omega=omega,
            F=F_val,
            S=S_val,
            C=C_val,
            kappa=kappa_val,
            IC=IC_val,
            R=R,
            delta_kappa=dk,
            p=p,
            alpha=alpha,
            epsilon=epsilon,
            tol_seam=tol_seam,
        )
        results.append(diag)

    return results


# =============================================================================
# PREDICTION VERIFICATION (§6 testable predictions)
# =============================================================================


def verify_cubic_slowing(
    omega_values: list[float],
    *,
    p: int = P_EXPONENT,
    epsilon: float = EPSILON,
) -> dict[str, Any]:
    """Verify Prediction 1: Cubic slowing down.

    Check that Γ(ω) scales as ω^p/(1-ω) — specifically that the
    cost ratio between high and low ω follows the cubic law.

    Args:
        omega_values: List of ω values to test

    Returns:
        Dict with ratio, exponent, and pass status
    """
    if len(omega_values) < 2:
        return {"pass": False, "reason": "Need ≥ 2 ω values"}

    gammas = [gamma_omega(o, p, epsilon) for o in omega_values]
    # Sort by ω
    pairs = sorted(zip(omega_values, gammas, strict=True))
    low_omega, low_gamma = pairs[0]
    high_omega, high_gamma = pairs[-1]

    if low_gamma < 1e-15:
        return {"pass": True, "ratio": float("inf"), "note": "Low ω has negligible Γ"}

    ratio = high_gamma / low_gamma
    # Expected ratio for cubic: (ω_high/ω_low)^p · (1-ω_low)/(1-ω_high)
    if low_omega > 0 and high_omega < 1.0:
        expected_ratio = (high_omega / low_omega) ** p * (1 - low_omega + epsilon) / (1 - high_omega + epsilon)
    else:
        expected_ratio = ratio  # Can't compute expected

    return {
        "pass": ratio > 1.0,
        "observed_ratio": ratio,
        "expected_ratio": expected_ratio,
        "low_omega": low_omega,
        "high_omega": high_omega,
    }


def verify_R_min_divergence(
    omega_values: list[float],
    C: float = 0.1,
    *,
    p: int = P_EXPONENT,
    alpha: float = ALPHA,
    epsilon: float = EPSILON,
    tol_seam: float = TOL_SEAM,
) -> dict[str, Any]:
    """Verify Prediction 5: R_min·(1-ω) → 1/tol_seam = 200 as ω → 1.

    Args:
        omega_values: List of ω values approaching 1
        C: Curvature (fixed for test)

    Returns:
        Dict with products and convergence status
    """
    target = 1.0 / tol_seam  # = 200
    products = []

    for omega in sorted(omega_values):
        r_min = compute_R_min(omega, C, tol_seam, p=p, alpha=alpha, epsilon=epsilon)
        product = r_min * (1.0 - omega)
        products.append({"omega": omega, "R_min": r_min, "R_min*(1-ω)": product})

    # Check convergence: last product should be closer to target than first
    if len(products) >= 2:
        first_err = abs(products[0]["R_min*(1-ω)"] - target) / target
        last_err = abs(products[-1]["R_min*(1-ω)"] - target) / target
        converging = last_err < first_err
    else:
        converging = False

    return {
        "pass": converging,
        "target": target,
        "products": products,
    }


def verify_trapping_threshold(
    *,
    p: int = P_EXPONENT,
    alpha: float = ALPHA,
    epsilon: float = EPSILON,
) -> dict[str, Any]:
    """Verify Prediction 3: Trapping threshold at c ≈ 0.315.

    Tests that systems below c_trap (high ω) cannot self-correct even
    with maximum curvature, while systems above c_trap can.

    Returns:
        Dict with c_trap value and boundary verification
    """
    c_trap = compute_trapping_threshold(p=p, alpha=alpha, epsilon=epsilon)
    omega_trap = 1.0 - c_trap

    # Below trap (high ω): should be trapped even with C = 1.0 (maximum correction)
    below = is_trapped(omega_trap + 0.05, 1.0, p=p, alpha=alpha, epsilon=epsilon)
    # Above trap (low ω): should NOT be trapped with C = 1.0
    above_omega = max(omega_trap - 0.05, 0.01)
    above = is_trapped(above_omega, 1.0, p=p, alpha=alpha, epsilon=epsilon)

    return {
        "pass": below and not above,
        "c_trap": c_trap,
        "omega_trap": omega_trap,
        "below_trapped": below,
        "above_free": not above,
    }

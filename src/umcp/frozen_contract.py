"""
Frozen Contract Constants

Canonical constants from "The Physics of Coherence" frozen contract snapshot.
These values define the measurement constitution and must be disclosed with
any PASS claim.

Reference: The Physics of Coherence, Clement Paulus, December 31, 2025

CRITICAL: If any of these change, a new contract variant must be declared
because comparability has changed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple

import numpy as np

# =============================================================================
# NORMALIZATION DOMAIN
# =============================================================================

DOMAIN_MIN: float = 0.0  # a
DOMAIN_MAX: float = 1.0  # b

# =============================================================================
# GUARD BAND AND CLIPPING
# =============================================================================

EPSILON: float = 1e-8  # ε - numerical guard band
FACE_POLICY: str = "pre_clip"  # Clipping policy

# =============================================================================
# CLOSURE CONSTANTS
# =============================================================================

# Drift cost exponent (cubic penalty)
P_EXPONENT: int = 3  # p in Γ(ω) = ω^p / (1 - ω + ε)

# Curvature cost coefficient
ALPHA: float = 1.0  # α in D_C = α·C

# Lambda (auxiliary coefficient)
LAMBDA: float = 0.2  # λ

# =============================================================================
# SEAM TOLERANCE
# =============================================================================

TOL_SEAM: float = 0.005  # |s| ≤ tol_seam for PASS

# =============================================================================
# TIMEZONE
# =============================================================================

TIMEZONE: str = "America/Chicago"

# =============================================================================
# REGIME THRESHOLDS (Portable Defaults)
# =============================================================================


class Regime(Enum):
    """Collapse regime classification.

    These are not severity labels — they are epistemic categories.
    STABLE means the system can demonstrate return with low cost.
    WATCH means return is structurally possible but epistemically
    uncertain (curvature cost dominates the budget). COLLAPSE means
    the epistemic trace has degraded past the point of viable return
    credit — this is not failure, it is the boundary condition that
    makes return meaningful. Without the possibility of COLLAPSE,
    STABLE would be trivial and the seam would audit nothing.

    CRITICAL is a severity overlay (IC < 0.30) that can accompany
    any regime — it flags that integrity is dangerously low regardless
    of the regime classification.

    See: epistemic_weld.py (EpistemicVerdict — the epistemic
    interpretation of regime + seam outcome).
    """

    STABLE = "STABLE"
    WATCH = "WATCH"
    COLLAPSE = "COLLAPSE"
    CRITICAL = "CRITICAL"  # Overlay: I < 0.30


@dataclass(frozen=True)
class RegimeThresholds:
    """Threshold values for regime classification.

    Default values from The Episteme of Return.
    """

    # Stable thresholds
    omega_stable_max: float = 0.038
    F_stable_min: float = 0.90
    S_stable_max: float = 0.15
    C_stable_max: float = 0.14

    # Watch thresholds
    omega_watch_min: float = 0.038
    omega_watch_max: float = 0.30

    # Collapse threshold
    omega_collapse_min: float = 0.30

    # Critical overlay
    I_critical_max: float = 0.30


# Default thresholds
DEFAULT_THRESHOLDS = RegimeThresholds()


def classify_regime(
    omega: float,
    F: float,
    S: float,
    C: float,
    integrity: float,
    thresholds: RegimeThresholds = DEFAULT_THRESHOLDS,
) -> Regime:
    """
    Classify the current state into a regime.

    Args:
        omega: Drift (1 - F)
        F: Weighted fidelity
        S: Weighted entropy
        C: Curvature (normalized dispersion)
        integrity: Integrity I = exp(κ)
        thresholds: Regime threshold values

    Returns:
        Regime classification

    Reference: The Episteme of Return, regime thresholds
    """
    # Critical overlay takes precedence
    if integrity < thresholds.I_critical_max:
        return Regime.CRITICAL

    # Collapse
    if omega >= thresholds.omega_collapse_min:
        return Regime.COLLAPSE

    # Watch
    if omega >= thresholds.omega_watch_min:
        return Regime.WATCH

    # Stable requires all conditions
    if (
        omega < thresholds.omega_stable_max
        and thresholds.F_stable_min < F
        and thresholds.S_stable_max > S
        and thresholds.C_stable_max > C
    ):
        return Regime.STABLE

    # Default to Watch if not clearly stable
    return Regime.WATCH


# =============================================================================
# COST CLOSURES
# =============================================================================


def gamma_omega(omega: float, p: int = P_EXPONENT, epsilon: float = EPSILON) -> float:
    """
    Drift cost closure Γ(ω).

    Canonical form from The Physics of Coherence:
        Γ(ω) = ω^p / (1 - ω + ε)

    Args:
        omega: Drift value (1 - F)
        p: Exponent (default 3, prime)
        epsilon: Guard band (default 1e-8)

    Returns:
        Drift cost D_ω = Γ(ω)

    Reference: Universal Measurement Contract Protocol, closure form
    """
    return float(omega**p / (1 - omega + epsilon))


def cost_curvature(C: float, alpha: float = ALPHA) -> float:
    """
    Curvature cost D_C = α·C.

    Args:
        C: Curvature (normalized dispersion)
        alpha: Coefficient (default 1.0)

    Returns:
        Curvature cost
    """
    return alpha * C


def compute_budget_delta_kappa(
    R: float,
    tau_R: float,
    D_omega: float,
    D_C: float,
) -> float:
    """
    Compute budget identity target Δκ_budget.

    Δκ_budget = R·τ_R - (D_ω + D_C)

    Args:
        R: Return credit
        tau_R: Re-entry delay
        D_omega: Drift cost
        D_C: Curvature cost

    Returns:
        Budget delta kappa
    """
    return R * tau_R - (D_omega + D_C)


def compute_seam_residual(
    delta_kappa_budget: float,
    delta_kappa_ledger: float,
) -> float:
    """
    Compute seam residual s.

    s = Δκ_budget - Δκ_ledger

    Args:
        delta_kappa_budget: Budget identity target
        delta_kappa_ledger: Ledger identity (κ(t1) - κ(t0))

    Returns:
        Seam residual
    """
    return delta_kappa_budget - delta_kappa_ledger


def check_seam_pass(
    residual: float,
    tau_R: float,
    I_ratio: float,
    delta_kappa: float,
    tol_seam: float = TOL_SEAM,
    tol_exp: float = 1e-6,
    tau_R_inf: float = float("inf"),
) -> tuple[bool, list[str]]:
    """
    Check PASS conditions for seam weld.

    The seam is the verification boundary between outbound collapse and
    demonstrated return. It does not *create* reality — it *recognizes*
    that return has occurred under the rules that were declared before
    collapse. The frozen parameters (ε, p, α, λ, tol_seam) must be
    identical on both sides of the seam; otherwise the comparison is
    incomparable and the weld is meaningless. "Frozen" means consistent
    across the seam, not constant for its own sake.

    PASS requires ALL of:
        1. |s| ≤ tol_seam — the budget identity closed
        2. τ_R is finite (not INF_REC) — something returned
        3. |I_post/I_pre - exp(Δκ)| < tol_exp — the exponential identity held

    If any condition fails, the emission is a GESTURE: it exists, it may
    be internally consistent, but it did not complete the collapse-return
    cycle under the frozen contract. A gesture has no epistemic standing —
    not because it is wrong, but because it did not weld.

    See: epistemic_weld.py for the full epistemic interpretation of
    PASS/FAIL verdicts (RETURN / GESTURE / DISSOLUTION trichotomy).

    Args:
        residual: Seam residual s
        tau_R: Re-entry delay
        I_ratio: Integrity ratio I_post/I_pre
        delta_kappa: Ledger change κ(t1) - κ(t0)
        tol_seam: Seam tolerance
        tol_exp: Exponential identity tolerance
        tau_R_inf: Value representing infinite τ_R

    Returns:
        (pass_status, list of failure reasons)

    Reference: The Physics of Coherence, PASS conditions;
               The Seam of Reality (Paulus, 2025) §3
    """
    failures: list[str] = []

    # Condition 1: |s| ≤ tol_seam
    if abs(residual) > tol_seam:
        failures.append(f"|s|={abs(residual):.6f} > tol_seam={tol_seam}")

    # Condition 2: τ_R is finite
    if tau_R == tau_R_inf or not np.isfinite(tau_R):
        failures.append(f"τ_R={tau_R} is not finite (INF_REC)")

    # Condition 3: Identity check
    exp_delta_kappa = np.exp(delta_kappa)
    identity_error = abs(I_ratio - exp_delta_kappa)
    if identity_error >= tol_exp:
        failures.append(f"|I_ratio - exp(Δκ)|={identity_error:.6e} >= tol_exp={tol_exp}")

    return (len(failures) == 0, failures)


# =============================================================================
# EQUATOR CLOSURE (DIAGNOSTIC)
# =============================================================================


def equator_phi(omega: float, F: float, C: float) -> float:
    """
    Equator closure Φ_eq(ω, F, C).

    Calibrated affine trade-off from The Episteme of Return:
        Φ_eq(ω, F, C) = F - (1.00 - 0.75ω - 0.55C)

    Sign interpretation:
        Φ > 0: Above equator (higher fidelity than expected)
        Φ < 0: Below equator (lower fidelity than expected)
        Φ = 0: On equator (balanced trade-off)

    The equator is the locus of maximum epistemic symmetry where
    four independent conditions converge on c = 1/2:
        (1) Fisher metric minimized:  g_F(1/2) = 4
        (2) Entropy maximized:        S(1/2) = ln 2  (Lemma 5)
        (3) S + κ = 0 exactly:        entropy-integrity cancel (Lemma 41)
        (4) Fano-Fisher duality:       h''(c) = -g_F(c)  (T19)
    This diagnostic measures deviation from that self-consistent axis.
    See: closures/rcft/information_geometry.py for the geodesic
    structure and thermodynamic efficiency relative to the equator.

    WARNING: This is a DIAGNOSTIC, not a gate. Promoting to gate
    requires explicit closure declaration.

    Args:
        omega: Drift (1 - F)
        F: Weighted fidelity
        C: Curvature

    Returns:
        Equator deviation Φ
    """
    return F - (1.00 - 0.75 * omega - 0.55 * C)


# =============================================================================
# RETURN METRIC
# =============================================================================


class ReturnMetric(NamedTuple):
    """Return metric specification.

    All parameters must be published for reproducibility.
    """

    norm: str  # e.g., "L2", "L1", "Linf"
    eta: float  # Threshold η
    delta_t: float  # Sampling cadence Δt
    H_rec: float  # Recovery horizon


def compute_tau_R(
    trace: np.ndarray,
    t: int,
    eta: float,
    H_rec: int,
    norm: str = "L2",
) -> float:
    """
    Compute re-entry delay τ_R.

    τ_R = min{Δt > 0 : ‖Ψ(t) - Ψ(t-Δt)‖ < η} within H_rec

    Args:
        trace: Bounded trace Ψ(t) shape (T, n)
        t: Current time index
        eta: Return threshold
        H_rec: Recovery horizon (max lookback)
        norm: Norm type ("L2", "L1", "Linf")

    Returns:
        Re-entry delay τ_R, or inf if no return within H_rec

    Reference: The Physics of Coherence, return metric
    """
    current = trace[t]

    for delta_t in range(1, min(H_rec + 1, t + 1)):
        past = trace[t - delta_t]
        diff = current - past

        if norm == "L2":
            distance = float(np.sqrt(np.sum(diff**2)))
        elif norm == "L1":
            distance = float(np.sum(np.abs(diff)))
        elif norm == "Linf":
            distance = float(np.max(np.abs(diff)))
        else:
            raise ValueError(f"Unknown norm: {norm}")

        if distance < eta:
            return float(delta_t)

    return float("inf")  # INF_REC - typed censoring


# =============================================================================
# TIER-1 KERNEL INVARIANTS
# =============================================================================


class KernelOutput(NamedTuple):
    """Tier-1 kernel invariant outputs."""

    omega: float  # Drift (1 - F)
    F: float  # Weighted fidelity
    S: float  # Weighted entropy
    C: float  # Curvature
    tau_R: float  # Re-entry delay
    kappa: float  # Log-integrity
    IC: float  # Integrity (exp(κ))


def compute_kernel(
    c: np.ndarray,
    w: np.ndarray,
    tau_R: float,
    epsilon: float = EPSILON,
) -> KernelOutput:
    """
    Compute Tier-1 kernel invariants.

    Given bounded trace Ψ(t) = (c_i(t)) and weights w_i.

    Args:
        c: Coordinate vector in [0,1]^n
        w: Weight vector (must sum to 1)
        tau_R: Pre-computed re-entry delay
        epsilon: Guard band for log

    Returns:
        KernelOutput with all invariants

    Reference: The Physics of Coherence, Tier-1 kernel
    """
    # Clip to avoid log(0)
    c_safe = np.clip(c, epsilon, 1 - epsilon)

    # Weighted fidelity: F = Σ w_i c_i
    F = float(np.dot(w, c_safe))

    # Drift: ω = 1 - F
    omega = 1.0 - F

    # Binary entropy: S = -Σ w_i [c_i ln(c_i) + (1-c_i) ln(1-c_i)]
    h_i = -c_safe * np.log(c_safe) - (1 - c_safe) * np.log(1 - c_safe)
    S = float(np.dot(w, h_i))

    # Curvature: C = stddev({c_i}) / 0.5
    C = float(np.std(c_safe) / 0.5)

    # Log-integrity: κ = Σ ln(c_i) (unweighted for IC)
    kappa = float(np.sum(np.log(c_safe)))

    # Integrity: I = exp(κ)
    IC = float(np.exp(kappa))

    return KernelOutput(
        omega=omega,
        F=F,
        S=S,
        C=C,
        tau_R=tau_R,
        kappa=kappa,
        IC=IC,
    )


# =============================================================================
# NONCONFORMANCE TRIGGERS
# =============================================================================


class NonconformanceType(Enum):
    """Types of nonconformance (decidable failure modes).

    These are the decidable reasons why an emission does not earn
    epistemic credit. Each maps to a specific structural failure
    in the collapse-return cycle.

    The distinction between GESTURE and NO_RETURN is important:
    NO_RETURN means τ_R = ∞_rec (nothing came back at all).
    GESTURE means something came back but the seam did not close —
    the emission exists but did not weld. A gesture may be internally
    consistent and structurally complex, but it did not return through
    collapse under the frozen rules. See: epistemic_weld.py.
    """

    SEAM_FAILURE = "seam_failure"  # |s| > tol
    NO_RETURN = "no_return"  # τ_R = INF_REC
    GESTURE = "gesture"  # Emission exists but seam did not close
    TIER0_FAILURE = "tier0_failure"  # Missing N_K or Ψ(t)
    CLOSURE_FAILURE = "closure_failure"  # Missing Γ, α, R, τ_R params
    SYMBOL_FAILURE = "symbol_failure"  # Redefining Tier-1 symbols
    DIAGNOSTIC_MISUSE = "diagnostic_misuse"  # Promoting diagnostic to gate


# =============================================================================
# FROZEN CONTRACT SNAPSHOT
# =============================================================================


@dataclass(frozen=True)
class FrozenContract:
    """
    Frozen contract snapshot.

    This is the "measurement constitution" - all parameters that
    must be fixed for comparability.
    """

    # Normalization domain
    domain_min: float = DOMAIN_MIN
    domain_max: float = DOMAIN_MAX

    # Face policy
    face_policy: str = FACE_POLICY

    # Guard band
    epsilon: float = EPSILON

    # Closure constants
    p: int = P_EXPONENT
    alpha: float = ALPHA
    lambda_: float = LAMBDA

    # Seam tolerance
    tol_seam: float = TOL_SEAM

    # Timezone
    timezone: str = TIMEZONE


# Default frozen contract (Physics of Coherence edition)
DEFAULT_CONTRACT = FrozenContract()

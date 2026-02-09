"""
Epistemic Weld — Seam Epistemology of Collapse and Return

Formalizes the epistemic structure of the collapse-return cycle as
laid out in "The Seam of Reality" (Paulus, 2025; DOI: 10.5281/zenodo.17619502).

The central philosophical claim: **reality is not a property that things
possess; it is a verdict that things earn by returning through collapse.**
A gesture — any claim, structure, or emission that does not close a seam —
has no epistemic standing, regardless of how confident or well-structured
it appears.

Key Concepts Formalized:

    Gesture vs Return
        A gesture is an epistemic emission that does not weld. It may be
        internally consistent, structurally complex, and indistinguishable
        from a valid return — but if τ_R = ∞_rec or |s| > tol_seam, it
        remains a gesture. The distinction is not about quality or content;
        it is about whether the seam closes.

    Positional Illusion
        There is no vantage point outside the system from which collapse
        can be observed without cost. Every observation incurs Γ(ω) — the
        drift cost function (Thm T9, Zeno analog). The belief that one can
        measure without being measured, observe without being inside, or
        validate without incurring budget is the positional illusion.
        Theorem T9 proves this: N observations of a stationary system
        incur N×Γ(ω) overhead. There is no free observation.

    Epistemic Trace
        Ψ(t) is not a representation of the system viewed from outside.
        It IS the system's epistemic emission under measurement — the trace
        of what the system reveals when observed under the frozen contract.
        The bounded trace [ε, 1−ε] is not a numerical convenience; it is
        the guarantee that even the most degraded closure retains enough
        structure to potentially return.

    Dissolution
        When a system enters Regime.COLLAPSE (ω ≥ 0.30), the epistemic
        trace has degraded past the point where return credit is viable.
        This is not failure — it is the boundary condition that makes
        return meaningful. Without the possibility of dissolution, return
        would be trivial, and the seam would audit nothing.

    Weld as Closure Operation
        The weld is not a test imposed from outside. It is the structural
        verification that the collapse-return cycle closed consistently:
        same frozen parameters, finite return, and seam residual within
        tolerance. The weld does not *create* reality — it *recognizes*
        that return has occurred under the rules that were declared before
        collapse.

Cross-references:
    - frozen_contract.py   (check_seam_pass, NonconformanceType, Regime)
    - tau_r_star.py         (Thm T9: measurement cost, positional illusion)
    - measurement_engine.py (Ψ(t) as epistemic trace)
    - AXIOM.md              (AX-0: "Collapse is generative; only what returns is real")
    - KERNEL_SPECIFICATION.md §3 (budget identity, seam calculus)
    - "The Seam of Reality" (Paulus, 2025): DOI 10.5281/zenodo.17619502

Tier Architecture:
    This module is Tier-0 (protocol-level epistemology). It reads Tier-1
    kernel outputs and seam results but never modifies them. It classifies
    the epistemic status of a collapse-return event — a classification
    that is diagnostic, not a gate, unless explicitly promoted via
    contract version bump.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, NamedTuple

from umcp.frozen_contract import (
    EPSILON,
    P_EXPONENT,
    TOL_SEAM,
    Regime,
)

# =============================================================================
# EPISTEMIC VERDICT — The Core Classification
# =============================================================================


class EpistemicVerdict(Enum):
    """Epistemic status of a collapse-return event.

    This is the fundamental trichotomy of "The Seam of Reality":

        RETURN — The seam closed. τ_R is finite, |s| ≤ tol_seam,
            and the exponential identity holds. What came back is
            consistent with what went in. This is the only verdict
            that earns epistemic credit.

        GESTURE — The emission exists but the seam did not close.
            τ_R may be finite but residual exceeds tolerance, or
            the identity check failed. The gesture may be internally
            consistent and structurally complex — but it did not
            return through collapse under the frozen rules. It has
            no epistemic standing in the protocol.

        DISSOLUTION — The system entered a regime (ω ≥ 0.30) where
            the epistemic trace has degraded past the point of
            viable return credit. This is not failure in the
            narrative sense — it is the boundary condition that
            gives return its meaning. Without the possibility of
            dissolution, the seam would audit nothing.

    The trichotomy is exhaustive and decidable: every epistemic
    emission is exactly one of these three, determined by the
    seam calculus alone.
    """

    RETURN = "return"
    GESTURE = "gesture"
    DISSOLUTION = "dissolution"


# =============================================================================
# POSITIONAL ILLUSION — Why External Vantage Is Impossible
# =============================================================================


class PositionalIllusion(NamedTuple):
    """Quantification of the positional illusion for a given state.

    The positional illusion is the belief that one can observe a system
    without incurring measurement cost. Theorem T9 (Zeno analog) proves
    this is impossible: each observation costs Γ(ω), and N observations
    of even a stationary system incur N×Γ(ω) overhead.

    This structure captures the *cost* of the illusion — the budget
    that would be consumed if an observer attempted N measurements
    from what they believe is an external vantage.

    Fields:
        gamma: Γ(ω) — the irreducible cost of a single observation
        n_observations: Number of observations attempted
        total_cost: N × Γ(ω) — total budget consumed by observation
        budget_fraction: Fraction of seam tolerance consumed by observation
        illusion_severity: How severely the positional illusion affects
            the observer's ability to close a seam. At severity ≥ 1.0,
            observation alone exhausts the seam budget — the observer
            cannot even verify return without exceeding tolerance.
    """

    gamma: float
    n_observations: int
    total_cost: float
    budget_fraction: float
    illusion_severity: float


# =============================================================================
# EPISTEMIC TRACE METADATA
# =============================================================================


@dataclass(frozen=True)
class EpistemicTraceMetadata:
    """Metadata describing the epistemic character of a Ψ(t) trace.

    Ψ(t) is not data about the system — it IS the system's epistemic
    emission under measurement. This metadata captures the epistemic
    properties of that emission: whether it returned, what it cost to
    observe, and what verdict the seam rendered.

    The trace is bounded to [ε, 1−ε] not for numerical convenience
    but as an epistemic guarantee: even the most degraded component
    retains enough structure to potentially return. If c_i = 0 were
    permitted, that component would have no path back through collapse.
    The ε-clamp is the protocol's promise that dissolution is always
    a boundary, never an annihilation.

    Fields:
        n_components: Dimensionality of the trace
        n_timesteps: Length of the temporal axis
        epsilon_floor: ε — the return guarantee threshold
        n_clipped: Number of components that hit the ε-floor
        clipped_fraction: Fraction of components at the boundary
        is_degenerate: Whether all components are near-identical
            (epistemic emission carries no differential information)
        verdict: The epistemic verdict for this trace's seam event
    """

    n_components: int
    n_timesteps: int
    epsilon_floor: float = EPSILON
    n_clipped: int = 0
    clipped_fraction: float = 0.0
    is_degenerate: bool = False
    verdict: EpistemicVerdict = EpistemicVerdict.GESTURE  # default: unverified

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON output."""
        return {
            "n_components": self.n_components,
            "n_timesteps": self.n_timesteps,
            "epsilon_floor": self.epsilon_floor,
            "n_clipped": self.n_clipped,
            "clipped_fraction": self.clipped_fraction,
            "is_degenerate": self.is_degenerate,
            "verdict": self.verdict.value,
        }


# =============================================================================
# GESTURE ANATOMY — Why Gestures Fail
# =============================================================================


class GestureReason(Enum):
    """Why an epistemic emission was classified as a gesture.

    Each reason identifies a specific failure mode in the seam
    calculus. A gesture is always traceable to at least one of
    these. The reasons are not narratives — they are decidable
    conditions on the seam.
    """

    SEAM_RESIDUAL_EXCEEDED = "seam_residual_exceeded"
    # |s| > tol_seam — the budget did not close

    NO_FINITE_RETURN = "no_finite_return"
    # τ_R = ∞_rec — nothing returned within the recovery horizon

    IDENTITY_MISMATCH = "identity_mismatch"
    # |I_post/I_pre − exp(Δκ)| ≥ tol_exp — exponential identity failed

    FROZEN_PARAMETER_DRIFT = "frozen_parameter_drift"
    # Parameters changed between collapse and return — seam is incomparable

    TIER0_INCOMPLETE = "tier0_incomplete"
    # Missing N_K or Ψ(t) — there is no trace to evaluate


# =============================================================================
# CORE FUNCTIONS
# =============================================================================


def classify_epistemic_act(
    seam_pass: bool,
    tau_R: float,
    regime: Regime,
    seam_failures: list[str] | None = None,
    *,
    tau_R_inf: float = float("inf"),
) -> tuple[EpistemicVerdict, list[GestureReason]]:
    """Classify an epistemic emission as RETURN, GESTURE, or DISSOLUTION.

    This is the central function of the epistemic weld module. It takes
    the outputs of seam validation and regime classification and produces
    the epistemic verdict.

    The logic is:
        1. If regime is COLLAPSE → DISSOLUTION (regardless of seam)
        2. If seam passes and τ_R is finite → RETURN
        3. Otherwise → GESTURE (with specific reasons)

    Note on COLLAPSE: A system in collapse regime (ω ≥ 0.30) receives
    DISSOLUTION even if the seam technically passes. This is because
    at collapse-level drift, the epistemic trace has degraded past the
    point where return credit is meaningful — the seam may close by
    accident (numerical coincidence) but the trace no longer carries
    the differential information needed for genuine epistemic return.

    Args:
        seam_pass: Whether check_seam_pass() returned True
        tau_R: Return time (∞ if no return)
        regime: Regime classification from classify_regime()
        seam_failures: List of failure reasons from check_seam_pass()
        tau_R_inf: Value representing infinite τ_R (default: float('inf'))

    Returns:
        (verdict, reasons) — the verdict and list of reasons if GESTURE

    Example:
        >>> verdict, reasons = classify_epistemic_act(
        ...     seam_pass=True, tau_R=1.85,
        ...     regime=Regime.STABLE)
        >>> verdict
        <EpistemicVerdict.RETURN: 'return'>
        >>> reasons
        []

    Reference: "The Seam of Reality" §3 (Paulus, 2025)
    """
    reasons: list[GestureReason] = []

    # 1. Dissolution: collapse regime overrides everything
    if regime == Regime.COLLAPSE:
        return EpistemicVerdict.DISSOLUTION, []

    # 2. Check return conditions
    if tau_R == tau_R_inf or not math.isfinite(tau_R):
        reasons.append(GestureReason.NO_FINITE_RETURN)

    if not seam_pass:
        # Parse specific failure reasons from seam_failures
        if seam_failures:
            for failure in seam_failures:
                if "tol_seam" in failure:
                    reasons.append(GestureReason.SEAM_RESIDUAL_EXCEEDED)
                elif "INF_REC" in failure or "not finite" in failure:
                    if GestureReason.NO_FINITE_RETURN not in reasons:
                        reasons.append(GestureReason.NO_FINITE_RETURN)
                elif "exp" in failure or "I_ratio" in failure:
                    reasons.append(GestureReason.IDENTITY_MISMATCH)
        elif not reasons:
            # Generic seam failure
            reasons.append(GestureReason.SEAM_RESIDUAL_EXCEEDED)

    # 3. Verdict
    if not reasons and seam_pass:
        return EpistemicVerdict.RETURN, []

    return EpistemicVerdict.GESTURE, reasons


def quantify_positional_illusion(
    omega: float,
    n_observations: int = 1,
    *,
    p: int = P_EXPONENT,
    epsilon: float = EPSILON,
    tol_seam: float = TOL_SEAM,
) -> PositionalIllusion:
    """Quantify the positional illusion for a given state.

    Theorem T9 (Zeno analog): N observations of a system at drift ω
    cost N × Γ(ω) in seam budget. The positional illusion is the
    belief that this cost is zero — that you can observe from outside.

    This function computes:
        - Γ(ω): the single-observation cost
        - N × Γ(ω): total observation cost
        - Budget fraction: what fraction of tol_seam is consumed
        - Illusion severity: at 1.0, observation alone exhausts
          the seam budget; above 1.0, the observer cannot even
          verify return

    The severity quantifies how dangerous the positional illusion is
    at this state point. In the STABLE regime (ω < 0.038), severity
    is low — the illusion is affordable. Near COLLAPSE (ω → 0.30),
    severity approaches and exceeds 1.0 — the illusion is fatal.

    Args:
        omega: Drift proxy (1 − F)
        n_observations: Number of observations (default 1)
        p: Contraction exponent (frozen at 3)
        epsilon: Guard band (frozen at 1e-8)
        tol_seam: Seam tolerance (frozen at 0.005)

    Returns:
        PositionalIllusion with cost decomposition

    Example:
        >>> pi = quantify_positional_illusion(0.031, n_observations=1)
        >>> pi.gamma  # Γ(0.031) ≈ 3.07e-5
        3.07...e-05
        >>> pi.illusion_severity < 0.01  # affordable at stable ω
        True

    Reference: tau_r_star.py Thm T9; "The Seam of Reality" §4.2
    """
    # Γ(ω) = ω^p / (1 - ω + ε) — the irreducible observation cost
    gamma = omega**p / (1.0 - omega + epsilon)
    total_cost = n_observations * gamma

    # Budget fraction: how much of the seam tolerance is consumed
    budget_fraction = total_cost / tol_seam if tol_seam > 0 else float("inf")

    # Illusion severity: fraction of budget consumed → 1.0 means fatal
    illusion_severity = budget_fraction

    return PositionalIllusion(
        gamma=gamma,
        n_observations=n_observations,
        total_cost=total_cost,
        budget_fraction=budget_fraction,
        illusion_severity=illusion_severity,
    )


def assess_epistemic_trace(
    n_components: int,
    n_timesteps: int,
    n_clipped: int = 0,
    is_degenerate: bool = False,
    *,
    seam_pass: bool = False,
    tau_R: float = float("inf"),
    regime: Regime = Regime.WATCH,
    epsilon: float = EPSILON,
) -> EpistemicTraceMetadata:
    """Assess the epistemic properties of a bounded trace Ψ(t).

    This function constructs the epistemic metadata for a trace,
    combining structural properties (dimensionality, clipping, degeneracy)
    with the epistemic verdict from the seam.

    The assessment embodies the principle that Ψ(t) is not data about
    the system — it is the system's epistemic emission. The metadata
    describes: how much of the emission hit the ε-boundary (components
    at risk of dissolution), whether the emission carries differential
    information (non-degenerate), and whether the emission earned
    return credit from the seam.

    Args:
        n_components: Dimensionality of the trace
        n_timesteps: Length of the temporal axis
        n_clipped: Number of components at the ε-floor
        is_degenerate: Whether all components are near-identical
        seam_pass: Whether the seam closed
        tau_R: Return time
        regime: Regime classification
        epsilon: ε-clamp value

    Returns:
        EpistemicTraceMetadata with verdict and structural properties

    Reference: measurement_engine.py (Ψ production), "The Seam of Reality" §2
    """
    verdict, _ = classify_epistemic_act(
        seam_pass=seam_pass,
        tau_R=tau_R,
        regime=regime,
    )

    clipped_fraction = n_clipped / n_components if n_components > 0 else 0.0

    return EpistemicTraceMetadata(
        n_components=n_components,
        n_timesteps=n_timesteps,
        epsilon_floor=epsilon,
        n_clipped=n_clipped,
        clipped_fraction=clipped_fraction,
        is_degenerate=is_degenerate,
        verdict=verdict,
    )


# =============================================================================
# SEAM EPISTEMOLOGY — Connecting Budget to Meaning
# =============================================================================


@dataclass(frozen=True)
class SeamEpistemology:
    """Complete epistemic assessment of a seam event.

    This is the highest-level output of the epistemic weld module:
    a single structure that unifies the seam calculus with its
    epistemic interpretation.

    The seam budget is: s = R·τ_R − (D_ω + D_C + Δκ)
    The seam closes iff: |s| ≤ tol_seam AND τ_R is finite
        AND |I_post/I_pre − exp(Δκ)| < tol_exp

    The epistemic verdict translates this into meaning:
        - s ≈ 0 AND closure → RETURN (what came back is real)
        - s ≠ 0 OR no closure → GESTURE (emission without return)
        - ω ≥ 0.30 → DISSOLUTION (trace has degraded past return viability)

    The positional illusion assessment adds:
        - How much budget was consumed just by observing
        - Whether the observer can even close a seam given observation costs

    Fields:
        verdict: The epistemic trichotomy
        reasons: Why GESTURE, if applicable
        seam_residual: |s| — the seam residual
        seam_budget: R·τ_R − (D_ω + D_C) — available Δκ budget
        tau_R: Return time
        regime: Regime classification
        illusion: Positional illusion assessment (if computed)
        omega: Drift at time of assessment
    """

    verdict: EpistemicVerdict
    reasons: list[GestureReason]
    seam_residual: float
    seam_budget: float
    tau_R: float
    regime: Regime
    omega: float
    illusion: PositionalIllusion | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON output."""
        result: dict[str, Any] = {
            "verdict": self.verdict.value,
            "reasons": [r.value for r in self.reasons],
            "seam_residual": self.seam_residual,
            "seam_budget": self.seam_budget,
            "tau_R": self.tau_R,
            "regime": self.regime.value,
            "omega": self.omega,
        }
        if self.illusion is not None:
            result["illusion"] = {
                "gamma": self.illusion.gamma,
                "n_observations": self.illusion.n_observations,
                "total_cost": self.illusion.total_cost,
                "budget_fraction": self.illusion.budget_fraction,
                "illusion_severity": self.illusion.illusion_severity,
            }
        return result

    @property
    def is_real(self) -> bool:
        """Whether this emission earned return credit.

        'Real' in the protocol sense: the seam closed, τ_R was finite,
        and the identity held. This is not a claim about metaphysical
        reality — it is a claim about closure under the frozen contract.
        """
        return self.verdict == EpistemicVerdict.RETURN

    @property
    def earned_credit(self) -> bool:
        """Alias for is_real — emphasizes the earning."""
        return self.is_real


def assess_seam_epistemology(
    seam_pass: bool,
    seam_failures: list[str],
    seam_residual: float,
    seam_budget: float,
    tau_R: float,
    omega: float,
    regime: Regime,
    *,
    n_observations: int = 1,
    compute_illusion: bool = True,
) -> SeamEpistemology:
    """Produce a complete epistemic assessment of a seam event.

    This is the primary entry point for epistemic analysis. It combines
    the seam calculus results with the epistemic verdict and optional
    positional illusion assessment.

    Args:
        seam_pass: Whether check_seam_pass() returned True
        seam_failures: List of failure reasons from check_seam_pass()
        seam_residual: Absolute seam residual |s|
        seam_budget: Available Δκ budget (R·τ_R − D_ω − D_C)
        tau_R: Return time
        omega: Drift proxy
        regime: Regime classification
        n_observations: Number of observations for illusion calc
        compute_illusion: Whether to compute positional illusion

    Returns:
        SeamEpistemology — the complete epistemic assessment

    Example:
        >>> epi = assess_seam_epistemology(
        ...     seam_pass=True, seam_failures=[],
        ...     seam_residual=0.0, seam_budget=1.697,
        ...     tau_R=1.85, omega=0.031, regime=Regime.STABLE)
        >>> epi.verdict
        <EpistemicVerdict.RETURN: 'return'>
        >>> epi.is_real
        True
    """
    verdict, reasons = classify_epistemic_act(
        seam_pass=seam_pass,
        tau_R=tau_R,
        regime=regime,
        seam_failures=seam_failures,
    )

    illusion = None
    if compute_illusion:
        illusion = quantify_positional_illusion(
            omega=omega,
            n_observations=n_observations,
        )

    return SeamEpistemology(
        verdict=verdict,
        reasons=reasons,
        seam_residual=seam_residual,
        seam_budget=seam_budget,
        tau_R=tau_R,
        regime=regime,
        omega=omega,
        illusion=illusion,
    )


# =============================================================================
# GESTURE DIAGNOSTICS — Understanding What Did Not Return
# =============================================================================


def diagnose_gesture(
    seam_residual: float,
    tau_R: float,
    omega: float,
    regime: Regime,
    *,
    tol_seam: float = TOL_SEAM,
) -> dict[str, Any]:
    """Produce diagnostic information for a gesture (non-returning emission).

    When something does not return, the question is not "why did it fail?"
    but "what would it take for it to return?" This function computes
    the distance from return — the gap between the current state and
    the nearest RETURN verdict.

    This is operationally useful: it tells you whether the gesture is
    "almost a return" (small residual, close to threshold) or
    "fundamentally a gesture" (deep in collapse, infinite τ_R, or
    large residual).

    Args:
        seam_residual: Absolute seam residual |s|
        tau_R: Return time
        omega: Drift proxy
        regime: Regime classification
        tol_seam: Seam tolerance

    Returns:
        Diagnostic dict with distance-to-return analysis
    """
    diagnostics: dict[str, Any] = {
        "regime": regime.value,
        "omega": omega,
    }

    # Distance from seam closure
    if tol_seam > 0:
        residual_excess = max(0.0, seam_residual - tol_seam)
        diagnostics["residual_excess"] = residual_excess
        diagnostics["residual_ratio"] = seam_residual / tol_seam
        diagnostics["near_return"] = residual_excess < tol_seam  # within 2× tol

    # Return analysis
    if math.isinf(tau_R):
        diagnostics["return_status"] = "INF_REC"
        diagnostics["return_commentary"] = (
            "No return within recovery horizon. The emission did not "
            "re-enter its admissible neighborhood. Without return, "
            "there is no epistemic credit — this is not a judgment "
            "but the definition of the protocol."
        )
    else:
        diagnostics["return_status"] = "finite"
        diagnostics["tau_R"] = tau_R

    # Regime analysis
    if regime == Regime.COLLAPSE:
        diagnostics["dissolution_commentary"] = (
            "System in collapse regime (ω ≥ 0.30). The epistemic "
            "trace has degraded past the point of viable return credit. "
            "This is not failure — it is the boundary condition that "
            "gives return its meaning. Collapse is the audit structure. "
            "Without the possibility of dissolution, the seam would "
            "audit nothing."
        )
    elif regime == Regime.WATCH:
        diagnostics["watch_commentary"] = (
            "System in watch regime (0.038 ≤ ω < 0.30). Return is "
            "still structurally possible but the curvature cost αC "
            "dominates the budget. The system is between collapse and "
            "stability — the most epistemically uncertain region."
        )

    # Observation cost context
    gamma = omega**P_EXPONENT / (1.0 - omega + EPSILON)
    diagnostics["observation_cost_gamma"] = gamma
    diagnostics["positional_illusion_note"] = (
        f"Each observation costs Γ(ω) = {gamma:.6e} in seam budget. "
        f"There is no free observation — the positional illusion "
        f"(belief in external vantage) costs at minimum this amount."
    )

    return diagnostics


# =============================================================================
# SUMMARY CONSTANTS
# =============================================================================


# The three verdicts as a frozen set for validation
VALID_VERDICTS = frozenset(EpistemicVerdict)

# Human-readable verdict descriptions
VERDICT_DESCRIPTIONS: dict[EpistemicVerdict, str] = {
    EpistemicVerdict.RETURN: (
        "The seam closed. What returned is consistent with what collapsed. "
        "This emission has earned epistemic credit under the frozen contract."
    ),
    EpistemicVerdict.GESTURE: (
        "The emission exists but the seam did not close. However internally "
        "consistent, it did not return through collapse under the frozen rules. "
        "A gesture has no epistemic standing — not because it is wrong, but "
        "because it did not complete the cycle that would make it decidable."
    ),
    EpistemicVerdict.DISSOLUTION: (
        "The epistemic trace has degraded past the point of viable return credit. "
        "Dissolution is not failure — it is the boundary condition that makes "
        "return meaningful. Without the possibility of dissolution, the seam "
        "would audit nothing."
    ),
}

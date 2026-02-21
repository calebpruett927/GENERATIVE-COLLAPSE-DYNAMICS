"""
Continuity Law Verifier — First-Class Budget Identity Check

Implements the continuity law verification as a reusable Tier-0 protocol
function, independent of any specific audit pipeline.

The continuity law states that under a frozen contract, the budget identity
must reconcile:

    Δκ = R·τ_R − (D_ω + D_C)

And the interpretive density ratio must satisfy:

    |ir − exp(Δκ)| ≤ tol_id

Where ir = I₁ / I₀ is the ratio of post-collapse to pre-collapse
integrity.

This module formalizes the paper's specification: "coherence can be computed,
compared, and audited across experiments, simulations, and document revisions
without interpretive drift."

Reference: The Physics of Coherence, §Continuity Laws, Clement Paulus (2025)
Cross-references:
    - frozen_contract.py   (budget identity, seam check)
    - seam_optimized.py    (accumulator)
    - KERNEL_SPECIFICATION.md §3 (budget identity derivation)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

from .frozen_contract import (
    ALPHA,
    EPSILON,
    LAMBDA,
    P_EXPONENT,
    TOL_SEAM,
    gamma_omega,
)


class ContinuityVerdict(NamedTuple):
    """Result of a continuity law verification.

    Fields:
        passes: Whether all continuity checks passed
        delta_kappa: Budget identity value Δκ = R·τ_R − (D_ω + D_C)
        ir: Interpretive density ratio I₁ / I₀
        ir_expected: Expected ratio exp(Δκ)
        identity_error: |ir − exp(Δκ)|
        residual: Seam residual (Δκ_budget − Δκ_ledger)
        failures: List of failure reasons (empty if passes)
    """

    passes: bool
    delta_kappa: float
    ir: float
    ir_expected: float
    identity_error: float
    residual: float
    failures: tuple[str, ...]


@dataclass(frozen=True)
class ContinuityLawSpec:
    """Specification for continuity law verification.

    All parameters are frozen per run — consistent across the seam.
    """

    tol_seam: float = TOL_SEAM
    tol_identity: float = 1e-6
    p: int = P_EXPONENT
    alpha: float = ALPHA
    lambda_drift: float = LAMBDA
    epsilon: float = EPSILON


# Default specification from frozen contract
DEFAULT_CONTINUITY_SPEC = ContinuityLawSpec()


def verify_continuity_law(
    I_pre: float,
    I_post: float,
    tau_R: float,
    omega: float,
    C: float,
    R: float = 1.0,
    delta_kappa_ledger: float | None = None,
    spec: ContinuityLawSpec = DEFAULT_CONTINUITY_SPEC,
) -> ContinuityVerdict:
    """
    Verify the continuity law for a collapse-return transition.

    Checks three conditions:
        1. Budget identity reconciles: |s| ≤ tol_seam
        2. τ_R is finite (not ∞_rec)
        3. Exponential identity: |I₁/I₀ − exp(Δκ)| ≤ tol_identity

    Args:
        I_pre: Pre-collapse integrity I₀ = exp(κ₀)
        I_post: Post-collapse integrity I₁ = exp(κ₁)
        tau_R: Return delay (finite for credit, inf for no return)
        omega: Drift ω = 1 − F
        C: Curvature
        R: Return credit [0, 1]
        delta_kappa_ledger: Observed ledger change κ₁ − κ₀.
            If None, computed from I_post/I_pre.
        spec: Continuity law specification (frozen parameters)

    Returns:
        ContinuityVerdict with pass/fail and all computed values
    """
    failures: list[str] = []

    # Compute costs from frozen parameters
    D_omega = gamma_omega(omega, p=spec.p, epsilon=spec.epsilon)
    D_C = spec.alpha * C

    # Budget identity: Δκ = R·τ_R − (D_ω + D_C)
    delta_kappa_budget = R * tau_R - (D_omega + D_C)

    # Interpretive density ratio
    if I_pre > 0:
        ir = I_post / I_pre
    else:
        ir = 0.0
        failures.append("I_pre = 0: cannot compute interpretive density ratio")

    # Expected ratio from budget
    ir_expected = math.exp(delta_kappa_budget)

    # Condition 1: τ_R is finite
    if not math.isfinite(tau_R):
        failures.append(f"τ_R = {tau_R}: not finite (∞_rec → zero credit)")

    # Condition 2: Budget reconciliation
    if delta_kappa_ledger is not None:
        residual = delta_kappa_budget - delta_kappa_ledger
    else:
        # Derive from integrity ratio
        if I_pre > 0 and I_post > 0:
            delta_kappa_ledger = math.log(I_post) - math.log(I_pre)
            residual = delta_kappa_budget - delta_kappa_ledger
        else:
            delta_kappa_ledger = 0.0
            residual = delta_kappa_budget
            failures.append("Cannot derive Δκ_ledger: I_pre or I_post ≤ 0")

    if abs(residual) > spec.tol_seam:
        failures.append(f"|s| = {abs(residual):.6f} > tol_seam = {spec.tol_seam}")

    # Condition 3: Exponential identity
    identity_error = abs(ir - ir_expected)
    if identity_error > spec.tol_identity:
        failures.append(f"|ir − exp(Δκ)| = {identity_error:.6e} > tol_id = {spec.tol_identity}")

    return ContinuityVerdict(
        passes=len(failures) == 0,
        delta_kappa=delta_kappa_budget,
        ir=ir,
        ir_expected=ir_expected,
        identity_error=identity_error,
        residual=residual,
        failures=tuple(failures),
    )


def verify_continuity_chain(
    transitions: list[dict[str, float]],
    spec: ContinuityLawSpec = DEFAULT_CONTINUITY_SPEC,
) -> list[ContinuityVerdict]:
    """
    Verify continuity law across a chain of transitions.

    Each transition dict must have keys:
        I_pre, I_post, tau_R, omega, C, R

    Args:
        transitions: List of transition parameter dicts
        spec: Continuity law specification

    Returns:
        List of ContinuityVerdict, one per transition
    """
    verdicts: list[ContinuityVerdict] = []
    for t in transitions:
        verdict = verify_continuity_law(
            I_pre=t["I_pre"],
            I_post=t["I_post"],
            tau_R=t["tau_R"],
            omega=t["omega"],
            C=t["C"],
            R=t.get("R", 1.0),
            delta_kappa_ledger=t.get("delta_kappa_ledger"),
            spec=spec,
        )
        verdicts.append(verdict)
    return verdicts

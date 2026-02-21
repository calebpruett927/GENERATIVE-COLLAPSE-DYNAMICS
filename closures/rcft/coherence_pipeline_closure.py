"""
Coherence Pipeline Closure — RCFT Tier-2

Formal closure proving coherence is a CONSEQUENCE of the measurement
pipeline, not an independent metric.

From "The Physics of Coherence" (Paulus, 2025):
    "Coherence begins only after observables are mapped through an explicit
    normalization into a bounded, dimensionless trace Ψ(t) ∈ [0,1]^n under
    a frozen normalization contract."

This closure takes raw observables and a frozen contract, runs them through
the full measurement pipeline (embedding → clip → Ψ(t) → kernel → τ_R →
regime), and produces a coherence derivation that demonstrates:

    1. Coherence is derived, never asserted
    2. The pipeline respects Tier-1 identity constraints (F + ω = 1, IC ≤ F)
    3. The budget identity reconciles under the frozen contract
    4. Return is measured, not assumed (τ_R computation)

The coherence verdict is: "Under this frozen contract, with these observables,
the system's Tier-1 invariants produce this regime — coherence is the name
we give to a Stable regime with verified return."

Tier-2 Constraints:
    - Does NOT redefine F, ω, S, C, κ, IC, τ_R, or regime
    - Does NOT modify Tier-0 protocol or Tier-1 identity behavior
    - Produces a diagnostic coherence derivation as a Tier-2 overlay
    - Equator deviation is included as a diagnostic, not a gate

Cross-references:
    - measurement_engine.py       (Ψ(t) production pipeline)
    - frozen_contract.py          (kernel, regime, equator_phi)
    - seam_optimized.py           (budget identity)
    - continuity_law.py           (verify_continuity_law)
    - closures/rcft/__init__.py   (RCFT registry)
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Add src to path for imports
_src_path = Path(__file__).parent.parent.parent / "src"
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

from umcp.frozen_contract import (  # noqa: E402
    EPSILON,
    Regime,
    classify_regime,
    compute_kernel,
    equator_phi,
)


@dataclass(frozen=True)
class CoherenceDerivation:
    """Result of the coherence pipeline closure.

    Coherence is derived from the measurement pipeline, not asserted.
    This dataclass records the full derivation chain from raw observables
    through kernel invariants to regime classification.

    The coherence_status is:
        - "COHERENT": Regime is STABLE with verified return (τ_R finite)
        - "DEGRADED": Regime is WATCH — return possible but uncertain
        - "INCOHERENT": Regime is COLLAPSE or CRITICAL — past viable return
        - "NON_EVALUABLE": Insufficient data for derivation

    This status is a Tier-2 diagnostic overlay. It does NOT override
    the Tier-0 regime classification. Promoting it to a gate requires
    explicit closure declaration.
    """

    # Pipeline inputs
    n_channels: int  # Number of measurement channels
    n_timesteps: int  # Number of timesteps in trace
    embedding_method: str  # How observables were normalized
    contract_epsilon: float  # Guard band ε from frozen contract

    # Tier-1 kernel invariants
    omega: float  # Drift ω = 1 - F
    F: float  # Fidelity (weighted mean)
    S: float  # Bernoulli field entropy
    C: float  # Curvature (coupling)
    kappa: float  # Log-integrity κ
    IC: float  # Integrity composite exp(κ)
    tau_R: float  # Return delay

    # Tier-1 identity checks
    duality_residual: float  # |F + ω − 1| (should be 0)
    integrity_bound_ok: bool  # IC ≤ F (limbus integritatis)
    exp_identity_ok: bool  # |IC − exp(κ)| < tol

    # Regime (Tier-0)
    regime: str  # STABLE / WATCH / COLLAPSE / CRITICAL

    # Tier-2 diagnostics (NOT gates)
    equator_deviation: float  # Φ_eq (diagnostic only)
    heterogeneity_gap: float  # Δ = F − IC

    # Coherence verdict (Tier-2 overlay)
    coherence_status: str  # COHERENT / DEGRADED / INCOHERENT / NON_EVALUABLE
    derivation_chain: str  # Human-readable derivation narrative

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "pipeline": {
                "n_channels": self.n_channels,
                "n_timesteps": self.n_timesteps,
                "embedding_method": self.embedding_method,
                "contract_epsilon": self.contract_epsilon,
            },
            "kernel_invariants": {
                "omega": self.omega,
                "F": self.F,
                "S": self.S,
                "C": self.C,
                "kappa": self.kappa,
                "IC": self.IC,
                "tau_R": self.tau_R,
            },
            "identity_checks": {
                "duality_residual": self.duality_residual,
                "integrity_bound_ok": self.integrity_bound_ok,
                "exp_identity_ok": self.exp_identity_ok,
            },
            "regime": self.regime,
            "diagnostics": {
                "equator_deviation": self.equator_deviation,
                "heterogeneity_gap": self.heterogeneity_gap,
            },
            "coherence_status": self.coherence_status,
            "derivation_chain": self.derivation_chain,
        }


def derive_coherence(
    trace: NDArray[np.floating[Any]],
    weights: NDArray[np.floating[Any]] | None = None,
    tau_R: float | None = None,
    eta: float = 0.1,
    H_rec: int = 64,
    embedding_method: str = "frozen_contract",
    epsilon: float = EPSILON,
    tol_exp: float = 1e-6,
) -> CoherenceDerivation:
    """
    Derive coherence from a bounded trace under the frozen contract.

    This is the central function of the closure. It takes a trace Ψ(t)
    that has already been embedded into [0,1]^n and produces a complete
    coherence derivation showing how coherence emerges from the pipeline.

    The derivation chain is:

        Raw observables
          → Frozen contract (normalization, guard band, face policy)
          → Bounded trace Ψ(t) ∈ [ε, 1−ε]^n
          → Tier-1 kernel: (ω, F, S, C, κ, IC)
          → Return computation: τ_R
          → Regime classification: (STABLE → COHERENT)
          → Identity checks: F + ω = 1, IC ≤ F, IC ≈ exp(κ)

    Args:
        trace: Bounded trace of shape (T, n) or (n,) for single timestep.
            Must be in [0, 1]^n (pre-embedded).
        weights: Channel weights (must sum to 1). If None, uniform weights.
        tau_R: Pre-computed return delay. If None, computed from trace.
        eta: Return threshold for τ_R computation.
        H_rec: Recovery horizon for τ_R computation.
        embedding_method: Name of the embedding used.
        epsilon: Guard band from frozen contract.
        tol_exp: Tolerance for exponential identity check.

    Returns:
        CoherenceDerivation recording the full pipeline output.
    """
    # Handle single timestep
    if trace.ndim == 1:
        trace = trace.reshape(1, -1)

    n_timesteps, n_channels = trace.shape

    # Default uniform weights
    if weights is None:
        weights = np.ones(n_channels) / n_channels
    else:
        weights = np.asarray(weights, dtype=float)
        weights = weights / weights.sum()

    # Clip trace to [ε, 1−ε] (frozen contract face policy)
    trace_safe = np.clip(trace, epsilon, 1 - epsilon)

    # Compute τ_R if not provided
    if tau_R is None:
        tau_R = (
            _compute_tau_R_simple(trace_safe, eta=eta, H_rec=H_rec)
            if n_timesteps > 1
            else float("inf")  # Single point cannot demonstrate return
        )

    # Compute kernel on the last timestep (representative)
    c = trace_safe[-1]
    kernel_result = compute_kernel(c, weights, tau_R, epsilon=epsilon)

    # Identity checks
    duality_residual = abs(kernel_result.F + kernel_result.omega - 1.0)
    integrity_bound_ok = kernel_result.IC <= kernel_result.F + 1e-12
    exp_identity_error = abs(kernel_result.IC - math.exp(kernel_result.kappa))
    exp_identity_ok = exp_identity_error < tol_exp

    # Regime classification
    regime = classify_regime(
        omega=kernel_result.omega,
        F=kernel_result.F,
        S=kernel_result.S,
        C=kernel_result.C,
        integrity=kernel_result.IC,
    )

    # Diagnostics (NOT gates)
    eq_dev = equator_phi(kernel_result.omega, kernel_result.F, kernel_result.C)
    het_gap = kernel_result.F - kernel_result.IC

    # Coherence status (Tier-2 overlay)
    if regime == Regime.STABLE and math.isfinite(tau_R):
        coherence_status = "COHERENT"
    elif regime == Regime.WATCH:
        coherence_status = "DEGRADED"
    elif regime in (Regime.COLLAPSE, Regime.CRITICAL):
        coherence_status = "INCOHERENT"
    else:
        coherence_status = "NON_EVALUABLE"

    # Build derivation chain narrative
    derivation = (
        f"Contract: ε={epsilon}, face=pre_clip → "
        f"Trace: Ψ(t) ∈ [{epsilon:.1e}, {1 - epsilon:.1e}]^{n_channels} "
        f"({n_timesteps} steps) → "
        f"Kernel: ω={kernel_result.omega:.6f}, F={kernel_result.F:.6f}, "
        f"S={kernel_result.S:.6f}, C={kernel_result.C:.6f}, "
        f"κ={kernel_result.kappa:.6f}, IC={kernel_result.IC:.6f} → "
        f"Return: τ_R={tau_R} → "
        f"Regime: {regime.value} → "
        f"Coherence: {coherence_status} "
        f"(derived, not asserted)"
    )

    return CoherenceDerivation(
        n_channels=n_channels,
        n_timesteps=n_timesteps,
        embedding_method=embedding_method,
        contract_epsilon=epsilon,
        omega=kernel_result.omega,
        F=kernel_result.F,
        S=kernel_result.S,
        C=kernel_result.C,
        kappa=kernel_result.kappa,
        IC=kernel_result.IC,
        tau_R=tau_R,
        duality_residual=duality_residual,
        integrity_bound_ok=integrity_bound_ok,
        exp_identity_ok=exp_identity_ok,
        regime=regime.value,
        equator_deviation=eq_dev,
        heterogeneity_gap=het_gap,
        coherence_status=coherence_status,
        derivation_chain=derivation,
    )


def _compute_tau_R_simple(
    trace: NDArray[np.floating[Any]],
    eta: float = 0.1,
    H_rec: int = 64,
) -> float:
    """
    Simple τ_R computation from trace.

    τ_R = min{Δt > 0 : ‖Ψ(t) − Ψ(t−Δt)‖₂ < η} within H_rec.

    Args:
        trace: Shape (T, n) bounded trace.
        eta: Return threshold.
        H_rec: Recovery horizon.

    Returns:
        Return delay τ_R, or float("inf") if no return.
    """
    T = trace.shape[0]
    if T < 2:
        return float("inf")

    current = trace[-1]
    for delta_t in range(1, min(H_rec + 1, T)):
        past = trace[-(delta_t + 1)]
        distance = float(np.sqrt(np.sum((current - past) ** 2)))
        if distance < eta:
            return float(delta_t)

    return float("inf")


def verify_coherence_is_derived(derivation: CoherenceDerivation) -> dict[str, Any]:
    """
    Meta-verification: confirm that coherence was derived, not asserted.

    Checks that all identity constraints hold and that the coherence
    status is consistent with the regime classification.

    Returns:
        Dict with verification results:
            - all_identities_hold: True if F+ω=1, IC≤F, IC≈exp(κ)
            - regime_consistent: True if coherence_status matches regime
            - verdict: "VERIFIED" or "INCONSISTENT"
    """
    identities_ok = derivation.duality_residual < 1e-10 and derivation.integrity_bound_ok and derivation.exp_identity_ok

    # Check consistency
    expected_status_map = {
        "STABLE": "COHERENT",
        "WATCH": "DEGRADED",
        "COLLAPSE": "INCOHERENT",
        "CRITICAL": "INCOHERENT",
    }
    expected = expected_status_map.get(derivation.regime, "NON_EVALUABLE")
    # Special case: STABLE with infinite τ_R is NON_EVALUABLE
    if derivation.regime == "STABLE" and not math.isfinite(derivation.tau_R):
        expected = "NON_EVALUABLE"

    regime_consistent = derivation.coherence_status == expected

    return {
        "all_identities_hold": identities_ok,
        "regime_consistent": regime_consistent,
        "verdict": "VERIFIED" if (identities_ok and regime_consistent) else "INCONSISTENT",
        "duality_residual": derivation.duality_residual,
        "integrity_bound_ok": derivation.integrity_bound_ok,
        "exp_identity_ok": derivation.exp_identity_ok,
    }

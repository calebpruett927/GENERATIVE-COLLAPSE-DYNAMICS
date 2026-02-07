"""Uncertainty Principle Closure — QM.INTSTACK.v1

Computes Heisenberg uncertainty bounds for conjugate variable pairs.

Physics:
  - Heisenberg:  Δx · Δp ≥ ℏ/2
  - Heisenberg ratio:  R = (Δx · Δp) / (ℏ/2), must be ≥ 1
  - Minimum uncertainty:  R = 1 (Gaussian wavepacket)
  - Energy-time:  ΔE · Δt ≥ ℏ/2

UMCP integration:
  ω = max(0, 1 - 1/R)  (closer to minimum uncertainty → lower drift)
  F = 1/R clamped       (minimum uncertainty state = maximum fidelity)
  S = log(R)            (excess uncertainty → entropy)
  If R < 1: NONCONFORMANT — Heisenberg violation indicates measurement error

Regime (based on heisenberg_ratio):
  Minimum:     1.0 ≤ R < 1.5   (near Gaussian limit)
  Moderate:    1.5 ≤ R < 5.0   (reasonable quantum state)
  Dispersed:   5.0 ≤ R < 20.0  (thermal broadening)
  Classical:   R ≥ 20.0        (effectively classical regime)
  Violation:   R < 1.0         (measurement error — NONCONFORMANT)

Cross-references:
  Contract:  contracts/QM.INTSTACK.v1.yaml
  Canon:     canon/qm_anchors.yaml
  Registry:  closures/registry.yaml (extensions.quantum_mechanics)
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Any, NamedTuple


class UncertaintyRegime(StrEnum):
    """Regime based on Heisenberg ratio R = (Δx·Δp)/(ℏ/2)."""

    MINIMUM = "Minimum"
    MODERATE = "Moderate"
    DISPERSED = "Dispersed"
    CLASSICAL = "Classical"
    VIOLATION = "Violation"


class UncertaintyResult(NamedTuple):
    """Result of uncertainty principle computation."""

    heisenberg_product: float  # Δx · Δp (J·s or eV·s)
    heisenberg_ratio: float  # (Δx · Δp) / (ℏ/2)
    min_uncertainty: float  # ℏ/2 (the minimum bound)
    regime: str  # Regime classification


# ── Constants ────────────────────────────────────────────────────
HBAR = 1.054571817e-34  # ℏ (J·s)
HBAR_OVER_2 = HBAR / 2.0  # ℏ/2 — the Heisenberg bound
EV_TO_J = 1.602176634e-19  # eV → J
NM_TO_M = 1e-9  # nm → m

# Regime thresholds (on ratio R)
THRESH_MINIMUM = 1.5
THRESH_MODERATE = 5.0
THRESH_DISPERSED = 20.0


def _classify_regime(ratio: float) -> UncertaintyRegime:
    """Classify Heisenberg ratio regime."""
    if ratio < 1.0:
        return UncertaintyRegime.VIOLATION
    if ratio < THRESH_MINIMUM:
        return UncertaintyRegime.MINIMUM
    if ratio < THRESH_MODERATE:
        return UncertaintyRegime.MODERATE
    if ratio < THRESH_DISPERSED:
        return UncertaintyRegime.DISPERSED
    return UncertaintyRegime.CLASSICAL


def compute_uncertainty(
    delta_x: float,
    delta_p: float,
    units: str = "SI",
) -> dict[str, Any]:
    """Compute uncertainty principle outputs for UMCP validation.

    Parameters
    ----------
    delta_x : float
        Position uncertainty. In "SI" mode: meters. In "natural" mode: nm.
    delta_p : float
        Momentum uncertainty. In "SI" mode: kg·m/s. In "natural" mode: eV/c.
    units : str
        "SI" (default) or "natural" (nm, eV/c).

    Returns
    -------
    dict with keys: heisenberg_product, heisenberg_ratio, min_uncertainty, regime
    """
    # Convert to SI if needed
    if units == "natural":
        dx_si = delta_x * NM_TO_M
        # p in eV/c → SI: p_SI = p_eV * eV_to_J / c
        dp_si = delta_p * EV_TO_J / 2.99792458e8
    else:
        dx_si = delta_x
        dp_si = delta_p

    # Heisenberg product
    product = dx_si * dp_si

    # Heisenberg ratio
    ratio = product / HBAR_OVER_2 if HBAR_OVER_2 > 0 else float("inf")

    regime = _classify_regime(ratio)

    return {
        "heisenberg_product": product,
        "heisenberg_ratio": round(ratio, 6),
        "min_uncertainty": HBAR_OVER_2,
        "regime": regime.value,
    }


# ── Self-test ────────────────────────────────────────────────────
if __name__ == "__main__":
    # Minimum uncertainty state: Δx · Δp = ℏ/2
    # Δx = √(ℏ/(2mω)) for ground state, let's use exact values
    dx = math.sqrt(HBAR_OVER_2)
    dp = math.sqrt(HBAR_OVER_2)
    result = compute_uncertainty(dx, dp)
    print(
        f"Minimum:   product={result['heisenberg_product']:.4e} "
        f"ratio={result['heisenberg_ratio']:.4f} "
        f"regime={result['regime']}"
    )
    assert result["regime"] == "Minimum"
    assert abs(result["heisenberg_ratio"] - 1.0) < 0.01

    # Thermal state (larger uncertainty)
    result = compute_uncertainty(1e-10, 1e-24)
    print(
        f"Thermal:   product={result['heisenberg_product']:.4e} "
        f"ratio={result['heisenberg_ratio']:.4f} "
        f"regime={result['regime']}"
    )
    assert result["heisenberg_ratio"] >= 1.0

    # Classical regime (macroscopic)
    result = compute_uncertainty(1e-6, 1e-20)
    print(
        f"Classical: product={result['heisenberg_product']:.4e} "
        f"ratio={result['heisenberg_ratio']:.4f} "
        f"regime={result['regime']}"
    )

    # Violation test (should be NONCONFORMANT indicator)
    result = compute_uncertainty(1e-40, 1e-40)
    print(
        f"Violation: product={result['heisenberg_product']:.4e} "
        f"ratio={result['heisenberg_ratio']:.6f} "
        f"regime={result['regime']}"
    )
    assert result["regime"] == "Violation"

    print("✓ uncertainty_principle self-test passed")

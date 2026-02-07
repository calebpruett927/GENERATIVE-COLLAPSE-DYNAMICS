"""Quantum Harmonic Oscillator Closure — QM.INTSTACK.v1

Computes energy quantization and coherent state properties for
the quantum harmonic oscillator.

Physics:
  - Energy levels:  E_n = ℏω(n + ½)
  - Zero-point energy:  E_0 = ℏω/2
  - Level spacing:  ΔE = ℏω (equidistant)
  - Coherent state:  |α⟩ = exp(-|α|²/2) Σ (αⁿ/√n!) |n⟩
  - Mean photon number:  ⟨n⟩ = |α|²

UMCP integration:
  ω_drift = |E_obs - E_predicted| / E_predicted  (energy deviation as drift)
  F = 1 - ω_drift                                 (fidelity to quantized levels)
  S = 1/(n+1) inverted                            (higher n → more states → more entropy)

Regime classification (coherence_quality — based on energy fidelity):
  Pure:       fidelity ≥ 0.99  (energy matches E_n perfectly)
  High:       0.90 ≤ fidelity < 0.99
  Mixed:      0.50 ≤ fidelity < 0.90
  Decoherent: fidelity < 0.50

Cross-references:
  Contract:  contracts/QM.INTSTACK.v1.yaml
  Canon:     canon/qm_anchors.yaml
  Registry:  closures/registry.yaml (extensions.quantum_mechanics)
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Any, NamedTuple


class OscillatorRegime(StrEnum):
    """Regime based on energy quantization fidelity."""

    PURE = "Pure"
    HIGH = "High"
    MIXED = "Mixed"
    DECOHERENT = "Decoherent"


class OscillatorResult(NamedTuple):
    """Result of harmonic oscillator computation."""

    E_predicted: float  # Predicted energy E_n = ℏω(n + ½)
    delta_E: float  # |E_obs - E_predicted| / E_predicted
    coherent_alpha: float  # Coherent state parameter |α|
    squeeze_r: float  # Squeeze parameter r (0 = vacuum, > 0 = squeezed)
    regime: str  # Regime classification


# ── Constants ────────────────────────────────────────────────────
HBAR = 1.054571817e-34  # ℏ (J·s)
EV_TO_J = 1.602176634e-19  # eV → J conversion

# Regime thresholds (on fidelity = 1 - delta_E)
THRESH_PURE = 0.99
THRESH_HIGH = 0.90
THRESH_MIXED = 0.50


def _classify_regime(fidelity: float) -> OscillatorRegime:
    """Classify coherence quality regime."""
    if fidelity >= THRESH_PURE:
        return OscillatorRegime.PURE
    if fidelity >= THRESH_HIGH:
        return OscillatorRegime.HIGH
    if fidelity >= THRESH_MIXED:
        return OscillatorRegime.MIXED
    return OscillatorRegime.DECOHERENT


def compute_harmonic_oscillator(
    n_quanta: int,
    omega_freq: float,
    e_observed: float,
    coherent_alpha: float = 0.0,
    squeeze_r: float = 0.0,
) -> dict[str, Any]:
    """Compute harmonic oscillator outputs for UMCP validation.

    Parameters
    ----------
    n_quanta : int
        Quantum number n (ground state = 0).
    omega_freq : float
        Angular frequency ω (rad/s) or if < 1e10, treated as eV for ℏω.
    e_observed : float
        Experimentally observed energy (eV).
    coherent_alpha : float
        Coherent state parameter |α|. If 0, computed from n → |α|² = ⟨n⟩.
    squeeze_r : float
        Squeeze parameter r. 0 = vacuum/coherent state.

    Returns
    -------
    dict with keys: E_predicted, delta_E, coherent_alpha, squeeze_r, regime
    """
    # Compute predicted energy E_n = ℏω(n + 1/2)
    # If omega_freq is small, interpret as ℏω in eV directly
    hw_ev = omega_freq if omega_freq < 1e10 else HBAR * omega_freq / EV_TO_J

    e_predicted = hw_ev * (n_quanta + 0.5)

    # Energy deviation
    delta_e = abs(e_observed - e_predicted) / e_predicted if e_predicted > 0 else 1.0 if e_observed != 0 else 0.0

    # Fidelity
    fidelity = max(0.0, 1.0 - delta_e)

    # Coherent state parameter
    alpha = coherent_alpha
    if alpha == 0.0 and n_quanta > 0:
        # For a Fock state |n⟩, mean photon number = n
        # Coherent state with same mean: |α|² = n
        alpha = math.sqrt(float(n_quanta))

    regime = _classify_regime(fidelity)

    return {
        "E_predicted": round(e_predicted, 8),
        "delta_E": round(delta_e, 8),
        "coherent_alpha": round(alpha, 6),
        "squeeze_r": round(squeeze_r, 6),
        "regime": regime.value,
    }


# ── Self-test ────────────────────────────────────────────────────
if __name__ == "__main__":
    # Ground state (n=0) of ℏω = 0.5 eV oscillator
    # E_0 = 0.5 * (0 + 0.5) = 0.25 eV
    result = compute_harmonic_oscillator(0, 0.5, 0.25)
    print(f"n=0: E_pred={result['E_predicted']:.4f} eV delta_E={result['delta_E']:.6f} regime={result['regime']}")
    assert result["regime"] == "Pure"
    assert abs(result["E_predicted"] - 0.25) < 1e-6

    # First excited state (n=1)
    result = compute_harmonic_oscillator(1, 0.5, 0.75)
    print(f"n=1: E_pred={result['E_predicted']:.4f} eV delta_E={result['delta_E']:.6f} regime={result['regime']}")
    assert result["regime"] == "Pure"

    # Slightly perturbed observation
    result = compute_harmonic_oscillator(3, 0.5, 1.80)
    print(f"n=3: E_pred={result['E_predicted']:.4f} eV delta_E={result['delta_E']:.6f} regime={result['regime']}")

    # Highly off measurement
    result = compute_harmonic_oscillator(5, 0.5, 0.5)
    print(f"n=5 bad: E_pred={result['E_predicted']:.4f} eV delta_E={result['delta_E']:.6f} regime={result['regime']}")

    print("✓ harmonic_oscillator self-test passed")

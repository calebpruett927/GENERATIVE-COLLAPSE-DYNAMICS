"""Wavefunction Collapse Closure — QM.INTSTACK.v1

Computes Born rule probabilities, state fidelity, and purity for
quantum measurement processes.

Physics:
  - Born rule:  P(φ) = |⟨φ|ψ⟩|²
  - State fidelity:  F_q = |⟨ψ_ideal|ψ_exp⟩|²
  - Purity:  γ = Tr(ρ²) ∈ [1/d, 1] (d = Hilbert space dimension)

UMCP integration:
  ω = delta_P = deviation from Born rule prediction (drift)
  F = 1 - delta_P = fidelity to theoretical prediction
  S = 1 - purity = entropy proxy (mixed-state fraction)

Regime classification (born_deviation):
  Faithful:    delta_P < 0.01
  Perturbed:   0.01 ≤ delta_P < 0.05
  Decoherent:  0.05 ≤ delta_P < 0.15
  Anomalous:   delta_P ≥ 0.15

Cross-references:
  Contract:  contracts/QM.INTSTACK.v1.yaml
  Canon:     canon/qm_anchors.yaml
  Registry:  closures/registry.yaml (extensions.quantum_mechanics)
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Any, NamedTuple


class BornRegime(StrEnum):
    """Regime based on deviation from Born rule prediction."""

    FAITHFUL = "Faithful"
    PERTURBED = "Perturbed"
    DECOHERENT = "Decoherent"
    ANOMALOUS = "Anomalous"


class CollapseResult(NamedTuple):
    """Result of wavefunction collapse computation."""

    P_born: float  # Born rule predicted probability for observed outcome
    delta_P: float  # Deviation from Born rule
    fidelity_state: float  # Quantum state fidelity
    purity: float  # Tr(ρ²)
    regime: str  # Regime classification


# ── Regime thresholds ────────────────────────────────────────────
THRESH_FAITHFUL = 0.01
THRESH_PERTURBED = 0.05
THRESH_DECOHERENT = 0.15


def _classify_regime(delta_p: float) -> BornRegime:
    """Classify Born rule deviation regime."""
    if delta_p < THRESH_FAITHFUL:
        return BornRegime.FAITHFUL
    if delta_p < THRESH_PERTURBED:
        return BornRegime.PERTURBED
    if delta_p < THRESH_DECOHERENT:
        return BornRegime.DECOHERENT
    return BornRegime.ANOMALOUS


def _compute_purity(eigenvalues: list[float]) -> float:
    """Compute purity Tr(ρ²) from density matrix eigenvalues.

    For a pure state: Tr(ρ²) = 1
    For maximally mixed: Tr(ρ²) = 1/d
    """
    return sum(ev**2 for ev in eigenvalues)


def _compute_fidelity(
    psi_amplitudes: list[float],
    measurement_probs: list[float],
) -> float:
    """Compute state fidelity between ideal and experimental distributions.

    Uses classical fidelity (Bhattacharyya coefficient):
      F = (Σ √(p_i · q_i))²
    """
    if len(psi_amplitudes) != len(measurement_probs):
        return 0.0

    bc = sum(math.sqrt(abs(p) * abs(q)) for p, q in zip(psi_amplitudes, measurement_probs, strict=True))
    return min(1.0, bc**2)


def compute_wavefunction_collapse(
    psi_amplitudes: list[float],
    measurement_probs: list[float],
    observed_outcome_idx: int = 0,
) -> dict[str, Any]:
    """Compute wavefunction collapse outputs for UMCP validation.

    Parameters
    ----------
    psi_amplitudes : list[float]
        Squared amplitudes |c_i|² of the wavefunction in measurement basis.
        Must sum to ~1 (Born rule normalization).
    measurement_probs : list[float]
        Experimentally observed probability distribution.
    observed_outcome_idx : int
        Index of the particular outcome being analyzed.

    Returns
    -------
    dict with keys: P_born, delta_P, fidelity_state, purity, regime
    """
    # Born rule prediction for the observed outcome
    p_born = abs(psi_amplitudes[observed_outcome_idx]) if 0 <= observed_outcome_idx < len(psi_amplitudes) else 0.0

    # Experimental probability for that outcome
    p_exp = abs(measurement_probs[observed_outcome_idx]) if 0 <= observed_outcome_idx < len(measurement_probs) else 0.0

    # Born rule deviation
    delta_p = abs(p_born - p_exp)

    # State fidelity between theoretical and experimental distributions
    fidelity = _compute_fidelity(psi_amplitudes, measurement_probs)

    # Purity from experimental distribution (treat as eigenvalues of effective ρ)
    purity = _compute_purity(measurement_probs)

    regime = _classify_regime(delta_p)

    return {
        "P_born": round(p_born, 6),
        "delta_P": round(delta_p, 6),
        "fidelity_state": round(fidelity, 6),
        "purity": round(purity, 6),
        "regime": regime.value,
    }


# ── Self-test ────────────────────────────────────────────────────
if __name__ == "__main__":
    # Perfect spin-1/2 measurement: |↑⟩ → 100% up
    result = compute_wavefunction_collapse(
        psi_amplitudes=[1.0, 0.0],
        measurement_probs=[1.0, 0.0],
        observed_outcome_idx=0,
    )
    print(
        f"Pure ↑: P={result['P_born']:.4f} delta_P={result['delta_P']:.4f} "
        f"fidelity={result['fidelity_state']:.4f} purity={result['purity']:.4f} "
        f"regime={result['regime']}"
    )
    assert result["regime"] == "Faithful"
    assert result["delta_P"] < 0.001

    # Equal superposition with slight deviation
    result = compute_wavefunction_collapse(
        psi_amplitudes=[0.5, 0.5],
        measurement_probs=[0.48, 0.52],
        observed_outcome_idx=0,
    )
    print(
        f"50/50: P={result['P_born']:.4f} delta_P={result['delta_P']:.4f} "
        f"fidelity={result['fidelity_state']:.4f} purity={result['purity']:.4f} "
        f"regime={result['regime']}"
    )

    print("✓ wavefunction_collapse self-test passed")

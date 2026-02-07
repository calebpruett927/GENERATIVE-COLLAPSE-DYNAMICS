"""Spin Measurement Closure — QM.INTSTACK.v1

Computes spin measurement outcomes, Zeeman splitting, and Larmor precession
for quantum spin systems in external magnetic fields.

Physics:
  - Spin projection:  S_z = mₛ ℏ, where mₛ ∈ {-s, -s+1, ..., s}
  - Zeeman splitting:  ΔE = g μ_B B (energy gap between adjacent mₛ levels)
  - Larmor precession:  ω_L = g e B / (2m) = g μ_B B / ℏ
  - Stern-Gerlach:  2s + 1 discrete deflections

UMCP integration:
  ω = 1 - spin_fidelity   (deviation from ideal eigenstate)
  F = spin_fidelity        (fidelity to expected mₛ eigenstate)

Regime (born_deviation regime applied to spin fidelity):
  Faithful:    spin_fidelity ≥ 0.99
  Perturbed:   0.95 ≤ spin_fidelity < 0.99
  Decoherent:  0.80 ≤ spin_fidelity < 0.95
  Anomalous:   spin_fidelity < 0.80

Cross-references:
  Contract:  contracts/QM.INTSTACK.v1.yaml
  Canon:     canon/qm_anchors.yaml
  Registry:  closures/registry.yaml (extensions.quantum_mechanics)
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Any, NamedTuple


class SpinRegime(StrEnum):
    """Regime based on spin measurement fidelity."""

    FAITHFUL = "Faithful"
    PERTURBED = "Perturbed"
    DECOHERENT = "Decoherent"
    ANOMALOUS = "Anomalous"


class SpinResult(NamedTuple):
    """Result of spin measurement computation."""

    S_z_predicted: float  # Predicted S_z eigenvalue (in ℏ units)
    spin_fidelity: float  # Fidelity to ideal eigenstate
    larmor_freq: float  # Larmor frequency (Hz)
    zeeman_split: float  # Zeeman energy splitting (eV)
    regime: str  # Regime classification


# ── Constants ────────────────────────────────────────────────────
HBAR = 1.054571817e-34  # ℏ (J·s)
MU_B = 9.2740100783e-24  # Bohr magneton (J/T)
EV_TO_J = 1.602176634e-19  # eV → J
G_ELECTRON = 2.00231930436  # Electron g-factor
G_PROTON = 5.5856946893  # Proton g-factor

# Regime thresholds
THRESH_FAITHFUL = 0.99
THRESH_PERTURBED = 0.95
THRESH_DECOHERENT = 0.80


def _classify_regime(fidelity: float) -> SpinRegime:
    """Classify spin measurement fidelity regime."""
    if fidelity >= THRESH_FAITHFUL:
        return SpinRegime.FAITHFUL
    if fidelity >= THRESH_PERTURBED:
        return SpinRegime.PERTURBED
    if fidelity >= THRESH_DECOHERENT:
        return SpinRegime.DECOHERENT
    return SpinRegime.ANOMALOUS


def compute_spin_measurement(
    s_total: float,
    s_z_observed: float,
    b_field: float,
    g_factor: float | None = None,
) -> dict[str, Any]:
    """Compute spin measurement outputs for UMCP validation.

    Parameters
    ----------
    s_total : float
        Total spin quantum number s (0.5 for electron, 1 for photon, etc.).
    s_z_observed : float
        Observed S_z value in ℏ units (should be one of -s, -s+1, ..., s).
    b_field : float
        External magnetic field strength (Tesla).
    g_factor : float | None
        Landé g-factor. Defaults to electron g-factor.

    Returns
    -------
    dict with keys: S_z_predicted, spin_fidelity, larmor_freq, zeeman_split, regime
    """
    g = g_factor if g_factor is not None else G_ELECTRON

    # Valid m_s values: -s, -s+1, ..., s (2s+1 values)
    valid_ms = [s_total - k for k in range(int(2 * s_total + 1))]

    # Find closest valid m_s eigenvalue
    s_z_predicted = min(valid_ms, key=lambda ms: abs(ms - s_z_observed)) if valid_ms else 0.0

    # Spin fidelity: how close is the observed value to the nearest eigenvalue
    # Perfect measurement → s_z_observed is exactly an eigenvalue → fidelity = 1
    deviation = abs(s_z_observed - s_z_predicted)
    max_range = 2 * s_total if s_total > 0 else 1.0
    spin_fidelity = max(0.0, 1.0 - deviation / max_range)

    # Zeeman splitting: ΔE = g μ_B B
    zeeman_j = g * MU_B * b_field
    zeeman_ev = zeeman_j / EV_TO_J if EV_TO_J > 0 else 0.0

    # Larmor frequency: ω_L = g μ_B B / ℏ → f_L = ω_L / (2π)
    larmor_hz = g * MU_B * b_field / (2.0 * math.pi * HBAR) if b_field > 0 else 0.0

    regime = _classify_regime(spin_fidelity)

    return {
        "S_z_predicted": round(s_z_predicted, 4),
        "spin_fidelity": round(spin_fidelity, 6),
        "larmor_freq": round(larmor_hz, 2),
        "zeeman_split": round(zeeman_ev, 8),
        "regime": regime.value,
    }


# ── Self-test ────────────────────────────────────────────────────
if __name__ == "__main__":
    # Perfect spin-1/2 measurement: observed exactly +0.5
    result = compute_spin_measurement(0.5, 0.5, 1.0)
    print(
        f"e⁻ ↑:    Sz_pred={result['S_z_predicted']:+.1f}ℏ "
        f"fidelity={result['spin_fidelity']:.4f} "
        f"f_L={result['larmor_freq']:.2e} Hz "
        f"ΔE={result['zeeman_split']:.6e} eV "
        f"regime={result['regime']}"
    )
    assert result["regime"] == "Faithful"
    assert result["spin_fidelity"] == 1.0

    # Slightly off measurement
    result = compute_spin_measurement(0.5, 0.48, 1.0)
    print(
        f"e⁻ ~↑:   Sz_pred={result['S_z_predicted']:+.1f}ℏ "
        f"fidelity={result['spin_fidelity']:.4f} regime={result['regime']}"
    )

    # Spin-1 particle in strong field
    result = compute_spin_measurement(1.0, 0.0, 5.0, g_factor=1.0)
    print(
        f"s=1 m=0: Sz_pred={result['S_z_predicted']:+.1f}ℏ "
        f"fidelity={result['spin_fidelity']:.4f} "
        f"ΔE={result['zeeman_split']:.6e} eV "
        f"regime={result['regime']}"
    )

    # Bad measurement: observed value between eigenstates
    result = compute_spin_measurement(0.5, 0.25, 1.0)
    print(
        f"e⁻ bad:  Sz_pred={result['S_z_predicted']:+.1f}ℏ "
        f"fidelity={result['spin_fidelity']:.4f} regime={result['regime']}"
    )

    print("✓ spin_measurement self-test passed")

"""Quantum Tunneling Closure — QM.INTSTACK.v1

Computes transmission coefficients for particles tunneling through
potential barriers.

Physics:
  - Rectangular barrier:  T ≈ exp(-2κL), κ = √(2m(V₀-E))/ℏ
  - Classical forbidden:  E < V₀ → T_classical = 0
  - T_ratio = T_quantum / T_classical → ∞ (division by zero → sentinel)

UMCP integration:
  ω = 1 - T_coeff  (opaque barrier → high drift from transmission)
  F = T_coeff       (transmission = fidelity to "particle passes through")
  For tunneling, the GCD mapping is inverted from the classical perspective:
  classically T = 0 is "faithful" to Newton, but in GCD, the generative
  collapse *through* the barrier is the real process.

Regime classification (tunneling_regime):
  Opaque:      T < 0.001
  Suppressed:  0.001 ≤ T < 0.05
  Moderate:    0.05 ≤ T < 0.3
  Transparent: T ≥ 0.3

Cross-references:
  Contract:  contracts/QM.INTSTACK.v1.yaml
  Canon:     canon/qm_anchors.yaml
  Registry:  closures/registry.yaml (extensions.quantum_mechanics)
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Any, NamedTuple


class TunnelingRegime(StrEnum):
    """Regime based on barrier transmission."""

    OPAQUE = "Opaque"
    SUPPRESSED = "Suppressed"
    MODERATE = "Moderate"
    TRANSPARENT = "Transparent"


class TunnelingResult(NamedTuple):
    """Result of tunneling computation."""

    T_coeff: float  # Transmission coefficient
    kappa_barrier: float  # Barrier decay constant (1/m)
    T_classical: float  # Classical transmission (0 if E < V₀)
    T_ratio: float  # Quantum/classical ratio (INF_REC if classical = 0)
    regime: str  # Regime classification


# ── Constants ────────────────────────────────────────────────────
HBAR = 1.054571817e-34  # ℏ (J·s)
M_ELECTRON = 9.1093837015e-31  # Electron mass (kg)
EV_TO_J = 1.602176634e-19  # eV → J conversion
NM_TO_M = 1e-9  # nm → m conversion
EXP_CUTOFF = 50.0  # Numerical stability cutoff for exp argument

# Regime thresholds
THRESH_OPAQUE = 0.001
THRESH_SUPPRESSED = 0.05
THRESH_MODERATE = 0.3


def _classify_regime(t_coeff: float) -> TunnelingRegime:
    """Classify tunneling transmission regime."""
    if t_coeff < THRESH_OPAQUE:
        return TunnelingRegime.OPAQUE
    if t_coeff < THRESH_SUPPRESSED:
        return TunnelingRegime.SUPPRESSED
    if t_coeff < THRESH_MODERATE:
        return TunnelingRegime.MODERATE
    return TunnelingRegime.TRANSPARENT


def compute_tunneling(
    e_particle: float,
    v_barrier: float,
    barrier_width: float,
    particle_mass: float | None = None,
) -> dict[str, Any]:
    """Compute quantum tunneling transmission for UMCP validation.

    Parameters
    ----------
    e_particle : float
        Particle kinetic energy (eV).
    v_barrier : float
        Barrier potential height (eV).
    barrier_width : float
        Barrier width (nm).
    particle_mass : float | None
        Particle mass (kg). Defaults to electron mass.

    Returns
    -------
    dict with keys: T_coeff, kappa_barrier, T_classical, T_ratio, regime
    """
    mass = particle_mass if particle_mass is not None else M_ELECTRON

    # Convert to SI
    e_j = e_particle * EV_TO_J
    v_j = v_barrier * EV_TO_J
    width_m = barrier_width * NM_TO_M

    # Classical transmission
    t_classical = 1.0 if e_j >= v_j else 0.0

    if e_j >= v_j:
        # Particle has enough energy — classically allowed, T ≈ 1
        # (ignoring reflection at barrier edges for simplicity)
        kappa = 0.0
        t_coeff = 1.0
    else:
        # Quantum tunneling: E < V₀
        # κ = √(2m(V₀ - E)) / ℏ
        delta_v = v_j - e_j
        kappa = math.sqrt(2.0 * mass * delta_v) / HBAR

        # T ≈ exp(-2κL)
        exponent = 2.0 * kappa * width_m
        t_coeff = 0.0 if exponent > EXP_CUTOFF else math.exp(-exponent)

    # Quantum/classical ratio
    # Classical says 0 but quantum says nonzero → ratio is infinite
    t_ratio = t_coeff / t_classical if t_classical > 0 else float("inf") if t_coeff > 0 else 0.0

    regime = _classify_regime(t_coeff)

    return {
        "T_coeff": round(t_coeff, 8),
        "kappa_barrier": round(kappa, 4) if kappa < 1e15 else round(kappa, 2),
        "T_classical": round(t_classical, 1),
        "T_ratio": t_ratio if not math.isinf(t_ratio) else "INF_REC",
        "regime": regime.value,
    }


# ── Self-test ────────────────────────────────────────────────────
if __name__ == "__main__":
    # Electron tunneling through 1 eV barrier at 0.5 eV, width 0.1 nm
    result = compute_tunneling(0.5, 1.0, 0.1)
    print(f"e⁻ thin:  T={result['T_coeff']:.6f} κ={result['kappa_barrier']:.2e} regime={result['regime']}")
    assert result["T_coeff"] > 0, "Should have nonzero tunneling"

    # Thick barrier — suppressed
    result = compute_tunneling(0.5, 1.0, 1.0)
    print(f"e⁻ thick: T={result['T_coeff']:.6f} κ={result['kappa_barrier']:.2e} regime={result['regime']}")
    assert result["T_coeff"] < 0.001

    # Classically allowed: E > V₀
    result = compute_tunneling(2.0, 1.0, 0.5)
    print(f"E > V₀:   T={result['T_coeff']:.6f} regime={result['regime']}")
    assert result["T_coeff"] == 1.0

    # Alpha decay: proton mass, 30 MeV barrier, 5 MeV energy, ~10 fm width
    result = compute_tunneling(5.0, 30.0, 0.01, particle_mass=1.67262192e-27)
    print(f"α-decay:  T={result['T_coeff']:.8f} κ={result['kappa_barrier']:.2e} regime={result['regime']}")

    print("✓ tunneling self-test passed")

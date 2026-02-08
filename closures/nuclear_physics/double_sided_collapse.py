"""Double-Sided Collapse Closure — NUC.INTSTACK.v1

Implements the O.NUC.DOUBLEFACE.v1 overlay: the double-sided
collapse dynamics where both fusion (light side, A < 62) and
decay/fission (heavy side, A > 62) converge on the iron peak
attractor (Ni-62, BE/A = 8.7945 MeV/nucleon).

Physics:
  signed_distance = (BE_per_A − BE_peak) / BE_peak
    > 0 → overbinding (impossible for SEMF, but shell effects)
    < 0 → underbinding → drift from peak
    = 0 → at peak (Ni-62)

  side:
    Light  → A < 62  → convergence via fusion (increases A)
    Peak   → A = 62  → at attractor
    Heavy  → A > 62  → convergence via decay/fission (decreases A)

  convergence_direction:
    "→Fe"  (light side, fusing toward iron)
    "≡Fe"  (at peak)
    "←Fe"  (heavy side, decaying toward iron)

This overlay tests the falsifiable prediction that ALL nuclides
converge toward the iron peak in the limit of sufficient time
and energy availability.

UMCP integration:
  The double-sided collapse map is a 1D signed coordinate system
  centered on the iron peak, with magnitude proportional to ω_eff.
  It is NOT a Ψ channel itself but an overlay that combines Ψ_BE
  with directional information.

Regime classification (convergence):
  AtPeak:     |signed_distance| < 0.005  (within 0.5% of peak)
  NearPeak:   0.005 ≤ |s.d.| < 0.02
  Convergent: 0.02 ≤ |s.d.| < 0.10
  Distant:    |s.d.| ≥ 0.10

Cross-references:
  Contract:  contracts/NUC.INTSTACK.v1.yaml  (AX-N4)
  Sources:   Burbidge et al. 1957 (B²FH); Woosley & Weaver 1995
"""

from __future__ import annotations

from enum import StrEnum
from typing import NamedTuple


class ConvergenceRegime(StrEnum):
    """Regime based on distance to iron peak."""

    AT_PEAK = "AtPeak"
    NEAR_PEAK = "NearPeak"
    CONVERGENT = "Convergent"
    DISTANT = "Distant"


class CollapseDirection(StrEnum):
    """Direction of convergence toward iron peak."""

    FUSING = "→Fe"  # Light side, A < 62, fusion path
    AT_PEAK = "≡Fe"  # At peak
    DECAYING = "←Fe"  # Heavy side, A > 62, decay/fission path


class DoubleSidedResult(NamedTuple):
    """Result of double-sided collapse analysis."""

    A: int  # Mass number
    BE_per_A: float  # Binding energy per nucleon
    signed_distance: float  # (BE/A - peak) / peak
    abs_distance: float  # |signed_distance|
    side: str  # "Light", "Peak", or "Heavy"
    convergence_direction: str  # "→Fe", "≡Fe", or "←Fe"
    omega_eff: float  # Effective drift from peak
    regime: str


# ── Frozen constants ─────────────────────────────────────────────
BE_PEAK_REF = 8.7945  # Ni-62 BE/A peak (MeV/nucleon)
A_PEAK = 62  # Peak mass number (Ni-62)

# Regime thresholds on |signed_distance|
THRESH_AT_PEAK = 0.005
THRESH_NEAR = 0.02
THRESH_CONVERGENT = 0.10


def _classify_regime(abs_dist: float) -> ConvergenceRegime:
    """Classify convergence regime."""
    if abs_dist < THRESH_AT_PEAK:
        return ConvergenceRegime.AT_PEAK
    if abs_dist < THRESH_NEAR:
        return ConvergenceRegime.NEAR_PEAK
    if abs_dist < THRESH_CONVERGENT:
        return ConvergenceRegime.CONVERGENT
    return ConvergenceRegime.DISTANT


def compute_double_sided(
    Z: int,
    A: int,
    BE_per_A: float,
) -> DoubleSidedResult:
    """Compute double-sided collapse position and convergence.

    Parameters
    ----------
    Z : int
        Atomic number (used for identification only).
    A : int
        Mass number.
    BE_per_A : float
        Binding energy per nucleon (MeV/nucleon).
        Use measured value from AME2020 when available.

    Returns
    -------
    DoubleSidedResult
    """
    # Signed distance from peak
    signed_dist = (BE_per_A - BE_PEAK_REF) / BE_PEAK_REF
    abs_dist = abs(signed_dist)

    # Side determination
    if A < A_PEAK:
        side = "Light"
        direction = CollapseDirection.FUSING
    elif A == A_PEAK:
        side = "Peak"
        direction = CollapseDirection.AT_PEAK
    else:
        side = "Heavy"
        direction = CollapseDirection.DECAYING

    # ω_eff = binding deficit from peak (always ≥ 0)
    omega_eff = max(0.0, 1.0 - BE_per_A / BE_PEAK_REF)

    regime = _classify_regime(abs_dist)

    return DoubleSidedResult(
        A=A,
        BE_per_A=round(BE_per_A, 4),
        signed_distance=round(signed_dist, 6),
        abs_distance=round(abs_dist, 6),
        side=side,
        convergence_direction=direction.value,
        omega_eff=round(omega_eff, 6),
        regime=regime.value,
    )


# ── Self-test ────────────────────────────────────────────────────
if __name__ == "__main__":
    # Representative nuclides spanning both sides
    tests = [
        (1, 1, 0.000, "H-1 (unbound)"),
        (1, 2, 1.112, "D-2"),
        (2, 4, 7.074, "He-4"),
        (6, 12, 7.680, "C-12"),
        (8, 16, 7.976, "O-16"),
        (14, 28, 8.448, "Si-28"),
        (26, 56, 8.790, "Fe-56"),
        (28, 58, 8.732, "Ni-58"),
        (28, 62, 8.795, "Ni-62 (peak)"),
        (37, 85, 8.697, "Rb-85"),
        (50, 120, 8.505, "Sn-120"),
        (82, 208, 7.868, "Pb-208"),
        (92, 238, 7.570, "U-238"),
    ]

    print(f"{'Nuclide':20s} {'A':>4s} {'BE/A':>7s} {'s.d.':>8s} {'Side':>6s} {'Dir':>4s} {'ω_eff':>7s} {'Regime':>12s}")
    print("-" * 80)
    for z, a, be, name in tests:
        r = compute_double_sided(z, a, be)
        print(
            f"{name:20s} {r.A:4d} {r.BE_per_A:7.3f} {r.signed_distance:+8.4f} "
            f"{r.side:>6s} {r.convergence_direction:>4s} {r.omega_eff:7.4f} {r.regime:>12s}"
        )

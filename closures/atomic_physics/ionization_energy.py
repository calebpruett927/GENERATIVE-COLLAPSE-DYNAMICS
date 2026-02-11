"""Ionization Energy Closure — ATOM.INTSTACK.v1

Computes ionization energies using the hydrogen-like model with quantum
defect corrections, maps to GCD invariants via binding deficit.

Physics:
  E_n = -13.6 eV · Z_eff² / n_eff²   (hydrogen-like with quantum defect)
  n_eff = n - δ_l                       (effective quantum number)
  Z_eff = Z - σ                         (Slater screening)
  IE₁ reference: H = 13.598 eV

UMCP integration:
  ω_eff = |IE_measured - IE_predicted| / IE_measured  (ionization drift)
  F_eff = 1 - ω_eff                                    (ionization fidelity)
  Ψ_IE = IE₁ / 13.598  clipped to [0,1]               (normalized trace)

Regime classification:
  Precise:       ω_eff < 0.01
  Approximate:   0.01 ≤ ω_eff < 0.05
  Screened:      0.05 ≤ ω_eff < 0.15
  Anomalous:     ω_eff ≥ 0.15

Cross-references:
  Contract:  contracts/ATOM.INTSTACK.v1.yaml
  Canon:     canon/atom_anchors.yaml
  Sources:   NIST ASD, Kramida et al. (2023)
"""

from __future__ import annotations

from enum import StrEnum
from typing import NamedTuple


class IonizationRegime(StrEnum):
    """Regime based on ionization energy prediction accuracy."""

    PRECISE = "Precise"
    APPROXIMATE = "Approximate"
    SCREENED = "Screened"
    ANOMALOUS = "Anomalous"


class IonizationResult(NamedTuple):
    """Result of ionization energy computation."""

    IE_predicted_eV: float  # Predicted ionization energy (eV)
    IE_measured_eV: float  # Measured/reference ionization energy (eV)
    Z_eff: float  # Effective nuclear charge (Slater)
    n_eff: float  # Effective principal quantum number
    omega_eff: float  # Drift from prediction
    F_eff: float  # Fidelity
    Psi_IE: float  # Normalized IE trace channel
    regime: str  # Regime classification


# ── Frozen constants ─────────────────────────────────────────────
RYDBERG_EV = 13.5984  # Hydrogen ground-state IE (eV)

# NIST reference first ionization energies (eV) for Z=1..36
NIST_IE1: dict[int, float] = {
    1: 13.598,
    2: 24.587,
    3: 5.392,
    4: 9.323,
    5: 8.298,
    6: 11.260,
    7: 14.534,
    8: 13.618,
    9: 17.423,
    10: 21.565,
    11: 5.139,
    12: 7.646,
    13: 5.986,
    14: 8.152,
    15: 10.487,
    16: 10.360,
    17: 12.968,
    18: 15.760,
    19: 4.341,
    20: 6.113,
    21: 6.562,
    22: 6.828,
    23: 6.746,
    24: 6.767,
    25: 7.434,
    26: 7.902,
    27: 7.881,
    28: 7.640,
    29: 7.726,
    30: 9.394,
    31: 5.999,
    32: 7.900,
    33: 9.789,
    34: 9.752,
    35: 11.814,
    36: 14.000,
}

# ── Aufbau filling order (n, l, capacity) ────────────────────────
_AUFBAU_FILL: list[tuple[int, int, int]] = [
    (1, 0, 2),  # 1s
    (2, 0, 2),  # 2s
    (2, 1, 6),  # 2p
    (3, 0, 2),  # 3s
    (3, 1, 6),  # 3p
    (4, 0, 2),  # 4s
    (3, 2, 10),  # 3d
    (4, 1, 6),  # 4p
    (5, 0, 2),  # 5s
    (4, 2, 10),  # 4d
    (5, 1, 6),  # 5p
    (6, 0, 2),  # 6s
    (4, 3, 14),  # 4f
    (5, 2, 10),  # 5d
    (6, 1, 6),  # 6p
    (7, 0, 2),  # 7s
    (5, 3, 14),  # 5f
    (6, 2, 10),  # 6d
    (7, 1, 6),  # 7p
]

# Regime thresholds
THRESH_PRECISE = 0.01
THRESH_APPROXIMATE = 0.05
THRESH_SCREENED = 0.15


def _classify_regime(omega_eff: float) -> IonizationRegime:
    """Classify ionization deviation regime."""
    if omega_eff < THRESH_PRECISE:
        return IonizationRegime.PRECISE
    if omega_eff < THRESH_APPROXIMATE:
        return IonizationRegime.APPROXIMATE
    if omega_eff < THRESH_SCREENED:
        return IonizationRegime.SCREENED
    return IonizationRegime.ANOMALOUS


def _slater_group_key(n: int, l_val: int) -> tuple[int, str]:
    """Map (n, l) to Slater group key.

    Slater groups: (1s)(2s,2p)(3s,3p)(3d)(4s,4p)(4d)(4f)…
    """
    if l_val <= 1:  # s or p
        return (n, "sp")
    if l_val == 2:  # d
        return (n, "d")
    return (n, "f")


def _slater_screening(Z: int, n: int, l_val: int = 0) -> float:
    """Compute Slater screening using proper Slater grouping rules (1930).

    Groups: (1s)(2s,2p)(3s,3p)(3d)(4s,4p)(4d)(4f)(5s,5p)…

    Rules for the electron being screened:
      1. Electrons in groups to the RIGHT contribute 0.
      2. Same group: 0.35 each (0.30 for 1s).
      3. For ns/np valence electron:
         - Electrons in (n-1) principal shell: 0.85 each
         - Electrons in ≤(n-2) shells: 1.00 each
      4. For nd/nf valence electron:
         - ALL electrons in inner groups: 1.00 each
    """
    # Fill electrons by Aufbau to determine group populations
    remaining = Z
    group_pops: dict[tuple[int, str], int] = {}
    last_n = 1
    for orb_n, orb_l, cap in _AUFBAU_FILL:
        if remaining <= 0:
            break
        pop = min(remaining, cap)
        gk = _slater_group_key(orb_n, orb_l)
        group_pops[gk] = group_pops.get(gk, 0) + pop
        remaining -= pop
        last_n = orb_n

    # Use the supplied (n, l) if given, else auto-detected last-filled
    val_n = n if n > 0 else last_n
    val_l = l_val
    val_gk = _slater_group_key(val_n, val_l)
    is_sp = val_l <= 1

    sigma = 0.0

    # Same-group screening (exclude the electron itself)
    same = group_pops.get(val_gk, 0) - 1
    if same > 0:
        sigma += same * (0.30 if val_gk == (1, "sp") else 0.35)

    # Inner-group screening
    for gk, pop in group_pops.items():
        if gk == val_gk:
            continue
        g_n, _g_type = gk

        if is_sp:
            # For s,p valence: n-1 shell → 0.85, ≤ n-2 → 1.00
            if g_n == val_n - 1:
                sigma += pop * 0.85
            elif g_n < val_n - 1:
                sigma += pop * 1.00
            # No contribution from groups with higher principal QN
        else:
            # For d,f valence: all inner groups → 1.00
            if g_n < val_n:
                sigma += pop * 1.00
            elif g_n == val_n and gk != val_gk:
                # Same-shell s,p group (e.g. 3s3p for 3d electron)
                sigma += pop * 1.00

    return sigma


def compute_ionization(
    Z: int,
    n: int = 0,
    *,
    IE_measured_eV: float | None = None,
    quantum_defect: float = 0.0,
) -> IonizationResult:
    """Compute ionization energy and GCD mapping.

    Parameters
    ----------
    Z : int
        Atomic number.
    n : int
        Principal quantum number of valence electron.  If 0, auto-assigned
        from periodic table period.
    IE_measured_eV : float | None
        Measured IE (eV).  Falls back to NIST reference or prediction.
    quantum_defect : float
        Quantum defect δ_l for the valence orbital.

    Returns
    -------
    IonizationResult
    """
    if Z < 1:
        msg = f"Atomic number Z must be ≥ 1, got {Z}"
        raise ValueError(msg)

    # Auto-detect valence (n, l) for ionization.
    # The ionized electron comes from the orbital with highest n
    # (and if tied, lowest l) — NOT the last Aufbau-filled orbital.
    # For transition metals: 4s ionizes before 3d despite 3d filling later.
    remaining = Z
    occupied: list[tuple[int, int, int]] = []  # (n, l, pop)
    for orb_n, orb_l, cap in _AUFBAU_FILL:
        if remaining <= 0:
            break
        pop = min(remaining, cap)
        occupied.append((orb_n, orb_l, pop))
        remaining -= pop

    # Find ionization orbital: highest n, then lowest l (most loosely bound)
    val_n, val_l = 1, 0
    if occupied:
        best = max(occupied, key=lambda x: (x[0], -x[1]))
        val_n, val_l = best[0], best[1]

    # Override n if explicitly provided
    if n > 0:
        val_n = n

    # Effective quantum number
    n_eff = val_n - quantum_defect

    # Proper Slater screening with l-awareness
    sigma = _slater_screening(Z, val_n, val_l)
    z_eff = max(1.0, Z - sigma)

    # Hydrogen-like prediction
    ie_predicted = RYDBERG_EV * z_eff**2 / n_eff**2

    # Measured/reference value
    if IE_measured_eV is not None:
        ie_measured = IE_measured_eV
    elif Z in NIST_IE1:
        ie_measured = NIST_IE1[Z]
    else:
        ie_measured = ie_predicted  # fallback

    # GCD mapping
    omega_eff = abs(ie_measured - ie_predicted) / ie_measured if ie_measured > 0 else 1.0

    omega_eff = min(1.0, omega_eff)
    f_eff = 1.0 - omega_eff
    psi_ie = max(0.0, min(1.0, ie_measured / RYDBERG_EV))

    regime = _classify_regime(omega_eff)

    return IonizationResult(
        IE_predicted_eV=round(ie_predicted, 4),
        IE_measured_eV=round(ie_measured, 4),
        Z_eff=round(z_eff, 4),
        n_eff=round(n_eff, 4),
        omega_eff=round(omega_eff, 6),
        F_eff=round(f_eff, 6),
        Psi_IE=round(psi_ie, 6),
        regime=regime.value,
    )


# ── Self-test ────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        (1, "Hydrogen"),
        (2, "Helium"),
        (6, "Carbon"),
        (11, "Sodium"),
        (26, "Iron"),
        (29, "Copper"),
    ]
    for z, name in tests:
        r = compute_ionization(z)
        print(
            f"{name:12s} (Z={z:2d})  IE_pred={r.IE_predicted_eV:8.3f}  "
            f"IE_meas={r.IE_measured_eV:8.3f}  ω={r.omega_eff:.4f}  {r.regime}"
        )

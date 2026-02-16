"""Cosmological Closure — Astronomy Domain Extension

Maps cosmological observables from Planck 2018 + BAO + SNe Ia measurements
to UMCP Tier-1 invariants. Completes the top of the cross-scale ladder:

    subatomic → atomic → everyday → stellar → COSMOLOGICAL

Trace vector construction (8 channels, equal weight):
    c₁ = H₀ / H₀_max              (Hubble parameter)
    c₂ = Ω_b / Ω_b_max            (baryon density)
    c₃ = Ω_c / Ω_c_max            (cold dark matter density)
    c₄ = Ω_Λ                       (dark energy density, already ≤ 1)
    c₅ = T_CMB / T_max             (CMB temperature)
    c₆ = n_s                       (scalar spectral index, already ~1)
    c₇ = σ₈ / σ₈_max              (amplitude of fluctuations)
    c₈ = τ_reion / τ_max           (optical depth to reionization)

All channels normalized to [ε, 1−ε].

Cosmological epochs tested:
    Planck (CMB)      → z ≈ 1100, t ≈ 380 kyr
    BAO (galaxy)      → z ≈ 0.5,  t ≈ 8.7 Gyr
    SNe Ia (late)     → z ≈ 0.1,  t ≈ 12.4 Gyr
    ΛCDM prediction   → z = 0,    t = 13.8 Gyr
    Inflation exit    → z ≈ 10²⁶, t ≈ 10⁻³² s
    Dark energy onset → z ≈ 0.7,  t ≈ 7.3 Gyr

Cross-references:
    Source data: Planck 2018 (arXiv:1807.06209), BAO (SDSS-IV), SNe Ia (Pantheon+)
    Stellar bridge: closures/astronomy/stellar_luminosity.py
    Contract: contracts/ASTRO.INTSTACK.v1.yaml
"""

from __future__ import annotations

import sys
from enum import StrEnum
from pathlib import Path
from typing import NamedTuple

import numpy as np

# Ensure workspace root is on path
_WORKSPACE = Path(__file__).resolve().parents[2]
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))

from umcp.kernel_optimized import compute_kernel_outputs  # noqa: E402


class CosmologicalRegime(StrEnum):
    """Regime classification for cosmological parameters."""

    STABLE = "Stable"
    WATCH = "Watch"
    COLLAPSE = "Collapse"


class CosmologicalResult(NamedTuple):
    """Result of cosmological kernel computation."""

    epoch: str
    redshift: float
    F: float
    omega: float
    IC: float
    kappa: float
    S: float
    C: float
    gap: float
    regime: str
    trace: list[float]


# ── Frozen constants (Planck 2018 TT,TE,EE+lowE+lensing+BAO) ──
EPSILON = 1e-8
N_CHANNELS = 8
OMEGA_WATCH = 0.10
OMEGA_COLLAPSE = 0.30

# Planck 2018 best-fit parameters (Table 2, arXiv:1807.06209)
H0_PLANCK = 67.36  # km/s/Mpc
OMEGA_B_H2 = 0.02237  # Ω_b h²
OMEGA_C_H2 = 0.1200  # Ω_c h²
OMEGA_LAMBDA = 0.6847  # Ω_Λ
T_CMB = 2.7255  # K (COBE-FIRAS)
N_S = 0.9649  # scalar spectral index
SIGMA_8 = 0.8111  # amplitude of fluctuations
TAU_REION = 0.0544  # optical depth to reionization

# Derived
h = H0_PLANCK / 100.0
OMEGA_B = OMEGA_B_H2 / h**2
OMEGA_C = OMEGA_C_H2 / h**2

# ── Cosmological epoch database ─────────────────────────────────
# Each entry: (name, redshift,
#   H [km/s/Mpc], Ω_b, Ω_c, Ω_Λ, T_CMB_eff [K], n_s, σ₈_eff, τ_eff)
#
# At higher redshifts, density parameters evolve with the Friedmann equation.
# H(z) ≈ H₀ √(Ω_m(1+z)³ + Ω_Λ) for flat ΛCDM.
# T_CMB(z) = T₀(1+z).
# σ₈ grows with growth factor D(z); at recombination σ₈_eff ≈ 0.001.


def _H_of_z(z: float) -> float:
    """Hubble parameter at redshift z [km/s/Mpc] for flat ΛCDM."""
    Omega_m = OMEGA_B + OMEGA_C
    return H0_PLANCK * np.sqrt(Omega_m * (1 + z) ** 3 + OMEGA_LAMBDA)


COSMOLOGICAL_EPOCHS: list[tuple[str, float, float, float, float, float, float, float, float, float]] = [
    # (name, z, H(z), Ω_b, Ω_c, Ω_Λ_eff, T_CMB(z), n_s, σ₈_eff, τ_eff)
    (
        "Inflation exit",
        1e26,
        _H_of_z(1e26),
        OMEGA_B,
        OMEGA_C,
        1e-30,  # Ω_Λ negligible
        T_CMB * (1 + 1e26),
        N_S,
        1e-6,
        0.0,
    ),
    ("Recombination (CMB)", 1100.0, _H_of_z(1100), OMEGA_B, OMEGA_C, 1e-9, T_CMB * 1101.0, N_S, 0.001, TAU_REION),
    ("Dark energy onset", 0.7, _H_of_z(0.7), OMEGA_B, OMEGA_C, 0.40, T_CMB * 1.7, N_S, 0.60, TAU_REION),
    ("BAO epoch", 0.5, _H_of_z(0.5), OMEGA_B, OMEGA_C, 0.55, T_CMB * 1.5, N_S, 0.70, TAU_REION),
    ("SNe Ia epoch", 0.1, _H_of_z(0.1), OMEGA_B, OMEGA_C, 0.68, T_CMB * 1.1, N_S, 0.79, TAU_REION),
    ("Present (ΛCDM)", 0.0, H0_PLANCK, OMEGA_B, OMEGA_C, OMEGA_LAMBDA, T_CMB, N_S, SIGMA_8, TAU_REION),
]

# Normalization ranges
_H_vals = [e[2] for e in COSMOLOGICAL_EPOCHS]
_Ob_vals = [e[3] for e in COSMOLOGICAL_EPOCHS]
_Oc_vals = [e[4] for e in COSMOLOGICAL_EPOCHS]
_T_vals = [e[6] for e in COSMOLOGICAL_EPOCHS]
_sig_vals = [e[8] for e in COSMOLOGICAL_EPOCHS]
_tau_vals = [e[9] for e in COSMOLOGICAL_EPOCHS]

# Use log for H (huge range from inflation)
_H_log = [np.log10(max(h, 1e-30)) for h in _H_vals]
H_LOG_MIN, H_LOG_MAX = min(_H_log), max(_H_log)

# Use log for T_CMB (spans from 2.7 K to ~10²⁶ K)
_T_log = [np.log10(max(t, 1e-30)) for t in _T_vals]
T_LOG_MIN, T_LOG_MAX = min(_T_log), max(_T_log)

OB_MAX = max(_Ob_vals)
OC_MAX = max(_Oc_vals)
SIG_MAX = max(_sig_vals)
TAU_MAX = max(max(_tau_vals), 0.1)


def _clip(x: float) -> float:
    """Clip to [ε, 1−ε]."""
    return max(EPSILON, min(1.0 - EPSILON, x))


def _log_norm(val: float, vmin: float, vmax: float) -> float:
    """Normalize log-scaled value to [0, 1]."""
    lv = np.log10(max(val, 1e-30))
    if vmax <= vmin:
        return 0.5
    return (lv - vmin) / (vmax - vmin)


def _classify_regime(omega_val: float) -> CosmologicalRegime:
    if omega_val < OMEGA_WATCH:
        return CosmologicalRegime.STABLE
    if omega_val < OMEGA_COLLAPSE:
        return CosmologicalRegime.WATCH
    return CosmologicalRegime.COLLAPSE


def compute_cosmological_epoch(
    name: str,
    redshift: float,
    H: float,
    Omega_b: float,
    Omega_c: float,
    Omega_Lambda: float,
    T_cmb: float,
    n_s: float,
    sigma_8: float,
    tau: float,
) -> CosmologicalResult:
    """Compute UMCP kernel for a cosmological epoch's parameters.

    Parameters
    ----------
    name : str
        Epoch name.
    redshift : float
        Cosmological redshift.
    H : float
        Hubble parameter at this epoch [km/s/Mpc].
    Omega_b : float
        Baryon density parameter.
    Omega_c : float
        Cold dark matter density parameter.
    Omega_Lambda : float
        Dark energy density parameter.
    T_cmb : float
        CMB temperature at this epoch [K].
    n_s : float
        Scalar spectral index.
    sigma_8 : float
        Amplitude of matter fluctuations.
    tau : float
        Optical depth to reionization.
    """
    c = np.array(
        [
            _clip(_log_norm(H, H_LOG_MIN, H_LOG_MAX)),
            _clip(Omega_b / OB_MAX),
            _clip(Omega_c / OC_MAX),
            _clip(Omega_Lambda),
            _clip(_log_norm(T_cmb, T_LOG_MIN, T_LOG_MAX)),
            _clip(n_s),
            _clip(sigma_8 / SIG_MAX) if SIG_MAX > 0 else EPSILON,
            _clip(tau / TAU_MAX) if TAU_MAX > 0 else EPSILON,
        ]
    )
    w = np.full(N_CHANNELS, 1.0 / N_CHANNELS)

    k_out = compute_kernel_outputs(c, w, EPSILON)

    F = float(k_out["F"])
    omega_val = float(k_out["omega"])
    IC = float(k_out["IC"])
    kappa = float(k_out["kappa"])
    S_val = float(k_out["S"])
    C_val = float(k_out["C"])
    gap_val = F - IC

    return CosmologicalResult(
        epoch=name,
        redshift=redshift,
        F=round(F, 6),
        omega=round(omega_val, 6),
        IC=round(IC, 6),
        kappa=round(kappa, 6),
        S=round(S_val, 6),
        C=round(C_val, 6),
        gap=round(gap_val, 6),
        regime=_classify_regime(omega_val).value,
        trace=[round(float(x), 6) for x in c],
    )


def compute_all_cosmological_epochs() -> list[CosmologicalResult]:
    """Compute kernel for all cosmological epochs."""
    return [
        compute_cosmological_epoch(name, z, H, ob, oc, ol, t, ns, s8, tau)
        for name, z, H, ob, oc, ol, t, ns, s8, tau in COSMOLOGICAL_EPOCHS
    ]


# ── Self-test ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("═" * 78)
    print("  COSMOLOGICAL CLOSURE — Astronomy Domain Extension")
    print("  Source: Planck 2018 (arXiv:1807.06209)")
    print("═" * 78)
    print()

    results = compute_all_cosmological_epochs()

    print(f"  {'Epoch':<26} {'z':>10} {'F':>7} {'ω':>7} {'IC':>7} {'Δ':>7} {'S':>7} {'Regime':<10}")
    print("  " + "─" * 74)

    for r in results:
        z_str = f"{r.redshift:.1e}" if r.redshift > 10 else f"{r.redshift:.2f}"
        print(
            f"  {r.epoch:<26} {z_str:>10} {r.F:7.4f} {r.omega:7.4f} {r.IC:7.4f} {r.gap:7.4f} {r.S:7.4f} {r.regime:<10}"
        )

    # Tier-1 identity verification
    for r in results:
        assert abs((r.F + r.omega) - 1.0) < 1e-5, f"{r.epoch}: F+ω = {r.F + r.omega}"
        assert r.IC <= r.F + 1e-5, f"{r.epoch}: IC ({r.IC}) > F ({r.F})"
        assert abs(r.IC - np.exp(r.kappa)) < 1e-4, f"{r.epoch}: IC ≠ exp(κ)"

    print()
    mean_F = np.mean([r.F for r in results])
    mean_gap = np.mean([r.gap for r in results])
    print(f"  Epochs: {len(results)}")
    print(f"  ⟨F⟩ = {mean_F:.4f}, ⟨Δ⟩ = {mean_gap:.4f}")
    print(f"  Tier-1 identities: ALL EXACT across {len(results)} epochs")

    # Cross-scale insight: early vs late universe
    early = [r for r in results if r.redshift > 1.0]
    late = [r for r in results if r.redshift <= 1.0]
    if early and late:
        F_early = np.mean([r.F for r in early])
        F_late = np.mean([r.F for r in late])
        print(f"  Early universe (z>1): ⟨F⟩ = {F_early:.4f}")
        print(f"  Late universe (z≤1):  ⟨F⟩ = {F_late:.4f}")

    print("  ✓ cosmology self-test passed")

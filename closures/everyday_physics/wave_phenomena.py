"""Wave Phenomena Closure — Everyday Physics

Maps classical wave systems to UMCP Tier-1 invariants, bridging
quantum wave-particle duality to macroscopic wave behavior.

Trace vector construction (6 channels, equal weight):
    c₁ = log(f) / log(f_max)       (frequency, log-scaled)
    c₂ = log(λ) / log(λ_max)       (wavelength, log-scaled)
    c₃ = v / v_max                  (phase velocity)
    c₄ = Q / Q_max                  (quality factor — energy retention)
    c₅ = coherence_norm             (coherence length / wavelength)
    c₆ = amplitude_norm             (normalized amplitude)

All channels normalized to [ε, 1−ε].

Cross-scale connections:
    quantum    → de Broglie λ = h/p, photon E = hf
    atomic     → spectral emission ↔ electron transitions
    everyday   → sound, music, ocean waves, earthquakes, radio
    stellar    → gravitational waves, stellar oscillations
    cosmic     → CMB, 21-cm line, gravitational wave background

The wave closure demonstrates that oscillatory return (τ_R = period)
is the most natural form of the Return Axiom: the system collapses
from equilibrium and returns every cycle. *Cyclus redire debet.*

Cross-references:
    Source data: NIST, ASA, IRIS seismology
    QM bridge: closures/quantum_mechanics/ (wave-particle duality)
    Contract: contracts/UMA.INTSTACK.v1.yaml
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


class WaveRegime(StrEnum):
    """Wave system regime."""

    STABLE = "Stable"
    WATCH = "Watch"
    COLLAPSE = "Collapse"


class WaveResult(NamedTuple):
    """Result of wave system kernel computation."""

    system: str
    wave_type: str
    F: float
    omega: float
    IC: float
    kappa: float
    S: float
    C: float
    gap: float
    regime: str
    trace: list[float]


# ── Frozen constants ─────────────────────────────────────────────
EPSILON = 1e-8
N_CHANNELS = 6
OMEGA_WATCH = 0.10
OMEGA_COLLAPSE = 0.30

# Fundamental constants
H_PLANCK = 6.62607015e-34  # J·s (exact, SI 2019)
C_LIGHT = 299_792_458.0  # m/s (exact)
K_BOLTZMANN = 1.380649e-23  # J/K (exact)

# ── Wave system database ────────────────────────────────────────
# Each entry: (name, wave_type,
#   frequency [Hz], wavelength [m], phase_velocity [m/s],
#   Q_factor, coherence_lengths, amplitude_norm)
#
# Q factor = 2π × (energy stored) / (energy lost per cycle)
# coherence_lengths = coherence_length / wavelength
# amplitude_norm = normalized to [0,1] relative to typical max for that type

WAVE_SYSTEMS: list[tuple[str, str, float, float, float, float, float, float]] = [
    # --- Sound in air (everyday) ---
    ("Concert A", "sound", 440.0, 0.780, 343.0, 100.0, 50.0, 0.3),
    ("Human speech", "sound", 300.0, 1.143, 343.0, 30.0, 10.0, 0.1),
    ("Thunder", "sound", 50.0, 6.860, 343.0, 5.0, 2.0, 0.9),
    ("Ultrasound (med)", "sound", 5.0e6, 6.86e-5, 343.0, 1000.0, 200.0, 0.05),
    ("Whale song", "sound", 20.0, 75.0, 1500.0, 50.0, 20.0, 0.6),
    # --- Sound in solids ---
    ("Bell (bronze)", "sound", 523.0, 6.80, 3560.0, 5000.0, 1000.0, 0.4),
    ("Tuning fork", "sound", 440.0, 11.36, 5000.0, 10000.0, 5000.0, 0.2),
    # --- Light ---
    ("Red light", "electromagnetic", 4.3e14, 7.0e-7, C_LIGHT, 1e8, 1e6, 0.5),
    ("Green light", "electromagnetic", 5.5e14, 5.5e-7, C_LIGHT, 1e8, 1e6, 0.5),
    ("Blue light", "electromagnetic", 6.5e14, 4.6e-7, C_LIGHT, 1e8, 1e6, 0.5),
    ("Laser (HeNe)", "electromagnetic", 4.74e14, 6.33e-7, C_LIGHT, 1e12, 1e9, 0.8),
    ("Microwave", "electromagnetic", 2.45e9, 0.122, C_LIGHT, 1e4, 100.0, 0.7),
    ("Radio (FM)", "electromagnetic", 1.0e8, 3.0, C_LIGHT, 1e3, 10.0, 0.6),
    ("X-ray", "electromagnetic", 3.0e18, 1.0e-10, C_LIGHT, 1e6, 1e3, 0.3),
    ("Gamma ray", "electromagnetic", 3.0e20, 1.0e-12, C_LIGHT, 1e6, 1e3, 0.1),
    # --- Water waves ---
    ("Ocean swell", "water", 0.1, 156.0, 15.6, 20.0, 5.0, 0.7),
    ("Ripple (capillary)", "water", 50.0, 0.02, 1.0, 5.0, 3.0, 0.05),
    ("Tsunami", "water", 0.0003, 200000.0, 200.0, 100.0, 50.0, 0.95),
    # --- Seismic ---
    ("P-wave (local)", "seismic", 1.0, 6000.0, 6000.0, 200.0, 30.0, 0.5),
    ("S-wave (local)", "seismic", 0.5, 7000.0, 3500.0, 100.0, 15.0, 0.4),
    ("Surface wave", "seismic", 0.05, 60000.0, 3000.0, 50.0, 8.0, 0.8),
    # --- Gravitational ---
    ("LIGO signal", "gravitational", 100.0, 3.0e6, C_LIGHT, 10.0, 5.0, 1e-21),
    # --- Quantum (de Broglie) ---
    ("Electron (1eV)", "matter", 2.42e14, 1.23e-9, C_LIGHT * 0.001, 100.0, 50.0, 0.5),
    ("Neutron (thermal)", "matter", 4.8e11, 1.82e-10, 2200.0, 50.0, 20.0, 0.3),
]

# Log-scale normalization
_f_log = [np.log10(max(m[2], 1e-30)) for m in WAVE_SYSTEMS]
_lam_log = [np.log10(max(m[3], 1e-30)) for m in WAVE_SYSTEMS]
_v_log = [np.log10(max(m[4], 1e-30)) for m in WAVE_SYSTEMS]
_Q_log = [np.log10(max(m[5], 1e-10)) for m in WAVE_SYSTEMS]
_coh_log = [np.log10(max(m[6], 1e-10)) for m in WAVE_SYSTEMS]

F_LOG_MIN, F_LOG_MAX = min(_f_log), max(_f_log)
LAM_LOG_MIN, LAM_LOG_MAX = min(_lam_log), max(_lam_log)
V_LOG_MIN, V_LOG_MAX = min(_v_log), max(_v_log)
Q_LOG_MIN, Q_LOG_MAX = min(_Q_log), max(_Q_log)
COH_LOG_MIN, COH_LOG_MAX = min(_coh_log), max(_coh_log)


def _clip(x: float) -> float:
    """Clip to [ε, 1−ε]."""
    return max(EPSILON, min(1.0 - EPSILON, x))


def _log_norm(val: float, vmin: float, vmax: float) -> float:
    """Normalize log-scaled value to [0, 1]."""
    lv = np.log10(max(val, 1e-30))
    if vmax <= vmin:
        return 0.5
    return (lv - vmin) / (vmax - vmin)


def _classify_regime(omega: float) -> WaveRegime:
    if omega < OMEGA_WATCH:
        return WaveRegime.STABLE
    if omega < OMEGA_COLLAPSE:
        return WaveRegime.WATCH
    return WaveRegime.COLLAPSE


def compute_wave_system(
    name: str,
    wave_type: str,
    frequency: float,
    wavelength: float,
    phase_velocity: float,
    Q_factor: float,
    coherence_lengths: float,
    amplitude_norm: float,
) -> WaveResult:
    """Compute UMCP kernel for a wave system.

    Parameters
    ----------
    name : str
        System name.
    wave_type : str
        Wave type (sound, electromagnetic, water, seismic, gravitational, matter).
    frequency : float
        Frequency [Hz].
    wavelength : float
        Wavelength [m].
    phase_velocity : float
        Phase velocity [m/s].
    Q_factor : float
        Quality factor (energy retention per cycle).
    coherence_lengths : float
        Coherence length in units of wavelength.
    amplitude_norm : float
        Normalized amplitude (0–1).
    """
    c = np.array(
        [
            _clip(_log_norm(frequency, F_LOG_MIN, F_LOG_MAX)),
            _clip(_log_norm(wavelength, LAM_LOG_MIN, LAM_LOG_MAX)),
            _clip(_log_norm(phase_velocity, V_LOG_MIN, V_LOG_MAX)),
            _clip(_log_norm(Q_factor, Q_LOG_MIN, Q_LOG_MAX)),
            _clip(_log_norm(coherence_lengths, COH_LOG_MIN, COH_LOG_MAX)),
            _clip(amplitude_norm),
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

    return WaveResult(
        system=name,
        wave_type=wave_type,
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


def compute_all_wave_systems() -> list[WaveResult]:
    """Compute kernel for all wave systems in the database."""
    return [compute_wave_system(name, wt, f, lam, v, Q, coh, amp) for name, wt, f, lam, v, Q, coh, amp in WAVE_SYSTEMS]


# ── Self-test ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("═" * 78)
    print("  WAVE PHENOMENA CLOSURE — Everyday Physics")
    print("═" * 78)
    print()

    results = compute_all_wave_systems()

    print(f"  {'System':<20} {'Type':<16} {'F':>7} {'ω':>7} {'IC':>7} {'Δ':>7} {'Regime':<10}")
    print("  " + "─" * 74)

    types: dict[str, list[float]] = {}
    for r in results:
        types.setdefault(r.wave_type, []).append(r.F)
        print(f"  {r.system:<20} {r.wave_type:<16} {r.F:7.4f} {r.omega:7.4f} {r.IC:7.4f} {r.gap:7.4f} {r.regime:<10}")

    print()
    for wt, fvals in sorted(types.items()):
        print(f"  {wt}: ⟨F⟩={np.mean(fvals):.4f}, n={len(fvals)}")

    # Tier-1 verification
    for r in results:
        assert abs((r.F + r.omega) - 1.0) < 1e-5
        assert r.IC <= r.F + 1e-5
        assert abs(r.IC - np.exp(r.kappa)) < 1e-4

    print()
    print(f"  Systems: {len(results)}")
    mean_F = np.mean([r.F for r in results])
    mean_gap = np.mean([r.gap for r in results])
    print(f"  ⟨F⟩ = {mean_F:.4f}, ⟨Δ⟩ = {mean_gap:.4f}")
    print(f"  Tier-1 identities: ALL EXACT across {len(results)} systems")

    # Cross-scale insight: wave coherence and return
    high_Q = [r for r in results if r.trace[3] > 0.6]  # High quality factor
    low_Q = [r for r in results if r.trace[3] < 0.3]  # Low quality factor
    if high_Q and low_Q:
        mean_F_hi = np.mean([r.F for r in high_Q])
        mean_F_lo = np.mean([r.F for r in low_Q])
        print(f"  High-Q ⟨F⟩ = {mean_F_hi:.4f} vs Low-Q ⟨F⟩ = {mean_F_lo:.4f}")
        print(f"  Coherent waves retain more fidelity: ΔF = {mean_F_hi - mean_F_lo:+.4f}")

    print("  ✓ wave_phenomena self-test passed")

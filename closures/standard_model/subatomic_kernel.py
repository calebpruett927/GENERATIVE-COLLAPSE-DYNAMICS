"""Subatomic Particle Kernel Analysis — GCD Tier-1 Applied to the Standard Model.

Extends the periodic table kernel analysis (118 elements) down to the
subatomic level: all 17 fundamental particles of the Standard Model,
plus 14 key composite hadrons (baryons + mesons).

For each particle, measurable properties are normalized into a trace
vector c ∈ [ε, 1−ε]^n, then fed through the GCD kernel to extract:

    F + ω = 1            (Conservation — closed process)
    IC ≤ F               (integrity bound — coherence bounded by fidelity)
    IC = exp(κ)          (Exp map — log/linear duality)
    Δ = F − IC           (heterogeneity gap — heterogeneity cost)

Normalization channels for FUNDAMENTAL particles (8 channels):
    1. mass_log:      log₁₀(m / m_floor) / log₁₀(m_ceil / m_floor)
    2. charge_abs:    |Q/e|
    3. spin_norm:     2s / 2   (0, 0.5, or 1.0)
    4. color_dof:     log₂(N_color + 1) / log₂(9)  (1→0.32, 3→0.64, 8→0.95)
    5. weak_T3:       (|T₃| + 0.5) / 1.5  (maps {0,½,1} → ~{0.33,0.67,1.0})
    6. hypercharge:   |Y / 2| / Y_max_norm
    7. generation:    gen / 3  (fermions: 0.33–1.0; bosons: ε)
    8. stability:     log₁₀(τ / τ_Planck) / log₁₀(τ_universe / τ_Planck)

Normalization channels for COMPOSITE particles (8 channels):
    1. mass_log:      same as fundamental
    2. charge_abs:    |Q/e|
    3. spin_norm:     2s / 2
    4. valence:       n_quarks / 3  (mesons=0.67, baryons=1.0)
    5. strangeness:   |S| / 3
    6. heavy_flavor:  (|C| + |B'|) / 2  (charm + beauty content)
    7. stability:     same as fundamental
    8. binding:       (Σm_quarks − m_hadron) / Σm_quarks  (binding fraction)

The kernel doesn't know what a quark is or what mass means.  It just sees
a vector of numbers in [ε, 1−ε] and computes F, IC, S, C, κ.
The patterns that emerge are pure mathematics applied to physics data.

Cross-references:
    Kernel:    src/umcp/kernel_optimized.py
    Catalog:   closures/standard_model/particle_catalog.py
    Contract:  contracts/SM.INTSTACK.v1.yaml
    Elements:  closures/atomic_physics/periodic_kernel.py (analogous for atoms)
    Proof:     closures/atomic_physics/tier1_proof.py (identity verification)
    Spec:      KERNEL_SPECIFICATION.md (Tier-1 identities, Lemmas 1-34)
    Axiom:     AXIOM.md (Axiom-0: collapse is generative)
"""

from __future__ import annotations

import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Ensure workspace root is on path
_WORKSPACE = Path(__file__).resolve().parents[2]
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))

from umcp.kernel_optimized import compute_kernel_outputs  # noqa: E402, I001


# ═══════════════════════════════════════════════════════════════════
# SECTION 1: PARTICLE DATABASE (PDG 2024)
# ═══════════════════════════════════════════════════════════════════

# Physical constants
TAU_PLANCK_S = 5.391e-44  # Planck time (s)
TAU_UNIVERSE_S = 4.35e17  # Age of universe (s)
M_FLOOR_GEV = 1e-11  # Neutrino mass floor (GeV)
M_CEIL_GEV = 200.0  # Above top quark (GeV)


@dataclass(frozen=True, slots=True)
class FundamentalParticle:
    """A fundamental Standard Model particle with full quantum numbers."""

    name: str
    symbol: str
    category: str  # Quark, Lepton, GaugeBoson, ScalarBoson
    mass_GeV: float  # Rest mass (GeV/c²), 0 for massless
    charge_e: float  # Electric charge in units of e
    spin: float  # Spin quantum number
    color_dof: int  # Color degrees of freedom: 1, 3, or 8
    generation: int  # 1, 2, 3 for fermions; 0 for bosons
    weak_T3: float  # Weak isospin 3rd component
    hypercharge_Y: float  # Weak hypercharge Y
    lifetime_s: float  # Mean lifetime (s), 0 = stable
    width_GeV: float  # Decay width (GeV), 0 = stable
    is_fermion: bool


@dataclass(frozen=True, slots=True)
class CompositeParticle:
    """A composite hadron (baryon or meson)."""

    name: str
    symbol: str
    hadron_type: str  # Baryon or Meson
    quark_content: str  # e.g. "uud", "ud̄"
    n_valence_quarks: int  # 2 for mesons, 3 for baryons
    mass_GeV: float
    charge_e: float
    spin: float
    strangeness: int  # |S| = number of strange quarks
    charm: int  # |C| content
    beauty: int  # |B'| content
    lifetime_s: float
    width_GeV: float
    constituent_mass_sum_GeV: float  # Sum of constituent quark masses


# ── PDG 2024 Quark masses (constituent / current) ──────────────
# Current quark masses (MS-bar) for fundamental
# Constituent masses for hadron binding fraction
_CONSTITUENT_MASS = {
    "u": 0.336,  # GeV (constituent)
    "d": 0.340,
    "s": 0.486,
    "c": 1.55,
    "b": 4.73,
    "t": 172.69,
}

# ── FUNDAMENTAL PARTICLES (17) ─────────────────────────────────
# All values: PDG 2024 (Particle Data Group)

FUNDAMENTAL_PARTICLES: tuple[FundamentalParticle, ...] = (
    # ── Quarks (Generation 1) ──
    FundamentalParticle(
        name="up",
        symbol="u",
        category="Quark",
        mass_GeV=0.00216,
        charge_e=2 / 3,
        spin=0.5,
        color_dof=3,
        generation=1,
        weak_T3=0.5,
        hypercharge_Y=1 / 3,
        lifetime_s=0,
        width_GeV=0,
        is_fermion=True,
    ),
    FundamentalParticle(
        name="down",
        symbol="d",
        category="Quark",
        mass_GeV=0.00467,
        charge_e=-1 / 3,
        spin=0.5,
        color_dof=3,
        generation=1,
        weak_T3=-0.5,
        hypercharge_Y=1 / 3,
        lifetime_s=0,
        width_GeV=0,
        is_fermion=True,
    ),
    # ── Quarks (Generation 2) ──
    FundamentalParticle(
        name="charm",
        symbol="c",
        category="Quark",
        mass_GeV=1.27,
        charge_e=2 / 3,
        spin=0.5,
        color_dof=3,
        generation=2,
        weak_T3=0.5,
        hypercharge_Y=1 / 3,
        lifetime_s=0,
        width_GeV=0,
        is_fermion=True,
    ),
    FundamentalParticle(
        name="strange",
        symbol="s",
        category="Quark",
        mass_GeV=0.093,
        charge_e=-1 / 3,
        spin=0.5,
        color_dof=3,
        generation=2,
        weak_T3=-0.5,
        hypercharge_Y=1 / 3,
        lifetime_s=0,
        width_GeV=0,
        is_fermion=True,
    ),
    # ── Quarks (Generation 3) ──
    FundamentalParticle(
        name="top",
        symbol="t",
        category="Quark",
        mass_GeV=172.69,
        charge_e=2 / 3,
        spin=0.5,
        color_dof=3,
        generation=3,
        weak_T3=0.5,
        hypercharge_Y=1 / 3,
        lifetime_s=5e-25,
        width_GeV=1.42,
        is_fermion=True,
    ),
    FundamentalParticle(
        name="bottom",
        symbol="b",
        category="Quark",
        mass_GeV=4.18,
        charge_e=-1 / 3,
        spin=0.5,
        color_dof=3,
        generation=3,
        weak_T3=-0.5,
        hypercharge_Y=1 / 3,
        lifetime_s=0,
        width_GeV=0,
        is_fermion=True,
    ),
    # ── Leptons (Generation 1) ──
    FundamentalParticle(
        name="electron",
        symbol="e⁻",
        category="Lepton",
        mass_GeV=0.000511,
        charge_e=-1.0,
        spin=0.5,
        color_dof=1,
        generation=1,
        weak_T3=-0.5,
        hypercharge_Y=-1.0,
        lifetime_s=0,
        width_GeV=0,
        is_fermion=True,
    ),
    FundamentalParticle(
        name="electron neutrino",
        symbol="ν_e",
        category="Lepton",
        mass_GeV=1e-11,
        charge_e=0.0,
        spin=0.5,
        color_dof=1,
        generation=1,
        weak_T3=0.5,
        hypercharge_Y=-1.0,
        lifetime_s=0,
        width_GeV=0,
        is_fermion=True,
    ),
    # ── Leptons (Generation 2) ──
    FundamentalParticle(
        name="muon",
        symbol="μ⁻",
        category="Lepton",
        mass_GeV=0.10566,
        charge_e=-1.0,
        spin=0.5,
        color_dof=1,
        generation=2,
        weak_T3=-0.5,
        hypercharge_Y=-1.0,
        lifetime_s=2.197e-6,
        width_GeV=3e-19,
        is_fermion=True,
    ),
    FundamentalParticle(
        name="muon neutrino",
        symbol="ν_μ",
        category="Lepton",
        mass_GeV=1e-11,
        charge_e=0.0,
        spin=0.5,
        color_dof=1,
        generation=2,
        weak_T3=0.5,
        hypercharge_Y=-1.0,
        lifetime_s=0,
        width_GeV=0,
        is_fermion=True,
    ),
    # ── Leptons (Generation 3) ──
    FundamentalParticle(
        name="tau",
        symbol="τ⁻",
        category="Lepton",
        mass_GeV=1.777,
        charge_e=-1.0,
        spin=0.5,
        color_dof=1,
        generation=3,
        weak_T3=-0.5,
        hypercharge_Y=-1.0,
        lifetime_s=2.903e-13,
        width_GeV=2.27e-12,
        is_fermion=True,
    ),
    FundamentalParticle(
        name="tau neutrino",
        symbol="ν_τ",
        category="Lepton",
        mass_GeV=1e-11,
        charge_e=0.0,
        spin=0.5,
        color_dof=1,
        generation=3,
        weak_T3=0.5,
        hypercharge_Y=-1.0,
        lifetime_s=0,
        width_GeV=0,
        is_fermion=True,
    ),
    # ── Gauge Bosons ──
    FundamentalParticle(
        name="photon",
        symbol="γ",
        category="GaugeBoson",
        mass_GeV=0.0,
        charge_e=0.0,
        spin=1.0,
        color_dof=1,
        generation=0,
        weak_T3=0.0,
        hypercharge_Y=0.0,
        lifetime_s=0,
        width_GeV=0,
        is_fermion=False,
    ),
    FundamentalParticle(
        name="W boson",
        symbol="W±",
        category="GaugeBoson",
        mass_GeV=80.377,
        charge_e=1.0,
        spin=1.0,
        color_dof=1,
        generation=0,
        weak_T3=1.0,
        hypercharge_Y=0.0,
        lifetime_s=3.17e-25,
        width_GeV=2.085,
        is_fermion=False,
    ),
    FundamentalParticle(
        name="Z boson",
        symbol="Z⁰",
        category="GaugeBoson",
        mass_GeV=91.1876,
        charge_e=0.0,
        spin=1.0,
        color_dof=1,
        generation=0,
        weak_T3=0.0,
        hypercharge_Y=0.0,
        lifetime_s=2.64e-25,
        width_GeV=2.4952,
        is_fermion=False,
    ),
    FundamentalParticle(
        name="gluon",
        symbol="g",
        category="GaugeBoson",
        mass_GeV=0.0,
        charge_e=0.0,
        spin=1.0,
        color_dof=8,
        generation=0,
        weak_T3=0.0,
        hypercharge_Y=0.0,
        lifetime_s=0,
        width_GeV=0,
        is_fermion=False,
    ),
    # ── Scalar Boson ──
    FundamentalParticle(
        name="Higgs",
        symbol="H⁰",
        category="ScalarBoson",
        mass_GeV=125.25,
        charge_e=0.0,
        spin=0.0,
        color_dof=1,
        generation=0,
        weak_T3=-0.5,
        hypercharge_Y=1.0,
        lifetime_s=1.56e-22,
        width_GeV=4.07e-3,
        is_fermion=False,
    ),
)

# ── COMPOSITE PARTICLES (14 hadrons) ──────────────────────────

COMPOSITE_PARTICLES: tuple[CompositeParticle, ...] = (
    # ── Baryons (qqq) ──
    CompositeParticle(
        name="proton",
        symbol="p",
        hadron_type="Baryon",
        quark_content="uud",
        n_valence_quarks=3,
        mass_GeV=0.93827,
        charge_e=1.0,
        spin=0.5,
        strangeness=0,
        charm=0,
        beauty=0,
        lifetime_s=0,
        width_GeV=0,  # stable (> 10^34 yr)
        constituent_mass_sum_GeV=0.336 + 0.336 + 0.340,
    ),
    CompositeParticle(
        name="neutron",
        symbol="n",
        hadron_type="Baryon",
        quark_content="udd",
        n_valence_quarks=3,
        mass_GeV=0.93957,
        charge_e=0.0,
        spin=0.5,
        strangeness=0,
        charm=0,
        beauty=0,
        lifetime_s=879.4,
        width_GeV=7.49e-28,
        constituent_mass_sum_GeV=0.336 + 0.340 + 0.340,
    ),
    CompositeParticle(
        name="Lambda",
        symbol="Λ⁰",
        hadron_type="Baryon",
        quark_content="uds",
        n_valence_quarks=3,
        mass_GeV=1.11568,
        charge_e=0.0,
        spin=0.5,
        strangeness=1,
        charm=0,
        beauty=0,
        lifetime_s=2.632e-10,
        width_GeV=2.50e-15,
        constituent_mass_sum_GeV=0.336 + 0.340 + 0.486,
    ),
    CompositeParticle(
        name="Sigma+",
        symbol="Σ⁺",
        hadron_type="Baryon",
        quark_content="uus",
        n_valence_quarks=3,
        mass_GeV=1.18937,
        charge_e=1.0,
        spin=0.5,
        strangeness=1,
        charm=0,
        beauty=0,
        lifetime_s=8.018e-11,
        width_GeV=8.21e-15,
        constituent_mass_sum_GeV=0.336 + 0.336 + 0.486,
    ),
    CompositeParticle(
        name="Xi0",
        symbol="Ξ⁰",
        hadron_type="Baryon",
        quark_content="uss",
        n_valence_quarks=3,
        mass_GeV=1.31486,
        charge_e=0.0,
        spin=0.5,
        strangeness=2,
        charm=0,
        beauty=0,
        lifetime_s=2.90e-10,
        width_GeV=2.27e-15,
        constituent_mass_sum_GeV=0.336 + 0.486 + 0.486,
    ),
    CompositeParticle(
        name="Omega-",
        symbol="Ω⁻",
        hadron_type="Baryon",
        quark_content="sss",
        n_valence_quarks=3,
        mass_GeV=1.67245,
        charge_e=-1.0,
        spin=1.5,
        strangeness=3,
        charm=0,
        beauty=0,
        lifetime_s=8.21e-11,
        width_GeV=8.02e-15,
        constituent_mass_sum_GeV=0.486 + 0.486 + 0.486,
    ),
    CompositeParticle(
        name="Lambda_c+",
        symbol="Λ_c⁺",
        hadron_type="Baryon",
        quark_content="udc",
        n_valence_quarks=3,
        mass_GeV=2.28646,
        charge_e=1.0,
        spin=0.5,
        strangeness=0,
        charm=1,
        beauty=0,
        lifetime_s=2.024e-13,
        width_GeV=3.25e-12,
        constituent_mass_sum_GeV=0.336 + 0.340 + 1.55,
    ),
    # ── Mesons (qq̄) ──
    CompositeParticle(
        name="pion+",
        symbol="π⁺",
        hadron_type="Meson",
        quark_content="ud̄",
        n_valence_quarks=2,
        mass_GeV=0.13957,
        charge_e=1.0,
        spin=0.0,
        strangeness=0,
        charm=0,
        beauty=0,
        lifetime_s=2.603e-8,
        width_GeV=2.53e-17,
        constituent_mass_sum_GeV=0.336 + 0.340,
    ),
    CompositeParticle(
        name="pion0",
        symbol="π⁰",
        hadron_type="Meson",
        quark_content="uū/dd̄",
        n_valence_quarks=2,
        mass_GeV=0.13498,
        charge_e=0.0,
        spin=0.0,
        strangeness=0,
        charm=0,
        beauty=0,
        lifetime_s=8.43e-17,
        width_GeV=7.81e-9,
        constituent_mass_sum_GeV=0.336 + 0.336,  # average
    ),
    CompositeParticle(
        name="kaon+",
        symbol="K⁺",
        hadron_type="Meson",
        quark_content="us̄",
        n_valence_quarks=2,
        mass_GeV=0.49368,
        charge_e=1.0,
        spin=0.0,
        strangeness=1,
        charm=0,
        beauty=0,
        lifetime_s=1.238e-8,
        width_GeV=5.32e-17,
        constituent_mass_sum_GeV=0.336 + 0.486,
    ),
    CompositeParticle(
        name="kaon0",
        symbol="K⁰",
        hadron_type="Meson",
        quark_content="ds̄",
        n_valence_quarks=2,
        mass_GeV=0.49761,
        charge_e=0.0,
        spin=0.0,
        strangeness=1,
        charm=0,
        beauty=0,
        lifetime_s=5.116e-8,
        width_GeV=1.29e-17,  # K_L dominant
        constituent_mass_sum_GeV=0.340 + 0.486,
    ),
    CompositeParticle(
        name="J/psi",
        symbol="J/ψ",
        hadron_type="Meson",
        quark_content="cc̄",
        n_valence_quarks=2,
        mass_GeV=3.09690,
        charge_e=0.0,
        spin=1.0,
        strangeness=0,
        charm=0,
        beauty=0,  # hidden charm (C=0)
        lifetime_s=7.09e-21,
        width_GeV=9.29e-5,
        constituent_mass_sum_GeV=1.55 + 1.55,
    ),
    CompositeParticle(
        name="Upsilon",
        symbol="Υ",
        hadron_type="Meson",
        quark_content="bb̄",
        n_valence_quarks=2,
        mass_GeV=9.4603,
        charge_e=0.0,
        spin=1.0,
        strangeness=0,
        charm=0,
        beauty=0,  # hidden beauty (B'=0)
        lifetime_s=1.22e-20,
        width_GeV=5.40e-5,
        constituent_mass_sum_GeV=4.73 + 4.73,
    ),
    CompositeParticle(
        name="D0",
        symbol="D⁰",
        hadron_type="Meson",
        quark_content="cū",
        n_valence_quarks=2,
        mass_GeV=1.86484,
        charge_e=0.0,
        spin=0.0,
        strangeness=0,
        charm=1,
        beauty=0,
        lifetime_s=4.10e-13,
        width_GeV=1.61e-12,
        constituent_mass_sum_GeV=1.55 + 0.336,
    ),
)

# Lookup dicts
_FUND_BY_NAME = {p.name: p for p in FUNDAMENTAL_PARTICLES}
_COMP_BY_NAME = {p.name: p for p in COMPOSITE_PARTICLES}


def get_fundamental(name: str) -> FundamentalParticle | None:
    """Look up a fundamental particle by name (case-insensitive)."""
    key = name.lower()
    for pname, p in _FUND_BY_NAME.items():
        if pname.lower() == key or p.symbol.lower() == key:
            return p
    return None


def get_composite(name: str) -> CompositeParticle | None:
    """Look up a composite particle by name (case-insensitive)."""
    key = name.lower()
    for pname, p in _COMP_BY_NAME.items():
        if pname.lower() == key or p.symbol.lower() == key:
            return p
    return None


# ═══════════════════════════════════════════════════════════════════
# SECTION 2: NORMALIZATION → TRACE VECTOR
# ═══════════════════════════════════════════════════════════════════

EPSILON = 1e-6

# Precompute normalization constants
_LOG_MASS_RANGE = math.log10(M_CEIL_GEV / M_FLOOR_GEV)  # ~13.3
_LOG_TAU_RANGE = math.log10(TAU_UNIVERSE_S / TAU_PLANCK_S)  # ~61

# Maximum |Y| across fundamental particles
_Y_MAX = max(abs(p.hypercharge_Y) for p in FUNDAMENTAL_PARTICLES)


def _mass_log_norm(m_gev: float) -> float:
    """Normalize mass via log-scale to [0, 1]."""
    if m_gev <= 0:
        return 0.0  # massless → clamped to ε later
    return min(1.0, max(0.0, math.log10(m_gev / M_FLOOR_GEV) / _LOG_MASS_RANGE))


def _stability_norm(lifetime_s: float) -> float:
    """Normalize lifetime to [0, 1].  Longer-lived → higher value."""
    if lifetime_s <= 0:
        return 1.0  # stable
    log_ratio = math.log10(lifetime_s / TAU_PLANCK_S) / _LOG_TAU_RANGE
    return min(1.0, max(0.0, log_ratio))


# ── Fundamental particle normalization (8 channels) ────────────

FUND_CHANNELS = [
    "mass_log",  # log-scale mass
    "charge_abs",  # |Q/e|
    "spin_norm",  # 2s/2
    "color_dof",  # log₂(N_c+1)/log₂(9)
    "weak_T3",  # (|T₃|+0.5)/1.5
    "hypercharge",  # |Y|/Y_max
    "generation",  # gen/3
    "stability",  # log(τ/τ_Planck) normalized
]


def normalize_fundamental(p: FundamentalParticle) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Normalize a fundamental particle's quantum numbers into a trace vector.

    Returns (c, w, channel_labels) where c ∈ [ε, 1−ε]^8.
    """
    raw = [
        _mass_log_norm(p.mass_GeV),  # mass_log
        abs(p.charge_e),  # charge_abs
        p.spin,  # spin_norm (0, 0.5, 1.0)
        math.log2(p.color_dof + 1) / math.log2(9),  # color_dof
        (abs(p.weak_T3) + 0.5) / 1.5,  # weak_T3
        abs(p.hypercharge_Y) / _Y_MAX if _Y_MAX > 0 else 0.0,  # hypercharge
        p.generation / 3.0 if p.generation > 0 else 0.0,  # generation
        _stability_norm(p.lifetime_s),  # stability
    ]

    c = np.clip(np.array(raw, dtype=np.float64), EPSILON, 1 - EPSILON)
    w = np.ones(len(c)) / len(c)
    return c, w, list(FUND_CHANNELS)


# ── Composite particle normalization (8 channels) ──────────────

COMP_CHANNELS = [
    "mass_log",  # log-scale mass
    "charge_abs",  # |Q/e|
    "spin_norm",  # 2s/2
    "valence",  # n_quarks/3
    "strangeness",  # |S|/3
    "heavy_flavor",  # (|C|+|B'|)/2
    "stability",  # log(τ/τ_Planck) normalized
    "binding",  # binding energy fraction
]


def normalize_composite(p: CompositeParticle) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Normalize a composite particle's properties into a trace vector.

    Returns (c, w, channel_labels) where c ∈ [ε, 1−ε]^8.
    """
    # Binding fraction: how much mass is "missing" due to binding
    if p.constituent_mass_sum_GeV > 0:
        binding = (p.constituent_mass_sum_GeV - p.mass_GeV) / p.constituent_mass_sum_GeV
        # Can be negative for heavy hadrons (mass > sum of current quark masses)
        # but constituent masses are used here, so it should be positive
        binding = max(0.0, binding)
    else:
        binding = 0.0

    raw = [
        _mass_log_norm(p.mass_GeV),  # mass_log
        abs(p.charge_e),  # charge_abs
        p.spin,  # spin_norm
        p.n_valence_quarks / 3.0,  # valence (2/3 or 1)
        p.strangeness / 3.0,  # strangeness |S|/3
        (p.charm + p.beauty) / 2.0,  # heavy_flavor
        _stability_norm(p.lifetime_s),  # stability
        binding,  # binding fraction
    ]

    c = np.clip(np.array(raw, dtype=np.float64), EPSILON, 1 - EPSILON)
    w = np.ones(len(c)) / len(c)
    return c, w, list(COMP_CHANNELS)


# ═══════════════════════════════════════════════════════════════════
# SECTION 3: KERNEL COMPUTATION + RESULTS
# ═══════════════════════════════════════════════════════════════════


@dataclass
class ParticleKernelResult:
    """Full Tier-1 kernel result for one particle."""

    # Identity
    name: str
    symbol: str
    particle_type: str  # "Fundamental" or "Composite"
    category: str  # Quark/Lepton/GaugeBoson/ScalarBoson or Baryon/Meson

    # Kernel input
    n_channels: int
    channel_labels: list[str]
    trace_vector: list[float]

    # Tier-1 invariants
    F: float
    omega: float
    S: float
    C: float
    kappa: float
    IC: float
    heterogeneity_gap: float

    # Identity checks
    F_plus_omega: float
    IC_leq_F: bool
    IC_eq_exp_kappa: bool

    # Regime + GCD classification
    regime: str
    gcd_category: str

    # Extra metadata
    mass_GeV: float
    charge_e: float
    spin: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _classify_regime(omega: float, F: float, S: float, C: float) -> str:
    """Standard Tier-0 regime classification."""
    if omega >= 0.30:
        return "Collapse"
    if omega < 0.038 and F > 0.90 and S < 0.15 and C < 0.14:
        return "Stable"
    return "Watch"


def _derive_gcd_category(F: float, IC: float, heterogeneity_gap: float, S: float, C: float) -> str:
    """Derive GCD-based classification from kernel invariants.

    These categories emerge from the shape of the property profile:
        Kernel-concentrated: Very high F, low gap  (homogeneous, high values)
        Kernel-structured:   High F, moderate gap  (high but varied)
        Kernel-balanced:     Mid F, low gap         (moderate, uniform)
        Kernel-split:        Mid F, high gap         (moderate, heterogeneous)
        Kernel-sparse:       Low F, low entropy      (low values, few channels active)
        Kernel-diffuse:      Low F, high entropy     (low values spread wide)
    """
    if F > 0.55:
        if heterogeneity_gap < 0.03:
            return "Kernel-concentrated"
        return "Kernel-structured"
    if F > 0.35:
        if heterogeneity_gap < 0.05:
            return "Kernel-balanced"
        return "Kernel-split"
    if S < 0.40:
        return "Kernel-sparse"
    return "Kernel-diffuse"


def compute_fundamental_kernel(p: FundamentalParticle) -> ParticleKernelResult:
    """Run the GCD kernel on a fundamental particle."""
    c, w, labels = normalize_fundamental(p)
    k = compute_kernel_outputs(c, w, EPSILON)

    F_po = k["F"] + k["omega"]
    ic_leq = k["IC"] <= k["F"] + 1e-12
    ic_exp = abs(k["IC"] - math.exp(k["kappa"])) < 1e-12

    regime = _classify_regime(k["omega"], k["F"], k["S"], k["C"])
    gcd_cat = _derive_gcd_category(k["F"], k["IC"], k["heterogeneity_gap"], k["S"], k["C"])

    return ParticleKernelResult(
        name=p.name,
        symbol=p.symbol,
        particle_type="Fundamental",
        category=p.category,
        n_channels=len(c),
        channel_labels=labels,
        trace_vector=c.tolist(),
        F=k["F"],
        omega=k["omega"],
        S=k["S"],
        C=k["C"],
        kappa=k["kappa"],
        IC=k["IC"],
        heterogeneity_gap=k["heterogeneity_gap"],
        F_plus_omega=F_po,
        IC_leq_F=ic_leq,
        IC_eq_exp_kappa=ic_exp,
        regime=regime,
        gcd_category=gcd_cat,
        mass_GeV=p.mass_GeV,
        charge_e=p.charge_e,
        spin=p.spin,
    )


def compute_composite_kernel(p: CompositeParticle) -> ParticleKernelResult:
    """Run the GCD kernel on a composite particle."""
    c, w, labels = normalize_composite(p)
    k = compute_kernel_outputs(c, w, EPSILON)

    F_po = k["F"] + k["omega"]
    ic_leq = k["IC"] <= k["F"] + 1e-12
    ic_exp = abs(k["IC"] - math.exp(k["kappa"])) < 1e-12

    regime = _classify_regime(k["omega"], k["F"], k["S"], k["C"])
    gcd_cat = _derive_gcd_category(k["F"], k["IC"], k["heterogeneity_gap"], k["S"], k["C"])

    return ParticleKernelResult(
        name=p.name,
        symbol=p.symbol,
        particle_type="Composite",
        category=p.hadron_type,
        n_channels=len(c),
        channel_labels=labels,
        trace_vector=c.tolist(),
        F=k["F"],
        omega=k["omega"],
        S=k["S"],
        C=k["C"],
        kappa=k["kappa"],
        IC=k["IC"],
        heterogeneity_gap=k["heterogeneity_gap"],
        F_plus_omega=F_po,
        IC_leq_F=ic_leq,
        IC_eq_exp_kappa=ic_exp,
        regime=regime,
        gcd_category=gcd_cat,
        mass_GeV=p.mass_GeV,
        charge_e=p.charge_e,
        spin=p.spin,
    )


def compute_all_fundamental() -> list[ParticleKernelResult]:
    """Run kernel on all 17 fundamental particles."""
    return [compute_fundamental_kernel(p) for p in FUNDAMENTAL_PARTICLES]


def compute_all_composite() -> list[ParticleKernelResult]:
    """Run kernel on all composite hadrons."""
    return [compute_composite_kernel(p) for p in COMPOSITE_PARTICLES]


def compute_all() -> list[ParticleKernelResult]:
    """Run kernel on ALL particles (fundamental + composite)."""
    return compute_all_fundamental() + compute_all_composite()


# ═══════════════════════════════════════════════════════════════════
# SECTION 4: ANALYSIS + DISPLAY
# ═══════════════════════════════════════════════════════════════════


def display_particle_table(results: list[ParticleKernelResult], title: str = "") -> None:
    """Print a compact table of kernel results."""
    if title:
        print(f"\n{'═' * 78}")
        print(f"  {title}")
        print(f"{'═' * 78}")

    hdr = (
        f"  {'Name':<20s} {'Sym':>5s}  {'F':>6s} {'ω':>6s} "
        f"{'IC':>6s} {'Δ':>6s} {'S':>5s} {'C':>5s}  {'Regime':<8s} {'GCD Category':<20s}"
    )
    print(hdr)
    print("  " + "─" * 96)

    for r in results:
        print(
            f"  {r.name:<20s} {r.symbol:>5s}  {r.F:6.4f} {r.omega:6.4f} "
            f"{r.IC:6.4f} {r.heterogeneity_gap:6.4f} {r.S:5.3f} {r.C:5.3f}  "
            f"{r.regime:<8s} {r.gcd_category:<20s}"
        )


def verify_tier1_identities(results: list[ParticleKernelResult]) -> tuple[int, int]:
    """Verify all three Tier-1 identities across results."""
    passed = 0
    failed = 0
    for r in results:
        ok = abs(r.F_plus_omega - 1.0) < 1e-10 and r.IC_leq_F and r.IC_eq_exp_kappa
        if ok:
            passed += 1
        else:
            failed += 1
            print(f"  FAIL: {r.name} (F+ω={r.F_plus_omega}, IC≤F={r.IC_leq_F}, IC=exp(κ)={r.IC_eq_exp_kappa})")
    return passed, failed


def analyze_by_category(results: list[ParticleKernelResult]) -> None:
    """Group particles by SM category and show kernel statistics."""
    from collections import defaultdict

    groups: dict[str, list[ParticleKernelResult]] = defaultdict(list)
    for r in results:
        groups[r.category].append(r)

    print(f"\n{'═' * 78}")
    print("  KERNEL STATISTICS BY CATEGORY")
    print(f"{'═' * 78}")
    print(
        f"  {'Category':<14s} {'N':>3s}  {'⟨F⟩':>6s} {'⟨ω⟩':>6s} {'⟨IC⟩':>6s} {'⟨Δ⟩':>6s} {'⟨S⟩':>5s}  {'F range':>14s}"
    )
    print("  " + "─" * 72)

    for cat in ["Quark", "Lepton", "GaugeBoson", "ScalarBoson", "Baryon", "Meson"]:
        if cat not in groups:
            continue
        gs = groups[cat]
        Fs = [r.F for r in gs]
        omegas = [r.omega for r in gs]
        ICs = [r.IC for r in gs]
        gaps = [r.heterogeneity_gap for r in gs]
        Ss = [r.S for r in gs]

        avg_F = sum(Fs) / len(Fs)
        avg_o = sum(omegas) / len(omegas)
        avg_IC = sum(ICs) / len(ICs)
        avg_gap = sum(gaps) / len(gaps)
        avg_S = sum(Ss) / len(Ss)
        F_range = f"[{min(Fs):.3f}, {max(Fs):.3f}]"

        print(
            f"  {cat:<14s} {len(gs):3d}  {avg_F:6.4f} {avg_o:6.4f} "
            f"{avg_IC:6.4f} {avg_gap:6.4f} {avg_S:5.3f}  {F_range:>14s}"
        )


def analyze_generation_structure(fund_results: list[ParticleKernelResult]) -> None:
    """Show how the three fermion generations appear in the kernel."""
    print(f"\n{'═' * 78}")
    print("  GENERATION STRUCTURE (Fermions only)")
    print(f"{'═' * 78}")

    fermions = [r for r in fund_results if r.category in ("Quark", "Lepton")]

    for gen in [1, 2, 3]:
        gen_particles = [r for r in fermions if _get_generation(r.name) == gen]
        if not gen_particles:
            continue

        Fs = [r.F for r in gen_particles]
        gaps = [r.heterogeneity_gap for r in gen_particles]
        avg_F = sum(Fs) / len(Fs)
        avg_gap = sum(gaps) / len(gaps)

        names = ", ".join(r.symbol for r in gen_particles)
        print(f"\n  Generation {gen}: {names}")
        print(f"    ⟨F⟩ = {avg_F:.4f}   ⟨Δ⟩ = {avg_gap:.4f}")

        for r in gen_particles:
            print(f"      {r.symbol:>5s}  F={r.F:.4f}  IC={r.IC:.4f}  Δ={r.heterogeneity_gap:.4f}  {r.gcd_category}")


def _get_generation(name: str) -> int:
    """Get generation from particle name."""
    gen_map = {
        "up": 1,
        "down": 1,
        "electron": 1,
        "electron neutrino": 1,
        "charm": 2,
        "strange": 2,
        "muon": 2,
        "muon neutrino": 2,
        "top": 3,
        "bottom": 3,
        "tau": 3,
        "tau neutrino": 3,
    }
    return gen_map.get(name, 0)


def analyze_mass_hierarchy(results: list[ParticleKernelResult]) -> None:
    """Show how the mass hierarchy maps to the kernel."""
    print(f"\n{'═' * 78}")
    print("  MASS HIERARCHY → KERNEL MAPPING")
    print(f"{'═' * 78}")

    # Sort by mass
    massive = [r for r in results if r.mass_GeV > 0]
    massive.sort(key=lambda r: r.mass_GeV)

    print(f"\n  {'Name':<20s} {'Mass (GeV)':>12s}  {'F':>6s} {'IC':>6s} {'Δ':>6s}  {'GCD Category'}")
    print("  " + "─" * 80)

    for r in massive:
        print(
            f"  {r.name:<20s} {r.mass_GeV:12.5g}  {r.F:6.4f} {r.IC:6.4f} {r.heterogeneity_gap:6.4f}  {r.gcd_category}"
        )


def analyze_fermion_boson_split(fund_results: list[ParticleKernelResult]) -> None:
    """Compare fermion vs boson kernel signatures."""
    print(f"\n{'═' * 78}")
    print("  FERMION vs BOSON KERNEL SIGNATURES")
    print(f"{'═' * 78}")

    fermions = [r for r in fund_results if r.category in ("Quark", "Lepton")]
    bosons = [r for r in fund_results if r.category in ("GaugeBoson", "ScalarBoson")]

    for label, group in [("Fermions (12)", fermions), ("Bosons (5)", bosons)]:
        Fs = [r.F for r in group]
        ICs = [r.IC for r in group]
        gaps = [r.heterogeneity_gap for r in group]
        Ss = [r.S for r in group]

        print(f"\n  {label}:")
        print(f"    ⟨F⟩  = {sum(Fs) / len(Fs):.4f}   [{min(Fs):.4f} – {max(Fs):.4f}]")
        print(f"    ⟨IC⟩ = {sum(ICs) / len(ICs):.4f}   [{min(ICs):.4f} – {max(ICs):.4f}]")
        print(f"    ⟨Δ⟩  = {sum(gaps) / len(gaps):.4f}")
        print(f"    ⟨S⟩  = {sum(Ss) / len(Ss):.4f}")


def analyze_quark_lepton_duality(fund_results: list[ParticleKernelResult]) -> None:
    """Check if quark-lepton pairs within each generation show kernel duality."""
    print(f"\n{'═' * 78}")
    print("  QUARK ↔ LEPTON DUALITY (within generations)")
    print(f"{'═' * 78}")

    by_name = {r.name: r for r in fund_results}

    pairs = [
        # (quark_up, quark_down, lepton_charged, lepton_neutrino)
        ("up", "down", "electron", "electron neutrino"),
        ("charm", "strange", "muon", "muon neutrino"),
        ("top", "bottom", "tau", "tau neutrino"),
    ]

    for gen, (qu, qd, lc, ln) in enumerate(pairs, 1):
        rqu = by_name.get(qu)
        rqd = by_name.get(qd)
        rlc = by_name.get(lc)
        rln = by_name.get(ln)

        if rqu is None or rqd is None or rlc is None or rln is None:
            continue

        quark_avg_F = (rqu.F + rqd.F) / 2
        lepton_avg_F = (rlc.F + rln.F) / 2
        F_sum = quark_avg_F + lepton_avg_F

        quark_avg_gap = (rqu.heterogeneity_gap + rqd.heterogeneity_gap) / 2
        lepton_avg_gap = (rlc.heterogeneity_gap + rln.heterogeneity_gap) / 2

        print(f"\n  Generation {gen}:")
        print(f"    Quarks  ({rqu.symbol}, {rqd.symbol}):  ⟨F⟩ = {quark_avg_F:.4f}  ⟨Δ⟩ = {quark_avg_gap:.4f}")
        print(f"    Leptons ({rlc.symbol}, {rln.symbol}):  ⟨F⟩ = {lepton_avg_F:.4f}  ⟨Δ⟩ = {lepton_avg_gap:.4f}")
        print(f"    Sum ⟨F_q⟩ + ⟨F_l⟩ = {F_sum:.4f}")


def analyze_composite_binding(comp_results: list[ParticleKernelResult]) -> None:
    """Show how binding energy fraction appears in the kernel."""
    print(f"\n{'═' * 78}")
    print("  COMPOSITE BINDING → KERNEL MAPPING")
    print(f"{'═' * 78}")

    print(
        f"\n  {'Name':<15s} {'Quarks':>6s} {'m (GeV)':>9s} {'Σm_q':>9s} {'Bind%':>6s}  {'F':>6s} {'IC':>6s} {'Δ':>6s}"
    )
    print("  " + "─" * 75)

    for p, r in zip(COMPOSITE_PARTICLES, comp_results, strict=True):
        bind_pct = 0.0
        if p.constituent_mass_sum_GeV > 0:
            bind_pct = (p.constituent_mass_sum_GeV - p.mass_GeV) / p.constituent_mass_sum_GeV * 100

        print(
            f"  {r.name:<15s} {p.quark_content:>6s} {p.mass_GeV:9.5f} "
            f"{p.constituent_mass_sum_GeV:9.5f} {bind_pct:5.1f}%  "
            f"{r.F:6.4f} {r.IC:6.4f} {r.heterogeneity_gap:6.4f}"
        )


def analyze_channel_profiles(fund_results: list[ParticleKernelResult]) -> None:
    """Show the raw trace vectors — which channels drive F and IC."""
    print(f"\n{'═' * 78}")
    print("  TRACE VECTOR PROFILES (Fundamental)")
    print(f"{'═' * 78}")
    print(f"\n  Channels: {', '.join(FUND_CHANNELS)}")
    print()

    for r in fund_results:
        c = np.array(r.trace_vector)
        max_ch = FUND_CHANNELS[np.argmax(c)]
        min_ch = FUND_CHANNELS[np.argmin(c)]
        print(f"  {r.symbol:>5s}  c = [{', '.join(f'{v:.3f}' for v in c)}]  max={max_ch}  min={min_ch}")


# ═══════════════════════════════════════════════════════════════════
# SECTION 5: MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║  SUBATOMIC PARTICLE KERNEL ANALYSIS — GCD Tier-1                           ║")
    print("║  17 Fundamental Particles + 14 Composite Hadrons                           ║")
    print("╚══════════════════════════════════════════════════════════════════════════════╝")

    # ─── Compute all kernels ───
    fund_results = compute_all_fundamental()
    comp_results = compute_all_composite()
    all_results = fund_results + comp_results

    # ─── Tier-1 Identity Verification ───
    print(f"\n{'━' * 78}")
    print("  TIER-1 IDENTITY VERIFICATION")
    print(f"{'━' * 78}")
    fp, ff = verify_tier1_identities(fund_results)
    cp, cf = verify_tier1_identities(comp_results)
    print(f"\n  Fundamental: {fp}/{fp + ff} pass")
    print(f"  Composite:   {cp}/{cp + cf} pass")
    print(f"  Total:       {fp + cp}/{fp + ff + cp + cf} pass")

    # ─── Full Tables ───
    display_particle_table(fund_results, "FUNDAMENTAL PARTICLES — KERNEL TABLE")
    display_particle_table(comp_results, "COMPOSITE HADRONS — KERNEL TABLE")

    # ─── Analysis ───
    analyze_by_category(all_results)
    analyze_generation_structure(fund_results)
    analyze_fermion_boson_split(fund_results)
    analyze_quark_lepton_duality(fund_results)
    analyze_mass_hierarchy(all_results)
    analyze_composite_binding(comp_results)
    analyze_channel_profiles(fund_results)

    # ─── Summary ───
    print(f"\n{'━' * 78}")
    print("  SUMMARY")
    print(f"{'━' * 78}")

    # GCD category distribution
    from collections import Counter

    cat_counts = Counter(r.gcd_category for r in all_results)
    regime_counts = Counter(r.regime for r in all_results)

    print("\n  GCD Category Distribution:")
    for cat, n in sorted(cat_counts.items()):
        members = [r.symbol for r in all_results if r.gcd_category == cat]
        print(f"    {cat:<22s}: {n:2d}  [{', '.join(members)}]")

    print("\n  Regime Distribution:")
    for reg, n in sorted(regime_counts.items()):
        members = [r.symbol for r in all_results if r.regime == reg]
        print(f"    {reg:<10s}: {n:2d}  [{', '.join(members)}]")

    # Key findings
    fund_Fs = [r.F for r in fund_results]
    comp_Fs = [r.F for r in comp_results]
    all_gaps = [r.heterogeneity_gap for r in all_results]

    max_F_all = max(all_results, key=lambda r: r.F)
    min_F_all = min(all_results, key=lambda r: r.F)
    max_gap = max(all_results, key=lambda r: r.heterogeneity_gap)

    print("\n  Key Findings:")
    print(f"    Highest F:     {max_F_all.symbol} ({max_F_all.name}) F={max_F_all.F:.4f}")
    print(f"    Lowest F:      {min_F_all.symbol} ({min_F_all.name}) F={min_F_all.F:.4f}")
    print(f"    Largest gap:   {max_gap.symbol} ({max_gap.name}) Δ={max_gap.heterogeneity_gap:.4f}")
    print(f"    ⟨F⟩ fundamental: {sum(fund_Fs) / len(fund_Fs):.4f}")
    print(f"    ⟨F⟩ composite:   {sum(comp_Fs) / len(comp_Fs):.4f}")
    print(f"    ⟨Δ⟩ all:         {sum(all_gaps) / len(all_gaps):.4f}")

    # Duality check
    print("\n  ─── DUALITY CHECK ───")
    print("    For each particle: F + F_complement = ?")
    for r in fund_results:
        c_comp = 1.0 - np.array(r.trace_vector)
        c_comp = np.clip(c_comp, EPSILON, 1 - EPSILON)
        w = np.ones(len(c_comp)) / len(c_comp)
        k_comp = compute_kernel_outputs(c_comp, w, EPSILON)
        F_sum = r.F + k_comp["F"]
        print(f"    {r.symbol:>5s}: F={r.F:.6f}  F_comp={k_comp['F']:.6f}  sum={F_sum:.9f}")

    print()

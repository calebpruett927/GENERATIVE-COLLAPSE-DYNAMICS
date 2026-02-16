"""Scale Ladder — From Minimal to Universal.

A complete, auditable map of material structure from the smallest
measurable scale (10⁻³⁵ m, Planck length) to the largest (10²⁶ m,
cosmic horizon) — 61 orders of magnitude — unified through the
GCD kernel.

═══════════════════════════════════════════════════════════════════
  THE SCALE LADDER
═══════════════════════════════════════════════════════════════════

  Rung 0   Planck           10⁻³⁵ m    The floor of measurability
  Rung 1   Subatomic        10⁻¹⁸ m    Quarks, leptons, bosons
  Rung 2   Hadronic         10⁻¹⁵ m    Protons, neutrons, mesons
  Rung 3   Nuclear          10⁻¹⁵ m    Nuclei, binding, shell closure
  Rung 4   Atomic           10⁻¹⁰ m    118 elements, electron shells
  Rung 5   Molecular        10⁻⁹  m    Chemical bonds, polymers
  Rung 6   Cellular         10⁻⁵  m    Cells, organelles, biology
  Rung 7   Human/Everyday   10⁰   m    What we see, touch, hear
  Rung 8   Geological       10⁶   m    Planets, tectonics
  Rung 9   Stellar          10¹⁰  m    Stars, stellar evolution
  Rung 10  Galactic         10²¹  m    Galaxies, clusters
  Rung 11  Cosmological     10²⁶  m    Observable universe

Each rung has:
  - Representative objects with measurable properties
  - A trace vector mapping to the GCD kernel
  - Kernel outputs: F, ω, IC, κ, S, C, Δ = F − IC
  - Regime classification: Stable / Watch / Collapse
  - Explicit BRIDGES to adjacent rungs (what connects them)

The Three Pillars hold at every rung with zero violations:
  PILLAR 1:  F + ω = 1           (duality identity)
  PILLAR 2:  IC ≤ F              (integrity bound)
  PILLAR 3:  IC = exp(κ)         (exponential bridge)

Purpose:
  This file exists so that anyone — physicist, biologist, engineer,
  philosopher, student — can trace the thread of minimal structure
  from the quantum foam to the cosmic web and see that it is ONE
  thread. Not by analogy. By measurement.

  *Scala naturae non narratur: mensuratur.*
  (The ladder of nature is not narrated: it is measured.)

Cross-references:
    Axiom:              AXIOM.md (Axiom-0)
    Tier system:        TIER_SYSTEM.md
    Unified structure:  closures/unified_minimal_structure.py (T17-T23)
    Subatomic:          closures/standard_model/subatomic_kernel.py
    Nuclear bridge:     closures/atomic_physics/cross_scale_kernel.py
    Elements:           closures/materials_science/element_database.py
    Everyday physics:   closures/everyday_physics/
    Stellar:            closures/astronomy/stellar_luminosity.py
    Cosmology:          closures/astronomy/cosmology.py
    Kernel:             src/umcp/kernel_optimized.py
    Contract:           contracts/UMA.INTSTACK.v1.yaml
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# ── Path setup ──────────────────────────────────────────────────
_WORKSPACE = Path(__file__).resolve().parents[1]
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))

from umcp.kernel_optimized import compute_kernel_outputs  # noqa: E402

# ── Frozen constants (consistent across the seam) ───────────────
EPSILON = 1e-8  # guard band
OMEGA_WATCH = 0.10  # Watch threshold
OMEGA_COLLAPSE = 0.30  # Collapse threshold


# ═══════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ScaleObject:
    """A single object at a specific scale, with kernel outputs."""

    name: str
    F: float
    omega: float
    IC: float
    kappa: float
    S: float
    C: float
    gap: float  # Δ = F − IC (heterogeneity gap)
    regime: str
    trace: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class Bridge:
    """An explicit connection between adjacent rungs.

    Bridges are not metaphors — they are measurable relationships
    where a property at one scale causally or structurally determines
    a property at the adjacent scale.
    """

    from_rung: str
    to_rung: str
    mechanism: str  # what connects them
    example: str  # concrete instance
    channel_mapping: str  # which channels flow across


@dataclass
class Rung:
    """One rung of the scale ladder."""

    number: int
    name: str
    scale_meters: float
    description: str
    channel_names: list[str]
    n_channels: int
    objects: list[ScaleObject] = field(default_factory=list)
    bridges_up: list[Bridge] = field(default_factory=list)
    bridges_down: list[Bridge] = field(default_factory=list)

    @property
    def n_objects(self) -> int:
        return len(self.objects)

    @property
    def mean_F(self) -> float:
        return sum(o.F for o in self.objects) / len(self.objects) if self.objects else 0.0

    @property
    def mean_omega(self) -> float:
        return sum(o.omega for o in self.objects) / len(self.objects) if self.objects else 0.0

    @property
    def mean_IC(self) -> float:
        return sum(o.IC for o in self.objects) / len(self.objects) if self.objects else 0.0

    @property
    def mean_gap(self) -> float:
        return sum(o.gap for o in self.objects) / len(self.objects) if self.objects else 0.0

    @property
    def mean_S(self) -> float:
        return sum(o.S for o in self.objects) / len(self.objects) if self.objects else 0.0

    @property
    def regime_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for o in self.objects:
            counts[o.regime] = counts.get(o.regime, 0) + 1
        return counts


@dataclass
class ScaleLadder:
    """The complete scale ladder from Planck to cosmos."""

    rungs: list[Rung]
    total_objects: int = 0
    total_tier1_violations: int = 0
    pillar_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    build_time_ms: float = 0.0

    def rung_by_name(self, name: str) -> Rung | None:
        for r in self.rungs:
            if r.name.lower() == name.lower():
                return r
        return None


# ═══════════════════════════════════════════════════════════════════
# KERNEL HELPERS
# ═══════════════════════════════════════════════════════════════════


def _clip(x: float) -> float:
    """Clip to [ε, 1−ε]."""
    return max(EPSILON, min(1.0 - EPSILON, x))


def _compute(c: np.ndarray, w: np.ndarray | None = None) -> dict[str, float]:
    """Compute kernel outputs from trace vector."""
    c_clipped = np.clip(c, EPSILON, 1 - EPSILON)
    if w is None:
        w = np.ones(len(c_clipped)) / len(c_clipped)
    return compute_kernel_outputs(c_clipped, w, EPSILON)


def _to_object(name: str, k: dict[str, float], c: np.ndarray) -> ScaleObject:
    """Convert kernel output dict to ScaleObject."""
    F = k["F"]
    omega = k["omega"]
    IC = k["IC"]
    if omega < OMEGA_WATCH:
        regime = "Stable"
    elif omega < OMEGA_COLLAPSE:
        regime = "Watch"
    else:
        regime = "Collapse"
    return ScaleObject(
        name=name,
        F=round(F, 6),
        omega=round(omega, 6),
        IC=round(IC, 6),
        kappa=round(k["kappa"], 6),
        S=round(k["S"], 6),
        C=round(k["C"], 6),
        gap=round(F - IC, 6),
        regime=regime,
        trace=[round(float(x), 6) for x in c],
    )


# ═══════════════════════════════════════════════════════════════════
# RUNG 0 — PLANCK SCALE (10⁻³⁵ m)
# ═══════════════════════════════════════════════════════════════════


def _build_planck() -> Rung:
    """The floor of measurability.

    At the Planck scale, the five fundamental constants (G, ℏ, c, kB, ε₀)
    define natural units. We map the dimensionless ratios between them
    as a 6-channel trace vector. These are not "objects" — they are
    the boundary conditions of measurement itself.

    Bridge upward: Planck mass → heaviest elementary particles;
    Planck length → smallest resolvable structure.
    """
    # Planck units in SI
    G = 6.67430e-11  # N·m²/kg²
    hbar = 1.054571817e-34  # J·s
    c = 2.99792458e8  # m/s
    kB = 1.380649e-23  # J/K
    eps0 = 8.8541878128e-12  # F/m

    l_P = math.sqrt(hbar * G / c**3)  # 1.616e-35 m
    m_P = math.sqrt(hbar * c / G)  # 2.176e-8 kg = 1.22e19 GeV
    t_P = l_P / c  # 5.39e-44 s
    T_P = m_P * c**2 / kB  # 1.417e32 K
    q_P = math.sqrt(4 * math.pi * eps0 * hbar * c)  # 1.876e-18 C
    e = 1.602176634e-19  # C

    # 6-channel trace: dimensionless ratios showing how our universe
    # is positioned relative to the Planck floor
    planck_ratios = [
        (
            "Planck Floor",
            [
                _clip(e / q_P),  # fine structure proximity: α^(1/2) ≈ 0.0854
                _clip(9.109e-31 / m_P),  # electron/Planck mass ≈ 4.2e-23 (near ε)
                _clip(2.725 / T_P),  # CMB/Planck temp ≈ 1.9e-32 (near ε)
                _clip(l_P / 1e-18),  # Planck/subatomic ≈ 1.6e-17 (near ε)
                _clip(t_P / 1e-24),  # Planck/yoctosecond ≈ 5.4e-20 (near ε)
                _clip(0.0854),  # √α itself — the coupling strength of EM
            ],
        ),
    ]

    objects = []
    for name, channels in planck_ratios:
        c_arr = np.array(channels)
        k = _compute(c_arr)
        objects.append(_to_object(name, k, c_arr))

    return Rung(
        number=0,
        name="Planck",
        scale_meters=1.616e-35,
        description=(
            "The floor of measurability. Below this, spacetime itself "
            "may not be smooth. The Planck scale sets the boundary "
            "conditions — not objects, but the limits of objecthood."
        ),
        channel_names=["α_proximity", "m_e/m_P", "T_CMB/T_P", "l_P/l_sub", "t_P/t_yocto", "√α"],
        n_channels=6,
        objects=objects,
        bridges_up=[
            Bridge(
                from_rung="Planck",
                to_rung="Subatomic",
                mechanism="Planck mass → electroweak symmetry breaking → particle masses",
                example="m_P = 1.22×10¹⁹ GeV sets the gravitational coupling; "
                "Higgs VEV = 246 GeV breaks electroweak → particle masses",
                channel_mapping="m_e/m_P → mass_log channel at subatomic rung",
            )
        ],
    )


# ═══════════════════════════════════════════════════════════════════
# RUNG 1 — SUBATOMIC (10⁻¹⁸ m)
# ═══════════════════════════════════════════════════════════════════


def _build_subatomic() -> Rung:
    """Fundamental particles: quarks, leptons, gauge bosons, Higgs.

    Imports directly from the Standard Model subatomic kernel (31 particles,
    8 channels each). This is the bottom rung of the *material* ladder —
    everything above is made from these.
    """
    from closures.standard_model.subatomic_kernel import (
        compute_all_composite,
        compute_all_fundamental,
    )

    objects = []
    for r in compute_all_fundamental():
        objects.append(
            ScaleObject(
                name=r.name,
                F=r.F,
                omega=r.omega,
                IC=r.IC,
                kappa=r.kappa,
                S=r.S,
                C=r.C,
                gap=round(r.F - r.IC, 6),
                regime=r.regime,
            )
        )
    for r in compute_all_composite():
        objects.append(
            ScaleObject(
                name=r.name,
                F=r.F,
                omega=r.omega,
                IC=r.IC,
                kappa=r.kappa,
                S=r.S,
                C=r.C,
                gap=round(r.F - r.IC, 6),
                regime=r.regime,
            )
        )

    return Rung(
        number=1,
        name="Subatomic",
        scale_meters=1e-18,
        description=(
            "31 particles — 17 fundamental (6 quarks, 6 leptons, "
            "4 gauge bosons, Higgs) + 14 composite hadrons. "
            "Everything you will ever touch is made from these."
        ),
        channel_names=[
            "mass_log",
            "charge_abs",
            "spin_norm",
            "color/valence",
            "T₃/strangeness",
            "Y/heavy_flavor",
            "generation/stability",
            "stability/binding",
        ],
        n_channels=8,
        objects=objects,
        bridges_up=[
            Bridge(
                from_rung="Subatomic",
                to_rung="Nuclear",
                mechanism="Strong force confinement binds quarks into nucleons; "
                "residual strong force binds nucleons into nuclei",
                example="3 quarks (uud) → proton; 3 quarks (udd) → neutron; protons + neutrons → nucleus",
                channel_mapping="color → confinement (IC cliff: 98% drop); mass_log → nuclear binding energy",
            )
        ],
        bridges_down=[
            Bridge(
                from_rung="Subatomic",
                to_rung="Planck",
                mechanism="Particle masses arise from Higgs mechanism (VEV=246 GeV), "
                "which sits 17 OOM below the Planck mass — the hierarchy problem",
                example="m_top/m_P ≈ 1.4×10⁻¹⁷; m_e/m_P ≈ 4.2×10⁻²³",
                channel_mapping="mass_log → Planck mass ratio",
            )
        ],
    )


# ═══════════════════════════════════════════════════════════════════
# RUNG 2 — NUCLEAR (10⁻¹⁵ m)
# ═══════════════════════════════════════════════════════════════════

_MAGIC = (2, 8, 20, 28, 50, 82, 126)


def _stable_A(Z: int) -> int:
    """Most stable mass number for element Z."""
    if Z <= 0:
        return 0
    if Z == 1:
        return 1
    if Z <= 20:
        return 2 * Z
    return round(2.0 * Z + 0.015 * Z**2)


def _bethe_weizsacker(Z: int, A: int) -> float:
    """Binding energy per nucleon (MeV) via Bethe-Weizsäcker formula."""
    if A <= 1:
        return 0.0
    N = A - Z
    B = 15.75 * A - 17.80 * A ** (2.0 / 3.0) - 0.711 * Z * (Z - 1) / A ** (1.0 / 3.0) - 23.70 * (A - 2 * Z) ** 2 / A
    if Z % 2 == 0 and N % 2 == 0:
        B += 11.18 / A**0.5
    elif Z % 2 == 1 and N % 2 == 1:
        B -= 11.18 / A**0.5
    return max(0.0, B / A)


def _magic_proximity(Z: int, N: int) -> float:
    """Shell closure proximity ∈ [0, 1]."""
    dZ = min(abs(Z - m) for m in _MAGIC)
    dN = min(abs(N - m) for m in _MAGIC)
    return (1.0 / (1.0 + dZ) + 1.0 / (1.0 + dN)) / 2.0


def _build_nuclear() -> Rung:
    """Atomic nuclei Z=1..92: protons + neutrons bound by residual strong force.

    The nuclear scale is where binding energy peaks (Fe-56) and
    shell structure (magic numbers) determines stability. The
    Bethe-Weizsäcker semi-empirical mass formula provides the
    trace vector channels.
    """
    objects = []
    for Z in range(1, 93):
        A = _stable_A(Z)
        if A <= 0:
            continue
        N = A - Z
        BE_A = _bethe_weizsacker(Z, A)
        magic_prox = _magic_proximity(Z, N)

        c = np.array(
            [
                _clip(Z / 92.0),
                _clip(N / 146.0),
                _clip(BE_A / 8.8),
                magic_prox,
                _clip(A / 238.0),
                abs(N - Z) / max(A, 1),
            ]
        )
        c = np.clip(c, EPSILON, 1 - EPSILON)
        w = np.ones(6) / 6.0
        k = _compute(c, w)
        objects.append(_to_object(f"Z={Z} A={A}", k, c))

    return Rung(
        number=2,
        name="Nuclear",
        scale_meters=1e-15,
        description=(
            "92 stable nuclei. Binding energy per nucleon peaks at "
            "Z≈26 (iron group). Magic numbers (2,8,20,28,50,82,126) "
            "mark shell closures where nuclei are especially stable — "
            "the nuclear analog of noble gases."
        ),
        channel_names=[
            "proton_frac",
            "neutron_frac",
            "BE/A",
            "magic_proximity",
            "mass_number_frac",
            "isospin_asymmetry",
        ],
        n_channels=6,
        objects=objects,
        bridges_up=[
            Bridge(
                from_rung="Nuclear",
                to_rung="Atomic",
                mechanism="Nuclear charge Z determines electron configuration; "
                "nuclear mass determines atomic mass; "
                "nuclear stability determines which elements exist",
                example="Z=26 (iron nucleus) → 26 electrons → [Ar] 3d⁶ 4s² → "
                "transition metal properties (magnetism, catalysis)",
                channel_mapping="Z → atomic number → electron config → electronegativity, ionization energy, radius",
            )
        ],
        bridges_down=[
            Bridge(
                from_rung="Nuclear",
                to_rung="Subatomic",
                mechanism="Nuclei are bound states of protons and neutrons; "
                "each nucleon is 3 confined quarks; "
                "nuclear force is residual QCD",
                example="He-4 (α particle): 2p + 2n = 12 quarks; doubly magic (Z=2, N=2) → exceptionally stable",
                channel_mapping="confinement IC-cliff → nuclear binding; "
                "quark mass → nucleon mass (99% from QCD, not Higgs)",
            )
        ],
    )


# ═══════════════════════════════════════════════════════════════════
# RUNG 3 — ATOMIC (10⁻¹⁰ m)
# ═══════════════════════════════════════════════════════════════════


def _build_atomic() -> Rung:
    """118 elements: the periodic table through the GCD kernel.

    Every element is mapped to 8 channels covering electronic,
    thermodynamic, and structural properties. This is the rung
    where quantum mechanics becomes chemistry — electron shells,
    orbital filling, bonding.
    """
    try:
        from closures.materials_science.element_database import ELEMENTS
    except ImportError:
        return Rung(
            number=3,
            name="Atomic",
            scale_meters=1e-10,
            description="(element_database not available)",
            channel_names=[],
            n_channels=0,
        )

    objects = []
    for elem in ELEMENTS:
        en = elem.electronegativity if elem.electronegativity else 0.5
        ie = elem.ionization_energy_eV if elem.ionization_energy_eV else 0.5
        dens = elem.density_g_cm3 if elem.density_g_cm3 else 0.5
        mp = elem.melting_point_K if elem.melting_point_K else 0.5
        bp = elem.boiling_point_K if elem.boiling_point_K else 0.5
        ar = elem.atomic_radius_pm if elem.atomic_radius_pm else 0.5
        ea = elem.electron_affinity_eV if elem.electron_affinity_eV and elem.electron_affinity_eV > 0 else 0.5

        c = np.array(
            [
                _clip(elem.Z / 118.0),
                _clip(en / 4.0),
                _clip(ie / 25.0),
                _clip(dens / 23000.0),
                _clip(mp / 3700.0),
                _clip(bp / 5900.0),
                _clip(ar / 300.0),
                _clip(ea / 4.0),
            ]
        )
        w = np.ones(8) / 8.0
        k = _compute(c, w)
        objects.append(_to_object(elem.symbol, k, c))

    return Rung(
        number=3,
        name="Atomic",
        scale_meters=1e-10,
        description=(
            "118 elements — the complete periodic table. Electron shell "
            "filling (Aufbau principle) determines chemical behavior. "
            "Noble gases have full shells (high IC); alkali metals have "
            "one lone electron (large heterogeneity gap)."
        ),
        channel_names=[
            "Z_norm",
            "electronegativity",
            "ionization_energy",
            "density",
            "melting_pt",
            "boiling_pt",
            "atomic_radius",
            "electron_affinity",
        ],
        n_channels=8,
        objects=objects,
        bridges_up=[
            Bridge(
                from_rung="Atomic",
                to_rung="Molecular",
                mechanism="Atomic properties (electronegativity, radius, valence) "
                "determine how atoms bond: ionic, covalent, metallic, "
                "van der Waals. Chemistry IS this bridge.",
                example="C (Z=6): 4 valence electrons → tetravalent bonding → "
                "diamond (sp³), graphite (sp²), fullerenes, life itself",
                channel_mapping="electronegativity → bond polarity; "
                "atomic_radius → bond length; "
                "ionization_energy → bond strength",
            )
        ],
        bridges_down=[
            Bridge(
                from_rung="Atomic",
                to_rung="Nuclear",
                mechanism="Nuclear charge Z = number of protons = number of "
                "electrons (neutral atom). The nucleus is the attractor "
                "around which electron probability clouds form.",
                example="Z=79 (gold): 79 protons attract 79 electrons into "
                "[Xe] 4f¹⁴ 5d¹⁰ 6s¹ — relativistic contraction of "
                "6s orbital gives gold its color",
                channel_mapping="Z → electron count → shell structure → all atomic properties",
            )
        ],
    )


# ═══════════════════════════════════════════════════════════════════
# RUNG 4 — MOLECULAR (10⁻⁹ m)
# ═══════════════════════════════════════════════════════════════════


def _build_molecular() -> Rung:
    """Molecules and simple materials: where atoms become substances.

    This rung maps molecules and small structures by their bonding
    characteristics, size, and emergent properties. Representative
    molecules span the diversity of chemical bonding and biological
    relevance.

    Channels:
      molecular_weight_norm — mass (proxy for size/complexity)
      n_atoms_norm          — number of atoms (structural complexity)
      bond_energy_norm      — average bond dissociation energy
      polarity_norm         — dipole moment (charge distribution)
      symmetry_norm         — point group order (structural regularity)
      bioactivity           — biological relevance (0 = inert, 1 = essential)
    """
    # Representative molecules: (name, MW [g/mol], n_atoms, avg_bond_E [kJ/mol],
    #                            dipole [D], symmetry_order, bio_score)
    # Sources: NIST Chemistry WebBook, CRC Handbook
    molecules = [
        # Simple diatomics
        ("H₂", 2.016, 2, 436.0, 0.0, 2, 0.3),
        ("O₂", 32.0, 2, 498.0, 0.0, 2, 0.9),
        ("N₂", 28.0, 2, 945.0, 0.0, 2, 0.8),
        ("CO", 28.01, 2, 1072.0, 0.11, 1, 0.2),
        ("HCl", 36.46, 2, 431.0, 1.08, 1, 0.3),
        # Water and simple compounds
        ("H₂O", 18.015, 3, 459.0, 1.85, 2, 1.0),
        ("CO₂", 44.01, 3, 799.0, 0.0, 4, 0.7),
        ("NH₃", 17.03, 4, 386.0, 1.47, 3, 0.8),
        ("CH₄", 16.04, 5, 411.0, 0.0, 12, 0.5),
        ("H₂SO₄", 98.08, 7, 465.0, 2.72, 2, 0.3),
        # Organic building blocks
        ("C₂H₅OH (ethanol)", 46.07, 9, 348.0, 1.69, 1, 0.6),
        ("CH₃COOH (acetic acid)", 60.05, 8, 360.0, 1.74, 1, 0.5),
        ("C₆H₁₂O₆ (glucose)", 180.16, 24, 350.0, 8.0, 1, 1.0),
        ("C₆H₆ (benzene)", 78.11, 12, 505.0, 0.0, 12, 0.2),
        # Biological macromolecule fragments
        ("Amino acid (avg)", 110.0, 15, 340.0, 5.0, 1, 1.0),
        ("Nucleotide (avg)", 330.0, 35, 350.0, 6.0, 1, 1.0),
        # Inorganic
        ("NaCl", 58.44, 2, 411.0, 9.0, 48, 0.7),
        ("SiO₂", 60.08, 3, 800.0, 0.0, 3, 0.2),
        ("CaCO₃", 100.09, 5, 500.0, 0.0, 3, 0.5),
        ("Fe₂O₃", 159.69, 5, 400.0, 0.0, 6, 0.3),
    ]

    # Normalization constants
    mw_max = max(m[1] for m in molecules)
    na_max = max(m[2] for m in molecules)
    be_max = max(m[3] for m in molecules)
    dip_max = max(m[4] for m in molecules) or 1.0
    sym_max = max(m[5] for m in molecules)

    objects = []
    for name, mw, na, be, dip, sym, bio in molecules:
        c = np.array(
            [
                _clip(mw / mw_max),
                _clip(na / na_max),
                _clip(be / be_max),
                _clip(dip / dip_max) if dip_max > 0 else EPSILON,
                _clip(sym / sym_max),
                _clip(bio),
            ]
        )
        k = _compute(c)
        objects.append(_to_object(name, k, c))

    return Rung(
        number=4,
        name="Molecular",
        scale_meters=1e-9,
        description=(
            "20 representative molecules — from H₂ to biomolecule fragments. "
            "This is where atomic properties become chemical behavior: "
            "bond angles, polarity, symmetry, reactivity. Water's dipole "
            "moment (1.85 D) makes life possible. Benzene's symmetry "
            "(order 12) gives aromatic stability."
        ),
        channel_names=[
            "molecular_weight",
            "n_atoms",
            "bond_energy",
            "polarity",
            "symmetry",
            "bioactivity",
        ],
        n_channels=6,
        objects=objects,
        bridges_up=[
            Bridge(
                from_rung="Molecular",
                to_rung="Cellular",
                mechanism="Molecules self-assemble into cellular structures: "
                "lipid bilayers (cell membranes), protein folding, "
                "DNA base pairing. Chemistry becomes biology.",
                example="Phospholipids (amphiphilic molecules) → lipid bilayer → "
                "cell membrane → compartmentalization → life",
                channel_mapping="polarity → membrane assembly; "
                "bioactivity → metabolic integration; "
                "bond_energy → thermal stability of biomolecules",
            )
        ],
        bridges_down=[
            Bridge(
                from_rung="Molecular",
                to_rung="Atomic",
                mechanism="Molecular properties emerge from atomic properties "
                "through quantum mechanical bonding: orbital overlap, "
                "electron sharing, electrostatic attraction",
                example="H₂O: O (EN=3.44) + 2×H (EN=2.20) → polar covalent bonds "
                "→ 104.5° angle → dipole moment → solvent properties",
                channel_mapping="electronegativity_diff → bond polarity; "
                "atomic_radius → bond length; "
                "valence_electrons → coordination",
            )
        ],
    )


# ═══════════════════════════════════════════════════════════════════
# RUNG 5 — CELLULAR (10⁻⁵ m)
# ═══════════════════════════════════════════════════════════════════


def _build_cellular() -> Rung:
    """Cells and biological micro-structures.

    This rung bridges molecular chemistry to the living world.
    Representative systems span from viruses (edge of life) to
    human cells (complex eukaryotes).

    Channels:
      size_norm         — physical size (log-scaled)
      complexity_norm   — number of distinct molecular components
      energy_norm       — metabolic rate (ATP turnover)
      replication_norm  — division time (inverse, faster = higher)
      organization_norm — structural hierarchy (membrane, organelles, etc.)
      information_norm  — genome size / information content
    """
    # (name, size_μm, n_components, ATP_rate_rel, div_time_hr, org_level, genome_bp)
    # org_level: 1=none, 2=membrane, 3=organelles, 4=tissue, 5=organ
    cells = [
        ("Virus (T4 phage)", 0.2, 50, 0.0, 0.5, 1, 169_000),
        ("Mycoplasma", 0.3, 500, 0.01, 4.0, 2, 580_000),
        ("E. coli", 2.0, 4000, 0.1, 0.33, 2, 4_600_000),
        ("Yeast", 5.0, 6000, 0.3, 1.5, 3, 12_000_000),
        ("Red blood cell", 7.0, 300, 0.05, 0, 2, 0),  # no nucleus
        ("White blood cell", 12.0, 8000, 0.5, 24.0, 3, 3_200_000_000),
        ("Epithelial cell", 15.0, 10000, 0.4, 24.0, 4, 3_200_000_000),
        ("Neuron", 50.0, 12000, 0.8, 0, 3, 3_200_000_000),
        ("Muscle fiber", 100.0, 8000, 1.0, 0, 4, 3_200_000_000),
        ("Plant cell (avg)", 40.0, 7000, 0.3, 48.0, 3, 135_000_000),
        ("Amoeba", 500.0, 5000, 0.2, 8.0, 3, 290_000_000_000),
        ("Human egg cell", 120.0, 15000, 0.3, 0, 3, 3_200_000_000),
    ]

    size_max = max(c[1] for c in cells)
    comp_max = max(c[2] for c in cells)
    atp_max = max(c[3] for c in cells) or 1.0
    gen_max = max(c[6] for c in cells) or 1.0

    objects = []
    for name, size, comp, atp, div_t, org, gen_bp in cells:
        # Invert division time (faster division → higher channel value)
        div_norm = (1.0 / (1.0 + div_t)) if div_t > 0 else EPSILON
        c_arr = np.array(
            [
                _clip(math.log10(max(size, 0.01)) / math.log10(size_max)),
                _clip(comp / comp_max),
                _clip(atp / atp_max),
                _clip(div_norm),
                _clip(org / 5.0),
                _clip(math.log10(max(gen_bp, 1)) / math.log10(gen_max)),
            ]
        )
        k = _compute(c_arr)
        objects.append(_to_object(name, k, c_arr))

    return Rung(
        number=5,
        name="Cellular",
        scale_meters=1e-5,
        description=(
            "12 cell types — from viruses (edge of life) to amoebae. "
            "This is where chemistry becomes biology: self-replication, "
            "metabolism, information storage (DNA), compartmentalization "
            "(membranes). The cell is the unit of life."
        ),
        channel_names=[
            "size_log",
            "complexity",
            "energy_rate",
            "replication",
            "organization",
            "information",
        ],
        n_channels=6,
        objects=objects,
        bridges_up=[
            Bridge(
                from_rung="Cellular",
                to_rung="Everyday",
                mechanism="Cells organize into tissues → organs → organisms. "
                "Trillions of cells with specialized functions "
                "produce macroscopic behavior: thought, movement, "
                "perception, the experience of being alive.",
                example="~37 trillion human cells → 78 organs → you reading this right now",
                channel_mapping="complexity → tissue specialization; "
                "energy_rate → metabolic heat (→ thermodynamics); "
                "organization → macro structure (→ materials)",
            )
        ],
        bridges_down=[
            Bridge(
                from_rung="Cellular",
                to_rung="Molecular",
                mechanism="Cells are molecular machines: lipid membranes, "
                "protein enzymes, DNA information storage, "
                "ATP energy currency — all molecular.",
                example="ATP synthase: a molecular rotary motor (F₁F₀) that converts proton gradient → ATP at ~100/s",
                channel_mapping="bioactivity → metabolic integration; bond_energy → biochemical pathway energetics",
            )
        ],
    )


# ═══════════════════════════════════════════════════════════════════
# RUNG 6 — EVERYDAY / HUMAN (10⁰ m)
# ═══════════════════════════════════════════════════════════════════


def _build_everyday() -> Rung:
    """The human scale: what we see, touch, hear, and experience.

    Integrates all four everyday physics closures:
    - Thermodynamics (20 materials)
    - Electromagnetism (20 materials)
    - Optics (20 materials)
    - Wave phenomena (24 systems)

    This is the rung you live on. Every other rung exists to
    explain why this one works the way it does.
    """
    from closures.everyday_physics.electromagnetism import EM_MATERIALS, compute_electromagnetic_material
    from closures.everyday_physics.optics import OPTICAL_MATERIALS, compute_optical_material
    from closures.everyday_physics.thermodynamics import THERMAL_MATERIALS, compute_thermal_material
    from closures.everyday_physics.wave_phenomena import WAVE_SYSTEMS, compute_wave_system

    objects = []

    # Thermodynamics (20 materials)
    for name, Cp, k_th, rho, Tm, Tb in THERMAL_MATERIALS:
        r_t = compute_thermal_material(name, Cp, k_th, rho, Tm, Tb)
        objects.append(
            ScaleObject(
                name=f"thermo:{r_t.material}",
                F=r_t.F,
                omega=r_t.omega,
                IC=r_t.IC,
                kappa=r_t.kappa,
                S=r_t.S,
                C=r_t.C,
                gap=r_t.gap,
                regime=r_t.regime,
                trace=r_t.trace,
            )
        )

    # Electromagnetism (20 materials)
    for name, cat, sigma, eps, wf, bg, mu, rho in EM_MATERIALS:
        r_e = compute_electromagnetic_material(name, cat, sigma, eps, wf, bg, mu, rho)
        objects.append(
            ScaleObject(
                name=f"em:{r_e.material}",
                F=r_e.F,
                omega=r_e.omega,
                IC=r_e.IC,
                kappa=r_e.kappa,
                S=r_e.S,
                C=r_e.C,
                gap=r_e.gap,
                regime=r_e.regime,
                trace=r_e.trace,
            )
        )

    # Optics (20 materials)
    for entry in OPTICAL_MATERIALS:
        r_o = compute_optical_material(*entry)
        objects.append(
            ScaleObject(
                name=f"optics:{r_o.material}",
                F=r_o.F,
                omega=r_o.omega,
                IC=r_o.IC,
                kappa=r_o.kappa,
                S=r_o.S,
                C=r_o.C,
                gap=r_o.gap,
                regime=r_o.regime,
                trace=r_o.trace,
            )
        )

    # Wave phenomena (24 systems)
    for entry in WAVE_SYSTEMS:
        r_w = compute_wave_system(*entry)
        objects.append(
            ScaleObject(
                name=f"wave:{r_w.system}",
                F=r_w.F,
                omega=r_w.omega,
                IC=r_w.IC,
                kappa=r_w.kappa,
                S=r_w.S,
                C=r_w.C,
                gap=r_w.gap,
                regime=r_w.regime,
                trace=r_w.trace,
            )
        )

    return Rung(
        number=6,
        name="Everyday",
        scale_meters=1e0,
        description=(
            "84 systems at human scale — thermodynamics (20), "
            "electromagnetism (20), optics (20), wave phenomena (24). "
            "This is the rung you live on. Copper conducts because of "
            "its electron band structure (atomic rung). Sound travels "
            "because molecules transmit vibrations (molecular rung). "
            "Light refracts because photons interact with electron "
            "clouds (subatomic + atomic rungs). Everything here is "
            "explained by the rungs below."
        ),
        channel_names=[
            "varies by subdomain (6 channels each): thermal, EM, optical, wave",
        ],
        n_channels=6,
        objects=objects,
        bridges_up=[
            Bridge(
                from_rung="Everyday",
                to_rung="Geological",
                mechanism="Aggregate enormous numbers of everyday-scale objects: "
                "10²⁵ molecules → a rock; 10⁵⁰ atoms → a planet. "
                "Thermodynamics scales up: convection → plate tectonics.",
                example="Mantle convection: ~10²³ silicate molecules per cm³, "
                "heated by radioactive decay (nuclear rung) → plate tectonics",
                channel_mapping="thermal_conductivity → heat transport; "
                "density → gravitational stratification; "
                "melting_pt → mantle/core boundary",
            )
        ],
        bridges_down=[
            Bridge(
                from_rung="Everyday",
                to_rung="Cellular",
                mechanism="Macroscopic materials are aggregates of cells (biology) "
                "or crystal lattices (minerals). Everyday experience "
                "is produced by cellular nervous systems.",
                example="Wood = dead plant cells (cellulose + lignin); "
                "your perception of wood = neural signaling across ~10¹⁰ neurons",
                channel_mapping="material_properties → cellular composition; wave perception → neural processing",
            )
        ],
    )


# ═══════════════════════════════════════════════════════════════════
# RUNG 7 — GEOLOGICAL (10⁶ m)
# ═══════════════════════════════════════════════════════════════════


def _build_geological() -> Rung:
    """Planetary-scale structures: geology, oceanography, atmosphere.

    The geological rung bridges everyday materials to astronomical
    objects. A planet is an enormous aggregate of atoms, differentiated
    by density (iron core, silicate mantle, gaseous atmosphere).

    Channels:
      mass_log       — log(M/M_earth)
      radius_log     — log(R/R_earth)
      density_norm   — mean density (kg/m³)
      surface_T_norm — surface temperature
      atm_pressure   — atmospheric pressure (log, bar)
      magnetic_norm  — magnetic field strength (normalized)
    """
    # (name, M/M_earth, R/R_earth, ρ [kg/m³], T_surf [K], P_atm [bar], B_rel)
    bodies = [
        ("Mercury", 0.055, 0.383, 5427, 440, 1e-15, 0.011),
        ("Venus", 0.815, 0.949, 5243, 737, 92.0, 0.0),
        ("Earth", 1.0, 1.0, 5514, 288, 1.0, 1.0),
        ("Moon", 0.012, 0.273, 3346, 250, 3e-15, 0.0),
        ("Mars", 0.107, 0.532, 3934, 210, 0.006, 0.0),
        ("Ceres", 0.00016, 0.074, 2162, 168, 0.0, 0.0),
        ("Jupiter", 317.8, 11.21, 1326, 165, 1000.0, 20000.0),
        ("Saturn", 95.16, 9.45, 687, 134, 1000.0, 600.0),
        ("Uranus", 14.54, 4.01, 1271, 76, 1000.0, 50.0),
        ("Neptune", 17.15, 3.88, 1638, 72, 1000.0, 25.0),
        ("Titan", 0.0225, 0.404, 1881, 94, 1.47, 0.0),
        ("Europa", 0.008, 0.245, 3013, 102, 1e-12, 0.0),
        ("Io", 0.015, 0.286, 3528, 130, 1e-9, 0.0),
        ("Pluto", 0.002, 0.186, 1854, 44, 1e-5, 0.0),
    ]

    m_max = max(b[1] for b in bodies)
    r_max = max(b[2] for b in bodies)
    rho_max = max(b[3] for b in bodies)
    t_max = max(b[4] for b in bodies)
    p_max = max(b[5] for b in bodies) or 1.0
    b_max = max(b[6] for b in bodies) or 1.0

    objects = []
    for name, mass, radius, rho, t_surf, p_atm, b_field in bodies:
        c = np.array(
            [
                _clip(math.log10(max(mass, 1e-6)) / math.log10(m_max))
                if m_max > 1
                else _clip(mass / max(m_max, 1e-10)),
                _clip(math.log10(max(radius, 0.01)) / math.log10(r_max))
                if r_max > 1
                else _clip(radius / max(r_max, 1e-10)),
                _clip(rho / rho_max),
                _clip(t_surf / t_max),
                _clip(math.log10(max(p_atm, 1e-20)) / math.log10(max(p_max, 2))),
                _clip(math.log10(max(b_field, 1e-20)) / math.log10(max(b_max, 2))),
            ]
        )
        k = _compute(c)
        objects.append(_to_object(name, k, c))

    return Rung(
        number=7,
        name="Geological",
        scale_meters=1e6,
        description=(
            "14 planetary bodies — from Ceres to Jupiter. Planets are "
            "where everyday materials aggregate into world-scale structures "
            "under gravity. Differentiation (iron sinks, silicates float) "
            "is thermodynamics at planetary scale."
        ),
        channel_names=[
            "mass_log",
            "radius_log",
            "density",
            "surface_temp",
            "atm_pressure_log",
            "magnetic_field_log",
        ],
        n_channels=6,
        objects=objects,
        bridges_up=[
            Bridge(
                from_rung="Geological",
                to_rung="Stellar",
                mechanism="Sufficient mass → gravitational collapse → nuclear "
                "fusion ignition. A planet that gains enough mass "
                "becomes a star. Jupiter is 0.08× too small.",
                example="Protostellar disk → gravitational accretion → ~0.08 M☉ → hydrogen fusion → main sequence star",
                channel_mapping="mass → stellar mass; "
                "density → core pressure → fusion conditions; "
                "magnetic → stellar magnetic activity",
            )
        ],
        bridges_down=[
            Bridge(
                from_rung="Geological",
                to_rung="Everyday",
                mechanism="Planetary properties emerge from the bulk properties "
                "of their constituent materials under self-gravity. "
                "Earth's iron core = 10²⁵ kg of iron (everyday material).",
                example="Earth's magnetic field: convecting liquid iron core "
                "(EM properties from everyday rung) → dynamo → "
                "magnetosphere → protects atmosphere → enables life",
                channel_mapping="material density → planetary density; "
                "material melting_pt → core/mantle boundary; "
                "thermal_conductivity → planetary heat flow",
            )
        ],
    )


# ═══════════════════════════════════════════════════════════════════
# RUNG 8 — STELLAR (10¹⁰ m)
# ═══════════════════════════════════════════════════════════════════


def _build_stellar() -> Rung:
    """Stars: where nuclear physics powers astronomical objects.

    Each star is a nuclear reactor held together by gravity.
    The HR diagram (luminosity vs temperature) IS the kernel
    signature of the stellar population — it maps directly to
    fidelity and regime classification.
    """
    # (name, M/M_sun, T_eff [K], L/L_sun, R/R_sun, metallicity_rel, age_Gyr)
    stars = [
        ("Proxima Cen", 0.12, 3042, 0.0017, 0.154, 0.21, 4.85),
        ("Barnard's Star", 0.16, 3134, 0.0035, 0.196, 0.10, 10.0),
        ("Sun", 1.0, 5778, 1.0, 1.0, 1.0, 4.6),
        ("Alpha Cen A", 1.1, 5790, 1.52, 1.22, 1.5, 5.3),
        ("Sirius A", 2.06, 9940, 25.4, 1.71, 1.0, 0.24),
        ("Vega", 2.14, 9602, 40.12, 2.36, 0.54, 0.45),
        ("Arcturus", 1.08, 4286, 170.0, 25.4, 0.32, 7.1),
        ("Betelgeuse", 11.6, 3600, 126000.0, 887.0, 0.05, 0.01),
        ("Rigel", 21.0, 12100, 120000.0, 78.9, 0.10, 0.008),
        ("Aldebaran", 1.16, 3910, 518.0, 44.13, 0.40, 6.6),
        ("Polaris", 5.4, 6015, 1260.0, 37.5, 1.0, 0.07),
        ("Deneb", 19.0, 8525, 196000.0, 203.0, 0.10, 0.03),
        ("Wolf 359", 0.09, 2800, 0.001, 0.16, 0.29, 1.0),
        ("Eta Carinae", 100.0, 36000, 5e6, 240.0, 2.0, 0.003),
        ("White dwarf (avg)", 0.6, 25000, 0.001, 0.01, 1.0, 8.0),
        ("Neutron star (avg)", 1.4, 1e6, 0.1, 1.5e-5, 1.0, 0.001),
    ]

    objects = []
    for name, mass, teff, lum, radius, metal, _age in stars:
        c = np.array(
            [
                _clip(math.log10(max(mass, 0.01)) / 2.5 + 0.5),
                _clip(teff / 1e6),
                _clip(math.log10(max(lum, 1e-4)) / 7.0 + 0.5),
                _clip(math.log10(max(radius, 1e-6)) / 4.0 + 0.5),
                _clip(metal / 2.5),
                _clip(mass / 150.0),
            ]
        )
        k = _compute(c)
        objects.append(_to_object(name, k, c))

    return Rung(
        number=8,
        name="Stellar",
        scale_meters=1e10,
        description=(
            "16 stars — from red dwarfs to hypergiants, plus compact "
            "remnants (white dwarfs, neutron stars). Stars fuse hydrogen "
            "→ helium → ... → iron (nuclear rung), then die: white dwarf, "
            "neutron star, or black hole. Supernovae scatter heavy elements "
            "back into space → next generation of stars + planets + life."
        ),
        channel_names=[
            "mass_log",
            "T_eff",
            "luminosity_log",
            "radius_log",
            "metallicity",
            "mass_frac",
        ],
        n_channels=6,
        objects=objects,
        bridges_up=[
            Bridge(
                from_rung="Stellar",
                to_rung="Galactic",
                mechanism="Stars cluster under gravity: open clusters → "
                "stellar associations → spiral arms → galaxies. "
                "A galaxy is ~10¹¹ stars orbiting a common center.",
                example="Sun orbits Milky Way center at 230 km/s, 26,000 ly from center, one of ~200 billion stars",
                channel_mapping="stellar mass → galactic mass function; "
                "metallicity → galactic chemical evolution; "
                "luminosity → galaxy luminosity function",
            )
        ],
        bridges_down=[
            Bridge(
                from_rung="Stellar",
                to_rung="Geological",
                mechanism="Stars produce planets from leftover disk material. "
                "Stellar radiation determines planetary climate. "
                "Stellar death enriches ISM with heavy elements.",
                example="Solar wind → magnetosphere interaction; "
                "solar luminosity → Earth's climate; "
                "stellar nucleosynthesis → planetary composition",
                channel_mapping="T_eff → planetary surface temperature; "
                "metallicity → planetary composition; "
                "luminosity → habitable zone boundaries",
            )
        ],
    )


# ═══════════════════════════════════════════════════════════════════
# RUNG 9 — GALACTIC (10²¹ m)
# ═══════════════════════════════════════════════════════════════════


def _build_galactic() -> Rung:
    """Galaxies: island universes of 10⁸-10¹² stars.

    Channels:
      mass_log       — log(M/M_sun) total (including dark matter)
      luminosity_log — log(L/L_sun) total
      size_log       — log(R/kpc) effective radius
      sfr_norm       — star formation rate (M_sun/yr, normalized)
      morphology     — Hubble type (0=E, 0.5=S0, 1=Sa..Sd)
      dm_fraction    — dark matter fraction of total mass
    """
    # (name, log M/M_sun, log L/L_sun, R_eff kpc, SFR M_sun/yr, morph, DM_frac)
    galaxies = [
        ("Milky Way", 12.0, 10.7, 13.0, 1.65, 0.7, 0.90),
        ("Andromeda (M31)", 12.3, 10.9, 21.0, 0.35, 0.7, 0.90),
        ("LMC", 10.6, 9.3, 4.5, 0.20, 0.9, 0.85),
        ("SMC", 10.0, 8.7, 2.0, 0.05, 0.9, 0.85),
        ("M87 (Virgo A)", 13.2, 11.2, 40.0, 0.0, 0.0, 0.95),
        ("NGC 1300 (barred)", 11.5, 10.5, 15.0, 2.5, 0.8, 0.90),
        ("M82 (starburst)", 11.0, 10.8, 3.6, 10.0, 0.9, 0.85),
        ("IC 1101 (giant E)", 13.8, 12.0, 200.0, 0.0, 0.0, 0.97),
        ("Segue 1 (dwarf)", 8.5, 5.0, 0.03, 0.0, 0.0, 0.999),
        ("Sombrero (M104)", 12.8, 11.0, 15.0, 0.1, 0.5, 0.90),
        ("Whirlpool (M51)", 11.8, 10.6, 12.0, 3.0, 0.7, 0.90),
        ("Cartwheel", 11.5, 10.8, 22.0, 15.0, 1.0, 0.85),
    ]

    m_max = max(g[1] for g in galaxies)
    l_max = max(g[2] for g in galaxies)
    r_max = max(g[3] for g in galaxies)
    sfr_max = max(g[4] for g in galaxies) or 1.0

    objects = []
    for name, log_m, log_l, r_eff, sfr, morph, dm_frac in galaxies:
        c = np.array(
            [
                _clip(log_m / m_max),
                _clip(log_l / l_max),
                _clip(math.log10(max(r_eff, 0.01)) / math.log10(r_max)),
                _clip(sfr / sfr_max),
                _clip(morph),
                _clip(dm_frac),
            ]
        )
        k = _compute(c)
        objects.append(_to_object(name, k, c))

    return Rung(
        number=9,
        name="Galactic",
        scale_meters=1e21,
        description=(
            "12 galaxies — from dwarf (Segue 1, ~10⁵ stars) to giant "
            "ellipticals (IC 1101, ~10¹⁴ stars). Dark matter dominates "
            "mass at this scale (85-99.9%). Galaxy morphology (spiral, "
            "elliptical, irregular) reflects formation history and "
            "merger dynamics."
        ),
        channel_names=[
            "mass_log",
            "luminosity_log",
            "size_log",
            "star_formation_rate",
            "morphology",
            "dark_matter_fraction",
        ],
        n_channels=6,
        objects=objects,
        bridges_up=[
            Bridge(
                from_rung="Galactic",
                to_rung="Cosmological",
                mechanism="Galaxies cluster → superclusters → cosmic web. "
                "Dark energy (Ω_Λ) determines whether clusters "
                "are bound or expanding apart.",
                example="Virgo Supercluster → Laniakea → cosmic web filaments → "
                "large-scale structure traces initial density fluctuations (σ₈)",
                channel_mapping="galaxy mass → Ω_m; dark_matter_fraction → Ω_c; luminosity_function → Ω_b",
            )
        ],
        bridges_down=[
            Bridge(
                from_rung="Galactic",
                to_rung="Stellar",
                mechanism="A galaxy is a gravitationally bound system of stars, "
                "gas, dust, and dark matter. Stars form from molecular "
                "clouds within galaxies (SFR channel).",
                example="Milky Way: ~200 billion stars + ~1.5 M_sun/yr new stars "
                "forming in spiral arm molecular clouds",
                channel_mapping="star_formation_rate → stellar population; "
                "metallicity (galactic evolution) → stellar metallicity; "
                "morphology → stellar orbits (disk vs bulge)",
            )
        ],
    )


# ═══════════════════════════════════════════════════════════════════
# RUNG 10 — COSMOLOGICAL (10²⁶ m)
# ═══════════════════════════════════════════════════════════════════


def _build_cosmological() -> Rung:
    """The observable universe: cosmological parameters across epochs.

    Integrates the cosmology closure (Planck 2018 + BAO + SNe Ia).
    """
    try:
        from closures.astronomy.cosmology import COSMOLOGICAL_EPOCHS, compute_cosmological_epoch
    except ImportError:
        # Fallback inline data
        return _build_cosmological_inline()

    objects = []
    for entry in COSMOLOGICAL_EPOCHS:
        r = compute_cosmological_epoch(*entry)
        objects.append(
            ScaleObject(
                name=r.epoch,
                F=r.F,
                omega=r.omega,
                IC=r.IC,
                kappa=r.kappa,
                S=r.S,
                C=r.C,
                gap=r.gap,
                regime=r.regime,
                trace=r.trace,
            )
        )

    return Rung(
        number=10,
        name="Cosmological",
        scale_meters=1e26,
        description=(
            "The observable universe across 6 cosmic epochs — from "
            "inflation exit (10⁻³² s) to the present (13.8 Gyr). "
            "Cosmological parameters (H₀, Ω_b, Ω_c, Ω_Λ, T_CMB, n_s, σ₈) "
            "are the trace vector. The universe IS a trace vector."
        ),
        channel_names=[
            "H₀_log",
            "Ω_b",
            "Ω_c",
            "Ω_Λ",
            "T_CMB_log",
            "n_s",
            "σ₈",
            "τ_reion",
        ],
        n_channels=8,
        objects=objects,
        bridges_down=[
            Bridge(
                from_rung="Cosmological",
                to_rung="Galactic",
                mechanism="Initial density fluctuations (σ₈, n_s) → gravitational "
                "collapse → dark matter halos → galaxy formation. "
                "Dark energy (Ω_Λ) determines cosmic expansion rate.",
                example="σ₈ = 0.811 → amplitude of matter fluctuations → galaxy clustering on 8 h⁻¹ Mpc scale",
                channel_mapping="σ₈ → galaxy mass function; "
                "Ω_c → dark matter halos; "
                "Ω_b → baryonic infall → star formation",
            )
        ],
    )


def _build_cosmological_inline() -> Rung:
    """Fallback cosmological data if cosmology closure is not available."""
    epochs = [
        ("Inflation exit", [0.674, 0.0493, 0.265, 0.685, 0.99, 0.965, 0.001, 0.054]),
        ("Recombination (z=1089)", [0.674, 0.0493, 0.265, 0.685, 0.83, 0.965, 0.811, 0.054]),
        ("Dark energy onset (z=0.7)", [0.674, 0.0493, 0.265, 0.685, 0.30, 0.965, 0.811, 0.054]),
        ("BAO epoch (z=0.5)", [0.674, 0.0493, 0.265, 0.685, 0.25, 0.965, 0.811, 0.054]),
        ("SNe Ia epoch (z=0.1)", [0.674, 0.0493, 0.265, 0.685, 0.10, 0.965, 0.811, 0.054]),
        ("Present (z=0)", [0.674, 0.0493, 0.265, 0.685, 0.074, 0.965, 0.811, 0.054]),
    ]

    objects = []
    for name, channels in epochs:
        c = np.array(channels)
        c = np.clip(c, EPSILON, 1 - EPSILON)
        k = _compute(c)
        objects.append(_to_object(name, k, c))

    return Rung(
        number=10,
        name="Cosmological",
        scale_meters=1e26,
        description="Observable universe across cosmic epochs (inline fallback).",
        channel_names=["H₀", "Ω_b", "Ω_c", "Ω_Λ", "T_CMB", "n_s", "σ₈", "τ_reion"],
        n_channels=8,
        objects=objects,
    )


# ═══════════════════════════════════════════════════════════════════
# BUILD THE COMPLETE LADDER
# ═══════════════════════════════════════════════════════════════════


def build_scale_ladder() -> ScaleLadder:
    """Construct the complete scale ladder from Planck to cosmos.

    Returns a ScaleLadder containing all 11 rungs, every object,
    every bridge, and Tier-1 verification across the entire ladder.
    """
    t0 = time.perf_counter()

    rungs = [
        _build_planck(),  # Rung 0:  10⁻³⁵ m
        _build_subatomic(),  # Rung 1:  10⁻¹⁸ m
        _build_nuclear(),  # Rung 2:  10⁻¹⁵ m
        _build_atomic(),  # Rung 3:  10⁻¹⁰ m
        _build_molecular(),  # Rung 4:  10⁻⁹  m
        _build_cellular(),  # Rung 5:  10⁻⁵  m
        _build_everyday(),  # Rung 6:  10⁰   m
        _build_geological(),  # Rung 7:  10⁶   m
        _build_stellar(),  # Rung 8:  10¹⁰  m
        _build_galactic(),  # Rung 9:  10²¹  m
        _build_cosmological(),  # Rung 10: 10²⁶  m
    ]

    # Total object count
    total_objects = sum(r.n_objects for r in rungs)

    # ── Tier-1 verification across every object at every rung ──
    tier1_violations = 0
    for rung in rungs:
        for obj in rung.objects:
            # Pillar 1: F + ω = 1
            if abs(obj.F + obj.omega - 1.0) > 1e-5:
                tier1_violations += 1
            # Pillar 2: IC ≤ F
            if obj.IC > obj.F + 1e-5:
                tier1_violations += 1
            # Pillar 3: IC ≈ exp(κ)
            if obj.kappa > -500 and abs(obj.IC - math.exp(obj.kappa)) > 1e-4:
                tier1_violations += 1

    # ── Three Pillars summary ──
    all_F = [o.F for r in rungs for o in r.objects]
    all_omega = [o.omega for r in rungs for o in r.objects]
    all_IC = [o.IC for r in rungs for o in r.objects]
    all_gap = [o.gap for r in rungs for o in r.objects]

    pillar_results = {
        "pillar_1_duality": {
            "identity": "F + ω = 1",
            "max_error": max(abs(f + o - 1.0) for f, o in zip(all_F, all_omega, strict=True)),
            "status": "EXACT" if tier1_violations == 0 else "VIOLATED",
        },
        "pillar_2_integrity_bound": {
            "identity": "IC ≤ F",
            "max_violation": max(ic - f for ic, f in zip(all_IC, all_F, strict=True)),
            "status": "PROVEN" if all(ic <= f + 1e-10 for ic, f in zip(all_IC, all_F, strict=True)) else "VIOLATED",
        },
        "pillar_3_exponential_bridge": {
            "identity": "IC = exp(κ)",
            "status": "EXACT" if tier1_violations == 0 else "VIOLATED",
        },
        "fidelity_range": (round(float(min(all_F)), 4), round(float(max(all_F)), 4)),
        "gap_range": (round(float(min(all_gap)), 4), round(float(max(all_gap)), 4)),
        "mean_gap": round(float(sum(all_gap)) / len(all_gap), 4),
        "total_objects": total_objects,
        "total_violations": tier1_violations,
        "dynamic_range_OOM": 61,  # 10⁻³⁵ to 10²⁶
    }

    dt = (time.perf_counter() - t0) * 1000

    return ScaleLadder(
        rungs=rungs,
        total_objects=total_objects,
        total_tier1_violations=tier1_violations,
        pillar_results=pillar_results,
        build_time_ms=round(dt, 1),
    )


# ═══════════════════════════════════════════════════════════════════
# BRIDGE MAP — Explicit connections between every adjacent pair
# ═══════════════════════════════════════════════════════════════════


def get_bridge_map(ladder: ScaleLadder) -> list[dict[str, str]]:
    """Extract the complete bridge map from the ladder.

    Returns a list of bridge descriptions showing the causal
    chain from Planck to cosmos.
    """
    bridges = []
    for rung in ladder.rungs:
        for b in rung.bridges_up:
            bridges.append(
                {
                    "from": b.from_rung,
                    "to": b.to_rung,
                    "mechanism": b.mechanism,
                    "example": b.example,
                    "channels": b.channel_mapping,
                }
            )
    return bridges


# ═══════════════════════════════════════════════════════════════════
# DISPLAY
# ═══════════════════════════════════════════════════════════════════


def display_ladder(ladder: ScaleLadder, *, verbose: bool = False) -> None:
    """Display the complete scale ladder."""
    print()
    print("╔═══════════════════════════════════════════════════════════════════════════════╗")
    print("║                                                                             ║")
    print("║   SCALE LADDER — From Minimal to Universal                                  ║")
    print("║   Scala naturae non narratur: mensuratur.                                   ║")
    print("║   (The ladder of nature is not narrated: it is measured.)                    ║")
    print("║                                                                             ║")
    print("╚═══════════════════════════════════════════════════════════════════════════════╝")

    # ── Overview ────────────────────────────────────────────────
    print(f"\n  Dynamic range:  {ladder.pillar_results.get('dynamic_range_OOM', 61)} orders of magnitude")
    print(f"  Total objects:  {ladder.total_objects}")
    print(f"  Tier-1 violations: {ladder.total_tier1_violations}")
    print(f"  Build time:     {ladder.build_time_ms:.0f} ms")

    # ── Rung-by-rung summary ────────────────────────────────────
    print("\n" + "═" * 80)
    print("  THE ELEVEN RUNGS")
    print("═" * 80)

    print(f"\n  {'#':<4} {'Rung':<15} {'Scale':>10} {'N':>5} {'⟨F⟩':>7} {'⟨ω⟩':>7} {'⟨IC⟩':>7} {'⟨Δ⟩':>7} {'Regimes'}")
    print("  " + "─" * 78)

    for rung in ladder.rungs:
        scale_str = f"10^{math.log10(rung.scale_meters):+.0f} m" if rung.scale_meters > 0 else "0 m"
        regimes = rung.regime_counts
        regimes_str = ", ".join(f"{k[0]}:{v}" for k, v in sorted(regimes.items()))

        print(
            f"  {rung.number:<4d} {rung.name:<15} {scale_str:>10} {rung.n_objects:>5d} "
            f"{rung.mean_F:>7.4f} {rung.mean_omega:>7.4f} "
            f"{rung.mean_IC:>7.4f} {rung.mean_gap:>7.4f} {regimes_str}"
        )

    # ── Bridge chain ────────────────────────────────────────────
    print("\n" + "═" * 80)
    print("  THE BRIDGE CHAIN — How Each Rung Connects to the Next")
    print("═" * 80)

    for rung in ladder.rungs:
        for bridge in rung.bridges_up:
            print(f"\n  {bridge.from_rung} → {bridge.to_rung}")
            print(f"    Mechanism: {bridge.mechanism}")
            print(f"    Example:   {bridge.example}")
            if verbose:
                print(f"    Channels:  {bridge.channel_mapping}")

    # ── Three Pillars ───────────────────────────────────────────
    print("\n" + "═" * 80)
    print("  THREE PILLARS — Verified Across All 11 Rungs")
    print("═" * 80)

    pr = ladder.pillar_results
    print(
        f"\n  Pillar 1 (Duality):        F + ω = 1     max error = {pr['pillar_1_duality']['max_error']:.2e}  [{pr['pillar_1_duality']['status']}]"
    )
    print(
        f"  Pillar 2 (Integrity Bound): IC ≤ F        max violation = {pr['pillar_2_integrity_bound']['max_violation']:.2e}  [{pr['pillar_2_integrity_bound']['status']}]"
    )
    print(f"  Pillar 3 (Exp Bridge):      IC = exp(κ)   [{pr['pillar_3_exponential_bridge']['status']}]")
    print(f"\n  Fidelity range:  F ∈ {pr['fidelity_range']}")
    print(f"  Gap range:       Δ ∈ {pr['gap_range']}")
    print(f"  Mean gap:        ⟨Δ⟩ = {pr['mean_gap']}")

    # ── Rung details (verbose) ──────────────────────────────────
    if verbose:
        for rung in ladder.rungs:
            print(f"\n{'─' * 80}")
            print(f"  RUNG {rung.number}: {rung.name} ({rung.scale_meters:.0e} m)")
            print(f"  {rung.description}")
            print(f"  Channels: {', '.join(rung.channel_names)}")
            print(f"\n  {'Object':<30} {'F':>7} {'ω':>7} {'IC':>7} {'Δ':>7} {'Regime':<10}")
            print("  " + "─" * 70)
            for obj in rung.objects[:20]:  # limit display
                print(
                    f"  {obj.name:<30} {obj.F:>7.4f} {obj.omega:>7.4f} {obj.IC:>7.4f} {obj.gap:>7.4f} {obj.regime:<10}"
                )
            if rung.n_objects > 20:
                print(f"  ... ({rung.n_objects - 20} more)")

    # ── Final summary ───────────────────────────────────────────
    print("\n" + "═" * 80)
    print("  VERDICT")
    print("═" * 80)
    print(f"\n  {ladder.total_objects} objects across 11 rungs spanning 61 orders of magnitude.")
    print(f"  {ladder.total_tier1_violations} Tier-1 violations.")

    if ladder.total_tier1_violations == 0:
        print("\n  The three pillars hold everywhere.")
        print("  From the Planck floor to the cosmic horizon,")
        print("  the same minimal structure persists.")
        print("\n  Solum quod redit, reale est.")
        print("  (Only what returns is real.)")
    else:
        print(f"\n  WARNING: {ladder.total_tier1_violations} violations detected.")

    print()


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scale Ladder — From Minimal to Universal")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show per-object details")
    args = parser.parse_args()

    ladder = build_scale_ladder()
    display_ladder(ladder, verbose=args.verbose)

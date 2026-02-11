"""Gap-Capture Duality & Seam Stamp 1 Mini (SS1M) — MATL.INTSTACK.v1

Derives the gap-capture measurement unit for any element by collecting
ω profiles across all available material closures, feeding both sides
(captures = 1−ω, gaps = ω) through the GCD kernel, and producing:

  1. A single scalar measurement unit: the IC asymmetry ratio
        R_IC = IC(captures) / IC(gaps)
     which quantifies how much more coherent the captured physics is
     than the missing physics.

  2. A full seam receipt with all meta-kernel invariants.

  3. An SS1M (Seam Stamp 1 Mini) — a compact data HUD for cross-domain
     portability.

Theory:
  The standard model uses ±1 (positive/negative forces, charges, spins).
  GCD does not assume sign — it derives structure from what is present
  (captures) and what is absent (gaps). The duality is exact:

      F_meta(captures) + F_meta(gaps) = 1.0        ∀ elements

  S (entropy) and C (curvature) are mirror-invariant — they describe the
  SHAPE of the profile, identical on both sides. Only IC (the geometric
  mean) breaks the symmetry, because the geometric mean is sensitive to
  zeros: a single channel where the model perfectly captures or perfectly
  misses dominates IC on its respective side.

  This means R_IC is the observable that tells you whether your models
  have more structure in what they found or in what they're missing.

  R_IC > 1:  captures are more coherent than gaps (model is working)
  R_IC ≈ 1:  captures and gaps are equally coherent (model at parity)
  R_IC < 1:  gaps are more coherent than captures (model is anti-correlated)

The SS1M (Seam Stamp 1 Mini) is a compact data HUD:

  ┌─────────────────────────────────────┐
  │  SS1M · Fe                          │
  │  ─────────────────────────────────  │
  │  channels: 4/8  [coh mag deb ban]   │
  │  F_cap: 0.7838   ω_cap: 0.2162     │
  │  IC_cap: 0.7581  IC_gap: 0.0109    │
  │  R_IC: 69.6      ln(R): 4.243      │
  │  S: 0.4008  C: 0.3773  (invariant) │
  │  Γ: 0.0129  budget: 0.9871         │
  │  regime: Watch   seam: OPEN         │
  └─────────────────────────────────────┘

Cross-references:
  Kernel:    src/umcp/kernel_optimized.py (compute_kernel_outputs)
  Seam:      src/umcp/seam_optimized.py (SeamRecord)
  Closures:  closures/materials_science/*.py (per-channel ω)
  Contract:  src/umcp/frozen_contract.py (ε, p, tol_seam)
  Axiom:     AXIOM.md ("What returns through collapse is real")
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np

from closures.materials_science.element_database import (
    get_element,
)

# ── Channel labels ───────────────────────────────────────────────
CHANNEL_LABELS = [
    "coh",  # cohesive energy
    "mag",  # magnetic properties
    "deb",  # debye thermal
    "ban",  # band structure
    "ela",  # elastic moduli
    "sur",  # surface catalysis
    "bcs",  # BCS superconductivity
    "pha",  # phase transitions
]

CHANNEL_FULL_NAMES = {
    "coh": "cohesive_energy",
    "mag": "magnetic_properties",
    "deb": "debye_thermal",
    "ban": "band_structure",
    "ela": "elastic_moduli",
    "sur": "surface_catalysis",
    "bcs": "bcs_superconductivity",
    "pha": "phase_transitions",
}

# ── Reference coverage maps ─────────────────────────────────────
# Which elements each closure actually has measured reference data for.
# TYPE-A closures (pred vs. measured): ω=0 for unlisted elements is a BUG
#   (self-referencing — prediction becomes its own reference → false perfect match).
# TYPE-B closures (physics-based): ω is physically meaningful but only precise
#   when the element's specific constants (Θ_D, T_c) are known.
# Elements NOT in these sets get available=False (excluded from kernel).
#
# Keys: pure element symbols. Compound keys (NaCl, MgO etc.) are not matched
# since we only feed element symbols through gap_capture.

HAS_COHESIVE: set[str] = {
    "Li",
    "Na",
    "K",
    "Rb",
    "Cs",
    "Be",
    "Mg",
    "Ca",
    "Sr",
    "Ba",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Zr",
    "Nb",
    "Mo",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Al",
    "Ga",
    "In",
    "Sn",
    "Pb",
    "C",
    "Si",
    "Ge",
    "Ne",
    "Ar",
    "Kr",
    "Xe",
}  # 48 entries (minus 4 compounds = 44 pure elements)

HAS_MAGNETIC: set[str] = {
    "Fe",
    "Co",
    "Ni",
    "Gd",
    "Dy",
    "Cr",
    "Mn",
    "Cu",
    "Au",
    "Bi",
    "Al",
    "Pt",
}  # 12 pure elements (5 compounds excluded)

HAS_DEBYE: set[str] = {
    "Li",
    "Na",
    "K",
    "Rb",
    "Cs",
    "Be",
    "Mg",
    "Ca",
    "Al",
    "Cu",
    "Ag",
    "Au",
    "Fe",
    "Co",
    "Ni",
    "Cr",
    "Ti",
    "V",
    "Nb",
    "Mo",
    "W",
    "Pt",
    "Pd",
    "Rh",
    "Ir",
    "Zn",
    "Cd",
    "Pb",
    "Sn",
    "Si",
    "Ge",
}  # 31 pure elements (7 compounds excluded)

HAS_BAND: set[str] = {
    "Li",
    "Na",
    "K",
    "Cu",
    "Ag",
    "Au",
    "Fe",
    "Al",
    "W",
    "Pt",
    "Bi",
    "Sb",
    "As",
    "Si",
    "Ge",
}  # 15 pure elements (28 compounds excluded)

HAS_ELASTIC: set[str] = {
    "Li",
    "Na",
    "K",
    "Rb",
    "Cs",
    "Be",
    "Mg",
    "Ca",
    "Sr",
    "Ba",
    "Ti",
    "V",
    "Cr",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Nb",
    "Mo",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Al",
    "Si",
    "Ge",
}  # 34 pure elements (7 compounds excluded)

HAS_SURFACE: set[str] = {
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Ir",
    "Pt",
    "Au",
    "W",
    "Mo",
    "Ti",
    "Al",
    "Zn",
}  # 16 pure elements

HAS_BCS: set[str] = {
    "Nb",
    "Pb",
    "Sn",
    "Al",
    "In",
    "V",
    "Ta",
    "Hg",
    "Zn",
}  # 9 pure elements (3 compounds excluded)

HAS_PHASE: set[str] = {
    "Fe",
    "Co",
    "Ni",
    "Gd",
    "Nb",
    "Pb",
}  # 6 pure elements with direct symbol match (11 use compound/variant keys)


# ── Element → Z mapping (periodic table) ────────────────────────
ELEMENT_Z: dict[str, int] = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Kr": 36,
    "Rb": 37,
    "Sr": 38,
    "Y": 39,
    "Zr": 40,
    "Nb": 41,
    "Mo": 42,
    "Tc": 43,
    "Ru": 44,
    "Rh": 45,
    "Pd": 46,
    "Ag": 47,
    "Cd": 48,
    "In": 49,
    "Sn": 50,
    "Sb": 51,
    "Te": 52,
    "I": 53,
    "Xe": 54,
    "Cs": 55,
    "Ba": 56,
    "La": 57,
    "Ce": 58,
    "Pr": 59,
    "Nd": 60,
    "Pm": 61,
    "Sm": 62,
    "Eu": 63,
    "Gd": 64,
    "Tb": 65,
    "Dy": 66,
    "Ho": 67,
    "Er": 68,
    "Tm": 69,
    "Yb": 70,
    "Lu": 71,
    "Hf": 72,
    "Ta": 73,
    "W": 74,
    "Re": 75,
    "Os": 76,
    "Ir": 77,
    "Pt": 78,
    "Au": 79,
    "Hg": 80,
    "Tl": 81,
    "Pb": 82,
    "Bi": 83,
    "Po": 84,
    "At": 85,
    "Rn": 86,
    "Fr": 87,
    "Ra": 88,
    "Ac": 89,
    "Th": 90,
    "Pa": 91,
    "U": 92,
    "Np": 93,
    "Pu": 94,
    "Am": 95,
    "Cm": 96,
    "Bk": 97,
    "Cf": 98,
    "Es": 99,
    "Fm": 100,
    "Md": 101,
    "No": 102,
    "Lr": 103,
    "Rf": 104,
    "Db": 105,
    "Sg": 106,
    "Bh": 107,
    "Hs": 108,
    "Mt": 109,
    "Ds": 110,
    "Rg": 111,
    "Cn": 112,
    "Nh": 113,
    "Fl": 114,
    "Mc": 115,
    "Lv": 116,
    "Ts": 117,
    "Og": 118,
}


# ── Regime gates (Tier-0, from KERNEL_SPECIFICATION.md) ─────────
def _classify_regime(omega: float, F: float, S: float, C: float) -> str:
    """Protocol-level regime classification (Tier-0)."""
    if omega >= 0.30:
        return "Collapse"
    if omega < 0.038 and F > 0.90 and S < 0.15 and C < 0.14:
        return "Stable"
    return "Watch"


# ── Result dataclasses ───────────────────────────────────────────


@dataclass
class ChannelResult:
    """Per-channel ω and source data."""

    label: str
    full_name: str
    omega: float
    available: bool
    raw_predicted: float | None = None
    raw_measured: float | None = None
    unit: str = ""
    omega_type: str = ""  # "model_vs_reality" or "physical_observable"


@dataclass
class SeamReceipt:
    """Full seam receipt — complete metadata for one element."""

    element: str
    Z: int
    timestamp: str
    channels_available: int
    channels_total: int
    channel_labels: list[str]
    omega_profile: list[float]
    capture_profile: list[float]
    weights: list[float]

    # Captures side (F_eff = 1 - ω fed as trace)
    F_cap: float
    omega_cap: float
    S_cap: float
    C_cap: float
    kappa_cap: float
    IC_cap: float

    # Gaps side (ω fed as trace)
    F_gap: float
    omega_gap: float
    S_gap: float
    C_gap: float
    kappa_gap: float
    IC_gap: float

    # Duality check
    duality_sum: float  # F_cap + F_gap (should be 1.0)
    duality_exact: bool

    # Mirror invariance
    S_invariant: bool  # S_cap == S_gap
    C_invariant: bool  # C_cap == C_gap

    # The derived measurement unit
    R_IC: float  # IC_cap / IC_gap
    ln_R_IC: float  # ln(R_IC) — the log-scale unit

    # Seam budget
    gamma_cap: float  # Γ(ω_cap) = ω³/(1−ω+ε)
    budget: float  # 1 - Γ

    # Regime
    regime_cap: str
    regime_gap: str

    # Seam status
    seam_status: str  # OPEN (first observation) or CLOSED (return verified)

    # Integrity hash
    receipt_hash: str

    # Fundamental atomic properties (from element_database)
    atomic_mass: float | None = None
    standard_state: str | None = None
    electron_config: str | None = None
    oxidation_states: tuple[int, ...] | None = None
    electronegativity: float | None = None
    atomic_radius_pm: float | None = None
    ionization_energy_eV: float | None = None
    electron_affinity_eV: float | None = None
    melting_point_K: float | None = None
    boiling_point_K: float | None = None
    density_g_cm3: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        d = self.to_dict()

        # Ensure numpy types are JSON-serializable
        def _convert(obj: Any) -> Any:
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            return obj

        return json.dumps({k: _convert(v) for k, v in d.items()}, indent=indent)


@dataclass
class SS1M:
    """Seam Stamp 1 Mini — compact data HUD.

    This is the portable version of the seam receipt.
    Everything you need to compare across domains in one glance.
    """

    element: str
    Z: int
    n_channels: int
    channel_tags: str  # e.g. "coh mag deb ban"
    F_cap: float
    omega_cap: float
    IC_cap: float
    IC_gap: float
    R_IC: float
    ln_R: float
    S: float  # mirror-invariant
    C: float  # mirror-invariant
    gamma: float
    budget: float
    regime: str
    seam: str  # OPEN or CLOSED

    def hud(self) -> str:
        """Render the SS1M as a compact text HUD."""
        w = 41
        lines = [
            f"┌{'─' * (w - 2)}┐",
            f"│  SS1M · {self.element:<{w - 12}}│",
            f"│  {'─' * (w - 6)}  │",
            f"│  channels: {self.n_channels}/8  [{self.channel_tags}]{' ' * max(0, w - 19 - len(self.channel_tags) - len(str(self.n_channels)))}│",
            f"│  F_cap: {self.F_cap:<8.4f}  ω_cap: {self.omega_cap:<8.4f}  │",
            f"│  IC_cap: {self.IC_cap:<8.4f} IC_gap: {self.IC_gap:<8.6f}│",
            f"│  R_IC: {self.R_IC:<9.1f} ln(R): {self.ln_R:<8.3f}  │",
            f"│  S: {self.S:<7.4f} C: {self.C:<7.4f} (invariant) │",
            f"│  Γ: {self.gamma:<7.4f} budget: {self.budget:<7.4f}       │",
            f"│  regime: {self.regime:<8s} seam: {self.seam:<8s}   │",
            f"└{'─' * (w - 2)}┘",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return asdict(self)


# ── Core computation ─────────────────────────────────────────────


def _collect_omega_profile(
    symbol: str,
    Z: int,
    temperature_K: float = 300.0,
) -> list[ChannelResult]:
    """Collect ω from every available closure for this element.

    Returns a list of ChannelResult, one per CHANNEL_LABELS.

    Availability rules:
    - TYPE-A closures (pred vs. measured): available ONLY if the element has
      real reference data in that closure's database. Otherwise ω=0 from
      self-referencing is a false "perfect match" — mark unavailable.
    - TYPE-B closures (physics-based, e.g. Debye, phase): available ONLY if
      the element has known physical constants (Θ_D, T_c) in the closure's
      database. Generic defaults are not "observations."
    - NaN or exception: always unavailable.
    """
    results: list[ChannelResult] = []

    def _unavailable(label: str, full_name: str) -> ChannelResult:
        return ChannelResult(
            label=label,
            full_name=full_name,
            omega=float("nan"),
            available=False,
        )

    # ── Channel 0: Cohesive energy ───────────────────────────────
    if symbol in HAS_COHESIVE:
        try:
            from closures.materials_science.cohesive_energy import compute_cohesive_energy

            r = compute_cohesive_energy(Z, symbol=symbol)
            results.append(
                ChannelResult(
                    label="coh",
                    full_name="cohesive_energy",
                    omega=r.omega_eff,
                    available=True,
                    raw_predicted=r.E_coh_eV,
                    raw_measured=r.E_coh_measured_eV,
                    unit="eV/atom",
                    omega_type="model_vs_reality",
                )
            )
        except Exception:
            results.append(_unavailable("coh", "cohesive_energy"))
    else:
        results.append(_unavailable("coh", "cohesive_energy"))

    # ── Channel 1: Magnetic properties ───────────────────────────
    if symbol in HAS_MAGNETIC:
        try:
            from closures.materials_science.magnetic_properties import compute_magnetic_properties

            r = compute_magnetic_properties(Z, symbol=symbol, T_K=temperature_K)
            results.append(
                ChannelResult(
                    label="mag",
                    full_name="magnetic_properties",
                    omega=r.omega_eff,
                    available=True,
                    raw_predicted=r.M_total_B,
                    raw_measured=r.M_sat_B,
                    unit="μ_B/atom",
                    omega_type="model_vs_reality",
                )
            )
        except Exception:
            results.append(_unavailable("mag", "magnetic_properties"))
    else:
        results.append(_unavailable("mag", "magnetic_properties"))

    # ── Channel 2: Debye thermal ─────────────────────────────────
    if symbol in HAS_DEBYE:
        try:
            from closures.materials_science.debye_thermal import compute_debye_thermal

            r = compute_debye_thermal(temperature_K, symbol=symbol)
            results.append(
                ChannelResult(
                    label="deb",
                    full_name="debye_thermal",
                    omega=r.omega_eff,
                    available=True,
                    raw_predicted=r.C_V_J_mol_K,
                    raw_measured=3.0 * 8.314,  # 3R (Dulong-Petit limit)
                    unit="J/(mol·K)",
                    omega_type="physical_observable",
                )
            )
        except Exception:
            results.append(_unavailable("deb", "debye_thermal"))
    else:
        results.append(_unavailable("deb", "debye_thermal"))

    # ── Channel 3: Band structure ────────────────────────────────
    if symbol in HAS_BAND:
        try:
            from closures.materials_science.band_structure import compute_band_structure

            r = compute_band_structure(Z, symbol=symbol)
            results.append(
                ChannelResult(
                    label="ban",
                    full_name="band_structure",
                    omega=r.omega_eff,
                    available=True,
                    raw_predicted=r.E_g_eV,
                    raw_measured=r.E_g_measured_eV,
                    unit="eV",
                    omega_type="model_vs_reality",
                )
            )
        except Exception:
            results.append(_unavailable("ban", "band_structure"))
    else:
        results.append(_unavailable("ban", "band_structure"))

    # ── Channel 4: Elastic moduli ────────────────────────────────
    # Elastic moduli requires E_coh and r0 from cohesive_energy closure.
    # Chain: cohesive_energy → elastic_moduli
    if symbol in HAS_ELASTIC:
        try:
            from closures.materials_science.elastic_moduli import compute_elastic_moduli

            # Get cohesive data first for chaining
            coh_ch = next((c for c in results if c.label == "coh" and c.available), None)
            e_coh = coh_ch.raw_predicted if coh_ch else 0.0
            # Estimate r0 from cohesive energy (Å) — rough scaling
            r0_est = 2.5  # default metallic nearest-neighbor
            if e_coh and e_coh > 0:
                r0_est = max(2.0, min(4.0, 2.5 * (4.0 / e_coh) ** 0.3))
            r = compute_elastic_moduli(e_coh or 4.0, r0_est, symbol=symbol)
            results.append(
                ChannelResult(
                    label="ela",
                    full_name="elastic_moduli",
                    omega=r.omega_eff,
                    available=True,
                    raw_predicted=r.K_GPa,
                    raw_measured=r.K_measured_GPa,
                    unit="GPa",
                    omega_type="model_vs_reality",
                )
            )
        except Exception:
            results.append(_unavailable("ela", "elastic_moduli"))
    else:
        results.append(_unavailable("ela", "elastic_moduli"))

    # ── Channel 5: Surface catalysis ─────────────────────────────
    if symbol in HAS_SURFACE:
        try:
            from closures.materials_science.surface_catalysis import compute_surface_catalysis

            r = compute_surface_catalysis(0, symbol=symbol)  # E_coh auto-looked up
            results.append(
                ChannelResult(
                    label="sur",
                    full_name="surface_catalysis",
                    omega=r.omega_eff,
                    available=True,
                    raw_predicted=r.gamma_J_m2,
                    raw_measured=r.gamma_measured_J_m2,
                    unit="J/m²",
                    omega_type="model_vs_reality",
                )
            )
        except Exception:
            results.append(_unavailable("sur", "surface_catalysis"))
    else:
        results.append(_unavailable("sur", "surface_catalysis"))

    # ── Channel 6: BCS superconductivity ─────────────────────────
    if symbol in HAS_BCS:
        try:
            from closures.materials_science.bcs_superconductivity import compute_bcs_superconductivity

            r = compute_bcs_superconductivity(0.0, 0.0, symbol=symbol)
            results.append(
                ChannelResult(
                    label="bcs",
                    full_name="bcs_superconductivity",
                    omega=r.omega_eff,
                    available=True,
                    raw_predicted=r.T_c_K,
                    raw_measured=r.T_c_measured_K,
                    unit="K",
                    omega_type="model_vs_reality",
                )
            )
        except Exception:
            results.append(_unavailable("bcs", "bcs_superconductivity"))
    else:
        results.append(_unavailable("bcs", "bcs_superconductivity"))

    # ── Channel 7: Phase transitions ─────────────────────────────
    if symbol in HAS_PHASE:
        try:
            from closures.materials_science.phase_transition import compute_phase_transition

            r = compute_phase_transition(temperature_K, 0.0, material_key=symbol)
            results.append(
                ChannelResult(
                    label="pha",
                    full_name="phase_transition",
                    omega=r.omega_eff,
                    available=True,
                    raw_predicted=temperature_K,
                    raw_measured=r.T_c_K,
                    unit="K",
                    omega_type="physical_observable",
                )
            )
        except Exception:
            results.append(_unavailable("pha", "phase_transition"))
    else:
        results.append(_unavailable("pha", "phase_transition"))

    return results


def compute_gap_capture(
    symbol: str,
    temperature_K: float = 300.0,
    epsilon: float = 1e-6,
) -> tuple[SeamReceipt, SS1M]:
    """Compute the gap-capture duality for an element.

    This is the core function. Given an element symbol, it:
    1. Collects ω from every available closure channel
    2. Builds the ω profile vector (gaps) and 1−ω profile (captures)
    3. Feeds both through the GCD kernel
    4. Computes R_IC = IC(captures) / IC(gaps) — the single measurement unit
    5. Returns a full SeamReceipt and a compact SS1M HUD

    Parameters
    ----------
    symbol : str
        Element symbol (e.g. "Fe", "Cu", "Au").
    temperature_K : float
        Temperature for temperature-dependent closures (Debye, magnetic, phase).
    epsilon : float
        ε-clamp for kernel computation.

    Returns
    -------
    (SeamReceipt, SS1M)
    """
    from src.umcp.kernel_optimized import compute_kernel_outputs

    Z = ELEMENT_Z.get(symbol, 0)

    # Look up fundamental atomic properties
    el_props = get_element(symbol)

    # Collect per-channel ω
    channels = _collect_omega_profile(symbol, Z, temperature_K)

    # Filter to available channels
    available = [ch for ch in channels if ch.available]
    n_available = len(available)

    timestamp = datetime.now(UTC).isoformat()

    if n_available == 0:
        # NON_EVALUABLE: no closure has reference data for this element.
        # Produce an honest receipt that says "0/8 channels, non-evaluable."
        hash_input = f"{symbol}:{Z}:NON_EVALUABLE"
        receipt_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:16]

        receipt = SeamReceipt(
            element=symbol,
            Z=Z,
            timestamp=timestamp,
            channels_available=0,
            channels_total=len(CHANNEL_LABELS),
            channel_labels=[],
            omega_profile=[],
            capture_profile=[],
            weights=[],
            F_cap=0.0,
            omega_cap=1.0,
            S_cap=0.0,
            C_cap=0.0,
            kappa_cap=0.0,
            IC_cap=0.0,
            F_gap=1.0,
            omega_gap=0.0,
            S_gap=0.0,
            C_gap=0.0,
            kappa_gap=0.0,
            IC_gap=0.0,
            duality_sum=1.0,
            duality_exact=True,
            S_invariant=True,
            C_invariant=True,
            R_IC=0.0,
            ln_R_IC=float("-inf"),
            gamma_cap=1.0,
            budget=0.0,
            regime_cap="NON_EVALUABLE",
            regime_gap="NON_EVALUABLE",
            seam_status="NO_DATA",
            receipt_hash=receipt_hash,
            atomic_mass=el_props.atomic_mass if el_props else None,
            standard_state=el_props.standard_state if el_props else None,
            electron_config=el_props.electron_config if el_props else None,
            oxidation_states=el_props.oxidation_states if el_props else None,
            electronegativity=el_props.electronegativity if el_props else None,
            atomic_radius_pm=el_props.atomic_radius_pm if el_props else None,
            ionization_energy_eV=el_props.ionization_energy_eV if el_props else None,
            electron_affinity_eV=el_props.electron_affinity_eV if el_props else None,
            melting_point_K=el_props.melting_point_K if el_props else None,
            boiling_point_K=el_props.boiling_point_K if el_props else None,
            density_g_cm3=el_props.density_g_cm3 if el_props else None,
        )
        stamp = SS1M(
            element=symbol,
            Z=Z,
            n_channels=0,
            channel_tags="(none)",
            F_cap=0.0,
            omega_cap=1.0,
            IC_cap=0.0,
            IC_gap=0.0,
            R_IC=0.0,
            ln_R=float("-inf"),
            S=0.0,
            C=0.0,
            gamma=1.0,
            budget=0.0,
            regime="NON_EVALUABLE",
            seam="NO_DATA",
        )
        return receipt, stamp

    # Build profile vectors (only available channels)
    omega_raw = np.array([ch.omega for ch in available])
    labels_used = [ch.label for ch in available]

    # ε-clamp: c ∈ [ε, 1-ε]  (protocol mandate: even degraded closures return)
    omega_clamped = np.clip(omega_raw, epsilon, 1.0 - epsilon)
    capture_clamped = 1.0 - omega_clamped

    # Equal weights across available channels
    w = np.ones(n_available) / n_available

    # ── Kernel: captures side ────────────────────────────────────
    r_cap = compute_kernel_outputs(capture_clamped, w, epsilon)

    # ── Kernel: gaps side ────────────────────────────────────────
    r_gap = compute_kernel_outputs(omega_clamped, w, epsilon)

    # ── Duality check ────────────────────────────────────────────
    duality_sum = r_cap["F"] + r_gap["F"]
    duality_exact = abs(duality_sum - 1.0) < 1e-9

    # ── Mirror invariance ────────────────────────────────────────
    s_inv = abs(r_cap["S"] - r_gap["S"]) < 1e-10
    c_inv = abs(r_cap["C"] - r_gap["C"]) < 1e-10

    # ── The derived measurement unit ─────────────────────────────
    R_IC = r_cap["IC"] / r_gap["IC"] if r_gap["IC"] > 0 else float("inf")
    ln_R = math.log(R_IC) if R_IC > 0 and math.isfinite(R_IC) else float("inf")

    # ── Seam budget ──────────────────────────────────────────────
    omega_cap = r_cap["omega"]
    gamma = omega_cap**3 / (1.0 - omega_cap + epsilon)
    budget = 1.0 - gamma

    # ── Regime ───────────────────────────────────────────────────
    regime_cap = _classify_regime(omega_cap, r_cap["F"], r_cap["S"], r_cap["C"])
    regime_gap = _classify_regime(r_gap["omega"], r_gap["F"], r_gap["S"], r_gap["C"])

    # ── Receipt hash ─────────────────────────────────────────────
    hash_input = f"{symbol}:{Z}:{omega_raw.tolist()}:{r_cap['F']}:{r_gap['F']}:{R_IC}"
    receipt_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    # ── Build SeamReceipt ────────────────────────────────────────
    receipt = SeamReceipt(
        element=symbol,
        Z=Z,
        timestamp=timestamp,
        channels_available=n_available,
        channels_total=len(CHANNEL_LABELS),
        channel_labels=labels_used,
        omega_profile=[round(float(x), 6) for x in omega_raw],
        capture_profile=[round(float(1.0 - x), 6) for x in omega_raw],
        weights=[round(float(x), 6) for x in w],
        F_cap=round(r_cap["F"], 6),
        omega_cap=round(float(omega_cap), 6),
        S_cap=round(r_cap["S"], 6),
        C_cap=round(r_cap["C"], 6),
        kappa_cap=round(r_cap["kappa"], 6),
        IC_cap=round(r_cap["IC"], 6),
        F_gap=round(r_gap["F"], 6),
        omega_gap=round(r_gap["omega"], 6),
        S_gap=round(r_gap["S"], 6),
        C_gap=round(r_gap["C"], 6),
        kappa_gap=round(r_gap["kappa"], 6),
        IC_gap=round(r_gap["IC"], 6),
        duality_sum=round(duality_sum, 9),
        duality_exact=duality_exact,
        S_invariant=s_inv,
        C_invariant=c_inv,
        R_IC=round(R_IC, 4),
        ln_R_IC=round(ln_R, 4),
        gamma_cap=round(gamma, 9),
        budget=round(budget, 9),
        regime_cap=regime_cap,
        regime_gap=regime_gap,
        seam_status="OPEN",
        receipt_hash=receipt_hash,
        atomic_mass=el_props.atomic_mass if el_props else None,
        standard_state=el_props.standard_state if el_props else None,
        electron_config=el_props.electron_config if el_props else None,
        oxidation_states=el_props.oxidation_states if el_props else None,
        electronegativity=el_props.electronegativity if el_props else None,
        atomic_radius_pm=el_props.atomic_radius_pm if el_props else None,
        ionization_energy_eV=el_props.ionization_energy_eV if el_props else None,
        electron_affinity_eV=el_props.electron_affinity_eV if el_props else None,
        melting_point_K=el_props.melting_point_K if el_props else None,
        boiling_point_K=el_props.boiling_point_K if el_props else None,
        density_g_cm3=el_props.density_g_cm3 if el_props else None,
    )

    # ── Build SS1M ───────────────────────────────────────────────
    stamp = SS1M(
        element=symbol,
        Z=Z,
        n_channels=n_available,
        channel_tags=" ".join(labels_used),
        F_cap=round(r_cap["F"], 4),
        omega_cap=round(float(omega_cap), 4),
        IC_cap=round(r_cap["IC"], 4),
        IC_gap=round(r_gap["IC"], 6),
        R_IC=round(R_IC, 1),
        ln_R=round(ln_R, 3),
        S=round(r_cap["S"], 4),
        C=round(r_cap["C"], 4),
        gamma=round(gamma, 4),
        budget=round(budget, 4),
        regime=regime_cap,
        seam="OPEN",
    )

    return receipt, stamp


def batch_compute(
    symbols: list[str],
    temperature_K: float = 300.0,
    epsilon: float = 1e-6,
) -> list[tuple[str, SeamReceipt | None, SS1M | None, str]]:
    """Compute gap-capture for a batch of elements.

    Returns list of (symbol, receipt, stamp, error_msg).
    On failure, receipt and stamp are None and error_msg is set.
    """
    results: list[tuple[str, SeamReceipt | None, SS1M | None, str]] = []
    for sym in symbols:
        try:
            receipt, stamp = compute_gap_capture(sym, temperature_K, epsilon)
            results.append((sym, receipt, stamp, ""))
        except Exception as e:
            results.append((sym, None, None, str(e)))
    return results


def print_ss1m_table(results: list[tuple[str, SeamReceipt | None, SS1M | None, str]]) -> None:
    """Print a summary table of SS1M results."""
    header = (
        f"{'Elem':>4} {'Z':>3} {'Ch':>2} {'F_cap':>7} {'ω_cap':>7} "
        f"{'IC_cap':>8} {'IC_gap':>9} {'R_IC':>8} {'ln(R)':>7} "
        f"{'S':>7} {'C':>7} {'Γ':>7} {'Regime':>8} {'F+F':>11}"
    )
    print(header)
    print("─" * len(header))
    for sym, receipt, stamp, err in results:
        if stamp is None:
            print(f"{sym:>4}  — {err}")
            continue
        assert receipt is not None
        print(
            f"{stamp.element:>4} {stamp.Z:>3} {stamp.n_channels:>2} "
            f"{stamp.F_cap:>7.4f} {stamp.omega_cap:>7.4f} "
            f"{stamp.IC_cap:>8.4f} {stamp.IC_gap:>9.6f} "
            f"{stamp.R_IC:>8.1f} {stamp.ln_R:>7.3f} "
            f"{stamp.S:>7.4f} {stamp.C:>7.4f} {stamp.gamma:>7.4f} "
            f"{stamp.regime:>8} {receipt.duality_sum:>11.9f}"
        )


# ── CLI self-test ────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    # Default: run the original 13 + expanded set
    if len(sys.argv) > 1:
        elements = sys.argv[1:]
    else:
        # All 118 elements, ordered by Z
        elements = [
            "H",
            "He",
            "Li",
            "Be",
            "B",
            "C",
            "N",
            "O",
            "F",
            "Ne",
            "Na",
            "Mg",
            "Al",
            "Si",
            "P",
            "S",
            "Cl",
            "Ar",
            "K",
            "Ca",
            "Sc",
            "Ti",
            "V",
            "Cr",
            "Mn",
            "Fe",
            "Co",
            "Ni",
            "Cu",
            "Zn",
            "Ga",
            "Ge",
            "As",
            "Se",
            "Br",
            "Kr",
            "Rb",
            "Sr",
            "Y",
            "Zr",
            "Nb",
            "Mo",
            "Tc",
            "Ru",
            "Rh",
            "Pd",
            "Ag",
            "Cd",
            "In",
            "Sn",
            "Sb",
            "Te",
            "I",
            "Xe",
            "Cs",
            "Ba",
            "La",
            "Ce",
            "Pr",
            "Nd",
            "Pm",
            "Sm",
            "Eu",
            "Gd",
            "Tb",
            "Dy",
            "Ho",
            "Er",
            "Tm",
            "Yb",
            "Lu",
            "Hf",
            "Ta",
            "W",
            "Re",
            "Os",
            "Ir",
            "Pt",
            "Au",
            "Hg",
            "Tl",
            "Pb",
            "Bi",
            "Po",
            "At",
            "Rn",
            "Fr",
            "Ra",
            "Ac",
            "Th",
            "Pa",
            "U",
            "Np",
            "Pu",
            "Am",
            "Cm",
            "Bk",
            "Cf",
            "Es",
            "Fm",
            "Md",
            "No",
            "Lr",
            "Rf",
            "Db",
            "Sg",
            "Bh",
            "Hs",
            "Mt",
            "Ds",
            "Rg",
            "Cn",
            "Nh",
            "Fl",
            "Mc",
            "Lv",
            "Ts",
            "Og",
        ]

    results = batch_compute(elements)

    # Print SS1M HUDs for first 3
    for _sym, _receipt, stamp, _err in results[:3]:
        if stamp is not None:
            print(stamp.hud())
            print()

    # Print summary table
    print_ss1m_table(results)

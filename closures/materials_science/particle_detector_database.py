"""Particle detector materials and radiation interaction database.

Encodes scintillator materials, superheated liquid detector (SLD) parameters,
and shielding material properties sourced from the MDPI *Particles* journal.
All data is mapped through the GCD kernel to produce Tier-1 invariants
(F, ω, S, C, κ, IC) via an 8-channel trace vector per detector material.

Trace vector channels (equal weights w_i = 1/8):
    0. density_norm — material density (g/cm³), rescaled to [0,1]
    1. Z_eff_norm — effective atomic number, rescaled to [0,1]
    2. light_yield_norm — light yield (ph/keV), log-scaled to [0,1]
    3. decay_time_norm — scintillation/nucleation decay time (ns), log-scaled
    4. energy_resolution_norm — energy resolution proxy, [0,1]
    5. radiation_hardness_norm — radiation tolerance score, [0,1]
    6. detection_efficiency_norm — detection efficiency or transmission, [0,1]
    7. spectral_match_norm — spectral match to SiPM/PMT readout, [0,1]

Sources (see paper/Bibliography.bib):
    - Rodrigues, A.F.; Felizardo, M. (2026). "Detection of Shielded Nuclear
      Materials Using Superheated Liquid Detectors."
      Particles 9(1):20. DOI: 10.3390/particles9010020
      [rodrigues2026sld]
    - Dimiccoli, F.; Ferro, F.; Zanet, D.; Beri, D.; Fenoglio, E. (2025).
      "A New Measurement of Light Yield Quenching Factors in EJ-200 and LYSO:Ce
      Scintillators." Particles 8(4):82.
      DOI: 10.3390/particles8040082  [dimiccoli2025quenching]
    - Birks, J.B. (1964). "The Theory and Practice of Scintillation Counting."
      Pergamon Press. [birks1964scintillation]
    - Particle Data Group (2024) — detector material properties
      [pdg2024]
    - Eljen Technology — EJ-200 datasheet (public domain specifications)
    - Saint-Gobain Crystals — LYSO:Ce datasheet (public domain specifications)

Copyright-free reference data: material densities, atomic numbers, scintillation
yields, and timing constants are factual scientific measurements not subject
to copyright.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, NamedTuple

# ═══════════════════════════════════════════════════════════════════
#  Enumerations
# ═══════════════════════════════════════════════════════════════════


class DetectorType(StrEnum):
    """Classification of particle detector technology."""

    PLASTIC_SCINTILLATOR = "Plastic scintillator"
    INORGANIC_SCINTILLATOR = "Inorganic scintillator"
    SUPERHEATED_LIQUID = "Superheated liquid detector"
    GAS_DETECTOR = "Gas detector"
    SEMICONDUCTOR = "Semiconductor detector"
    CHERENKOV = "Cherenkov detector"
    CALORIMETER = "Calorimeter"


class RadiationType(StrEnum):
    """Primary radiation type the detector is sensitive to."""

    NEUTRON = "Neutron"
    GAMMA = "Gamma"
    CHARGED_PARTICLE = "Charged particle"
    MUON = "Muon"
    MIXED = "Mixed"
    DARK_MATTER = "Dark matter"


class QuenchingModel(StrEnum):
    """Light yield quenching model used for parameterization."""

    BIRKS = "Birks"
    BIRKS_ONSAGER = "Birks-Onsager"
    GENERALIZED_LOGISTIC = "Generalized logistic"
    NONE = "None"


class ShieldingCategory(StrEnum):
    """Shielding material category for nuclear material detection."""

    HYDROGENOUS = "Hydrogenous"  # Paraffin, polyethylene, water
    METALLIC = "Metallic"  # Lead, aluminum, steel
    ORGANIC = "Organic"  # Wood, plastic
    COMPOSITE = "Composite"  # Multi-layer / graded


# ═══════════════════════════════════════════════════════════════════
#  Data Classes — Scintillator Materials
# ═══════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class ScintillatorMaterial:
    """Scintillator material with optical and radiation response properties.

    Sources:
        EJ-200 properties: Dimiccoli et al. (2025) [dimiccoli2025quenching]
            Table 1, Eqs. 7-10, Fig. 5-6
        LYSO:Ce properties: Dimiccoli et al. (2025) [dimiccoli2025quenching]
            Table 2, Eqs. 11-13, Fig. 7-8
        Birks model: Birks (1964) [birks1964scintillation]
        General properties: PDG 2024 [pdg2024], manufacturer datasheets
    """

    name: str
    chemical_formula: str
    detector_type: DetectorType
    density_g_cm3: float
    Z_eff: float | None  # Effective atomic number
    light_yield_ph_per_keV: float  # Photons per keV deposited
    peak_emission_nm: float  # Peak emission wavelength (nm)

    # ── Timing properties ────────────────────────────────────────
    rise_time_ns: float | None  # Scintillation rise time (ns)
    decay_time_ns: float  # Primary decay constant (ns)
    attenuation_length_m: float | None  # Optical attenuation length (m)

    # ── Quenching parameters ─────────────────────────────────────
    quenching_model: QuenchingModel
    birks_inv_MeV_cm: float | None  # 1/kB (MeV/cm) from Birks fit
    birks_inv_err: float | None  # Uncertainty on 1/kB
    eta_e_h: float | None  # Birks-Onsager: geminate escape ratio
    eta_H: float | None  # Birks-Onsager: hadronic quenching
    dEdx_0_MeV_cm: float | None  # Onsager critical dE/dx (MeV/cm)

    # ── Logistic quenching parameterization ──────────────────────
    logistic_K_MeV_cm: float | None  # Logistic saturation (MeV/cm)
    logistic_alpha: float | None  # Logistic power exponent
    logistic_L_inf: float | None  # Logistic asymptotic fraction

    # ── Physical properties ──────────────────────────────────────
    radiation_sensitivity: RadiationType = RadiationType.MIXED
    radiation_hardness_Mrad: float | None = None  # Rad hardness threshold

    # ── Metadata ─────────────────────────────────────────────────
    manufacturer: str = ""
    source_reference: str = ""  # BibTeX key

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON output."""
        return {
            "name": self.name,
            "chemical_formula": self.chemical_formula,
            "detector_type": self.detector_type.value,
            "density_g_cm3": self.density_g_cm3,
            "Z_eff": self.Z_eff,
            "light_yield_ph_per_keV": self.light_yield_ph_per_keV,
            "peak_emission_nm": self.peak_emission_nm,
            "rise_time_ns": self.rise_time_ns,
            "decay_time_ns": self.decay_time_ns,
            "attenuation_length_m": self.attenuation_length_m,
            "quenching_model": self.quenching_model.value,
            "birks_inv_MeV_cm": self.birks_inv_MeV_cm,
            "birks_inv_err": self.birks_inv_err,
            "eta_e_h": self.eta_e_h,
            "eta_H": self.eta_H,
            "dEdx_0_MeV_cm": self.dEdx_0_MeV_cm,
            "logistic_K_MeV_cm": self.logistic_K_MeV_cm,
            "logistic_alpha": self.logistic_alpha,
            "logistic_L_inf": self.logistic_L_inf,
            "radiation_sensitivity": self.radiation_sensitivity.value,
            "radiation_hardness_Mrad": self.radiation_hardness_Mrad,
            "manufacturer": self.manufacturer,
            "source_reference": self.source_reference,
        }


# ═══════════════════════════════════════════════════════════════════
#  Data Classes — Superheated Liquid Detectors
# ═══════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class SuperheatedDetector:
    """Superheated liquid (bubble) detector properties.

    Superheated droplet detectors (SDDs) exploit the nucleation of
    superheated liquid droplets by incident radiation.  The phase
    transition from metastable liquid to gas is triggered by ionizing
    particles above a threshold LET, making them inherently insensitive
    to minimum-ionizing particles (gammas, muons).

    Sources:
        Rodrigues & Felizardo (2026) [rodrigues2026sld]
            Table 1: C3F8 properties, fabrication, acoustic readout
            Table 2: Shielding transmission factors
            Figs. 3-6: Distance dependence, shielding effectiveness
    """

    name: str
    active_liquid: str  # Active superheated liquid name
    chemical_formula: str  # Chemical formula of active liquid
    molecular_weight: float  # g/mol
    boiling_point_K: float  # Normal boiling point (K)
    density_g_cm3: float  # Liquid density at operating temp

    # ── Droplet/gel properties ───────────────────────────────────
    droplet_diameter_um: float  # Mean droplet diameter (μm)
    gel_matrix: str  # Gel composition
    droplet_loading_pct: float  # Volume fraction of active liquid (%)

    # ── Nucleation/acoustic properties ───────────────────────────
    nucleation_freq_min_Hz: float  # Min nucleation acoustic frequency (Hz)
    nucleation_freq_max_Hz: float  # Max nucleation acoustic frequency (Hz)
    nucleation_decay_min_ms: float  # Min nucleation decay time (ms)
    nucleation_decay_max_ms: float  # Max nucleation decay time (ms)
    acoustic_sensitivity_dB: tuple[float, float] = (-70.0, 20.0)  # dB range

    # ── Radiation sensitivity ────────────────────────────────────
    neutron_sensitive: bool = True
    alpha_sensitive: bool = True
    gamma_sensitive: bool = False  # Key property: gamma-blind
    min_recoil_energy_keV: float = 0.0  # Threshold recoil energy

    # ── Source and reference ─────────────────────────────────────
    source_reference: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON output."""
        return {
            "name": self.name,
            "active_liquid": self.active_liquid,
            "chemical_formula": self.chemical_formula,
            "molecular_weight": self.molecular_weight,
            "boiling_point_K": self.boiling_point_K,
            "density_g_cm3": self.density_g_cm3,
            "droplet_diameter_um": self.droplet_diameter_um,
            "gel_matrix": self.gel_matrix,
            "droplet_loading_pct": self.droplet_loading_pct,
            "nucleation_freq_min_Hz": self.nucleation_freq_min_Hz,
            "nucleation_freq_max_Hz": self.nucleation_freq_max_Hz,
            "nucleation_decay_min_ms": self.nucleation_decay_min_ms,
            "nucleation_decay_max_ms": self.nucleation_decay_max_ms,
            "acoustic_sensitivity_dB": list(self.acoustic_sensitivity_dB),
            "neutron_sensitive": self.neutron_sensitive,
            "alpha_sensitive": self.alpha_sensitive,
            "gamma_sensitive": self.gamma_sensitive,
            "min_recoil_energy_keV": self.min_recoil_energy_keV,
            "source_reference": self.source_reference,
        }


# ═══════════════════════════════════════════════════════════════════
#  Data Classes — Shielding Materials
# ═══════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class ShieldingMaterial:
    """Neutron shielding material with measured transmission factors.

    Sources:
        Rodrigues & Felizardo (2026) [rodrigues2026sld]
            Table 2: Transmission factors for 5 shielding materials
            with AmBe and Cf-252 neutron sources at multiple distances
    """

    name: str
    category: ShieldingCategory
    density_g_cm3: float
    thickness_cm: float  # Thickness tested
    neutron_transmission: float  # Measured transmission factor [0,1]
    neutron_transmission_err: float | None  # Uncertainty
    composition_notes: str = ""
    source_reference: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON output."""
        return {
            "name": self.name,
            "category": self.category.value,
            "density_g_cm3": self.density_g_cm3,
            "thickness_cm": self.thickness_cm,
            "neutron_transmission": self.neutron_transmission,
            "neutron_transmission_err": self.neutron_transmission_err,
            "composition_notes": self.composition_notes,
            "source_reference": self.source_reference,
        }


class DetectorKernelResult(NamedTuple):
    """GCD kernel output for a detector material."""

    name: str
    detector_type: str
    trace: tuple[float, ...]
    F: float  # Fidelity
    omega: float  # Drift
    S: float  # Bernoulli field entropy
    C: float  # Curvature
    kappa: float  # Log-integrity
    IC: float  # Integrity composite
    regime: str  # Stable / Watch / Collapse


# ═══════════════════════════════════════════════════════════════════
#  SCINTILLATOR MATERIAL DATABASE
# ═══════════════════════════════════════════════════════════════════
# Primary sources:
#   Dimiccoli et al. (2025) [dimiccoli2025quenching]
#   PDG 2024 [pdg2024]
#   Manufacturer datasheets (Eljen Technology, Saint-Gobain)
# ═══════════════════════════════════════════════════════════════════

SCINTILLATORS: tuple[ScintillatorMaterial, ...] = (
    # ── Plastic scintillators ────────────────────────────────────
    # EJ-200: Dimiccoli et al. (2025) Table 1, Eqs. 7-10
    ScintillatorMaterial(
        name="EJ-200",
        chemical_formula="C9H10 (PVT base)",
        detector_type=DetectorType.PLASTIC_SCINTILLATOR,
        density_g_cm3=1.023,
        Z_eff=3.4,
        light_yield_ph_per_keV=10.0,
        peak_emission_nm=425.0,
        rise_time_ns=0.9,
        decay_time_ns=2.1,
        attenuation_length_m=3.8,
        quenching_model=QuenchingModel.BIRKS_ONSAGER,
        birks_inv_MeV_cm=12.0,
        birks_inv_err=1.3,
        eta_e_h=0.853,
        eta_H=0.0,  # Fixed to 0 in fit
        dEdx_0_MeV_cm=134.0,
        logistic_K_MeV_cm=65.0,
        logistic_alpha=0.75,
        logistic_L_inf=0.0,  # Fixed to 0 in fit
        radiation_sensitivity=RadiationType.CHARGED_PARTICLE,
        radiation_hardness_Mrad=10.0,
        manufacturer="Eljen Technology",
        source_reference="dimiccoli2025quenching",
    ),
    # EJ-208: Similar PVT, longer attenuation for large volumes
    ScintillatorMaterial(
        name="EJ-208",
        chemical_formula="C9H10 (PVT base)",
        detector_type=DetectorType.PLASTIC_SCINTILLATOR,
        density_g_cm3=1.023,
        Z_eff=3.4,
        light_yield_ph_per_keV=9.2,
        peak_emission_nm=435.0,
        rise_time_ns=0.9,
        decay_time_ns=3.3,
        attenuation_length_m=4.0,
        quenching_model=QuenchingModel.BIRKS,
        birks_inv_MeV_cm=10.8,
        birks_inv_err=1.5,
        eta_e_h=None,
        eta_H=None,
        dEdx_0_MeV_cm=None,
        logistic_K_MeV_cm=None,
        logistic_alpha=None,
        logistic_L_inf=None,
        radiation_sensitivity=RadiationType.CHARGED_PARTICLE,
        radiation_hardness_Mrad=10.0,
        manufacturer="Eljen Technology",
        source_reference="pdg2024",
    ),
    # BC-408: Equivalent to EJ-200 (Saint-Gobain)
    ScintillatorMaterial(
        name="BC-408",
        chemical_formula="C9H10 (PVT base)",
        detector_type=DetectorType.PLASTIC_SCINTILLATOR,
        density_g_cm3=1.032,
        Z_eff=3.4,
        light_yield_ph_per_keV=10.0,
        peak_emission_nm=425.0,
        rise_time_ns=0.9,
        decay_time_ns=2.1,
        attenuation_length_m=3.8,
        quenching_model=QuenchingModel.BIRKS,
        birks_inv_MeV_cm=12.0,
        birks_inv_err=1.5,
        eta_e_h=None,
        eta_H=None,
        dEdx_0_MeV_cm=None,
        logistic_K_MeV_cm=None,
        logistic_alpha=None,
        logistic_L_inf=None,
        radiation_sensitivity=RadiationType.CHARGED_PARTICLE,
        radiation_hardness_Mrad=10.0,
        manufacturer="Saint-Gobain Crystals",
        source_reference="pdg2024",
    ),
    # ── Inorganic scintillators ──────────────────────────────────
    # LYSO:Ce — Dimiccoli et al. (2025) Table 2, Eqs. 11-13
    ScintillatorMaterial(
        name="LYSO:Ce",
        chemical_formula="Lu1.8Y0.2SiO5:Ce",
        detector_type=DetectorType.INORGANIC_SCINTILLATOR,
        density_g_cm3=7.1,
        Z_eff=65.0,
        light_yield_ph_per_keV=30.0,
        peak_emission_nm=420.0,
        rise_time_ns=None,
        decay_time_ns=40.0,
        attenuation_length_m=None,
        quenching_model=QuenchingModel.BIRKS_ONSAGER,
        birks_inv_MeV_cm=361.0,
        birks_inv_err=16.0,
        eta_e_h=0.0,  # Fixed to 0 in fit
        eta_H=0.053,
        dEdx_0_MeV_cm=None,
        logistic_K_MeV_cm=380.0,
        logistic_alpha=1.00,
        logistic_L_inf=0.050,
        radiation_sensitivity=RadiationType.GAMMA,
        radiation_hardness_Mrad=1.0,
        manufacturer="Saint-Gobain Crystals",
        source_reference="dimiccoli2025quenching",
    ),
    # NaI:Tl — Classic gamma spectrometer
    ScintillatorMaterial(
        name="NaI:Tl",
        chemical_formula="NaI:Tl",
        detector_type=DetectorType.INORGANIC_SCINTILLATOR,
        density_g_cm3=3.67,
        Z_eff=51.0,
        light_yield_ph_per_keV=38.0,
        peak_emission_nm=415.0,
        rise_time_ns=None,
        decay_time_ns=250.0,
        attenuation_length_m=None,
        quenching_model=QuenchingModel.BIRKS,
        birks_inv_MeV_cm=None,
        birks_inv_err=None,
        eta_e_h=None,
        eta_H=None,
        dEdx_0_MeV_cm=None,
        logistic_K_MeV_cm=None,
        logistic_alpha=None,
        logistic_L_inf=None,
        radiation_sensitivity=RadiationType.GAMMA,
        radiation_hardness_Mrad=0.01,
        manufacturer="Saint-Gobain Crystals",
        source_reference="pdg2024",
    ),
    # CsI:Tl — Compact calorimetry
    ScintillatorMaterial(
        name="CsI:Tl",
        chemical_formula="CsI:Tl",
        detector_type=DetectorType.INORGANIC_SCINTILLATOR,
        density_g_cm3=4.51,
        Z_eff=54.0,
        light_yield_ph_per_keV=54.0,
        peak_emission_nm=540.0,
        rise_time_ns=None,
        decay_time_ns=1000.0,
        attenuation_length_m=None,
        quenching_model=QuenchingModel.BIRKS,
        birks_inv_MeV_cm=None,
        birks_inv_err=None,
        eta_e_h=None,
        eta_H=None,
        dEdx_0_MeV_cm=None,
        logistic_K_MeV_cm=None,
        logistic_alpha=None,
        logistic_L_inf=None,
        radiation_sensitivity=RadiationType.CHARGED_PARTICLE,
        radiation_hardness_Mrad=0.1,
        manufacturer="Various",
        source_reference="pdg2024",
    ),
    # BGO — Bismuth germanate (high density, no self-activity)
    ScintillatorMaterial(
        name="BGO",
        chemical_formula="Bi4Ge3O12",
        detector_type=DetectorType.INORGANIC_SCINTILLATOR,
        density_g_cm3=7.13,
        Z_eff=74.0,
        light_yield_ph_per_keV=8.0,
        peak_emission_nm=480.0,
        rise_time_ns=None,
        decay_time_ns=300.0,
        attenuation_length_m=None,
        quenching_model=QuenchingModel.NONE,
        birks_inv_MeV_cm=None,
        birks_inv_err=None,
        eta_e_h=None,
        eta_H=None,
        dEdx_0_MeV_cm=None,
        logistic_K_MeV_cm=None,
        logistic_alpha=None,
        logistic_L_inf=None,
        radiation_sensitivity=RadiationType.GAMMA,
        radiation_hardness_Mrad=1.0,
        manufacturer="Various",
        source_reference="pdg2024",
    ),
    # PWO — Lead tungstate (CMS ECAL)
    ScintillatorMaterial(
        name="PbWO4 (PWO)",
        chemical_formula="PbWO4",
        detector_type=DetectorType.INORGANIC_SCINTILLATOR,
        density_g_cm3=8.28,
        Z_eff=75.6,
        light_yield_ph_per_keV=0.3,  # Very low — fast timing compensates
        peak_emission_nm=420.0,
        rise_time_ns=None,
        decay_time_ns=10.0,
        attenuation_length_m=None,
        quenching_model=QuenchingModel.NONE,
        birks_inv_MeV_cm=None,
        birks_inv_err=None,
        eta_e_h=None,
        eta_H=None,
        dEdx_0_MeV_cm=None,
        logistic_K_MeV_cm=None,
        logistic_alpha=None,
        logistic_L_inf=None,
        radiation_sensitivity=RadiationType.GAMMA,
        radiation_hardness_Mrad=100.0,
        manufacturer="Various (Bogoroditsk)",
        source_reference="pdg2024",
    ),
)


# ═══════════════════════════════════════════════════════════════════
#  SUPERHEATED LIQUID DETECTOR DATABASE
# ═══════════════════════════════════════════════════════════════════
# Source: Rodrigues & Felizardo (2026) [rodrigues2026sld]
#   C3F8 SDD fabrication, acoustic readout, optical (YOLOv5)
# ═══════════════════════════════════════════════════════════════════

SUPERHEATED_DETECTORS: tuple[SuperheatedDetector, ...] = (
    SuperheatedDetector(
        name="SLD-C3F8 (standard)",
        active_liquid="Octafluoropropane",
        chemical_formula="C3F8",
        molecular_weight=188.02,
        boiling_point_K=236.6,  # −36.7°C
        density_g_cm3=1.601,  # Liquid at −36.7°C
        droplet_diameter_um=20.4,
        gel_matrix="Gelatin + PVP (polyvinylpyrrolidone)",
        droplet_loading_pct=1.0,  # ~1% by volume
        nucleation_freq_min_Hz=450.0,
        nucleation_freq_max_Hz=750.0,
        nucleation_decay_min_ms=5.0,
        nucleation_decay_max_ms=40.0,
        acoustic_sensitivity_dB=(-70.0, 20.0),
        neutron_sensitive=True,
        alpha_sensitive=True,
        gamma_sensitive=False,  # Confirmed: Cs-137 test → 0 nucleations
        min_recoil_energy_keV=8.0,  # Typical threshold for neutron recoils
        source_reference="rodrigues2026sld",
    ),
)


# ═══════════════════════════════════════════════════════════════════
#  SHIELDING MATERIAL DATABASE
# ═══════════════════════════════════════════════════════════════════
# Source: Rodrigues & Felizardo (2026) [rodrigues2026sld]
#   Table 2: Measured neutron transmission factors
#   AmBe source (0.1 mCi, ~2.2×10⁶ n/s), Cf-252 (0.1 mCi)
# ═══════════════════════════════════════════════════════════════════

SHIELDING_MATERIALS: tuple[ShieldingMaterial, ...] = (
    ShieldingMaterial(
        name="Aluminum",
        category=ShieldingCategory.METALLIC,
        density_g_cm3=2.70,
        thickness_cm=5.0,
        neutron_transmission=0.9998,
        neutron_transmission_err=0.001,
        composition_notes="Pure Al; nearly transparent to fast neutrons",
        source_reference="rodrigues2026sld",
    ),
    ShieldingMaterial(
        name="Wood",
        category=ShieldingCategory.ORGANIC,
        density_g_cm3=0.55,
        thickness_cm=10.0,
        neutron_transmission=0.74,
        neutron_transmission_err=0.02,
        composition_notes="Cellulose (~6.2% H); moderate neutron thermalizer",
        source_reference="rodrigues2026sld",
    ),
    ShieldingMaterial(
        name="Plastic (generic)",
        category=ShieldingCategory.ORGANIC,
        density_g_cm3=1.10,
        thickness_cm=5.0,
        neutron_transmission=0.9048,
        neutron_transmission_err=0.015,
        composition_notes="Generic hydrocarbon plastic; moderate H content",
        source_reference="rodrigues2026sld",
    ),
    ShieldingMaterial(
        name="Polyethylene (HDPE)",
        category=ShieldingCategory.HYDROGENOUS,
        density_g_cm3=0.95,
        thickness_cm=10.0,
        neutron_transmission=0.8187,
        neutron_transmission_err=0.01,
        composition_notes="(C2H4)n; high H/C ratio, effective neutron moderator",
        source_reference="rodrigues2026sld",
    ),
    ShieldingMaterial(
        name="Paraffin wax",
        category=ShieldingCategory.HYDROGENOUS,
        density_g_cm3=0.90,
        thickness_cm=10.0,
        neutron_transmission=0.0498,
        neutron_transmission_err=0.005,
        composition_notes="CnH2n+2 (n~25); highest H density → best neutron shield",
        source_reference="rodrigues2026sld",
    ),
)


# ═══════════════════════════════════════════════════════════════════
#  Lookup Utilities
# ═══════════════════════════════════════════════════════════════════

_SCINT_BY_NAME: dict[str, ScintillatorMaterial] = {s.name: s for s in SCINTILLATORS}
_SHIELD_BY_NAME: dict[str, ShieldingMaterial] = {s.name: s for s in SHIELDING_MATERIALS}


def get_scintillator(name: str) -> ScintillatorMaterial | None:
    """Look up a scintillator material by exact name."""
    return _SCINT_BY_NAME.get(name)


def get_scintillators_by_type(
    dtype: DetectorType,
) -> list[ScintillatorMaterial]:
    """Filter scintillators by detector type."""
    return [s for s in SCINTILLATORS if s.detector_type == dtype]


def get_shielding_material(name: str) -> ShieldingMaterial | None:
    """Look up a shielding material by name."""
    return _SHIELD_BY_NAME.get(name)


def get_shielding_by_category(
    category: ShieldingCategory,
) -> list[ShieldingMaterial]:
    """Filter shielding materials by category."""
    return [s for s in SHIELDING_MATERIALS if s.category == category]


# ═══════════════════════════════════════════════════════════════════
#  Normalization Constants (for trace vector construction)
# ═══════════════════════════════════════════════════════════════════

# Density: [0.5, 10.0] → [0, 1]
_DENSITY_MIN = 0.5
_DENSITY_MAX = 10.0

# Z_eff: [3, 80] → [0, 1]
_ZEFF_MIN = 3.0
_ZEFF_MAX = 80.0

# Light yield: log [0.1, 100] ph/keV → [0, 1]
_LY_MIN_LOG = math.log(0.1)
_LY_MAX_LOG = math.log(100.0)

# Decay time: log [1, 5000] ns → [0, 1]
_DT_MIN_LOG = math.log(1.0)
_DT_MAX_LOG = math.log(5000.0)

# Guard band
_EPSILON = 1e-8


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp value to [lo, hi]."""
    return max(lo, min(hi, x))


def _log_normalize(value: float, min_log: float, max_log: float) -> float:
    """Log-scale normalize value to [0, 1], clamped."""
    if value <= 0:
        return _EPSILON
    lv = math.log(value)
    return _clamp((lv - min_log) / (max_log - min_log), _EPSILON, 1.0 - _EPSILON)


def _linear_normalize(value: float, vmin: float, vmax: float) -> float:
    """Linear normalize value to [0, 1], clamped."""
    return _clamp((value - vmin) / (vmax - vmin), _EPSILON, 1.0 - _EPSILON)


# ═══════════════════════════════════════════════════════════════════
#  Trace Vector & Kernel Computation — Scintillators
# ═══════════════════════════════════════════════════════════════════


def build_scintillator_trace(mat: ScintillatorMaterial) -> tuple[float, ...]:
    """Build 8-channel trace vector for a scintillator material.

    Channels:
        0. density (linear)
        1. Z_eff (linear)
        2. light_yield (log-normalized)
        3. decay_time (log-normalized, inverted: faster = higher)
        4. energy_resolution proxy = light_yield / sqrt(decay_time)
        5. radiation_hardness (log-normalized)
        6. detection_efficiency proxy from Z_eff and density
        7. spectral_match: peak_emission closeness to 420 nm (SiPM peak)
    """
    c0 = _linear_normalize(mat.density_g_cm3, _DENSITY_MIN, _DENSITY_MAX)
    c1 = _linear_normalize(mat.Z_eff or 10.0, _ZEFF_MIN, _ZEFF_MAX)
    c2 = _log_normalize(mat.light_yield_ph_per_keV, _LY_MIN_LOG, _LY_MAX_LOG)

    # Faster decay → higher fidelity (inverted)
    c3 = 1.0 - _log_normalize(mat.decay_time_ns, _DT_MIN_LOG, _DT_MAX_LOG)

    # Energy resolution proxy: LY / sqrt(τ) normalized
    er_raw = mat.light_yield_ph_per_keV / math.sqrt(max(mat.decay_time_ns, 1.0))
    c4 = _clamp(er_raw / 10.0, _EPSILON, 1.0 - _EPSILON)

    # Radiation hardness (log-scale)
    rh = mat.radiation_hardness_Mrad or 0.01
    c5 = _log_normalize(rh, math.log(0.001), math.log(1000.0))

    # Detection efficiency proxy: density × Z_eff (for stopping power)
    de_raw = mat.density_g_cm3 * (mat.Z_eff or 10.0)
    c6 = _clamp(de_raw / 700.0, _EPSILON, 1.0 - _EPSILON)  # PWO ~625

    # Spectral match to 420nm SiPM peak
    spectral_dist = abs(mat.peak_emission_nm - 420.0)
    c7 = _clamp(1.0 - spectral_dist / 200.0, _EPSILON, 1.0 - _EPSILON)

    return (c0, c1, c2, c3, c4, c5, c6, c7)


def compute_scintillator_kernel(
    mat: ScintillatorMaterial,
) -> DetectorKernelResult:
    """Compute GCD kernel invariants for a scintillator material.

    Maps material properties through an 8-channel trace vector,
    then derives Tier-1 invariants:
        F = Σ wᵢ·cᵢ        (fidelity)
        ω = 1 − F           (drift; duality identity)
        S = −Σ wᵢ[cᵢ ln cᵢ + (1−cᵢ) ln(1−cᵢ)]  (Bernoulli field entropy)
        C = stddev(cᵢ) / 0.5  (curvature)
        κ = Σ wᵢ ln(cᵢ)     (log-integrity)
        IC = exp(κ)          (integrity composite)
    """
    trace = build_scintillator_trace(mat)
    n = len(trace)
    w = 1.0 / n

    F = sum(w * c for c in trace)
    omega = 1.0 - F

    S = 0.0
    for c in trace:
        ce = _clamp(c, _EPSILON, 1.0 - _EPSILON)
        S -= w * (ce * math.log(ce) + (1.0 - ce) * math.log(1.0 - ce))

    mean_c = F
    variance = sum(w * (c - mean_c) ** 2 for c in trace)
    C = _clamp(math.sqrt(variance) / 0.5, 0.0, 1.0)

    kappa = sum(w * math.log(_clamp(c, _EPSILON, 1.0)) for c in trace)
    IC = math.exp(kappa)

    if omega >= 0.30:
        regime = "Collapse"
    elif omega < 0.038 and F > 0.90 and S < 0.15 and C < 0.14:
        regime = "Stable"
    else:
        regime = "Watch"

    return DetectorKernelResult(
        name=mat.name,
        detector_type=mat.detector_type.value,
        trace=trace,
        F=F,
        omega=omega,
        S=S,
        C=C,
        kappa=kappa,
        IC=IC,
        regime=regime,
    )


def compute_all_scintillator_kernels() -> list[DetectorKernelResult]:
    """Compute kernel invariants for all scintillators in database."""
    return [compute_scintillator_kernel(s) for s in SCINTILLATORS]


# ═══════════════════════════════════════════════════════════════════
#  Quenching Physics
# ═══════════════════════════════════════════════════════════════════


def birks_quenching_factor(
    dEdx: float,
    kB_inv: float,
    *,
    eta_e_h: float = 1.0,
    eta_H: float = 0.0,
    dEdx_0: float | None = None,
) -> float:
    """Compute Birks-Onsager quenching factor Q(dE/dx).

    Standard Birks:
        Q = 1 / (1 + kB · dE/dx)

    Birks-Onsager (from Dimiccoli et al. 2025):
        Q = η_{e/h} / (1 + kB · dE/dx) + η_H · exp(-dE/dx / (dE/dx)_0)

    Parameters
    ----------
    dEdx : float
        Stopping power in MeV/cm.
    kB_inv : float
        Inverse Birks constant 1/kB in MeV/cm.
    eta_e_h : float
        Electron/hadron efficiency ratio (Birks-Onsager).
    eta_H : float
        Hadronic quenching contribution.
    dEdx_0 : float | None
        Onsager critical dE/dx. If None, uses standard Birks.

    Returns
    -------
    float
        Quenching factor Q ∈ [0, 1].

    Source: Dimiccoli et al. (2025) Eq. 5 [dimiccoli2025quenching]
    """
    if kB_inv <= 0:
        return 1.0

    kB = 1.0 / kB_inv
    birks_term = eta_e_h / (1.0 + kB * dEdx)

    onsager_term = eta_H * math.exp(-dEdx / dEdx_0) if dEdx_0 is not None and dEdx_0 > 0 else 0.0

    return _clamp(birks_term + onsager_term, 0.0, 1.0)


def logistic_quenching_factor(
    dEdx: float,
    K: float,
    alpha: float,
    L_inf: float = 0.0,
) -> float:
    """Compute generalized logistic quenching factor.

    Q = (1 − L∞) / (1 + (dE/dx / K)^α) + L∞

    Source: Dimiccoli et al. (2025) Eq. 6 [dimiccoli2025quenching]

    Parameters
    ----------
    dEdx : float
        Stopping power in MeV/cm.
    K : float
        Logistic saturation parameter (MeV/cm).
    alpha : float
        Power exponent controlling steepness.
    L_inf : float
        Asymptotic quenching fraction at high dE/dx.
    """
    if K <= 0:
        return 1.0
    ratio = (dEdx / K) ** alpha
    return (1.0 - L_inf) / (1.0 + ratio) + L_inf


# ═══════════════════════════════════════════════════════════════════
#  Shielding Analysis
# ═══════════════════════════════════════════════════════════════════


def shielding_effectiveness(material: ShieldingMaterial) -> float:
    """Compute shielding effectiveness (1 − transmission).

    Returns the fraction of neutrons stopped by the material.
    Source: Rodrigues & Felizardo (2026) Table 2 [rodrigues2026sld]
    """
    return 1.0 - material.neutron_transmission


def rank_shielding_materials() -> list[tuple[str, float, float]]:
    """Rank shielding materials by effectiveness (descending).

    Returns list of (name, effectiveness, transmission).
    """
    ranked = [(s.name, shielding_effectiveness(s), s.neutron_transmission) for s in SHIELDING_MATERIALS]
    return sorted(ranked, key=lambda x: x[1], reverse=True)


# ═══════════════════════════════════════════════════════════════════
#  Detector Comparison & Analysis
# ═══════════════════════════════════════════════════════════════════


def compare_scintillators(
    names: list[str] | None = None,
) -> dict[str, Any]:
    """Compare scintillator materials by kernel invariants.

    If names is None, compares all scintillators.
    Highlights which material has highest fidelity (F) and
    smallest heterogeneity gap (Δ = F − IC).
    """
    materials = [s for s in SCINTILLATORS if s.name in names] if names else list(SCINTILLATORS)

    results = [(m.name, compute_scintillator_kernel(m)) for m in materials]

    comparison = {
        "materials_compared": len(results),
        "entries": [
            {
                "name": name,
                "detector_type": kr.detector_type,
                "F": kr.F,
                "omega": kr.omega,
                "IC": kr.IC,
                "heterogeneity_gap": kr.F - kr.IC,
                "S": kr.S,
                "C": kr.C,
                "regime": kr.regime,
            }
            for name, kr in results
        ],
    }

    if results:
        comparison["highest_F"] = max(results, key=lambda x: x[1].F)[0]
        comparison["smallest_gap"] = min(results, key=lambda x: x[1].F - x[1].IC)[0]

    return comparison


# ═══════════════════════════════════════════════════════════════════
#  Validation
# ═══════════════════════════════════════════════════════════════════


def validate_database() -> dict[str, Any]:
    """Validate the particle detector database.

    Checks:
        1. All scintillators have positive density and light yield
        2. Kernel identity F + ω = 1 holds for all traces
        3. Integrity bound IC ≤ F holds for all traces
        4. Shielding transmission factors ∈ [0, 1]
        5. No NaN or infinite values in kernel outputs
    """
    errors: list[str] = []
    warnings: list[str] = []

    # ── Scintillator checks ──────────────────────────────────────
    for mat in SCINTILLATORS:
        if mat.density_g_cm3 <= 0:
            errors.append(f"{mat.name}: density must be > 0")
        if mat.light_yield_ph_per_keV <= 0:
            errors.append(f"{mat.name}: light_yield must be > 0")
        if mat.decay_time_ns <= 0:
            errors.append(f"{mat.name}: decay_time must be > 0")

        kr = compute_scintillator_kernel(mat)
        duality_residual = abs(kr.F + kr.omega - 1.0)
        if duality_residual > 1e-12:
            errors.append(f"{mat.name}: duality identity violated, |F + ω − 1| = {duality_residual:.2e}")
        if kr.IC > kr.F + 1e-12:
            errors.append(f"{mat.name}: integrity bound violated, IC ({kr.IC:.6f}) > F ({kr.F:.6f})")
        if math.isnan(kr.F) or math.isinf(kr.F):
            errors.append(f"{mat.name}: F is NaN/Inf")

        # Warn if quenching parameters are missing
        if mat.birks_inv_MeV_cm is None and mat.quenching_model != QuenchingModel.NONE:
            warnings.append(f"{mat.name}: quenching model set but no Birks parameter")

    # ── Shielding checks ─────────────────────────────────────────
    for shield in SHIELDING_MATERIALS:
        if not 0.0 <= shield.neutron_transmission <= 1.0:
            errors.append(f"{shield.name}: transmission {shield.neutron_transmission} out of [0, 1]")
        if shield.density_g_cm3 <= 0:
            errors.append(f"{shield.name}: density must be > 0")

    # ── Superheated detector checks ──────────────────────────────
    for det in SUPERHEATED_DETECTORS:
        if det.density_g_cm3 <= 0:
            errors.append(f"{det.name}: density must be > 0")
        if det.droplet_diameter_um <= 0:
            errors.append(f"{det.name}: droplet_diameter must be > 0")
        if det.nucleation_freq_min_Hz >= det.nucleation_freq_max_Hz:
            errors.append(
                f"{det.name}: nucleation freq range invalid "
                f"({det.nucleation_freq_min_Hz} >= {det.nucleation_freq_max_Hz})"
            )

    return {
        "total_scintillators": len(SCINTILLATORS),
        "total_shielding": len(SHIELDING_MATERIALS),
        "total_superheated": len(SUPERHEATED_DETECTORS),
        "errors": errors,
        "warnings": warnings,
        "status": "CONFORMANT" if not errors else "NONCONFORMANT",
    }

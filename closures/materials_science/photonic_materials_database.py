"""Photonic materials database — Waveguides, resonators, lasers, and metasurfaces.

Encodes photonic device material properties, optical performance metrics, and
integration characteristics sourced from the MDPI Photonics journal (EISSN
2304-6732, CiteScore 3.5, Impact Factor 1.9).  All data is mapped through the
GCD kernel to produce Tier-1 invariants (F, ω, S, C, κ, IC) via an 8-channel
trace vector per photonic material/device.

Trace vector channels (equal weights w_i = 1/8):
    0. wavelength_norm — operating wavelength (nm), log-scaled to [0,1]
    1. refractive_index_norm — refractive index n, linear-normalized [1.0, 4.0]
    2. loss_performance — optical loss (dB/cm), inverse-log (low loss → high value)
    3. quality_factor_norm — Q factor, log-scaled to [0,1]
    4. bandwidth_norm — spectral bandwidth or tunability, log-normalized
    5. efficiency_norm — coupling/gain/conversion efficiency, normalized
    6. thermal_robustness — process/operating temperature stability
    7. integration_density — miniaturization metric (inverse footprint)

Sources (see paper/Bibliography.bib):
    - Li, Y.; Zhang, Z.; Xu, J.; Zhang, R.; Zhang, Y.; Yang, J. (2026).
      "Low-Loss Silicon Nitride Bent Waveguides at O-Band with Modified
      Hermite Curves." Photonics 13(2):175.
      DOI: 10.3390/photonics13020175
    - Fan, X.; Yu, G.; Yu, Z.; Wang, J.; Zuo, D.; Peng, R. (2026).
      "An 850 nm Grating Coupler on Thin-Film Lithium Niobate Enabled by
      Topological Unidirectional Guided Resonance."
      Photonics 13(2):199.  DOI: 10.3390/photonics13020199
    - Wang, L.; Li, H.; Li, P.; Han, X.; Gu, H.; Wang, X.; Zhang, D. (2026).
      "On-Chip Optical Signal Enhancement in Micro-Ring Resonators Using a
      NaYF4:Er3+-Doped Polymer Nanocomposite."
      Photonics 13(2):200.  DOI: 10.3390/photonics13020200
    - Li, A.; Zhang, Y.; Lu, X. (2026).
      "Whispering-Gallery-Mode Microcavity Lasers from Visible to
      Mid-Infrared: Applications." Photonics 13(2):174.
      DOI: 10.3390/photonics13020174
    - Amoateng, E.; Mubarak Sani, E.; Kwakye, K.S.O.; Pitilakis, A. (2026).
      "Analysis and Design of a Hybrid Graphene/VO2 Terahertz Metasurface
      with Independently Reconfigurable Reflection Phase and Magnitude."
      Photonics 13(2):195.  DOI: 10.3390/photonics13020195

Copyright-free reference data: refractive indices, absorption coefficients,
wavelengths, and Q factors are factual scientific measurements not subject
to copyright.  Device performance data extracted from CC BY 4.0 open-access
articles published by MDPI.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, NamedTuple

# ═══════════════════════════════════════════════════════════════════
#  Enumerations
# ═══════════════════════════════════════════════════════════════════


class DeviceType(StrEnum):
    """Photonic device classification by function."""

    WAVEGUIDE = "Waveguide"
    GRATING_COUPLER = "Grating coupler"
    MICRO_RING_RESONATOR = "Micro-ring resonator"
    WGM_RESONATOR = "WGM resonator"
    WGM_LASER = "WGM laser"
    MICRODISK_LASER = "Microdisk laser"
    PARAMETRIC_OSCILLATOR = "Parametric oscillator"
    METASURFACE = "Metasurface"
    PHOTONIC_WAVEGUIDE = "Photonic waveguide"
    OPTICAL_AMPLIFIER = "Optical amplifier"


class MaterialPlatform(StrEnum):
    """Material platform for photonic device."""

    SILICON_NITRIDE = "Silicon nitride (Si₃N₄)"
    LITHIUM_NIOBATE = "Lithium niobate (LiNbO₃)"
    POLYMER_NANOCOMPOSITE = "Polymer nanocomposite"
    FUSED_SILICA = "Fused silica (SiO₂)"
    FLUORIDE_GLASS = "Fluoride glass (ZBLAN)"
    GRAPHENE_VO2 = "Graphene/VO₂ hybrid"
    SILICON = "Silicon (Si)"
    III_V_SEMICONDUCTOR = "III-V semiconductor (InP)"
    GROUP_IV_ALLOY = "Group IV alloy (GeSn)"
    CHALCOGENIDE = "Chalcogenide glass"
    CRYSTALLINE_FLUORIDE = "Crystalline fluoride (MgF₂)"
    III_NITRIDE = "III-Nitride (GaN)"
    PEROVSKITE = "Perovskite (CsPbBr₃)"
    NONLINEAR_CRYSTAL = "Nonlinear crystal (AgGaSe₂)"


class SpectralBand(StrEnum):
    """Operating spectral band."""

    VISIBLE = "Visible (380-780 nm)"
    NEAR_IR_O = "Near-IR O-band (1260-1360 nm)"
    NEAR_IR_C = "Near-IR C-band (1530-1565 nm)"
    NEAR_IR_S = "Near-IR S-band (1460-1530 nm)"
    SHORT_WAVE = "Short-wave (780-1260 nm)"
    MID_IR = "Mid-IR (2-8 μm)"
    TERAHERTZ = "Terahertz (0.1-10 THz)"


class ApplicationField(StrEnum):
    """Primary application domain."""

    TELECOM = "Optical communications"
    SENSING = "Sensing and detection"
    DISPLAY = "Display and illumination"
    NONLINEAR = "Nonlinear optics"
    QUANTUM = "Quantum photonics"
    RECONFIGURABLE = "Reconfigurable devices"
    DATA_INTERCONNECT = "Data interconnects"
    SPECTROSCOPY = "Spectroscopy"
    MID_IR_LASER = "Mid-IR laser source"
    BIOPHOTONICS = "Biophotonics"


# ═══════════════════════════════════════════════════════════════════
#  Normalization Helpers
# ═══════════════════════════════════════════════════════════════════

_EPSILON = 1e-8  # Frozen guard band

# Wavelength range: 380 nm (visible) to 300,000 nm (300 μm THz)
_WL_MIN_LOG = math.log(380.0)
_WL_MAX_LOG = math.log(300_000.0)

# Refractive index range: 1.0 (air) to 4.0 (silicon, GaAs)
_N_MIN = 1.0
_N_MAX = 4.0

# Loss range: 0.001 dB/cm (ultra-low) to 100 dB/cm (high loss)
_LOSS_MIN_LOG = math.log(0.001)
_LOSS_MAX_LOG = math.log(100.0)

# Q factor range: 10 to 10^10
_Q_MIN_LOG = math.log(10.0)
_Q_MAX_LOG = math.log(1e10)

# Bandwidth range: 0.01 nm to 10,000 nm
_BW_MIN_LOG = math.log(0.01)
_BW_MAX_LOG = math.log(10_000.0)

# Footprint range: 1 μm² to 1e8 μm² (10 cm²)
_FP_MIN_LOG = math.log(1.0)
_FP_MAX_LOG = math.log(1e8)

# Temperature range for thermal robustness: 4 K to 1400 K
_TEMP_MIN = 4.0
_TEMP_MAX = 1400.0


def _clamp(val: float, lo: float, hi: float) -> float:
    """Clamp value to [lo, hi]."""
    return max(lo, min(val, hi))


def _log_normalize(val: float, log_min: float, log_max: float) -> float:
    """Log-scale normalization to [ε, 1-ε]."""
    if val <= 0:
        return _EPSILON
    log_val = math.log(val)
    raw = (log_val - log_min) / (log_max - log_min)
    return _clamp(raw, _EPSILON, 1.0 - _EPSILON)


def _linear_normalize(
    val: float | None,
    vmin: float,
    vmax: float,
) -> float:
    """Linear normalization to [ε, 1-ε]."""
    if val is None:
        return 0.5
    raw = (val - vmin) / (vmax - vmin)
    return _clamp(raw, _EPSILON, 1.0 - _EPSILON)


def _inverse_log_normalize(val: float, log_min: float, log_max: float) -> float:
    """Inverse log-scale: lower raw value → higher channel value."""
    if val <= 0:
        return 1.0 - _EPSILON
    log_val = math.log(val)
    raw = (log_val - log_min) / (log_max - log_min)
    return _clamp(1.0 - raw, _EPSILON, 1.0 - _EPSILON)


# ═══════════════════════════════════════════════════════════════════
#  Data Classes
# ═══════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class PhotonicMaterial:
    """Photonic material/device with optical and integration properties.

    Sources:
        Material data: established literature values (n, absorption)
        Device data: MDPI Photonics Vol. 13 (CC BY 4.0)
        WGM review data: Li, Zhang, Lu (Photonics 2026, 13(2), 174)
    """

    name: str
    material_formula: str
    device_type: DeviceType
    platform: MaterialPlatform
    spectral_band: SpectralBand
    application: ApplicationField

    # ── Optical properties ───────────────────────────────────────
    operating_wavelength_nm: float  # Primary operating wavelength (nm)
    refractive_index: float  # Effective or material refractive index
    optical_loss_dB_cm: float  # Propagation/insertion loss (dB/cm)
    quality_factor: float | None = None  # Q factor (resonators)

    # ── Performance metrics ──────────────────────────────────────
    bandwidth_nm: float | None = None  # Spectral bandwidth or FSR (nm)
    efficiency_dB: float | None = None  # Coupling/gain efficiency (dB)
    threshold_power: float | None = None  # Lasing threshold (μW or mW)
    threshold_unit: str = ""

    # ── Thermal / environmental ──────────────────────────────────
    max_operating_temp_K: float = 300.0  # Max operating temperature
    process_temp_K: float = 300.0  # Fabrication/process temperature

    # ── Integration metrics ──────────────────────────────────────
    footprint_um2: float = 10000.0  # Device footprint (μm²)
    cmos_compatible: bool = False  # CMOS process compatibility

    # ── Provenance ───────────────────────────────────────────────
    source_article: str = ""  # DOI
    source_journal: str = "Photonics"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON output."""
        return {
            "name": self.name,
            "material_formula": self.material_formula,
            "device_type": self.device_type.value,
            "platform": self.platform.value,
            "spectral_band": self.spectral_band.value,
            "application": self.application.value,
            "operating_wavelength_nm": self.operating_wavelength_nm,
            "refractive_index": self.refractive_index,
            "optical_loss_dB_cm": self.optical_loss_dB_cm,
            "quality_factor": self.quality_factor,
            "bandwidth_nm": self.bandwidth_nm,
            "efficiency_dB": self.efficiency_dB,
            "threshold_power": self.threshold_power,
            "threshold_unit": self.threshold_unit,
            "max_operating_temp_K": self.max_operating_temp_K,
            "process_temp_K": self.process_temp_K,
            "footprint_um2": self.footprint_um2,
            "cmos_compatible": self.cmos_compatible,
            "source_article": self.source_article,
            "source_journal": self.source_journal,
        }


class PhotonicKernelResult(NamedTuple):
    """GCD kernel output for a photonic material/device."""

    name: str
    material_formula: str
    device_type: str
    platform: str
    trace: tuple[float, ...]
    F: float  # Fidelity
    omega: float  # Drift
    S: float  # Bernoulli field entropy
    C: float  # Curvature
    kappa: float  # Log-integrity
    IC: float  # Integrity composite
    regime: str  # Stable / Watch / Collapse


# Thermal robustness mapping: higher process/operating temp → more robust
_THERMAL_WEIGHT = {
    "cryo": 0.15,  # Requires cryogenic cooling
    "room": 0.40,  # Room temp only
    "moderate": 0.60,  # Up to ~500 K
    "high": 0.80,  # Up to ~1000 K (LPCVD, MOCVD)
    "very_high": 0.95,  # >1000 K (Si₃N₄ LPCVD)
}


# ═══════════════════════════════════════════════════════════════════
#  PHOTONIC MATERIALS DATABASE  (14 entries)
# ═══════════════════════════════════════════════════════════════════
#
# Data sources by article:
#
# ── SiN Bent Waveguides (Photonics 13(2), 175) ──────────────────
#   Li et al. (2026), Beijing Univ. Posts & Telecomm.
#   Si₃N₄ on SiO₂ (LPCVD), 800 nm × 300 nm cross-section
#   O-band 1311 nm, 0.076 dB/90° bend (R=15 μm)
#   Propagation loss: 0.034 dB/m state-of-art
#   Modified Hermite curves with PSO optimization
#   Fabrication tolerance ±20 nm width deviation
#
# ── TFLN Grating Coupler (Photonics 13(2), 199) ─────────────────
#   Fan et al. (2026), Peking Univ. / Legend Holdings
#   Thin-film LiNbO₃ (TFLN), topological BIC-based design
#   850 nm coupling, CE = −0.6 dB (best TFLN without mirrors)
#   Up/down ratio: 63.7 dB, back-scattering 1.5%
#   VCSEL-matched for short-reach optical interconnects
#
# ── Er-Doped Micro-Ring (Photonics 13(2), 200) ──────────────────
#   Wang et al. (2026), Jilin University
#   NaYF₄:5%Er³⁺ in SU-8 2002 + PMMA cladding
#   Q = 5.72×10⁴, FSR = 0.081 nm, gain 8.92 dB @ 1527 nm
#   Propagation loss: 3.14 dB/cm @ 1530 nm
#   Overlay factors: signal 0.79, pump 0.80
#
# ── WGM Review Data (Photonics 13(2), 174) ──────────────────────
#   Li, Zhang, Lu (2026) — comprehensive review
#   Silica microsphere: Q ~ 10⁸, visible to near-IR
#   ZBLAN: Q ~ 5.4×10⁸ @ 1.55 μm, mid-IR @ 2.7 μm
#   MgF₂: Q > 10⁸ @ 1550 nm, Brillouin/Raman
#   GaN microdisk: visible CW WGM lasing
#   QD on Si: 1.3 μm, submilliamp threshold (0.6 μA)
#   AgGaSe₂: parametric oscillator to 8 μm
#   HgCdTe: ~4 μm, up to 230 K
#   Perovskite: visible, broad tunability, low threshold
#
# ── Graphene/VO₂ THz Metasurface (Photonics 13(2), 195) ────────
#   Amoateng et al. (2026), GCTU / KNUST / Aristotle Univ.
#   Phase modulation: ~250°, magnitude: >20 dB
#   ~3 dB to near-perfect absorption
#   Equivalent circuit + transmission line modeling
#   6G and beyond wireless communications
# ═══════════════════════════════════════════════════════════════════

MATERIALS: tuple[PhotonicMaterial, ...] = (
    # ── 1. Silicon Nitride Bent Waveguide ────────────────────────
    # Li et al. (2026) Photonics 13(2):175
    # LPCVD Si₃N₄, n ≈ 2.0, 800×300 nm, O-band
    # Bending loss: 0.076 dB/90° bend at R=15 μm
    # State-of-art propagation: 0.034 dB/m = 0.0034 dB/cm
    PhotonicMaterial(
        name="SiN Bent Waveguide (O-band)",
        material_formula="Si₃N₄",
        device_type=DeviceType.WAVEGUIDE,
        platform=MaterialPlatform.SILICON_NITRIDE,
        spectral_band=SpectralBand.NEAR_IR_O,
        application=ApplicationField.TELECOM,
        operating_wavelength_nm=1311.0,
        refractive_index=2.0,
        optical_loss_dB_cm=0.0034,
        quality_factor=None,
        bandwidth_nm=100.0,  # O-band window ~100 nm
        efficiency_dB=-0.076,  # Per 90° bend loss
        max_operating_temp_K=400.0,
        process_temp_K=1100.0,  # LPCVD 900-1100°C
        footprint_um2=240.0,  # 800 nm × 300 nm cross-section
        cmos_compatible=True,
        source_article="10.3390/photonics13020175",
    ),
    # ── 2. TFLN Grating Coupler (850 nm) ────────────────────────
    # Fan et al. (2026) Photonics 13(2):199
    # Thin-film LiNbO₃, topological UGR design
    # CE = −0.6 dB (highest for TFLN without mirrors)
    # Up/down ratio: 63.7 dB, back-scattering: 1.5%
    PhotonicMaterial(
        name="TFLN Grating Coupler (850 nm)",
        material_formula="LiNbO₃",
        device_type=DeviceType.GRATING_COUPLER,
        platform=MaterialPlatform.LITHIUM_NIOBATE,
        spectral_band=SpectralBand.SHORT_WAVE,
        application=ApplicationField.DATA_INTERCONNECT,
        operating_wavelength_nm=850.0,
        refractive_index=2.2,
        optical_loss_dB_cm=0.5,  # Typical TFLN waveguide loss
        quality_factor=80.0,  # UGR unit cell Q
        bandwidth_nm=50.0,  # 1 dB bandwidth window
        efficiency_dB=-0.6,  # Coupling efficiency
        max_operating_temp_K=350.0,
        process_temp_K=600.0,  # Ion-slicing + etching
        footprint_um2=7700.0,  # Grating area ~277 nm period × 28 periods
        cmos_compatible=True,
        source_article="10.3390/photonics13020199",
    ),
    # ── 3. Er-Doped Polymer Micro-Ring ──────────────────────────
    # Wang et al. (2026) Photonics 13(2):200
    # NaYF₄:5%Er³⁺ in SU-8 2002, PMMA cladding
    # Q = 5.72×10⁴, FSR = 0.081 nm, gain 8.92 dB @ 1527 nm
    PhotonicMaterial(
        name="Er-Doped Polymer Micro-Ring",
        material_formula="NaYF₄:Er³⁺/SU-8",
        device_type=DeviceType.MICRO_RING_RESONATOR,
        platform=MaterialPlatform.POLYMER_NANOCOMPOSITE,
        spectral_band=SpectralBand.NEAR_IR_C,
        application=ApplicationField.TELECOM,
        operating_wavelength_nm=1527.0,
        refractive_index=1.56,
        optical_loss_dB_cm=3.14,
        quality_factor=5.72e4,
        bandwidth_nm=0.081,  # FSR
        efficiency_dB=8.92,  # On-chip gain
        threshold_power=108.0,
        threshold_unit="mW",
        max_operating_temp_K=370.0,  # Polymer limit
        process_temp_K=340.0,  # SU-8 processing
        footprint_um2=12566.0,  # π × (2000/2)² ≈ π×10⁶ nm² ≈ 12566 μm²
        cmos_compatible=False,
        source_article="10.3390/photonics13020200",
    ),
    # ── 4. Silica WGM Microsphere ───────────────────────────────
    # Li, Zhang, Lu (2026) review; Photonics 13(2):174
    # SiO₂ microsphere, Q ~ 10⁸, visible to C-band
    # Tapered-fiber coupling ~99% efficiency
    # Raman lasing threshold ~100 μW
    PhotonicMaterial(
        name="Silica WGM Microsphere",
        material_formula="SiO₂",
        device_type=DeviceType.WGM_RESONATOR,
        platform=MaterialPlatform.FUSED_SILICA,
        spectral_band=SpectralBand.NEAR_IR_C,
        application=ApplicationField.SENSING,
        operating_wavelength_nm=1550.0,
        refractive_index=1.45,
        optical_loss_dB_cm=0.001,
        quality_factor=1e8,
        bandwidth_nm=0.015,  # Ultra-narrow linewidth
        efficiency_dB=-0.04,  # ~99% coupling → −0.04 dB
        max_operating_temp_K=1200.0,  # Silica softening ~1400 K
        process_temp_K=1800.0,  # Flame melting
        footprint_um2=78540.0,  # ~100 μm radius sphere
        cmos_compatible=False,
        source_article="10.3390/photonics13020174",
    ),
    # ── 5. ZBLAN Fluoride Microsphere (Mid-IR) ──────────────────
    # Tebeneva et al. (2022) via Li review; Q ~ 5.4×10⁸ @ 1.55 μm
    # Deng et al. (2014): Er-doped ZBLAN, 2.7 μm lasing
    # Threshold < 150 μW at 2.7 μm
    PhotonicMaterial(
        name="ZBLAN Fluoride Microsphere",
        material_formula="ZrF₄-BaF₂-LaF₃-AlF₃-NaF",
        device_type=DeviceType.WGM_LASER,
        platform=MaterialPlatform.FLUORIDE_GLASS,
        spectral_band=SpectralBand.MID_IR,
        application=ApplicationField.MID_IR_LASER,
        operating_wavelength_nm=2700.0,
        refractive_index=1.50,
        optical_loss_dB_cm=0.005,
        quality_factor=5.4e8,
        bandwidth_nm=10.0,  # Mid-IR emission bandwidth
        efficiency_dB=-8.2,  # ~150 μW threshold (moderate)
        threshold_power=150.0,
        threshold_unit="μW",
        max_operating_temp_K=350.0,  # Glass transition ~540 K
        process_temp_K=600.0,  # Melt-drawing
        footprint_um2=196350.0,  # ~250 μm radius sphere
        cmos_compatible=False,
        source_article="10.3390/photonics13020174",
    ),
    # ── 6. MgF₂ Brillouin/Raman WGM Resonator ──────────────────
    # Tian & Lin (2024) via Li review; Q > 10⁸ @ 1550 nm
    # Simultaneous Brillouin + Raman lasing
    # Disk radius 180 μm, z-cut crystal
    PhotonicMaterial(
        name="MgF₂ Brillouin-Raman Resonator",
        material_formula="MgF₂",
        device_type=DeviceType.WGM_RESONATOR,
        platform=MaterialPlatform.CRYSTALLINE_FLUORIDE,
        spectral_band=SpectralBand.NEAR_IR_C,
        application=ApplicationField.NONLINEAR,
        operating_wavelength_nm=1550.0,
        refractive_index=1.37,
        optical_loss_dB_cm=0.0005,
        quality_factor=1e8,
        bandwidth_nm=0.001,  # Ultra-narrow Brillouin linewidth
        efficiency_dB=-3.0,  # Moderate conversion efficiency
        max_operating_temp_K=600.0,  # Crystal stability
        process_temp_K=800.0,  # Polishing / machining
        footprint_um2=101788.0,  # π × 180² μm²
        cmos_compatible=False,
        source_article="10.3390/photonics13020174",
    ),
    # ── 7. GaN WGM Microdisk (Visible) ─────────────────────────
    # Tamboli et al. (2007) via Li review
    # Room-temp CW lasing in GaN/InGaN microdisks
    # Visible emission, direct bandgap III-Nitride
    PhotonicMaterial(
        name="GaN/InGaN WGM Microdisk",
        material_formula="GaN/InGaN",
        device_type=DeviceType.MICRODISK_LASER,
        platform=MaterialPlatform.III_NITRIDE,
        spectral_band=SpectralBand.VISIBLE,
        application=ApplicationField.DISPLAY,
        operating_wavelength_nm=450.0,
        refractive_index=2.45,
        optical_loss_dB_cm=5.0,
        quality_factor=3000.0,
        bandwidth_nm=1.0,  # Visible laser linewidth
        efficiency_dB=-6.0,  # Moderate coupling
        max_operating_temp_K=400.0,
        process_temp_K=1050.0,  # MOCVD growth
        footprint_um2=78.5,  # ~5 μm radius disk
        cmos_compatible=False,
        source_article="10.3390/photonics13020174",
    ),
    # ── 8. QD-on-Si Micro-Laser (1.3 μm) ───────────────────────
    # Wan et al. (2017) via Li review
    # InAs QDs on (001) Si, CW lasing ~1.3 μm
    # Threshold ~0.6 mA, operation up to ~100°C
    PhotonicMaterial(
        name="InAs QD-on-Si Micro-Laser",
        material_formula="InAs/GaAs/Si",
        device_type=DeviceType.MICRODISK_LASER,
        platform=MaterialPlatform.III_V_SEMICONDUCTOR,
        spectral_band=SpectralBand.NEAR_IR_O,
        application=ApplicationField.TELECOM,
        operating_wavelength_nm=1300.0,
        refractive_index=3.5,
        optical_loss_dB_cm=2.0,
        quality_factor=5000.0,
        bandwidth_nm=5.0,  # QD emission bandwidth
        efficiency_dB=-10.0,  # Submilliamp threshold
        threshold_power=600.0,
        threshold_unit="μA",
        max_operating_temp_K=373.0,  # Up to ~100°C
        process_temp_K=700.0,  # MBE growth
        footprint_um2=78.5,  # ~5 μm radius
        cmos_compatible=False,
        source_article="10.3390/photonics13020174",
    ),
    # ── 9. CsPbBr₃ Perovskite Micro-Laser ──────────────────────
    # Tang et al. (2017) via Li review
    # Submicron spheres, visible single-mode lasing
    # High optical gain, bandgap tunability, low threshold
    PhotonicMaterial(
        name="CsPbBr₃ Perovskite Microlaser",
        material_formula="CsPbBr₃",
        device_type=DeviceType.WGM_LASER,
        platform=MaterialPlatform.PEROVSKITE,
        spectral_band=SpectralBand.VISIBLE,
        application=ApplicationField.DISPLAY,
        operating_wavelength_nm=530.0,
        refractive_index=2.3,
        optical_loss_dB_cm=10.0,  # Higher loss in perovskites
        quality_factor=2000.0,
        bandwidth_nm=3.0,  # Visible emission
        efficiency_dB=-5.0,  # Low threshold
        max_operating_temp_K=310.0,  # Environmental sensitivity
        process_temp_K=350.0,  # Solution processing
        footprint_um2=0.8,  # ~0.5 μm radius submicron sphere
        cmos_compatible=False,
        source_article="10.3390/photonics13020174",
    ),
    # ── 10. AgGaSe₂ Parametric Oscillator (Mid-IR) ─────────────
    # Meisenheimer et al. (2017) via Li review
    # CW parametric oscillation, idler to 8 μm
    # 3.5 mm disk, mW thresholds, 10-800 μW output
    PhotonicMaterial(
        name="AgGaSe₂ Mid-IR OPO",
        material_formula="AgGaSe₂",
        device_type=DeviceType.PARAMETRIC_OSCILLATOR,
        platform=MaterialPlatform.NONLINEAR_CRYSTAL,
        spectral_band=SpectralBand.MID_IR,
        application=ApplicationField.MID_IR_LASER,
        operating_wavelength_nm=8000.0,
        refractive_index=2.59,
        optical_loss_dB_cm=0.1,  # Moderate loss at mid-IR
        quality_factor=1e6,
        bandwidth_nm=100.0,  # 100 nm temperature tuning
        efficiency_dB=-15.0,  # mW pump → μW output
        threshold_power=5.0,
        threshold_unit="mW",
        max_operating_temp_K=400.0,
        process_temp_K=850.0,  # Crystal growth
        footprint_um2=9.62e6,  # 3.5 mm diameter disk
        cmos_compatible=False,
        source_article="10.3390/photonics13020174",
    ),
    # ── 11. Graphene/VO₂ THz Metasurface ───────────────────────
    # Amoateng et al. (2026) Photonics 13(2):195
    # Phase modulation ~250°, magnitude >20 dB
    # Graphene patches + VO₂ overlayer
    # Equivalent circuit / transmission line modeling
    PhotonicMaterial(
        name="Graphene/VO₂ THz Metasurface",
        material_formula="Graphene/VO₂/SiO₂",
        device_type=DeviceType.METASURFACE,
        platform=MaterialPlatform.GRAPHENE_VO2,
        spectral_band=SpectralBand.TERAHERTZ,
        application=ApplicationField.RECONFIGURABLE,
        operating_wavelength_nm=100_000.0,  # ~3 THz → 100 μm
        refractive_index=1.5,  # Effective index of stack
        optical_loss_dB_cm=3.0,  # ~−3 dB reflection at resonance
        quality_factor=50.0,  # Broad resonance
        bandwidth_nm=30_000.0,  # THz bandwidth ~30 μm
        efficiency_dB=-3.0,  # −3 dB to near-perfect absorption
        max_operating_temp_K=340.0,  # VO₂ phase transition ~340 K
        process_temp_K=500.0,  # CVD graphene + sputtering
        footprint_um2=10000.0,  # Unit cell array
        cmos_compatible=False,
        source_article="10.3390/photonics13020195",
    ),
    # ── 12. Silicon Photonic Waveguide (C-band) ─────────────────
    # Well-established SOI platform
    # n = 3.48, 450×220 nm wire, loss ~1 dB/cm
    # Dominant commercial photonic platform
    PhotonicMaterial(
        name="Silicon SOI Waveguide (C-band)",
        material_formula="Si",
        device_type=DeviceType.PHOTONIC_WAVEGUIDE,
        platform=MaterialPlatform.SILICON,
        spectral_band=SpectralBand.NEAR_IR_C,
        application=ApplicationField.TELECOM,
        operating_wavelength_nm=1550.0,
        refractive_index=3.48,
        optical_loss_dB_cm=1.0,
        quality_factor=None,
        bandwidth_nm=100.0,  # C+L band
        efficiency_dB=-0.5,  # Typical insertion
        max_operating_temp_K=500.0,
        process_temp_K=1000.0,  # SOI fabrication
        footprint_um2=99.0,  # 450 nm × 220 nm
        cmos_compatible=True,
        source_article="10.3390/photonics13020175",
    ),
    # ── 13. Chalcogenide Glass Microsphere ──────────────────────
    # Ge-Ga-Sb-S glass, mid-IR platform
    # Yang et al. (2018) via Li review: lasing ~1.9 μm
    # Mid-IR transparent, high nonlinearity
    PhotonicMaterial(
        name="Chalcogenide Glass Microsphere",
        material_formula="Ge-Ga-Sb-S",
        device_type=DeviceType.WGM_LASER,
        platform=MaterialPlatform.CHALCOGENIDE,
        spectral_band=SpectralBand.MID_IR,
        application=ApplicationField.MID_IR_LASER,
        operating_wavelength_nm=1900.0,
        refractive_index=2.4,
        optical_loss_dB_cm=0.1,
        quality_factor=1e5,
        bandwidth_nm=20.0,
        efficiency_dB=-8.0,
        max_operating_temp_K=350.0,  # Low glass transition
        process_temp_K=500.0,  # Glass melting
        footprint_um2=31416.0,  # ~100 μm radius
        cmos_compatible=False,
        source_article="10.3390/photonics13020174",
    ),
    # ── 14. Er:YAG WGM Microdisk Laser ─────────────────────────
    # Wang et al. (2025) via Li review: 35 μm diameter
    # Low-threshold single-mode at C-band
    # Crystalline gain medium, femtosecond-laser written
    PhotonicMaterial(
        name="Er:YAG WGM Microdisk Laser",
        material_formula="Er:Y₃Al₅O₁₂",
        device_type=DeviceType.WGM_LASER,
        platform=MaterialPlatform.CRYSTALLINE_FLUORIDE,  # Crystalline WGM
        spectral_band=SpectralBand.NEAR_IR_C,
        application=ApplicationField.TELECOM,
        operating_wavelength_nm=1535.0,
        refractive_index=1.82,
        optical_loss_dB_cm=0.05,
        quality_factor=1e6,
        bandwidth_nm=2.0,  # Single-mode emission
        efficiency_dB=-5.0,  # 27% slope efficiency (Yb:YAG reference)
        max_operating_temp_K=500.0,
        process_temp_K=900.0,  # Femtosecond laser processing
        footprint_um2=962.0,  # π × 17.5² for 35 μm diameter
        cmos_compatible=False,
        source_article="10.3390/photonics13020174",
    ),
)


# ═══════════════════════════════════════════════════════════════════
#  Kernel Computation
# ═══════════════════════════════════════════════════════════════════


def build_trace(material: PhotonicMaterial) -> tuple[float, ...]:
    """Build 8-channel trace vector for a photonic material/device.

    Channels:
        0. wavelength_norm — operating wavelength, log-normalized
        1. refractive_index_norm — n, linear [1.0, 4.0]
        2. loss_performance — inverse-log loss (low loss → high value)
        3. quality_factor_norm — Q factor, log-normalized (0.5 if None)
        4. bandwidth_norm — spectral bandwidth, log-normalized
        5. efficiency_norm — coupling/gain efficiency, normalized
        6. thermal_robustness — combined process + operating temp
        7. integration_density — inverse footprint, log-normalized
    """
    # Ch 0: Operating wavelength (log-normalized across UV to THz)
    c0 = _log_normalize(material.operating_wavelength_nm, _WL_MIN_LOG, _WL_MAX_LOG)

    # Ch 1: Refractive index (linear [1.0, 4.0])
    c1 = _linear_normalize(material.refractive_index, _N_MIN, _N_MAX)

    # Ch 2: Loss performance (inverse: low loss → high channel value)
    c2 = _inverse_log_normalize(material.optical_loss_dB_cm, _LOSS_MIN_LOG, _LOSS_MAX_LOG)

    # Ch 3: Quality factor (log-normalized; 0.5 if None)
    if material.quality_factor is not None and material.quality_factor > 0:
        c3 = _log_normalize(material.quality_factor, _Q_MIN_LOG, _Q_MAX_LOG)
    else:
        c3 = 0.5

    # Ch 4: Bandwidth (log-normalized; 0.5 if None)
    if material.bandwidth_nm is not None and material.bandwidth_nm > 0:
        c4 = _log_normalize(material.bandwidth_nm, _BW_MIN_LOG, _BW_MAX_LOG)
    else:
        c4 = 0.5

    # Ch 5: Efficiency (dB to linear fraction)
    if material.efficiency_dB is not None:
        # Convert from dB: 0 dB = 100%, -3 dB = 50%, +10 dB = 1000% (gain)
        eff_db = material.efficiency_dB
        if eff_db >= 0:
            # Gain: normalize 0..20 dB → 0.5..0.99
            c5 = _clamp(0.5 + eff_db / 40.0, _EPSILON, 1.0 - _EPSILON)
        else:
            # Loss: -30 dB → ~0.001, -0 dB → ~0.5
            linear_eff = 10 ** (eff_db / 10.0)
            c5 = _clamp(linear_eff * 0.5, _EPSILON, 1.0 - _EPSILON)
        # Clamp to valid range
        c5 = _clamp(c5, _EPSILON, 1.0 - _EPSILON)
    else:
        c5 = 0.5

    # Ch 6: Thermal robustness (max of process and operating, normalized)
    max_temp = max(material.process_temp_K, material.max_operating_temp_K)
    c6 = _linear_normalize(max_temp, _TEMP_MIN, _TEMP_MAX)

    # Ch 7: Integration density (inverse footprint — smaller = denser)
    # Invert: small footprint → high integration density
    c7 = _inverse_log_normalize(material.footprint_um2, _FP_MIN_LOG, _FP_MAX_LOG)

    return (c0, c1, c2, c3, c4, c5, c6, c7)


def compute_photonic_kernel(material: PhotonicMaterial) -> PhotonicKernelResult:
    """Compute GCD kernel invariants for a photonic material/device.

    Maps material properties through an 8-channel trace vector,
    then derives Tier-1 invariants:
        F = Σ wᵢ·cᵢ        (fidelity)
        ω = 1 − F           (drift; duality identity)
        S = −Σ wᵢ[cᵢ ln cᵢ + (1−cᵢ) ln(1−cᵢ)]  (Bernoulli field entropy)
        C = stddev(cᵢ) / 0.5  (curvature)
        κ = Σ wᵢ ln(cᵢ)     (log-integrity)
        IC = exp(κ)          (integrity composite)

    Regime classification from frozen contract thresholds:
        Stable:   ω < 0.038 AND F > 0.90 AND S < 0.15 AND C < 0.14
        Collapse: ω ≥ 0.30
        Watch:    otherwise
    """
    trace = build_trace(material)
    n = len(trace)
    w = 1.0 / n  # Equal weights

    # Tier-1 invariants
    F = sum(w * c for c in trace)
    omega = 1.0 - F

    # Bernoulli field entropy
    S = 0.0
    for c in trace:
        ce = _clamp(c, _EPSILON, 1.0 - _EPSILON)
        S -= w * (ce * math.log(ce) + (1.0 - ce) * math.log(1.0 - ce))

    # Curvature
    mean_c = F  # F equals mean since equal weights
    variance = sum(w * (c - mean_c) ** 2 for c in trace)
    C_val = _clamp(math.sqrt(variance) / 0.5, 0.0, 1.0)

    # Log-integrity and IC
    kappa = sum(w * math.log(_clamp(c, _EPSILON, 1.0)) for c in trace)
    IC = math.exp(kappa)

    # Regime classification
    if omega >= 0.30:
        regime = "Collapse"
    elif omega < 0.038 and F > 0.90 and S < 0.15 and C_val < 0.14:
        regime = "Stable"
    else:
        regime = "Watch"

    return PhotonicKernelResult(
        name=material.name,
        material_formula=material.material_formula,
        device_type=material.device_type.value,
        platform=material.platform.value,
        trace=trace,
        F=F,
        omega=omega,
        S=S,
        C=C_val,
        kappa=kappa,
        IC=IC,
        regime=regime,
    )


def compute_all_photonic_kernels() -> list[PhotonicKernelResult]:
    """Compute kernel invariants for all photonic materials in database."""
    return [compute_photonic_kernel(m) for m in MATERIALS]


# ═══════════════════════════════════════════════════════════════════
#  Cross-Domain Analysis
# ═══════════════════════════════════════════════════════════════════


def analyze_device_types() -> dict[str, dict[str, float]]:
    """Average kernel invariants per device type.

    Reveals how different photonic device architectures map to
    distinct fidelity structures:
    - Waveguides: high uniformity → high IC/F ratio
    - WGM resonators: extreme Q → specific channel dominance
    - Metasurfaces: broad tunability → moderate heterogeneity
    """
    from collections import defaultdict

    accum: dict[str, list[PhotonicKernelResult]] = defaultdict(list)
    for m in MATERIALS:
        kr = compute_photonic_kernel(m)
        accum[kr.device_type].append(kr)

    stats: dict[str, dict[str, float]] = {}
    for dtype, results in sorted(accum.items()):
        n_results = len(results)
        stats[dtype] = {
            "count": n_results,
            "mean_F": sum(r.F for r in results) / n_results,
            "mean_omega": sum(r.omega for r in results) / n_results,
            "mean_IC": sum(r.IC for r in results) / n_results,
            "mean_heterogeneity_gap": sum(r.F - r.IC for r in results) / n_results,
            "mean_S": sum(r.S for r in results) / n_results,
            "mean_C": sum(r.C for r in results) / n_results,
        }
    return stats


def analyze_spectral_bands() -> dict[str, dict[str, float]]:
    """Average kernel invariants per spectral band.

    Tests whether operating wavelength regime correlates with
    kernel structure — visible vs. telecom vs. mid-IR vs. THz.
    """
    from collections import defaultdict

    accum: dict[str, list[PhotonicKernelResult]] = defaultdict(list)
    for m in MATERIALS:
        kr = compute_photonic_kernel(m)
        accum[m.spectral_band.value].append(kr)

    stats: dict[str, dict[str, float]] = {}
    for band, results in sorted(accum.items()):
        n_results = len(results)
        stats[band] = {
            "count": n_results,
            "mean_F": sum(r.F for r in results) / n_results,
            "mean_omega": sum(r.omega for r in results) / n_results,
            "mean_IC": sum(r.IC for r in results) / n_results,
            "mean_heterogeneity_gap": sum(r.F - r.IC for r in results) / n_results,
            "mean_S": sum(r.S for r in results) / n_results,
            "mean_C": sum(r.C for r in results) / n_results,
        }
    return stats


def analyze_platform_landscape() -> dict[str, Any]:
    """Kernel landscape across material platforms.

    Maps the photonics hierarchy:
    - Passive dielectrics (SiN, SiO₂, MgF₂): low loss, high Q
    - Active semiconductors (InP, GaN, perovskite): gain + loss trade-off
    - Emerging hybrid (graphene/VO₂, polymer NCs): tunability vs. loss
    """
    results_list: list[dict[str, Any]] = []
    for m in MATERIALS:
        kr = compute_photonic_kernel(m)
        results_list.append(
            {
                "name": kr.name,
                "platform": kr.platform,
                "device_type": kr.device_type,
                "F": round(kr.F, 4),
                "omega": round(kr.omega, 4),
                "IC": round(kr.IC, 4),
                "gap": round(kr.F - kr.IC, 4),
                "IC_F_ratio": round(kr.IC / kr.F, 4) if kr.F > 0 else 0.0,
                "regime": kr.regime,
            }
        )
    # Sort by IC/F ratio (channel uniformity)
    results_list.sort(key=lambda x: x["IC_F_ratio"], reverse=True)
    return {
        "landscape": results_list,
        "n_materials": len(results_list),
        "n_platforms": len({r["platform"] for r in results_list}),
    }


# ═══════════════════════════════════════════════════════════════════
#  CLI Entry Point
# ═══════════════════════════════════════════════════════════════════


def main() -> None:
    """Print kernel invariants for all photonic materials."""
    print("=" * 82)
    print("PHOTONIC MATERIALS DATABASE — GCD Kernel Analysis")
    print("14 photonic devices/materials from MDPI Photonics (2026)")
    print("=" * 82)

    results = compute_all_photonic_kernels()
    for r in results:
        print(
            f"\n{r.name} ({r.platform})"
            f"\n  Device: {r.device_type} | Material: {r.material_formula}"
            f"\n  Trace: [{', '.join(f'{v:.3f}' for v in r.trace)}]"
            f"\n  F={r.F:.4f}  ω={r.omega:.4f}  S={r.S:.4f}  "
            f"C={r.C:.4f}  κ={r.kappa:.4f}  IC={r.IC:.4f}"
            f"\n  Δ={r.F - r.IC:.4f}  IC/F={r.IC / r.F:.4f}  "
            f"Regime: {r.regime}"
        )

    # Summary
    print("\n" + "=" * 82)
    print("SUMMARY")
    print("-" * 82)
    mean_F = sum(r.F for r in results) / len(results)
    mean_IC = sum(r.IC for r in results) / len(results)
    mean_gap = sum(r.F - r.IC for r in results) / len(results)
    print(
        f"  n={len(results)}  <F>={mean_F:.4f}  <IC>={mean_IC:.4f}  <Δ>={mean_gap:.4f}  <IC/F>={mean_IC / mean_F:.4f}"
    )

    # Device type analysis
    print("\nBy Device Type:")
    for dtype, stats in analyze_device_types().items():
        print(
            f"  {dtype}: n={stats['count']:.0f}  "
            f"<F>={stats['mean_F']:.4f}  <IC>={stats['mean_IC']:.4f}  "
            f"<Δ>={stats['mean_heterogeneity_gap']:.4f}"
        )

    # Spectral band analysis
    print("\nBy Spectral Band:")
    for band, stats in analyze_spectral_bands().items():
        print(
            f"  {band}: n={stats['count']:.0f}  "
            f"<F>={stats['mean_F']:.4f}  <IC>={stats['mean_IC']:.4f}  "
            f"<Δ>={stats['mean_heterogeneity_gap']:.4f}"
        )


if __name__ == "__main__":
    main()

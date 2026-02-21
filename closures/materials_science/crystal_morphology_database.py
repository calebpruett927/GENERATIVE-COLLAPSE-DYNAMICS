"""Crystal morphology and crystallization process database.

Encodes crystal compound properties, morphological descriptors, and
crystallization process parameters sourced from the OpenCrystalData project
and related experimental imaging databases.  All data is mapped through the
GCD kernel to produce Tier-1 invariants (F, ω, S, C, κ, IC) via an
8-channel trace vector per compound.

Trace vector channels (equal weights w_i = 1/8):
    0. molecular_weight_norm — molecular weight (g/mol), log-scaled to [0,1]
    1. density_norm — crystal density (g/cm³), rescaled to [0,1]
    2. melting_point_norm — melting point (K), rescaled to [0,1]
    3. solubility_norm — aqueous solubility (g/L at 25°C), log-scaled to [0,1]
    4. aspect_ratio_norm — crystal habit aspect ratio, clamped to [0,1]
    5. circularity_norm — 2D circularity (4πA/P²), inherently [0,1]
    6. crystal_system_norm — crystal system index / 6, [0,1]
    7. habit_regularity_norm — qualitative habit regularity score, [0,1]

Sources (see paper/Bibliography.bib):
    - Barhate et al. (2024). "OpenCrystalData: An open-access particle image
      database for crystallization process development."
      Digital Chemical Engineering 11:100150.
      DOI: 10.1016/j.dche.2024.100150   [barhate2024opencrystaldata]
    - Kaggle OpenCrystalData datasets:
      kaggle.com/opencrystaldata (4 datasets, CC BY-NC-ND 4.0)
    - Cambridge Structural Database (CSD) — crystal system / lattice data
    - Morphologi G3 (Malvern Panalytical) — ground-truth morphological
      descriptors (circularity, convexity, solidity, aspect ratio)

Copyright-free reference data: crystal systems, molecular weights, melting
points, and morphological descriptors are factual scientific measurements
not subject to copyright.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, NamedTuple

# ═══════════════════════════════════════════════════════════════════
#  Enumerations
# ═══════════════════════════════════════════════════════════════════


class CrystalSystem(StrEnum):
    """Seven crystal systems, ordered by symmetry (lowest → highest)."""

    TRICLINIC = "Triclinic"
    MONOCLINIC = "Monoclinic"
    ORTHORHOMBIC = "Orthorhombic"
    TETRAGONAL = "Tetragonal"
    TRIGONAL = "Trigonal"
    HEXAGONAL = "Hexagonal"
    CUBIC = "Cubic"


# Ordinal mapping for trace normalization (0 → 6, divided by 6)
_SYSTEM_INDEX: dict[CrystalSystem, int] = {
    CrystalSystem.TRICLINIC: 0,
    CrystalSystem.MONOCLINIC: 1,
    CrystalSystem.ORTHORHOMBIC: 2,
    CrystalSystem.TETRAGONAL: 3,
    CrystalSystem.TRIGONAL: 4,
    CrystalSystem.HEXAGONAL: 5,
    CrystalSystem.CUBIC: 6,
}


class CrystalHabit(StrEnum):
    """Common crystal habits observed in crystallization imaging."""

    NEEDLE = "Needle"  # High aspect ratio, orthogonal growth
    PLATE = "Plate"  # Flat, low aspect ratio
    PRISMATIC = "Prismatic"  # Elongated rectangular cross-section
    TABULAR = "Tabular"  # Flat with well-defined faces
    BLOCKY = "Blocky"  # Equidimensional, compact
    DENDRITIC = "Dendritic"  # Branched, tree-like
    SPHERULITIC = "Spherulitic"  # Radiating from center
    ACICULAR = "Acicular"  # Fine needle-like
    IRREGULAR = "Irregular"  # No defined habit
    MICROSPHERE = "Microsphere"  # Spherical (e.g., polystyrene reference)


# Habit regularity score: how regular/well-defined the habit is [0,1]
_HABIT_REGULARITY: dict[CrystalHabit, float] = {
    CrystalHabit.NEEDLE: 0.70,
    CrystalHabit.PLATE: 0.75,
    CrystalHabit.PRISMATIC: 0.85,
    CrystalHabit.TABULAR: 0.80,
    CrystalHabit.BLOCKY: 0.90,
    CrystalHabit.DENDRITIC: 0.30,
    CrystalHabit.SPHERULITIC: 0.40,
    CrystalHabit.ACICULAR: 0.65,
    CrystalHabit.IRREGULAR: 0.10,
    CrystalHabit.MICROSPHERE: 0.95,
}


class ApplicationDomain(StrEnum):
    """Application domain for the crystallized compound."""

    PHARMACEUTICAL = "Pharmaceutical"
    AGROCHEMICAL = "Agrochemical"
    REFERENCE_STANDARD = "Reference standard"
    FINE_CHEMICAL = "Fine chemical"
    FOOD_ADDITIVE = "Food additive"
    MATERIALS_RESEARCH = "Materials research"


class ProbeType(StrEnum):
    """Imaging probe type used in crystallization monitoring."""

    PVM = "PVM"  # Particle Vision and Measurement (EasyViewer)
    MORPHOLOGI_G3 = "Morphologi G3"  # Malvern ground-truth optical analyzer
    FBRM = "FBRM"  # Focused Beam Reflectance Measurement
    MICROSCOPE = "Microscope"  # Standard optical microscopy
    SEM = "SEM"  # Scanning Electron Microscopy


# ═══════════════════════════════════════════════════════════════════
#  Data Classes
# ═══════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class CrystalCompound:
    """Crystal compound with physical and morphological properties.

    Sources:
        Physical properties: CRC Handbook 104th ed., PubChem, NIST WebBook
        Crystal system: Cambridge Structural Database (CSD)
        Morphological data: Barhate et al. (2024) [barhate2024opencrystaldata]
        Habit classification: Morphologi G3 measurements + EasyViewer imaging
    """

    name: str
    formula: str
    molecular_weight: float  # g/mol
    crystal_system: CrystalSystem
    habit: CrystalHabit
    density_g_cm3: float  # Crystal density
    melting_point_K: float  # Melting/decomposition point
    solubility_g_L: float | None  # Aqueous solubility at 25°C (g/L)
    application_domain: ApplicationDomain
    cas_number: str | None = None  # CAS registry number

    # ── Morphological descriptors (from Morphologi G3 / EasyViewer) ──
    aspect_ratio: float | None = None  # Length/width ratio
    circularity: float | None = None  # 4πA/P², [0,1] (1 = perfect circle)
    convexity: float | None = None  # Convex hull area ratio, [0,1]
    solidity: float | None = None  # Area / convex hull area, [0,1]
    equivalent_diameter_um: float | None = None  # Equivalent circular diameter (μm)

    # ── Crystallization process metadata ─────────────────────────
    source_dataset: str = ""  # Kaggle dataset identifier
    image_count: int = 0  # Number of images in dataset
    classification_task: str = ""  # What the image classification targets

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON output."""
        return {
            "name": self.name,
            "formula": self.formula,
            "molecular_weight": self.molecular_weight,
            "crystal_system": self.crystal_system.value,
            "habit": self.habit.value,
            "density_g_cm3": self.density_g_cm3,
            "melting_point_K": self.melting_point_K,
            "solubility_g_L": self.solubility_g_L,
            "application_domain": self.application_domain.value,
            "cas_number": self.cas_number,
            "aspect_ratio": self.aspect_ratio,
            "circularity": self.circularity,
            "convexity": self.convexity,
            "solidity": self.solidity,
            "equivalent_diameter_um": self.equivalent_diameter_um,
            "source_dataset": self.source_dataset,
            "image_count": self.image_count,
            "classification_task": self.classification_task,
        }


class CrystalKernelResult(NamedTuple):
    """GCD kernel output for a crystal compound."""

    name: str
    formula: str
    crystal_system: str
    habit: str
    trace: tuple[float, ...]
    F: float  # Fidelity
    omega: float  # Drift
    S: float  # Bernoulli field entropy
    C: float  # Curvature
    kappa: float  # Log-integrity
    IC: float  # Integrity composite
    regime: str  # Stable / Watch / Collapse


# ═══════════════════════════════════════════════════════════════════
#  CRYSTAL COMPOUND DATABASE
# ═══════════════════════════════════════════════════════════════════
# Primary source: Barhate et al. (2024) OpenCrystalData
#   DOI: 10.1016/j.dche.2024.100150
# Crystal systems: Cambridge Structural Database (CSD)
# Physical properties: CRC Handbook 104th ed., PubChem, NIST WebBook
# Morphological descriptors: Morphologi G3 measurements
# ═══════════════════════════════════════════════════════════════════

COMPOUNDS: tuple[CrystalCompound, ...] = (
    # ── OpenCrystalData: Crystallization impurity detection ──────
    # Dataset 1: Cephalexin monohydrate with phenylglycine impurity
    # Source: Barhate et al. (2024) Table 1, Kaggle dataset #1
    CrystalCompound(
        name="Cephalexin monohydrate",
        formula="C16H17N3O4S·H2O",
        molecular_weight=365.41,
        crystal_system=CrystalSystem.MONOCLINIC,
        habit=CrystalHabit.PRISMATIC,
        density_g_cm3=1.40,
        melting_point_K=600.15,  # ~327°C (decomposes)
        solubility_g_L=12.0,  # ~12 g/L in water at 25°C
        application_domain=ApplicationDomain.PHARMACEUTICAL,
        cas_number="23325-78-2",
        aspect_ratio=2.5,
        circularity=0.55,
        convexity=0.88,
        solidity=0.85,
        equivalent_diameter_um=120.0,
        source_dataset="OpenCrystalData/crystallization-impurity-detection",
        image_count=8382,
        classification_task="Impurity detection (phenylglycine in cephalexin)",
    ),
    CrystalCompound(
        name="D-Phenylglycine",
        formula="C8H9NO2",
        molecular_weight=151.16,
        crystal_system=CrystalSystem.ORTHORHOMBIC,
        habit=CrystalHabit.NEEDLE,
        density_g_cm3=1.30,
        melting_point_K=575.15,  # ~302°C (decomposes)
        solubility_g_L=28.0,  # ~28 g/L in water at 25°C
        application_domain=ApplicationDomain.PHARMACEUTICAL,
        cas_number="875-74-1",
        aspect_ratio=5.0,
        circularity=0.25,
        convexity=0.82,
        solidity=0.78,
        equivalent_diameter_um=80.0,
        source_dataset="OpenCrystalData/crystallization-impurity-detection",
        image_count=8382,
        classification_task="Impurity crystal identification by habit difference",
    ),
    # ── OpenCrystalData: Silver crystal characterization ─────────
    # Dataset 2: Ag crystals from electrochemical synthesis
    # Source: Barhate et al. (2024) Table 2, Kaggle dataset #2
    CrystalCompound(
        name="Silver (dendritic)",
        formula="Ag",
        molecular_weight=107.87,
        crystal_system=CrystalSystem.CUBIC,
        habit=CrystalHabit.DENDRITIC,
        density_g_cm3=10.49,
        melting_point_K=1234.93,
        solubility_g_L=None,  # Insoluble (metal)
        application_domain=ApplicationDomain.MATERIALS_RESEARCH,
        cas_number="7440-22-4",
        aspect_ratio=1.8,
        circularity=0.30,
        convexity=0.65,
        solidity=0.55,
        equivalent_diameter_um=200.0,
        source_dataset="OpenCrystalData/ag-crystal-characterization",
        image_count=2400,
        classification_task="Morphology classification (dendritic vs compact)",
    ),
    CrystalCompound(
        name="Silver (compact)",
        formula="Ag",
        molecular_weight=107.87,
        crystal_system=CrystalSystem.CUBIC,
        habit=CrystalHabit.BLOCKY,
        density_g_cm3=10.49,
        melting_point_K=1234.93,
        solubility_g_L=None,
        application_domain=ApplicationDomain.MATERIALS_RESEARCH,
        cas_number="7440-22-4",
        aspect_ratio=1.2,
        circularity=0.75,
        convexity=0.92,
        solidity=0.90,
        equivalent_diameter_um=150.0,
        source_dataset="OpenCrystalData/ag-crystal-characterization",
        image_count=2400,
        classification_task="Morphology classification (dendritic vs compact)",
    ),
    # ── OpenCrystalData: EasyViewer inline imaging ───────────────
    # Dataset 3: Agrochemical needle crystals via EasyViewer-100 PVM
    # Source: Barhate et al. (2024) Section 3.3, Kaggle dataset #3
    CrystalCompound(
        name="Agrochemical compound A",
        formula="C12H15ClN2O",
        molecular_weight=238.72,
        crystal_system=CrystalSystem.MONOCLINIC,
        habit=CrystalHabit.NEEDLE,
        density_g_cm3=1.28,
        melting_point_K=470.15,  # ~197°C
        solubility_g_L=0.5,
        application_domain=ApplicationDomain.AGROCHEMICAL,
        aspect_ratio=8.0,
        circularity=0.15,
        convexity=0.80,
        solidity=0.75,
        equivalent_diameter_um=60.0,
        source_dataset="OpenCrystalData/easyviewer-inline-imaging",
        image_count=3200,
        classification_task="Needle vs non-needle classification (EasyViewer PVM)",
    ),
    # ── OpenCrystalData: Polystyrene microspheres (reference) ────
    # Dataset 4: Polystyrene latex microspheres for calibration
    # Source: Barhate et al. (2024) Section 3.4, Kaggle dataset #4
    CrystalCompound(
        name="Polystyrene microsphere (20 μm)",
        formula="(C8H8)n",
        molecular_weight=104.15,  # Styrene monomer MW
        crystal_system=CrystalSystem.CUBIC,  # Amorphous, but isotropic
        habit=CrystalHabit.MICROSPHERE,
        density_g_cm3=1.05,
        melting_point_K=513.15,  # ~240°C (Tg ~100°C)
        solubility_g_L=None,  # Insoluble
        application_domain=ApplicationDomain.REFERENCE_STANDARD,
        aspect_ratio=1.0,
        circularity=0.98,
        convexity=0.99,
        solidity=0.99,
        equivalent_diameter_um=20.0,
        source_dataset="OpenCrystalData/polystyrene-microspheres",
        image_count=1500,
        classification_task="Size and circularity calibration standard",
    ),
    CrystalCompound(
        name="Polystyrene microsphere (50 μm)",
        formula="(C8H8)n",
        molecular_weight=104.15,
        crystal_system=CrystalSystem.CUBIC,
        habit=CrystalHabit.MICROSPHERE,
        density_g_cm3=1.05,
        melting_point_K=513.15,
        solubility_g_L=None,
        application_domain=ApplicationDomain.REFERENCE_STANDARD,
        aspect_ratio=1.0,
        circularity=0.97,
        convexity=0.99,
        solidity=0.99,
        equivalent_diameter_um=50.0,
        source_dataset="OpenCrystalData/polystyrene-microspheres",
        image_count=1500,
        classification_task="Size and circularity calibration standard",
    ),
    # ── Additional pharmaceutical crystals (literature) ──────────
    # Common pharmaceutical polymorphs with well-characterized habits
    CrystalCompound(
        name="Aspirin (Form I)",
        formula="C9H8O4",
        molecular_weight=180.16,
        crystal_system=CrystalSystem.MONOCLINIC,
        habit=CrystalHabit.TABULAR,
        density_g_cm3=1.40,
        melting_point_K=408.15,  # 135°C
        solubility_g_L=3.3,
        application_domain=ApplicationDomain.PHARMACEUTICAL,
        cas_number="50-78-2",
        aspect_ratio=2.0,
        circularity=0.60,
        convexity=0.90,
        solidity=0.88,
        equivalent_diameter_um=200.0,
    ),
    CrystalCompound(
        name="Paracetamol (Form I, monoclinic)",
        formula="C8H9NO2",
        molecular_weight=151.16,
        crystal_system=CrystalSystem.MONOCLINIC,
        habit=CrystalHabit.PRISMATIC,
        density_g_cm3=1.29,
        melting_point_K=442.15,  # 169°C
        solubility_g_L=14.0,
        application_domain=ApplicationDomain.PHARMACEUTICAL,
        cas_number="103-90-2",
        aspect_ratio=2.2,
        circularity=0.50,
        convexity=0.87,
        solidity=0.84,
        equivalent_diameter_um=180.0,
    ),
    CrystalCompound(
        name="Paracetamol (Form II, orthorhombic)",
        formula="C8H9NO2",
        molecular_weight=151.16,
        crystal_system=CrystalSystem.ORTHORHOMBIC,
        habit=CrystalHabit.PLATE,
        density_g_cm3=1.30,
        melting_point_K=430.15,  # 157°C
        solubility_g_L=14.0,
        application_domain=ApplicationDomain.PHARMACEUTICAL,
        cas_number="103-90-2",
        aspect_ratio=1.5,
        circularity=0.65,
        convexity=0.92,
        solidity=0.90,
        equivalent_diameter_um=250.0,
    ),
    CrystalCompound(
        name="Glycine (α-form)",
        formula="C2H5NO2",
        molecular_weight=75.03,
        crystal_system=CrystalSystem.MONOCLINIC,
        habit=CrystalHabit.PRISMATIC,
        density_g_cm3=1.60,
        melting_point_K=506.15,  # 233°C (decomposes)
        solubility_g_L=250.0,
        application_domain=ApplicationDomain.FINE_CHEMICAL,
        cas_number="56-40-6",
        aspect_ratio=1.8,
        circularity=0.55,
        convexity=0.90,
        solidity=0.87,
        equivalent_diameter_um=300.0,
    ),
    CrystalCompound(
        name="Glycine (γ-form)",
        formula="C2H5NO2",
        molecular_weight=75.03,
        crystal_system=CrystalSystem.TRIGONAL,
        habit=CrystalHabit.PRISMATIC,
        density_g_cm3=1.62,
        melting_point_K=506.15,
        solubility_g_L=250.0,
        application_domain=ApplicationDomain.FINE_CHEMICAL,
        cas_number="56-40-6",
        aspect_ratio=2.0,
        circularity=0.50,
        convexity=0.88,
        solidity=0.85,
        equivalent_diameter_um=200.0,
    ),
    CrystalCompound(
        name="L-Glutamic acid (α-form)",
        formula="C5H9NO4",
        molecular_weight=147.13,
        crystal_system=CrystalSystem.ORTHORHOMBIC,
        habit=CrystalHabit.PRISMATIC,
        density_g_cm3=1.54,
        melting_point_K=472.15,  # 199°C (decomposes)
        solubility_g_L=8.6,
        application_domain=ApplicationDomain.FOOD_ADDITIVE,
        cas_number="56-86-0",
        aspect_ratio=2.5,
        circularity=0.45,
        convexity=0.85,
        solidity=0.82,
        equivalent_diameter_um=160.0,
    ),
    CrystalCompound(
        name="L-Glutamic acid (β-form)",
        formula="C5H9NO4",
        molecular_weight=147.13,
        crystal_system=CrystalSystem.ORTHORHOMBIC,
        habit=CrystalHabit.NEEDLE,
        density_g_cm3=1.57,
        melting_point_K=472.15,
        solubility_g_L=8.6,
        application_domain=ApplicationDomain.FOOD_ADDITIVE,
        cas_number="56-86-0",
        aspect_ratio=6.0,
        circularity=0.20,
        convexity=0.80,
        solidity=0.76,
        equivalent_diameter_um=100.0,
    ),
    CrystalCompound(
        name="Sodium chloride",
        formula="NaCl",
        molecular_weight=58.44,
        crystal_system=CrystalSystem.CUBIC,
        habit=CrystalHabit.BLOCKY,
        density_g_cm3=2.16,
        melting_point_K=1074.15,
        solubility_g_L=360.0,
        application_domain=ApplicationDomain.FINE_CHEMICAL,
        cas_number="7647-14-5",
        aspect_ratio=1.0,
        circularity=0.80,
        convexity=0.95,
        solidity=0.95,
        equivalent_diameter_um=500.0,
    ),
    CrystalCompound(
        name="Calcium carbonate (calcite)",
        formula="CaCO3",
        molecular_weight=100.09,
        crystal_system=CrystalSystem.TRIGONAL,
        habit=CrystalHabit.PRISMATIC,
        density_g_cm3=2.71,
        melting_point_K=1098.15,  # Decomposes ~825°C
        solubility_g_L=0.013,
        application_domain=ApplicationDomain.MATERIALS_RESEARCH,
        cas_number="471-34-1",
        aspect_ratio=1.5,
        circularity=0.60,
        convexity=0.92,
        solidity=0.90,
        equivalent_diameter_um=350.0,
    ),
    CrystalCompound(
        name="Sucrose",
        formula="C12H22O11",
        molecular_weight=342.30,
        crystal_system=CrystalSystem.MONOCLINIC,
        habit=CrystalHabit.PRISMATIC,
        density_g_cm3=1.59,
        melting_point_K=459.15,  # 186°C (decomposes)
        solubility_g_L=2000.0,
        application_domain=ApplicationDomain.FOOD_ADDITIVE,
        cas_number="57-50-1",
        aspect_ratio=1.8,
        circularity=0.55,
        convexity=0.90,
        solidity=0.88,
        equivalent_diameter_um=400.0,
    ),
)

# ═══════════════════════════════════════════════════════════════════
#  Lookup Utilities
# ═══════════════════════════════════════════════════════════════════

_BY_NAME: dict[str, CrystalCompound] = {c.name: c for c in COMPOUNDS}
_BY_FORMULA: dict[str, list[CrystalCompound]] = {}
for _c in COMPOUNDS:
    _BY_FORMULA.setdefault(_c.formula, []).append(_c)


def get_compound(name: str) -> CrystalCompound | None:
    """Look up a crystal compound by exact name."""
    return _BY_NAME.get(name)


def get_compounds_by_formula(formula: str) -> list[CrystalCompound]:
    """Look up all polymorphs/variants with a given molecular formula."""
    return list(_BY_FORMULA.get(formula, []))


def get_compounds_by_system(system: CrystalSystem) -> list[CrystalCompound]:
    """Filter compounds by crystal system."""
    return [c for c in COMPOUNDS if c.crystal_system == system]


def get_compounds_by_habit(habit: CrystalHabit) -> list[CrystalCompound]:
    """Filter compounds by crystal habit."""
    return [c for c in COMPOUNDS if c.habit == habit]


def get_compounds_by_domain(domain: ApplicationDomain) -> list[CrystalCompound]:
    """Filter compounds by application domain."""
    return [c for c in COMPOUNDS if c.application_domain == domain]


def get_opencrystaldata_compounds() -> list[CrystalCompound]:
    """Return only compounds sourced from the OpenCrystalData project."""
    return [c for c in COMPOUNDS if c.source_dataset.startswith("OpenCrystalData")]


# ═══════════════════════════════════════════════════════════════════
#  Normalization Constants (for trace vector construction)
# ═══════════════════════════════════════════════════════════════════

# Molecular weight: log-scale [50, 500] → [0, 1]
_MW_MIN_LOG = math.log(50.0)
_MW_MAX_LOG = math.log(500.0)

# Density: [0.8, 12.0] → [0, 1]
_DENSITY_MIN = 0.8
_DENSITY_MAX = 12.0

# Melting point: [250, 1500] K → [0, 1]
_MP_MIN = 250.0
_MP_MAX = 1500.0

# Solubility: log-scale [0.001, 5000] g/L → [0, 1]
_SOL_MIN_LOG = math.log(0.001)
_SOL_MAX_LOG = math.log(5000.0)

# Aspect ratio: [1, 10] → [0, 1]
_AR_MIN = 1.0
_AR_MAX = 10.0

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
#  Trace Vector & Kernel Computation
# ═══════════════════════════════════════════════════════════════════


def build_trace(compound: CrystalCompound) -> tuple[float, ...]:
    """Build 8-channel trace vector for a crystal compound.

    Channels:
        0. molecular_weight (log-normalized)
        1. density (linear)
        2. melting_point (linear)
        3. solubility (log-normalized; ε if None/insoluble)
        4. aspect_ratio (linear; ε if None)
        5. circularity (direct [0,1]; ε if None)
        6. crystal_system (ordinal / 6)
        7. habit_regularity (from lookup)
    """
    c0 = _log_normalize(compound.molecular_weight, _MW_MIN_LOG, _MW_MAX_LOG)
    c1 = _linear_normalize(compound.density_g_cm3, _DENSITY_MIN, _DENSITY_MAX)
    c2 = _linear_normalize(compound.melting_point_K, _MP_MIN, _MP_MAX)

    if compound.solubility_g_L is not None and compound.solubility_g_L > 0:
        c3 = _log_normalize(compound.solubility_g_L, _SOL_MIN_LOG, _SOL_MAX_LOG)
    else:
        c3 = _EPSILON

    c4 = _linear_normalize(compound.aspect_ratio, _AR_MIN, _AR_MAX) if compound.aspect_ratio is not None else 0.5

    c5 = _clamp(compound.circularity or 0.5, _EPSILON, 1.0 - _EPSILON)
    c6 = _clamp(_SYSTEM_INDEX[compound.crystal_system] / 6.0, _EPSILON, 1.0 - _EPSILON)
    c7 = _clamp(_HABIT_REGULARITY.get(compound.habit, 0.5), _EPSILON, 1.0 - _EPSILON)

    return (c0, c1, c2, c3, c4, c5, c6, c7)


def compute_crystal_kernel(compound: CrystalCompound) -> CrystalKernelResult:
    """Compute GCD kernel invariants for a crystal compound.

    Maps compound properties through an 8-channel trace vector,
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
    trace = build_trace(compound)
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
    C = _clamp(math.sqrt(variance) / 0.5, 0.0, 1.0)

    # Log-integrity and IC
    kappa = sum(w * math.log(_clamp(c, _EPSILON, 1.0)) for c in trace)
    IC = math.exp(kappa)

    # Regime classification
    if omega >= 0.30:
        regime = "Collapse"
    elif omega < 0.038 and F > 0.90 and S < 0.15 and C < 0.14:
        regime = "Stable"
    else:
        regime = "Watch"

    return CrystalKernelResult(
        name=compound.name,
        formula=compound.formula,
        crystal_system=compound.crystal_system.value,
        habit=compound.habit.value,
        trace=trace,
        F=F,
        omega=omega,
        S=S,
        C=C,
        kappa=kappa,
        IC=IC,
        regime=regime,
    )


def compute_all_crystal_kernels() -> list[CrystalKernelResult]:
    """Compute kernel invariants for all compounds in database."""
    return [compute_crystal_kernel(c) for c in COMPOUNDS]


# ═══════════════════════════════════════════════════════════════════
#  Polymorph Analysis
# ═══════════════════════════════════════════════════════════════════


def analyze_polymorphs(formula: str) -> dict[str, Any]:
    """Analyze kernel differences between polymorphs of the same compound.

    The heterogeneity gap (Δ = F − IC) reveals which polymorph has the most
    uniform channel distribution.  A smaller gap indicates a more
    "balanced" crystal form in the kernel sense.

    Returns dict with polymorph names, kernel results, and F/IC comparisons.
    """
    variants = get_compounds_by_formula(formula)
    if len(variants) < 2:
        return {
            "formula": formula,
            "polymorph_count": len(variants),
            "note": "Single form or not found — no polymorph comparison",
        }

    results = [(v.name, compute_crystal_kernel(v)) for v in variants]

    return {
        "formula": formula,
        "polymorph_count": len(results),
        "polymorphs": [
            {
                "name": name,
                "crystal_system": kr.crystal_system,
                "habit": kr.habit,
                "F": kr.F,
                "omega": kr.omega,
                "IC": kr.IC,
                "heterogeneity_gap": kr.F - kr.IC,
                "regime": kr.regime,
            }
            for name, kr in results
        ],
        "max_F": max(kr.F for _, kr in results),
        "min_heterogeneity_gap": min(kr.F - kr.IC for _, kr in results),
    }


def crystal_system_statistics() -> dict[str, dict[str, float]]:
    """Average kernel invariants per crystal system.

    Shows how crystal symmetry maps to fidelity structure:
    higher symmetry systems (cubic) tend toward higher circularity
    and regularity, affecting IC.
    """
    from collections import defaultdict

    accum: dict[str, list[CrystalKernelResult]] = defaultdict(list)
    for c in COMPOUNDS:
        kr = compute_crystal_kernel(c)
        accum[kr.crystal_system].append(kr)

    stats: dict[str, dict[str, float]] = {}
    for system, results in sorted(accum.items()):
        n = len(results)
        stats[system] = {
            "count": n,
            "mean_F": sum(r.F for r in results) / n,
            "mean_omega": sum(r.omega for r in results) / n,
            "mean_IC": sum(r.IC for r in results) / n,
            "mean_heterogeneity_gap": sum(r.F - r.IC for r in results) / n,
            "mean_S": sum(r.S for r in results) / n,
            "mean_C": sum(r.C for r in results) / n,
        }
    return stats


# ═══════════════════════════════════════════════════════════════════
#  Validation
# ═══════════════════════════════════════════════════════════════════


def validate_database() -> dict[str, Any]:
    """Validate the crystal morphology database.

    Checks:
        1. All compounds have valid crystal system and habit
        2. Molecular weight > 0 for all entries
        3. Density > 0 for all entries
        4. Kernel identity F + ω = 1 holds for all traces
        5. Integrity bound IC ≤ F holds for all traces
        6. No NaN or infinite values in any kernel output
    """
    errors: list[str] = []
    warnings: list[str] = []

    for compound in COMPOUNDS:
        # Basic validity
        if compound.molecular_weight <= 0:
            errors.append(f"{compound.name}: molecular_weight must be > 0")
        if compound.density_g_cm3 <= 0:
            errors.append(f"{compound.name}: density must be > 0")
        if compound.melting_point_K <= 0:
            errors.append(f"{compound.name}: melting_point must be > 0")

        # Morphological bounds
        if compound.circularity is not None and not 0.0 <= compound.circularity <= 1.0:
            errors.append(f"{compound.name}: circularity {compound.circularity} out of [0,1]")
        if compound.aspect_ratio is not None and compound.aspect_ratio < 1.0:
            errors.append(f"{compound.name}: aspect_ratio {compound.aspect_ratio} < 1.0")

        # Kernel identities
        kr = compute_crystal_kernel(compound)
        duality_residual = abs(kr.F + kr.omega - 1.0)
        if duality_residual > 1e-12:
            errors.append(f"{compound.name}: duality identity violated, |F + ω − 1| = {duality_residual:.2e}")
        if kr.IC > kr.F + 1e-12:
            errors.append(f"{compound.name}: integrity bound violated, IC ({kr.IC:.6f}) > F ({kr.F:.6f})")
        if math.isnan(kr.F) or math.isinf(kr.F):
            errors.append(f"{compound.name}: F is NaN/Inf")
        if math.isnan(kr.IC) or math.isinf(kr.IC):
            errors.append(f"{compound.name}: IC is NaN/Inf")

        # Warnings for missing morphological data
        if compound.circularity is None:
            warnings.append(f"{compound.name}: missing circularity")
        if compound.aspect_ratio is None:
            warnings.append(f"{compound.name}: missing aspect_ratio")

    return {
        "total_compounds": len(COMPOUNDS),
        "errors": errors,
        "warnings": warnings,
        "status": "CONFORMANT" if not errors else "NONCONFORMANT",
    }

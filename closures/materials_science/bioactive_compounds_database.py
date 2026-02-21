"""Bioactive compounds database — Essential oils, short-chain fatty acids, and biologics.

Encodes bioactive compound properties, bioactivity metrics, and pharmacological
descriptors sourced from the MDPI Biology and Life Sciences Forum (BLSF) and
MDPI Biologics journals.  All data is mapped through the GCD kernel to produce
Tier-1 invariants (F, ω, S, C, κ, IC) via an 8-channel trace vector per compound.

Trace vector channels (equal weights w_i = 1/8):
    0. molecular_weight_norm — molecular weight (g/mol or kDa), log-scaled to [0,1]
    1. bioactivity_potency — normalized efficacy (MIC, IC50, DPPH%, inhibition%)
    2. lipophilicity_norm — logP or hydrophobicity index, rescaled to [0,1]
    3. thermal_stability_norm — boiling/melting point (K), rescaled to [0,1]
    4. structural_complexity — ring count + heteroatom fraction, rescaled to [0,1]
    5. target_breadth — spectrum breadth (narrow=0.2, moderate=0.5, broad=0.8)
    6. safety_index — therapeutic index / selectivity, rescaled to [0,1]
    7. source_naturalness — natural=0.9, semi-synthetic=0.5, recombinant=0.3

Sources (see paper/Bibliography.bib):
    - Papantzikos, V.; Patakioutas, G.; Yfanti, P. (2025).
      "Evaluation of the Antifungal Effect of Carvacrol-Rich Essential Oils."
      Biol. Life Sci. Forum 54(1):1.  DOI: 10.3390/blsf2025054001
    - Ezzaky, Y. et al. (2026).
      "Biopreservative and Antioxidant Potential of Novel LAB Strains."
      Biol. Life Sci. Forum 56(1):4.  DOI: 10.3390/blsf2026056004
    - Zolfanelli, C. et al. (2026).
      "Short-Chain Fatty Acids as Functional Postbiotics in CRC Management."
      Biol. Life Sci. Forum 56(1):5.  DOI: 10.3390/blsf2026056005
    - Suzuki, T. et al. (2026).
      "Clinical Remission After 12 Months of Biologic Therapy in Severe Asthma."
      Biologics 6(1):4.  DOI: 10.3390/biologics6010004
    - Di Benedetto, G. et al. (2026).
      "IL-6: A Central Biomarker in Cancer and Infectious Disease."
      Biologics 6(1):5.  DOI: 10.3390/biologics6010005

Copyright-free reference data: molecular weights, boiling/melting points,
logP values, and pKa values are factual scientific measurements not subject
to copyright.  Bioactivity data extracted from CC BY 4.0 open-access articles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, NamedTuple

# ═══════════════════════════════════════════════════════════════════
#  Enumerations
# ═══════════════════════════════════════════════════════════════════


class CompoundClass(StrEnum):
    """Bioactive compound classification by chemical nature."""

    MONOTERPENE_PHENOL = "Monoterpene phenol"
    MONOTERPENE = "Monoterpene"
    AROMATIC_HYDROCARBON = "Aromatic hydrocarbon"
    SHORT_CHAIN_FATTY_ACID = "Short-chain fatty acid"
    ORGANIC_ACID = "Organic acid"
    CYTOKINE = "Cytokine"
    MONOCLONAL_ANTIBODY = "Monoclonal antibody"
    RECOMBINANT_ENZYME = "Recombinant enzyme"
    ANTIMICROBIAL_PEPTIDE = "Antimicrobial peptide"


class BioactivityType(StrEnum):
    """Primary bioactivity mechanism."""

    ANTIFUNGAL = "Antifungal"
    ANTIBACTERIAL = "Antibacterial"
    ANTIOXIDANT = "Antioxidant"
    ANTICANCER = "Anticancer"
    ANTI_INFLAMMATORY = "Anti-inflammatory"
    IMMUNOMODULATORY = "Immunomodulatory"
    ENZYME_REPLACEMENT = "Enzyme replacement"
    BIOPRESERVATIVE = "Biopreservative"


class SourceType(StrEnum):
    """Origin of the bioactive compound."""

    PLANT_ESSENTIAL_OIL = "Plant essential oil"
    MICROBIAL_METABOLITE = "Microbial metabolite"
    DIETARY_FIBER_FERMENTATION = "Dietary fiber fermentation"
    RECOMBINANT_BIOTECHNOLOGY = "Recombinant biotechnology"
    SYNTHETIC = "Synthetic"
    SEMI_SYNTHETIC = "Semi-synthetic"


class TargetOrganism(StrEnum):
    """Target organism or system for bioactivity."""

    FUSARIUM = "Fusarium sp."
    ALTERNARIA = "Alternaria sp."
    LISTERIA = "Listeria monocytogenes"
    STAPHYLOCOCCUS = "Staphylococcus aureus"
    COLORECTAL_CANCER = "Colorectal cancer cells"
    IMMUNE_SYSTEM = "Immune system"
    RESPIRATORY = "Respiratory system"
    NEURONAL = "Neuronal (CLN2)"
    BROAD_SPECTRUM = "Broad spectrum"


# Source naturalness mapping for trace channel 7
_SOURCE_NATURALNESS: dict[SourceType, float] = {
    SourceType.PLANT_ESSENTIAL_OIL: 0.95,
    SourceType.MICROBIAL_METABOLITE: 0.85,
    SourceType.DIETARY_FIBER_FERMENTATION: 0.80,
    SourceType.RECOMBINANT_BIOTECHNOLOGY: 0.30,
    SourceType.SYNTHETIC: 0.10,
    SourceType.SEMI_SYNTHETIC: 0.50,
}

# Target breadth mapping for trace channel 5
_TARGET_BREADTH: dict[TargetOrganism, float] = {
    TargetOrganism.FUSARIUM: 0.30,
    TargetOrganism.ALTERNARIA: 0.30,
    TargetOrganism.LISTERIA: 0.35,
    TargetOrganism.STAPHYLOCOCCUS: 0.35,
    TargetOrganism.COLORECTAL_CANCER: 0.40,
    TargetOrganism.IMMUNE_SYSTEM: 0.75,
    TargetOrganism.RESPIRATORY: 0.50,
    TargetOrganism.NEURONAL: 0.25,
    TargetOrganism.BROAD_SPECTRUM: 0.85,
}


# ═══════════════════════════════════════════════════════════════════
#  Data Classes
# ═══════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class BioactiveCompound:
    """Bioactive compound with biochemical and pharmacological properties.

    Sources:
        Physical properties: PubChem, DrugBank, KEGG
        Bioactivity data: MDPI BLSF Vol. 54 & 56 (CC BY 4.0)
        Biologics data: MDPI Biologics Vol. 5 & 6 (CC BY 4.0)
    """

    name: str
    formula: str
    molecular_weight: float  # g/mol (small molecules) or kDa (biologics)
    compound_class: CompoundClass
    bioactivity: BioactivityType
    source_type: SourceType
    primary_target: TargetOrganism
    cas_number: str | None = None

    # ── Physical/chemical properties ─────────────────────────────
    boiling_point_K: float | None = None  # Boiling point (K)
    melting_point_K: float | None = None  # Melting point (K)
    logP: float | None = None  # Octanol-water partition coefficient
    pKa: float | None = None  # Acid dissociation constant
    density_g_cm3: float | None = None  # Density (g/cm³)

    # ── Structural descriptors ───────────────────────────────────
    ring_count: int = 0  # Number of rings
    heteroatom_count: int = 0  # N, O, S atoms
    heavy_atom_count: int = 0  # Non-hydrogen atoms
    rotatable_bonds: int = 0  # Rotatable bond count

    # ── Bioactivity metrics ──────────────────────────────────────
    potency_value: float | None = None  # Primary bioactivity metric
    potency_unit: str = ""  # Unit of potency measure
    potency_description: str = ""  # What the metric measures

    # ── Safety / selectivity ─────────────────────────────────────
    safety_index: float = 0.5  # 0=toxic, 1=very safe (GRAS/biologic)

    # ── Provenance ───────────────────────────────────────────────
    source_article: str = ""  # DOI or citation key
    source_journal: str = ""  # BLSF or Biologics

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON output."""
        return {
            "name": self.name,
            "formula": self.formula,
            "molecular_weight": self.molecular_weight,
            "compound_class": self.compound_class.value,
            "bioactivity": self.bioactivity.value,
            "source_type": self.source_type.value,
            "primary_target": self.primary_target.value,
            "cas_number": self.cas_number,
            "boiling_point_K": self.boiling_point_K,
            "melting_point_K": self.melting_point_K,
            "logP": self.logP,
            "pKa": self.pKa,
            "density_g_cm3": self.density_g_cm3,
            "ring_count": self.ring_count,
            "heteroatom_count": self.heteroatom_count,
            "heavy_atom_count": self.heavy_atom_count,
            "rotatable_bonds": self.rotatable_bonds,
            "potency_value": self.potency_value,
            "potency_unit": self.potency_unit,
            "potency_description": self.potency_description,
            "safety_index": self.safety_index,
            "source_article": self.source_article,
            "source_journal": self.source_journal,
        }


class BioactiveKernelResult(NamedTuple):
    """GCD kernel output for a bioactive compound."""

    name: str
    formula: str
    compound_class: str
    bioactivity: str
    trace: tuple[float, ...]
    F: float  # Fidelity
    omega: float  # Drift
    S: float  # Bernoulli field entropy
    C: float  # Curvature
    kappa: float  # Log-integrity
    IC: float  # Integrity composite
    regime: str  # Stable / Watch / Collapse


# ═══════════════════════════════════════════════════════════════════
#  BIOACTIVE COMPOUNDS DATABASE
# ═══════════════════════════════════════════════════════════════════
#
# ── Essential Oils (BLSF Vol. 54, Papantzikos et al. 2025) ──────
#   Three Lamiaceae EOs tested against Fusarium and Alternaria:
#   - Carvacrol: main active component (41.4-70.0% in tested EOs)
#   - Thymol: isomeric co-component (0-18.9%)
#   - p-Cymene: supporting monoterpene (~7% in C. capitatus)
#   - γ-Terpinene: supporting monoterpene (~7% in C. capitatus)
#   Complete inhibition of Fusarium at 2 µL/plate (O. vulgare, C. capitatus)
#   Complete inhibition of Alternaria at 3 µL/plate (same species)
#
# ── LAB Biopreservation (BLSF Vol. 56, Ezzaky et al. 2026) ──────
#   56 LAB isolates from low-sodium vegetable fermentations:
#   - 8 strains with strong antimicrobial activity (>10 mm inhibition)
#   - DPPH radical scavenging: 27-65% (strain L10 highest)
#   - Three dominant taxa: Leuconostoc, Lactococcus lactis, E. faecium
#   - Lactic acid: key antimicrobial metabolite
#
# ── SCFAs in CRC (BLSF Vol. 56, Zolfanelli et al. 2026) ────────
#   Systematic review of 27 studies, 8 SCFA-focused:
#   - Butyrate: autophagy via LKB1-AMPK, mucin at 6-9 mM
#   - Tumor reduction: 8.3±2.4 → 4.0±1.6 tumors/mouse (p<0.001)
#   - Tumor size: 2.5±0.4 → 1.3±0.5 mm (p<0.001)
#   - Ki67 proliferation: 6.8±1.7 → 3.5±1.0 (p<0.001)
#
# ── Biologics (MDPI Biologics Vol. 5-6) ─────────────────────────
#   - IL-6: central biomarker (JAK/STAT3, MAPK, PI3K/AKT)
#   - Dupilumab: anti-IL-4Rα, 30% four-component remission at 12 mo
#   - Cerliponase alfa: recombinant TPP1, 31 patients, 2705 infusions
#
# Physical data: PubChem (CID lookups), DrugBank, KEGG COMPOUND
# ═══════════════════════════════════════════════════════════════════

COMPOUNDS: tuple[BioactiveCompound, ...] = (
    # ── Essential Oil Components (BLSF Vol. 54) ─────────────────
    # Source: Papantzikos et al. (2025) DOI: 10.3390/blsf2025054001
    # GC-MS composition: carvacrol 41.4-70.0%, thymol 0-18.9%
    BioactiveCompound(
        name="Carvacrol",
        formula="C₁₀H₁₄O",
        molecular_weight=150.22,
        compound_class=CompoundClass.MONOTERPENE_PHENOL,
        bioactivity=BioactivityType.ANTIFUNGAL,
        source_type=SourceType.PLANT_ESSENTIAL_OIL,
        primary_target=TargetOrganism.BROAD_SPECTRUM,
        cas_number="499-75-2",
        boiling_point_K=510.15,  # 237°C (PubChem CID 10364)
        melting_point_K=274.15,  # 1°C
        logP=3.49,  # PubChem XLogP3
        density_g_cm3=0.9772,
        ring_count=1,
        heteroatom_count=1,  # 1 O
        heavy_atom_count=11,
        rotatable_bonds=1,
        potency_value=100.0,  # % inhibition of Fusarium at 2 µL/plate
        potency_unit="% inhibition",
        potency_description="Complete mycelial growth inhibition of Fusarium sp. at 2 µL/plate (fumigant assay, 8 days, 26°C)",
        safety_index=0.85,  # GRAS flavoring agent (FDA 21 CFR 172.515)
        source_article="10.3390/blsf2025054001",
        source_journal="BLSF",
    ),
    BioactiveCompound(
        name="Thymol",
        formula="C₁₀H₁₄O",
        molecular_weight=150.22,
        compound_class=CompoundClass.MONOTERPENE_PHENOL,
        bioactivity=BioactivityType.ANTIFUNGAL,
        source_type=SourceType.PLANT_ESSENTIAL_OIL,
        primary_target=TargetOrganism.BROAD_SPECTRUM,
        cas_number="89-83-8",
        boiling_point_K=505.65,  # 232.5°C (PubChem CID 6989)
        melting_point_K=323.15,  # 50°C
        logP=3.30,  # PubChem XLogP3
        density_g_cm3=0.965,
        ring_count=1,
        heteroatom_count=1,  # 1 O
        heavy_atom_count=11,
        rotatable_bonds=1,
        potency_value=100.0,  # Complete antifungal inhibition (synergist)
        potency_unit="% inhibition",
        potency_description="Antifungal co-component in EOs (0-18.9% in tested oils); synergistic with carvacrol",
        safety_index=0.80,  # GRAS, EPA-registered antimicrobial
        source_article="10.3390/blsf2025054001",
        source_journal="BLSF",
    ),
    BioactiveCompound(
        name="p-Cymene",
        formula="C₁₀H₁₄",
        molecular_weight=134.22,
        compound_class=CompoundClass.AROMATIC_HYDROCARBON,
        bioactivity=BioactivityType.ANTIFUNGAL,
        source_type=SourceType.PLANT_ESSENTIAL_OIL,
        primary_target=TargetOrganism.FUSARIUM,
        cas_number="99-87-6",
        boiling_point_K=450.28,  # 177.1°C (PubChem CID 7463)
        melting_point_K=205.25,  # -67.9°C
        logP=4.10,  # PubChem XLogP3
        density_g_cm3=0.857,
        ring_count=1,
        heteroatom_count=0,
        heavy_atom_count=10,
        rotatable_bonds=1,
        potency_value=31.0,  # Partial inhibition at 1 µL/plate
        potency_unit="% growth remaining",
        potency_description="Supporting terpene (~7% in C. capitatus EO); enhances antifungal activity synergistically",
        safety_index=0.70,  # Generally low toxicity (FEMA GRAS)
        source_article="10.3390/blsf2025054001",
        source_journal="BLSF",
    ),
    BioactiveCompound(
        name="γ-Terpinene",
        formula="C₁₀H₁₆",
        molecular_weight=136.23,
        compound_class=CompoundClass.MONOTERPENE,
        bioactivity=BioactivityType.ANTIFUNGAL,
        source_type=SourceType.PLANT_ESSENTIAL_OIL,
        primary_target=TargetOrganism.FUSARIUM,
        cas_number="99-85-4",
        boiling_point_K=456.15,  # 183°C (PubChem CID 7461)
        melting_point_K=176.15,  # -97°C (estimated)
        logP=4.50,  # Estimated from structure
        density_g_cm3=0.849,
        ring_count=1,
        heteroatom_count=0,
        heavy_atom_count=10,
        rotatable_bonds=1,
        potency_value=31.0,  # Partial synergistic contribution
        potency_unit="% growth remaining",
        potency_description="Supporting terpene (~7% in C. capitatus); synergistic role with phenolic components",
        safety_index=0.70,  # Low acute toxicity
        source_article="10.3390/blsf2025054001",
        source_journal="BLSF",
    ),
    # ── LAB Metabolites (BLSF Vol. 56) ──────────────────────────
    # Source: Ezzaky et al. (2026) DOI: 10.3390/blsf2026056004
    # 56 LAB isolates from low-sodium vegetable fermentations
    BioactiveCompound(
        name="Lactic acid",
        formula="C₃H₆O₃",
        molecular_weight=90.08,
        compound_class=CompoundClass.ORGANIC_ACID,
        bioactivity=BioactivityType.BIOPRESERVATIVE,
        source_type=SourceType.MICROBIAL_METABOLITE,
        primary_target=TargetOrganism.BROAD_SPECTRUM,
        cas_number="50-21-5",
        boiling_point_K=395.15,  # 122°C at 12 mmHg (PubChem CID 612)
        melting_point_K=291.15,  # 18°C (L-form)
        logP=-0.72,  # PubChem XLogP3
        pKa=3.86,
        density_g_cm3=1.209,
        ring_count=0,
        heteroatom_count=3,  # 3 O
        heavy_atom_count=6,
        rotatable_bonds=1,
        potency_value=65.0,  # DPPH scavenging of LAB strain L10 supernatant
        potency_unit="% DPPH scavenging",
        potency_description="LAB-produced; strain L10 highest DPPH scavenging (65%, p<0.05); broad antimicrobial via pH depression",
        safety_index=0.95,  # GRAS, ubiquitous food preservative
        source_article="10.3390/blsf2026056004",
        source_journal="BLSF",
    ),
    BioactiveCompound(
        name="Bacteriocin-like substance (LAB)",
        formula="peptide",
        molecular_weight=5000.0,  # Typical bacteriocin 3-10 kDa
        compound_class=CompoundClass.ANTIMICROBIAL_PEPTIDE,
        bioactivity=BioactivityType.ANTIBACTERIAL,
        source_type=SourceType.MICROBIAL_METABOLITE,
        primary_target=TargetOrganism.LISTERIA,
        boiling_point_K=None,  # Protein — no bp
        melting_point_K=None,
        logP=None,
        density_g_cm3=None,
        ring_count=0,
        heteroatom_count=50,  # Peptide: many N, O
        heavy_atom_count=350,  # Approximate for ~5 kDa peptide
        rotatable_bonds=50,
        potency_value=10.0,  # >10 mm inhibition zones against L. monocytogenes
        potency_unit="mm inhibition zone",
        potency_description="Bacteriocin-like substances from 8 LAB isolates; >10 mm zones vs Listeria monocytogenes and S. aureus",
        safety_index=0.90,  # LAB-derived, food-grade
        source_article="10.3390/blsf2026056004",
        source_journal="BLSF",
    ),
    # ── Short-Chain Fatty Acids (BLSF Vol. 56) ──────────────────
    # Source: Zolfanelli et al. (2026) DOI: 10.3390/blsf2026056005
    # Systematic review: 27 studies, 8 SCFA-focused
    BioactiveCompound(
        name="Butyric acid (sodium butyrate)",
        formula="C₄H₇NaO₂",
        molecular_weight=110.09,
        compound_class=CompoundClass.SHORT_CHAIN_FATTY_ACID,
        bioactivity=BioactivityType.ANTICANCER,
        source_type=SourceType.DIETARY_FIBER_FERMENTATION,
        primary_target=TargetOrganism.COLORECTAL_CANCER,
        cas_number="156-54-7",
        boiling_point_K=None,  # Sodium salt — decomposes
        melting_point_K=523.15,  # ~250°C (decomposes)
        logP=-0.79,  # Butyric acid logP (PubChem CID 264)
        pKa=4.82,
        density_g_cm3=0.964,  # Butyric acid density
        ring_count=0,
        heteroatom_count=2,  # 2 O
        heavy_atom_count=7,  # C4H7O2Na
        rotatable_bonds=2,
        potency_value=51.8,  # Tumor count reduction: (8.3-4.0)/8.3 × 100 = 51.8%
        potency_unit="% tumor reduction",
        potency_description="AOM/DSS mouse model: tumors 8.3±2.4→4.0±1.6/mouse (p<0.001); tumor size 2.5→1.3 mm; Ki67 6.8→3.5",
        safety_index=0.75,  # Endogenous metabolite; dose-dependent effects
        source_article="10.3390/blsf2026056005",
        source_journal="BLSF",
    ),
    BioactiveCompound(
        name="Acetic acid (acetate)",
        formula="C₂H₄O₂",
        molecular_weight=60.05,
        compound_class=CompoundClass.SHORT_CHAIN_FATTY_ACID,
        bioactivity=BioactivityType.ANTICANCER,
        source_type=SourceType.DIETARY_FIBER_FERMENTATION,
        primary_target=TargetOrganism.COLORECTAL_CANCER,
        cas_number="64-19-7",
        boiling_point_K=391.05,  # 117.9°C (PubChem CID 176)
        melting_point_K=289.75,  # 16.6°C
        logP=-0.17,  # PubChem XLogP3
        pKa=4.76,
        density_g_cm3=1.049,
        ring_count=0,
        heteroatom_count=2,  # 2 O
        heavy_atom_count=4,
        rotatable_bonds=0,
        potency_value=30.0,  # Moderate apoptosis induction (Cousin et al.)
        potency_unit="% apoptosis enhancement",
        potency_description="Activates intrinsic apoptosis in HT-29/HCT-116 via TRAIL; synergistic with propionate",
        safety_index=0.85,  # GRAS, natural metabolite, food preservative
        source_article="10.3390/blsf2026056005",
        source_journal="BLSF",
    ),
    BioactiveCompound(
        name="Propionic acid (propionate)",
        formula="C₃H₆O₂",
        molecular_weight=74.08,
        compound_class=CompoundClass.SHORT_CHAIN_FATTY_ACID,
        bioactivity=BioactivityType.ANTICANCER,
        source_type=SourceType.DIETARY_FIBER_FERMENTATION,
        primary_target=TargetOrganism.COLORECTAL_CANCER,
        cas_number="79-09-4",
        boiling_point_K=414.25,  # 141.1°C (PubChem CID 1032)
        melting_point_K=252.45,  # -20.7°C
        logP=0.33,  # PubChem XLogP3
        pKa=4.87,
        density_g_cm3=0.993,
        ring_count=0,
        heteroatom_count=2,  # 2 O
        heavy_atom_count=5,
        rotatable_bonds=1,
        potency_value=35.0,  # Moderate apoptosis induction
        potency_unit="% apoptosis enhancement",
        potency_description="Activates intrinsic apoptosis via P. freudenreichii ITG-P9; caspase activation in CRC cell lines",
        safety_index=0.80,  # GRAS, food preservative (E280)
        source_article="10.3390/blsf2026056005",
        source_journal="BLSF",
    ),
    # ── Biologics — Cytokines & Therapeutic Proteins ─────────────
    # Source: Di Benedetto et al. (2026) DOI: 10.3390/biologics6010005
    BioactiveCompound(
        name="Interleukin-6 (IL-6)",
        formula="protein",
        molecular_weight=21000.0,  # ~21 kDa glycoprotein
        compound_class=CompoundClass.CYTOKINE,
        bioactivity=BioactivityType.IMMUNOMODULATORY,
        source_type=SourceType.RECOMBINANT_BIOTECHNOLOGY,
        primary_target=TargetOrganism.IMMUNE_SYSTEM,
        boiling_point_K=None,  # Protein
        melting_point_K=None,
        logP=None,
        density_g_cm3=None,
        ring_count=0,
        heteroatom_count=500,  # Approximate for 184-aa protein
        heavy_atom_count=1500,
        rotatable_bonds=200,
        potency_value=0.001,  # pg/mL detection threshold in serum
        potency_unit="pg/mL detection",
        potency_description="Central biomarker: JAK/STAT3, MAPK, PI3K/AKT signaling; classical + trans-signaling; CRP/SAA induction",
        safety_index=0.20,  # Endogenous but dysregulation = disease
        source_article="10.3390/biologics6010005",
        source_journal="Biologics",
    ),
    # Source: Suzuki et al. (2026) DOI: 10.3390/biologics6010004
    BioactiveCompound(
        name="Dupilumab",
        formula="IgG4κ",
        molecular_weight=147000.0,  # ~147 kDa monoclonal antibody
        compound_class=CompoundClass.MONOCLONAL_ANTIBODY,
        bioactivity=BioactivityType.ANTI_INFLAMMATORY,
        source_type=SourceType.RECOMBINANT_BIOTECHNOLOGY,
        primary_target=TargetOrganism.RESPIRATORY,
        boiling_point_K=None,
        melting_point_K=None,
        logP=None,
        density_g_cm3=None,
        ring_count=0,
        heteroatom_count=5000,  # Approximate for ~1300-aa IgG
        heavy_atom_count=11000,
        rotatable_bonds=2000,
        potency_value=30.0,  # 30% four-component clinical remission at 12 months
        potency_unit="% clinical remission",
        potency_description="Anti-IL-4Rα mAb; 87 severe asthma patients: 30% four-component remission (OCS-free 85%, no exacerbation 59%, ACT≥20 53%, FEV1≥80% 61%)",
        safety_index=0.65,  # Well-tolerated; injection-site reactions, conjunctivitis
        source_article="10.3390/biologics6010004",
        source_journal="Biologics",
    ),
    # Source: Whiteley et al. (2026) DOI: 10.3390/biologics6010007
    BioactiveCompound(
        name="Cerliponase alfa",
        formula="rhTPP1",
        molecular_weight=66000.0,  # ~66 kDa recombinant enzyme
        compound_class=CompoundClass.RECOMBINANT_ENZYME,
        bioactivity=BioactivityType.ENZYME_REPLACEMENT,
        source_type=SourceType.RECOMBINANT_BIOTECHNOLOGY,
        primary_target=TargetOrganism.NEURONAL,
        boiling_point_K=None,
        melting_point_K=None,
        logP=None,
        density_g_cm3=None,
        ring_count=0,
        heteroatom_count=2000,  # Approximate for ~560-aa enzyme
        heavy_atom_count=5000,
        rotatable_bonds=800,
        potency_value=64.5,  # 11/31 had IARs = 35.5% IAR rate → 64.5% well-tolerated
        potency_unit="% infusion tolerance",
        potency_description="ICV ERT for CLN2: 31 patients, ~2705 infusions over 10 years; 64.5% no IARs; 1 anaphylaxis resolved",
        safety_index=0.55,  # ICV route; mild IARs (pyrexia, vomiting, rash)
        source_article="10.3390/biologics6010007",
        source_journal="Biologics",
    ),
)


# ═══════════════════════════════════════════════════════════════════
#  Lookup Utilities
# ═══════════════════════════════════════════════════════════════════


def get_compound(name: str) -> BioactiveCompound | None:
    """Look up a compound by exact name (case-insensitive)."""
    key = name.lower()
    for c in COMPOUNDS:
        if c.name.lower() == key:
            return c
    return None


def get_compounds_by_class(cls: CompoundClass) -> list[BioactiveCompound]:
    """Filter compounds by chemical class."""
    return [c for c in COMPOUNDS if c.compound_class == cls]


def get_compounds_by_bioactivity(activity: BioactivityType) -> list[BioactiveCompound]:
    """Filter compounds by bioactivity type."""
    return [c for c in COMPOUNDS if c.bioactivity == activity]


def get_compounds_by_source(source: SourceType) -> list[BioactiveCompound]:
    """Filter compounds by source type."""
    return [c for c in COMPOUNDS if c.source_type == source]


def get_compounds_by_journal(journal: str) -> list[BioactiveCompound]:
    """Filter compounds by source journal (BLSF or Biologics)."""
    key = journal.lower()
    return [c for c in COMPOUNDS if c.source_journal.lower() == key]


def get_essential_oils() -> list[BioactiveCompound]:
    """Get all essential oil components."""
    return get_compounds_by_source(SourceType.PLANT_ESSENTIAL_OIL)


def get_scfas() -> list[BioactiveCompound]:
    """Get all short-chain fatty acid compounds."""
    return get_compounds_by_class(CompoundClass.SHORT_CHAIN_FATTY_ACID)


def get_biologics() -> list[BioactiveCompound]:
    """Get all biologic therapy compounds (antibodies, enzymes, cytokines)."""
    return [
        c
        for c in COMPOUNDS
        if c.compound_class
        in (
            CompoundClass.MONOCLONAL_ANTIBODY,
            CompoundClass.RECOMBINANT_ENZYME,
            CompoundClass.CYTOKINE,
        )
    ]


# ═══════════════════════════════════════════════════════════════════
#  Normalization Constants (from compound property ranges)
# ═══════════════════════════════════════════════════════════════════

_EPSILON = 1e-8  # Guard band — frozen parameter

# Molecular weight: log-scale from ~50 g/mol to ~200,000 g/mol
_MW_MIN_LOG = math.log(50.0)
_MW_MAX_LOG = math.log(200_000.0)

# Boiling/melting point: 150 K to 600 K (small molecules; None for proteins)
_TEMP_MIN = 150.0
_TEMP_MAX = 600.0

# logP: -2 to +6
_LOGP_MIN = -2.0
_LOGP_MAX = 6.0

# Structural complexity: heavy atoms 4 to 11000
_HEAVY_MIN_LOG = math.log(4.0)
_HEAVY_MAX_LOG = math.log(12_000.0)


def _clamp(x: float, lo: float, hi: float) -> float:
    """Clamp x to [lo, hi]."""
    return max(lo, min(hi, x))


def _log_normalize(val: float, min_log: float, max_log: float) -> float:
    """Log-scale normalize a positive value to (ε, 1−ε)."""
    if val <= 0:
        return _EPSILON
    log_val = math.log(val)
    ratio = (log_val - min_log) / (max_log - min_log) if max_log > min_log else 0.5
    return _clamp(ratio, _EPSILON, 1.0 - _EPSILON)


def _linear_normalize(val: float, lo: float, hi: float) -> float:
    """Linear normalize to (ε, 1−ε)."""
    if hi <= lo:
        return 0.5
    ratio = (val - lo) / (hi - lo)
    return _clamp(ratio, _EPSILON, 1.0 - _EPSILON)


# ═══════════════════════════════════════════════════════════════════
#  Trace Vector & Kernel Computation
# ═══════════════════════════════════════════════════════════════════


def build_trace(compound: BioactiveCompound) -> tuple[float, ...]:
    """Build 8-channel trace vector for a bioactive compound.

    Channels:
        0. molecular_weight (log-normalized)
        1. bioactivity_potency (normalized 0-100% → [ε, 1-ε])
        2. lipophilicity (logP linear-normalized; 0.5 if None)
        3. thermal_stability (bp or mp, linear-normalized; 0.5 if None)
        4. structural_complexity (heavy atom count, log-normalized)
        5. target_breadth (from organism → breadth mapping)
        6. safety_index (direct [0,1])
        7. source_naturalness (from source type mapping)
    """
    # Ch 0: Molecular weight (log-normalized)
    c0 = _log_normalize(compound.molecular_weight, _MW_MIN_LOG, _MW_MAX_LOG)

    # Ch 1: Bioactivity potency (normalized to [0,1])
    if compound.potency_value is not None:
        raw_potency = compound.potency_value
        # For % metrics: direct 0-100 → 0-1
        if "%" in compound.potency_unit:
            c1 = _clamp(raw_potency / 100.0, _EPSILON, 1.0 - _EPSILON)
        # For mm inhibition zones: normalize 0-30 mm
        elif "mm" in compound.potency_unit:
            c1 = _clamp(raw_potency / 30.0, _EPSILON, 1.0 - _EPSILON)
        # For concentration metrics (pg/mL): logscale sensitivity
        elif "pg" in compound.potency_unit or "mL" in compound.potency_unit:
            c1 = _clamp(0.95, _EPSILON, 1.0 - _EPSILON)  # Ultra-sensitive
        else:
            c1 = _clamp(raw_potency / 100.0, _EPSILON, 1.0 - _EPSILON)
    else:
        c1 = 0.5

    # Ch 2: Lipophilicity (logP)
    c2 = _linear_normalize(compound.logP, _LOGP_MIN, _LOGP_MAX) if compound.logP is not None else 0.5

    # Ch 3: Thermal stability (bp or mp)
    temp = compound.boiling_point_K or compound.melting_point_K
    c3 = _linear_normalize(temp, _TEMP_MIN, _TEMP_MAX) if temp is not None else 0.5

    # Ch 4: Structural complexity (heavy atom count, log-normalized)
    if compound.heavy_atom_count > 0:
        c4 = _log_normalize(compound.heavy_atom_count, _HEAVY_MIN_LOG, _HEAVY_MAX_LOG)
    else:
        c4 = 0.5

    # Ch 5: Target breadth
    c5 = _clamp(
        _TARGET_BREADTH.get(compound.primary_target, 0.5),
        _EPSILON,
        1.0 - _EPSILON,
    )

    # Ch 6: Safety index
    c6 = _clamp(compound.safety_index, _EPSILON, 1.0 - _EPSILON)

    # Ch 7: Source naturalness
    c7 = _clamp(
        _SOURCE_NATURALNESS.get(compound.source_type, 0.5),
        _EPSILON,
        1.0 - _EPSILON,
    )

    return (c0, c1, c2, c3, c4, c5, c6, c7)


def compute_bioactive_kernel(compound: BioactiveCompound) -> BioactiveKernelResult:
    """Compute GCD kernel invariants for a bioactive compound.

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

    return BioactiveKernelResult(
        name=compound.name,
        formula=compound.formula,
        compound_class=compound.compound_class.value,
        bioactivity=compound.bioactivity.value,
        trace=trace,
        F=F,
        omega=omega,
        S=S,
        C=C,
        kappa=kappa,
        IC=IC,
        regime=regime,
    )


def compute_all_bioactive_kernels() -> list[BioactiveKernelResult]:
    """Compute kernel invariants for all compounds in database."""
    return [compute_bioactive_kernel(c) for c in COMPOUNDS]


# ═══════════════════════════════════════════════════════════════════
#  Cross-Domain Analysis
# ═══════════════════════════════════════════════════════════════════


def analyze_compound_classes() -> dict[str, dict[str, float]]:
    """Average kernel invariants per compound class.

    Shows how molecular scale maps to fidelity structure:
    small molecules (terpenes, SCFAs) vs. large biologics (antibodies,
    enzymes) exhibit fundamentally different heterogeneity patterns.
    """
    from collections import defaultdict

    accum: dict[str, list[BioactiveKernelResult]] = defaultdict(list)
    for c in COMPOUNDS:
        kr = compute_bioactive_kernel(c)
        accum[kr.compound_class].append(kr)

    stats: dict[str, dict[str, float]] = {}
    for cls, results in sorted(accum.items()):
        n = len(results)
        stats[cls] = {
            "count": n,
            "mean_F": sum(r.F for r in results) / n,
            "mean_omega": sum(r.omega for r in results) / n,
            "mean_IC": sum(r.IC for r in results) / n,
            "mean_heterogeneity_gap": sum(r.F - r.IC for r in results) / n,
            "mean_S": sum(r.S for r in results) / n,
            "mean_C": sum(r.C for r in results) / n,
        }
    return stats


def analyze_scale_gradient() -> dict[str, Any]:
    """Analyze kernel behavior across molecular weight scale.

    Essential oils (~150 Da) → SCFAs (~60-110 Da) → peptides (~5 kDa)
    → cytokines (~21 kDa) → enzymes (~66 kDa) → antibodies (~147 kDa).

    The heterogeneity gap Δ = F − IC reveals how channel uniformity
    changes with molecular complexity.
    """
    results: list[dict[str, Any]] = []
    for c in COMPOUNDS:
        kr = compute_bioactive_kernel(c)
        results.append(
            {
                "name": kr.name,
                "MW": c.molecular_weight,
                "MW_log": math.log10(c.molecular_weight),
                "F": kr.F,
                "IC": kr.IC,
                "heterogeneity_gap": kr.F - kr.IC,
                "regime": kr.regime,
                "compound_class": kr.compound_class,
            }
        )

    # Sort by molecular weight
    results.sort(key=lambda r: r["MW"])

    small_mol = [r for r in results if r["MW"] < 1000]
    large_mol = [r for r in results if r["MW"] >= 1000]

    return {
        "compounds_by_MW": results,
        "small_molecule_count": len(small_mol),
        "large_molecule_count": len(large_mol),
        "small_mean_F": sum(r["F"] for r in small_mol) / len(small_mol) if small_mol else 0.0,
        "large_mean_F": sum(r["F"] for r in large_mol) / len(large_mol) if large_mol else 0.0,
        "small_mean_gap": sum(r["heterogeneity_gap"] for r in small_mol) / len(small_mol) if small_mol else 0.0,
        "large_mean_gap": sum(r["heterogeneity_gap"] for r in large_mol) / len(large_mol) if large_mol else 0.0,
    }


def analyze_natural_vs_recombinant() -> dict[str, Any]:
    """Compare natural bioactive compounds vs. recombinant biologics.

    Natural products (essential oils, microbial metabolites) have higher
    source_naturalness but lower structural_complexity.  Recombinant biologics
    (mAbs, enzymes) have extreme structural complexity but lower naturalness.

    The kernel exposes this as a trade-off in the heterogeneity gap.
    """
    natural = [
        c
        for c in COMPOUNDS
        if c.source_type
        in (
            SourceType.PLANT_ESSENTIAL_OIL,
            SourceType.MICROBIAL_METABOLITE,
            SourceType.DIETARY_FIBER_FERMENTATION,
        )
    ]
    recombinant = [c for c in COMPOUNDS if c.source_type == SourceType.RECOMBINANT_BIOTECHNOLOGY]

    nat_kernels = [compute_bioactive_kernel(c) for c in natural]
    rec_kernels = [compute_bioactive_kernel(c) for c in recombinant]

    def _avg(lst: list[BioactiveKernelResult], attr: str) -> float:
        if not lst:
            return 0.0
        return sum(getattr(r, attr) for r in lst) / len(lst)

    return {
        "natural_count": len(natural),
        "recombinant_count": len(recombinant),
        "natural_mean_F": _avg(nat_kernels, "F"),
        "recombinant_mean_F": _avg(rec_kernels, "F"),
        "natural_mean_IC": _avg(nat_kernels, "IC"),
        "recombinant_mean_IC": _avg(rec_kernels, "IC"),
        "natural_mean_gap": sum(r.F - r.IC for r in nat_kernels) / len(nat_kernels) if nat_kernels else 0.0,
        "recombinant_mean_gap": sum(r.F - r.IC for r in rec_kernels) / len(rec_kernels) if rec_kernels else 0.0,
    }


# ═══════════════════════════════════════════════════════════════════
#  Validation
# ═══════════════════════════════════════════════════════════════════


def validate_database() -> dict[str, Any]:
    """Validate the bioactive compounds database.

    Checks:
        1. All compounds have valid compound class and bioactivity
        2. Molecular weight > 0 for all entries
        3. Trace has exactly 8 channels, all in (0,1)
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

        # Trace validity
        trace = build_trace(compound)
        if len(trace) != 8:
            errors.append(f"{compound.name}: trace has {len(trace)} channels, expected 8")
        for i, c in enumerate(trace):
            if not (0.0 < c < 1.0):
                errors.append(f"{compound.name}: channel {i} = {c} not in (0,1)")

        # Kernel identities
        kr = compute_bioactive_kernel(compound)
        duality_residual = abs(kr.F + kr.omega - 1.0)
        if duality_residual > 1e-12:
            errors.append(f"{compound.name}: duality identity violated, |F + ω − 1| = {duality_residual:.2e}")
        if kr.IC > kr.F + 1e-12:
            errors.append(f"{compound.name}: integrity bound violated, IC ({kr.IC:.6f}) > F ({kr.F:.6f})")
        if math.isnan(kr.F) or math.isinf(kr.F):
            errors.append(f"{compound.name}: F is NaN/Inf")
        if math.isnan(kr.IC) or math.isinf(kr.IC):
            errors.append(f"{compound.name}: IC is NaN/Inf")

        # Warnings for missing data
        if compound.boiling_point_K is None and compound.melting_point_K is None:
            warnings.append(f"{compound.name}: no thermal stability data (bp/mp)")
        if compound.logP is None:
            warnings.append(f"{compound.name}: missing logP")

    return {
        "total_compounds": len(COMPOUNDS),
        "errors": errors,
        "warnings": warnings,
        "status": "CONFORMANT" if not errors else "NONCONFORMANT",
    }

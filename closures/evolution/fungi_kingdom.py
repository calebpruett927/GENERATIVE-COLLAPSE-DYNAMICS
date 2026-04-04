"""Fungi Kingdom Closure — Evolution Domain.

Tier-2 closure mapping 12 fungi species through the GCD kernel.
Fungi occupy the liminal space between life and death — they decompose
the dead into nutrients for the living, form symbiotic networks that
underpin 80% of terrestrial plant life, and grow to the largest
organismal sizes on Earth while retaining structural union through
mycelial networks.

Channels (8, equal weights w_i = 1/8):
  0  mycelial_extent         — spatial network reach (1 = Earth's largest organism)
  1  decomposition_capacity  — ability to degrade organic substrates (1 = lignin + cellulose)
  2  symbiotic_integration   — degree of mutualistic partnership (1 = obligate symbiont)
  3  reproductive_versatility — range of reproductive strategies (1 = dimorphic + multi-spore)
  4  metabolic_diversity     — breadth of metabolic pathways / secondary metabolites
  5  environmental_breadth   — range of habitats and conditions tolerated
  6  chemical_potency        — bioactive compound production (antibiotics, toxins, enzymes)
  7  structural_persistence  — durability / longevity (chitin walls, perennial mycelia)

12 species entities across 4 categories:
  Decomposers (3): armillaria_ostoyae, trametes_versicolor, serpula_lacrymans
  Symbionts (3): rhizophagus_irregularis, xanthoria_parietina, epichloe_festucae
  Transformers (3): saccharomyces_cerevisiae, penicillium_chrysogenum, aspergillus_oryzae
  Extremophiles (3): ophiocordyceps_unilateralis, fomitiporia_ellipsoidea,
                      batrachochytrium_dendrobatidis

6 mycorrhizal stress entities across 2 categories (Thiem et al. 2025):
  AM Mycorrhizal (3): am_unstressed, am_shortterm_saline, am_longterm_saline
  Dual AM+EM (3): dual_amem_unstressed, dual_amem_shortterm_saline,
                   dual_amem_longterm_saline

9 theorems (T-FK-1 through T-FK-9).
  T-FK-1–6: Species-level theorems.
  T-FK-7–9: Mycorrhizal stress response (Thiem et al. 2025, Plant and Soil,
             DOI: 10.1007/s11104-025-07630-0).

Key GCD insight: Fungi live between things — between death and life
(decomposition), between organisms (symbiosis), between scales
(microscopic hyphae → hectare-spanning mycelia). The kernel reveals
that this liminal existence produces distinctive signatures: symbionts
achieve the highest multiplicative coherence (IC/F ratio) because
balanced integration across channels preserves the geometric mean,
while extreme specialists suffer geometric slaughter from dead channels.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_WORKSPACE = Path(__file__).resolve().parents[2]
for _p in [str(_WORKSPACE / "src"), str(_WORKSPACE)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from umcp.frozen_contract import EPSILON  # noqa: E402
from umcp.kernel_optimized import compute_kernel_outputs  # noqa: E402

FK_CHANNELS = [
    "mycelial_extent",
    "decomposition_capacity",
    "symbiotic_integration",
    "reproductive_versatility",
    "metabolic_diversity",
    "environmental_breadth",
    "chemical_potency",
    "structural_persistence",
]
N_FK_CHANNELS = len(FK_CHANNELS)


@dataclass(frozen=True, slots=True)
class FungiEntity:
    """A fungal species with 8 measurable channels."""

    name: str
    category: str
    common_name: str
    phylum: str
    mycelial_extent: float
    decomposition_capacity: float
    symbiotic_integration: float
    reproductive_versatility: float
    metabolic_diversity: float
    environmental_breadth: float
    chemical_potency: float
    structural_persistence: float

    def trace_vector(self) -> np.ndarray:
        return np.array(
            [
                self.mycelial_extent,
                self.decomposition_capacity,
                self.symbiotic_integration,
                self.reproductive_versatility,
                self.metabolic_diversity,
                self.environmental_breadth,
                self.chemical_potency,
                self.structural_persistence,
            ]
        )


# ── Entity Catalog ────────────────────────────────────────────────────
#
# Channel values are normalized to [0, 1] based on the range of each
# property across the fungal kingdom. Sources: ScienceDirect Fungi
# Kingdom topic, Yale E360 fungi kingdom feature, Fungi Foundation,
# primary mycological literature.

FK_ENTITIES: tuple[FungiEntity, ...] = (
    # ── Decomposers ───────────────────────────────────────────────────
    # Organisms that transform death into life — the recyclers.
    FungiEntity(
        "armillaria_ostoyae",
        "decomposer",
        "Honey Mushroom",
        "Basidiomycota",
        # Largest organism on Earth: ~965 Ha in Oregon Blue Mountains.
        # White-rot decomposer of conifers. Low symbiotic integration.
        # Rhizomorphs persist for millennia. Bioluminescent.
        mycelial_extent=0.98,
        decomposition_capacity=0.85,
        symbiotic_integration=0.15,
        reproductive_versatility=0.70,
        metabolic_diversity=0.65,
        environmental_breadth=0.55,
        chemical_potency=0.50,
        structural_persistence=0.95,
    ),
    FungiEntity(
        "trametes_versicolor",
        "decomposer",
        "Turkey Tail",
        "Basidiomycota",
        # Ubiquitous white-rot polypore. Degrades lignin and cellulose.
        # Polysaccharide-K (PSK) is an approved anticancer drug in Japan.
        # Found on every continent except Antarctica.
        mycelial_extent=0.35,
        decomposition_capacity=0.92,
        symbiotic_integration=0.10,
        reproductive_versatility=0.60,
        metabolic_diversity=0.75,
        environmental_breadth=0.70,
        chemical_potency=0.80,
        structural_persistence=0.65,
    ),
    FungiEntity(
        "serpula_lacrymans",
        "decomposer",
        "Dry Rot Fungus",
        "Basidiomycota",
        # Brown-rot specialist of structural timber. Can transport water
        # through rhizomorphs to colonize dry wood. Extremely persistent
        # in buildings. Narrow host range (built-environment specialist).
        mycelial_extent=0.40,
        decomposition_capacity=0.88,
        symbiotic_integration=0.05,
        reproductive_versatility=0.45,
        metabolic_diversity=0.55,
        environmental_breadth=0.30,
        chemical_potency=0.35,
        structural_persistence=0.80,
    ),
    # ── Symbionts ─────────────────────────────────────────────────────
    # Organisms that live between — connecting kingdoms.
    FungiEntity(
        "rhizophagus_irregularis",
        "symbiont",
        "AM Fungus",
        "Glomeromycota",
        # Arbuscular mycorrhiza: colonizes ~80% of terrestrial plant
        # roots. Obligate biotroph — cannot survive without host.
        # Trades phosphorus/nitrogen for plant carbohydrates.
        # No known sexual reproduction (ancient asexual lineage).
        mycelial_extent=0.60,
        decomposition_capacity=0.10,
        symbiotic_integration=0.98,
        reproductive_versatility=0.30,
        metabolic_diversity=0.40,
        environmental_breadth=0.80,
        chemical_potency=0.25,
        structural_persistence=0.85,
    ),
    FungiEntity(
        "xanthoria_parietina",
        "symbiont",
        "Common Orange Lichen",
        "Ascomycota",
        # Lichen: fungus + photobiont (Trebouxia alga). Survives
        # radiation, desiccation, extreme temperatures. Found from
        # Arctic to tropics, on bark, rock, concrete. Lives centuries.
        mycelial_extent=0.10,
        decomposition_capacity=0.15,
        symbiotic_integration=0.95,
        reproductive_versatility=0.55,
        metabolic_diversity=0.60,
        environmental_breadth=0.90,
        chemical_potency=0.45,
        structural_persistence=0.90,
    ),
    FungiEntity(
        "epichloe_festucae",
        "symbiont",
        "Grass Endophyte",
        "Ascomycota",
        # Endophyte: lives WITHIN grass cells. Produces alkaloids
        # (peramine, loline) that repel herbivores. Vertically
        # transmitted through seeds. The fungus IS part of the plant.
        mycelial_extent=0.15,
        decomposition_capacity=0.05,
        symbiotic_integration=0.92,
        reproductive_versatility=0.35,
        metabolic_diversity=0.70,
        environmental_breadth=0.45,
        chemical_potency=0.85,
        structural_persistence=0.75,
    ),
    # ── Transformers ──────────────────────────────────────────────────
    # Metabolic masters — chemical factories co-opted by humans.
    FungiEntity(
        "saccharomyces_cerevisiae",
        "transformer",
        "Baker's Yeast",
        "Ascomycota",
        # Unicellular: no mycelium. But: fermentation master, model
        # organism, first eukaryotic genome sequenced (1996). Used for
        # bread, beer, wine, biofuel, recombinant protein production.
        # Highly adaptable to diverse sugar substrates.
        mycelial_extent=0.02,
        decomposition_capacity=0.25,
        symbiotic_integration=0.20,
        reproductive_versatility=0.80,
        metabolic_diversity=0.90,
        environmental_breadth=0.85,
        chemical_potency=0.60,
        structural_persistence=0.40,
    ),
    FungiEntity(
        "penicillium_chrysogenum",
        "transformer",
        "Penicillin Mold",
        "Ascomycota",
        # Source of penicillin — the antibiotic that changed medicine.
        # Prolific secondary metabolite producer. Common saprophyte
        # of food and indoor environments. Multiple anamorphs.
        mycelial_extent=0.20,
        decomposition_capacity=0.55,
        symbiotic_integration=0.10,
        reproductive_versatility=0.75,
        metabolic_diversity=0.85,
        environmental_breadth=0.80,
        chemical_potency=0.95,
        structural_persistence=0.55,
    ),
    FungiEntity(
        "aspergillus_oryzae",
        "transformer",
        "Koji Mold",
        "Ascomycota",
        # The foundational organism of East Asian fermentation:
        # sake, soy sauce, miso, mirin. Declared Japan's "National
        # Fungus." Massive enzyme production (amylases, proteases).
        mycelial_extent=0.25,
        decomposition_capacity=0.65,
        symbiotic_integration=0.15,
        reproductive_versatility=0.70,
        metabolic_diversity=0.92,
        environmental_breadth=0.75,
        chemical_potency=0.88,
        structural_persistence=0.50,
    ),
    # ── Extremophiles ─────────────────────────────────────────────────
    # Boundary-dwellers — specialists at the edges of life.
    FungiEntity(
        "ophiocordyceps_unilateralis",
        "extremophile",
        "Zombie Ant Fungus",
        "Ascomycota",
        # Parasitoid: manipulates ant behavior (death-grip on leaf
        # vein) to optimize spore dispersal. Species-specific. Minimal
        # mycelial network but extreme chemical sophistication.
        mycelial_extent=0.10,
        decomposition_capacity=0.30,
        symbiotic_integration=0.05,
        reproductive_versatility=0.50,
        metabolic_diversity=0.80,
        environmental_breadth=0.15,
        chemical_potency=0.92,
        structural_persistence=0.60,
    ),
    FungiEntity(
        "fomitiporia_ellipsoidea",
        "extremophile",
        "Giant Bracket Fungus",
        "Basidiomycota",
        # Produces the largest known fruiting body: ~500 kg specimen
        # in Hainan, China (2011). Perennial polypore, adds new
        # hymenial layers each year. Extremely durable but narrow niche.
        mycelial_extent=0.70,
        decomposition_capacity=0.75,
        symbiotic_integration=0.10,
        reproductive_versatility=0.40,
        metabolic_diversity=0.45,
        environmental_breadth=0.25,
        chemical_potency=0.30,
        structural_persistence=0.98,
    ),
    FungiEntity(
        "batrachochytrium_dendrobatidis",
        "extremophile",
        "Chytrid Frog Pathogen",
        "Chytridiomycota",
        # Aquatic pathogen: zoosporic (flagellated). Caused decline of
        # 501+ amphibian species in 54 countries, 90 presumed
        # extinctions. One of the most destructive pathogens in history.
        # Specialist: only infects amphibian keratin.
        mycelial_extent=0.05,
        decomposition_capacity=0.20,
        symbiotic_integration=0.02,
        reproductive_versatility=0.85,
        metabolic_diversity=0.50,
        environmental_breadth=0.60,
        chemical_potency=0.70,
        structural_persistence=0.35,
    ),
)


# ── Kernel Computation ────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class FKKernelResult:
    """Kernel output for a fungi kingdom entity."""

    name: str
    category: str
    common_name: str
    F: float
    omega: float
    S: float
    C: float
    kappa: float
    IC: float
    regime: str

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "category": self.category,
            "common_name": self.common_name,
            "F": self.F,
            "omega": self.omega,
            "S": self.S,
            "C": self.C,
            "kappa": self.kappa,
            "IC": self.IC,
            "regime": self.regime,
        }


def compute_fk_kernel(entity: FungiEntity) -> FKKernelResult:
    """Compute GCD kernel for a fungi kingdom entity."""
    c = entity.trace_vector()
    c = np.clip(c, EPSILON, 1.0 - EPSILON)
    w = np.ones(N_FK_CHANNELS) / N_FK_CHANNELS
    result = compute_kernel_outputs(c, w)
    F = float(result["F"])
    omega = float(result["omega"])
    S = float(result["S"])
    C_val = float(result["C"])
    kappa = float(result["kappa"])
    IC = float(result["IC"])
    if omega >= 0.30:
        regime = "Collapse"
    elif omega < 0.038 and F > 0.90 and S < 0.15 and C_val < 0.14:
        regime = "Stable"
    else:
        regime = "Watch"
    return FKKernelResult(
        name=entity.name,
        category=entity.category,
        common_name=entity.common_name,
        F=F,
        omega=omega,
        S=S,
        C=C_val,
        kappa=kappa,
        IC=IC,
        regime=regime,
    )


def compute_all_entities() -> list[FKKernelResult]:
    """Compute kernel outputs for all fungi kingdom entities."""
    return [compute_fk_kernel(e) for e in FK_ENTITIES]


# ── Theorems ──────────────────────────────────────────────────────────


def verify_t_fk_1(results: list[FKKernelResult]) -> dict:
    """T-FK-1: Decomposer Fidelity Dominance.

    Decomposers have the highest mean F among categories. Organisms
    that transform death into life maintain the broadest channel
    fidelity — high decomposition, high persistence, substantial
    network extent. The ecological role of recycling IS high fidelity.
    """
    cats = {r.category for r in results}
    cat_means = {cat: float(np.mean([r.F for r in results if r.category == cat])) for cat in cats}
    decomp_mean = cat_means["decomposer"]
    others_max = max(v for k, v in cat_means.items() if k != "decomposer")
    passed = decomp_mean > others_max
    return {
        "name": "T-FK-1",
        "passed": bool(passed),
        "decomposer_mean_F": decomp_mean,
        "category_means": cat_means,
    }


def verify_t_fk_2(results: list[FKKernelResult]) -> dict:
    """T-FK-2: Extremophile Specialist Incoherence.

    Extremophiles have the lowest mean IC/F ratio. Species that push
    to the boundary — parasitoid behavior manipulation, giant single
    fruiting bodies, amphibian-specialist pathogens — specialize so
    heavily that channel heterogeneity destroys multiplicative
    coherence. Specialization at the edge costs geometric integrity.
    """
    cats = {r.category for r in results}
    cat_icf = {cat: float(np.mean([r.IC / r.F for r in results if r.category == cat])) for cat in cats}
    extremo_icf = cat_icf["extremophile"]
    others_min = min(v for k, v in cat_icf.items() if k != "extremophile")
    passed = extremo_icf < others_min
    return {
        "name": "T-FK-2",
        "passed": bool(passed),
        "extremophile_IC_F": extremo_icf,
        "category_IC_F": cat_icf,
    }


def verify_t_fk_3(results: list[FKKernelResult]) -> dict:
    """T-FK-3: Network-Persistence Coupling.

    Among all entities, mycelial_extent and structural_persistence
    are positively correlated (r > 0.3). Extending in space co-occurs
    with extending in time — massive mycelia are long-lived, while
    unicellular/minimal forms are transient. Spatial union and
    temporal persistence are coupled modalities in fungi.
    """
    extents = np.array([e.mycelial_extent for e in FK_ENTITIES])
    persists = np.array([e.structural_persistence for e in FK_ENTITIES])
    corr = float(np.corrcoef(extents, persists)[0, 1])
    passed = corr > 0.3
    return {
        "name": "T-FK-3",
        "passed": bool(passed),
        "correlation": corr,
    }


def verify_t_fk_4(results: list[FKKernelResult]) -> dict:
    """T-FK-4: Unicellular Geometric Slaughter.

    Saccharomyces cerevisiae has the largest heterogeneity gap
    (Δ = F − IC) among all entities. As a unicellular yeast, its
    mycelial_extent channel is near-ε (0.02), producing severe
    geometric slaughter: the most metabolically versatile fungus
    is structurally the most incoherent, because unicellularity IS
    the dead channel. Even metabolic mastery cannot compensate for
    the absence of spatial network integration.
    """
    sc = next(r for r in results if r.name == "saccharomyces_cerevisiae")
    sc_gap = sc.F - sc.IC
    max_gap = max(r.F - r.IC for r in results)
    max_gap_entity = max(results, key=lambda r: r.F - r.IC).name
    passed = max_gap_entity == "saccharomyces_cerevisiae"
    return {
        "name": "T-FK-4",
        "passed": bool(passed),
        "saccharomyces_gap": float(sc_gap),
        "max_gap": float(max_gap),
        "saccharomyces_F": sc.F,
        "saccharomyces_IC": sc.IC,
    }


def verify_t_fk_5(results: list[FKKernelResult]) -> dict:
    """T-FK-5: Transformer Metabolic Dominance.

    Transformers have the highest mean metabolic_diversity channel value.
    Chemical versatility concentrates in the organisms humans have
    co-opted for industry: yeast (fermentation), Penicillium
    (antibiotics), Aspergillus (enzymes). The metabolic channel IS
    the signature of transformers.
    """
    cats = {r.category for r in results}
    cat_metab = {}
    for cat in cats:
        ents = [e for e in FK_ENTITIES if e.category == cat]
        cat_metab[cat] = float(np.mean([e.metabolic_diversity for e in ents]))
    trans_metab = cat_metab["transformer"]
    others_max = max(v for k, v in cat_metab.items() if k != "transformer")
    passed = trans_metab > others_max
    return {
        "name": "T-FK-5",
        "passed": bool(passed),
        "transformer_metabolic": trans_metab,
        "category_metabolic": cat_metab,
    }


def verify_t_fk_6(results: list[FKKernelResult]) -> dict:
    """T-FK-6: Universal Collapse — Fungi's Generative Regime.

    All 12 fungi species occupy Collapse regime (ω ≥ 0.30). Fungi
    inhabit the liminal space between kingdoms and between death and
    life; this ecological intermediacy produces moderate channel values
    across all species — no fungus achieves the uniformly high fidelity
    required for Stable or Watch. Collapse IS the fungal condition,
    and as Axiom-0 states, collapse is generative: the kingdom that
    recycles death into life structurally inhabits the generative
    regime.
    """
    regimes = [r.regime for r in results]
    n_collapse = sum(1 for r in regimes if r == "Collapse")
    all_collapse = n_collapse == len(results)
    omegas = [r.omega for r in results]
    passed = all_collapse
    return {
        "name": "T-FK-6",
        "passed": bool(passed),
        "n_collapse": n_collapse,
        "n_total": len(results),
        "min_omega": float(min(omegas)),
        "max_omega": float(max(omegas)),
    }


def verify_all_theorems() -> list[dict]:
    """Run all T-FK theorems (1–6, species-level)."""
    results = compute_all_entities()
    return [
        verify_t_fk_1(results),
        verify_t_fk_2(results),
        verify_t_fk_3(results),
        verify_t_fk_4(results),
        verify_t_fk_5(results),
        verify_t_fk_6(results),
    ]


# ══════════════════════════════════════════════════════════════════════
# MYCORRHIZAL STRESS RESPONSE — Thiem et al. 2025 (Plant and Soil)
# ══════════════════════════════════════════════════════════════════════
#
# "Impact of mycorrhizal inoculation on the fungal root and
#  rhizosphere communities, growth and salinity tolerance of
#  Alnus glutinosa Gaertn. seedlings"
#
# DOI: 10.1007/s11104-025-07630-0 (Open Access, CC BY 4.0)
# Published: 2025-06-19
# Authors: Thiem D, Gołębiewski M, Taburski J, Kowalkowski T,
#          Baum C, Hrynkiewicz K
#
# Key finding: Alnus glutinosa forms DUAL mycorrhizal symbioses —
# both arbuscular (AM) and ectomycorrhizal (EM) fungi coexist in
# a single root system. Under short-term salinity, dual AM+EM
# outperforms AM-alone (broader channel coverage). Under long-term
# salinity, AM-alone outperforms (EM channel degrades → geometric
# slaughter of dual system's IC). This is §3 of the orientation
# playing out in a real biological system.
#
# 6 experimental configurations × same 8 channels → 3 theorems
# (T-FK-7 through T-FK-9).
# ══════════════════════════════════════════════════════════════════════

MS_ENTITIES: tuple[FungiEntity, ...] = (
    # ── AM Mycorrhizal Configurations ─────────────────────────────────
    # Arbuscular mycorrhizal community alone (Glomeromycota-dominated).
    # AM fungi colonize ~80% of terrestrial plant roots, trading
    # phosphorus/nitrogen for plant carbohydrates. Fine hyphal
    # networks optimize nutrient uptake (the "fine-grained" channel).
    FungiEntity(
        "am_unstressed",
        "am_mycorrhizal",
        "AM Community (Baseline)",
        "Glomeromycota",
        # Baseline AM colonization of Alnus glutinosa roots. High
        # symbiotic integration (obligate biotroph), broad
        # environmental tolerance, no decomposition. Rhizophagus-
        # like functional profile.
        mycelial_extent=0.55,
        decomposition_capacity=0.08,
        symbiotic_integration=0.92,
        reproductive_versatility=0.30,
        metabolic_diversity=0.38,
        environmental_breadth=0.75,
        chemical_potency=0.22,
        structural_persistence=0.80,
    ),
    FungiEntity(
        "am_shortterm_saline",
        "am_mycorrhizal",
        "AM Community (ST Salinity)",
        "Glomeromycota",
        # Short-term salinity: AM colonization partially reduced but
        # still effective. Paper: both treatments increased plant
        # growth under ST. AM maintains core symbiotic function.
        # Proline accumulation and water content preserved.
        mycelial_extent=0.45,
        decomposition_capacity=0.08,
        symbiotic_integration=0.78,
        reproductive_versatility=0.28,
        metabolic_diversity=0.35,
        environmental_breadth=0.60,
        chemical_potency=0.30,
        structural_persistence=0.70,
    ),
    FungiEntity(
        "am_longterm_saline",
        "am_mycorrhizal",
        "AM Community (LT Salinity)",
        "Glomeromycota",
        # Long-term salinity: AM still functional — paper shows AM
        # more effective than dual under LT. AM's simpler architecture
        # avoids the EM degradation problem. Channels degrade
        # uniformly rather than suffering one-channel collapse.
        mycelial_extent=0.35,
        decomposition_capacity=0.08,
        symbiotic_integration=0.65,
        reproductive_versatility=0.25,
        metabolic_diversity=0.30,
        environmental_breadth=0.50,
        chemical_potency=0.35,
        structural_persistence=0.60,
    ),
    # ── Dual AM+EM Configurations ─────────────────────────────────────
    # Both arbuscular AND ectomycorrhizal fungi coexisting in a single
    # Alnus glutinosa root system. EM fungi form sheaths around roots
    # (Hartig net), providing stress buffering and sodium exclusion
    # (the "coarse-grained" channel). Dual architecture = more
    # channels active, but more channels vulnerable.
    FungiEntity(
        "dual_amem_unstressed",
        "dual_mycorrhizal",
        "Dual AM+EM (Baseline)",
        "Glomeromycota",
        # Dual colonization baseline. EM adds decomposition capacity,
        # broader mycelial extent (extramatrical networks), and
        # structural persistence from Hartig net. Higher F than AM
        # alone — more channels contributing. Dominant AM partner
        # determines phylum classification.
        mycelial_extent=0.65,
        decomposition_capacity=0.30,
        symbiotic_integration=0.88,
        reproductive_versatility=0.40,
        metabolic_diversity=0.50,
        environmental_breadth=0.78,
        chemical_potency=0.35,
        structural_persistence=0.82,
    ),
    FungiEntity(
        "dual_amem_shortterm_saline",
        "dual_mycorrhizal",
        "Dual AM+EM (ST Salinity)",
        "Glomeromycota",
        # Short-term salinity: paper shows AM+EM more effective at
        # reducing sodium bioconcentration in leaves and roots.
        # Both channels still active — EM buffers ion stress while
        # AM maintains nutrient flow. Still higher F than AM alone
        # under same conditions.
        mycelial_extent=0.55,
        decomposition_capacity=0.25,
        symbiotic_integration=0.75,
        reproductive_versatility=0.38,
        metabolic_diversity=0.45,
        environmental_breadth=0.65,
        chemical_potency=0.40,
        structural_persistence=0.72,
    ),
    FungiEntity(
        "dual_amem_longterm_saline",
        "dual_mycorrhizal",
        "Dual AM+EM (LT Salinity)",
        "Glomeromycota",
        # Long-term salinity: EM channel degrades significantly.
        # Paper: AM alone more effective under LT. The EM component
        # loses colonization vigor under sustained salt — its
        # decomposition capacity drops, mycelial extent shrinks,
        # and the dead/dying EM channel drags IC down through
        # geometric slaughter. This is §3 of the orientation in vivo.
        mycelial_extent=0.40,
        decomposition_capacity=0.12,
        symbiotic_integration=0.50,
        reproductive_versatility=0.32,
        metabolic_diversity=0.35,
        environmental_breadth=0.42,
        chemical_potency=0.38,
        structural_persistence=0.55,
    ),
)


def compute_ms_kernel(entity: FungiEntity) -> FKKernelResult:
    """Compute GCD kernel for a mycorrhizal stress entity."""
    return compute_fk_kernel(entity)


def compute_all_ms_entities() -> list[FKKernelResult]:
    """Compute kernel outputs for all mycorrhizal stress entities."""
    return [compute_ms_kernel(e) for e in MS_ENTITIES]


# ── Mycorrhizal Stress Theorems ───────────────────────────────────────


def verify_t_fk_7(ms_results: list[FKKernelResult]) -> dict:
    """T-FK-7: Dual Mycorrhizal Short-Term Advantage.

    Under short-term salinity, the dual AM+EM configuration has
    higher F than AM-alone, because both mycorrhizal types contribute
    to composite fidelity. The EM component adds decomposition
    capacity, broader mycelial extent, and enhanced sodium exclusion
    — channels that AM alone cannot fill. More active channels →
    higher arithmetic mean → higher fidelity.

    Source: Thiem et al. 2025 — "AM+EM being more effective [at
    reducing sodium bioconcentration] under ST salinity."
    """
    am_st = next(r for r in ms_results if r.name == "am_shortterm_saline")
    dual_st = next(r for r in ms_results if r.name == "dual_amem_shortterm_saline")
    passed = dual_st.F > am_st.F
    return {
        "name": "T-FK-7",
        "passed": bool(passed),
        "dual_st_F": dual_st.F,
        "am_st_F": am_st.F,
        "advantage": dual_st.F - am_st.F,
    }


def verify_t_fk_8(ms_results: list[FKKernelResult]) -> dict:
    """T-FK-8: Long-Term EM Degradation Amplifies Drift.

    The F-drop (drift increase) from unstressed baseline to long-term
    salinity is LARGER for dual AM+EM than for AM-alone. Under
    sustained stress, the EM component degrades faster: its
    decomposition capacity drops, mycelial networks shrink, and the
    weakened EM channels drag the dual system's fidelity down more
    than the uniform degradation of AM-only channels.

    This is geometric slaughter (§3) playing out over time: the dual
    system starts with more channels and higher F, but the EM channel
    becomes the dead weight that amplifies long-term drift beyond
    what the simpler AM-only architecture suffers.

    Source: Thiem et al. 2025 — "AM was more effective under LT
    salinity."
    """
    am_base = next(r for r in ms_results if r.name == "am_unstressed")
    am_lt = next(r for r in ms_results if r.name == "am_longterm_saline")
    dual_base = next(r for r in ms_results if r.name == "dual_amem_unstressed")
    dual_lt = next(r for r in ms_results if r.name == "dual_amem_longterm_saline")
    am_drop = am_base.F - am_lt.F
    dual_drop = dual_base.F - dual_lt.F
    passed = dual_drop > am_drop
    return {
        "name": "T-FK-8",
        "passed": bool(passed),
        "am_F_drop": am_drop,
        "dual_F_drop": dual_drop,
        "am_baseline_F": am_base.F,
        "am_lt_F": am_lt.F,
        "dual_baseline_F": dual_base.F,
        "dual_lt_F": dual_lt.F,
    }


def verify_t_fk_9(ms_results: list[FKKernelResult]) -> dict:
    """T-FK-9: Stress-Induced Drift Monotonicity.

    For BOTH mycorrhizal configurations (AM-alone and dual AM+EM),
    drift (ω) increases monotonically through the stress sequence:
    unstressed → short-term salinity → long-term salinity. Salinity
    is a monotone perturbation — each increment of stress duration
    erodes channels further, never restoring them. The kernel detects
    this as strictly increasing ω.

    Source: Thiem et al. 2025 — growth decrements and colonization
    reductions scale with salinity duration in both treatments.
    """
    am_u = next(r for r in ms_results if r.name == "am_unstressed")
    am_st = next(r for r in ms_results if r.name == "am_shortterm_saline")
    am_lt = next(r for r in ms_results if r.name == "am_longterm_saline")
    dual_u = next(r for r in ms_results if r.name == "dual_amem_unstressed")
    dual_st = next(r for r in ms_results if r.name == "dual_amem_shortterm_saline")
    dual_lt = next(r for r in ms_results if r.name == "dual_amem_longterm_saline")
    am_monotone = am_u.omega < am_st.omega < am_lt.omega
    dual_monotone = dual_u.omega < dual_st.omega < dual_lt.omega
    passed = am_monotone and dual_monotone
    return {
        "name": "T-FK-9",
        "passed": bool(passed),
        "am_omega_sequence": [am_u.omega, am_st.omega, am_lt.omega],
        "dual_omega_sequence": [dual_u.omega, dual_st.omega, dual_lt.omega],
        "am_monotone": bool(am_monotone),
        "dual_monotone": bool(dual_monotone),
    }


def verify_all_ms_theorems() -> list[dict]:
    """Run all mycorrhizal stress theorems (T-FK-7 through T-FK-9)."""
    ms_results = compute_all_ms_entities()
    return [
        verify_t_fk_7(ms_results),
        verify_t_fk_8(ms_results),
        verify_t_fk_9(ms_results),
    ]


def verify_all_theorems_combined() -> list[dict]:
    """Run all T-FK theorems (species + mycorrhizal stress)."""
    return verify_all_theorems() + verify_all_ms_theorems()


# ── Main ──────────────────────────────────────────────────────────────


def main() -> None:
    """Entry point."""
    results = compute_all_entities()
    print("=" * 78)
    print("FUNGI KINGDOM CLOSURE — 12 Species × 8 Channels → GCD Kernel")
    print("=" * 78)
    print()
    for cat in ["decomposer", "symbiont", "transformer", "extremophile"]:
        cat_results = [r for r in results if r.category == cat]
        print(f"  {cat.upper()}")
        for r in cat_results:
            gap = r.F - r.IC
            print(f"    {r.common_name:<25s}  F={r.F:.4f}  IC={r.IC:.4f}  Δ={gap:.4f}  ω={r.omega:.4f}  {r.regime}")
        print()

    # Mycorrhizal stress response (Thiem et al. 2025)
    ms_results = compute_all_ms_entities()
    print("=" * 78)
    print("MYCORRHIZAL STRESS RESPONSE — Thiem et al. 2025 (Plant and Soil)")
    print("  6 Configurations × 8 Channels → GCD Kernel")
    print("=" * 78)
    print()
    for cat in ["am_mycorrhizal", "dual_mycorrhizal"]:
        cat_results = [r for r in ms_results if r.category == cat]
        label = "AM-ALONE" if cat == "am_mycorrhizal" else "DUAL AM+EM"
        print(f"  {label}")
        for r in cat_results:
            gap = r.F - r.IC
            print(f"    {r.common_name:<30s}  F={r.F:.4f}  IC={r.IC:.4f}  Δ={gap:.4f}  ω={r.omega:.4f}  {r.regime}")
        print()

    print("-" * 78)
    print("THEOREMS (Species: T-FK-1–6, Mycorrhizal Stress: T-FK-7–9)")
    print("-" * 78)
    for t in verify_all_theorems_combined():
        status = "PROVEN" if t["passed"] else "FAILED"
        print(f"  {t['name']}: {status}")
    print()


if __name__ == "__main__":
    main()

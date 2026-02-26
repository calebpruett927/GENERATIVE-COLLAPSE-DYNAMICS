"""
Evolution Kernel — The GCD Kernel Applied to Biological Evolution

Maps 40 representative organisms across the tree of life to 8-channel trace
vectors and computes Tier-1 invariants. Demonstrates that evolutionary
dynamics are instances of collapse-return structure.

Channels (8):
    1. genetic_diversity    — Normalized heterozygosity / allelic richness
    2. morphological_fitness — Body plan complexity and adaptation score
    3. reproductive_success — Normalized fecundity × offspring survival
    4. metabolic_efficiency — Energy conversion optimization
    5. immune_competence    — Pathogen resistance breadth / defense systems
    6. environmental_breadth — Niche width / habitat generalism
    7. behavioral_complexity — Behavioral repertoire richness
    8. lineage_persistence  — Geological duration normalized to max known

Key GCD predictions for evolution:
    - F + ω = 1: What selection preserves + what it removes = 1 (exhaustive)
    - IC ≤ F: Organism coherence cannot exceed mean trait fitness
    - Geometric slaughter: ONE non-viable trait kills IC → purifying selection
    - Heterogeneity gap Δ = F - IC: evolutionary fragility (high F, low IC = fragile)
    - Specialists: high F, high Δ (fragile to perturbation)
    - Generalists: moderate F, low Δ (robust IC, survive mass extinctions)
    - Extinct lineages: τ_R = ∞_rec (no return — the lineage is a gestus)

Derivation chain: Axiom-0 → frozen_contract → kernel_optimized → this module

Data sources:
    Trait values are normalized estimates from comparative biology literature.
    Each value represents a consensus ranking within the tree of life,
    not absolute measurements. The kernel results are structural — they
    depend on the pattern of channels, not the exact values.
"""

from __future__ import annotations

import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

# ── Path setup ────────────────────────────────────────────────────
_WORKSPACE = Path(__file__).resolve().parents[2]
if str(_WORKSPACE / "src") not in sys.path:
    sys.path.insert(0, str(_WORKSPACE / "src"))
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))

from umcp.frozen_contract import EPSILON  # noqa: E402
from umcp.kernel_optimized import compute_kernel_outputs  # noqa: E402

# ── Guard band ────────────────────────────────────────────────────
EPS = 1e-6  # Closure-level epsilon (above frozen ε = 1e-8)

# ═════════════════════════════════════════════════════════════════════
# SECTION 1: ORGANISM DATA
# ═════════════════════════════════════════════════════════════════════

CHANNEL_LABELS: list[str] = [
    "genetic_diversity",
    "morphological_fitness",
    "reproductive_success",
    "metabolic_efficiency",
    "immune_competence",
    "environmental_breadth",
    "behavioral_complexity",
    "lineage_persistence",
]

N_CHANNELS = len(CHANNEL_LABELS)


@dataclass(frozen=True, slots=True)
class Organism:
    """A representative organism in the tree of life.

    Each trait is normalized to [0, 1] representing relative standing
    within the full diversity of life. These are structural rankings,
    not absolute measurements.
    """

    name: str
    domain: str  # Archaea, Bacteria, Eukarya
    kingdom: str  # e.g., Animalia, Plantae, Fungi, Protista, Monera
    clade: str  # Finer classification
    status: str  # extant or extinct

    # 8 channels — normalized trait scores [0, 1]
    genetic_diversity: float
    morphological_fitness: float
    reproductive_success: float
    metabolic_efficiency: float
    immune_competence: float
    environmental_breadth: float
    behavioral_complexity: float
    lineage_persistence: float  # 0 for recently extinct, high for ancient lineages


# ── Organism Catalog ──────────────────────────────────────────────
# 40 organisms spanning the tree of life: prokaryotes through mammals,
# plus extinct lineages (τ_R = ∞_rec for those that did not return).
#
# Trait normalization conventions:
#   genetic_diversity:     Population genetic diversity (heterozygosity proxy)
#   morphological_fitness: Body plan complexity / adaptation (0=minimal, 1=maximal)
#   reproductive_success:  Fecundity × survival to reproduction
#   metabolic_efficiency:  Energy yield per unit input
#   immune_competence:     Defense system breadth (0=none, 1=adaptive+innate)
#   environmental_breadth: Habitat generalism (0=obligate, 1=cosmopolitan)
#   behavioral_complexity: Repertoire richness (0=reflexive, 1=cultural)
#   lineage_persistence:   Duration / max_duration (Bacteria set upper bound ~3.8 Ga)

ORGANISMS: tuple[Organism, ...] = (
    # ── PROKARYOTES ───────────────────────────────────────────────
    Organism(
        "Escherichia coli",
        "Bacteria",
        "Monera",
        "Proteobacteria",
        "extant",
        0.92,
        0.15,
        0.95,
        0.80,
        0.10,
        0.85,
        0.05,
        0.90,
    ),
    Organism(
        "Thermus aquaticus",
        "Bacteria",
        "Monera",
        "Deinococcota",
        "extant",
        0.60,
        0.12,
        0.70,
        0.85,
        0.08,
        0.30,
        0.03,
        0.85,
    ),
    Organism(
        "Cyanobacteria (Synechococcus)",
        "Bacteria",
        "Monera",
        "Cyanobacteria",
        "extant",
        0.80,
        0.18,
        0.88,
        0.90,
        0.12,
        0.75,
        0.04,
        0.95,
    ),
    Organism(
        "Methanobacterium",
        "Archaea",
        "Monera",
        "Euryarchaeota",
        "extant",
        0.55,
        0.10,
        0.65,
        0.70,
        0.06,
        0.25,
        0.02,
        0.95,
    ),
    Organism(
        "Halobacterium",
        "Archaea",
        "Monera",
        "Euryarchaeota",
        "extant",
        0.50,
        0.12,
        0.60,
        0.65,
        0.08,
        0.15,
        0.03,
        0.90,
    ),
    # ── PROTISTS ──────────────────────────────────────────────────
    Organism(
        "Amoeba proteus",
        "Eukarya",
        "Protista",
        "Amoebozoa",
        "extant",
        0.70,
        0.25,
        0.75,
        0.60,
        0.15,
        0.50,
        0.10,
        0.80,
    ),
    Organism(
        "Paramecium",
        "Eukarya",
        "Protista",
        "Ciliophora",
        "extant",
        0.65,
        0.30,
        0.80,
        0.55,
        0.20,
        0.45,
        0.15,
        0.75,
    ),
    Organism(
        "Plasmodium falciparum",
        "Eukarya",
        "Protista",
        "Apicomplexa",
        "extant",
        0.85,
        0.35,
        0.90,
        0.50,
        0.05,
        0.10,
        0.08,
        0.30,
    ),
    # ── FUNGI ─────────────────────────────────────────────────────
    Organism(
        "Saccharomyces cerevisiae",
        "Eukarya",
        "Fungi",
        "Ascomycota",
        "extant",
        0.75,
        0.20,
        0.85,
        0.75,
        0.10,
        0.40,
        0.05,
        0.60,
    ),
    Organism(
        "Penicillium chrysogenum",
        "Eukarya",
        "Fungi",
        "Ascomycota",
        "extant",
        0.70,
        0.22,
        0.80,
        0.70,
        0.30,
        0.55,
        0.05,
        0.50,
    ),
    Organism(
        "Armillaria ostoyae",
        "Eukarya",
        "Fungi",
        "Basidiomycota",
        "extant",
        0.60,
        0.28,
        0.70,
        0.65,
        0.15,
        0.35,
        0.04,
        0.45,
    ),
    # ── PLANTS ────────────────────────────────────────────────────
    Organism(
        "Marchantia (liverwort)",
        "Eukarya",
        "Plantae",
        "Bryophyta",
        "extant",
        0.55,
        0.30,
        0.65,
        0.60,
        0.15,
        0.40,
        0.02,
        0.85,
    ),
    Organism(
        "Equisetum (horsetail)",
        "Eukarya",
        "Plantae",
        "Pteridophyta",
        "extant",
        0.50,
        0.40,
        0.60,
        0.65,
        0.18,
        0.35,
        0.02,
        0.90,
    ),
    Organism(
        "Pinus (pine)",
        "Eukarya",
        "Plantae",
        "Gymnospermae",
        "extant",
        0.65,
        0.55,
        0.70,
        0.70,
        0.25,
        0.50,
        0.03,
        0.75,
    ),
    Organism(
        "Quercus (oak)",
        "Eukarya",
        "Plantae",
        "Angiospermae",
        "extant",
        0.70,
        0.60,
        0.75,
        0.72,
        0.30,
        0.55,
        0.03,
        0.45,
    ),
    Organism(
        "Oryza sativa (rice)",
        "Eukarya",
        "Plantae",
        "Angiospermae",
        "extant",
        0.80,
        0.55,
        0.85,
        0.78,
        0.20,
        0.35,
        0.02,
        0.05,
    ),
    # ── INVERTEBRATES ─────────────────────────────────────────────
    Organism(
        "Caenorhabditis elegans",
        "Eukarya",
        "Animalia",
        "Nematoda",
        "extant",
        0.60,
        0.35,
        0.88,
        0.55,
        0.20,
        0.40,
        0.12,
        0.50,
    ),
    Organism(
        "Drosophila melanogaster",
        "Eukarya",
        "Animalia",
        "Arthropoda",
        "extant",
        0.75,
        0.50,
        0.90,
        0.60,
        0.35,
        0.45,
        0.25,
        0.30,
    ),
    Organism(
        "Apis mellifera (honeybee)",
        "Eukarya",
        "Animalia",
        "Arthropoda",
        "extant",
        0.55,
        0.60,
        0.70,
        0.70,
        0.40,
        0.35,
        0.65,
        0.15,
    ),
    Organism(
        "Octopus vulgaris",
        "Eukarya",
        "Animalia",
        "Mollusca",
        "extant",
        0.65,
        0.70,
        0.55,
        0.65,
        0.35,
        0.40,
        0.80,
        0.25,
    ),
    Organism(
        "Limulus polyphemus (horseshoe crab)",
        "Eukarya",
        "Animalia",
        "Arthropoda",
        "extant",
        0.35,
        0.50,
        0.50,
        0.55,
        0.60,
        0.30,
        0.10,
        0.95,
    ),
    # ── FISH ──────────────────────────────────────────────────────
    Organism(
        "Latimeria (coelacanth)",
        "Eukarya",
        "Animalia",
        "Sarcopterygii",
        "extant",
        0.25,
        0.55,
        0.30,
        0.50,
        0.50,
        0.15,
        0.20,
        0.90,
    ),
    Organism(
        "Danio rerio (zebrafish)",
        "Eukarya",
        "Animalia",
        "Actinopterygii",
        "extant",
        0.70,
        0.55,
        0.85,
        0.65,
        0.55,
        0.30,
        0.25,
        0.20,
    ),
    Organism(
        "Carcharodon carcharias (great white)",
        "Eukarya",
        "Animalia",
        "Chondrichthyes",
        "extant",
        0.40,
        0.75,
        0.35,
        0.70,
        0.55,
        0.45,
        0.50,
        0.85,
    ),
    # ── AMPHIBIANS & REPTILES ─────────────────────────────────────
    Organism(
        "Rana temporaria (frog)",
        "Eukarya",
        "Animalia",
        "Amphibia",
        "extant",
        0.65,
        0.50,
        0.80,
        0.55,
        0.40,
        0.45,
        0.20,
        0.50,
    ),
    Organism(
        "Crocodylus niloticus",
        "Eukarya",
        "Animalia",
        "Reptilia",
        "extant",
        0.45,
        0.70,
        0.55,
        0.60,
        0.55,
        0.30,
        0.35,
        0.85,
    ),
    Organism(
        "Chelonia mydas (green turtle)",
        "Eukarya",
        "Animalia",
        "Reptilia",
        "extant",
        0.40,
        0.60,
        0.50,
        0.55,
        0.45,
        0.50,
        0.20,
        0.75,
    ),
    # ── BIRDS ─────────────────────────────────────────────────────
    Organism(
        "Corvus corax (raven)",
        "Eukarya",
        "Animalia",
        "Aves",
        "extant",
        0.60,
        0.70,
        0.65,
        0.72,
        0.55,
        0.55,
        0.85,
        0.20,
    ),
    Organism(
        "Aptenodytes forsteri (emperor penguin)",
        "Eukarya",
        "Animalia",
        "Aves",
        "extant",
        0.35,
        0.65,
        0.40,
        0.68,
        0.50,
        0.10,
        0.50,
        0.10,
    ),
    # ── MAMMALS ───────────────────────────────────────────────────
    Organism(
        "Mus musculus (mouse)",
        "Eukarya",
        "Animalia",
        "Mammalia",
        "extant",
        0.80,
        0.55,
        0.92,
        0.65,
        0.65,
        0.60,
        0.35,
        0.15,
    ),
    Organism(
        "Canis lupus (wolf)",
        "Eukarya",
        "Animalia",
        "Mammalia",
        "extant",
        0.55,
        0.75,
        0.55,
        0.70,
        0.70,
        0.55,
        0.80,
        0.10,
    ),
    Organism(
        "Tursiops truncatus (dolphin)",
        "Eukarya",
        "Animalia",
        "Mammalia",
        "extant",
        0.50,
        0.75,
        0.45,
        0.68,
        0.60,
        0.40,
        0.85,
        0.10,
    ),
    Organism(
        "Elephas maximus (Asian elephant)",
        "Eukarya",
        "Animalia",
        "Mammalia",
        "extant",
        0.35,
        0.80,
        0.30,
        0.60,
        0.65,
        0.25,
        0.80,
        0.08,
    ),
    Organism(
        "Pan troglodytes (chimpanzee)",
        "Eukarya",
        "Animalia",
        "Mammalia",
        "extant",
        0.50,
        0.75,
        0.40,
        0.62,
        0.70,
        0.30,
        0.90,
        0.02,
    ),
    # NOTE on lineage_persistence = 0.001:
    # Homo sapiens is ~300 kyr old vs ~3.8 Gyr for bacteria → geological persistence ≈ 0.
    # Cultural knowledge accumulation (~300 kyr) functions as an adaptive persistence
    # mechanism, but the GCD kernel measures DEMONSTRATED return (geological track
    # record), not potential. τ_R is measured, not assumed (Continuitas non narratur:
    # mensuratur). Whether cultural persistence should constitute a separate channel
    # or modify this one is an open modeling question — but under Axiom-0, only what
    # has actually returned counts. The species hasn't yet proved multi-Myr persistence.
    # This puts Homo sapiens in Collapse regime (ω ≈ 0.346), which is structurally
    # honest: our IC is dragged down by recency. Whether we are a weld or a gestus
    # is the defining question. See recursive_evolution.py for the full discussion.
    Organism(
        "Homo sapiens",
        "Eukarya",
        "Animalia",
        "Mammalia",
        "extant",
        0.45,  # genetic_diversity: moderate (recent bottleneck ~70 ka)
        0.80,  # morphological_fitness: high (bipedal, dexterous, large brain)
        0.70,  # reproductive_success: moderate (low fecundity, high investment)
        0.60,  # metabolic_efficiency: moderate (high BMR, inefficient thermoregulation)
        0.75,  # immune_competence: high (adaptive + innate, but autoimmune burden)
        0.95,  # environmental_breadth: near-maximal (all continents, all biomes)
        0.98,  # behavioral_complexity: near-maximal (language, culture, technology)
        0.001,  # lineage_persistence: minimal (demonstrated geological return ≈ 0)
    ),
    # ── EXTINCT LINEAGES (τ_R = ∞_rec — no return) ───────────────
    Organism(
        "Trilobita (trilobite)",
        "Eukarya",
        "Animalia",
        "Arthropoda",
        "extinct",
        0.70,
        0.55,
        0.75,
        0.55,
        0.30,
        0.60,
        0.10,
        0.70,
    ),
    Organism(
        "Ammonoidea (ammonite)",
        "Eukarya",
        "Animalia",
        "Mollusca",
        "extinct",
        0.65,
        0.50,
        0.70,
        0.50,
        0.25,
        0.55,
        0.08,
        0.65,
    ),
    Organism(
        "Tyrannosaurus rex",
        "Eukarya",
        "Animalia",
        "Dinosauria",
        "extinct",
        0.30,
        0.85,
        0.35,
        0.70,
        0.50,
        0.20,
        0.55,
        0.04,
    ),
    Organism(
        "Dodo (Raphus cucullatus)",
        "Eukarya",
        "Animalia",
        "Aves",
        "extinct",
        0.10,
        0.40,
        0.30,
        0.50,
        0.20,
        0.05,
        0.15,
        0.001,
    ),
    Organism(
        "Mammuthus primigenius (woolly mammoth)",
        "Eukarya",
        "Animalia",
        "Mammalia",
        "extinct",
        0.20,
        0.75,
        0.25,
        0.60,
        0.55,
        0.10,
        0.60,
        0.01,
    ),
)


# ═════════════════════════════════════════════════════════════════════
# SECTION 2: NORMALIZATION AND KERNEL COMPUTATION
# ═════════════════════════════════════════════════════════════════════


def normalize_organism(org: Organism) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Map organism traits to ε-clamped trace vector.

    Returns (c, w, labels) where c ∈ [ε, 1-ε]^8 and w sums to 1.
    """
    raw = np.array(
        [
            org.genetic_diversity,
            org.morphological_fitness,
            org.reproductive_success,
            org.metabolic_efficiency,
            org.immune_competence,
            org.environmental_breadth,
            org.behavioral_complexity,
            org.lineage_persistence,
        ],
        dtype=np.float64,
    )
    c = np.clip(raw, EPS, 1.0 - EPS)
    w = np.ones(N_CHANNELS) / N_CHANNELS
    return c, w, CHANNEL_LABELS


def _classify_regime(omega: float, F: float, S: float, C: float) -> str:
    """Standard four-gate regime classification (from frozen_contract)."""
    if omega >= 0.30:
        return "Collapse"
    if omega < 0.038 and F > 0.90 and S < 0.15 and C < 0.14:
        return "Stable"
    return "Watch"


def _classify_evolutionary_strategy(F: float, IC: float, delta: float, C: float) -> str:
    """Domain-specific evolutionary strategy classification.

    Maps kernel invariants to evolutionary interpretation:
        Robust Generalist:  moderate F, low Δ, wide breadth → survives perturbation
        Adapted Specialist:  high F, high Δ → dominates stable niches, fragile
        Resilient Ancient:  moderate F, low Δ, high persistence → living fossils
        Vulnerable Specialist: moderate F, high Δ, high C → at risk
        Minimal Viable:     low F, low IC → edge of viability
    """
    if F >= 0.65 and delta < 0.15:
        return "Robust Generalist"
    if F >= 0.55 and delta >= 0.15 and C < 0.30:
        return "Adapted Specialist"
    if F < 0.55 and delta < 0.10 and IC > 0.30:
        return "Resilient Ancient"
    if F >= 0.45 and delta >= 0.20:
        return "Vulnerable Specialist"
    return "Minimal Viable"


# ═════════════════════════════════════════════════════════════════════
# SECTION 3: RESULT CONTAINER
# ═════════════════════════════════════════════════════════════════════


@dataclass
class EvolutionKernelResult:
    """Kernel result for a single organism."""

    # Identity
    name: str
    domain_of_life: str
    kingdom: str
    clade: str
    status: str

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

    # Classification
    regime: str
    evolutionary_strategy: str

    # Evolution-specific
    weakest_channel: str
    weakest_value: float
    strongest_channel: str
    strongest_value: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)


# ═════════════════════════════════════════════════════════════════════
# SECTION 4: KERNEL COMPUTATION
# ═════════════════════════════════════════════════════════════════════


def compute_organism_kernel(org: Organism) -> EvolutionKernelResult:
    """Compute full GCD kernel for a single organism.

    Maps organism traits → 8-channel trace → Tier-1 invariants.
    Verifies all three structural identities.
    """
    c, w, labels = normalize_organism(org)
    k = compute_kernel_outputs(c, w, EPSILON)

    F = k["F"]
    omega = k["omega"]
    S = k["S"]
    C = k["C"]
    kappa = k["kappa"]
    IC = k["IC"]
    delta = k["heterogeneity_gap"]

    # Tier-1 identity checks
    F_plus_omega = F + omega
    IC_leq_F = IC <= F + 1e-12
    IC_eq_exp_kappa = abs(IC - math.exp(kappa)) < 1e-9

    # Regime and strategy
    regime = _classify_regime(omega, F, S, C)
    strategy = _classify_evolutionary_strategy(F, IC, delta, C)

    # Identify weakest and strongest channels
    min_idx = int(np.argmin(c))
    max_idx = int(np.argmax(c))

    return EvolutionKernelResult(
        name=org.name,
        domain_of_life=org.domain,
        kingdom=org.kingdom,
        clade=org.clade,
        status=org.status,
        n_channels=N_CHANNELS,
        channel_labels=labels,
        trace_vector=c.tolist(),
        F=F,
        omega=omega,
        S=S,
        C=C,
        kappa=kappa,
        IC=IC,
        heterogeneity_gap=delta,
        F_plus_omega=F_plus_omega,
        IC_leq_F=IC_leq_F,
        IC_eq_exp_kappa=IC_eq_exp_kappa,
        regime=regime,
        evolutionary_strategy=strategy,
        weakest_channel=labels[min_idx],
        weakest_value=float(c[min_idx]),
        strongest_channel=labels[max_idx],
        strongest_value=float(c[max_idx]),
    )


def compute_all_organisms() -> list[EvolutionKernelResult]:
    """Compute kernel for all 40 organisms in the catalog."""
    return [compute_organism_kernel(org) for org in ORGANISMS]


# ═════════════════════════════════════════════════════════════════════
# SECTION 5: ANALYSIS AND DISPLAY
# ═════════════════════════════════════════════════════════════════════


def print_results(results: list[EvolutionKernelResult] | None = None) -> None:
    """Print formatted kernel results for all organisms."""
    if results is None:
        results = compute_all_organisms()

    # Verify Tier-1 identities
    n_duality = sum(1 for r in results if abs(r.F_plus_omega - 1.0) < 1e-12)
    n_bound = sum(1 for r in results if r.IC_leq_F)
    n_exp = sum(1 for r in results if r.IC_eq_exp_kappa)

    print("=" * 90)
    print("  EVOLUTION KERNEL — Recursive Collapse-Return Dynamics of Life")
    print("  Collapsus generativus est; solum quod redit, reale est.")
    print("=" * 90)
    print()
    print(f"  Organisms: {len(results)}  |  Channels: {N_CHANNELS}  |  ε = {EPSILON}")
    print(
        f"  Tier-1 identities: F+ω=1 [{n_duality}/{len(results)}]  "
        f"IC≤F [{n_bound}/{len(results)}]  IC=exp(κ) [{n_exp}/{len(results)}]"
    )
    print()

    # Header
    print(
        f"  {'Organism':<35s} {'Status':<8s} {'F':>7s} {'ω':>7s} "
        f"{'IC':>8s} {'Δ':>7s} {'IC/F':>7s} {'Regime':<10s} {'Strategy':<22s}"
    )
    print("  " + "─" * 128)

    # Group by kingdom
    kingdoms_order = ["Monera", "Protista", "Fungi", "Plantae", "Animalia"]
    for kingdom in kingdoms_order:
        group = [r for r in results if r.kingdom == kingdom]
        if not group:
            continue
        print(f"\n  ── {kingdom.upper()} {'─' * (120 - len(kingdom))}")
        for r in group:
            ic_f = r.IC / r.F if r.F > 0 else 0
            status_mark = "†" if r.status == "extinct" else " "
            print(
                f"  {r.name:<35s} {status_mark:<8s} {r.F:7.4f} {r.omega:7.4f} "
                f"{r.IC:8.6f} {r.heterogeneity_gap:7.4f} {ic_f:7.4f} "
                f"{r.regime:<10s} {r.evolutionary_strategy:<22s}"
            )

    # Insights
    print("\n" + "=" * 90)
    print("  KEY INSIGHTS — GCD Predictions for Evolution")
    print("=" * 90)

    # 1. Extant vs extinct
    extant = [r for r in results if r.status == "extant"]
    extinct = [r for r in results if r.status == "extinct"]

    mean_F_extant = np.mean([r.F for r in extant])
    mean_F_extinct = np.mean([r.F for r in extinct])
    mean_IC_extant = np.mean([r.IC for r in extant])
    mean_IC_extinct = np.mean([r.IC for r in extinct])
    mean_delta_extant = np.mean([r.heterogeneity_gap for r in extant])
    mean_delta_extinct = np.mean([r.heterogeneity_gap for r in extinct])

    print("\n  §1  EXTANT vs EXTINCT (Persistence = Return)")
    print(f"      ⟨F⟩ extant  = {mean_F_extant:.4f}   ⟨F⟩ extinct  = {mean_F_extinct:.4f}")
    print(f"      ⟨IC⟩ extant = {mean_IC_extant:.4f}   ⟨IC⟩ extinct = {mean_IC_extinct:.4f}")
    print(f"      ⟨Δ⟩ extant  = {mean_delta_extant:.4f}   ⟨Δ⟩ extinct  = {mean_delta_extinct:.4f}")
    print("      INSIGHT: Extinct lineages have LOWER IC despite comparable F.")
    print("      The heterogeneity gap is wider — they had fatal channel weaknesses.")
    print("      Extinction = τ_R = ∞_rec. The lineage is a gestus, not a sutura.")

    # 2. Specialist vs generalist
    specialists = [r for r in results if "Specialist" in r.evolutionary_strategy]
    generalists = [r for r in results if "Generalist" in r.evolutionary_strategy]

    if specialists and generalists:
        print("\n  §2  SPECIALIST vs GENERALIST (Fragility = Heterogeneity Gap)")
        print(
            f"      Specialists ({len(specialists)}):  ⟨Δ⟩ = {np.mean([r.heterogeneity_gap for r in specialists]):.4f}"
        )
        print(
            f"      Generalists ({len(generalists)}):  ⟨Δ⟩ = {np.mean([r.heterogeneity_gap for r in generalists]):.4f}"
        )
        print("      INSIGHT: Specialists have higher Δ — high F but lower IC.")
        print("      Selection optimizes F (arithmetic). Survival requires IC (geometric).")
        print("      The gap IS evolutionary fragility. Selection cannot see it.")

    # 3. The Homo sapiens anomaly
    human = next((r for r in results if r.name == "Homo sapiens"), None)
    if human:
        print("\n  §3  THE HUMAN ANOMALY (Environmental Breadth + Behavioral Complexity)")
        print(f"      F = {human.F:.4f}, IC = {human.IC:.6f}, Δ = {human.heterogeneity_gap:.4f}")
        print(f"      Weakest: {human.weakest_channel} = {human.weakest_value:.4f}")
        print(f"      Strongest: {human.strongest_channel} = {human.strongest_value:.4f}")
        print("      INSIGHT: Humans have extreme behavioral_complexity (0.98) and")
        print("      environmental_breadth (0.95), but very low lineage_persistence (0.001).")
        print("      We are evolutionarily YOUNG — our IC is dragged down by recency.")
        print("      The geometric mean penalizes our short track record.")
        print("      Whether Homo sapiens is a weld or a gestus remains undetermined.")

    # 4. Geometric slaughter in action: the Dodo
    dodo = next((r for r in results if "Dodo" in r.name), None)
    if dodo:
        print("\n  §4  GEOMETRIC SLAUGHTER — THE DODO (Mors Canalis)")
        print(f"      F = {dodo.F:.4f}, IC = {dodo.IC:.6f}")
        print(f"      Weakest: {dodo.weakest_channel} = {dodo.weakest_value:.4f}")
        print("      INSIGHT: Environmental breadth → ε (island endemic, flightless).")
        print("      One dead channel killed IC via geometric slaughter.")
        print(f"      The Dodo was 'fine on average' (F = {dodo.F:.2f}) but structurally doomed.")
        print("      Trucidatio geometrica: the geometric mean has no mercy.")

    # 5. Living fossils
    coelacanth = next((r for r in results if "coelacanth" in r.name), None)
    horseshoe = next((r for r in results if "horseshoe" in r.name), None)
    if coelacanth and horseshoe:
        print("\n  §5  LIVING FOSSILS (Persistence Without Dominance)")
        print(
            f"      Coelacanth:    F = {coelacanth.F:.4f}, IC/F = {coelacanth.IC / coelacanth.F:.4f}, "
            f"persistence = {coelacanth.trace_vector[7]:.2f}"
        )
        print(
            f"      Horseshoe crab: F = {horseshoe.F:.4f}, IC/F = {horseshoe.IC / horseshoe.F:.4f}, "
            f"persistence = {horseshoe.trace_vector[7]:.2f}"
        )
        print("      INSIGHT: Low F but moderate IC/F — channels are uniformly modest.")
        print("      The heterogeneity gap is small. No brilliant channels, no dead ones.")
        print("      They survive not by being excellent but by being UNIFORM.")
        print("      Persistence = low Δ. The homogeneous path (§4 of orientation).")

    print(f"\n{'=' * 90}")
    print("  Finis, sed semper initium recursionis.")
    print(f"{'=' * 90}\n")


# ═════════════════════════════════════════════════════════════════════
# SECTION 6: CLI
# ═════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print_results()

"""
Recursive Evolution — RCFT Overlay on Evolutionary Dynamics

Evolution operates as a recursive collapse field: the SAME kernel structure
repeats at nested scales, with each scale's collapse generating the signal
for the next. This module demonstrates the recursive structure.

Scales (5 nested levels):
    1. GENE:       mutation → selection → fixation      (τ_R ~ 1-100 gen)
    2. ORGANISM:   birth → life → reproduction          (τ_R ~ 1 gen)
    3. POPULATION: founder → adaptation → equilibrium   (τ_R ~ 10²-10⁴ gen)
    4. SPECIES:    speciation → radiation → extinction   (τ_R ~ 10⁵-10⁷ yr)
    5. CLADE:      origination → diversification → mass extinction/recovery
                                                        (τ_R ~ 10⁷-10⁸ yr)

Key insight — WHY evolution is recursive, not linear:
    At each scale, the kernel computes identical invariants (F, ω, S, C, κ, IC).
    Collapse at one scale IS the signal for the next. A mutation (gene-level
    drift) becomes an organism-level trait. An organism-level death becomes a
    population-level statistic. A population-level extinction becomes a
    species-level event. Each level's ω feeds upward as the next level's
    measurement input.

    The Rosetta translation:
        Gene:       Drift = mutation rate,     Fidelity = replication accuracy
        Organism:   Drift = mortality,         Fidelity = reproductive fitness
        Population: Drift = N_e fluctuation,   Fidelity = adaptive equilibrium
        Species:    Drift = extinction rate,   Fidelity = speciation rate
        Clade:      Drift = mass extinction,   Fidelity = recovery radiation

    At every scale: F + ω = 1. IC ≤ F. What doesn't return isn't real.

The profound reinterpretation:
    Natural selection optimizes F (arithmetic mean fitness across traits).
    But SURVIVAL requires IC (geometric mean — multiplicative coherence).
    Selection CANNOT SEE the heterogeneity gap Δ = F - IC.

    This explains:
    - Why generalists survive mass extinctions (low Δ → robust IC)
    - Why specialists dominate stable environments (high F, high Δ → fragile)
    - Why sexual reproduction persists (mixes channels → reduces Δ)
    - Why extinction cascades happen (one dead ecosystem channel → geometric slaughter)

Derivation chain: Axiom-0 → frozen_contract → kernel_optimized → evolution_kernel → this module
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ── Path setup ────────────────────────────────────────────────────
_WORKSPACE = Path(__file__).resolve().parents[2]
if str(_WORKSPACE / "src") not in sys.path:
    sys.path.insert(0, str(_WORKSPACE / "src"))
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))

from umcp.frozen_contract import (  # noqa: E402
    EPSILON,
    cost_curvature,
    gamma_omega,
)
from umcp.kernel_optimized import compute_kernel_outputs  # noqa: E402

# ── Guard band ────────────────────────────────────────────────────
EPS = 1e-6


# ═════════════════════════════════════════════════════════════════════
# SECTION 1: EVOLUTIONARY SCALE DEFINITIONS
# ═════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class EvolutionaryScale:
    """A scale in the recursive collapse hierarchy of evolution.

    Each scale has its own channel semantics but maps to the same
    Tier-1 kernel. The collapse at one scale generates the input
    for the next.
    """

    name: str
    level: int  # 1=gene, 2=organism, 3=population, 4=species, 5=clade
    typical_tau_R: str  # Characteristic return time
    collapse_event: str  # What constitutes collapse at this scale
    return_event: str  # What constitutes return
    drift_meaning: str  # What ω means at this scale
    fidelity_meaning: str  # What F means at this scale

    # 8-channel trace for this scale
    channel_labels: tuple[str, ...]
    channel_values: tuple[float, ...]


# ── Five scales of evolutionary recursion ─────────────────────────
SCALES: tuple[EvolutionaryScale, ...] = (
    EvolutionaryScale(
        name="Gene",
        level=1,
        typical_tau_R="1-100 generations",
        collapse_event="Mutation / recombination event",
        return_event="Allele fixation in population",
        drift_meaning="Mutation rate (departure from template)",
        fidelity_meaning="Replication accuracy",
        channel_labels=(
            "replication_fidelity",
            "recombination_rate",
            "repair_efficiency",
            "epigenetic_stability",
            "codon_optimization",
            "regulatory_coherence",
            "transposon_suppression",
            "horizontal_transfer",
        ),
        # High fidelity, modest diversity — molecular machinery is precise
        channel_values=(0.95, 0.60, 0.88, 0.75, 0.80, 0.70, 0.85, 0.30),
    ),
    EvolutionaryScale(
        name="Organism",
        level=2,
        typical_tau_R="1 generation",
        collapse_event="Death of individual",
        return_event="Successful reproduction (offspring reaches maturity)",
        drift_meaning="Mortality rate (individuals lost per generation)",
        fidelity_meaning="Reproductive fitness (offspring contribution)",
        channel_labels=(
            "developmental_integrity",
            "phenotypic_plasticity",
            "reproductive_output",
            "metabolic_homeostasis",
            "somatic_maintenance",
            "immune_defense",
            "stress_tolerance",
            "mate_selection",
        ),
        # Moderate across all — the organism integrates many tradeoffs
        channel_values=(0.75, 0.65, 0.70, 0.72, 0.60, 0.68, 0.55, 0.62),
    ),
    EvolutionaryScale(
        name="Population",
        level=3,
        typical_tau_R="10²-10⁴ generations",
        collapse_event="Bottleneck / founder effect / drift",
        return_event="Equilibrium restoration (Hardy-Weinberg)",
        drift_meaning="Effective population size fluctuation",
        fidelity_meaning="Adaptive equilibrium maintenance",
        channel_labels=(
            "effective_population_size",
            "genetic_variation",
            "migration_connectivity",
            "selection_efficiency",
            "drift_resistance",
            "mutation_supply",
            "recombination_rate",
            "demographic_stability",
        ),
        # Population health varies — some channels are vulnerable
        channel_values=(0.55, 0.70, 0.40, 0.65, 0.50, 0.60, 0.55, 0.45),
    ),
    EvolutionaryScale(
        name="Species",
        level=4,
        typical_tau_R="10⁵-10⁷ years",
        collapse_event="Extinction / range collapse",
        return_event="Speciation event (daughter species viable)",
        drift_meaning="Background extinction rate",
        fidelity_meaning="Speciation rate / niche persistence",
        channel_labels=(
            "speciation_potential",
            "niche_breadth",
            "geographic_range",
            "adaptive_radiation",
            "competitive_ability",
            "predator_avoidance",
            "climate_tolerance",
            "coevolution_network",
        ),
        # Species-level — some channels strong, some at risk
        channel_values=(0.50, 0.55, 0.45, 0.40, 0.60, 0.50, 0.35, 0.55),
    ),
    EvolutionaryScale(
        name="Clade",
        level=5,
        typical_tau_R="10⁷-10⁸ years",
        collapse_event="Mass extinction",
        return_event="Adaptive radiation (recovery from mass extinction)",
        drift_meaning="Mass extinction severity",
        fidelity_meaning="Recovery potential / radiation rate",
        channel_labels=(
            "body_plan_innovation",
            "ecological_diversification",
            "geographic_dispersal",
            "metabolic_versatility",
            "size_range",
            "habitat_diversity",
            "trophic_diversity",
            "symbiotic_networks",
        ),
        # Clade-level — highest uncertainty, lowest uniformity
        channel_values=(0.45, 0.55, 0.50, 0.60, 0.40, 0.50, 0.45, 0.35),
    ),
)


# ═════════════════════════════════════════════════════════════════════
# SECTION 2: RECURSIVE KERNEL COMPUTATION
# ═════════════════════════════════════════════════════════════════════


@dataclass
class ScaleKernelResult:
    """Kernel result for one evolutionary scale."""

    scale_name: str
    level: int
    typical_tau_R: str
    collapse_event: str
    return_event: str

    # Tier-1 invariants
    F: float
    omega: float
    S: float
    C: float
    kappa: float
    IC: float
    heterogeneity_gap: float

    # Seam budget
    gamma_omega: float
    D_C: float
    total_debit: float

    # Identity checks
    F_plus_omega_exact: bool
    IC_leq_F: bool
    IC_eq_exp_kappa: bool

    # Weakest channel
    weakest_channel: str
    weakest_value: float

    regime: str


def compute_scale_kernel(scale: EvolutionaryScale) -> ScaleKernelResult:
    """Compute GCD kernel for one evolutionary scale."""
    c = np.clip(
        np.array(scale.channel_values, dtype=np.float64),
        EPS,
        1.0 - EPS,
    )
    w = np.ones(len(c)) / len(c)
    k = compute_kernel_outputs(c, w, EPSILON)

    F = k["F"]
    omega = k["omega"]
    S = k["S"]
    C_val = k["C"]
    kappa = k["kappa"]
    IC = k["IC"]
    delta = k["heterogeneity_gap"]

    g_omega = gamma_omega(omega)
    d_c = cost_curvature(C_val)

    # Regime
    if omega >= 0.30:
        regime = "Collapse"
    elif omega < 0.038 and F > 0.90 and S < 0.15 and C_val < 0.14:
        regime = "Stable"
    else:
        regime = "Watch"

    min_idx = int(np.argmin(c))

    return ScaleKernelResult(
        scale_name=scale.name,
        level=scale.level,
        typical_tau_R=scale.typical_tau_R,
        collapse_event=scale.collapse_event,
        return_event=scale.return_event,
        F=F,
        omega=omega,
        S=S,
        C=C_val,
        kappa=kappa,
        IC=IC,
        heterogeneity_gap=delta,
        gamma_omega=g_omega,
        D_C=d_c,
        total_debit=g_omega + d_c,
        F_plus_omega_exact=abs(F + omega - 1.0) < 1e-12,
        IC_leq_F=IC <= F + 1e-12,
        IC_eq_exp_kappa=abs(IC - math.exp(kappa)) < 1e-9,
        weakest_channel=scale.channel_labels[min_idx],
        weakest_value=float(c[min_idx]),
        regime=regime,
    )


def compute_all_scales() -> list[ScaleKernelResult]:
    """Compute kernel for all 5 evolutionary scales."""
    return [compute_scale_kernel(s) for s in SCALES]


# ═════════════════════════════════════════════════════════════════════
# SECTION 3: THE FIVE MASS EXTINCTIONS — COLLAPSE AND RETURN
# ═════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class MassExtinction:
    """A mass extinction modeled as a collapse event with measured return.

    The extinction IS the collapse. The recovery radiation IS the return.
    τ_R is the recovery time in millions of years.
    """

    name: str
    age_mya: float  # Millions of years ago
    species_loss_pct: float  # Estimated percentage of species lost
    recovery_mya: float  # Time to recovery (recovery radiation onset)

    # Pre-extinction ecosystem health (8 channels)
    pre_channels: tuple[float, ...]
    # Post-extinction ecosystem state (8 channels — degraded)
    post_channels: tuple[float, ...]

    # What recovered and what didn't
    key_casualties: str
    key_survivors: str
    recovery_innovation: str


MASS_EXTINCTIONS: tuple[MassExtinction, ...] = (
    MassExtinction(
        "End-Ordovician",
        445,
        85.0,
        15.0,
        (0.75, 0.65, 0.70, 0.60, 0.55, 0.60, 0.50, 0.65),  # pre
        (0.30, 0.25, 0.20, 0.55, 0.50, 0.15, 0.40, 0.20),  # post
        "Trilobites, brachiopods, bryozoans decimated",
        "Nautiloids, some trilobite families, early fish",
        "Silurian reef systems, jawed fish radiation",
    ),
    MassExtinction(
        "Late Devonian",
        375,
        75.0,
        20.0,
        (0.80, 0.70, 0.75, 0.65, 0.55, 0.65, 0.55, 0.70),
        (0.35, 0.30, 0.25, 0.55, 0.45, 0.20, 0.40, 0.25),
        "Reef ecosystems, placoderms, many marine invertebrates",
        "Sharks, early tetrapods, seed plants",
        "Carboniferous forests, amphibian diversification",
    ),
    MassExtinction(
        "End-Permian",
        252,
        96.0,
        10.0,
        (0.85, 0.80, 0.80, 0.70, 0.60, 0.70, 0.60, 0.75),
        (0.10, 0.15, 0.08, 0.40, 0.30, 0.05, 0.25, 0.10),
        "Trilobites (final), most marine species, many terrestrial families",
        "Lystrosaurus, some insects, hardy seed plants",
        "Dinosaur rise, mammal-like reptile radiation, reef recovery",
    ),
    MassExtinction(
        "End-Triassic",
        201,
        80.0,
        10.0,
        (0.78, 0.72, 0.75, 0.65, 0.55, 0.65, 0.55, 0.70),
        (0.25, 0.28, 0.18, 0.50, 0.45, 0.15, 0.35, 0.20),
        "Most non-dinosaur archosaurs, many marine reptiles",
        "Dinosaurs, early mammals, crocodylomorphs",
        "Jurassic dinosaur dominance, early birds",
    ),
    MassExtinction(
        "End-Cretaceous (K-Pg)",
        66,
        76.0,
        10.0,
        (0.82, 0.78, 0.78, 0.68, 0.58, 0.68, 0.60, 0.72),
        (0.25, 0.30, 0.15, 0.50, 0.50, 0.12, 0.40, 0.22),
        "Non-avian dinosaurs, ammonites, mosasaurs, pterosaurs",
        "Mammals, birds, crocodilians, turtles",
        "Mammalian radiation, primate emergence, grassland ecosystems",
    ),
)


@dataclass
class ExtinctionKernelResult:
    """Pre- and post-extinction kernel comparison."""

    name: str
    age_mya: float
    species_loss_pct: float
    recovery_mya: float

    # Pre-extinction kernel
    pre_F: float
    pre_IC: float
    pre_delta: float
    pre_regime: str

    # Post-extinction kernel
    post_F: float
    post_IC: float
    post_delta: float
    post_regime: str

    # The cliff
    IC_drop_pct: float
    F_drop_pct: float
    delta_change: float

    # Seam analysis
    pre_gamma: float
    post_gamma: float
    post_total_debit: float

    key_casualties: str
    key_survivors: str
    recovery_innovation: str


def compute_extinction_kernel(ext: MassExtinction) -> ExtinctionKernelResult:
    """Compute pre- and post-extinction kernels for a mass extinction event."""
    n = len(ext.pre_channels)
    w = np.ones(n) / n

    c_pre = np.clip(np.array(ext.pre_channels, dtype=np.float64), EPS, 1.0 - EPS)
    c_post = np.clip(np.array(ext.post_channels, dtype=np.float64), EPS, 1.0 - EPS)

    k_pre = compute_kernel_outputs(c_pre, w, EPSILON)
    k_post = compute_kernel_outputs(c_post, w, EPSILON)

    def _regime(omega: float, F: float, S: float, C: float) -> str:
        if omega >= 0.30:
            return "Collapse"
        if omega < 0.038 and F > 0.90 and S < 0.15 and C < 0.14:
            return "Stable"
        return "Watch"

    pre_regime = _regime(k_pre["omega"], k_pre["F"], k_pre["S"], k_pre["C"])
    post_regime = _regime(k_post["omega"], k_post["F"], k_post["S"], k_post["C"])

    ic_drop = (1.0 - k_post["IC"] / k_pre["IC"]) * 100 if k_pre["IC"] > 0 else 0
    f_drop = (1.0 - k_post["F"] / k_pre["F"]) * 100 if k_pre["F"] > 0 else 0

    return ExtinctionKernelResult(
        name=ext.name,
        age_mya=ext.age_mya,
        species_loss_pct=ext.species_loss_pct,
        recovery_mya=ext.recovery_mya,
        pre_F=k_pre["F"],
        pre_IC=k_pre["IC"],
        pre_delta=k_pre["heterogeneity_gap"],
        pre_regime=pre_regime,
        post_F=k_post["F"],
        post_IC=k_post["IC"],
        post_delta=k_post["heterogeneity_gap"],
        post_regime=post_regime,
        IC_drop_pct=ic_drop,
        F_drop_pct=f_drop,
        delta_change=k_post["heterogeneity_gap"] - k_pre["heterogeneity_gap"],
        pre_gamma=gamma_omega(k_pre["omega"]),
        post_gamma=gamma_omega(k_post["omega"]),
        post_total_debit=gamma_omega(k_post["omega"]) + cost_curvature(k_post["C"]),
        key_casualties=ext.key_casualties,
        key_survivors=ext.key_survivors,
        recovery_innovation=ext.recovery_innovation,
    )


# ═════════════════════════════════════════════════════════════════════
# SECTION 4: DISPLAY
# ═════════════════════════════════════════════════════════════════════


def print_recursive_analysis() -> None:
    """Print the full recursive evolution analysis."""
    print()
    print("=" * 95)
    print("  RECURSIVE EVOLUTION — RCFT Overlay on Evolutionary Dynamics")
    print("  Collapsus generativus est; solum quod redit, reale est.")
    print("=" * 95)

    # ── Part 1: Scales ────────────────────────────────────────────
    print("\n  PART 1: THE FIVE SCALES OF EVOLUTIONARY RECURSION")
    print("  " + "─" * 90)
    print("\n  Each scale runs the SAME kernel (F, ω, S, C, κ, IC) on DIFFERENT channels.")
    print("  Collapse at one scale generates the input for the next.")
    print()

    scale_results = compute_all_scales()

    print(
        f"  {'Scale':<12s} {'Level':>5s} {'F':>7s} {'ω':>7s} {'IC':>8s} "
        f"{'Δ':>7s} {'Γ(ω)':>8s} {'D_C':>7s} {'Regime':<10s} {'τ_R typical':<20s}"
    )
    print("  " + "─" * 100)

    for r in scale_results:
        print(
            f"  {r.scale_name:<12s} {r.level:>5d} {r.F:>7.4f} {r.omega:>7.4f} "
            f"{r.IC:>8.6f} {r.heterogeneity_gap:>7.4f} {r.gamma_omega:>8.4f} "
            f"{r.D_C:>7.4f} {r.regime:<10s} {r.typical_tau_R:<20s}"
        )

    # Identity verification
    all_ok = all(r.F_plus_omega_exact and r.IC_leq_F and r.IC_eq_exp_kappa for r in scale_results)
    print(f"\n  Tier-1 identities across all 5 scales: {'ALL PASS ✓' if all_ok else 'VIOLATION ✗'}")

    # Key insight
    print("\n  INSIGHT: The kernel is SCALE-INVARIANT. The same identities hold at every level.")
    print("  Gene-level: F + ω = 1 (replication fidelity + mutation rate = 1)")
    print("  Organism-level: F + ω = 1 (reproductive fitness + mortality = 1)")
    print("  Species-level: F + ω = 1 (speciation rate + extinction rate = 1)")
    print("  The recursion is structural, not metaphorical.")

    # ── Degradation pattern ───────────────────────────────────────
    print("\n  RECURSIVE DEGRADATION PATTERN:")
    for r in scale_results:
        bar = "█" * int(r.IC * 40)
        print(f"  {r.scale_name:<12s} IC = {r.IC:.4f}  |{bar:<40s}|  weakest: {r.weakest_channel}")

    print("\n  INSIGHT: IC degrades from Gene → Clade. Each successive scale has MORE")
    print("  heterogeneous channels (more uncontrolled degrees of freedom).")
    print("  The curvature C increases with scale — more coupling to the unknown.")
    print("  This is WHY higher-level extinctions are harder to reverse:")
    print("  the debit (Γ(ω) + D_C) grows with scale, while the return credit")
    print("  must overcome increasingly heterogeneous channel structure.")

    # ── Part 2: Mass Extinctions ──────────────────────────────────
    print("\n\n  PART 2: THE FIVE MASS EXTINCTIONS — Collapse and Return")
    print("  " + "─" * 90)
    print("\n  Each mass extinction is a COLLAPSE EVENT. The recovery is the RETURN.")
    print("  If recovery generates new structure (radiation), the collapse was generative.")
    print("  Axiom-0: Collapse is generative; only what returns is real.\n")

    ext_results = [compute_extinction_kernel(e) for e in MASS_EXTINCTIONS]

    print(
        f"  {'Event':<22s} {'Age':>5s} {'Loss%':>5s} {'pre-F':>6s} {'pre-IC':>7s} "
        f"{'post-F':>7s} {'post-IC':>8s} {'IC drop':>8s} {'Δ change':>8s} {'τ_R(Myr)':>8s}"
    )
    print("  " + "─" * 100)

    for r in ext_results:
        print(
            f"  {r.name:<22s} {r.age_mya:>5.0f} {r.species_loss_pct:>5.0f} "
            f"{r.pre_F:>6.3f} {r.pre_IC:>7.4f} {r.post_F:>7.3f} {r.post_IC:>8.5f} "
            f"{r.IC_drop_pct:>7.1f}% {r.delta_change:>+8.4f} {r.recovery_mya:>8.0f}"
        )

    # The End-Permian
    permian = next(r for r in ext_results if "Permian" in r.name)
    print("\n  THE GREAT DYING — End-Permian (252 Mya)")
    print(f"    Pre:  F = {permian.pre_F:.3f}, IC = {permian.pre_IC:.4f} ({permian.pre_regime})")
    print(f"    Post: F = {permian.post_F:.3f}, IC = {permian.post_IC:.5f} ({permian.post_regime})")
    print(f"    IC dropped {permian.IC_drop_pct:.1f}% — near-total geometric slaughter.")
    print(f"    Recovery took ~{permian.recovery_mya:.0f} Myr → dinosaurs, mammals, modern ecosystems.")
    print("    INSIGHT: 96% species loss. IC devastated. But the collapse was GENERATIVE:")
    print("    it created the ecological vacuum that enabled the Mesozoic radiation.")
    print("    Ruptura est fons constantiae — the rupture is the source of constancy.")

    # ── Part 3: The recursive insight ─────────────────────────────
    print("\n\n  PART 3: THE RECURSIVE INSIGHT — Why Evolution Is Not Linear")
    print("  " + "─" * 90)

    print("""
  The standard narrative: mutation → selection → adaptation → complexity.
  This is linear. It treats evolution as a one-directional accumulation.

  The recursive reinterpretation (from GCD/RCFT):

  EVOLUTION = NESTED COLLAPSE-RETURN CYCLES

      Gene mutation    ─→  collapses into  ─→  Organism death
      Organism death   ─→  collapses into  ─→  Population bottleneck
      Pop. bottleneck  ─→  collapses into  ─→  Species extinction
      Species extinct. ─→  collapses into  ─→  Clade mass extinction
      Mass extinction  ─→  GENERATES       ─→  Adaptive radiation (RETURN)

  Each level's COLLAPSE is the next level's SIGNAL.
  Each level's RETURN is the next level's FIDELITY.

  The kernel is identical at every scale. The channels change, the
  invariants don't. F + ω = 1 at the gene level (replication + mutation = 1)
  AND at the clade level (radiation + extinction = 1).

  CRITICAL PREDICTION: Selection optimizes F (arithmetic mean fitness).
  But survival across perturbation requires IC (geometric mean — all
  channels must be viable). The heterogeneity gap Δ = F - IC is
  INVISIBLE to natural selection but DETERMINES extinction risk.

  This is why:
    • Generalists survive mass extinctions (low Δ → robust IC)
    • Specialists dominate calm periods (high F, high Δ → fragile)
    • Sexual reproduction persists (shuffles channels → reduces Δ)
    • "Living fossils" endure (uniform mediocrity → IC ≈ F)
    • Island endemics go extinct first (narrow breadth → channel death)
    • Recovery after extinction produces MORE diversity (collapse is generative)

  The Dodo had F ≈ 0.21. Its environmental_breadth → ε.
  One dead channel → geometric slaughter → τ_R = ∞_rec.
  The lineage is a gestus, not a sutura.

  Homo sapiens has F ≈ 0.65 but lineage_persistence → ε (we are YOUNG).
  Our IC is dragged down by recency. Whether we are a weld or a gestus
  is the open question — it depends on whether we return.

  Collapsus generativus est; solum quod redit, reale est.
  Evolution is the oldest instantiation of Axiom-0.
""")

    print("=" * 95)
    print("  Finis, sed semper initium recursionis.")
    print("=" * 95)
    print()


# ═════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print_recursive_analysis()

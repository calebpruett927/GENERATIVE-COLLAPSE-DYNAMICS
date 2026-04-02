"""Malbolge Kernel Closure — Dynamic Semiotics Domain.

Tier-2 closure mapping 12 esoteric programming languages through the GCD kernel.
Each language is characterized by 8 channels measuring structural properties that
determine whether a program can return from the maximally hostile dissolution that
esoteric language design imposes.

Malbolge is the paradigmatic case: its self-encrypting memory, ternary arithmetic,
and adversarial instruction encoding create near-total channel destruction. A working
Malbolge program is a demonstrated return (τ_R < ∞_rec) from the Collapse regime —
proof that coherence can be forced to re-emerge from a maximally hostile manifold.
*Ruptura est fons constantiae.*

Channels (8, equal weights w_i = 1/8):
  0  instruction_fidelity     — how faithfully instructions encode intent [0,1]
  1  memory_coherence         — stability of memory across execution cycles [0,1]
  2  cipher_transparency      — readability of any self-modification layer [0,1]
  3  control_flow_regularity  — predictability of execution path [0,1]
  4  compositionality         — meaning derivable from program parts [0,1]
  5  halting_predictability   — likelihood program halts as intended [0,1]
  6  debug_observability      — ability to inspect/trace state [0,1]
  7  generative_expressiveness — range of computable outputs [0,1]

12 entities across 4 categories:
  Adversarial (3):        malbolge, malbolge_unshackled, intercal
  Self-modifying (3):     befunge, brainfuck_selfmod, unlambda
  Minimalist (3):         brainfuck, whitespace, iota
  Structured esoteric (3): shakespeare, chef, piet

6 theorems (T-MB-1 through T-MB-6).
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

MB_CHANNELS = [
    "instruction_fidelity",
    "memory_coherence",
    "cipher_transparency",
    "control_flow_regularity",
    "compositionality",
    "halting_predictability",
    "debug_observability",
    "generative_expressiveness",
]
N_MB_CHANNELS = len(MB_CHANNELS)


@dataclass(frozen=True, slots=True)
class EsotericLanguageEntity:
    """An esoteric programming language with 8 measurable channels."""

    name: str
    category: str
    instruction_fidelity: float
    memory_coherence: float
    cipher_transparency: float
    control_flow_regularity: float
    compositionality: float
    halting_predictability: float
    debug_observability: float
    generative_expressiveness: float

    def trace_vector(self) -> np.ndarray:
        return np.array(
            [
                self.instruction_fidelity,
                self.memory_coherence,
                self.cipher_transparency,
                self.control_flow_regularity,
                self.compositionality,
                self.halting_predictability,
                self.debug_observability,
                self.generative_expressiveness,
            ]
        )


# ── Entity Catalog ───────────────────────────────────────────────────
#
# Channel values are derived from structural properties of each language:
#
# Malbolge: Self-encrypting ternary machine. Every instruction is encrypted
#   after execution (cipher_transparency → 0.02). Memory uses balanced ternary
#   with crazy operation (memory_coherence → 0.03). Instructions encode as
#   ASCII mod 94 with position-dependent meaning (instruction_fidelity → 0.05).
#   Control flow is non-linear via jump tables (control_flow_regularity → 0.04).
#   Programs are effectively non-compositional (compositionality → 0.03).
#   Finding a halting program required AI search (halting_predictability → 0.02).
#   State is unobservable mid-execution (debug_observability → 0.01).
#   Turing-complete but practically inaccessible (generative_expressiveness → 0.15).
#
# Malbolge Unshackled: Removes word-size limit from Malbolge. Same adversarial
#   design but unbounded memory makes generative space slightly larger while
#   making halting even less predictable.
#
# INTERCAL: Deliberately confusing syntax ("PLEASE" modifier, COME FROM,
#   random statement ignoring). Less adversarial than Malbolge but intentionally
#   frustrating. Moderate cipher transparency (no self-modification).

MB_ENTITIES: tuple[EsotericLanguageEntity, ...] = (
    # ── Adversarial ──────────────────────────────────────────────────
    # Designed to be maximally hostile to the programmer.
    EsotericLanguageEntity(
        "malbolge",
        "adversarial",
        0.05,
        0.03,
        0.02,
        0.04,
        0.03,
        0.02,
        0.01,
        0.15,
    ),
    EsotericLanguageEntity(
        "malbolge_unshackled",
        "adversarial",
        0.04,
        0.03,
        0.02,
        0.03,
        0.02,
        0.01,
        0.01,
        0.20,
    ),
    EsotericLanguageEntity(
        "intercal",
        "adversarial",
        0.25,
        0.60,
        0.55,
        0.20,
        0.15,
        0.30,
        0.35,
        0.40,
    ),
    # ── Self-modifying ───────────────────────────────────────────────
    # Languages where the program text mutates during execution.
    EsotericLanguageEntity(
        "befunge",
        "self_modifying",
        0.55,
        0.50,
        0.40,
        0.30,
        0.35,
        0.45,
        0.50,
        0.65,
    ),
    EsotericLanguageEntity(
        "brainfuck_selfmod",
        "self_modifying",
        0.40,
        0.35,
        0.30,
        0.50,
        0.20,
        0.35,
        0.40,
        0.55,
    ),
    EsotericLanguageEntity(
        "unlambda",
        "self_modifying",
        0.30,
        0.70,
        0.60,
        0.15,
        0.65,
        0.25,
        0.20,
        0.50,
    ),
    # ── Minimalist ───────────────────────────────────────────────────
    # Extremely reduced instruction sets, Turing-complete by construction.
    EsotericLanguageEntity(
        "brainfuck",
        "minimalist",
        0.70,
        0.85,
        0.90,
        0.75,
        0.30,
        0.55,
        0.65,
        0.60,
    ),
    EsotericLanguageEntity(
        "whitespace",
        "minimalist",
        0.50,
        0.80,
        0.10,
        0.65,
        0.25,
        0.50,
        0.15,
        0.55,
    ),
    EsotericLanguageEntity(
        "iota",
        "minimalist",
        0.20,
        0.90,
        0.85,
        0.10,
        0.80,
        0.20,
        0.10,
        0.45,
    ),
    # ── Structured esoteric ──────────────────────────────────────────
    # Themed languages with readable (if absurd) surface syntax.
    EsotericLanguageEntity(
        "shakespeare",
        "structured_esoteric",
        0.60,
        0.75,
        0.80,
        0.55,
        0.40,
        0.50,
        0.55,
        0.50,
    ),
    EsotericLanguageEntity(
        "chef",
        "structured_esoteric",
        0.55,
        0.70,
        0.75,
        0.50,
        0.35,
        0.45,
        0.50,
        0.50,
    ),
    EsotericLanguageEntity(
        "piet",
        "structured_esoteric",
        0.45,
        0.65,
        0.70,
        0.35,
        0.25,
        0.40,
        0.60,
        0.55,
    ),
)


@dataclass(frozen=True, slots=True)
class MBKernelResult:
    """Kernel output for an esoteric language entity."""

    name: str
    category: str
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
            "F": self.F,
            "omega": self.omega,
            "S": self.S,
            "C": self.C,
            "kappa": self.kappa,
            "IC": self.IC,
            "regime": self.regime,
        }


def compute_mb_kernel(entity: EsotericLanguageEntity) -> MBKernelResult:
    """Compute GCD kernel for an esoteric language entity."""
    c = entity.trace_vector()
    c = np.clip(c, EPSILON, 1.0 - EPSILON)
    w = np.ones(N_MB_CHANNELS) / N_MB_CHANNELS
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
    return MBKernelResult(
        name=entity.name,
        category=entity.category,
        F=F,
        omega=omega,
        S=S,
        C=C_val,
        kappa=kappa,
        IC=IC,
        regime=regime,
    )


def compute_all_entities() -> list[MBKernelResult]:
    """Compute kernel outputs for all esoteric language entities."""
    return [compute_mb_kernel(e) for e in MB_ENTITIES]


# ── Theorems ─────────────────────────────────────────────────────────


def verify_t_mb_1(results: list[MBKernelResult]) -> dict:
    """T-MB-1: Malbolge and Malbolge Unshackled are in Collapse regime —
    adversarial self-encryption destroys all channels simultaneously,
    placing ω ≥ 0.30 by construction. This is geometric slaughter:
    near-zero cipher_transparency and debug_observability kill IC
    regardless of generative_expressiveness.
    """
    mb = next(r for r in results if r.name == "malbolge")
    mbu = next(r for r in results if r.name == "malbolge_unshackled")
    passed = mb.regime == "Collapse" and mbu.regime == "Collapse"
    return {
        "name": "T-MB-1",
        "passed": bool(passed),
        "malbolge_regime": mb.regime,
        "malbolge_omega": mb.omega,
        "malbolge_unshackled_regime": mbu.regime,
        "malbolge_unshackled_omega": mbu.omega,
    }


def verify_t_mb_2(results: list[MBKernelResult]) -> dict:
    """T-MB-2: Adversarial category has lowest mean F across all categories —
    languages designed to be hostile preserve the least structural fidelity.
    """
    cats = {e.category for e in MB_ENTITIES}
    cat_means: dict[str, float] = {}
    for cat in cats:
        vals = [r.F for r in results if r.category == cat]
        cat_means[cat] = float(np.mean(vals))
    adv_mean = cat_means["adversarial"]
    passed = all(adv_mean <= v + 1e-9 for v in cat_means.values())
    return {
        "name": "T-MB-2",
        "passed": bool(passed),
        "adversarial_mean_F": adv_mean,
        "all_means": cat_means,
    }


def verify_t_mb_3(results: list[MBKernelResult]) -> dict:
    """T-MB-3: Iota has the largest heterogeneity gap (Δ = F − IC) among
    all entities — despite being Turing-complete via a single combinator,
    extreme channel variance (memory_coherence 0.90, compositionality 0.80
    vs. instruction_fidelity 0.20, debug_observability 0.10) creates maximum
    divergence between arithmetic and geometric means.

    This demonstrates that geometric slaughter requires *heterogeneity*, not
    uniform destruction. Malbolge destroys all channels uniformly, so its
    gap is small. Iota's mix of brilliant and blind channels is what kills IC.
    """
    gaps = {r.name: r.F - r.IC for r in results}
    iota_gap = gaps["iota"]
    passed = all(iota_gap >= v - 1e-9 for v in gaps.values())
    return {
        "name": "T-MB-3",
        "passed": bool(passed),
        "iota_gap": float(iota_gap),
        "all_gaps": {k: float(v) for k, v in gaps.items()},
    }


def verify_t_mb_4(results: list[MBKernelResult]) -> dict:
    """T-MB-4: Brainfuck (minimalist) has the highest F among all entities —
    its extreme simplicity (8 instructions, linear tape, transparent semantics)
    means every channel carries moderate-to-high fidelity with no adversarial
    destruction. Minimalism preserves what matters.
    """
    bf = next(r for r in results if r.name == "brainfuck")
    max_F = max(r.F for r in results)
    passed = abs(bf.F - max_F) < 0.02
    return {
        "name": "T-MB-4",
        "passed": bool(passed),
        "brainfuck_F": bf.F,
        "global_max_F": float(max_F),
    }


def verify_t_mb_5(results: list[MBKernelResult]) -> dict:
    """T-MB-5: Structured esoteric category has highest mean debug_observability —
    themed languages (Shakespeare, Chef, Piet) use human-readable surface syntax
    that allows state inspection even when the computational model is unusual.
    """
    cats = {e.category for e in MB_ENTITIES}
    cat_means: dict[str, float] = {}
    for cat in cats:
        vals = [e.debug_observability for e in MB_ENTITIES if e.category == cat]
        cat_means[cat] = float(np.mean(vals))
    se_mean = cat_means["structured_esoteric"]
    passed = all(se_mean >= v - 1e-9 for v in cat_means.values())
    return {
        "name": "T-MB-5",
        "passed": bool(passed),
        "structured_esoteric_debug": se_mean,
        "all_means": cat_means,
    }


def verify_t_mb_6(results: list[MBKernelResult]) -> dict:
    """T-MB-6: Malbolge has ω > 0.95 — the deepest Collapse in the catalog.
    Over 95% of structural fidelity is lost to drift. This is not geometric
    slaughter (which requires heterogeneity) but *uniform dissolution*:
    every channel is simultaneously near-zero. The distinction matters —
    uniform destruction preserves IC/F ratio while maximizing absolute ω.

    Malbolge demonstrates that Collapse regime has internal structure:
    heterogeneous collapse (large Δ, as in iota) differs from homogeneous
    collapse (small Δ, extreme ω, as in Malbolge).
    """
    mb = next(r for r in results if r.name == "malbolge")
    passed = mb.omega > 0.95
    return {
        "name": "T-MB-6",
        "passed": bool(passed),
        "malbolge_omega": mb.omega,
        "malbolge_F": mb.F,
        "malbolge_IC": mb.IC,
        "malbolge_gap": mb.F - mb.IC,
    }


def verify_all_theorems() -> list[dict]:
    """Run all T-MB theorems."""
    results = compute_all_entities()
    return [
        verify_t_mb_1(results),
        verify_t_mb_2(results),
        verify_t_mb_3(results),
        verify_t_mb_4(results),
        verify_t_mb_5(results),
        verify_t_mb_6(results),
    ]


def main() -> None:
    """Entry point."""
    results = compute_all_entities()
    print("=" * 78)
    print("MALBOLGE KERNEL — ESOTERIC LANGUAGES AS COLLAPSE LABORATORIES")
    print("=" * 78)
    print(f"{'Entity':<24} {'Cat':<20} {'F':>6} {'ω':>6} {'IC':>6} {'Δ':>6} {'IC/F':>6} {'Regime'}")
    print("-" * 78)
    for r in results:
        gap = r.F - r.IC
        ratio = r.IC / r.F if r.F > EPSILON else 0.0
        print(f"{r.name:<24} {r.category:<20} {r.F:6.3f} {r.omega:6.3f} {r.IC:6.3f} {gap:6.3f} {ratio:6.3f} {r.regime}")
    print("\n── Theorems ──")
    for t in verify_all_theorems():
        status = "PROVEN" if t["passed"] else "FAILED"
        print(f"  {t['name']}: {status}")


if __name__ == "__main__":
    main()

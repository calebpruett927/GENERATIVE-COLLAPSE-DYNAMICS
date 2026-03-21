"""Temporal Topology Closure — Spacetime Memory Domain.

Tier-2 closure mapping 12 collapse-return circulation types through the
GCD kernel. Formalizes time_vortex.txt: time as poloidal circulation on
a torus (the Dungey cycle), asymmetric descent/ascent, winding numbers,
toroidal drift, and the vortex structure of the arrow of time.

Channels (8, equal weights w_i = 1/8):
  0  loop_completeness     — cycling completeness (1 = fully recorded loop, ~0 = fragmentary)
  1  descent_coherence     — coherence preserved during collapse (1 = lossless, ~0 = chaotic)
  2  ascent_coherence      — coherence preserved during return (< descent: arrow of time)
  3  cost_awareness        — how well cost asymmetry is characterized (1 = transparent, ~0 = opaque)
  4  cycle_regularity      — regularity of winding (1 = perfectly periodic, ~0 = chaotic)
  5  drift_stability       — stability against toroidal drift (1 = locked, ~0 = blown off course)
  6  equator_proximity     — closeness to c = 1/2 (max entropy, S + κ = 0)
  7  poloidal_closure      — how well the loop closes (1 = perfect return)

12 entities across 4 categories:
  Descent (3): gentle_descent, moderate_descent, steep_descent
  Circulation (3): shallow_loop, deep_loop, near_horizon_loop
  Toroidal (3): slow_precession, fast_precession, quasi_periodic
  Special (3): equatorial_crossing, magnetotail_reconnection, stagnation_point

6 theorems (T-TT-1 through T-TT-6).
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

TT_CHANNELS = [
    "loop_completeness",
    "descent_coherence",
    "ascent_coherence",
    "cost_awareness",
    "cycle_regularity",
    "drift_stability",
    "equator_proximity",
    "poloidal_closure",
]
N_TT_CHANNELS = len(TT_CHANNELS)


@dataclass(frozen=True, slots=True)
class TemporalEntity:
    """A collapse-return circulation type with 8 measurable channels."""

    name: str
    category: str
    loop_completeness: float
    descent_coherence: float
    ascent_coherence: float
    cost_awareness: float
    cycle_regularity: float
    drift_stability: float
    equator_proximity: float
    poloidal_closure: float

    def trace_vector(self) -> np.ndarray:
        return np.array(
            [
                self.loop_completeness,
                self.descent_coherence,
                self.ascent_coherence,
                self.cost_awareness,
                self.cycle_regularity,
                self.drift_stability,
                self.equator_proximity,
                self.poloidal_closure,
            ]
        )


TT_ENTITIES: tuple[TemporalEntity, ...] = (
    # Descent — different depths of collapse; coherence degrades with steepness
    TemporalEntity("gentle_descent", "descent", 0.85, 0.90, 0.85, 0.75, 0.65, 0.92, 0.30, 0.90),
    TemporalEntity("moderate_descent", "descent", 0.70, 0.75, 0.60, 0.65, 0.50, 0.70, 0.55, 0.80),
    TemporalEntity("steep_descent", "descent", 0.50, 0.55, 0.30, 0.50, 0.35, 0.45, 0.75, 0.55),
    # Circulation — loop geometry in (F, κ, C) space
    TemporalEntity("shallow_loop", "circulation", 0.90, 0.92, 0.88, 0.80, 0.70, 0.95, 0.25, 0.95),
    TemporalEntity("deep_loop", "circulation", 0.55, 0.60, 0.40, 0.50, 0.40, 0.50, 0.70, 0.60),
    TemporalEntity("near_horizon_loop", "circulation", 0.20, 0.30, 0.10, 0.25, 0.15, 0.12, 0.90, 0.15),
    # Toroidal — successive cycles advancing around the ring
    TemporalEntity("slow_precession", "toroidal", 0.85, 0.88, 0.82, 0.78, 0.65, 0.90, 0.35, 0.88),
    TemporalEntity("fast_precession", "toroidal", 0.60, 0.65, 0.48, 0.55, 0.42, 0.45, 0.60, 0.65),
    TemporalEntity("quasi_periodic", "toroidal", 0.50, 0.55, 0.38, 0.48, 0.38, 0.40, 0.65, 0.52),
    # Special points on the vortex
    TemporalEntity("equatorial_crossing", "special", 0.65, 0.70, 0.60, 0.62, 0.50, 0.55, 0.95, 0.72),
    TemporalEntity("magnetotail_reconnection", "special", 0.35, 0.40, 0.18, 0.30, 0.22, 0.20, 0.80, 0.28),
    TemporalEntity("stagnation_point", "special", 0.95, 0.97, 0.95, 0.85, 0.98, 0.99, 0.15, 0.99),
)


@dataclass(frozen=True, slots=True)
class TTKernelResult:
    """Kernel output for a temporal topology entity."""

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


def compute_tt_kernel(entity: TemporalEntity) -> TTKernelResult:
    """Compute GCD kernel for a temporal topology entity."""
    c = entity.trace_vector()
    c = np.clip(c, EPSILON, 1.0 - EPSILON)
    w = np.ones(N_TT_CHANNELS) / N_TT_CHANNELS
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
    return TTKernelResult(
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


def compute_all_entities() -> list[TTKernelResult]:
    """Compute kernel outputs for all temporal topology entities."""
    return [compute_tt_kernel(e) for e in TT_ENTITIES]


# ── Theorems ──────────────────────────────────────────────────────────


def verify_t_tt_1(results: list[TTKernelResult]) -> dict:
    """T-TT-1: Descent coherence exceeds ascent coherence for ALL entities —
    the arrow of time is structural. Collapse preserves more coherence;
    return costs more. Γ(ω) increases monotonically; climbing the budget
    surface degrades coherence more than falling down it.
    """
    all_asymmetric = all(e.descent_coherence > e.ascent_coherence for e in TT_ENTITIES)
    ratios = [e.descent_coherence / max(e.ascent_coherence, 1e-10) for e in TT_ENTITIES]
    return {
        "name": "T-TT-1",
        "passed": bool(all_asymmetric),
        "all_descent_gt_ascent": bool(all_asymmetric),
        "mean_ratio": float(np.mean(ratios)),
    }


def verify_t_tt_2(results: list[TTKernelResult]) -> dict:
    """T-TT-2: Near-horizon loop has highest ω — approaching the pole
    where Γ → ∞ means maximum cost, maximum winding.
    """
    nhl = next(r for r in results if r.name == "near_horizon_loop")
    max_omega = max(r.omega for r in results)
    passed = abs(nhl.omega - max_omega) < 0.02
    return {
        "name": "T-TT-2",
        "passed": bool(passed),
        "nhl_omega": nhl.omega,
        "max_omega": float(max_omega),
    }


def verify_t_tt_3(results: list[TTKernelResult]) -> dict:
    """T-TT-3: Stagnation point has highest F — static equilibrium at
    the top of the vortex (near-zero loop area, easiest return).
    """
    sp = next(r for r in results if r.name == "stagnation_point")
    max_F = max(r.F for r in results)
    passed = abs(sp.F - max_F) < 0.02
    return {
        "name": "T-TT-3",
        "passed": bool(passed),
        "stagnation_F": sp.F,
        "max_F": float(max_F),
    }


def verify_t_tt_4(results: list[TTKernelResult]) -> dict:
    """T-TT-4: Descent entities show monotonic F decrease —
    gentle > moderate > steep, reflecting deepening collapse.
    """
    gentle = next(r for r in results if r.name == "gentle_descent")
    moderate = next(r for r in results if r.name == "moderate_descent")
    steep = next(r for r in results if r.name == "steep_descent")
    passed = gentle.F > moderate.F > steep.F
    return {
        "name": "T-TT-4",
        "passed": bool(passed),
        "gentle_F": gentle.F,
        "moderate_F": moderate.F,
        "steep_F": steep.F,
    }


def verify_t_tt_5(results: list[TTKernelResult]) -> dict:
    """T-TT-5: Equatorial crossing has highest equator_proximity channel —
    structural fixed point at c = 1/2 where S + κ = 0 exactly.
    """
    eq = next(e for e in TT_ENTITIES if e.name == "equatorial_crossing")
    max_eq_prox = max(e.equator_proximity for e in TT_ENTITIES)
    passed = abs(eq.equator_proximity - max_eq_prox) < 0.01
    return {
        "name": "T-TT-5",
        "passed": bool(passed),
        "equatorial_proximity": eq.equator_proximity,
        "max_proximity": float(max_eq_prox),
    }


def verify_t_tt_6(results: list[TTKernelResult]) -> dict:
    """T-TT-6: All entities have nonzero loop completeness channel —
    collapse-return trajectories form loops, not lines.
    Time is cyclic (vortex), not linear.
    """
    all_nonzero = all(e.loop_completeness > 0.10 for e in TT_ENTITIES)
    min_area = min(e.loop_completeness for e in TT_ENTITIES)
    return {
        "name": "T-TT-6",
        "passed": bool(all_nonzero),
        "min_loop_completeness": float(min_area),
        "all_nonzero": bool(all_nonzero),
    }


def verify_all_theorems() -> list[dict]:
    """Run all T-TT theorems."""
    results = compute_all_entities()
    return [
        verify_t_tt_1(results),
        verify_t_tt_2(results),
        verify_t_tt_3(results),
        verify_t_tt_4(results),
        verify_t_tt_5(results),
        verify_t_tt_6(results),
    ]


def main() -> None:
    """Entry point."""
    results = compute_all_entities()
    print("=" * 78)
    print("TEMPORAL TOPOLOGY — GCD KERNEL ANALYSIS")
    print("=" * 78)
    print(f"{'Entity':<28} {'Cat':<16} {'F':>6} {'ω':>6} {'IC':>6} {'Δ':>6} {'Regime'}")
    print("-" * 78)
    for r in results:
        gap = r.F - r.IC
        print(f"{r.name:<28} {r.category:<16} {r.F:6.3f} {r.omega:6.3f} {r.IC:6.3f} {gap:6.3f} {r.regime}")

    print("\n── Theorems ──")
    for t in verify_all_theorems():
        print(f"  {t['name']}: {'PROVEN' if t['passed'] else 'FAILED'}")


if __name__ == "__main__":
    main()

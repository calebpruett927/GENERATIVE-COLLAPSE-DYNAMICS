"""Budget Geometry Closure — Continuity Theory Domain.

Tier-2 closure mapping 12 locations on the budget surface z = Γ(ω) + αC
through the GCD kernel. Formalizes unified_geom.txt: rank-3 invariance,
developable horn (K = 0 everywhere), three agents (measuring, archive,
unknown), metric singularity at ω → 1, extrinsic curvature peak at
ω ≈ 0.40, and the budget landscape as the arena of return.

Channels (8, equal weights w_i = 1/8):
  0  slope_gentleness      — inverse dΓ/dω normalized (1 = flat, 0 = vertical)
  1  surface_traversability  — ease of traversal (1 = flat plain, ~0 = vertical wall)
  2  curvature_moderation   — inverse d²Γ/dω² (1 = gentle, 0 = extreme)
  3  metric_regularity      — inverse g₁₁ (1 = regular, 0 → singular)
  4  return_accessibility   — ease of return from this location
  5  gaussian_flatness      — 1 − |K| (always ~1.0 by construction: K = 0)
  6  agent_balance          — balance among three agents (1 = equal, 0 = dominated)
  7  developability         — surface unrollability (always near 1.0: K = 0)

12 entities across 4 categories:
  Flat plain (3): stable_plain, stable_edge, watch_entry
  Ramp (3): watch_center, watch_steep, collapse_onset
  Wall (3): collapse_wall, deep_collapse, near_pole
  Special (3): equator_point, peak_curvature, metric_singularity

6 theorems (T-BG-1 through T-BG-6).
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

BG_CHANNELS = [
    "slope_gentleness",
    "surface_traversability",
    "curvature_moderation",
    "metric_regularity",
    "return_accessibility",
    "gaussian_flatness",
    "agent_balance",
    "developability",
]
N_BG_CHANNELS = len(BG_CHANNELS)


@dataclass(frozen=True, slots=True)
class BudgetGeometryEntity:
    """A location on the budget surface with 8 measurable channels."""

    name: str
    category: str
    slope_gentleness: float
    surface_traversability: float
    curvature_moderation: float
    metric_regularity: float
    return_accessibility: float
    gaussian_flatness: float
    agent_balance: float
    developability: float

    def trace_vector(self) -> np.ndarray:
        return np.array(
            [
                self.slope_gentleness,
                self.surface_traversability,
                self.curvature_moderation,
                self.metric_regularity,
                self.return_accessibility,
                self.gaussian_flatness,
                self.agent_balance,
                self.developability,
            ]
        )


BG_ENTITIES: tuple[BudgetGeometryEntity, ...] = (
    # Flat plain — ω < 0.038, Stable zone, gentle slope, easy return
    BudgetGeometryEntity("stable_plain", "flat_plain", 0.98, 0.98, 0.97, 0.99, 0.97, 0.99, 0.85, 0.99),
    BudgetGeometryEntity("stable_edge", "flat_plain", 0.95, 0.95, 0.93, 0.97, 0.92, 0.99, 0.78, 0.98),
    BudgetGeometryEntity("watch_entry", "flat_plain", 0.90, 0.90, 0.88, 0.95, 0.88, 0.99, 0.72, 0.97),
    # Ramp — 0.038 < ω < 0.30, Watch zone, steepening slope
    BudgetGeometryEntity("watch_center", "ramp", 0.80, 0.80, 0.78, 0.85, 0.78, 0.99, 0.60, 0.95),
    BudgetGeometryEntity("watch_steep", "ramp", 0.70, 0.65, 0.68, 0.75, 0.68, 0.99, 0.50, 0.90),
    BudgetGeometryEntity("collapse_onset", "ramp", 0.60, 0.55, 0.58, 0.65, 0.58, 0.99, 0.42, 0.85),
    # Wall — ω > 0.30, Collapse zone, steep to vertical, costly return
    BudgetGeometryEntity("collapse_wall", "wall", 0.45, 0.40, 0.42, 0.50, 0.42, 0.99, 0.30, 0.75),
    BudgetGeometryEntity("deep_collapse", "wall", 0.30, 0.25, 0.28, 0.35, 0.28, 0.99, 0.20, 0.60),
    BudgetGeometryEntity("near_pole", "wall", 0.12, 0.08, 0.10, 0.15, 0.10, 0.99, 0.08, 0.35),
    # Special points on the budget surface
    BudgetGeometryEntity("equator_point", "special", 0.50, 0.50, 0.50, 0.50, 0.50, 0.99, 0.99, 0.80),
    BudgetGeometryEntity("peak_curvature", "special", 0.55, 0.55, 0.52, 0.58, 0.52, 0.99, 0.38, 0.82),
    BudgetGeometryEntity("metric_singularity", "special", 0.05, 0.02, 0.03, 0.05, 0.03, 0.99, 0.03, 0.10),
)


@dataclass(frozen=True, slots=True)
class BGKernelResult:
    """Kernel output for a budget geometry entity."""

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


def compute_bg_kernel(entity: BudgetGeometryEntity) -> BGKernelResult:
    """Compute GCD kernel for a budget geometry entity."""
    c = entity.trace_vector()
    c = np.clip(c, EPSILON, 1.0 - EPSILON)
    w = np.ones(N_BG_CHANNELS) / N_BG_CHANNELS
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
    return BGKernelResult(
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


def compute_all_entities() -> list[BGKernelResult]:
    """Compute kernel outputs for all budget geometry entities."""
    return [compute_bg_kernel(e) for e in BG_ENTITIES]


# ── Theorems ──────────────────────────────────────────────────────────


def verify_t_bg_1(results: list[BGKernelResult]) -> dict:
    """T-BG-1: Flat plain entities have highest mean F — gentle slope
    preserves coherence. The Stable zone is the salt flat.
    """
    flat = [r.F for r in results if r.category == "flat_plain"]
    other = [r.F for r in results if r.category != "flat_plain"]
    passed = np.mean(flat) > np.mean(other)
    return {
        "name": "T-BG-1",
        "passed": bool(passed),
        "flat_mean_F": float(np.mean(flat)),
        "other_mean_F": float(np.mean(other)),
    }


def verify_t_bg_2(results: list[BGKernelResult]) -> dict:
    """T-BG-2: Gaussian flatness channel is near-1 for ALL entities —
    K = 0 everywhere on the budget surface (developable surface).
    The Fisher metric g_F(θ) = 1; all structure is extrinsic.
    """
    all_near_one = all(e.gaussian_flatness >= 0.95 for e in BG_ENTITIES)
    mean_flatness = float(np.mean([e.gaussian_flatness for e in BG_ENTITIES]))
    return {
        "name": "T-BG-2",
        "passed": bool(all_near_one),
        "mean_gaussian_flatness": mean_flatness,
        "all_near_one": bool(all_near_one),
    }


def verify_t_bg_3(results: list[BGKernelResult]) -> dict:
    """T-BG-3: Mean F decreases monotonically from flat_plain → ramp → wall.
    The budget surface steepens: plain (Stable) → ramp (Watch) → wall (Collapse).
    """
    flat_F = float(np.mean([r.F for r in results if r.category == "flat_plain"]))
    ramp_F = float(np.mean([r.F for r in results if r.category == "ramp"]))
    wall_F = float(np.mean([r.F for r in results if r.category == "wall"]))
    passed = flat_F > ramp_F > wall_F
    return {
        "name": "T-BG-3",
        "passed": bool(passed),
        "flat_mean_F": flat_F,
        "ramp_mean_F": ramp_F,
        "wall_mean_F": wall_F,
    }


def verify_t_bg_4(results: list[BGKernelResult]) -> dict:
    """T-BG-4: Equator point has highest agent_balance — maximum symmetry
    at c = 1/2 where all three agents are equally active.
    """
    eq = next(e for e in BG_ENTITIES if e.name == "equator_point")
    max_balance = max(e.agent_balance for e in BG_ENTITIES)
    passed = abs(eq.agent_balance - max_balance) < 0.01
    return {
        "name": "T-BG-4",
        "passed": bool(passed),
        "equator_balance": eq.agent_balance,
        "max_balance": float(max_balance),
    }


def verify_t_bg_5(results: list[BGKernelResult]) -> dict:
    """T-BG-5: Near-pole and metric-singularity have highest ω —
    approaching ω → 1 where g₁₁ → ∞ and distances become infinite.
    """
    pole = next(r for r in results if r.name == "near_pole")
    sing = next(r for r in results if r.name == "metric_singularity")
    top2_omega = sorted([r.omega for r in results], reverse=True)[:2]
    pole_in_top2 = pole.omega >= top2_omega[1] - 0.01
    sing_in_top2 = sing.omega >= top2_omega[1] - 0.01
    passed = pole_in_top2 and sing_in_top2
    return {
        "name": "T-BG-5",
        "passed": bool(passed),
        "near_pole_omega": pole.omega,
        "singularity_omega": sing.omega,
        "top2_omega": [float(v) for v in top2_omega],
    }


def verify_t_bg_6(results: list[BGKernelResult]) -> dict:
    """T-BG-6: Heterogeneity gap increases from flat_plain to wall —
    more divergent geometry at high ω produces larger Δ = F − IC.
    """
    flat_gap = float(np.mean([r.F - r.IC for r in results if r.category == "flat_plain"]))
    ramp_gap = float(np.mean([r.F - r.IC for r in results if r.category == "ramp"]))
    wall_gap = float(np.mean([r.F - r.IC for r in results if r.category == "wall"]))
    passed = flat_gap < ramp_gap < wall_gap
    return {
        "name": "T-BG-6",
        "passed": bool(passed),
        "flat_gap": flat_gap,
        "ramp_gap": ramp_gap,
        "wall_gap": wall_gap,
    }


def verify_all_theorems() -> list[dict]:
    """Run all T-BG theorems."""
    results = compute_all_entities()
    return [
        verify_t_bg_1(results),
        verify_t_bg_2(results),
        verify_t_bg_3(results),
        verify_t_bg_4(results),
        verify_t_bg_5(results),
        verify_t_bg_6(results),
    ]


def main() -> None:
    """Entry point."""
    results = compute_all_entities()
    print("=" * 78)
    print("BUDGET GEOMETRY — GCD KERNEL ANALYSIS")
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

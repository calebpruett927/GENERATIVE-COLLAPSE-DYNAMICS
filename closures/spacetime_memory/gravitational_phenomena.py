"""Gravitational Phenomena Closure — Spacetime Memory Domain.

Tier-2 closure mapping 12 gravitational systems through the GCD kernel.
Formalizes gravity.txt and lensing.txt: gravity as budget gradient,
always attractive (κ ≤ 0), cubic suppression, time dilation,
equivalence principle, tidal forces, infinite range, lensing.

Channels (8, equal weights w_i = 1/8):
  0  structural_persistence — how well structure survives at this location
  1  temporal_coherence      — clock rate fidelity (1 = normal time, ~0 = frozen)
  2  spatial_regularity      — geometry regularity (1 = flat, ~0 = extreme curvature)
  3  tidal_tolerance         — survival of extended objects (1 = safe, ~0 = torn apart)
  4  information_escape      — can signals escape the well (1 = free, ~0 = trapped)
  5  measurement_access      — observational accessibility (1 = easy, ~0 = redshifted away)
  6  geometric_uniformity    — well profile uniformity (1 = ring, ~0 = distorted arc)
  7  range_coverage          — fraction of gravitational range contributing

12 entities across 4 categories:
  Weak field (3): satellite_orbit, solar_system, stellar_interior
  Strong field (3): neutron_star, black_hole_photosphere, event_horizon
  Lensing (3): point_mass_lens, galaxy_cluster_lens, microlens_event
  Cosmological (3): cosmic_void, cosmic_filament, dark_energy_horizon

6 theorems (T-GP-1 through T-GP-6).
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

GP_CHANNELS = [
    "structural_persistence",
    "temporal_coherence",
    "spatial_regularity",
    "tidal_tolerance",
    "information_escape",
    "measurement_access",
    "geometric_uniformity",
    "range_coverage",
]
N_GP_CHANNELS = len(GP_CHANNELS)


@dataclass(frozen=True, slots=True)
class GravitationalEntity:
    """A gravitational system with 8 measurable channels."""

    name: str
    category: str
    structural_persistence: float
    temporal_coherence: float
    spatial_regularity: float
    tidal_tolerance: float
    information_escape: float
    measurement_access: float
    geometric_uniformity: float
    range_coverage: float

    def trace_vector(self) -> np.ndarray:
        return np.array(
            [
                self.structural_persistence,
                self.temporal_coherence,
                self.spatial_regularity,
                self.tidal_tolerance,
                self.information_escape,
                self.measurement_access,
                self.geometric_uniformity,
                self.range_coverage,
            ]
        )


GP_ENTITIES: tuple[GravitationalEntity, ...] = (
    # Weak field — all channels high, structure fully preserved
    GravitationalEntity("satellite_orbit", "weak_field", 0.98, 0.99, 0.97, 0.99, 0.99, 0.98, 0.95, 0.85),
    GravitationalEntity("solar_system", "weak_field", 0.95, 0.97, 0.93, 0.97, 0.97, 0.95, 0.92, 0.82),
    GravitationalEntity("stellar_interior", "weak_field", 0.88, 0.92, 0.85, 0.90, 0.90, 0.88, 0.88, 0.78),
    # Strong field — temporal coherence, tidal tolerance destroyed near pole
    GravitationalEntity("neutron_star", "strong_field", 0.55, 0.40, 0.45, 0.35, 0.60, 0.50, 0.70, 0.75),
    GravitationalEntity("black_hole_photosphere", "strong_field", 0.30, 0.15, 0.25, 0.12, 0.25, 0.20, 0.55, 0.70),
    GravitationalEntity("event_horizon", "strong_field", 0.10, 0.02, 0.08, 0.02, 0.05, 0.05, 0.45, 0.65),
    # Lensing — profile uniformity determines ring vs arc
    GravitationalEntity("point_mass_lens", "lensing", 0.75, 0.80, 0.70, 0.78, 0.82, 0.78, 0.95, 0.85),
    GravitationalEntity("galaxy_cluster_lens", "lensing", 0.65, 0.72, 0.60, 0.68, 0.75, 0.70, 0.35, 0.88),
    GravitationalEntity("microlens_event", "lensing", 0.92, 0.95, 0.90, 0.95, 0.95, 0.93, 0.85, 0.60),
    # Cosmological — void is nearly flat, horizon involves information trapping
    GravitationalEntity("cosmic_void", "cosmological", 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.98, 0.80),
    GravitationalEntity("cosmic_filament", "cosmological", 0.85, 0.88, 0.82, 0.88, 0.90, 0.85, 0.55, 0.90),
    GravitationalEntity("dark_energy_horizon", "cosmological", 0.50, 0.55, 0.48, 0.52, 0.45, 0.42, 0.42, 0.95),
)


@dataclass(frozen=True, slots=True)
class GPKernelResult:
    """Kernel output for a gravitational phenomenon entity."""

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


def compute_gp_kernel(entity: GravitationalEntity) -> GPKernelResult:
    """Compute GCD kernel for a gravitational phenomenon entity."""
    c = entity.trace_vector()
    c = np.clip(c, EPSILON, 1.0 - EPSILON)
    w = np.ones(N_GP_CHANNELS) / N_GP_CHANNELS
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
    return GPKernelResult(
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


def compute_all_entities() -> list[GPKernelResult]:
    """Compute kernel outputs for all gravitational phenomenon entities."""
    return [compute_gp_kernel(e) for e in GP_ENTITIES]


# ── Theorems ──────────────────────────────────────────────────────────


def verify_t_gp_1(results: list[GPKernelResult]) -> dict:
    """T-GP-1: Strong field entities have highest mean ω — extreme gravity
    corresponds to high drift (approaching the budget pole at ω = 1).
    """
    strong = [r.omega for r in results if r.category == "strong_field"]
    other = [r.omega for r in results if r.category != "strong_field"]
    passed = np.mean(strong) > np.mean(other)
    return {
        "name": "T-GP-1",
        "passed": bool(passed),
        "strong_mean_omega": float(np.mean(strong)),
        "other_mean_omega": float(np.mean(other)),
    }


def verify_t_gp_2(results: list[GPKernelResult]) -> dict:
    """T-GP-2: κ < 0 for ALL entities — gravity is always attractive.

    Structural: κ = Σ wᵢ ln(cᵢ), and ln(cᵢ) ≤ 0 for cᵢ ∈ [0,1].
    No configuration of channels produces positive κ.
    """
    all_negative = all(r.kappa < 0 for r in results)
    max_kappa = max(r.kappa for r in results)
    return {
        "name": "T-GP-2",
        "passed": bool(all_negative),
        "max_kappa": float(max_kappa),
        "all_negative": bool(all_negative),
    }


def verify_t_gp_3(results: list[GPKernelResult]) -> dict:
    """T-GP-3: Weak field entities have highest mean F — fidelity is
    preserved in gentle gravitational fields (low ω, low budget gradient).
    """
    weak = [r.F for r in results if r.category == "weak_field"]
    other = [r.F for r in results if r.category != "weak_field"]
    passed = np.mean(weak) > np.mean(other)
    return {
        "name": "T-GP-3",
        "passed": bool(passed),
        "weak_mean_F": float(np.mean(weak)),
        "other_mean_F": float(np.mean(other)),
    }


def verify_t_gp_4(results: list[GPKernelResult]) -> dict:
    """T-GP-4: Event horizon has lowest IC among all entities —
    extreme heterogeneity near ω → 1 (temporal_dilation channel near ε).
    """
    eh = next(r for r in results if r.name == "event_horizon")
    min_IC = min(r.IC for r in results)
    passed = abs(eh.IC - min_IC) < 0.01
    return {
        "name": "T-GP-4",
        "passed": bool(passed),
        "event_horizon_IC": eh.IC,
        "min_IC": float(min_IC),
    }


def verify_t_gp_5(results: list[GPKernelResult]) -> dict:
    """T-GP-5: Lensing profile uniformity predicts heterogeneity gap —
    galaxy_cluster_lens (low uniformity) has larger Δ than
    point_mass_lens (high uniformity). Δ determines ring vs arc.
    """
    cluster = next(r for r in results if r.name == "galaxy_cluster_lens")
    point = next(r for r in results if r.name == "point_mass_lens")
    gap_cluster = cluster.F - cluster.IC
    gap_point = point.F - point.IC
    passed = gap_cluster > gap_point
    return {
        "name": "T-GP-5",
        "passed": bool(passed),
        "cluster_gap": float(gap_cluster),
        "point_gap": float(gap_point),
    }


def verify_t_gp_6(results: list[GPKernelResult]) -> dict:
    """T-GP-6: Range coverage is high across all entities —
    gravity has infinite range. Mean range_coverage ≥ 0.70.
    Γ(ω) is nonzero for all ω > 0; no cutoff exists.
    """
    mean_range = float(np.mean([e.range_coverage for e in GP_ENTITIES]))
    passed = mean_range >= 0.70
    return {
        "name": "T-GP-6",
        "passed": bool(passed),
        "mean_range_coverage": mean_range,
    }


def verify_all_theorems() -> list[dict]:
    """Run all T-GP theorems."""
    results = compute_all_entities()
    return [
        verify_t_gp_1(results),
        verify_t_gp_2(results),
        verify_t_gp_3(results),
        verify_t_gp_4(results),
        verify_t_gp_5(results),
        verify_t_gp_6(results),
    ]


def main() -> None:
    """Entry point."""
    results = compute_all_entities()
    print("=" * 78)
    print("GRAVITATIONAL PHENOMENA — GCD KERNEL ANALYSIS")
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

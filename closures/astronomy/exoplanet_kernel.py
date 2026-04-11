"""Exoplanet Characterization Closure — Astronomy Domain.

Tier-2 closure mapping 12 exoplanetary systems through the GCD kernel.
Each exoplanet is characterized by 8 detection and habitability channels.

Channels (8, equal weights w_i = 1/8):
  0  equilibrium_temp_norm  — T_eq / 3000 K (1 = hot)
  1  mass_log_norm          — log(M/M_Earth) / log(4000) (1 = massive)
  2  radius_norm            — R / 2.5 R_Jup (1 = large)
  3  orbital_stability      — 1 − e (eccentricity complement; 1 = circular)
  4  atmosphere_constraint  — atmospheric characterization confidence (1 = well-constrained)
  5  stellar_activity_low   — 1 − stellar activity index (1 = quiet star)
  6  hz_proximity           — 1 − |d − d_HZ| / d_HZ (1 = centered in HZ)
  7  detection_confidence   — detection SNR normalized (1 = unambiguous)

12 entities across 4 categories:
  Hot Gas Giants (3):   51_Peg_b, WASP_121b, HD_209458b
  Rocky HZ (3):         TRAPPIST_1e, Proxima_Cen_b, Kepler_442b
  Sub-Neptunes (3):     GJ_1214b, TOI_700d, K2_18b
  Extreme (3):          PSR_B1257_12b, OGLE_2016_rogue, HD_80606b

6 theorems (T-EP-1 through T-EP-6).
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

EP_CHANNELS = [
    "equilibrium_temp_norm",
    "mass_log_norm",
    "radius_norm",
    "orbital_stability",
    "atmosphere_constraint",
    "stellar_activity_low",
    "hz_proximity",
    "detection_confidence",
]
N_EP_CHANNELS = len(EP_CHANNELS)


@dataclass(frozen=True, slots=True)
class ExoplanetEntity:
    """An exoplanet with 8 characterization channels."""

    name: str
    category: str
    equilibrium_temp_norm: float
    mass_log_norm: float
    radius_norm: float
    orbital_stability: float
    atmosphere_constraint: float
    stellar_activity_low: float
    hz_proximity: float
    detection_confidence: float

    def trace_vector(self) -> np.ndarray:
        return np.array(
            [
                self.equilibrium_temp_norm,
                self.mass_log_norm,
                self.radius_norm,
                self.orbital_stability,
                self.atmosphere_constraint,
                self.stellar_activity_low,
                self.hz_proximity,
                self.detection_confidence,
            ]
        )


EP_ENTITIES: tuple[ExoplanetEntity, ...] = (
    # Hot Gas Giants — tidally locked, well-characterized atmospheres, high F
    ExoplanetEntity("51_Peg_b", "hot_giant", 0.70, 0.80, 0.75, 0.98, 0.70, 0.85, 0.30, 0.95),
    ExoplanetEntity("WASP_121b", "hot_giant", 0.90, 0.85, 0.80, 0.97, 0.85, 0.70, 0.05, 0.98),
    ExoplanetEntity("HD_209458b", "hot_giant", 0.75, 0.82, 0.78, 0.96, 0.80, 0.78, 0.10, 0.97),
    # Rocky habitable zone — moderate temp, atmosphere poorly constrained
    ExoplanetEntity("TRAPPIST_1e", "rocky_hz", 0.08, 0.10, 0.04, 0.99, 0.15, 0.40, 0.92, 0.85),
    ExoplanetEntity("Proxima_Cen_b", "rocky_hz", 0.07, 0.12, 0.05, 0.85, 0.08, 0.25, 0.88, 0.75),
    ExoplanetEntity("Kepler_442b", "rocky_hz", 0.06, 0.15, 0.06, 0.93, 0.05, 0.70, 0.95, 0.80),
    # Sub-Neptunes — intermediate, possible water worlds
    ExoplanetEntity("GJ_1214b", "sub_neptune", 0.18, 0.28, 0.10, 0.97, 0.40, 0.60, 0.30, 0.90),
    ExoplanetEntity("TOI_700d", "sub_neptune", 0.08, 0.14, 0.05, 0.98, 0.20, 0.72, 0.85, 0.88),
    ExoplanetEntity("K2_18b", "sub_neptune", 0.10, 0.32, 0.09, 0.95, 0.55, 0.55, 0.70, 0.92),
    # Extreme — rogue planets, pulsar planets, eccentric orbits
    ExoplanetEntity("PSR_B1257_12b", "extreme", 0.01, 0.20, 0.05, 0.99, 0.01, 0.01, 0.01, 0.70),
    ExoplanetEntity("OGLE_2016_rogue", "extreme", 0.001, 0.08, 0.04, 0.50, 0.001, 0.50, 0.001, 0.40),
    ExoplanetEntity("HD_80606b", "extreme", 0.30, 0.70, 0.50, 0.07, 0.30, 0.80, 0.10, 0.90),
)


@dataclass(frozen=True, slots=True)
class EPKernelResult:
    """Kernel output for an exoplanet entity."""

    name: str
    category: str
    F: float
    omega: float
    S: float
    C: float
    kappa: float
    IC: float
    regime: str


def _classify_regime(omega: float, F: float, S: float, C: float) -> str:
    if omega >= 0.30:
        return "Collapse"
    if omega < 0.038 and F > 0.90 and S < 0.15 and C < 0.14:
        return "Stable"
    return "Watch"


def compute_ep_kernel(entity: ExoplanetEntity) -> EPKernelResult:
    """Compute kernel invariants for an exoplanet entity."""
    c = np.clip(entity.trace_vector(), EPSILON, 1 - EPSILON)
    w = np.ones(N_EP_CHANNELS) / N_EP_CHANNELS
    result = compute_kernel_outputs(c, w)
    F = float(result["F"])
    omega = float(result["omega"])
    S = float(result["S"])
    C = float(result["C"])
    kappa = float(result["kappa"])
    IC = float(result["IC"])
    regime = _classify_regime(omega, F, S, C)
    return EPKernelResult(
        name=entity.name,
        category=entity.category,
        F=F,
        omega=omega,
        S=S,
        C=C,
        kappa=kappa,
        IC=IC,
        regime=regime,
    )


def compute_all_entities() -> list[EPKernelResult]:
    """Compute kernel for all exoplanet entities."""
    return [compute_ep_kernel(e) for e in EP_ENTITIES]


# ---------------------------------------------------------------------------
# Theorems T-EP-1 through T-EP-6
# ---------------------------------------------------------------------------


def verify_t_ep_1(results: list[EPKernelResult]) -> dict:
    """T-EP-1: Extreme category has lowest mean IC (dead channels destroy coherence)."""
    cats: dict[str, list[float]] = {}
    for r in results:
        cats.setdefault(r.category, []).append(r.IC)
    extreme_ic = np.mean(cats["extreme"])
    others_ic = np.mean([ic for cat, ics in cats.items() if cat != "extreme" for ic in ics])
    passed = extreme_ic < others_ic
    return {
        "name": "T-EP-1",
        "passed": bool(passed),
        "extreme_mean_IC": float(extreme_ic),
        "others_mean_IC": float(others_ic),
    }


def verify_t_ep_2(results: list[EPKernelResult]) -> dict:
    """T-EP-2: Rocky HZ planets have high Δ (atmosphere channel kills IC while F moderate)."""
    rocky = [r for r in results if r.category == "rocky_hz"]
    rocky_delta = np.mean([r.F - r.IC for r in rocky])
    hot = [r for r in results if r.category == "hot_giant"]
    hot_delta = np.mean([r.F - r.IC for r in hot])
    passed = rocky_delta > hot_delta
    return {
        "name": "T-EP-2",
        "passed": bool(passed),
        "rocky_hz_mean_delta": float(rocky_delta),
        "hot_giant_mean_delta": float(hot_delta),
    }


def verify_t_ep_3(results: list[EPKernelResult]) -> dict:
    """T-EP-3: At least 2 distinct regimes present across the catalog."""
    regimes = {r.regime for r in results}
    passed = len(regimes) >= 2
    return {
        "name": "T-EP-3",
        "passed": bool(passed),
        "regimes": sorted(regimes),
        "count": len(regimes),
    }


def verify_t_ep_4(results: list[EPKernelResult]) -> dict:
    """T-EP-4: OGLE_2016_rogue has highest ω (most drift — unbound planet)."""
    rogue = next(r for r in results if r.name == "OGLE_2016_rogue")
    max_omega = max(r.omega for r in results)
    passed = rogue.omega >= max_omega - 1e-12
    return {
        "name": "T-EP-4",
        "passed": bool(passed),
        "rogue_omega": rogue.omega,
        "max_omega": float(max_omega),
    }


def verify_t_ep_5(results: list[EPKernelResult]) -> dict:
    """T-EP-5: Hot giants have highest mean detection_confidence → highest mean F in their class."""
    hot = [r for r in results if r.category == "hot_giant"]
    sub = [r for r in results if r.category == "sub_neptune"]
    hot_f = np.mean([r.F for r in hot])
    sub_f = np.mean([r.F for r in sub])
    passed = hot_f > sub_f
    return {
        "name": "T-EP-5",
        "passed": bool(passed),
        "hot_giant_mean_F": float(hot_f),
        "sub_neptune_mean_F": float(sub_f),
    }


def verify_t_ep_6(results: list[EPKernelResult]) -> dict:
    """T-EP-6: Duality identity F + ω = 1 holds for all entities."""
    residuals = [abs(r.F + r.omega - 1.0) for r in results]
    max_res = max(residuals)
    passed = max_res < 1e-12
    return {
        "name": "T-EP-6",
        "passed": bool(passed),
        "max_residual": float(max_res),
    }


def verify_all_theorems() -> list[dict]:
    """Run all T-EP theorems."""
    results = compute_all_entities()
    return [
        verify_t_ep_1(results),
        verify_t_ep_2(results),
        verify_t_ep_3(results),
        verify_t_ep_4(results),
        verify_t_ep_5(results),
        verify_t_ep_6(results),
    ]


if __name__ == "__main__":
    for t in verify_all_theorems():
        status = "PROVEN" if t["passed"] else "FAILED"
        print(f"  {t['name']}: {status}  {t}")

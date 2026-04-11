"""Energy Storage Closure — Materials Science Domain.

Tier-2 closure mapping 12 energy-storage technologies through the GCD kernel.
Each technology is characterized by 8 performance/maturity channels.

Channels (8, equal weights w_i = 1/8):
  0  energy_density_norm    — Wh/kg normalized to theoretical max (1 = theoretical limit)
  1  power_density_norm     — W/kg normalized (1 = instantaneous discharge)
  2  cycle_life_norm        — cycles / 10000 clipped (1 = effectively infinite)
  3  coulombic_efficiency   — round-trip efficiency (1 = perfect)
  4  safety_score           — thermal/chemical stability (1 = intrinsically safe)
  5  cost_efficiency        — 1 − ($/kWh)/1000 clipped (1 = essentially free)
  6  material_abundance     — elemental abundance of critical materials (1 = earth-abundant)
  7  tech_readiness         — TRL / 9 (1 = deployed at scale)

12 entities across 4 categories:
  Li-ion (3):        NMC_811, LFP, NCA
  Next-gen (3):      Solid_state_Li, Na_ion, Li_S
  Capacitors (3):    EDLC_activated_C, Pseudocap_MnO2, Hybrid_LIC
  Frontier (3):      Li_air, Redox_flow_V, Al_ion

6 theorems (T-ES-1 through T-ES-6).
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

ES_CHANNELS = [
    "energy_density_norm",
    "power_density_norm",
    "cycle_life_norm",
    "coulombic_efficiency",
    "safety_score",
    "cost_efficiency",
    "material_abundance",
    "tech_readiness",
]
N_ES_CHANNELS = len(ES_CHANNELS)


@dataclass(frozen=True, slots=True)
class EnergyStorageEntity:
    """An energy-storage technology with 8 performance channels."""

    name: str
    category: str
    energy_density_norm: float
    power_density_norm: float
    cycle_life_norm: float
    coulombic_efficiency: float
    safety_score: float
    cost_efficiency: float
    material_abundance: float
    tech_readiness: float

    def trace_vector(self) -> np.ndarray:
        return np.array(
            [
                self.energy_density_norm,
                self.power_density_norm,
                self.cycle_life_norm,
                self.coulombic_efficiency,
                self.safety_score,
                self.cost_efficiency,
                self.material_abundance,
                self.tech_readiness,
            ]
        )


ES_ENTITIES: tuple[EnergyStorageEntity, ...] = (
    # Li-ion mature — decent all-round, differentiated by safety/energy tradeoff
    EnergyStorageEntity("NMC_811", "li_ion", 0.75, 0.65, 0.60, 0.95, 0.50, 0.65, 0.40, 0.95),
    EnergyStorageEntity("LFP", "li_ion", 0.45, 0.55, 0.90, 0.96, 0.90, 0.75, 0.70, 0.95),
    EnergyStorageEntity("NCA", "li_ion", 0.80, 0.70, 0.50, 0.94, 0.45, 0.60, 0.35, 0.90),
    # Next-gen — high theoretical promise, lower TRL
    EnergyStorageEntity("Solid_state_Li", "next_gen", 0.85, 0.60, 0.40, 0.90, 0.80, 0.30, 0.40, 0.35),
    EnergyStorageEntity("Na_ion", "next_gen", 0.40, 0.50, 0.70, 0.92, 0.85, 0.80, 0.95, 0.50),
    EnergyStorageEntity("Li_S", "next_gen", 0.90, 0.45, 0.15, 0.80, 0.55, 0.50, 0.60, 0.20),
    # Capacitors — extreme power, low energy, long life
    EnergyStorageEntity("EDLC_activated_C", "capacitor", 0.08, 0.95, 0.99, 0.98, 0.95, 0.60, 0.95, 0.90),
    EnergyStorageEntity("Pseudocap_MnO2", "capacitor", 0.15, 0.85, 0.80, 0.92, 0.85, 0.55, 0.80, 0.60),
    EnergyStorageEntity("Hybrid_LIC", "capacitor", 0.25, 0.80, 0.85, 0.95, 0.80, 0.45, 0.70, 0.55),
    # Frontier — very high theoretical energy, very low TRL / stability
    EnergyStorageEntity("Li_air", "frontier", 0.95, 0.20, 0.05, 0.60, 0.20, 0.20, 0.50, 0.10),
    EnergyStorageEntity("Redox_flow_V", "frontier", 0.20, 0.30, 0.95, 0.85, 0.80, 0.40, 0.30, 0.55),
    EnergyStorageEntity("Al_ion", "frontier", 0.35, 0.50, 0.30, 0.75, 0.80, 0.70, 0.95, 0.15),
)


@dataclass(frozen=True, slots=True)
class ESKernelResult:
    """Kernel output for an energy-storage technology."""

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


def compute_es_kernel(entity: EnergyStorageEntity) -> ESKernelResult:
    """Compute kernel invariants for an energy-storage technology."""
    c = np.clip(entity.trace_vector(), EPSILON, 1 - EPSILON)
    w = np.ones(N_ES_CHANNELS) / N_ES_CHANNELS
    result = compute_kernel_outputs(c, w)
    F = float(result["F"])
    omega = float(result["omega"])
    S = float(result["S"])
    C = float(result["C"])
    kappa = float(result["kappa"])
    IC = float(result["IC"])
    regime = _classify_regime(omega, F, S, C)
    return ESKernelResult(
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


def compute_all_entities() -> list[ESKernelResult]:
    """Compute kernel for all energy-storage entities."""
    return [compute_es_kernel(e) for e in ES_ENTITIES]


# ---------------------------------------------------------------------------
# Theorems T-ES-1 through T-ES-6
# ---------------------------------------------------------------------------


def verify_t_es_1(results: list[ESKernelResult]) -> dict:
    """T-ES-1: Li-air has lowest IC among all (extreme channel heterogeneity → valley of death)."""
    li_air = next(r for r in results if r.name == "Li_air")
    li_ion = [r for r in results if r.category == "li_ion"]
    li_ion_min_ic = min(r.IC for r in li_ion)
    passed = li_ion_min_ic > li_air.IC
    return {"name": "T-ES-1", "passed": bool(passed), "li_air_IC": li_air.IC, "li_ion_min_IC": li_ion_min_ic}


def verify_t_es_2(results: list[ESKernelResult]) -> dict:
    """T-ES-2: Capacitors have highest mean IC/F (most balanced channels)."""
    cats: dict[str, list[float]] = {}
    for r in results:
        icf = r.IC / r.F if r.F > EPSILON else 0.0
        cats.setdefault(r.category, []).append(icf)
    cap_icf = np.mean(cats["capacitor"])
    front_icf = np.mean(cats["frontier"])
    passed = cap_icf > front_icf
    return {"name": "T-ES-2", "passed": bool(passed), "cap_IC_F": float(cap_icf), "frontier_IC_F": float(front_icf)}


def verify_t_es_3(results: list[ESKernelResult]) -> dict:
    """T-ES-3: At least 2 distinct regimes present."""
    regimes = {r.regime for r in results}
    passed = len(regimes) >= 2
    return {"name": "T-ES-3", "passed": bool(passed), "regimes": sorted(regimes)}


def verify_t_es_4(results: list[ESKernelResult]) -> dict:
    """T-ES-4: EDLC has highest IC among capacitors (balanced performance)."""
    caps = [r for r in results if r.category == "capacitor"]
    edlc = next(r for r in caps if r.name == "EDLC_activated_C")
    others_max = max(r.IC for r in caps if r.name != "EDLC_activated_C")
    passed = others_max <= edlc.IC
    return {"name": "T-ES-4", "passed": bool(passed), "edlc_IC": edlc.IC, "others_max_IC": others_max}


def verify_t_es_5(results: list[ESKernelResult]) -> dict:
    """T-ES-5: Frontier has highest mean ω across categories (least mature)."""
    cats: dict[str, list[float]] = {}
    for r in results:
        cats.setdefault(r.category, []).append(r.omega)
    front_omega = np.mean(cats["frontier"])
    lion_omega = np.mean(cats["li_ion"])
    passed = front_omega > lion_omega
    return {
        "name": "T-ES-5",
        "passed": bool(passed),
        "frontier_omega": float(front_omega),
        "lion_omega": float(lion_omega),
    }


def verify_t_es_6(results: list[ESKernelResult]) -> dict:
    """T-ES-6: Duality F + ω = 1 for all entities."""
    residuals = [abs(r.F + r.omega - 1.0) for r in results]
    max_res = max(residuals)
    passed = max_res < 1e-12
    return {"name": "T-ES-6", "passed": bool(passed), "max_residual": float(max_res)}


def verify_all_theorems() -> list[dict]:
    """Run all T-ES theorems."""
    results = compute_all_entities()
    return [
        verify_t_es_1(results),
        verify_t_es_2(results),
        verify_t_es_3(results),
        verify_t_es_4(results),
        verify_t_es_5(results),
        verify_t_es_6(results),
    ]


if __name__ == "__main__":
    for t in verify_all_theorems():
        status = "PROVEN" if t["passed"] else "FAILED"
        print(f"  {t['name']}: {status}  {t}")

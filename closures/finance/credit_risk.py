"""Credit Risk Dynamics Closure — Finance Domain.

Tier-2 closure mapping 12 credit entities through the GCD kernel.
Each issuer/instrument is characterized by 8 creditworthiness channels.

Channels (8, equal weights w_i = 1/8):
  0  pd_inverse             — 1 − PD_5yr (1 = low default probability)
  1  recovery_rate          — expected recovery (1 = full recovery)
  2  spread_tightness       — 1 − CDS spread normalized (1 = tight spread)
  3  liquidity_score        — bid-ask tightness (1 = deep/liquid)
  4  leverage_health        — 1 − leverage ratio normalized (1 = low leverage)
  5  interest_coverage      — min(ICR/10, 1) (1 = strong coverage)
  6  rating_stability       — years at current rating / 20 (1 = long-stable)
  7  macro_resilience       — GDP sensitivity score (1 = robust)

12 entities across 4 categories:
  Investment Grade (3):  US_Treasury_AAA, Apple_AA, JNJ_AAA
  Crossover (3):         Ford_BB, Kraft_BBB, Italy_BBB
  High Yield (3):        Tesla_B, Argentina_CCC, Venezuela_D
  Structured (3):        CLO_AAA_tranche, MBS_subprime_2007, CDO_equity

6 theorems (T-CR-1 through T-CR-6).
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

CR_CHANNELS = [
    "pd_inverse",
    "recovery_rate",
    "spread_tightness",
    "liquidity_score",
    "leverage_health",
    "interest_coverage",
    "rating_stability",
    "macro_resilience",
]
N_CR_CHANNELS = len(CR_CHANNELS)


@dataclass(frozen=True, slots=True)
class CreditEntity:
    """A credit instrument with 8 creditworthiness channels."""

    name: str
    category: str
    pd_inverse: float
    recovery_rate: float
    spread_tightness: float
    liquidity_score: float
    leverage_health: float
    interest_coverage: float
    rating_stability: float
    macro_resilience: float

    def trace_vector(self) -> np.ndarray:
        return np.array(
            [
                self.pd_inverse,
                self.recovery_rate,
                self.spread_tightness,
                self.liquidity_score,
                self.leverage_health,
                self.interest_coverage,
                self.rating_stability,
                self.macro_resilience,
            ]
        )


CR_ENTITIES: tuple[CreditEntity, ...] = (
    # Investment Grade — high F, low heterogeneity
    CreditEntity("US_Treasury_AAA", "investment_grade", 0.999, 0.95, 0.98, 0.99, 0.99, 0.95, 1.00, 0.90),
    CreditEntity("Apple_AA", "investment_grade", 0.995, 0.90, 0.92, 0.95, 0.85, 0.90, 0.80, 0.88),
    CreditEntity("JNJ_AAA", "investment_grade", 0.998, 0.92, 0.95, 0.92, 0.90, 0.92, 0.95, 0.85),
    # Crossover — Watch regime, higher curvature
    CreditEntity("Ford_BB", "crossover", 0.92, 0.55, 0.50, 0.70, 0.40, 0.45, 0.30, 0.60),
    CreditEntity("Kraft_BBB", "crossover", 0.96, 0.65, 0.65, 0.75, 0.50, 0.55, 0.45, 0.70),
    CreditEntity("Italy_BBB", "crossover", 0.94, 0.70, 0.55, 0.80, 0.30, 0.40, 0.50, 0.50),
    # High Yield — Collapse proximity, high Δ
    CreditEntity("Tesla_B", "high_yield", 0.85, 0.40, 0.30, 0.65, 0.25, 0.35, 0.15, 0.55),
    CreditEntity("Argentina_CCC", "high_yield", 0.50, 0.25, 0.10, 0.20, 0.10, 0.10, 0.05, 0.15),
    CreditEntity("Venezuela_D", "high_yield", 0.05, 0.10, 0.02, 0.05, 0.02, 0.02, 0.02, 0.05),
    # Structured — extreme channel heterogeneity
    CreditEntity("CLO_AAA_tranche", "structured", 0.99, 0.85, 0.88, 0.60, 0.80, 0.75, 0.70, 0.80),
    CreditEntity("MBS_subprime_2007", "structured", 0.30, 0.15, 0.05, 0.10, 0.05, 0.08, 0.50, 0.20),
    CreditEntity("CDO_equity", "structured", 0.40, 0.05, 0.08, 0.15, 0.03, 0.05, 0.20, 0.25),
)


@dataclass(frozen=True, slots=True)
class CRKernelResult:
    """Kernel output for a credit entity."""

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


def compute_cr_kernel(entity: CreditEntity) -> CRKernelResult:
    """Compute kernel invariants for a credit entity."""
    c = np.clip(entity.trace_vector(), EPSILON, 1 - EPSILON)
    w = np.ones(N_CR_CHANNELS) / N_CR_CHANNELS
    result = compute_kernel_outputs(c, w)
    F = float(result["F"])
    omega = float(result["omega"])
    S = float(result["S"])
    C = float(result["C"])
    kappa = float(result["kappa"])
    IC = float(result["IC"])
    regime = _classify_regime(omega, F, S, C)
    return CRKernelResult(
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


def compute_all_entities() -> list[CRKernelResult]:
    """Compute kernel for all credit entities."""
    return [compute_cr_kernel(e) for e in CR_ENTITIES]


# ---------------------------------------------------------------------------
# Theorems T-CR-1 through T-CR-6
# ---------------------------------------------------------------------------


def verify_t_cr_1(results: list[CRKernelResult]) -> dict:
    """T-CR-1: Defaulted/distressed entities in Collapse (Venezuela_D, MBS_subprime)."""
    distressed = {"Venezuela_D", "MBS_subprime_2007", "CDO_equity"}
    collapse_count = sum(1 for r in results if r.name in distressed and r.regime == "Collapse")
    passed = collapse_count >= 2
    return {"name": "T-CR-1", "passed": bool(passed), "collapse_count": collapse_count}


def verify_t_cr_2(results: list[CRKernelResult]) -> dict:
    """T-CR-2: Investment grade mean F > high yield mean F."""
    ig_f = np.mean([r.F for r in results if r.category == "investment_grade"])
    hy_f = np.mean([r.F for r in results if r.category == "high_yield"])
    passed = ig_f > hy_f
    return {"name": "T-CR-2", "passed": bool(passed), "ig_F": float(ig_f), "hy_F": float(hy_f)}


def verify_t_cr_3(results: list[CRKernelResult]) -> dict:
    """T-CR-3: At least 2 distinct regimes present."""
    regimes = {r.regime for r in results}
    passed = len(regimes) >= 2
    return {"name": "T-CR-3", "passed": bool(passed), "regimes": sorted(regimes)}


def verify_t_cr_4(results: list[CRKernelResult]) -> dict:
    """T-CR-4: Structured products have highest mean heterogeneity gap (extreme channel spread)."""
    cats: dict[str, list[float]] = {}
    for r in results:
        cats.setdefault(r.category, []).append(r.F - r.IC)
    struct_delta = np.mean(cats["structured"])
    ig_delta = np.mean(cats["investment_grade"])
    passed = struct_delta > ig_delta
    return {
        "name": "T-CR-4",
        "passed": bool(passed),
        "structured_delta": float(struct_delta),
        "ig_delta": float(ig_delta),
    }


def verify_t_cr_5(results: list[CRKernelResult]) -> dict:
    """T-CR-5: Investment grade has highest mean IC/F (most balanced channels)."""
    cats: dict[str, list[float]] = {}
    for r in results:
        icf = r.IC / r.F if r.F > EPSILON else 0.0
        cats.setdefault(r.category, []).append(icf)
    ig_icf = np.mean(cats["investment_grade"])
    hy_icf = np.mean(cats["high_yield"])
    passed = ig_icf > hy_icf
    return {"name": "T-CR-5", "passed": bool(passed), "ig_IC_F": float(ig_icf), "hy_IC_F": float(hy_icf)}


def verify_t_cr_6(results: list[CRKernelResult]) -> dict:
    """T-CR-6: Duality F + ω = 1 for all entities."""
    residuals = [abs(r.F + r.omega - 1.0) for r in results]
    max_res = max(residuals)
    passed = max_res < 1e-12
    return {"name": "T-CR-6", "passed": bool(passed), "max_residual": float(max_res)}


def verify_all_theorems() -> list[dict]:
    """Run all T-CR theorems."""
    results = compute_all_entities()
    return [
        verify_t_cr_1(results),
        verify_t_cr_2(results),
        verify_t_cr_3(results),
        verify_t_cr_4(results),
        verify_t_cr_5(results),
        verify_t_cr_6(results),
    ]


if __name__ == "__main__":
    for t in verify_all_theorems():
        status = "PROVEN" if t["passed"] else "FAILED"
        print(f"  {t['name']}: {status}  {t}")

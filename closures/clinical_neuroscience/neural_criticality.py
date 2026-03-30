"""Neural Criticality Closure — Clinical Neuroscience Domain.

Tier-2 closure mapping 12 brain states through the GCD kernel with
criticality-specific channels.  Motivated by the PRL finding (March 2026)
that the human brain operates *near*, but not *at*, the critical point.

In GCD terms: healthy waking maps to Watch regime — close to the
Stable/Watch boundary but never crossing it.  Pathological deviations
(seizure = supercritical runaway; anesthesia/coma = subcritical collapse)
both land in Collapse regime, demonstrating that the critical manifold
is the Watch boundary itself.

Channels (8, equal weights w_i = 1/8):
  0  branching_ratio       — avalanche propagation σ (1 = critical, <1 sub, >1 super)
  1  power_law_exponent    — avalanche size τ normalized (τ≈1.5 → 1.0 = critical)
  2  long_range_correlation — DFA exponent α (0.5 = uncorrelated, 1.0 = 1/f)
  3  spectral_slope        — 1/f^β slope (β≈1 → 1.0 = near-critical)
  4  susceptibility        — neural response amplitude (1 = maximal near criticality)
  5  correlation_length    — spatial extent of correlated activity (1 = maximal)
  6  entropy_rate          — temporal entropy production (1 = maximal)
  7  dynamic_range         — stimulus discrimination capacity (1 = maximal)

12 entities across 4 categories:
  Healthy waking (5): Alert_waking, Focused_attention, Mind_wandering,
                      Deep_meditation, Flow_state
  Sleep (2):          NREM3_slow_wave, REM_sleep
  Pathological (3):   Psychedelic_psilocybin, General_anesthesia, Coma
  Extreme (2):        Epileptic_seizure, Infant_6mo

7 theorems (T-NCR-1 through T-NCR-7).

References:
  - Beggs & Plenz (2003). Neuronal avalanches in neocortical circuits.
  - Shew & Plenz (2013). The functional benefits of criticality.
  - Kinouchi & Copelli (2006). Optimal dynamical range of excitable networks.
  - Carhart-Harris et al. (2014). The entropic brain.
  - PRL March 2026: Human brain operates near, but not at, the critical point.
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

NCR_CHANNELS = [
    "branching_ratio",
    "power_law_exponent",
    "long_range_correlation",
    "spectral_slope",
    "susceptibility",
    "correlation_length",
    "entropy_rate",
    "dynamic_range",
]
N_NCR_CHANNELS = len(NCR_CHANNELS)


@dataclass(frozen=True, slots=True)
class CriticalityEntity:
    """A brain state characterized by 8 criticality channels."""

    name: str
    category: str
    branching_ratio: float
    power_law_exponent: float
    long_range_correlation: float
    spectral_slope: float
    susceptibility: float
    correlation_length: float
    entropy_rate: float
    dynamic_range: float

    def trace_vector(self) -> np.ndarray:
        return np.array(
            [
                self.branching_ratio,
                self.power_law_exponent,
                self.long_range_correlation,
                self.spectral_slope,
                self.susceptibility,
                self.correlation_length,
                self.entropy_rate,
                self.dynamic_range,
            ]
        )


# ---------------------------------------------------------------------------
# Entity catalog — 12 brain states
# ---------------------------------------------------------------------------
# Channel values represent normalized proximity to criticality.
# σ ≈ 1.0 = critical branching; τ ≈ 1.5 mapped to 1.0; etc.
# Values are informed by the neuroscience literature on criticality
# biomarkers across brain states.

NCR_ENTITIES: tuple[CriticalityEntity, ...] = (
    # --- Healthy waking (5) — near-critical, Watch regime expected ---
    #                                  σ     τ     α     β     χ     ξ     h     Δ_R
    CriticalityEntity(
        "Alert_waking",
        "healthy",
        0.95,
        0.90,
        0.85,
        0.82,
        0.88,
        0.80,
        0.75,
        0.90,
    ),
    CriticalityEntity(
        "Focused_attention",
        "healthy",
        0.92,
        0.88,
        0.80,
        0.78,
        0.85,
        0.75,
        0.65,
        0.92,
    ),
    CriticalityEntity(
        "Mind_wandering",
        "healthy",
        0.88,
        0.85,
        0.82,
        0.80,
        0.82,
        0.78,
        0.82,
        0.85,
    ),
    CriticalityEntity(
        "Deep_meditation",
        "healthy",
        0.90,
        0.92,
        0.90,
        0.88,
        0.80,
        0.85,
        0.60,
        0.82,
    ),
    CriticalityEntity(
        "Flow_state",
        "healthy",
        0.93,
        0.91,
        0.88,
        0.85,
        0.92,
        0.82,
        0.70,
        0.95,
    ),
    # --- Sleep (2) — REM near-critical, NREM3 ordered/subcritical ---
    CriticalityEntity(
        "NREM3_slow_wave",
        "sleep",
        0.55,
        0.40,
        0.60,
        0.45,
        0.30,
        0.35,
        0.25,
        0.30,
    ),
    CriticalityEntity(
        "REM_sleep",
        "sleep",
        0.85,
        0.82,
        0.78,
        0.75,
        0.80,
        0.72,
        0.80,
        0.78,
    ),
    # --- Pathological (3) — deviations from criticality ---
    CriticalityEntity(
        "Psychedelic_psilocybin",
        "pathological",
        0.70,
        0.60,
        0.72,
        0.65,
        0.75,
        0.68,
        0.95,
        0.70,
    ),
    CriticalityEntity(
        "General_anesthesia",
        "pathological",
        0.15,
        0.20,
        0.30,
        0.25,
        0.10,
        0.15,
        0.12,
        0.10,
    ),
    CriticalityEntity(
        "Coma",
        "pathological",
        0.08,
        0.10,
        0.15,
        0.12,
        0.05,
        0.08,
        0.05,
        0.05,
    ),
    # --- Extreme (2) — seizure (supercritical) + infant (developing) ---
    CriticalityEntity(
        "Epileptic_seizure",
        "extreme",
        0.60,
        0.35,
        0.55,
        0.40,
        0.90,
        0.95,
        0.92,
        0.20,
    ),
    CriticalityEntity(
        "Infant_6mo",
        "extreme",
        0.55,
        0.50,
        0.50,
        0.48,
        0.45,
        0.40,
        0.60,
        0.45,
    ),
)


@dataclass(frozen=True, slots=True)
class NCRKernelResult:
    """Kernel output for a neural criticality entity."""

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


def compute_ncr_kernel(entity: CriticalityEntity) -> NCRKernelResult:
    """Compute kernel invariants for a criticality entity."""
    c = np.clip(entity.trace_vector(), EPSILON, 1 - EPSILON)
    w = np.ones(N_NCR_CHANNELS) / N_NCR_CHANNELS
    result = compute_kernel_outputs(c, w)
    F = float(result["F"])
    omega = float(result["omega"])
    S = float(result["S"])
    C = float(result["C"])
    kappa = float(result["kappa"])
    IC = float(result["IC"])
    regime = _classify_regime(omega, F, S, C)
    return NCRKernelResult(
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


def compute_all_entities() -> list[NCRKernelResult]:
    """Compute kernel for all criticality entities."""
    return [compute_ncr_kernel(e) for e in NCR_ENTITIES]


# ---------------------------------------------------------------------------
# Theorems T-NCR-1 through T-NCR-7
# ---------------------------------------------------------------------------


def verify_t_ncr_1(results: list[NCRKernelResult]) -> dict:
    """T-NCR-1: Near-Critical Watch — All healthy waking states occupy Watch regime.

    None reach Stable. The brain lives near, but not at, the critical point.
    This is the GCD translation of the PRL March 2026 finding.
    """
    healthy = [r for r in results if r.category == "healthy"]
    all_watch = all(r.regime == "Watch" for r in healthy)
    return {
        "name": "T-NCR-1",
        "passed": bool(all_watch),
        "healthy_regimes": {r.name: r.regime for r in healthy},
        "count_watch": sum(1 for r in healthy if r.regime == "Watch"),
        "count_total": len(healthy),
    }


def verify_t_ncr_2(results: list[NCRKernelResult]) -> dict:
    """T-NCR-2: Criticality = Watch Boundary.

    Mean ω of healthy waking states lies in Watch range [0.038, 0.30),
    and branching ratio σ ≈ 1 maps to the lower end of Watch (near Stable).
    """
    healthy = [r for r in results if r.category == "healthy"]
    mean_omega = float(np.mean([r.omega for r in healthy]))
    in_watch = 0.038 <= mean_omega < 0.30
    return {
        "name": "T-NCR-2",
        "passed": bool(in_watch),
        "mean_omega_healthy": mean_omega,
        "watch_range": (0.038, 0.30),
    }


def verify_t_ncr_3(results: list[NCRKernelResult]) -> dict:
    """T-NCR-3: Dynamic Range Maximum in Watch.

    Mean dynamic_range channel value for Watch-regime entities exceeds
    that of Collapse-regime entities.  Criticality theory predicts maximal
    dynamic range near the critical point.
    """
    watch_ents = [e for e in NCR_ENTITIES if compute_ncr_kernel(e).regime == "Watch"]
    collapse_ents = [e for e in NCR_ENTITIES if compute_ncr_kernel(e).regime == "Collapse"]
    watch_dr = float(np.mean([e.dynamic_range for e in watch_ents])) if watch_ents else 0.0
    collapse_dr = float(np.mean([e.dynamic_range for e in collapse_ents])) if collapse_ents else 0.0
    passed = watch_dr > collapse_dr
    return {
        "name": "T-NCR-3",
        "passed": bool(passed),
        "watch_mean_dynamic_range": watch_dr,
        "collapse_mean_dynamic_range": collapse_dr,
        "n_watch": len(watch_ents),
        "n_collapse": len(collapse_ents),
    }


def verify_t_ncr_4(results: list[NCRKernelResult]) -> dict:
    """T-NCR-4: Pathological Deviation — Opposite Extremes, Same Regime.

    Seizure (supercritical) and coma (subcritical) both land in Collapse
    regime despite being on opposite sides of criticality.  The regime
    classification detects deviation from near-criticality regardless
    of direction.
    """
    seizure = next(r for r in results if r.name == "Epileptic_seizure")
    coma = next(r for r in results if r.name == "Coma")
    anesthesia = next(r for r in results if r.name == "General_anesthesia")
    all_collapse = all(r.regime == "Collapse" for r in [seizure, coma, anesthesia])
    return {
        "name": "T-NCR-4",
        "passed": bool(all_collapse),
        "seizure_regime": seizure.regime,
        "coma_regime": coma.regime,
        "anesthesia_regime": anesthesia.regime,
        "seizure_omega": seizure.omega,
        "coma_omega": coma.omega,
    }


def verify_t_ncr_5(results: list[NCRKernelResult]) -> dict:
    """T-NCR-5: Heterogeneity Gap Amplification at Extremes.

    Seizure (supercritical runaway) produces the largest heterogeneity
    gap Δ = F − IC of all entities.  Supercritical dynamics drive
    some channels (susceptibility, correlation_length) high while
    crushing others (dynamic_range), creating maximal channel
    divergence.  The gap for seizure exceeds that of any healthy state.
    """
    seizure = next(r for r in results if r.name == "Epileptic_seizure")
    healthy = [r for r in results if r.category == "healthy"]
    sz_delta = seizure.F - seizure.IC
    max_healthy_delta = max(r.F - r.IC for r in healthy)
    passed = sz_delta > max_healthy_delta
    return {
        "name": "T-NCR-5",
        "passed": bool(passed),
        "seizure_delta": float(sz_delta),
        "max_healthy_delta": float(max_healthy_delta),
        "amplification_ratio": float(sz_delta / max_healthy_delta) if max_healthy_delta > 0 else float("inf"),
    }


def verify_t_ncr_6(results: list[NCRKernelResult]) -> dict:
    """T-NCR-6: Developmental Trajectory.

    Infant → adult follows Collapse → Watch trajectory as the brain
    develops toward near-criticality.  Infant ω > healthy adult mean ω.
    """
    infant = next(r for r in results if r.name == "Infant_6mo")
    healthy = [r for r in results if r.category == "healthy"]
    mean_adult_omega = float(np.mean([r.omega for r in healthy]))
    passed = infant.omega > mean_adult_omega
    return {
        "name": "T-NCR-6",
        "passed": bool(passed),
        "infant_omega": infant.omega,
        "mean_adult_omega": mean_adult_omega,
        "infant_regime": infant.regime,
    }


def verify_t_ncr_7(results: list[NCRKernelResult]) -> dict:
    """T-NCR-7: Watch as the Cognitive Tension Zone.

    Cognition requires three simultaneously non-trivial arms of tension:
      - S > 0.30  (input: meaningful uncertainty is present and being processed)
      - C > 0.05  (implication: channels differ, things point in different directions)
      - F > 0.80  (conviction: the system commits despite the uncertainty)

    Watch is the only regime where all three hold together.  The brain is
    bound between two failure modes: Stable (kills S and C — zero tension,
    nothing implies anything) and Collapse (kills F — conviction lost,
    return blocked).  Cognition lives in the tension, not despite it.

    Vita est reditus continuus.
    """
    healthy = [r for r in results if r.category == "healthy"]
    # All healthy (Watch) entities must have all three arms non-trivial
    s_threshold = 0.30
    c_threshold = 0.05
    f_threshold = 0.80
    all_watch_have_tension = all(s_threshold < r.S and c_threshold < r.C and f_threshold < r.F for r in healthy)
    # Brain never enters Stable — the zero-tension frozen zone
    no_stable = not any(r.regime == "Stable" for r in results)
    # Pathological Collapse loses conviction (F) — not just S or C
    path_collapse = [r for r in results if r.category == "pathological" and r.regime == "Collapse"]
    conviction_lost = all(r.F < 0.20 for r in path_collapse)
    passed = all_watch_have_tension and no_stable and conviction_lost
    return {
        "name": "T-NCR-7",
        "passed": bool(passed),
        "all_watch_have_tension": bool(all_watch_have_tension),
        "no_stable_entities": bool(no_stable),
        "pathological_collapse_conviction_lost": bool(conviction_lost),
        "healthy_s_min": float(min(r.S for r in healthy)),
        "healthy_c_min": float(min(r.C for r in healthy)),
        "healthy_f_min": float(min(r.F for r in healthy)),
    }


def verify_all_theorems() -> list[dict]:
    """Run all T-NCR theorems."""
    results = compute_all_entities()
    return [
        verify_t_ncr_1(results),
        verify_t_ncr_2(results),
        verify_t_ncr_3(results),
        verify_t_ncr_4(results),
        verify_t_ncr_5(results),
        verify_t_ncr_6(results),
        verify_t_ncr_7(results),
    ]


if __name__ == "__main__":
    print("Neural Criticality Closure — Computing all entities...\n")
    results = compute_all_entities()
    for r in results:
        delta = r.F - r.IC
        ic_f = r.IC / r.F if r.F > 0 else 0.0
        print(
            f"  {r.name:<30s}  F={r.F:.4f}  ω={r.omega:.4f}  "
            f"IC={r.IC:.4f}  Δ={delta:.4f}  IC/F={ic_f:.4f}  [{r.regime}]"
        )
    print("\nTheorems:")
    for t in verify_all_theorems():
        status = "PASS" if t["passed"] else "FAIL"
        print(f"  {t['name']}: {status}")

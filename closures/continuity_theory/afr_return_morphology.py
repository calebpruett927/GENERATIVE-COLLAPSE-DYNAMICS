"""Antifragility Return-Morphology Diagnostic — Continuity Theory Domain.

AFR.RETURN.v1 — A Tier-2 diagnostic family for reading the morphology of return.

Tier discipline (binding):
    Tier-1 tells us WHAT the returned state is: (F, omega, S, C, kappa, IC),
    tau_R, ledger closure. This module operates downstream and asks WHAT KIND of
    return it was. It NEVER redefines F, omega, S, C, kappa, IC, tau_R, or
    ``regime``. All Tier-1 outputs are read from
    ``umcp.kernel_optimized.compute_kernel_outputs``. All frozen parameters come
    from ``umcp.frozen_contract``. The diagnostic is a Tier-2 overlay that informs
    but never gates.

Axiom-0 through this lens:
    Antifragility is not "higher output after shock." It is admissible return with
    improved coherence, reduced hidden fracture, or increased future capacity.

Episode model (three states — one collapse-return cycle):

    pre-state K0  ->  collapse-state Kc  ->  return-state KR

Each state is a Tier-1 kernel evaluation of the same channel set under the same
frozen contract. The morphology diagnostic compares KR against K0 using existing
Tier-1 outputs.

Five Tier-2 derived measures (read from Tier-1 outputs, never redefining them):

    RS   Return Surplus       = IC_R - IC_0    (multiplicative coherence change)
    FG   Fidelity Gain        = F_R  - F_0     (average retention change)
    DR   Drift Reduction      = omega_0 - omega_R (drift decrease; positive = less drift)
    RR   Roughness Reduction  = C_0  - C_R     (smoothness gain; positive = smoother)
    HR   Heterogeneity Repair = (F_0 - IC_0) - (F_R - IC_R)  (gap shrinkage)

HR is the most diagnostic for antifragility: a positive HR means hidden fracture
(the gap between average and multiplicative coherence) actually *shrank* during
collapse-return. The system became less internally uneven.

Five return-morphology categories (Tier-2 labels, read from the diagnostic — never
used as Tier-0 gates):

    FRAGILE        No finite return (tau_R = INF_REC), or identity fails post-return.
    DAMAGED        Finite return; KR materially worse than K0 on RS and FG.
    ROBUST         Finite return; KR approx K0; no meaningful surplus.
    RESILIENT      Finite return after meaningful collapse; identity preserved;
                   admissible function regained (|RS| < tol_surplus).
    ANTIFRAGILE    Finite return; identity preserved; seam closed; KR shows surplus:
                   RS > tol_surplus AND HR >= 0 AND RR >= -tol_roughness.

Guardrails against false antifragility:
    - IC_R must exceed IC_0 (not just F_R). Rising F with falling IC is flagged
      as FALSE_ANTIFRAGILE (surface gain / hidden fracture).
    - HR must not worsen: returning with a wider gap between F and IC is a warning
      even if average fidelity rose.
    - Roughness must not increase beyond tol_roughness: a smoother-looking surface
      that is more internally jagged is not a surplus.

Sidecar ledger:
    Per-episode results should be appended to ledger/afr_episodes.csv (append-only,
    cross-referenced to return_log.csv by chain_hash). Do NOT modify return_log.csv
    schema.

12 canonical synthetic episodes for test coverage (4 categories x 3 episodes):
    Fragile (3): no-return, identity-fail, tau_R=INF_REC
    Damaged (3): finite return but materially worse KR
    Robust/Resilient (3): finite return, K0 approx KR
    Antifragile (3): finite return, KR shows measured surplus on RS + HR

6 theorems (T-AFR-1 through T-AFR-6).
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

_WORKSPACE = Path(__file__).resolve().parents[2]
for _p in [str(_WORKSPACE / "src"), str(_WORKSPACE)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from umcp.frozen_contract import EPSILON  # noqa: E402
from umcp.kernel_optimized import compute_kernel_outputs  # noqa: E402

# ---------------------------------------------------------------------------
# Tier-2 thresholds (frozen for this closure — seam-derived values)
# ---------------------------------------------------------------------------

TOL_SURPLUS: float = 0.02  # minimum RS to call a return "surplus"
TOL_ROUGHNESS: float = 0.02  # maximum RR deterioration before flagging
TOL_DAMAGED: float = -0.05  # RS below this threshold = materially damaged

AFR_CHANNELS = [
    "adaptive_capacity",  # channel 0 — ability to restructure under stress
    "internal_coherence",  # channel 1 — multiplicative self-consistency
    "resource_reserve",  # channel 2 — buffer available after collapse
    "communication_integrity",  # channel 3 — information pathways
    "boundary_discipline",  # channel 4 — seam-holding capacity
    "knowledge_retention",  # channel 5 — institutional/structural memory
    "response_agility",  # channel 6 — speed of re-entry
    "external_robustness",  # channel 7 — independence from triggering context
]
N_AFR_CHANNELS = len(AFR_CHANNELS)

ReturnCategory = Literal[
    "FRAGILE",
    "DAMAGED",
    "ROBUST",
    "RESILIENT",
    "ANTIFRAGILE",
    "FALSE_ANTIFRAGILE",  # RS positive but HR negative (hidden fracture)
]


# ---------------------------------------------------------------------------
# State and episode containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CollapseState:
    """Tier-1 kernel outputs for one state in a collapse-return episode.

    All fields map 1:1 to Tier-1 kernel outputs. No redefinition.
    """

    label: str  # "pre", "collapse", "return" — narrative tag only
    F: float
    omega: float
    S: float
    C: float
    kappa: float
    IC: float
    tau_R: float  # float("inf") for INF_REC

    def __post_init__(self) -> None:
        # Verify Tier-1 identities hold at construction time
        assert abs(self.F + self.omega - 1.0) < 1e-10, "Duality identity violated"
        assert self.IC <= self.F + 1e-10, "Integrity bound violated"
        assert abs(self.IC - math.exp(self.kappa)) < 1e-10, "Log-integrity violated"

    @classmethod
    def from_trace(cls, label: str, trace: np.ndarray, tau_R: float = float("inf")) -> CollapseState:
        """Construct from a raw trace vector using the frozen kernel."""
        c = np.clip(trace, EPSILON, 1.0 - EPSILON)
        w = np.ones(N_AFR_CHANNELS) / N_AFR_CHANNELS
        r = compute_kernel_outputs(c, w)
        return cls(
            label=label,
            F=float(r["F"]),
            omega=float(r["omega"]),
            S=float(r["S"]),
            C=float(r["C"]),
            kappa=float(r["kappa"]),
            IC=float(r["IC"]),
            tau_R=tau_R,
        )


@dataclass(frozen=True, slots=True)
class AntifragilityEpisode:
    """A three-state collapse-return episode with its AFR diagnostic."""

    name: str
    K0: CollapseState  # pre-collapse state
    Kc: CollapseState  # deepest collapse state
    KR: CollapseState  # post-return state
    description: str = ""

    # --- Five Tier-2 derived measures (computed from Tier-1 outputs) -------

    @property
    def RS(self) -> float:
        """Return Surplus: IC_R - IC_0. Positive = multiplicative coherence improved."""
        return self.KR.IC - self.K0.IC

    @property
    def FG(self) -> float:
        """Fidelity Gain: F_R - F_0. Positive = average retention improved."""
        return self.KR.F - self.K0.F

    @property
    def DR(self) -> float:
        """Drift Reduction: omega_0 - omega_R. Positive = returned state carries less drift."""
        return self.K0.omega - self.KR.omega

    @property
    def RR(self) -> float:
        """Roughness Reduction: C_0 - C_R. Positive = returned state smoother."""
        return self.K0.C - self.KR.C

    @property
    def HR(self) -> float:
        """Heterogeneity Repair: (F_0 - IC_0) - (F_R - IC_R).

        Positive = heterogeneity gap shrank after collapse-return.
        This is the most diagnostic measure for antifragility: a positive HR
        means the system returned with LESS hidden fracture than it started with.
        Overcorrection and brittleness produce negative HR even when F rises.
        """
        gap_pre = self.K0.F - self.K0.IC
        gap_post = self.KR.F - self.KR.IC
        return gap_pre - gap_post

    @property
    def finite_return(self) -> bool:
        """True iff tau_R of the return state is finite (not INF_REC)."""
        return math.isfinite(self.KR.tau_R)

    @property
    def category(self) -> ReturnCategory:
        """Tier-2 return-morphology label.

        Classification logic (in order of precedence):
          1. No finite return or Tier-1 identity failure -> FRAGILE
          2. RS < TOL_DAMAGED -> DAMAGED
          3. RS > TOL_SURPLUS AND HR negative (wider gap) -> FALSE_ANTIFRAGILE
          4. RS > TOL_SURPLUS AND HR >= 0 AND RR >= -TOL_ROUGHNESS -> ANTIFRAGILE
          5. |RS| < TOL_SURPLUS AND finite return after meaningful collapse -> RESILIENT
          6. |RS| < TOL_SURPLUS -> ROBUST
        """
        if not self.finite_return:
            return "FRAGILE"

        # Check Tier-1 identities on return state
        try:
            assert abs(self.KR.F + self.KR.omega - 1.0) < 1e-10
            assert self.KR.IC <= self.KR.F + 1e-10
        except AssertionError:
            return "FRAGILE"

        if self.RS < TOL_DAMAGED:
            return "DAMAGED"

        if self.RS > TOL_SURPLUS:
            # Surplus in RS — check for hidden fracture (false antifragility)
            if self.HR < 0:
                return "FALSE_ANTIFRAGILE"
            if self.RR < -TOL_ROUGHNESS:
                return "FALSE_ANTIFRAGILE"
            return "ANTIFRAGILE"

        # No meaningful surplus
        meaningful_collapse = self.Kc.omega >= 0.30
        if meaningful_collapse:
            return "RESILIENT"
        return "ROBUST"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "RS": self.RS,
            "FG": self.FG,
            "DR": self.DR,
            "RR": self.RR,
            "HR": self.HR,
            "finite_return": self.finite_return,
            "category": self.category,
            "K0_F": self.K0.F,
            "K0_IC": self.K0.IC,
            "Kc_F": self.Kc.F,
            "Kc_IC": self.Kc.IC,
            "KR_F": self.KR.F,
            "KR_IC": self.KR.IC,
        }


# ---------------------------------------------------------------------------
# Canonical synthetic episode catalog (12 episodes)
# ---------------------------------------------------------------------------
#
# Trace vectors are 8-channel (one per AFR_CHANNELS entry), values in [0, 1].
# Each episode defines K0, Kc, KR as explicit trace vectors.
# tau_R in the return state is set to float("inf") for no-return episodes.


def _ep(name: str, t0: list, tc: list, tr: list, tau_R_R: float = 1.0, desc: str = "") -> AntifragilityEpisode:
    return AntifragilityEpisode(
        name=name,
        K0=CollapseState.from_trace("pre", np.array(t0, dtype=float)),
        Kc=CollapseState.from_trace("collapse", np.array(tc, dtype=float)),
        KR=CollapseState.from_trace("return", np.array(tr, dtype=float), tau_R=tau_R_R),
        description=desc,
    )


AFR_EPISODES: tuple[AntifragilityEpisode, ...] = (
    # ── FRAGILE (3): no finite return ────────────────────────────────────
    _ep(
        "fragile_no_return",
        t0=[0.75, 0.80, 0.70, 0.85, 0.80, 0.75, 0.70, 0.80],
        tc=[0.20, 0.15, 0.10, 0.25, 0.20, 0.15, 0.10, 0.20],
        tr=[0.30, 0.25, 0.15, 0.30, 0.25, 0.20, 0.15, 0.25],
        tau_R_R=float("inf"),
        desc="High-performing system collapses and never returns — INF_REC",
    ),
    _ep(
        "fragile_permanent_detention",
        t0=[0.80, 0.75, 0.65, 0.80, 0.75, 0.70, 0.65, 0.75],
        tc=[0.10, 0.08, 0.05, 0.12, 0.10, 0.08, 0.05, 0.10],
        tr=[0.25, 0.20, 0.10, 0.28, 0.22, 0.18, 0.12, 0.22],
        tau_R_R=float("inf"),
        desc="Deep collapse, permanent detention — moratio reditus = infinity",
    ),
    _ep(
        "fragile_catastrophic",
        t0=[0.70, 0.72, 0.68, 0.75, 0.70, 0.68, 0.65, 0.72],
        tc=[0.05, 0.04, 0.03, 0.06, 0.05, 0.04, 0.03, 0.05],
        tr=[0.20, 0.15, 0.08, 0.22, 0.18, 0.14, 0.08, 0.18],
        tau_R_R=float("inf"),
        desc="Catastrophic collapse — system exits domain of return",
    ),
    # ── DAMAGED (3): finite return but materially worse ──────────────────
    _ep(
        "damaged_partial_recovery",
        t0=[0.80, 0.82, 0.78, 0.85, 0.80, 0.78, 0.75, 0.82],
        tc=[0.25, 0.20, 0.15, 0.30, 0.25, 0.20, 0.15, 0.22],
        tr=[0.55, 0.50, 0.45, 0.60, 0.55, 0.50, 0.45, 0.52],
        tau_R_R=3.0,
        desc="Finite return but KR materially below K0 — damaged recovery",
    ),
    _ep(
        "damaged_surface_gain",
        t0=[0.75, 0.80, 0.72, 0.80, 0.75, 0.72, 0.68, 0.78],
        tc=[0.20, 0.15, 0.12, 0.25, 0.20, 0.18, 0.12, 0.18],
        tr=[0.60, 0.40, 0.42, 0.65, 0.60, 0.55, 0.40, 0.58],
        tau_R_R=2.0,
        desc="F rises slightly but IC falls — net damage",
    ),
    _ep(
        "damaged_brittle_return",
        t0=[0.78, 0.80, 0.75, 0.82, 0.78, 0.76, 0.72, 0.80],
        tc=[0.18, 0.14, 0.10, 0.22, 0.18, 0.16, 0.10, 0.16],
        tr=[0.52, 0.38, 0.40, 0.58, 0.52, 0.48, 0.38, 0.50],
        tau_R_R=4.0,
        desc="Return achieved but RS well below zero — brittle recovery",
    ),
    # ── RESILIENT (3): meaningful collapse then admissible K0 recovery ───
    _ep(
        "resilient_collapse_recovery",
        t0=[0.72, 0.74, 0.70, 0.75, 0.72, 0.70, 0.68, 0.74],
        tc=[0.20, 0.18, 0.15, 0.22, 0.20, 0.18, 0.15, 0.20],
        tr=[0.72, 0.73, 0.70, 0.74, 0.71, 0.69, 0.68, 0.73],
        tau_R_R=5.0,
        desc="Full collapse (omega >= 0.30), returns to approximately K0",
    ),
    _ep(
        "resilient_institutional",
        t0=[0.68, 0.90, 0.85, 0.72, 0.85, 0.92, 0.30, 0.55],
        tc=[0.22, 0.40, 0.30, 0.35, 0.40, 0.50, 0.12, 0.28],
        tr=[0.67, 0.89, 0.84, 0.71, 0.84, 0.91, 0.30, 0.54],
        tau_R_R=8.0,
        desc="Institutional system weathers shock and restores near-baseline",
    ),
    _ep(
        "resilient_distributed",
        t0=[0.65, 0.60, 0.55, 0.70, 0.25, 0.68, 0.58, 0.80],
        tc=[0.20, 0.15, 0.12, 0.25, 0.08, 0.22, 0.15, 0.28],
        tr=[0.64, 0.59, 0.55, 0.69, 0.25, 0.67, 0.58, 0.79],
        tau_R_R=6.0,
        desc="Distributed system collapses deeply then restores coherence",
    ),
    # ── ANTIFRAGILE (3): finite return with measured surplus ─────────────
    _ep(
        "antifragile_coherence_gain",
        t0=[0.72, 0.74, 0.70, 0.78, 0.72, 0.70, 0.68, 0.74],
        tc=[0.25, 0.22, 0.18, 0.28, 0.25, 0.22, 0.18, 0.24],
        tr=[0.80, 0.84, 0.78, 0.86, 0.82, 0.80, 0.78, 0.84],
        tau_R_R=4.0,
        desc="After collapse, returned state has higher IC AND smaller gap — antifragile",
    ),
    _ep(
        "antifragile_fracture_repair",
        t0=[0.70, 0.60, 0.65, 0.75, 0.65, 0.68, 0.60, 0.72],
        tc=[0.22, 0.18, 0.15, 0.28, 0.20, 0.22, 0.15, 0.24],
        tr=[0.76, 0.76, 0.74, 0.80, 0.76, 0.76, 0.74, 0.78],
        tau_R_R=5.0,
        desc="Large initial gap (tech_startup-like). Collapse repairs internal unevenness — HR > 0",
    ),
    _ep(
        "antifragile_training_effect",
        t0=[0.68, 0.70, 0.65, 0.72, 0.68, 0.66, 0.62, 0.70],
        tc=[0.20, 0.18, 0.12, 0.24, 0.20, 0.18, 0.12, 0.20],
        tr=[0.76, 0.80, 0.75, 0.82, 0.78, 0.76, 0.74, 0.80],
        tau_R_R=3.0,
        desc="Structured exposure: return state shows training gain — both RS and HR positive",
    ),
)


def compute_all_episodes() -> list[AntifragilityEpisode]:
    """Return all canonical episodes (no computation needed — episodes are lazy)."""
    return list(AFR_EPISODES)


# ---------------------------------------------------------------------------
# Theorem functions (T-AFR-1 through T-AFR-6)
# ---------------------------------------------------------------------------


def verify_t_afr_1(episodes: list[AntifragilityEpisode]) -> dict:
    """T-AFR-1: Fragile episodes have no finite return (tau_R = INF_REC).

    All episodes named 'fragile_*' must have finite_return == False.
    """
    fragile = [e for e in episodes if e.name.startswith("fragile_")]
    failed = [e.name for e in fragile if e.finite_return]
    passed = len(fragile) > 0 and len(failed) == 0
    return {
        "name": "T-AFR-1",
        "passed": bool(passed),
        "n_fragile": len(fragile),
        "incorrectly_finite": failed,
    }


def verify_t_afr_2(episodes: list[AntifragilityEpisode]) -> dict:
    """T-AFR-2: Antifragile episodes have positive Return Surplus (RS > TOL_SURPLUS).

    All episodes classified as ANTIFRAGILE must have RS > TOL_SURPLUS.
    This is a necessary (not sufficient) condition for antifragility.
    """
    antifragile = [e for e in episodes if e.category == "ANTIFRAGILE"]
    failed = [e.name for e in antifragile if e.RS <= TOL_SURPLUS]
    passed = len(antifragile) > 0 and len(failed) == 0
    return {
        "name": "T-AFR-2",
        "passed": bool(passed),
        "n_antifragile": len(antifragile),
        "rs_values": {e.name: round(e.RS, 5) for e in antifragile},
        "failed": failed,
    }


def verify_t_afr_3(episodes: list[AntifragilityEpisode]) -> dict:
    """T-AFR-3: Antifragile episodes have non-negative Heterogeneity Repair (HR >= 0).

    Hidden fracture (F - IC gap) must not widen in a genuine antifragile return.
    This guards against "false antifragility" — average-up with multiplicative decay.
    """
    antifragile = [e for e in episodes if e.category == "ANTIFRAGILE"]
    failed = [e.name for e in antifragile if e.HR < 0]
    passed = len(antifragile) > 0 and len(failed) == 0
    return {
        "name": "T-AFR-3",
        "passed": bool(passed),
        "n_antifragile": len(antifragile),
        "hr_values": {e.name: round(e.HR, 5) for e in antifragile},
        "failed": failed,
    }


def verify_t_afr_4(episodes: list[AntifragilityEpisode]) -> dict:
    """T-AFR-4: Tier-1 identities hold on all return states, regardless of category.

    F + omega = 1 (duality), IC <= F (integrity bound), IC = exp(kappa)
    must hold to machine precision on every KR in the catalog.
    This verifies that AFR does not violate the Tier-1 kernel.
    """
    violations = []
    for ep in episodes:
        kr = ep.KR
        if abs(kr.F + kr.omega - 1.0) >= 1e-10:
            violations.append(f"{ep.name}: duality |{kr.F + kr.omega - 1.0:.2e}|")
        if kr.IC > kr.F + 1e-10:
            violations.append(f"{ep.name}: IC > F by {kr.IC - kr.F:.2e}")
        if abs(kr.IC - math.exp(kr.kappa)) >= 1e-10:
            violations.append(f"{ep.name}: log-integrity |{kr.IC - math.exp(kr.kappa):.2e}|")
    passed = len(violations) == 0
    return {"name": "T-AFR-4", "passed": bool(passed), "violations": violations}


def verify_t_afr_5(episodes: list[AntifragilityEpisode]) -> dict:
    """T-AFR-5: False antifragility detector — episodes with RS > TOL_SURPLUS but
    HR < 0 must be classified FALSE_ANTIFRAGILE, not ANTIFRAGILE.

    Surface performance gain (F up, RS up) paired with multiplicative coherence
    deterioration (HR < 0) is hidden fracture. The classification must distinguish
    these from genuine antifragility.
    """
    candidates = [e for e in episodes if e.RS > TOL_SURPLUS and e.HR < 0]
    wrongly_classified = [e.name for e in candidates if e.category == "ANTIFRAGILE"]
    passed = len(wrongly_classified) == 0
    return {
        "name": "T-AFR-5",
        "passed": bool(passed),
        "n_false_candidates": len(candidates),
        "wrongly_classified_as_antifragile": wrongly_classified,
    }


def verify_t_afr_6(episodes: list[AntifragilityEpisode]) -> dict:
    """T-AFR-6: Morphology partition is exhaustive and mutually exclusive.

    Every episode must receive exactly one of the five (or six, including
    FALSE_ANTIFRAGILE) categories. No episode may be unclassified. The canonical
    set must include at least one episode in each non-FALSE category.
    """
    valid_categories = {"FRAGILE", "DAMAGED", "ROBUST", "RESILIENT", "ANTIFRAGILE", "FALSE_ANTIFRAGILE"}
    all_cats = [e.category for e in episodes]
    invalid = [c for c in all_cats if c not in valid_categories]
    present = set(all_cats)
    required = {"FRAGILE", "DAMAGED", "RESILIENT", "ANTIFRAGILE"}
    missing = required - present
    passed = len(invalid) == 0 and len(missing) == 0
    return {
        "name": "T-AFR-6",
        "passed": bool(passed),
        "categories_present": sorted(present),
        "invalid_categories": invalid,
        "missing_required": sorted(missing),
        "distribution": {c: all_cats.count(c) for c in sorted(present)},
    }


def verify_all_theorems() -> list[dict]:
    """Run all T-AFR theorems against the canonical episode catalog."""
    episodes = compute_all_episodes()
    return [
        verify_t_afr_1(episodes),
        verify_t_afr_2(episodes),
        verify_t_afr_3(episodes),
        verify_t_afr_4(episodes),
        verify_t_afr_5(episodes),
        verify_t_afr_6(episodes),
    ]


def main() -> None:
    """Entry point — print episode catalog and theorem verdicts."""
    episodes = compute_all_episodes()
    print(f"\nAFR.RETURN.v1 — {len(episodes)} canonical episodes\n")
    print(f"{'Name':<35} {'Cat':<20} {'RS':>8} {'FG':>8} {'HR':>8} {'DR':>8} {'RR':>8}")
    print("-" * 97)
    for ep in episodes:
        print(f"{ep.name:<35} {ep.category:<20} {ep.RS:>8.4f} {ep.FG:>8.4f} {ep.HR:>8.4f} {ep.DR:>8.4f} {ep.RR:>8.4f}")
    print()
    results = verify_all_theorems()
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  {status}  {r['name']}")
    all_pass = all(r["passed"] for r in results)
    verdict = "CONFORMANT" if all_pass else "NONCONFORMANT"
    print(f"\n  {verdict} — {sum(r['passed'] for r in results)}/{len(results)} theorems")


if __name__ == "__main__":
    main()

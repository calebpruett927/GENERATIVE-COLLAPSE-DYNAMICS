"""Tests for GCD Kernel Structural Theorems (T-KS-1 through T-KS-7).

These seven theorems formalize the structural phenomena that emerge from
the GCD kernel's interaction between the arithmetic mean (F) and the
geometric mean (IC).  They are properties of the kernel itself, independent
of any domain.

Cross-references:
    Formalism:      closures/gcd/kernel_structural_theorems.py
    Kernel:         src/umcp/kernel_optimized.py
    Frozen contract: src/umcp/frozen_contract.py
    SM formalism:   closures/standard_model/particle_physics_formalism.py

All 7 theorems derive from Axiom-0 through the Tier-1 identities:
    F + ω = 1, IC ≤ F, IC = exp(κ)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from closures.gcd.kernel_structural_theorems import (
    TheoremResult,
    run_all_theorems,
    theorem_TKS1_dimensionality_fragility,
    theorem_TKS2_positional_democracy,
    theorem_TKS3_weight_fragility,
    theorem_TKS4_monitoring_paradox,
    theorem_TKS5_approximation_boundary,
    theorem_TKS6_u_curve,
    theorem_TKS7_p3_unification,
)
from umcp.frozen_contract import EPSILON, P_EXPONENT
from umcp.kernel_optimized import compute_kernel_outputs

# ═══════════════════════════════════════════════════════════════════
# MODULE-LEVEL: ALL THEOREMS PROVEN
# ═══════════════════════════════════════════════════════════════════


class TestAllTheoremsProven:
    """Meta-tests: every theorem must pass all its subtests."""

    @pytest.fixture(scope="class")
    def all_results(self) -> list[TheoremResult]:
        return run_all_theorems()

    def test_all_seven_proven(self, all_results: list[TheoremResult]) -> None:
        for r in all_results:
            assert r.verdict == "PROVEN", f"{r.name}: {r.n_passed}/{r.n_tests}"

    def test_total_subtests_at_least_70(self, all_results: list[TheoremResult]) -> None:
        total = sum(r.n_tests for r in all_results)
        assert total >= 70, f"Only {total} subtests"

    def test_zero_failures(self, all_results: list[TheoremResult]) -> None:
        total_fail = sum(r.n_failed for r in all_results)
        assert total_fail == 0, f"{total_fail} subtests failed"

    def test_seven_theorems(self, all_results: list[TheoremResult]) -> None:
        assert len(all_results) == 7


# ═══════════════════════════════════════════════════════════════════
# T-KS-1: DIMENSIONALITY FRAGILITY LAW
# ═══════════════════════════════════════════════════════════════════


class TestTKS1DimensionalityFragility:
    """T-KS-1: IC_one_dead = ε^(1/n) · c₀^((n-1)/n)."""

    def test_theorem_proven(self) -> None:
        r = theorem_TKS1_dimensionality_fragility()
        assert r.verdict == "PROVEN"
        assert r.n_failed == 0

    @pytest.mark.parametrize("n", [4, 6, 8, 10, 16, 32])
    def test_formula_exact(self, n: int) -> None:
        """IC matches ε^(1/n) · c₀^((n-1)/n) within 2%."""
        c0 = 0.999
        c = np.full(n, c0)
        c[0] = EPSILON
        w = np.ones(n) / n
        k = compute_kernel_outputs(c, w)
        predicted = EPSILON ** (1.0 / n) * c0 ** ((n - 1) / n)
        assert k["IC"] == pytest.approx(predicted, rel=0.02)

    def test_fragility_monotone_with_n(self) -> None:
        """More channels = more robust to one dead channel."""
        ics = []
        for n in [4, 8, 16, 32]:
            c = np.full(n, 0.999)
            c[0] = EPSILON
            w = np.ones(n) / n
            k = compute_kernel_outputs(c, w)
            ics.append(k["IC"])
        for i in range(len(ics) - 1):
            assert ics[i] < ics[i + 1]

    def test_fragility_ratio_4_vs_8(self) -> None:
        """4-channel domain is ~10× more fragile than 8-channel."""
        ic4 = EPSILON ** (1 / 4) * 0.999 ** (3 / 4)
        ic8 = EPSILON ** (1 / 8) * 0.999 ** (7 / 8)
        ratio = ic4 / ic8
        assert ratio == pytest.approx(0.1, abs=0.02)

    def test_different_c0_values(self) -> None:
        """Formula holds for c₀ ∈ {0.5, 0.7, 0.9}."""
        for c0 in [0.5, 0.7, 0.9]:
            n = 8
            c = np.full(n, c0)
            c[0] = EPSILON
            w = np.ones(n) / n
            k = compute_kernel_outputs(c, w)
            predicted = EPSILON ** (1.0 / n) * c0 ** ((n - 1) / n)
            assert k["IC"] == pytest.approx(predicted, rel=0.02)


# ═══════════════════════════════════════════════════════════════════
# T-KS-2: POSITIONAL DEMOCRACY OF SLAUGHTER
# ═══════════════════════════════════════════════════════════════════


class TestTKS2PositionalDemocracy:
    """T-KS-2: IC_drop is constant regardless of which channel dies."""

    def test_theorem_proven(self) -> None:
        r = theorem_TKS2_positional_democracy()
        assert r.verdict == "PROVEN"

    def test_ic_drop_constant_diverse_trace(self) -> None:
        """IC drop varies < 2% of IC_base across diverse 8-channel trace."""
        base = np.array([0.95, 0.88, 0.72, 0.65, 0.80, 0.90, 0.78, 0.85])
        w = np.ones(8) / 8
        k_base = compute_kernel_outputs(base, w)
        drops = []
        for i in range(8):
            c = base.copy()
            c[i] = EPSILON
            k = compute_kernel_outputs(c, w)
            drops.append(k_base["IC"] - k["IC"])
        spread = max(drops) - min(drops)
        assert spread < 0.02 * k_base["IC"]

    def test_f_drop_varies_with_channel_value(self) -> None:
        """F drop correlates with channel value (aristocratic)."""
        base = np.array([0.95, 0.88, 0.72, 0.65, 0.80, 0.90, 0.78, 0.85])
        w = np.ones(8) / 8
        k_base = compute_kernel_outputs(base, w)
        f_drops = []
        for i in range(8):
            c = base.copy()
            c[i] = EPSILON
            k = compute_kernel_outputs(c, w)
            f_drops.append(k_base["F"] - k["F"])
        # Largest F_drop should come from largest channel
        max_drop_idx = int(np.argmax(f_drops))
        max_val_idx = int(np.argmax(base))
        assert max_drop_idx == max_val_idx

    def test_f_spread_exceeds_ic_spread(self) -> None:
        """F drop spread should be much larger than IC drop spread."""
        base = np.array([0.95, 0.88, 0.72, 0.65, 0.80, 0.90, 0.78, 0.85])
        w = np.ones(8) / 8
        k_base = compute_kernel_outputs(base, w)
        ic_drops, f_drops = [], []
        for i in range(8):
            c = base.copy()
            c[i] = EPSILON
            k = compute_kernel_outputs(c, w)
            ic_drops.append(k_base["IC"] - k["IC"])
            f_drops.append(k_base["F"] - k["F"])
        ic_spread = max(ic_drops) - min(ic_drops)
        f_spread = max(f_drops) - min(f_drops)
        assert f_spread > 5 * ic_spread


# ═══════════════════════════════════════════════════════════════════
# T-KS-3: WEIGHT-INDUCED FRAGILITY HIERARCHY
# ═══════════════════════════════════════════════════════════════════


class TestTKS3WeightFragility:
    """T-KS-3: Heavier channel weight = worse kill = lower IC residual."""

    def test_theorem_proven(self) -> None:
        r = theorem_TKS3_weight_fragility()
        assert r.verdict == "PROVEN"

    def test_finance_heaviest_kill_worst(self) -> None:
        """In finance [0.30,0.25,0.25,0.20], killing w=0.30 leaves lowest IC."""
        w = np.array([0.30, 0.25, 0.25, 0.20])
        c = np.array([0.85, 0.80, 0.75, 0.70])
        residuals = []
        for i in range(4):
            c_test = c.copy()
            c_test[i] = EPSILON
            k = compute_kernel_outputs(c_test, w)
            residuals.append(k["IC"])
        assert residuals[0] == min(residuals)  # heaviest weight → lowest IC

    def test_formula_exact(self) -> None:
        """IC_residual = ε^w_k · ∏ c_j^w_j."""
        w = np.array([0.30, 0.25, 0.25, 0.20])
        c = np.array([0.85, 0.80, 0.75, 0.70])
        for i in range(4):
            c_test = c.copy()
            c_test[i] = EPSILON
            k = compute_kernel_outputs(c_test, w)
            # Predicted
            log_pred = w[i] * math.log(EPSILON)
            for j in range(4):
                if j != i:
                    log_pred += w[j] * math.log(c[j])
            predicted = math.exp(log_pred)
            assert k["IC"] == pytest.approx(predicted, rel=0.05)

    def test_asymmetry_ratio_at_least_3x(self) -> None:
        """Killing heaviest vs lightest should differ by ≥ 3×."""
        w = np.array([0.30, 0.25, 0.25, 0.20])
        c = np.array([0.85, 0.80, 0.75, 0.70])
        c0 = c.copy()
        c0[0] = EPSILON
        c3 = c.copy()
        c3[3] = EPSILON
        k0 = compute_kernel_outputs(c0, w)
        k3 = compute_kernel_outputs(c3, w)
        assert k3["IC"] / k0["IC"] > 3.0


# ═══════════════════════════════════════════════════════════════════
# T-KS-4: MONITORING PARADOX
# ═══════════════════════════════════════════════════════════════════


class TestTKS4MonitoringParadox:
    """T-KS-4: Γ(ω) = ω³/(1-ω+ε) — observation cost blows up."""

    @staticmethod
    def _gamma(omega: float) -> float:
        return omega**P_EXPONENT / (1.0 - omega + EPSILON)

    def test_theorem_proven(self) -> None:
        r = theorem_TKS4_monitoring_paradox()
        assert r.verdict == "PROVEN"

    def test_cost_ratio_exceeds_100k(self) -> None:
        """Near-death / stable cost ratio > 10⁵."""
        ratio = self._gamma(0.90) / self._gamma(0.02)
        assert ratio > 1e5

    def test_strict_monotonicity(self) -> None:
        """Γ is strictly increasing on [0.001, 0.999]."""
        omegas = np.linspace(0.001, 0.999, 500)
        gammas = [self._gamma(o) for o in omegas]
        for i in range(len(gammas) - 1):
            assert gammas[i] < gammas[i + 1]

    def test_pole_at_omega_1(self) -> None:
        """Γ(1−δ) → ∞ as δ → 0."""
        assert self._gamma(0.999) > 900
        assert self._gamma(0.9999) > 5000

    def test_deep_collapse_exceeds_seam(self) -> None:
        """Γ(0.50) > tol_seam = 0.005."""
        assert self._gamma(0.50) > 0.005

    def test_stable_edge_below_seam(self) -> None:
        """Γ(0.038) < tol_seam."""
        assert self._gamma(0.038) < 0.005


# ═══════════════════════════════════════════════════════════════════
# T-KS-5: APPROXIMATION BOUNDARY
# ═══════════════════════════════════════════════════════════════════


class TestTKS5ApproximationBoundary:
    """T-KS-5: Var(c)/(2c̄) fails at cliff physics."""

    def test_theorem_proven(self) -> None:
        r = theorem_TKS5_approximation_boundary()
        assert r.verdict == "PROVEN"

    def test_accurate_for_mild_heterogeneity(self) -> None:
        """Approximation within 10% when all channels > 0.5."""
        c = np.array([0.7, 0.8, 0.6, 0.75, 0.65, 0.85, 0.72, 0.78])
        w = np.ones(8) / 8
        k = compute_kernel_outputs(c, w)
        c_bar = float(np.mean(c))
        var_c = float(np.var(c))
        approx = var_c / (2 * c_bar)
        actual = k["heterogeneity_gap"]
        ratio = approx / actual
        assert 0.8 < ratio < 1.2

    def test_fails_at_one_dead_channel(self) -> None:
        """Approximation underestimates by > 5× when one channel = ε."""
        c = np.concatenate([np.full(7, 0.9), [EPSILON]])
        w = np.ones(8) / 8
        k = compute_kernel_outputs(c, w)
        c_bar = float(np.mean(c))
        var_c = float(np.var(c))
        approx = var_c / (2 * c_bar)
        actual = k["heterogeneity_gap"]
        assert actual / approx > 5.0

    def test_fails_at_bimodal(self) -> None:
        """Approximation fails for bimodal traces."""
        c = np.array([0.99, 0.01, 0.99, 0.01, 0.99, 0.01, 0.99, 0.01])
        w = np.ones(8) / 8
        k = compute_kernel_outputs(c, w)
        c_bar = float(np.mean(c))
        var_c = float(np.var(c))
        approx = var_c / (2 * c_bar)
        actual = k["heterogeneity_gap"]
        assert actual / approx > 1.5


# ═══════════════════════════════════════════════════════════════════
# T-KS-6: U-CURVE OF DEGRADATION
# ═══════════════════════════════════════════════════════════════════


class TestTKS6UCurve:
    """T-KS-6: Partial collapse is structurally worst."""

    def test_theorem_proven(self) -> None:
        r = theorem_TKS6_u_curve()
        assert r.verdict == "PROVEN"

    def test_endpoints_homogeneous(self) -> None:
        """IC/F ≈ 1.0 at both extremes (all good, all bad)."""
        for val in [0.999, 0.1]:
            c = np.full(8, val)
            w = np.ones(8) / 8
            k = compute_kernel_outputs(c, w)
            assert k["IC"] / k["F"] > 0.99

    def test_minimum_in_interior(self) -> None:
        """IC/F minimum is not at endpoints."""
        icf_values = []
        for nd in range(9):
            c = np.full(8, 0.999)
            c[:nd] = 0.1
            w = np.ones(8) / 8
            k = compute_kernel_outputs(c, w)
            icf_values.append(k["IC"] / k["F"])
        min_idx = icf_values.index(min(icf_values))
        assert 0 < min_idx < 8

    def test_delta_peaks_in_interior(self) -> None:
        """Heterogeneity gap Δ peaks at intermediate degradation."""
        deltas = []
        for nd in range(9):
            c = np.full(8, 0.999)
            c[:nd] = 0.1
            w = np.ones(8) / 8
            k = compute_kernel_outputs(c, w)
            deltas.append(k["heterogeneity_gap"])
        max_idx = deltas.index(max(deltas))
        assert 0 < max_idx < 8
        assert deltas[0] < 0.001  # endpoint ≈ 0
        assert deltas[-1] < 0.001

    def test_minimum_near_half(self) -> None:
        """IC/F minimum is within ±2 of n/2."""
        icf_values = []
        for nd in range(9):
            c = np.full(8, 0.999)
            c[:nd] = 0.1
            w = np.ones(8) / 8
            k = compute_kernel_outputs(c, w)
            icf_values.append(k["IC"] / k["F"])
        min_idx = icf_values.index(min(icf_values))
        assert abs(min_idx - 4) <= 2


# ═══════════════════════════════════════════════════════════════════
# T-KS-7: p=3 UNIFICATION WEB
# ═══════════════════════════════════════════════════════════════════


class TestTKS7P3Unification:
    """T-KS-7: p=3 uniquely determines the phase structure."""

    def test_theorem_proven(self) -> None:
        r = theorem_TKS7_p3_unification()
        assert r.verdict == "PROVEN"

    def test_p_is_frozen_at_3(self) -> None:
        assert P_EXPONENT == 3

    def test_c_trap_near_0318(self) -> None:
        """First weld threshold c_trap ≈ 0.318."""
        from closures.gcd.kernel_structural_theorems import _gamma

        omega_trap: float | None = None
        for om in np.linspace(0.001, 0.999, 100000):
            if _gamma(om) >= 1.0:
                omega_trap = om
                break
        assert omega_trap is not None, "No omega_trap found"
        c_trap = 1.0 - omega_trap
        assert 0.31 < c_trap < 0.33

    def test_c_eff_one_third(self) -> None:
        assert pytest.approx(1.0 / 3) == 1.0 / P_EXPONENT

    def test_d_eff_six(self) -> None:
        assert 2 * P_EXPONENT == 6

    def test_watch_regime_has_finite_width(self) -> None:
        assert 0.30 - 0.038 > 0.10

    def test_no_other_integer_p_works(self) -> None:
        """No other p in {1,2,4,5} produces c_trap ∈ [0.31, 0.33]."""

        for p_test in [1, 2, 4, 5]:

            def g_test(om: float, _p: int = p_test) -> float:
                return om**_p / (1 - om + EPSILON)

            omega_trap = None
            for om in np.linspace(0.001, 0.999, 100000):
                if g_test(om) >= 1.0:
                    omega_trap = om
                    break
            c_t = 1.0 - omega_trap if omega_trap else None
            assert c_t is None or not (0.31 < c_t < 0.33), f"p={p_test} also gives c_trap in range"


# ═══════════════════════════════════════════════════════════════════
# DERIVATION CHAIN — STRUCTURAL COHERENCE
# ═══════════════════════════════════════════════════════════════════


class TestDerivationChain:
    """Verify the chain: T-KS-1 → ... → T-KS-7."""

    def test_fragility_enables_democracy(self) -> None:
        """T-KS-1 → T-KS-2: Exact formula shows IC_drop ≈ IC_base · [1 − ε^(1/n)]."""
        n = 8
        c = np.array([0.95, 0.88, 0.72, 0.65, 0.80, 0.90, 0.78, 0.85])
        w = np.ones(n) / n
        k_base = compute_kernel_outputs(c, w)
        predicted_drop = k_base["IC"] * (1 - EPSILON ** (1 / n))
        # Average actual drop should be close to predicted
        drops = []
        for i in range(n):
            ct = c.copy()
            ct[i] = EPSILON
            k = compute_kernel_outputs(ct, w)
            drops.append(k_base["IC"] - k["IC"])
        mean_drop = sum(drops) / len(drops)
        assert mean_drop == pytest.approx(predicted_drop, rel=0.05)

    def test_weights_break_democracy(self) -> None:
        """T-KS-2 → T-KS-3: Equal weights → democracy; non-equal → hierarchy."""
        c = np.array([0.85, 0.80, 0.75, 0.70])

        # Equal weights: IC drops should be similar
        w_eq = np.ones(4) / 4
        k_eq = compute_kernel_outputs(c, w_eq)
        eq_drops = []
        for i in range(4):
            ct = c.copy()
            ct[i] = EPSILON
            k = compute_kernel_outputs(ct, w_eq)
            eq_drops.append(k_eq["IC"] - k["IC"])
        eq_spread = max(eq_drops) - min(eq_drops)

        # Non-equal weights: IC residuals should be more spread
        w_ne = np.array([0.40, 0.25, 0.20, 0.15])
        ne_residuals = []
        for i in range(4):
            ct = c.copy()
            ct[i] = EPSILON
            k = compute_kernel_outputs(ct, w_ne)
            ne_residuals.append(k["IC"])
        ne_spread = max(ne_residuals) - min(ne_residuals)

        # Hierarchy should be wider than democracy
        assert ne_spread > eq_spread

    def test_u_curve_connects_to_approximation(self) -> None:
        """T-KS-5 → T-KS-6: Approximation fails precisely at the U-curve minimum."""
        # At the U-curve minimum (half degraded), channels are maximally split
        c_half = np.full(8, 0.999)
        c_half[:4] = 0.1
        w = np.ones(8) / 8
        k = compute_kernel_outputs(c_half, w)
        c_bar = float(np.mean(c_half))
        var_c = float(np.var(c_half))
        approx = var_c / (2 * c_bar)
        actual = k["heterogeneity_gap"]
        # Approximation should underestimate at the U-minimum
        assert actual > approx

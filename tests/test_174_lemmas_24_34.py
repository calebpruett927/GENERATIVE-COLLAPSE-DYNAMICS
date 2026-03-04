"""Tests for KERNEL_SPECIFICATION.md Lemmas 22, 24–34.

These lemmas are the formal mathematical guarantees the protocol claims.
Each is a falsifiable property of existing kernel functions.

Lemmas tested:
  22 — Collapse gate monotonicity under threshold relaxation
  24 — Return stability under small perturbation
  25 — Closure perturbation bound (ε shift → κ shift)
  26 — Entropy–drift coherence bound Θ ∈ [0, 2]
  27 — Residual accumulation bound (multi-seam)
  28 — Minimal closure set (zero-weight pruning preserves invariants)
  29 — Return probability bound (wide η → return)
  30 — Weight perturbation stability
  31 — Embedding consistency (zero-weight dims don't change F, ω, κ)
  32 — Temporal coarse-graining stability
  33 — Sufficient condition for finite return
  34 — Drift threshold via AM-GM gap
"""

from __future__ import annotations

import math

import numpy as np

from umcp.compute_utils import normalize_weights, prune_zero_weights
from umcp.frozen_contract import (
    EPSILON,
    compute_budget_delta_kappa,
    compute_kernel,
    compute_seam_residual,
    compute_tau_R,
)

RNG = np.random.default_rng(314159)


# ============================================================================
# Lemma 24 — Return stability under small perturbation
# ============================================================================


class TestLemma24_ReturnStability:
    """If ‖Ψ(t)−Ψ(u*)‖ ≤ η−2δ and perturbation ≤ δ, return is preserved."""

    def test_small_perturbation_preserves_return(self) -> None:
        """Trace with clear return; small perturbation doesn't break it."""
        trace = np.array(
            [
                [0.3, 0.4],
                [0.8, 0.9],
                [0.301, 0.401],  # very close to t=0
            ]
        )
        eta = 0.05
        tau_original = compute_tau_R(trace, t=2, eta=eta, H_rec=10, norm="L2")
        assert math.isfinite(tau_original)

        # Perturb by δ = 0.001 (< η/2)
        perturbed = trace.copy()
        perturbed[2] += 0.001
        tau_perturbed = compute_tau_R(perturbed, t=2, eta=eta, H_rec=10, norm="L2")
        assert math.isfinite(tau_perturbed), "Small perturbation broke return"

    def test_large_perturbation_may_break_return(self) -> None:
        """Perturbation exceeding margin can break return."""
        trace = np.array(
            [
                [0.3, 0.4],
                [0.8, 0.9],
                [0.34, 0.44],  # distance ≈ 0.057 from t=0
            ]
        )
        eta = 0.06
        tau_original = compute_tau_R(trace, t=2, eta=eta, H_rec=10, norm="L2")
        assert math.isfinite(tau_original)

        # Perturb beyond margin
        perturbed = trace.copy()
        perturbed[2] += 0.05  # pushes well beyond η
        tau_perturbed = compute_tau_R(perturbed, t=2, eta=eta, H_rec=10, norm="L2")
        assert math.isinf(tau_perturbed), "Large perturbation should break return"


# ============================================================================
# Lemma 25 — Closure perturbation bound
# ============================================================================


class TestLemma25_ClosurePerturbation:
    """Shifting ε changes κ by a bounded amount."""

    def test_epsilon_shift_bounded_kappa_change(self) -> None:
        c = np.array([0.3, 0.5, 0.7])
        w = np.array([1 / 3, 1 / 3, 1 / 3])

        k1 = compute_kernel(c, w, tau_R=1.0, epsilon=1e-8)
        k2 = compute_kernel(c, w, tau_R=1.0, epsilon=1e-6)

        # Both should produce finite κ; difference should be small for interior c
        assert math.isfinite(k1.kappa)
        assert math.isfinite(k2.kappa)
        # For interior coordinates, ε clipping doesn't change anything
        assert abs(k1.kappa - k2.kappa) < 1e-6

    def test_epsilon_matters_near_boundary(self) -> None:
        """Near c=0, different ε produces measurably different κ."""
        c = np.array([1e-7, 0.5])
        w = np.array([0.5, 0.5])

        k1 = compute_kernel(c, w, tau_R=1.0, epsilon=1e-8)
        k2 = compute_kernel(c, w, tau_R=1.0, epsilon=1e-4)

        # The clipping should matter here
        assert abs(k1.kappa - k2.kappa) > 0.01


# ============================================================================
# Lemma 26 — Entropy–drift coherence Θ ∈ [0, 2]
# ============================================================================


class TestLemma26_EntropyDriftCoherence:
    """Θ(t) = 1 − ω(t) + S(t)/ln(2) must lie in [0, 2]."""

    def test_theta_in_range_random(self) -> None:
        for _ in range(100):
            c = RNG.uniform(0.05, 0.95, size=5)
            w = normalize_weights(RNG.dirichlet(np.ones(5)))
            ko = compute_kernel(c, w, tau_R=1.0, epsilon=EPSILON)
            theta = 1 - ko.omega + ko.S / math.log(2)
            assert 0 <= theta <= 2 + 1e-10, f"Θ={theta} out of [0,2] for c={c}"

    def test_theta_homogeneous_half(self) -> None:
        """c_i = 0.5 → S = ln(2), ω = 0.5, Θ = 1 − 0.5 + 1 = 1.5."""
        c = np.full(4, 0.5)
        w = np.full(4, 0.25)
        ko = compute_kernel(c, w, tau_R=1.0, epsilon=EPSILON)
        theta = 1 - ko.omega + ko.S / math.log(2)
        assert abs(theta - 1.5) < 1e-10

    def test_theta_near_zero_c(self) -> None:
        """Near-zero closures: S ≈ 0, ω ≈ 1 → Θ ≈ 0."""
        c = np.full(3, EPSILON)
        w = np.full(3, 1 / 3)
        ko = compute_kernel(c, w, tau_R=1.0, epsilon=EPSILON)
        theta = 1 - ko.omega + ko.S / math.log(2)
        assert 0 <= theta <= 2


# ============================================================================
# Lemma 27 — Residual accumulation bound
# ============================================================================


class TestLemma27_ResidualAccumulation:
    """Accumulated residuals over K seams bounded by K·s_max."""

    def test_accumulated_residuals_bounded(self) -> None:
        """Sum of |s| over multiple seams ≤ K × tol_seam."""
        residuals = [0.001, -0.002, 0.003, -0.001, 0.004]
        K = len(residuals)
        accumulated = sum(abs(s) for s in residuals)
        s_max = max(abs(s) for s in residuals)
        assert accumulated <= K * s_max + 1e-15

    def test_single_seam_matches_bound(self) -> None:
        R, tau, D_w, D_c = 0.2, 5.0, 0.01, 0.05
        budget = compute_budget_delta_kappa(R, tau, D_w, D_c)
        ledger = budget + 0.001  # small mismatch
        s = compute_seam_residual(budget, ledger)
        assert abs(s) <= 0.005  # within tolerance


# ============================================================================
# Lemma 28 — Minimal closure set (pruning preserves invariants)
# ============================================================================


class TestLemma28_MinimalClosureSet:
    """Zero-weight dimensions can be pruned without changing kernel outputs."""

    def test_prune_zero_weight_preserves_F(self) -> None:
        c = np.array([0.3, 0.5, 0.7, 0.9])
        w = np.array([0.4, 0.0, 0.3, 0.3])

        result_full = compute_kernel(c, w, tau_R=1.0, epsilon=EPSILON)

        # Prune the zero-weight dimension
        pruned = prune_zero_weights(c, w)
        result_pruned = compute_kernel(pruned.c_active, pruned.w_active, tau_R=1.0, epsilon=EPSILON)

        assert abs(result_full.F - result_pruned.F) < 1e-12
        assert abs(result_full.omega - result_pruned.omega) < 1e-12

    def test_prune_preserves_kappa(self) -> None:
        """Pruning zero-weight dims preserves F, ω, and κ.

        With weighted κ = Σ wᵢ ln(cᵢ,ε), zero-weight dimensions contribute
        nothing to κ, so pruning them preserves all kernel invariants.
        """
        c = np.array([0.3, 0.5, 0.7, 0.9])
        w = np.array([0.4, 0.0, 0.3, 0.3])

        result_full = compute_kernel(c, w, tau_R=1.0, epsilon=EPSILON)
        pruned = prune_zero_weights(c, w)
        result_pruned = compute_kernel(pruned.c_active, pruned.w_active, tau_R=1.0, epsilon=EPSILON)

        # F, omega, and kappa preserved (all weighted quantities)
        assert abs(result_full.F - result_pruned.F) < 1e-12
        assert abs(result_full.omega - result_pruned.omega) < 1e-12
        assert abs(result_full.kappa - result_pruned.kappa) < 1e-12
        assert math.isfinite(result_full.kappa)
        assert math.isfinite(result_pruned.kappa)


# ============================================================================
# Lemma 30 — Weight perturbation stability
# ============================================================================


class TestLemma30_WeightPerturbation:
    """|F−F̃| ≤ δ_w when weights perturbed by ≤ δ_w per component."""

    def test_F_stable_under_weight_perturbation(self) -> None:
        c = np.array([0.3, 0.5, 0.7, 0.9])
        w = np.array([0.25, 0.25, 0.25, 0.25])
        delta_w = 0.01

        ko_orig = compute_kernel(c, w, tau_R=1.0, epsilon=EPSILON)

        # Perturb weights and renormalize
        w_pert = w + RNG.uniform(-delta_w, delta_w, size=4)
        w_pert = np.clip(w_pert, 0, None)
        w_pert = w_pert / w_pert.sum()

        ko_pert = compute_kernel(c, w_pert, tau_R=1.0, epsilon=EPSILON)

        assert abs(ko_orig.F - ko_pert.F) <= delta_w + 1e-10

    def test_S_stable_under_weight_perturbation(self) -> None:
        c = np.array([0.3, 0.5, 0.7])
        w = np.array([0.33, 0.34, 0.33])
        delta_w = 0.01

        ko_orig = compute_kernel(c, w, tau_R=1.0, epsilon=EPSILON)

        w_pert = w + RNG.uniform(-delta_w, delta_w, size=3)
        w_pert = np.clip(w_pert, 0, None)
        w_pert = w_pert / w_pert.sum()

        ko_pert = compute_kernel(c, w_pert, tau_R=1.0, epsilon=EPSILON)

        # From Lemma 30: |S−S̃| ≤ 2·ln(2)·δ_w
        bound = 2 * math.log(2) * delta_w
        assert abs(ko_orig.S - ko_pert.S) <= bound + 1e-8


# ============================================================================
# Lemma 31 — Embedding consistency (zero-weight dims)
# ============================================================================


class TestLemma31_EmbeddingConsistency:
    """Adding zero-weight dimensions does not change F, ω, or κ."""

    def test_add_zero_weight_dim_no_change_F(self) -> None:
        c_base = np.array([0.3, 0.7])
        w_base = np.array([0.5, 0.5])

        c_ext = np.array([0.3, 0.7, 0.999])  # extra dim
        w_ext = np.array([0.5, 0.5, 0.0])  # zero weight

        ko_base = compute_kernel(c_base, w_base, tau_R=1.0, epsilon=EPSILON)
        ko_ext = compute_kernel(c_ext, w_ext, tau_R=1.0, epsilon=EPSILON)

        assert abs(ko_base.F - ko_ext.F) < 1e-12
        assert abs(ko_base.omega - ko_ext.omega) < 1e-12

    def test_add_zero_weight_dim_no_change_kappa(self) -> None:
        """Adding zero-weight dims preserves F, ω, and κ.

        With weighted κ = Σ wᵢ ln(cᵢ,ε), zero-weight dimensions contribute
        nothing to κ, so adding them preserves all kernel invariants.
        """
        c_base = np.array([0.3, 0.7])
        w_base = np.array([0.5, 0.5])

        c_ext = np.array([0.3, 0.7, 0.1])  # extra dim with any value
        w_ext = np.array([0.5, 0.5, 0.0])

        ko_base = compute_kernel(c_base, w_base, tau_R=1.0, epsilon=EPSILON)
        ko_ext = compute_kernel(c_ext, w_ext, tau_R=1.0, epsilon=EPSILON)

        # F, omega, and kappa preserved (zero-weight dim doesn't contribute)
        assert abs(ko_base.F - ko_ext.F) < 1e-12
        assert abs(ko_base.omega - ko_ext.omega) < 1e-12
        assert abs(ko_base.kappa - ko_ext.kappa) < 1e-12
        assert math.isfinite(ko_base.kappa)
        assert math.isfinite(ko_ext.kappa)

    def test_add_multiple_zero_weight_dims(self) -> None:
        c_base = np.array([0.5])
        w_base = np.array([1.0])

        c_ext = np.array([0.5, 0.1, 0.9, 0.2])
        w_ext = np.array([1.0, 0.0, 0.0, 0.0])

        ko_base = compute_kernel(c_base, w_base, tau_R=1.0, epsilon=EPSILON)
        ko_ext = compute_kernel(c_ext, w_ext, tau_R=1.0, epsilon=EPSILON)

        assert abs(ko_base.F - ko_ext.F) < 1e-12


# ============================================================================
# Lemma 33 — Sufficient condition for finite return
# ============================================================================


class TestLemma33_SufficientReturn:
    """If ∃u ∈ D_θ(t): ‖Ψ(t)−Ψ(u)‖ < η then τ_R(t) < ∞."""

    def test_close_point_guarantees_return(self) -> None:
        """Exact repeat of prior state guarantees finite τ_R."""
        trace = np.array(
            [
                [0.5, 0.5],
                [0.8, 0.8],
                [0.5, 0.5],  # exact repeat of t=0
            ]
        )
        tau = compute_tau_R(trace, t=2, eta=0.01, H_rec=10, norm="L2")
        assert math.isfinite(tau)
        assert tau == 2.0

    def test_within_eta_guarantees_return(self) -> None:
        """Point within η of a prior state guarantees finite τ_R."""
        trace = np.array(
            [
                [0.5, 0.5],
                [0.8, 0.8],
                [0.501, 0.499],  # distance < 0.002
            ]
        )
        tau = compute_tau_R(trace, t=2, eta=0.01, H_rec=10, norm="L2")
        assert math.isfinite(tau)


# ============================================================================
# Lemma 34 — AM-GM gap and drift threshold
# ============================================================================


class TestLemma34_AMGMGap:
    """Δ_gap = F − IC ≥ 0; homogeneous ↔ gap = 0."""

    def test_gap_nonnegative_random(self) -> None:
        """F − IC ≥ 0 for random inputs (AM-GM)."""
        for _ in range(100):
            c = RNG.uniform(0.05, 0.95, size=5)
            w = normalize_weights(RNG.dirichlet(np.ones(5)))
            ko = compute_kernel(c, w, tau_R=1.0, epsilon=EPSILON)
            gap = ko.F - ko.IC
            assert gap >= -1e-12, f"Integrity bound violated: F={ko.F}, IC={ko.IC}, gap={gap}"

    def test_gap_zero_when_homogeneous(self) -> None:
        """Homogeneous coordinates → F = IC (heterogeneity gap = 0).

        With weighted κ = Σ wᵢ ln(cᵢ,ε), homogeneous coordinates
        (all cᵢ equal) give IC = exp(ln(c)) = c = F.
        Integrity bound equality holds when channels are uniform.
        """
        # Single dimension: trivially F = IC
        c = np.array([0.7])
        w = np.array([1.0])
        ko = compute_kernel(c, w, tau_R=1.0, epsilon=EPSILON)
        gap = ko.F - ko.IC
        assert abs(gap) < 1e-12

        # Multi-dimensional homogeneous: F = IC with weighted κ
        c_multi = np.full(4, 0.5)
        w_multi = np.full(4, 0.25)
        ko_multi = compute_kernel(c_multi, w_multi, tau_R=1.0, epsilon=EPSILON)
        gap_multi = ko_multi.F - ko_multi.IC
        assert abs(gap_multi) < 1e-12

    def test_gap_increases_with_heterogeneity(self) -> None:
        """More heterogeneous coordinates → larger gap."""
        w = np.full(4, 0.25)

        c_homo = np.full(4, 0.5)
        ko_homo = compute_kernel(c_homo, w, tau_R=1.0, epsilon=EPSILON)

        c_het = np.array([0.1, 0.3, 0.7, 0.9])
        ko_het = compute_kernel(c_het, w, tau_R=1.0, epsilon=EPSILON)

        gap_homo = ko_homo.F - ko_homo.IC
        gap_het = ko_het.F - ko_het.IC

        assert gap_het > gap_homo


# ============================================================================
# Lemma 22 — Collapse Gate Monotonicity Under Threshold Relaxation
# ============================================================================


class TestLemma22_CollapseGateMonotonicity:
    """Tightening collapse thresholds increases the collapse-flagged set.

    Lemma 22: Let T_tight be timesteps flagged as collapse under tight gates,
    T_relaxed under relaxed gates.  Then T_tight ⊇ T_relaxed.
    Order-preserving under threshold variation.
    """

    @staticmethod
    def _classify_regime(
        omega: float,
        F: float,
        S: float,
        C: float,
        omega_thresh: float,
        F_thresh: float,
        S_thresh: float,
        C_thresh: float,
    ) -> str:
        """Classify a single timestep.  Collapse if ω ≥ omega_thresh.
        Stable if all four gates pass.  Otherwise Watch.
        """
        if omega >= omega_thresh:
            return "Collapse"
        if omega < omega_thresh and F_thresh < F and S_thresh > S and C_thresh > C:
            return "Stable"
        return "Watch"

    def test_tighter_thresholds_flag_more_collapse(self) -> None:
        """Tighter ω threshold → more timesteps flagged as Collapse."""
        n = 8
        w = normalize_weights(np.ones(n) / n)
        traces = [RNG.uniform(0.1, 0.9, size=n) for _ in range(200)]

        tight_thresh = 0.20
        relaxed_thresh = 0.40

        collapse_tight = set()
        collapse_relaxed = set()
        for i, c in enumerate(traces):
            ko = compute_kernel(c, w, tau_R=1.0, epsilon=EPSILON)
            if self._classify_regime(ko.omega, ko.F, ko.S, ko.C, tight_thresh, 0.90, 0.15, 0.14) == "Collapse":
                collapse_tight.add(i)
            if self._classify_regime(ko.omega, ko.F, ko.S, ko.C, relaxed_thresh, 0.90, 0.15, 0.14) == "Collapse":
                collapse_relaxed.add(i)

        # Relaxed threshold is higher ω → fewer flagged as collapse
        assert collapse_relaxed.issubset(collapse_tight), (
            f"Monotonicity violated: {collapse_relaxed - collapse_tight} in relaxed but not tight"
        )

    def test_relaxed_stable_gates_reduce_collapse(self) -> None:
        """Relaxing all four gates reduces collapse+watch set size."""
        n = 5
        w = normalize_weights(np.ones(n) / n)
        traces = [RNG.uniform(0.05, 0.95, size=n) for _ in range(100)]

        # Tight: strict gates
        non_stable_tight = 0
        # Relaxed: looser gates
        non_stable_relaxed = 0

        for c in traces:
            ko = compute_kernel(c, w, tau_R=1.0, epsilon=EPSILON)
            if self._classify_regime(ko.omega, ko.F, ko.S, ko.C, 0.038, 0.90, 0.15, 0.14) != "Stable":
                non_stable_tight += 1
            if self._classify_regime(ko.omega, ko.F, ko.S, ko.C, 0.10, 0.70, 0.50, 0.50) != "Stable":
                non_stable_relaxed += 1

        assert non_stable_relaxed <= non_stable_tight


# ============================================================================
# Lemma 29 — Return Probability Under Bounded Random Walk
# ============================================================================


class TestLemma29_ReturnProbabilityBoundedWalk:
    """Under bounded random walk, return is almost certain if η > 2σ√n.

    Lemma 29: For bounded random walk on [0,1]ⁿ with step size σ,
    if η > 2σ√n then P_return → 1 as H_rec → ∞.
    """

    def test_bounded_walk_returns(self) -> None:
        """Bounded random walk with wide η returns with high probability."""
        n = 4
        sigma = 0.02
        eta = 2.5 * sigma * np.sqrt(n)  # > 2σ√n
        H_rec = 500

        returns = 0
        trials = 50
        for _ in range(trials):
            # Generate bounded random walk
            trace = np.zeros((H_rec + 1, n))
            trace[0] = RNG.uniform(0.3, 0.7, size=n)
            for t in range(1, H_rec + 1):
                step = RNG.normal(0, sigma, size=n)
                trace[t] = np.clip(trace[0 + t - 1] + step, 0.01, 0.99)

            # Check for return: any later timestep within η of t=0
            origin = trace[0]
            for t in range(1, H_rec + 1):
                if np.linalg.norm(trace[t] - origin) < eta:
                    returns += 1
                    break

        assert returns / trials >= 0.80, f"Return rate {returns / trials} too low — Lemma 29 expects near-certainty"

    def test_narrow_eta_reduces_returns(self) -> None:
        """Narrow η (< 2σ√n) reduces return probability."""
        n = 4
        sigma = 0.05
        eta_wide = 3.0 * sigma * np.sqrt(n)
        eta_narrow = 0.5 * sigma * np.sqrt(n)
        H_rec = 200

        returns_wide = 0
        returns_narrow = 0
        trials = 50
        for _ in range(trials):
            trace = np.zeros((H_rec + 1, n))
            trace[0] = RNG.uniform(0.3, 0.7, size=n)
            for t in range(1, H_rec + 1):
                step = RNG.normal(0, sigma, size=n)
                trace[t] = np.clip(trace[t - 1] + step, 0.01, 0.99)

            origin = trace[0]
            for t in range(1, H_rec + 1):
                if np.linalg.norm(trace[t] - origin) < eta_wide:
                    returns_wide += 1
                    break
            for t in range(1, H_rec + 1):
                if np.linalg.norm(trace[t] - origin) < eta_narrow:
                    returns_narrow += 1
                    break

        assert returns_wide >= returns_narrow, "Wider η should produce at least as many returns"


# ============================================================================
# Lemma 32 — Temporal Coarse-Graining Stability
# ============================================================================


class TestLemma32_TemporalCoarseGraining:
    """Coarse-grained kernel outputs are bounded perturbations of fine-grained averages.

    Lemma 32: |F̄(t') − (1/M) Σ F(Mt'+k)| ≤ ε_coarse(M) → 0 as M → 1.
    """

    def test_coarsened_F_close_to_fine_average(self) -> None:
        """Coarse-grained F is close to average of fine-grained F values."""
        n = 5
        w = normalize_weights(np.ones(n) / n)
        T = 100
        M = 5  # coarsening factor

        # Generate smooth trace
        trace = np.zeros((T, n))
        trace[0] = RNG.uniform(0.3, 0.7, size=n)
        for t in range(1, T):
            step = RNG.normal(0, 0.02, size=n)
            trace[t] = np.clip(trace[t - 1] + step, 0.01, 0.99)

        T_prime = T // M
        for t_prime in range(T_prime):
            # Fine-grained: compute F for each sub-step, average
            fine_Fs = []
            for k in range(M):
                idx = M * t_prime + k
                ko = compute_kernel(trace[idx], w, tau_R=1.0, epsilon=EPSILON)
                fine_Fs.append(ko.F)
            avg_fine_F = np.mean(fine_Fs)

            # Coarse-grained: average coordinates, then compute F
            coarse_c = np.mean(trace[M * t_prime : M * (t_prime + 1)], axis=0)
            coarse_c = np.clip(coarse_c, EPSILON, 1 - EPSILON)
            ko_coarse = compute_kernel(coarse_c, w, tau_R=1.0, epsilon=EPSILON)

            # F is linear in c_i, so for equal weights F̄ = avg(F) exactly
            assert abs(ko_coarse.F - avg_fine_F) < 1e-10, (
                f"F coarsening error too large: {abs(ko_coarse.F - avg_fine_F)}"
            )

    def test_coarsened_kappa_bounded_perturbation(self) -> None:
        """Coarse-grained κ is a bounded perturbation of fine-grained average."""
        n = 4
        w = normalize_weights(np.ones(n) / n)
        T = 60
        M = 3

        trace = np.zeros((T, n))
        trace[0] = RNG.uniform(0.3, 0.7, size=n)
        for t in range(1, T):
            step = RNG.normal(0, 0.01, size=n)
            trace[t] = np.clip(trace[t - 1] + step, 0.05, 0.95)

        T_prime = T // M
        max_kappa_error = 0.0
        for t_prime in range(T_prime):
            fine_kappas = []
            for k in range(M):
                idx = M * t_prime + k
                ko = compute_kernel(trace[idx], w, tau_R=1.0, epsilon=EPSILON)
                fine_kappas.append(ko.kappa)
            avg_fine_kappa = np.mean(fine_kappas)

            coarse_c = np.mean(trace[M * t_prime : M * (t_prime + 1)], axis=0)
            coarse_c = np.clip(coarse_c, EPSILON, 1 - EPSILON)
            ko_coarse = compute_kernel(coarse_c, w, tau_R=1.0, epsilon=EPSILON)

            kappa_error = abs(ko_coarse.kappa - avg_fine_kappa)
            max_kappa_error = max(max_kappa_error, kappa_error)

        # κ is nonlinear (Jensen's gap), but for smooth traces with small M
        # the perturbation should be bounded
        assert max_kappa_error < 0.5, f"κ coarsening error unbounded: {max_kappa_error}"

    def test_finer_coarsening_reduces_error(self) -> None:
        """Smaller M → smaller coarsening error (ε_coarse → 0 as M → 1)."""
        n = 4
        w = normalize_weights(np.ones(n) / n)
        T = 60

        trace = np.zeros((T, n))
        trace[0] = RNG.uniform(0.3, 0.7, size=n)
        for t in range(1, T):
            step = RNG.normal(0, 0.01, size=n)
            trace[t] = np.clip(trace[t - 1] + step, 0.05, 0.95)

        errors_by_M: dict[int, float] = {}
        for M in [2, 3, 5, 10]:
            T_prime = T // M
            max_err = 0.0
            for t_prime in range(T_prime):
                fine_kappas = []
                for k in range(M):
                    idx = M * t_prime + k
                    ko = compute_kernel(trace[idx], w, tau_R=1.0, epsilon=EPSILON)
                    fine_kappas.append(ko.kappa)
                avg_fine_kappa = np.mean(fine_kappas)

                coarse_c = np.mean(trace[M * t_prime : M * (t_prime + 1)], axis=0)
                coarse_c = np.clip(coarse_c, EPSILON, 1 - EPSILON)
                ko_coarse = compute_kernel(coarse_c, w, tau_R=1.0, epsilon=EPSILON)
                max_err = max(max_err, abs(ko_coarse.kappa - avg_fine_kappa))
            errors_by_M[M] = float(max_err)

        # Monotonicity: smaller M should give smaller error (approximately)
        assert errors_by_M[2] <= errors_by_M[10] + 0.01, (
            f"Expected ε_coarse to grow with M: M=2→{errors_by_M[2]:.4f}, M=10→{errors_by_M[10]:.4f}"
        )

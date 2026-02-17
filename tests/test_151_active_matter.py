"""Tests for Active Matter Frictional Cooling closure module.

Validates the corrected UMCP analysis of Antonov et al. (2025)
vibrated macroscopic robots, ensuring:

  1. Embedding: 4D Ψ(t) from particle velocity data
  2. Kernel: Canonical IC = exp(κ), not redundant IC formula
  3. Regimes: ω-primary cascade, correct for each phase
  4. Fisher distances: geodesic metric between phases
  5. Grammar: ORDERED at all β values
  6. Budget: Γ(ω) drift cost + D_C curvature cost
  7. Full audit pipeline integrity

Cross-references:
  closures/rcft/active_matter.py
  tests/test_149_rcft_universality.py  (RCFT primitives)
  KERNEL_SPECIFICATION.md §5
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from closures.rcft.active_matter import (
    ActiveMatterAudit,
    ActiveMatterConfig,
    PhaseResult,
    activity_to_beta,
    analyze_phase,
    cluster_stability,
    compute_invariants,
    embed_particle_velocities,
    frictional_cooling_rate,
    generate_synthetic_velocities,
    run_audit,
)

# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture(scope="module")
def config() -> ActiveMatterConfig:
    """Default Antonov et al. configuration."""
    return ActiveMatterConfig()


@pytest.fixture(scope="module")
def synthetic_data() -> tuple[np.ndarray, list[tuple[str, int, int]]]:
    """Generate synthetic velocity data with default parameters."""
    return generate_synthetic_velocities(seed=42)


@pytest.fixture(scope="module")
def psi_trace(
    synthetic_data: tuple[np.ndarray, list[tuple[str, int, int]]],
    config: ActiveMatterConfig,
) -> np.ndarray:
    """Embed synthetic data into 4D Ψ-trace."""
    velocities, _ = synthetic_data
    return embed_particle_velocities(velocities, config=config)


@pytest.fixture(scope="module")
def phase_invariants(
    psi_trace: np.ndarray,
    synthetic_data: tuple[np.ndarray, list[tuple[str, int, int]]],
) -> dict[str, list[dict]]:
    """Compute invariants for each phase."""
    _, boundaries = synthetic_data
    w = np.ones(4) / 4
    all_inv = compute_invariants(psi_trace, weights=w)
    return {name: all_inv[start:end] for name, start, end in boundaries}


# =====================================================================
# Config Tests
# =====================================================================


class TestActiveMatterConfig:
    """Verify experimental configuration defaults."""

    def test_particle_count(self, config: ActiveMatterConfig) -> None:
        assert config.n_particles == 180

    def test_dimensions(self, config: ActiveMatterConfig) -> None:
        assert config.n_dims == 4

    def test_weights_sum_to_one(self, config: ActiveMatterConfig) -> None:
        assert abs(sum(config.weights) - 1.0) < 1e-12

    def test_packing_fraction(self, config: ActiveMatterConfig) -> None:
        assert config.packing_fraction == 0.45

    def test_arrest_threshold_positive(self, config: ActiveMatterConfig) -> None:
        assert config.arrest_threshold > 0


# =====================================================================
# Embedding Tests
# =====================================================================


class TestEmbedding:
    """Verify 4D Ψ-embedding from particle velocities."""

    def test_output_shape(self, psi_trace: np.ndarray) -> None:
        assert psi_trace.ndim == 2
        assert psi_trace.shape[1] == 4

    def test_timestep_count(
        self,
        psi_trace: np.ndarray,
        synthetic_data: tuple[np.ndarray, list[tuple[str, int, int]]],
    ) -> None:
        velocities, _ = synthetic_data
        assert psi_trace.shape[0] == velocities.shape[0]

    def test_values_in_unit_interval(self, psi_trace: np.ndarray) -> None:
        assert np.all(psi_trace > 0)
        assert np.all(psi_trace < 1)

    def test_cooled_has_high_fidelity(
        self,
        psi_trace: np.ndarray,
        synthetic_data: tuple[np.ndarray, list[tuple[str, int, int]]],
    ) -> None:
        """Cooled phase: near-zero speeds → c₁ ≈ 1."""
        _, boundaries = synthetic_data
        start, end = boundaries[0][1], boundaries[0][2]
        c1_mean = np.mean(psi_trace[start:end, 0])
        assert c1_mean > 0.8, f"Cooled c₁ = {c1_mean}, expected > 0.8"

    def test_heated_has_lower_fidelity_than_cooled(
        self,
        psi_trace: np.ndarray,
        synthetic_data: tuple[np.ndarray, list[tuple[str, int, int]]],
    ) -> None:
        """Heated phase: higher speeds → lower c₁ than cooled."""
        _, boundaries = synthetic_data
        cooled_start, cooled_end = boundaries[0][1], boundaries[0][2]
        heated_start, heated_end = boundaries[-1][1], boundaries[-1][2]
        c1_cooled = np.mean(psi_trace[cooled_start:cooled_end, 0])
        c1_heated = np.mean(psi_trace[heated_start:heated_end, 0])
        assert c1_heated < c1_cooled, f"Heated c₁ = {c1_heated} should be < Cooled c₁ = {c1_cooled}"

    def test_cooled_arrest_fraction_high(
        self,
        psi_trace: np.ndarray,
        synthetic_data: tuple[np.ndarray, list[tuple[str, int, int]]],
    ) -> None:
        """Cooled phase: most particles below arrest threshold."""
        _, boundaries = synthetic_data
        start, end = boundaries[0][1], boundaries[0][2]
        c3_mean = np.mean(psi_trace[start:end, 2])
        assert c3_mean > 0.5, f"Cooled arrest fraction = {c3_mean}, expected > 0.5"

    def test_heated_arrest_fraction_low(
        self,
        psi_trace: np.ndarray,
        synthetic_data: tuple[np.ndarray, list[tuple[str, int, int]]],
    ) -> None:
        """Heated phase: few particles arrested."""
        _, boundaries = synthetic_data
        start, end = boundaries[-1][1], boundaries[-1][2]
        c3_mean = np.mean(psi_trace[start:end, 2])
        assert c3_mean < 0.2, f"Heated arrest fraction = {c3_mean}, expected < 0.2"


# =====================================================================
# Kernel Invariant Tests
# =====================================================================


class TestKernelInvariants:
    """Verify Tier-1 kernel invariants at each timestep."""

    def test_F_equals_1_minus_omega(self, phase_invariants: dict[str, list[dict]]) -> None:
        """F = 1 − ω (fundamental identity)."""
        for phase, invs in phase_invariants.items():
            for r in invs:
                assert abs(r["F"] - (1 - r["omega"])) < 1e-10, f"{phase} t={r['t']}: F={r['F']}, 1-ω={1 - r['omega']}"

    def test_IC_le_F(self, phase_invariants: dict[str, list[dict]]) -> None:
        """IC ≤ F (AM-GM inequality)."""
        for phase, invs in phase_invariants.items():
            for r in invs:
                assert r["IC"] <= r["F"] + 1e-10, f"{phase} t={r['t']}: IC={r['IC']} > F={r['F']}"

    def test_IC_approx_exp_kappa(self, phase_invariants: dict[str, list[dict]]) -> None:
        """IC ≈ exp(κ) — the CANONICAL formula."""
        for phase, invs in phase_invariants.items():
            for r in invs:
                expected = math.exp(r["kappa"])
                assert abs(r["IC"] - expected) < 1e-8, f"{phase} t={r['t']}: IC={r['IC']}, exp(κ)={expected}"

    def test_omega_in_unit_interval(self, phase_invariants: dict[str, list[dict]]) -> None:
        for _phase, invs in phase_invariants.items():
            for r in invs:
                assert 0 <= r["omega"] <= 1

    def test_entropy_non_negative(self, phase_invariants: dict[str, list[dict]]) -> None:
        for _phase, invs in phase_invariants.items():
            for r in invs:
                assert r["S"] >= 0

    def test_curvature_non_negative(self, phase_invariants: dict[str, list[dict]]) -> None:
        for _phase, invs in phase_invariants.items():
            for r in invs:
                assert r["C"] >= 0


# =====================================================================
# Regime Classification Tests
# =====================================================================


class TestRegimeClassification:
    """Verify ω-primary cascade regime classification."""

    def test_cooled_is_watch(self, phase_invariants: dict[str, list[dict]]) -> None:
        """Cooled phase: ω ≈ 0.13 → WATCH (not STABLE as paper claims)."""
        regimes = [r["regime"] for r in phase_invariants["Cooled"]]
        from collections import Counter

        dominant = Counter(regimes).most_common(1)[0][0]
        assert dominant == "WATCH", f"Cooled dominant regime = {dominant}, expected WATCH"

    def test_mixed_is_watch_or_higher(self, phase_invariants: dict[str, list[dict]]) -> None:
        """Mixed phase: bimodal velocities → WATCH or higher (intermediate ω)."""
        regimes = [r["regime"] for r in phase_invariants["Mixed"]]
        from collections import Counter

        counts = Counter(regimes)
        # Mixed phase has intermediate ω → predominantly WATCH (or CRITICAL/COLLAPSE)
        non_stable = counts.get("WATCH", 0) + counts.get("CRITICAL", 0) + counts.get("COLLAPSE", 0)
        assert non_stable > len(regimes) * 0.5, f"Expected mostly non-STABLE, got {counts}"

    def test_heated_is_critical(self, phase_invariants: dict[str, list[dict]]) -> None:
        """Heated phase: high ω → CRITICAL."""
        regimes = [r["regime"] for r in phase_invariants["Heated"]]
        from collections import Counter

        counts = Counter(regimes)
        assert counts.get("CRITICAL", 0) + counts.get("COLLAPSE", 0) > len(regimes) * 0.5

    def test_omega_increases_cooled_to_heated(self, phase_invariants: dict[str, list[dict]]) -> None:
        """ω increases monotonically: cooled < mixed < heated."""
        omega_cooled = np.mean([r["omega"] for r in phase_invariants["Cooled"]])
        omega_mixed = np.mean([r["omega"] for r in phase_invariants["Mixed"]])
        omega_heated = np.mean([r["omega"] for r in phase_invariants["Heated"]])
        assert omega_cooled < omega_mixed < omega_heated


# =====================================================================
# Phase Analysis Tests
# =====================================================================


class TestPhaseAnalysis:
    """Verify phase-level summary statistics."""

    def test_analyze_returns_phase_result(self, phase_invariants: dict[str, list[dict]]) -> None:
        result = analyze_phase("Cooled", phase_invariants["Cooled"])
        assert isinstance(result, PhaseResult)

    def test_phase_name_preserved(self, phase_invariants: dict[str, list[dict]]) -> None:
        result = analyze_phase("Heated", phase_invariants["Heated"])
        assert result.name == "Heated"

    def test_timestep_count(self, phase_invariants: dict[str, list[dict]]) -> None:
        result = analyze_phase("Cooled", phase_invariants["Cooled"])
        assert result.n_timesteps == len(phase_invariants["Cooled"])

    def test_IC_mean_canonical(self, phase_invariants: dict[str, list[dict]]) -> None:
        """IC mean matches exp(κ) mean."""
        result = analyze_phase("Cooled", phase_invariants["Cooled"])
        expected = np.mean([math.exp(r["kappa"]) for r in phase_invariants["Cooled"]])
        assert abs(result.IC_mean - expected) < 1e-6

    def test_equator_deviation_non_negative_for_low_omega(self, phase_invariants: dict[str, list[dict]]) -> None:
        """Low-ω phases should be above equator (Φ_eq > 0)."""
        result = analyze_phase("Cooled", phase_invariants["Cooled"])
        assert result.phi_eq_mean > 0

    def test_drift_cost_increases(self, phase_invariants: dict[str, list[dict]]) -> None:
        """Γ(ω) increases from Cooled to Heated."""
        cooled = analyze_phase("Cooled", phase_invariants["Cooled"])
        heated = analyze_phase("Heated", phase_invariants["Heated"])
        assert heated.drift_cost_mean > cooled.drift_cost_mean


# =====================================================================
# Frictional Cooling Rate Tests
# =====================================================================


class TestFrictionalCooling:
    """Verify dissipation rate computation."""

    def test_output_shape(
        self,
        synthetic_data: tuple[np.ndarray, list[tuple[str, int, int]]],
    ) -> None:
        velocities, _ = synthetic_data
        rate = frictional_cooling_rate(velocities)
        assert rate.shape == (velocities.shape[0],)

    def test_cooling_rate_non_negative(
        self,
        synthetic_data: tuple[np.ndarray, list[tuple[str, int, int]]],
    ) -> None:
        velocities, _ = synthetic_data
        rate = frictional_cooling_rate(velocities)
        assert np.all(rate >= 0)

    def test_uniform_velocities_zero_dissipation(self) -> None:
        """Uniform speeds → zero dissipation."""
        v = np.ones((10, 50)) * 0.5
        rate = frictional_cooling_rate(v)
        np.testing.assert_allclose(rate, 0.0, atol=1e-10)

    def test_heated_higher_dissipation_than_cooled(
        self,
        synthetic_data: tuple[np.ndarray, list[tuple[str, int, int]]],
    ) -> None:
        velocities, boundaries = synthetic_data
        rate = frictional_cooling_rate(velocities)
        cooled_start, cooled_end = boundaries[0][1], boundaries[0][2]
        heated_start, heated_end = boundaries[-1][1], boundaries[-1][2]
        assert np.mean(rate[heated_start:heated_end]) > np.mean(rate[cooled_start:cooled_end])


# =====================================================================
# Cluster Stability Tests
# =====================================================================


class TestClusterStability:
    """Verify corrected cluster stability (= canonical IC)."""

    def test_returns_IC_values(self, phase_invariants: dict[str, list[dict]]) -> None:
        ic_vals = cluster_stability(phase_invariants["Cooled"])
        expected = [r["IC"] for r in phase_invariants["Cooled"]]
        assert ic_vals == expected

    def test_cooled_higher_stability(self, phase_invariants: dict[str, list[dict]]) -> None:
        """Cooled → higher IC (more stable) than heated."""
        ic_cooled = np.mean(cluster_stability(phase_invariants["Cooled"]))
        ic_heated = np.mean(cluster_stability(phase_invariants["Heated"]))
        assert ic_cooled > ic_heated


# =====================================================================
# Activity-to-Beta Mapping Tests
# =====================================================================


class TestActivityToBeta:
    """Verify f₀ → β mapping."""

    def test_reference_point(self) -> None:
        beta = activity_to_beta(2.0, f0_ref=2.0, beta_ref=10.0)
        assert abs(beta - 10.0) < 1e-10

    def test_higher_activity_lower_beta(self) -> None:
        beta_low = activity_to_beta(1.0)
        beta_high = activity_to_beta(4.0)
        assert beta_low > beta_high

    def test_zero_activity_infinite_beta(self) -> None:
        beta = activity_to_beta(0.0)
        assert math.isinf(beta)

    def test_negative_activity_infinite_beta(self) -> None:
        beta = activity_to_beta(-1.0)
        assert math.isinf(beta)

    def test_inversely_proportional(self) -> None:
        """β ∝ 1/f₀."""
        b1 = activity_to_beta(1.0)
        b2 = activity_to_beta(2.0)
        assert abs(b1 / b2 - 2.0) < 1e-10


# =====================================================================
# Synthetic Data Generator Tests
# =====================================================================


class TestSyntheticData:
    """Verify synthetic velocity data generator."""

    def test_output_shapes(self) -> None:
        velocities, _boundaries = generate_synthetic_velocities(n_particles=50, T_per_phase=20)
        T = 3 * 20 + 2 * 20  # 3 phases + 2 transitions
        assert velocities.shape == (T, 50)

    def test_five_phases(self) -> None:
        _, boundaries = generate_synthetic_velocities()
        assert len(boundaries) == 5

    def test_boundaries_cover_full_range(self) -> None:
        velocities, boundaries = generate_synthetic_velocities()
        assert boundaries[0][1] == 0
        assert boundaries[-1][2] == velocities.shape[0]

    def test_deterministic_seed(self) -> None:
        v1, _ = generate_synthetic_velocities(seed=123)
        v2, _ = generate_synthetic_velocities(seed=123)
        np.testing.assert_array_equal(v1, v2)

    def test_velocities_non_negative(self) -> None:
        velocities, _ = generate_synthetic_velocities()
        assert np.all(velocities >= 0)

    def test_cooled_slower_than_heated(self) -> None:
        velocities, boundaries = generate_synthetic_velocities()
        cooled = velocities[boundaries[0][1] : boundaries[0][2]]
        heated = velocities[boundaries[-1][1] : boundaries[-1][2]]
        assert np.mean(cooled) < np.mean(heated)


# =====================================================================
# Full Audit Pipeline Tests
# =====================================================================


class TestFullAudit:
    """Verify the complete audit pipeline."""

    @pytest.fixture(scope="class")
    def audit_result(self) -> ActiveMatterAudit:
        velocities, boundaries = generate_synthetic_velocities(n_particles=50, T_per_phase=30, T_transition=10, seed=42)
        return run_audit(velocities, boundaries)

    def test_returns_audit_type(self, audit_result: ActiveMatterAudit) -> None:
        assert isinstance(audit_result, ActiveMatterAudit)

    def test_phase_count(self, audit_result: ActiveMatterAudit) -> None:
        assert len(audit_result.phases) == 5  # 3 stable + 2 transitions

    def test_fisher_distances_positive(self, audit_result: ActiveMatterAudit) -> None:
        for key, d in audit_result.fisher_distances.items():
            assert d > 0, f"Fisher distance {key} = {d}, expected > 0"

    def test_path_efficiency_in_range(self, audit_result: ActiveMatterAudit) -> None:
        assert 0 < audit_result.path_efficiency <= 1.0

    def test_universality_c_eff(self, audit_result: ActiveMatterAudit) -> None:
        assert abs(audit_result.universality["c_eff"] - 1 / 3) < 1e-10

    def test_grammar_all_ordered(self, audit_result: ActiveMatterAudit) -> None:
        for key, g in audit_result.grammar.items():
            assert g["complexity"] == "ORDERED", f"{key}: {g['complexity']}"

    def test_corrected_thresholds_present(self, audit_result: ActiveMatterAudit) -> None:
        assert "STABLE" in audit_result.corrected_thresholds
        assert "WATCH" in audit_result.corrected_thresholds
        assert "COLLAPSE" in audit_result.corrected_thresholds
        assert "CRITICAL" in audit_result.corrected_thresholds

    def test_config_preserved(self, audit_result: ActiveMatterAudit) -> None:
        # run_audit uses default config when none passed; n_particles comes from default
        assert audit_result.config.n_particles == 180


# =====================================================================
# Budget Thermodynamics Tests
# =====================================================================


class TestBudgetThermodynamics:
    """Verify Γ(ω) drift cost and D_C curvature cost."""

    def test_drift_cost_formula(self) -> None:
        """Γ(ω) = ω³/(1−ω+ε); canonical formula from frozen_contract."""
        from umcp.frozen_contract import EPSILON, gamma_omega

        omega = 0.3
        expected = omega**3 / (1 - omega + EPSILON)
        assert abs(gamma_omega(omega) - expected) < 1e-10

    def test_curvature_cost_formula(self) -> None:
        """D_C = α·C; ALPHA = 1.0 (not 1.5 as paper claims)."""
        from umcp.frozen_contract import ALPHA, cost_curvature

        C = 0.5
        assert abs(cost_curvature(C) - ALPHA * C) < 1e-10
        assert abs(ALPHA - 1.0) < 1e-10  # NOT 1.5

    def test_equator_separates_phases(self, phase_invariants: dict[str, list[dict]]) -> None:
        """All phases above equator (Φ_eq > 0) for this embedding."""
        for phase in ["Cooled", "Heated"]:
            if phase in phase_invariants:
                phis = [r["phi_eq"] for r in phase_invariants[phase]]
                assert np.mean(phis) > 0, f"{phase} Φ_eq = {np.mean(phis)}, expected > 0"


# =====================================================================
# Paper Correction Tests (the key divergences)
# =====================================================================


class TestPaperCorrections:
    """Explicitly test that the canonical kernel differs from the paper."""

    def test_alpha_is_1_not_1_5(self) -> None:
        """Paper uses α = 1.5, canonical α = 1.0."""
        from umcp.frozen_contract import ALPHA

        assert abs(ALPHA - 1.0) < 1e-10

    def test_p_exponent_is_3(self) -> None:
        """Canonical p = 3."""
        from umcp.frozen_contract import P_EXPONENT

        assert P_EXPONENT == 3

    def test_epsilon_is_1e_8(self) -> None:
        from umcp.frozen_contract import EPSILON

        assert abs(EPSILON - 1e-8) < 1e-12

    def test_IC_is_exp_kappa_not_paper_formula(self) -> None:
        """The paper's IC includes F·(1-ω) = F² — this is redundant.
        The canonical IC = exp(κ) where κ = Σ wᵢ ln(cᵢ) (weighted)."""
        from umcp.frozen_contract import compute_kernel

        c = np.array([0.6, 0.4, 0.8, 0.5])
        w = np.array([0.25, 0.25, 0.25, 0.25])
        ko = compute_kernel(c, w, 1.0, 1e-8)

        # Canonical IC = exp(κ) where κ = Σ wᵢ ln(cᵢ) (weighted sum)
        canonical_IC = math.exp(float(np.dot(w, np.log(c))))

        # Paper's WRONG formula
        F = np.sum(w * c)
        S = -np.sum(w * c * np.log(c + 1e-8))
        C_val = np.std(c) / 0.5
        paper_IC = F * math.exp(-S) * (1 - ko.omega) * math.exp(-1.5 * C_val / (1 + 1.0))

        # Canonical matches kernel, paper doesn't
        assert abs(ko.IC - canonical_IC) < 1e-8
        assert abs(ko.IC - paper_IC) > 0.01  # they DIFFER

    def test_inf_rec_not_zero(self) -> None:
        """τ_R returns INF_REC on no match, not 0."""
        from umcp.frozen_contract import compute_tau_R

        # Single-point trace — no return possible
        trace = np.array([[0.5, 0.5, 0.5, 0.5]])
        tau = compute_tau_R(trace, 0, 0.1, 50, "L2")
        assert math.isinf(tau), f"τ_R = {tau}, expected inf"

    def test_regime_is_omega_cascade_not_or_chain(self) -> None:
        """Regime classification uses ω-primary cascade, not OR-chain."""
        from umcp.frozen_contract import classify_regime

        # integrity < 0.30 triggers CRITICAL regardless of other values
        regime = classify_regime(omega=0.10, F=0.90, S=0.05, C=0.05, integrity=0.20)
        assert regime.value == "CRITICAL"

        # ω ≥ 0.30 triggers COLLAPSE
        regime2 = classify_regime(omega=0.35, F=0.65, S=0.20, C=0.30, integrity=0.40)
        assert regime2.value == "COLLAPSE"

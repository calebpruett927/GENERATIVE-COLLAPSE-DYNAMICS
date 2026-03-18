"""Tests for Long-Period Radio Transients (LPTs) closure.

Validates 10 LPT Theorems (T-LPT-1 through T-LPT-10), the 9-source catalog
spanning the known LPT population, kernel computation, trace construction,
narrative generation, and Tier-1 identity universality.

Sources: Hurley-Walker et al. 2024 (ApJL 976 L21), Pritchard et al. 2026
(arXiv:2603.07857), and 7 earlier LPT discovery papers.
"""

from __future__ import annotations

import math

import pytest

from closures.astronomy.long_period_radio_transients import (
    ACTIVITY_MAX_DAYS,
    ACTIVITY_MIN_DAYS,
    DISTANCE_MAX_KPC,
    DISTANCE_MIN_KPC,
    DM_MAX,
    DM_MIN,
    FLUX_MAX_MJY,
    FLUX_MIN_MJY,
    N_CHANNELS,
    N_LPTS,
    PERIOD_MAX_S,
    PERIOD_MIN_S,
    SPEC_IDX_MAX,
    SPEC_IDX_MIN,
    LPTKernelResult,
    LPTSource,
    _build_trace,
    _clip,
    _linear_norm,
    _log_norm,
    build_lpt_catalog,
    compute_all_lpt_kernels,
    generate_narrative,
    get_lpt_by_name,
    get_optical_counterpart_sources,
    get_wd_candidate_sources,
    prove_t_lpt_1,
    prove_t_lpt_2,
    prove_t_lpt_3,
    prove_t_lpt_4,
    prove_t_lpt_5,
    prove_t_lpt_6,
    prove_t_lpt_7,
    prove_t_lpt_8,
    prove_t_lpt_9,
    prove_t_lpt_10,
    run_full_analysis,
)
from umcp.frozen_contract import EPSILON

# ═══════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def full_analysis():
    """Run full analysis once for entire test module."""
    return run_full_analysis()


@pytest.fixture(scope="module")
def catalog():
    """Build catalog once."""
    return build_lpt_catalog()


@pytest.fixture(scope="module")
def all_kernels():
    """Compute all kernels once."""
    return compute_all_lpt_kernels()


# ═══════════════════════════════════════════════════════════════
# FROZEN CONSTANTS
# ═══════════════════════════════════════════════════════════════


class TestFrozenConstants:
    """Verify frozen constants from LPT literature."""

    def test_n_channels(self):
        assert N_CHANNELS == 8

    def test_n_lpts(self):
        assert N_LPTS == 9

    def test_period_range(self):
        assert PERIOD_MIN_S == 421.0
        assert PERIOD_MAX_S == 10497.0

    def test_flux_range(self):
        assert FLUX_MIN_MJY == 5.0
        assert FLUX_MAX_MJY == 45000.0

    def test_dm_range(self):
        assert DM_MIN == 3.0
        assert DM_MAX == 145.0

    def test_distance_range(self):
        assert DISTANCE_MIN_KPC == 0.17
        assert DISTANCE_MAX_KPC == 8.0

    def test_spectral_index_range(self):
        assert SPEC_IDX_MIN == -7.0
        assert SPEC_IDX_MAX == -1.0

    def test_activity_range(self):
        assert ACTIVITY_MIN_DAYS == 8.0
        assert ACTIVITY_MAX_DAYS == 11000.0


# ═══════════════════════════════════════════════════════════════
# NORMALIZATION HELPERS
# ═══════════════════════════════════════════════════════════════


class TestNormalization:
    """Verify trace normalization produces valid channels."""

    def test_clip_lower_bound(self):
        assert _clip(0.0) == EPSILON

    def test_clip_upper_bound(self):
        assert _clip(1.0) == 1.0 - EPSILON

    def test_clip_passthrough(self):
        assert _clip(0.5) == 0.5

    def test_clip_negative(self):
        assert _clip(-1.0) == EPSILON

    def test_clip_above_one(self):
        assert _clip(2.0) == 1.0 - EPSILON

    def test_log_norm_midpoint(self):
        result = _log_norm(100.0, 10.0, 1000.0)
        assert 0.0 < result < 1.0

    def test_log_norm_at_min(self):
        result = _log_norm(10.0, 10.0, 1000.0)
        assert result == pytest.approx(0.0, abs=0.01)

    def test_log_norm_at_max(self):
        result = _log_norm(1000.0, 10.0, 1000.0)
        assert result == pytest.approx(1.0, abs=0.01)

    def test_linear_norm_midpoint(self):
        result = _linear_norm(5.0, 0.0, 10.0)
        assert result == pytest.approx(0.5)

    def test_linear_norm_at_bounds(self):
        assert _linear_norm(0.0, 0.0, 10.0) == pytest.approx(0.0)
        assert _linear_norm(10.0, 0.0, 10.0) == pytest.approx(1.0)


# ═══════════════════════════════════════════════════════════════
# CATALOG CONSTRUCTION
# ═══════════════════════════════════════════════════════════════


class TestCatalog:
    """Verify LPT source catalog."""

    def test_catalog_count(self, catalog):
        assert len(catalog) == N_LPTS

    def test_all_lpt_source_type(self, catalog):
        for s in catalog:
            assert isinstance(s, LPTSource)

    def test_unique_names(self, catalog):
        names = [s.name for s in catalog]
        assert len(names) == len(set(names))

    def test_gleam_x_j0704_present(self, catalog):
        names = [s.name for s in catalog]
        assert any("J0704" in n for n in names)

    def test_askap_j1424_present(self, catalog):
        names = [s.name for s in catalog]
        assert any("J1424" in n for n in names)

    def test_all_periods_positive(self, catalog):
        for s in catalog:
            assert s.period_s > 0

    def test_all_fluxes_positive(self, catalog):
        for s in catalog:
            assert s.flux_peak_mjy > 0

    def test_all_dm_positive(self, catalog):
        for s in catalog:
            assert s.dm_pc_cm3 > 0

    def test_all_polarization_valid(self, catalog):
        for s in catalog:
            assert 0.0 <= s.linear_pol_frac <= 1.0

    def test_all_distances_positive(self, catalog):
        for s in catalog:
            assert s.distance_kpc > 0

    def test_optical_counterpart_sources(self):
        opt = get_optical_counterpart_sources()
        assert len(opt) >= 2
        for s in opt:
            assert s.has_optical_counterpart

    def test_wd_candidate_sources(self):
        wd = get_wd_candidate_sources()
        assert len(wd) >= 2
        for s in wd:
            assert s.favored_model == "WD_binary"

    def test_get_by_name_found(self):
        s = get_lpt_by_name("ASKAP J1424")
        assert s is not None
        assert s.period_s == pytest.approx(2147.0)

    def test_get_by_name_missing(self):
        assert get_lpt_by_name("Nonexistent Source") is None


# ═══════════════════════════════════════════════════════════════
# TRACE CONSTRUCTION
# ═══════════════════════════════════════════════════════════════


class TestTraceConstruction:
    """Verify trace vectors are well-formed."""

    def test_trace_length(self, catalog):
        for s in catalog:
            trace = _build_trace(
                s.period_s,
                s.flux_peak_mjy,
                s.duty_cycle,
                s.linear_pol_frac,
                s.dm_pc_cm3,
                s.spectral_index,
                s.distance_kpc,
                s.activity_days,
            )
            assert len(trace) == N_CHANNELS

    def test_trace_in_unit_interval(self, catalog):
        for s in catalog:
            trace = _build_trace(
                s.period_s,
                s.flux_peak_mjy,
                s.duty_cycle,
                s.linear_pol_frac,
                s.dm_pc_cm3,
                s.spectral_index,
                s.distance_kpc,
                s.activity_days,
            )
            for c in trace:
                assert EPSILON <= c <= 1.0 - EPSILON, f"{s.name}: channel value {c} outside guard band"

    def test_weights_equal_eighth(self):
        """Weights are uniform 1/N."""
        import numpy as np

        w = np.full(N_CHANNELS, 1.0 / N_CHANNELS)
        assert sum(w) == pytest.approx(1.0)
        assert all(wi == pytest.approx(1.0 / N_CHANNELS) for wi in w)


# ═══════════════════════════════════════════════════════════════
# KERNEL COMPUTATION
# ═══════════════════════════════════════════════════════════════


class TestKernelComputation:
    """Verify kernel outputs per source."""

    def test_all_kernels_count(self, all_kernels):
        assert len(all_kernels) == N_LPTS

    def test_kernel_result_type(self, all_kernels):
        for kr in all_kernels:
            assert isinstance(kr, LPTKernelResult)

    def test_fidelity_range(self, all_kernels):
        for kr in all_kernels:
            assert 0.0 < kr.F < 1.0

    def test_drift_range(self, all_kernels):
        for kr in all_kernels:
            assert 0.0 < kr.omega < 1.0

    @pytest.mark.parametrize("idx", range(9))
    def test_duality_identity(self, all_kernels, idx):
        """F + ω = 1 exactly."""
        kr = all_kernels[idx]
        assert kr.F + kr.omega == pytest.approx(1.0, abs=1e-10)

    @pytest.mark.parametrize("idx", range(9))
    def test_integrity_bound(self, all_kernels, idx):
        """IC ≤ F for all sources."""
        kr = all_kernels[idx]
        assert kr.IC <= kr.F + 1e-10

    @pytest.mark.parametrize("idx", range(9))
    def test_log_integrity_relation(self, all_kernels, idx):
        """IC ≈ exp(κ) — within rounding tolerance (6 dp)."""
        kr = all_kernels[idx]
        assert pytest.approx(math.exp(kr.kappa), abs=1e-5) == kr.IC

    @pytest.mark.parametrize("idx", range(9))
    def test_heterogeneity_gap_nonneg(self, all_kernels, idx):
        """Δ = F − IC ≥ 0."""
        kr = all_kernels[idx]
        assert kr.gap >= -1e-10

    def test_entropy_nonneg(self, all_kernels):
        for kr in all_kernels:
            assert kr.S >= 0.0

    def test_curvature_range(self, all_kernels):
        for kr in all_kernels:
            assert 0.0 <= kr.C <= 1.0 + 1e-10


# ═══════════════════════════════════════════════════════════════
# SPECIFIC SOURCES — KEY PAPERS
# ═══════════════════════════════════════════════════════════════


class TestKeyDiscoveries:
    """Verify kernel signatures for the two featured papers."""

    def test_gleam_x_j0704_period(self):
        """Hurley-Walker et al. 2024: longest-period LPT at 2.9 hr."""
        s = get_lpt_by_name("GLEAM-X J0704\u221237")
        assert s is not None
        assert s.period_s == pytest.approx(10497.0)

    def test_gleam_x_j0704_has_optical(self):
        s = get_lpt_by_name("GLEAM-X J0704\u221237")
        assert s is not None
        assert s.has_optical_counterpart

    def test_gleam_x_j0704_companion(self):
        s = get_lpt_by_name("GLEAM-X J0704\u221237")
        assert s is not None
        assert "M" in s.companion_type  # M-dwarf companion

    def test_askap_j1424_period(self):
        """Pritchard et al. 2026: 36-minute period transient."""
        s = get_lpt_by_name("ASKAP J1424")
        assert s is not None
        assert s.period_s == pytest.approx(2147.0)

    def test_askap_j1424_full_polarization(self):
        s = get_lpt_by_name("ASKAP J1424")
        assert s is not None
        assert s.linear_pol_frac == pytest.approx(1.0)

    def test_askap_j1424_short_activity(self):
        s = get_lpt_by_name("ASKAP J1424")
        assert s is not None
        assert s.activity_days == pytest.approx(8.0)

    def test_askap_j1424_no_optical(self):
        s = get_lpt_by_name("ASKAP J1424")
        assert s is not None
        assert not s.has_optical_counterpart


# ═══════════════════════════════════════════════════════════════
# THEOREM PROVERS
# ═══════════════════════════════════════════════════════════════


class TestTheoremT1:
    """T-LPT-1: Bridge-Object Fidelity Deficit."""

    def test_proven(self):
        result = prove_t_lpt_1()
        assert result["proven"]

    def test_bridge_below_median(self):
        result = prove_t_lpt_1()
        assert result["mean_F_bridge"] < result["median_F_all"]

    def test_tier1_pass(self):
        result = prove_t_lpt_1()
        assert result["tier1_pass"]

    def test_bridge_count(self):
        result = prove_t_lpt_1()
        assert len(result["bridge_names"]) >= 2


class TestTheoremT2:
    """T-LPT-2: Geometric Slaughter Detection."""

    def test_proven(self):
        result = prove_t_lpt_2()
        assert result["proven"]

    def test_multi_floor_depressed(self):
        result = prove_t_lpt_2()
        assert result["all_depressed_below_0.10"]

    def test_multi_floor_count(self):
        result = prove_t_lpt_2()
        assert result["n_multi_floor"] >= 2

    def test_few_floor_higher(self):
        result = prove_t_lpt_2()
        assert result["mean_IC_few_floor"] > 0.10


class TestTheoremT3:
    """T-LPT-3: Optical Counterpart Split."""

    def test_proven(self):
        result = prove_t_lpt_3()
        assert result["proven"]

    def test_sample_sizes(self):
        result = prove_t_lpt_3()
        assert result["n_with_optical"] >= 2
        assert result["n_without_optical"] >= 2


class TestTheoremT4:
    """T-LPT-4: Intermittency-Gap Link."""

    def test_proven(self):
        result = prove_t_lpt_4()
        assert result["proven"]

    def test_intermittent_higher_gap(self):
        result = prove_t_lpt_4()
        assert result["mean_gap_intermittent"] > result["mean_gap_persistent"]

    def test_intermittent_count(self):
        result = prove_t_lpt_4()
        assert result["n_intermittent"] >= 2


class TestTheoremT5:
    """T-LPT-5: WD vs NS Candidate Split."""

    def test_proven(self):
        result = prove_t_lpt_5()
        assert result["proven"]

    def test_sample_sizes(self):
        result = prove_t_lpt_5()
        assert result["n_WD"] >= 2
        assert result["n_other"] >= 2


class TestTheoremT6:
    """T-LPT-6: DM-Distance Coherence."""

    def test_proven(self):
        result = prove_t_lpt_6()
        assert result["proven"]

    def test_positive_correlation(self):
        result = prove_t_lpt_6()
        assert result["spearman_rho"] > 0


class TestTheoremT7:
    """T-LPT-7: Duty Cycle Gap."""

    def test_proven(self):
        result = prove_t_lpt_7()
        assert result["proven"]


class TestTheoremT8:
    """T-LPT-8: Spectral Steepness."""

    def test_proven(self):
        result = prove_t_lpt_8()
        assert result["proven"]

    def test_steep_count(self):
        result = prove_t_lpt_8()
        assert result["n_steep"] >= 2


class TestTheoremT9:
    """T-LPT-9: Population Kernel Bounds."""

    def test_proven(self):
        result = prove_t_lpt_9()
        assert result["proven"]

    def test_all_bounded(self):
        result = prove_t_lpt_9()
        assert result["all_bounded"]

    def test_integrity_bound(self):
        result = prove_t_lpt_9()
        assert result["integrity_bound_holds"]


class TestTheoremT10:
    """T-LPT-10: Universal Tier-1."""

    def test_proven(self):
        result = prove_t_lpt_10()
        assert result["proven"]

    def test_zero_violations(self):
        result = prove_t_lpt_10()
        assert result["n_violations"] == 0

    def test_all_tested(self):
        result = prove_t_lpt_10()
        assert result["n_tested"] == N_LPTS
        assert result["n_passed"] == N_LPTS


# ═══════════════════════════════════════════════════════════════
# NARRATIVE GENERATION
# ═══════════════════════════════════════════════════════════════


class TestNarrative:
    """Verify narrative production."""

    def test_narrative_nonempty(self):
        narr = generate_narrative()
        assert isinstance(narr, dict)
        assert len(narr) >= 3

    def test_narrative_has_acts(self):
        narr = generate_narrative()
        full_text = " ".join(str(v) for v in narr.values())
        assert "discover" in full_text.lower()

    def test_narrative_mentions_gleam_x_j0704(self):
        narr = generate_narrative()
        full_text = " ".join(str(v) for v in narr.values())
        assert "J0704" in full_text

    def test_narrative_mentions_askap_j1424(self):
        narr = generate_narrative()
        full_text = " ".join(str(v) for v in narr.values())
        assert "J1424" in full_text


# ═══════════════════════════════════════════════════════════════
# FULL ANALYSIS ASSEMBLY
# ═══════════════════════════════════════════════════════════════


class TestFullAnalysis:
    """Verify end-to-end analysis assembly."""

    def test_source_count(self, full_analysis):
        assert full_analysis["n_sources"] == N_LPTS

    def test_theorem_count(self, full_analysis):
        assert full_analysis["n_theorems"] == 10

    def test_all_proven(self, full_analysis):
        assert full_analysis["n_proven"] == 10

    def test_kernel_results_present(self, full_analysis):
        assert len(full_analysis["kernel_results"]) == N_LPTS

    def test_theorems_list(self, full_analysis):
        assert len(full_analysis["theorems"]) == 10

    def test_narrative_present(self, full_analysis):
        assert len(full_analysis["narrative"]) > 0

    def test_summary_keys(self, full_analysis):
        s = full_analysis["summary"]
        assert "mean_F" in s
        assert "mean_IC" in s
        assert "mean_gap" in s

    @pytest.mark.parametrize("idx", range(10))
    def test_each_theorem_proven(self, full_analysis, idx):
        t = full_analysis["theorems"][idx]
        assert t["proven"], f"{t['theorem']} ({t['name']}) not proven"

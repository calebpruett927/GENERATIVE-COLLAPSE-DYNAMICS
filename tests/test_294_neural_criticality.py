"""Tests for neural criticality closure (clinical neuroscience domain).

Validates 12 entities, 8-channel trace construction,
Tier-1 kernel identities, and 7 theorems (T-NCR-1 through T-NCR-7).

Motivated by PRL March 2026: the human brain operates near, but not at,
the critical point.  GCD translation: healthy waking = Watch regime.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from closures.clinical_neuroscience.neural_criticality import (
    N_NCR_CHANNELS,
    NCR_CHANNELS,
    NCR_ENTITIES,
    NCRKernelResult,
    compute_all_entities,
    compute_ncr_kernel,
    verify_all_theorems,
    verify_t_ncr_1,
    verify_t_ncr_2,
    verify_t_ncr_3,
    verify_t_ncr_4,
    verify_t_ncr_5,
    verify_t_ncr_6,
    verify_t_ncr_7,
)


@pytest.fixture(scope="module")
def all_results() -> list[NCRKernelResult]:
    return compute_all_entities()


# ---------------------------------------------------------------------------
# Entity catalog structure tests
# ---------------------------------------------------------------------------


class TestEntityCatalog:
    def test_entity_count(self):
        assert len(NCR_ENTITIES) == 12

    def test_channel_count(self):
        assert N_NCR_CHANNELS == 8
        assert len(NCR_CHANNELS) == 8

    def test_all_categories_present(self):
        cats = {e.category for e in NCR_ENTITIES}
        assert cats == {"extreme", "healthy", "pathological", "sleep"}

    @pytest.mark.parametrize("entity", NCR_ENTITIES, ids=lambda e: e.name)
    def test_trace_vector_shape(self, entity):
        assert entity.trace_vector().shape == (8,)

    @pytest.mark.parametrize("entity", NCR_ENTITIES, ids=lambda e: e.name)
    def test_trace_vector_bounds(self, entity):
        c = entity.trace_vector()
        assert np.all(c >= 0.0) and np.all(c <= 1.0)

    @pytest.mark.parametrize("entity", NCR_ENTITIES, ids=lambda e: e.name)
    def test_unique_names(self, entity):
        names = [e.name for e in NCR_ENTITIES]
        assert names.count(entity.name) == 1


# ---------------------------------------------------------------------------
# Tier-1 kernel identity tests (F + ω = 1, IC ≤ F, IC = exp(κ))
# ---------------------------------------------------------------------------


class TestTier1Identities:
    @pytest.mark.parametrize("entity", NCR_ENTITIES, ids=lambda e: e.name)
    def test_duality_identity(self, entity):
        r = compute_ncr_kernel(entity)
        assert abs(r.F + r.omega - 1.0) < 1e-12

    @pytest.mark.parametrize("entity", NCR_ENTITIES, ids=lambda e: e.name)
    def test_integrity_bound(self, entity):
        r = compute_ncr_kernel(entity)
        assert r.IC <= r.F + 1e-12

    @pytest.mark.parametrize("entity", NCR_ENTITIES, ids=lambda e: e.name)
    def test_log_integrity_relation(self, entity):
        r = compute_ncr_kernel(entity)
        assert abs(r.IC - math.exp(r.kappa)) < 1e-10

    @pytest.mark.parametrize("entity", NCR_ENTITIES, ids=lambda e: e.name)
    def test_kernel_values_finite(self, entity):
        r = compute_ncr_kernel(entity)
        assert all(math.isfinite(v) for v in [r.F, r.omega, r.S, r.C, r.kappa, r.IC])

    @pytest.mark.parametrize("entity", NCR_ENTITIES, ids=lambda e: e.name)
    def test_regime_valid(self, entity):
        r = compute_ncr_kernel(entity)
        assert r.regime in {"Stable", "Watch", "Collapse"}


# ---------------------------------------------------------------------------
# Theorem tests (T-NCR-1 through T-NCR-6)
# ---------------------------------------------------------------------------


class TestTheorems:
    def test_t_ncr_1(self, all_results):
        assert verify_t_ncr_1(all_results)["passed"]

    def test_t_ncr_2(self, all_results):
        assert verify_t_ncr_2(all_results)["passed"]

    def test_t_ncr_3(self, all_results):
        assert verify_t_ncr_3(all_results)["passed"]

    def test_t_ncr_4(self, all_results):
        assert verify_t_ncr_4(all_results)["passed"]

    def test_t_ncr_5(self, all_results):
        assert verify_t_ncr_5(all_results)["passed"]

    def test_t_ncr_6(self, all_results):
        assert verify_t_ncr_6(all_results)["passed"]

    def test_t_ncr_7(self, all_results):
        assert verify_t_ncr_7(all_results)["passed"]

    def test_all_theorems_pass(self, all_results):
        for t in verify_all_theorems():
            assert t["passed"], f"{t['name']} failed"


# ---------------------------------------------------------------------------
# Structural tests — regime distribution, ordering, etc.
# ---------------------------------------------------------------------------


class TestStructural:
    def test_at_least_two_regimes(self, all_results):
        regimes = {r.regime for r in all_results}
        assert len(regimes) >= 2

    def test_healthy_all_watch(self, all_results):
        healthy = [r for r in all_results if r.category == "healthy"]
        assert all(r.regime == "Watch" for r in healthy)

    def test_coma_is_collapse(self, all_results):
        coma = next(r for r in all_results if r.name == "Coma")
        assert coma.regime == "Collapse"

    def test_flow_highest_f_among_healthy(self, all_results):
        healthy = [r for r in all_results if r.category == "healthy"]
        flow = next(r for r in all_results if r.name == "Flow_state")
        assert max(r.F for r in healthy) == flow.F

    def test_coma_lowest_f(self, all_results):
        coma = next(r for r in all_results if r.name == "Coma")
        assert min(r.F for r in all_results) == coma.F

    def test_rem_in_watch(self, all_results):
        rem = next(r for r in all_results if r.name == "REM_sleep")
        assert rem.regime == "Watch"

    def test_nrem3_in_collapse(self, all_results):
        nrem = next(r for r in all_results if r.name == "NREM3_slow_wave")
        assert nrem.regime == "Collapse"

    def test_seizure_largest_delta(self, all_results):
        deltas = {r.name: r.F - r.IC for r in all_results}
        assert max(deltas, key=lambda x: deltas[x]) == "Epileptic_seizure"

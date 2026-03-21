"""Tests for budget geometry closure (continuity theory domain).

Validates 12 budget surface locations, 8-channel trace construction,
Tier-1 kernel identities, and 6 theorems (T-BG-1 through T-BG-6).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from closures.continuity_theory.budget_geometry import (
    BG_CHANNELS,
    BG_ENTITIES,
    N_BG_CHANNELS,
    BGKernelResult,
    compute_all_entities,
    compute_bg_kernel,
    verify_all_theorems,
    verify_t_bg_1,
    verify_t_bg_2,
    verify_t_bg_3,
    verify_t_bg_4,
    verify_t_bg_5,
    verify_t_bg_6,
)


@pytest.fixture(scope="module")
def all_results() -> list[BGKernelResult]:
    return compute_all_entities()


class TestEntityCatalog:
    def test_entity_count(self):
        assert len(BG_ENTITIES) == 12

    def test_channel_count(self):
        assert N_BG_CHANNELS == 8
        assert len(BG_CHANNELS) == 8

    def test_all_categories_present(self):
        cats = {e.category for e in BG_ENTITIES}
        assert cats == {"flat_plain", "ramp", "wall", "special"}

    @pytest.mark.parametrize("entity", BG_ENTITIES, ids=lambda e: e.name)
    def test_trace_vector_shape(self, entity):
        assert entity.trace_vector().shape == (8,)

    @pytest.mark.parametrize("entity", BG_ENTITIES, ids=lambda e: e.name)
    def test_trace_vector_bounds(self, entity):
        c = entity.trace_vector()
        assert np.all(c >= 0.0) and np.all(c <= 1.0)


class TestTier1Identities:
    @pytest.mark.parametrize("entity", BG_ENTITIES, ids=lambda e: e.name)
    def test_duality_identity(self, entity):
        r = compute_bg_kernel(entity)
        assert abs(r.F + r.omega - 1.0) < 1e-12

    @pytest.mark.parametrize("entity", BG_ENTITIES, ids=lambda e: e.name)
    def test_integrity_bound(self, entity):
        r = compute_bg_kernel(entity)
        assert r.IC <= r.F + 1e-12

    @pytest.mark.parametrize("entity", BG_ENTITIES, ids=lambda e: e.name)
    def test_log_integrity_relation(self, entity):
        r = compute_bg_kernel(entity)
        assert abs(r.IC - math.exp(r.kappa)) < 1e-10


class TestTheorems:
    def test_t_bg_1(self, all_results):
        assert verify_t_bg_1(all_results)["passed"]

    def test_t_bg_2(self, all_results):
        assert verify_t_bg_2(all_results)["passed"]

    def test_t_bg_3(self, all_results):
        assert verify_t_bg_3(all_results)["passed"]

    def test_t_bg_4(self, all_results):
        assert verify_t_bg_4(all_results)["passed"]

    def test_t_bg_5(self, all_results):
        assert verify_t_bg_5(all_results)["passed"]

    def test_t_bg_6(self, all_results):
        assert verify_t_bg_6(all_results)["passed"]

    def test_all_theorems_pass(self):
        for t in verify_all_theorems():
            assert t["passed"], f"{t['name']} failed: {t}"


class TestRegimeClassification:
    @pytest.mark.parametrize("entity", BG_ENTITIES, ids=lambda e: e.name)
    def test_regime_is_valid(self, entity):
        r = compute_bg_kernel(entity)
        assert r.regime in ("Stable", "Watch", "Collapse")

"""Tests for fungi kingdom closure (evolution domain).

Validates 12 fungi species, 6 mycorrhizal stress configurations,
8-channel trace construction, Tier-1 kernel identities,
and 9 theorems (T-FK-1 through T-FK-9).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from closures.evolution.fungi_kingdom import (
    FK_CHANNELS,
    FK_ENTITIES,
    N_FK_CHANNELS,
    FKKernelResult,
    compute_all_entities,
    compute_fk_kernel,
    verify_all_theorems,
    verify_t_fk_1,
    verify_t_fk_2,
    verify_t_fk_3,
    verify_t_fk_4,
    verify_t_fk_5,
    verify_t_fk_6,
)


@pytest.fixture(scope="module")
def all_results() -> list[FKKernelResult]:
    return compute_all_entities()


class TestEntityCatalog:
    def test_entity_count(self):
        assert len(FK_ENTITIES) == 12

    def test_channel_count(self):
        assert N_FK_CHANNELS == 8
        assert len(FK_CHANNELS) == 8

    def test_all_categories_present(self):
        cats = {e.category for e in FK_ENTITIES}
        assert cats == {"decomposer", "symbiont", "transformer", "extremophile"}

    def test_three_per_category(self):
        for cat in ["decomposer", "symbiont", "transformer", "extremophile"]:
            count = sum(1 for e in FK_ENTITIES if e.category == cat)
            assert count == 3, f"{cat} has {count} entities, expected 3"

    @pytest.mark.parametrize("entity", FK_ENTITIES, ids=lambda e: e.name)
    def test_trace_vector_shape(self, entity):
        assert entity.trace_vector().shape == (8,)

    @pytest.mark.parametrize("entity", FK_ENTITIES, ids=lambda e: e.name)
    def test_trace_vector_bounds(self, entity):
        c = entity.trace_vector()
        assert np.all(c >= 0.0) and np.all(c <= 1.0)

    @pytest.mark.parametrize("entity", FK_ENTITIES, ids=lambda e: e.name)
    def test_entity_has_phylum(self, entity):
        assert entity.phylum in (
            "Basidiomycota",
            "Ascomycota",
            "Glomeromycota",
            "Chytridiomycota",
        )


class TestTier1Identities:
    @pytest.mark.parametrize("entity", FK_ENTITIES, ids=lambda e: e.name)
    def test_duality_identity(self, entity):
        r = compute_fk_kernel(entity)
        assert abs(r.F + r.omega - 1.0) < 1e-12

    @pytest.mark.parametrize("entity", FK_ENTITIES, ids=lambda e: e.name)
    def test_integrity_bound(self, entity):
        r = compute_fk_kernel(entity)
        assert r.IC <= r.F + 1e-12

    @pytest.mark.parametrize("entity", FK_ENTITIES, ids=lambda e: e.name)
    def test_log_integrity_relation(self, entity):
        r = compute_fk_kernel(entity)
        assert abs(r.IC - math.exp(r.kappa)) < 1e-10


class TestTheorems:
    def test_t_fk_1(self, all_results):
        assert verify_t_fk_1(all_results)["passed"]

    def test_t_fk_2(self, all_results):
        assert verify_t_fk_2(all_results)["passed"]

    def test_t_fk_3(self, all_results):
        assert verify_t_fk_3(all_results)["passed"]

    def test_t_fk_4(self, all_results):
        assert verify_t_fk_4(all_results)["passed"]

    def test_t_fk_5(self, all_results):
        assert verify_t_fk_5(all_results)["passed"]

    def test_t_fk_6(self, all_results):
        assert verify_t_fk_6(all_results)["passed"]

    def test_all_theorems_pass(self):
        for t in verify_all_theorems():
            assert t["passed"], f"{t['name']} failed: {t}"


class TestRegimeClassification:
    @pytest.mark.parametrize("entity", FK_ENTITIES, ids=lambda e: e.name)
    def test_regime_is_valid(self, entity):
        r = compute_fk_kernel(entity)
        assert r.regime in ("Stable", "Watch", "Collapse")

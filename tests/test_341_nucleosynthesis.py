"""Tests for Nucleosynthesis Pathways Closure (T-NS-1 through T-NS-6).

Tier-2 closure test: 12 nucleosynthesis processes through the GCD kernel.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_WORKSPACE = Path(__file__).resolve().parents[1]
for _p in [str(_WORKSPACE / "src"), str(_WORKSPACE)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from closures.nuclear_physics.nucleosynthesis import (
    NS_CHANNELS,
    NS_ENTITIES,
    NSKernelResult,
    compute_all_entities,
    compute_ns_kernel,
    verify_t_ns_1,
    verify_t_ns_2,
    verify_t_ns_3,
    verify_t_ns_4,
    verify_t_ns_5,
    verify_t_ns_6,
)


@pytest.fixture(scope="module")
def all_results() -> list[NSKernelResult]:
    return compute_all_entities()


# ---- Entity catalog -------------------------------------------------------


class TestEntityCatalog:
    def test_entity_count(self):
        assert len(NS_ENTITIES) == 12

    def test_channel_count(self):
        assert len(NS_CHANNELS) == 8

    def test_categories(self):
        cats = {e.category for e in NS_ENTITIES}
        assert cats == {"bbn", "stellar", "s_process", "r_process"}

    def test_trace_vector_shape(self):
        for e in NS_ENTITIES:
            assert e.trace_vector().shape == (8,)

    def test_trace_vector_bounds(self):
        for e in NS_ENTITIES:
            v = e.trace_vector()
            assert np.all(v >= 0.0) and np.all(v <= 1.0)


# ---- Tier-1 identities (parametrized) ------------------------------------


@pytest.mark.parametrize("entity", NS_ENTITIES, ids=[e.name for e in NS_ENTITIES])
class TestTier1Identities:
    def test_duality(self, entity):
        r = compute_ns_kernel(entity)
        assert abs(r.F + r.omega - 1.0) < 1e-12

    def test_integrity_bound(self, entity):
        r = compute_ns_kernel(entity)
        assert r.IC <= r.F + 1e-12

    def test_log_integrity(self, entity):
        r = compute_ns_kernel(entity)
        assert abs(r.IC - math.exp(r.kappa)) < 1e-12


# ---- Theorems -------------------------------------------------------------


class TestTheorems:
    def test_t_ns_1(self, all_results):
        assert verify_t_ns_1(all_results)["passed"]

    def test_t_ns_2(self, all_results):
        assert verify_t_ns_2(all_results)["passed"]

    def test_t_ns_3(self, all_results):
        assert verify_t_ns_3(all_results)["passed"]

    def test_t_ns_4(self, all_results):
        assert verify_t_ns_4(all_results)["passed"]

    def test_t_ns_5(self, all_results):
        assert verify_t_ns_5(all_results)["passed"]

    def test_t_ns_6(self, all_results):
        assert verify_t_ns_6(all_results)["passed"]

    def test_all_theorems_pass(self, all_results):
        theorems = [
            verify_t_ns_1(all_results),
            verify_t_ns_2(all_results),
            verify_t_ns_3(all_results),
            verify_t_ns_4(all_results),
            verify_t_ns_5(all_results),
            verify_t_ns_6(all_results),
        ]
        for t in theorems:
            assert t["passed"], f"{t['name']} failed: {t}"


# ---- Regime classification ------------------------------------------------


@pytest.mark.parametrize("entity", NS_ENTITIES, ids=[e.name for e in NS_ENTITIES])
class TestRegimeClassification:
    def test_regime_is_valid(self, entity):
        r = compute_ns_kernel(entity)
        assert r.regime in ("Stable", "Watch", "Collapse")

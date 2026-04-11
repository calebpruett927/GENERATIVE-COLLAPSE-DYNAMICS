"""Tests for Quantum Error Correction Closure (T-QEC-1 through T-QEC-6).

Tier-2 closure test: 12 QEC codes through the GCD kernel.
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

from closures.quantum_mechanics.quantum_error_correction import (
    QEC_CHANNELS,
    QEC_ENTITIES,
    QECKernelResult,
    compute_all_entities,
    compute_qec_kernel,
    verify_t_qec_1,
    verify_t_qec_2,
    verify_t_qec_3,
    verify_t_qec_4,
    verify_t_qec_5,
    verify_t_qec_6,
)


@pytest.fixture(scope="module")
def all_results() -> list[QECKernelResult]:
    return compute_all_entities()


# ---- Entity catalog -------------------------------------------------------


class TestEntityCatalog:
    def test_entity_count(self):
        assert len(QEC_ENTITIES) == 12

    def test_channel_count(self):
        assert len(QEC_CHANNELS) == 8

    def test_categories(self):
        cats = {e.category for e in QEC_ENTITIES}
        assert cats == {"surface", "topological", "block", "bosonic"}

    def test_trace_vector_shape(self):
        for e in QEC_ENTITIES:
            assert e.trace_vector().shape == (8,)

    def test_trace_vector_bounds(self):
        for e in QEC_ENTITIES:
            v = e.trace_vector()
            assert np.all(v >= 0.0) and np.all(v <= 1.0)


# ---- Tier-1 identities (parametrized) ------------------------------------


@pytest.mark.parametrize("entity", QEC_ENTITIES, ids=[e.name for e in QEC_ENTITIES])
class TestTier1Identities:
    def test_duality(self, entity):
        r = compute_qec_kernel(entity)
        assert abs(r.F + r.omega - 1.0) < 1e-12

    def test_integrity_bound(self, entity):
        r = compute_qec_kernel(entity)
        assert r.IC <= r.F + 1e-12

    def test_log_integrity(self, entity):
        r = compute_qec_kernel(entity)
        assert abs(r.IC - math.exp(r.kappa)) < 1e-12


# ---- Theorems -------------------------------------------------------------


class TestTheorems:
    def test_t_qec_1(self, all_results):
        assert verify_t_qec_1(all_results)["passed"]

    def test_t_qec_2(self, all_results):
        assert verify_t_qec_2(all_results)["passed"]

    def test_t_qec_3(self, all_results):
        assert verify_t_qec_3(all_results)["passed"]

    def test_t_qec_4(self, all_results):
        assert verify_t_qec_4(all_results)["passed"]

    def test_t_qec_5(self, all_results):
        assert verify_t_qec_5(all_results)["passed"]

    def test_t_qec_6(self, all_results):
        assert verify_t_qec_6(all_results)["passed"]

    def test_all_theorems_pass(self, all_results):
        theorems = [
            verify_t_qec_1(all_results),
            verify_t_qec_2(all_results),
            verify_t_qec_3(all_results),
            verify_t_qec_4(all_results),
            verify_t_qec_5(all_results),
            verify_t_qec_6(all_results),
        ]
        for t in theorems:
            assert t["passed"], f"{t['name']} failed: {t}"


# ---- Regime classification ------------------------------------------------


@pytest.mark.parametrize("entity", QEC_ENTITIES, ids=[e.name for e in QEC_ENTITIES])
class TestRegimeClassification:
    def test_regime_is_valid(self, entity):
        r = compute_qec_kernel(entity)
        assert r.regime in ("Stable", "Watch", "Collapse")

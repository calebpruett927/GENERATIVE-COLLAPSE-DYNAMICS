"""Tests for Malbolge kernel closure (dynamic semiotics domain).

Validates 12 esoteric programming language entities, 8-channel trace
construction, Tier-1 kernel identities, and 6 theorems (T-MB-1 through
T-MB-6).

Esoteric languages are native dissolution laboratories — every entity
lives in the Collapse regime. The closure distinguishes *uniform dissolution*
(Malbolge: all channels near-zero, small Δ) from *heterogeneous collapse*
(Iota: extreme channel variance, large Δ).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from closures.dynamic_semiotics.malbolge_kernel import (
    MB_CHANNELS,
    MB_ENTITIES,
    N_MB_CHANNELS,
    MBKernelResult,
    compute_all_entities,
    compute_mb_kernel,
    verify_all_theorems,
    verify_t_mb_1,
    verify_t_mb_2,
    verify_t_mb_3,
    verify_t_mb_4,
    verify_t_mb_5,
    verify_t_mb_6,
)


@pytest.fixture(scope="module")
def all_results() -> list[MBKernelResult]:
    return compute_all_entities()


# ── Entity Catalog ───────────────────────────────────────────────────


class TestEntityCatalog:
    def test_entity_count(self):
        assert len(MB_ENTITIES) == 12

    def test_channel_count(self):
        assert N_MB_CHANNELS == 8
        assert len(MB_CHANNELS) == 8

    def test_all_categories_present(self):
        cats = {e.category for e in MB_ENTITIES}
        assert cats == {"adversarial", "self_modifying", "minimalist", "structured_esoteric"}

    def test_three_per_category(self):
        for cat in {"adversarial", "self_modifying", "minimalist", "structured_esoteric"}:
            count = sum(1 for e in MB_ENTITIES if e.category == cat)
            assert count == 3, f"{cat} has {count} entities, expected 3"

    @pytest.mark.parametrize("entity", MB_ENTITIES, ids=lambda e: e.name)
    def test_trace_vector_shape(self, entity):
        assert entity.trace_vector().shape == (8,)

    @pytest.mark.parametrize("entity", MB_ENTITIES, ids=lambda e: e.name)
    def test_trace_vector_bounds(self, entity):
        c = entity.trace_vector()
        assert np.all(c >= 0.0) and np.all(c <= 1.0)

    @pytest.mark.parametrize("entity", MB_ENTITIES, ids=lambda e: e.name)
    def test_entity_has_name(self, entity):
        assert entity.name and isinstance(entity.name, str)

    @pytest.mark.parametrize("entity", MB_ENTITIES, ids=lambda e: e.name)
    def test_entity_has_category(self, entity):
        assert entity.category in {
            "adversarial",
            "self_modifying",
            "minimalist",
            "structured_esoteric",
        }


# ── Tier-1 Identities ───────────────────────────────────────────────


class TestTier1Identities:
    """Verify the three algebraic identities for every entity."""

    @pytest.mark.parametrize("entity", MB_ENTITIES, ids=lambda e: e.name)
    def test_duality_identity(self, entity):
        """F + ω = 1 (exact)."""
        r = compute_mb_kernel(entity)
        assert abs(r.F + r.omega - 1.0) < 1e-12

    @pytest.mark.parametrize("entity", MB_ENTITIES, ids=lambda e: e.name)
    def test_integrity_bound(self, entity):
        """IC ≤ F (the solvability condition)."""
        r = compute_mb_kernel(entity)
        assert r.IC <= r.F + 1e-12

    @pytest.mark.parametrize("entity", MB_ENTITIES, ids=lambda e: e.name)
    def test_log_integrity_relation(self, entity):
        """IC = exp(κ)."""
        r = compute_mb_kernel(entity)
        assert abs(r.IC - math.exp(r.kappa)) < 1e-10

    @pytest.mark.parametrize("entity", MB_ENTITIES, ids=lambda e: e.name)
    def test_F_in_unit_interval(self, entity):
        r = compute_mb_kernel(entity)
        assert 0.0 <= r.F <= 1.0

    @pytest.mark.parametrize("entity", MB_ENTITIES, ids=lambda e: e.name)
    def test_omega_in_unit_interval(self, entity):
        r = compute_mb_kernel(entity)
        assert 0.0 <= r.omega <= 1.0

    @pytest.mark.parametrize("entity", MB_ENTITIES, ids=lambda e: e.name)
    def test_entropy_non_negative(self, entity):
        r = compute_mb_kernel(entity)
        assert r.S >= -1e-12

    @pytest.mark.parametrize("entity", MB_ENTITIES, ids=lambda e: e.name)
    def test_curvature_in_unit_interval(self, entity):
        r = compute_mb_kernel(entity)
        assert 0.0 <= r.C <= 1.0 + 1e-12


# ── Theorems ─────────────────────────────────────────────────────────


class TestTheorems:
    def test_t_mb_1(self, all_results):
        """T-MB-1: Malbolge & Malbolge Unshackled are in Collapse regime."""
        assert verify_t_mb_1(all_results)["passed"]

    def test_t_mb_2(self, all_results):
        """T-MB-2: Adversarial category has lowest mean F."""
        assert verify_t_mb_2(all_results)["passed"]

    def test_t_mb_3(self, all_results):
        """T-MB-3: Iota has the largest heterogeneity gap (Δ = F − IC)."""
        assert verify_t_mb_3(all_results)["passed"]

    def test_t_mb_4(self, all_results):
        """T-MB-4: Brainfuck has the highest F."""
        assert verify_t_mb_4(all_results)["passed"]

    def test_t_mb_5(self, all_results):
        """T-MB-5: Structured esoteric has highest mean debug_observability."""
        assert verify_t_mb_5(all_results)["passed"]

    def test_t_mb_6(self, all_results):
        """T-MB-6: Malbolge has ω > 0.95 — deepest Collapse."""
        assert verify_t_mb_6(all_results)["passed"]

    def test_all_theorems_pass(self):
        for t in verify_all_theorems():
            assert t["passed"], f"{t['name']} failed: {t}"


# ── Regime Classification ────────────────────────────────────────────


class TestRegimeClassification:
    @pytest.mark.parametrize("entity", MB_ENTITIES, ids=lambda e: e.name)
    def test_regime_is_valid(self, entity):
        r = compute_mb_kernel(entity)
        assert r.regime in ("Stable", "Watch", "Collapse")

    def test_all_entities_in_collapse(self, all_results):
        """Every esoteric language should be in Collapse regime —
        these are dissolution laboratories by construction."""
        for r in all_results:
            assert r.regime == "Collapse", f"{r.name} in {r.regime} (ω={r.omega:.3f}), expected Collapse"


# ── Structural Properties ───────────────────────────────────────────


class TestStructuralProperties:
    def test_malbolge_uniform_dissolution(self, all_results):
        """Malbolge has near-uniform low channels → small Δ, extreme ω."""
        mb = next(r for r in all_results if r.name == "malbolge")
        assert mb.omega > 0.95
        assert (mb.F - mb.IC) < 0.05  # Small gap = uniform dissolution

    def test_iota_heterogeneous_collapse(self, all_results):
        """Iota has extreme channel variance → largest Δ."""
        iota = next(r for r in all_results if r.name == "iota")
        assert (iota.F - iota.IC) > 0.10  # Large gap = heterogeneous collapse

    def test_adversarial_vs_minimalist_ordering(self, all_results):
        """Adversarial languages have lower mean F than minimalist ones."""
        adv_F = np.mean([r.F for r in all_results if r.category == "adversarial"])
        min_F = np.mean([r.F for r in all_results if r.category == "minimalist"])
        assert adv_F < min_F

    def test_to_dict_roundtrip(self, all_results):
        """MBKernelResult.to_dict() produces a valid dict with all fields."""
        for r in all_results:
            d = r.to_dict()
            assert d["name"] == r.name
            assert d["F"] == r.F
            assert d["omega"] == r.omega
            assert d["IC"] == r.IC
            assert d["regime"] == r.regime

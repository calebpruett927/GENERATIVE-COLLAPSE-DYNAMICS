"""Tests for AFR.RETURN.v1 — Antifragility Return-Morphology Diagnostic.

Validates 12 synthetic collapse-return episodes, 8-channel trace construction,
Tier-1 kernel identities on all three states (K0, Kc, KR), five derived measures
(RS, FG, DR, RR, HR), morphology classification, and 6 theorems (T-AFR-1 through
T-AFR-6).

Tier discipline: this test file never redefines F, omega, S, C, kappa, IC, tau_R,
or regime. It reads from the closure's public API only.
"""

from __future__ import annotations

import math
import typing

import pytest

from closures.continuity_theory.afr_return_morphology import (
    AFR_CHANNELS,
    AFR_EPISODES,
    N_AFR_CHANNELS,
    TOL_SURPLUS,
    AntifragilityEpisode,
    CollapseState,
    compute_all_episodes,
    verify_all_theorems,
    verify_t_afr_1,
    verify_t_afr_2,
    verify_t_afr_3,
    verify_t_afr_4,
    verify_t_afr_5,
    verify_t_afr_6,
)


@pytest.fixture(scope="module")
def all_episodes() -> list[AntifragilityEpisode]:
    return compute_all_episodes()


# ---------------------------------------------------------------------------
# Episode catalog
# ---------------------------------------------------------------------------


class TestEpisodeCatalog:
    def test_episode_count(self):
        assert len(AFR_EPISODES) == 12

    def test_channel_count(self):
        assert N_AFR_CHANNELS == 8
        assert len(AFR_CHANNELS) == 8

    def test_compute_all_returns_same(self, all_episodes):
        assert len(all_episodes) == len(AFR_EPISODES)

    def test_episode_names_unique(self, all_episodes):
        names = [e.name for e in all_episodes]
        assert len(names) == len(set(names))

    def test_fragile_episodes_present(self, all_episodes):
        fragile = [e for e in all_episodes if e.name.startswith("fragile_")]
        assert len(fragile) >= 3

    def test_antifragile_episodes_present(self, all_episodes):
        af = [e for e in all_episodes if e.name.startswith("antifragile_")]
        assert len(af) >= 3

    def test_damaged_episodes_present(self, all_episodes):
        dmg = [e for e in all_episodes if e.name.startswith("damaged_")]
        assert len(dmg) >= 3

    def test_resilient_episodes_present(self, all_episodes):
        res = [e for e in all_episodes if e.name.startswith("resilient_")]
        assert len(res) >= 3


# ---------------------------------------------------------------------------
# Tier-1 identities on all three states of every episode
# ---------------------------------------------------------------------------


def _check_state(s: CollapseState) -> None:
    assert abs(s.F + s.omega - 1.0) < 1e-10, f"Duality: {s.label}"
    assert s.IC <= s.F + 1e-10, f"IC <= F: {s.label}"
    assert abs(s.IC - math.exp(s.kappa)) < 1e-10, f"IC=exp(kappa): {s.label}"


class TestTier1Identities:
    @pytest.mark.parametrize("ep", AFR_EPISODES, ids=lambda e: e.name)
    def test_pre_state_duality(self, ep):
        assert abs(ep.K0.F + ep.K0.omega - 1.0) < 1e-10

    @pytest.mark.parametrize("ep", AFR_EPISODES, ids=lambda e: e.name)
    def test_pre_state_integrity_bound(self, ep):
        assert ep.K0.IC <= ep.K0.F + 1e-10

    @pytest.mark.parametrize("ep", AFR_EPISODES, ids=lambda e: e.name)
    def test_pre_state_log_integrity(self, ep):
        assert abs(ep.K0.IC - math.exp(ep.K0.kappa)) < 1e-10

    @pytest.mark.parametrize("ep", AFR_EPISODES, ids=lambda e: e.name)
    def test_collapse_state_duality(self, ep):
        assert abs(ep.Kc.F + ep.Kc.omega - 1.0) < 1e-10

    @pytest.mark.parametrize("ep", AFR_EPISODES, ids=lambda e: e.name)
    def test_collapse_state_integrity_bound(self, ep):
        assert ep.Kc.IC <= ep.Kc.F + 1e-10

    @pytest.mark.parametrize("ep", AFR_EPISODES, ids=lambda e: e.name)
    def test_return_state_duality(self, ep):
        if ep.finite_return:
            assert abs(ep.KR.F + ep.KR.omega - 1.0) < 1e-10

    @pytest.mark.parametrize("ep", AFR_EPISODES, ids=lambda e: e.name)
    def test_return_state_integrity_bound(self, ep):
        if ep.finite_return:
            assert ep.KR.IC <= ep.KR.F + 1e-10

    @pytest.mark.parametrize("ep", AFR_EPISODES, ids=lambda e: e.name)
    def test_return_state_log_integrity(self, ep):
        if ep.finite_return:
            assert abs(ep.KR.IC - math.exp(ep.KR.kappa)) < 1e-10


# ---------------------------------------------------------------------------
# Tier-2 derived measures (RS, FG, DR, RR, HR)
# ---------------------------------------------------------------------------


class TestDerivedMeasures:
    @pytest.mark.parametrize("ep", AFR_EPISODES, ids=lambda e: e.name)
    def test_rs_is_ic_difference(self, ep):
        expected = ep.KR.IC - ep.K0.IC
        assert abs(ep.RS - expected) < 1e-12

    @pytest.mark.parametrize("ep", AFR_EPISODES, ids=lambda e: e.name)
    def test_fg_is_f_difference(self, ep):
        expected = ep.KR.F - ep.K0.F
        assert abs(ep.FG - expected) < 1e-12

    @pytest.mark.parametrize("ep", AFR_EPISODES, ids=lambda e: e.name)
    def test_dr_is_omega_reduction(self, ep):
        expected = ep.K0.omega - ep.KR.omega
        assert abs(ep.DR - expected) < 1e-12

    @pytest.mark.parametrize("ep", AFR_EPISODES, ids=lambda e: e.name)
    def test_rr_is_c_reduction(self, ep):
        expected = ep.K0.C - ep.KR.C
        assert abs(ep.RR - expected) < 1e-12

    @pytest.mark.parametrize("ep", AFR_EPISODES, ids=lambda e: e.name)
    def test_hr_is_gap_repair(self, ep):
        gap_pre = ep.K0.F - ep.K0.IC
        gap_post = ep.KR.F - ep.KR.IC
        expected = gap_pre - gap_post
        assert abs(ep.HR - expected) < 1e-12

    @pytest.mark.parametrize("ep", AFR_EPISODES, ids=lambda e: e.name)
    def test_ic_never_exceeds_f_in_gap(self, ep):
        # Derived from Tier-1: gap must be non-negative
        assert ep.K0.F - ep.K0.IC >= -1e-10
        assert ep.KR.F - ep.KR.IC >= -1e-10


# ---------------------------------------------------------------------------
# Morphology classification
# ---------------------------------------------------------------------------


class TestMorphologyClassification:
    VALID_CATEGORIES: typing.ClassVar[set[str]] = {
        "FRAGILE",
        "DAMAGED",
        "ROBUST",
        "RESILIENT",
        "ANTIFRAGILE",
        "FALSE_ANTIFRAGILE",
    }

    @pytest.mark.parametrize("ep", AFR_EPISODES, ids=lambda e: e.name)
    def test_category_is_valid(self, ep):
        assert ep.category in self.VALID_CATEGORIES

    def test_fragile_episodes_classified_fragile(self, all_episodes):
        fragile = [e for e in all_episodes if e.name.startswith("fragile_")]
        for ep in fragile:
            assert ep.category == "FRAGILE", f"{ep.name} should be FRAGILE"

    def test_antifragile_episodes_have_rs_positive(self, all_episodes):
        af = [e for e in all_episodes if e.category == "ANTIFRAGILE"]
        for ep in af:
            assert ep.RS > TOL_SURPLUS, f"{ep.name} RS={ep.RS} <= TOL_SURPLUS"

    def test_antifragile_episodes_have_hr_nonnegative(self, all_episodes):
        af = [e for e in all_episodes if e.category == "ANTIFRAGILE"]
        for ep in af:
            assert ep.HR >= 0, f"{ep.name} HR={ep.HR} < 0 — hidden fracture"

    def test_damaged_have_finite_return(self, all_episodes):
        dmg = [e for e in all_episodes if e.category == "DAMAGED"]
        for ep in dmg:
            assert ep.finite_return, f"{ep.name} should have finite return"

    def test_resilient_had_meaningful_collapse(self, all_episodes):
        res = [e for e in all_episodes if e.category == "RESILIENT"]
        for ep in res:
            # Meaningful collapse means Kc.omega >= 0.30
            assert ep.Kc.omega >= 0.30, f"{ep.name} collapse not deep enough"

    def test_to_dict_contains_required_keys(self, all_episodes):
        required = {
            "name",
            "RS",
            "FG",
            "DR",
            "RR",
            "HR",
            "finite_return",
            "category",
            "K0_F",
            "K0_IC",
            "Kc_F",
            "Kc_IC",
            "KR_F",
            "KR_IC",
        }
        for ep in all_episodes:
            d = ep.to_dict()
            assert required.issubset(set(d.keys())), f"{ep.name}: missing keys"


# ---------------------------------------------------------------------------
# Theorems
# ---------------------------------------------------------------------------


class TestTheorems:
    def test_t_afr_1(self, all_episodes):
        result = verify_t_afr_1(all_episodes)
        assert result["passed"], f"T-AFR-1 failed: {result}"

    def test_t_afr_2(self, all_episodes):
        result = verify_t_afr_2(all_episodes)
        assert result["passed"], f"T-AFR-2 failed: {result}"

    def test_t_afr_3(self, all_episodes):
        result = verify_t_afr_3(all_episodes)
        assert result["passed"], f"T-AFR-3 failed: {result}"

    def test_t_afr_4(self, all_episodes):
        result = verify_t_afr_4(all_episodes)
        assert result["passed"], f"T-AFR-4 failed: {result}"

    def test_t_afr_5(self, all_episodes):
        result = verify_t_afr_5(all_episodes)
        assert result["passed"], f"T-AFR-5 failed: {result}"

    def test_t_afr_6(self, all_episodes):
        result = verify_t_afr_6(all_episodes)
        assert result["passed"], f"T-AFR-6 failed: {result}"

    def test_all_theorems_pass(self):
        for t in verify_all_theorems():
            assert t["passed"], f"{t['name']} failed: {t}"

    def test_theorem_count(self):
        results = verify_all_theorems()
        assert len(results) == 6

    def test_theorem_names(self):
        results = verify_all_theorems()
        names = [r["name"] for r in results]
        expected = [f"T-AFR-{i}" for i in range(1, 7)]
        assert names == expected

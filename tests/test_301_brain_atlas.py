"""Tests for brain atlas closure (clinical-neuroscience domain).

Validates 35 brain anatomical structures, 8-channel trace construction,
Tier-1 kernel identities, 6 theorems (T-BA-1 through T-BA-6), and
3 systems (atlas comparison, segmentation protocols, clinical diagnostics).

Data sourced from HoliAtlas (Manjón et al. 2026, Sci Rep 16:9457)
and NextBrain (Iglesias et al. 2025, Nature).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from closures.clinical_neuroscience.brain_atlas_closure import (
    BA_CHANNELS,
    BA_ENTITIES,
    HOLIATLAS_METADATA,
    N_BA_CHANNELS,
    NEXTBRAIN_METADATA,
    BAKernelResult,
    atlas_comparison_system,
    clinical_diagnostic_system,
    compute_all_entities,
    compute_ba_kernel,
    segmentation_protocol_system,
    verify_all_theorems,
    verify_t_ba_1,
    verify_t_ba_2,
    verify_t_ba_3,
    verify_t_ba_4,
    verify_t_ba_5,
    verify_t_ba_6,
)


@pytest.fixture(scope="module")
def all_results() -> list[BAKernelResult]:
    return compute_all_entities()


# ── Entity Catalog ────────────────────────────────────────────────


class TestEntityCatalog:
    def test_entity_count(self):
        assert len(BA_ENTITIES) == 35

    def test_channel_count(self):
        assert N_BA_CHANNELS == 8
        assert len(BA_CHANNELS) == 8

    def test_all_categories_present(self):
        cats = {e.category for e in BA_ENTITIES}
        assert cats == {
            "cortical",
            "subcortical",
            "limbic",
            "brainstem",
            "cerebellar",
            "white_matter",
            "specialized",
        }

    def test_category_counts(self):
        from collections import Counter

        counts = Counter(e.category for e in BA_ENTITIES)
        for cat in counts:
            assert counts[cat] == 5, f"{cat} has {counts[cat]} entities, expected 5"

    @pytest.mark.parametrize("entity", BA_ENTITIES, ids=lambda e: e.name)
    def test_trace_vector_shape(self, entity):
        assert entity.trace_vector().shape == (8,)

    @pytest.mark.parametrize("entity", BA_ENTITIES, ids=lambda e: e.name)
    def test_trace_vector_bounds(self, entity):
        c = entity.trace_vector()
        assert np.all(c >= 0.0) and np.all(c <= 1.0)

    @pytest.mark.parametrize("entity", BA_ENTITIES, ids=lambda e: e.name)
    def test_unique_names(self, entity):
        names = [e.name for e in BA_ENTITIES]
        assert names.count(entity.name) == 1


# ── Tier-1 Identities ────────────────────────────────────────────


class TestTier1Identities:
    @pytest.mark.parametrize("entity", BA_ENTITIES, ids=lambda e: e.name)
    def test_duality_identity(self, entity):
        r = compute_ba_kernel(entity)
        assert abs(r.F + r.omega - 1.0) < 1e-12

    @pytest.mark.parametrize("entity", BA_ENTITIES, ids=lambda e: e.name)
    def test_integrity_bound(self, entity):
        r = compute_ba_kernel(entity)
        assert r.IC <= r.F + 1e-12

    @pytest.mark.parametrize("entity", BA_ENTITIES, ids=lambda e: e.name)
    def test_log_integrity_relation(self, entity):
        r = compute_ba_kernel(entity)
        assert abs(r.IC - math.exp(r.kappa)) < 1e-10

    @pytest.mark.parametrize("entity", BA_ENTITIES, ids=lambda e: e.name)
    def test_entropy_non_negative(self, entity):
        r = compute_ba_kernel(entity)
        assert r.S >= -1e-12

    @pytest.mark.parametrize("entity", BA_ENTITIES, ids=lambda e: e.name)
    def test_curvature_bounded(self, entity):
        r = compute_ba_kernel(entity)
        assert 0.0 <= r.C <= 1.0 + 1e-12


# ── Theorems ──────────────────────────────────────────────────────


class TestTheorems:
    def test_t_ba_1(self, all_results):
        assert verify_t_ba_1(all_results)["passed"]

    def test_t_ba_2(self, all_results):
        assert verify_t_ba_2(all_results)["passed"]

    def test_t_ba_3(self, all_results):
        assert verify_t_ba_3(all_results)["passed"]

    def test_t_ba_4(self, all_results):
        assert verify_t_ba_4(all_results)["passed"]

    def test_t_ba_5(self, all_results):
        assert verify_t_ba_5(all_results)["passed"]

    def test_t_ba_6(self, all_results):
        assert verify_t_ba_6(all_results)["passed"]

    def test_all_theorems_pass(self):
        for t in verify_all_theorems():
            assert t["passed"], f"{t['name']} failed: {t}"


# ── Regime Classification ─────────────────────────────────────────


class TestRegimeClassification:
    @pytest.mark.parametrize("entity", BA_ENTITIES, ids=lambda e: e.name)
    def test_regime_is_valid(self, entity):
        r = compute_ba_kernel(entity)
        assert r.regime in ("Stable", "Watch", "Collapse")

    def test_thalamus_is_watch(self, all_results):
        thal = next(r for r in all_results if r.name == "thalamus")
        assert thal.regime == "Watch"

    def test_no_stable_entities(self, all_results):
        """No brain structure achieves Stable — anatomical heterogeneity
        across 8 channels prevents all gates from being simultaneously met."""
        stable = [r for r in all_results if r.regime == "Stable"]
        assert len(stable) == 0


# ── Systems ───────────────────────────────────────────────────────


class TestAtlasComparisonSystem:
    def test_atlas_count(self):
        comp = atlas_comparison_system()
        assert len(comp["atlases"]) == 10

    def test_holiatlas_resolution_advantage(self):
        comp = atlas_comparison_system()
        # HoliAtlas 8× better resolution than MNI152
        mni_gain = comp["holiatlas_advantages"]["MNI152"]["resolution_gain"]
        assert abs(mni_gain - 8.0) < 0.01

    def test_holiatlas_label_advantage(self):
        comp = atlas_comparison_system()
        aal_gain = comp["holiatlas_advantages"]["AAL"]["label_gain"]
        assert aal_gain > 3.0  # 350 / 90

    def test_holiatlas_multimodal_advantage(self):
        comp = atlas_comparison_system()
        mni_mod = comp["holiatlas_advantages"]["MNI152"]["modality_gain"]
        assert abs(mni_mod - 3.0) < 0.01  # 3 modalities vs 1


class TestSegmentationProtocolSystem:
    def test_protocol_count(self):
        seg = segmentation_protocol_system()
        assert seg["n_protocols"] == 7

    def test_total_regions(self):
        seg = segmentation_protocol_system()
        assert seg["total_regions"] == 315

    def test_mean_dice(self):
        seg = segmentation_protocol_system()
        assert 0.80 < seg["mean_dice"] < 0.90

    def test_pbrain_highest_dice(self):
        seg = segmentation_protocol_system()
        pbrain = seg["protocols"]["pBrain"]
        assert pbrain["avg_dice"] == 0.89


class TestClinicalDiagnosticSystem:
    def test_disease_count(self):
        clin = clinical_diagnostic_system()
        assert clin["n_diseases"] == 4

    def test_alzheimers_targets_hippocampus(self):
        clin = clinical_diagnostic_system()
        ad = clin["disease_targets"]["alzheimers"]
        assert "hippocampus_ca1" in ad["primary_targets"]
        assert "hippocampus_ca23" in ad["primary_targets"]

    def test_parkinsons_targets_substantia_nigra(self):
        clin = clinical_diagnostic_system()
        pd = clin["disease_targets"]["parkinsons"]
        assert "substantia_nigra" in pd["primary_targets"]

    def test_ms_targets_white_matter(self):
        clin = clinical_diagnostic_system()
        ms = clin["disease_targets"]["multiple_sclerosis"]
        assert "corpus_callosum" in ms["primary_targets"]


# ── Metadata ──────────────────────────────────────────────────────


class TestMetadata:
    def test_holiatlas_resolution(self):
        assert HOLIATLAS_METADATA["resolution_mm3"] == 0.125

    def test_holiatlas_labels(self):
        assert HOLIATLAS_METADATA["multiscale_labels"]["substructure"] == 350

    def test_holiatlas_modalities(self):
        assert HOLIATLAS_METADATA["modalities"] == ["T1w", "T2w", "WMn"]

    def test_holiatlas_subjects(self):
        assert HOLIATLAS_METADATA["n_subjects"] == 75

    def test_holiatlas_voxel_size(self):
        assert HOLIATLAS_METADATA["voxel_size_mm"] == [0.5, 0.5, 0.5]

    def test_nextbrain_regions(self):
        assert NEXTBRAIN_METADATA["n_regions"] == 333

    def test_nextbrain_resolution(self):
        assert NEXTBRAIN_METADATA["resolution_um"] == 100

    def test_nextbrain_brains(self):
        assert NEXTBRAIN_METADATA["n_postmortem_brains"] == 5

    def test_nextbrain_validation(self):
        assert NEXTBRAIN_METADATA["validation_scans"] == 3000


# ── Kernel Result ─────────────────────────────────────────────────


class TestKernelResult:
    def test_to_dict(self, all_results):
        d = all_results[0].to_dict()
        assert "name" in d
        assert "F" in d
        assert "omega" in d
        assert "IC" in d
        assert "regime" in d

    def test_all_results_count(self, all_results):
        assert len(all_results) == 35

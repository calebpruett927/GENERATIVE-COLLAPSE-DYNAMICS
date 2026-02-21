"""Tests for the bioactive compounds database.

Covers closures/materials_science/bioactive_compounds_database.py which
encodes essential oils, short-chain fatty acids, LAB metabolites, and
therapeutic biologics from MDPI BLSF and Biologics journals, mapped
through the GCD kernel to Tier-1 invariants.

Test structure follows the manifold layering:
  Layer 0 — Import & construction (modules loadable, data well-formed)
  Layer 1 — Kernel identities (F + ω = 1, IC ≤ F for all entries)
  Layer 2 — Database consistency (lookups, filters, enums)
  Layer 3 — Science consistency (scale gradient, natural vs recombinant)
  Layer 4 — Edge cases & validation

References:
  Papantzikos et al. (2025) DOI: 10.3390/blsf2025054001
  Ezzaky et al. (2026)      DOI: 10.3390/blsf2026056004
  Zolfanelli et al. (2026)  DOI: 10.3390/blsf2026056005
  Suzuki et al. (2026)      DOI: 10.3390/biologics6010004
  Di Benedetto et al. (2026) DOI: 10.3390/biologics6010005
"""

from __future__ import annotations

import math

import pytest

from closures.materials_science.bioactive_compounds_database import (
    COMPOUNDS,
    BioactiveCompound,
    BioactiveKernelResult,
    BioactivityType,
    CompoundClass,
    SourceType,
    TargetOrganism,
    analyze_compound_classes,
    analyze_natural_vs_recombinant,
    analyze_scale_gradient,
    build_trace,
    compute_all_bioactive_kernels,
    compute_bioactive_kernel,
    get_biologics,
    get_compound,
    get_compounds_by_bioactivity,
    get_compounds_by_class,
    get_compounds_by_journal,
    get_compounds_by_source,
    get_essential_oils,
    get_scfas,
    validate_database,
)

# ═══════════════════════════════════════════════════════════════════
#  Layer 0 — Import & Construction
# ═══════════════════════════════════════════════════════════════════


class TestBioactiveImports:
    """Verify bioactive compounds database imports and basic structure."""

    def test_compounds_tuple_nonempty(self) -> None:
        assert len(COMPOUNDS) >= 12

    def test_compound_is_frozen_dataclass(self) -> None:
        c = COMPOUNDS[0]
        assert isinstance(c, BioactiveCompound)
        with pytest.raises(AttributeError):
            c.name = "modified"  # type: ignore[misc]

    def test_all_compounds_have_required_fields(self) -> None:
        for c in COMPOUNDS:
            assert c.name, "Compound missing name"
            assert c.formula, "Compound missing formula"
            assert c.molecular_weight > 0, f"{c.name}: MW must be > 0"
            assert isinstance(c.compound_class, CompoundClass)
            assert isinstance(c.bioactivity, BioactivityType)
            assert isinstance(c.source_type, SourceType)
            assert isinstance(c.primary_target, TargetOrganism)

    def test_all_compounds_have_provenance(self) -> None:
        for c in COMPOUNDS:
            assert c.source_article, f"{c.name}: missing source_article"
            assert c.source_journal, f"{c.name}: missing source_journal"

    def test_enum_values_valid(self) -> None:
        assert len(CompoundClass) >= 7
        assert len(BioactivityType) >= 7
        assert len(SourceType) >= 4
        assert len(TargetOrganism) >= 7

    def test_kernel_result_is_namedtuple(self) -> None:
        kr = compute_bioactive_kernel(COMPOUNDS[0])
        assert isinstance(kr, BioactiveKernelResult)
        assert hasattr(kr, "F")
        assert hasattr(kr, "omega")
        assert hasattr(kr, "IC")
        assert hasattr(kr, "kappa")
        assert hasattr(kr, "regime")

    def test_build_trace_returns_8_channels(self) -> None:
        for c in COMPOUNDS:
            trace = build_trace(c)
            assert len(trace) == 8, f"{c.name}: trace has {len(trace)} channels"

    def test_trace_channels_in_open_unit_interval(self) -> None:
        for c in COMPOUNDS:
            trace = build_trace(c)
            for i, val in enumerate(trace):
                assert 0.0 < val < 1.0, f"{c.name}: ch{i} = {val} not in (0,1)"


# ═══════════════════════════════════════════════════════════════════
#  Layer 1 — Kernel Identities (Tier-1)
# ═══════════════════════════════════════════════════════════════════


class TestBioactiveKernelIdentities:
    """Verify Tier-1 structural identities for all bioactive compounds."""

    @pytest.fixture(params=list(range(len(COMPOUNDS))), ids=[c.name for c in COMPOUNDS])
    def kernel_result(self, request: pytest.FixtureRequest) -> BioactiveKernelResult:
        return compute_bioactive_kernel(COMPOUNDS[request.param])

    def test_duality_identity(self, kernel_result: BioactiveKernelResult) -> None:
        """F + ω = 1 exactly (duality identity, not unitarity)."""
        residual = abs(kernel_result.F + kernel_result.omega - 1.0)
        assert residual < 1e-12, f"{kernel_result.name}: |F + ω − 1| = {residual:.2e}"

    def test_integrity_bound(self, kernel_result: BioactiveKernelResult) -> None:
        """IC ≤ F (integrity bound, not AM-GM)."""
        assert kernel_result.IC <= kernel_result.F + 1e-12, (
            f"{kernel_result.name}: IC ({kernel_result.IC:.6f}) > F ({kernel_result.F:.6f})"
        )

    def test_fidelity_in_unit_interval(self, kernel_result: BioactiveKernelResult) -> None:
        """F ∈ [0, 1]."""
        assert 0.0 <= kernel_result.F <= 1.0

    def test_drift_in_unit_interval(self, kernel_result: BioactiveKernelResult) -> None:
        """ω ∈ [0, 1]."""
        assert 0.0 <= kernel_result.omega <= 1.0

    def test_entropy_nonnegative(self, kernel_result: BioactiveKernelResult) -> None:
        """Bernoulli field entropy S ≥ 0."""
        assert kernel_result.S >= -1e-12

    def test_curvature_bounded(self, kernel_result: BioactiveKernelResult) -> None:
        """C ∈ [0, 1]."""
        assert 0.0 <= kernel_result.C <= 1.0 + 1e-12

    def test_ic_positive(self, kernel_result: BioactiveKernelResult) -> None:
        """IC > 0."""
        assert kernel_result.IC > 0.0

    def test_kappa_nonpositive(self, kernel_result: BioactiveKernelResult) -> None:
        """κ ≤ 0 (since all c_i ∈ (0,1))."""
        assert kernel_result.kappa <= 1e-12

    def test_ic_equals_exp_kappa(self, kernel_result: BioactiveKernelResult) -> None:
        """IC ≈ exp(κ) (log-integrity relation)."""
        expected = math.exp(kernel_result.kappa)
        assert abs(kernel_result.IC - expected) < 1e-12, (
            f"{kernel_result.name}: IC ({kernel_result.IC}) != exp(κ) ({expected})"
        )

    def test_regime_valid(self, kernel_result: BioactiveKernelResult) -> None:
        """Regime is one of Stable / Watch / Collapse."""
        assert kernel_result.regime in {"Stable", "Watch", "Collapse"}

    def test_regime_consistent_with_gates(self, kernel_result: BioactiveKernelResult) -> None:
        """Regime classification matches frozen contract gates."""
        kr = kernel_result
        if kr.omega >= 0.30:
            assert kr.regime == "Collapse"
        elif kr.omega < 0.038 and kr.F > 0.90 and kr.S < 0.15 and kr.C < 0.14:
            assert kr.regime == "Stable"
        else:
            assert kr.regime == "Watch"


class TestBioactiveBulkKernels:
    """Bulk kernel properties across all compounds."""

    def test_compute_all_returns_correct_count(self) -> None:
        results = compute_all_bioactive_kernels()
        assert len(results) == len(COMPOUNDS)

    def test_no_nan_in_bulk_results(self) -> None:
        for kr in compute_all_bioactive_kernels():
            assert not math.isnan(kr.F), f"{kr.name}: F is NaN"
            assert not math.isnan(kr.omega), f"{kr.name}: ω is NaN"
            assert not math.isnan(kr.IC), f"{kr.name}: IC is NaN"
            assert not math.isnan(kr.S), f"{kr.name}: S is NaN"
            assert not math.isnan(kr.C), f"{kr.name}: C is NaN"

    def test_heterogeneity_gap_nonnegative(self) -> None:
        """Δ = F − IC ≥ 0 for all compounds."""
        for kr in compute_all_bioactive_kernels():
            assert kr.F - kr.IC >= -1e-12, f"{kr.name}: Δ < 0"


# ═══════════════════════════════════════════════════════════════════
#  Layer 2 — Database Consistency
# ═══════════════════════════════════════════════════════════════════


class TestBioactiveLookups:
    """Verify compound lookup and filter functions."""

    def test_get_compound_case_insensitive(self) -> None:
        c = get_compound("carvacrol")
        assert c is not None
        assert c.name == "Carvacrol"

    def test_get_compound_by_exact_name(self) -> None:
        c = get_compound("Thymol")
        assert c is not None
        assert c.formula == "C₁₀H₁₄O"

    def test_get_compound_returns_none_for_unknown(self) -> None:
        assert get_compound("NonexistentCompound") is None

    def test_get_compounds_by_class_monoterpene_phenol(self) -> None:
        results = get_compounds_by_class(CompoundClass.MONOTERPENE_PHENOL)
        assert len(results) >= 2  # Carvacrol, Thymol
        for c in results:
            assert c.compound_class == CompoundClass.MONOTERPENE_PHENOL

    def test_get_compounds_by_bioactivity_antifungal(self) -> None:
        results = get_compounds_by_bioactivity(BioactivityType.ANTIFUNGAL)
        assert len(results) >= 3  # Carvacrol, Thymol, p-Cymene, γ-Terpinene
        for c in results:
            assert c.bioactivity == BioactivityType.ANTIFUNGAL

    def test_get_compounds_by_bioactivity_anticancer(self) -> None:
        results = get_compounds_by_bioactivity(BioactivityType.ANTICANCER)
        assert len(results) >= 3  # Butyrate, acetate, propionate
        for c in results:
            assert c.bioactivity == BioactivityType.ANTICANCER

    def test_get_compounds_by_source_plant(self) -> None:
        results = get_compounds_by_source(SourceType.PLANT_ESSENTIAL_OIL)
        assert len(results) >= 4  # Carvacrol, Thymol, p-Cymene, γ-Terpinene
        for c in results:
            assert c.source_type == SourceType.PLANT_ESSENTIAL_OIL

    def test_get_compounds_by_journal_blsf(self) -> None:
        blsf = get_compounds_by_journal("BLSF")
        assert len(blsf) >= 8

    def test_get_compounds_by_journal_biologics(self) -> None:
        bio = get_compounds_by_journal("Biologics")
        assert len(bio) >= 3

    def test_get_essential_oils(self) -> None:
        oils = get_essential_oils()
        assert len(oils) >= 4
        for c in oils:
            assert c.source_type == SourceType.PLANT_ESSENTIAL_OIL

    def test_get_scfas(self) -> None:
        scfas = get_scfas()
        assert len(scfas) >= 3  # butyrate, acetate, propionate
        for c in scfas:
            assert c.compound_class == CompoundClass.SHORT_CHAIN_FATTY_ACID

    def test_get_biologics(self) -> None:
        biol = get_biologics()
        assert len(biol) >= 3  # IL-6, Dupilumab, Cerliponase alfa
        for c in biol:
            assert c.compound_class in (
                CompoundClass.MONOCLONAL_ANTIBODY,
                CompoundClass.RECOMBINANT_ENZYME,
                CompoundClass.CYTOKINE,
            )

    def test_to_dict_roundtrip(self) -> None:
        c = COMPOUNDS[0]
        d = c.to_dict()
        assert d["name"] == c.name
        assert d["molecular_weight"] == c.molecular_weight
        assert d["compound_class"] == c.compound_class.value
        assert isinstance(d, dict)


# ═══════════════════════════════════════════════════════════════════
#  Layer 3 — Science Consistency
# ═══════════════════════════════════════════════════════════════════


class TestBioactiveScience:
    """Verify scientific consistency of compound properties and kernel behavior."""

    # ── Molecular weight ordering ────────────────────────────────
    def test_essential_oils_lighter_than_biologics(self) -> None:
        """Small molecules (EOs, SCFAs) << macromolecules (mAbs, enzymes)."""
        oils = get_essential_oils()
        biol = get_biologics()
        max_oil_mw = max(c.molecular_weight for c in oils)
        min_bio_mw = min(c.molecular_weight for c in biol)
        assert max_oil_mw < min_bio_mw, "EOs should be lighter than biologics"

    def test_scfa_mw_ordering(self) -> None:
        """Acetic acid < propionic acid < butyric acid by MW."""
        acetic = get_compound("Acetic acid (acetate)")
        propionic = get_compound("Propionic acid (propionate)")
        butyric = get_compound("Butyric acid (sodium butyrate)")
        assert acetic is not None and propionic is not None and butyric is not None
        assert acetic.molecular_weight < propionic.molecular_weight < butyric.molecular_weight

    def test_carvacrol_thymol_isomers(self) -> None:
        """Carvacrol and thymol are structural isomers — same formula, same MW."""
        carv = get_compound("Carvacrol")
        thym = get_compound("Thymol")
        assert carv is not None and thym is not None
        assert carv.formula == thym.formula
        assert carv.molecular_weight == thym.molecular_weight

    # ── Lipophilicity ordering ───────────────────────────────────
    def test_terpenes_more_lipophilic_than_scfas(self) -> None:
        """Terpenes (logP ~3-4.5) >> SCFAs (logP ~ -0.8 to 0.3)."""
        oils = get_essential_oils()
        scfas = get_scfas()
        for oil in oils:
            if oil.logP is not None:
                for scfa in scfas:
                    if scfa.logP is not None:
                        assert oil.logP > scfa.logP

    # ── Kernel behavior patterns ─────────────────────────────────
    def test_natural_vs_recombinant_analysis_runs(self) -> None:
        result = analyze_natural_vs_recombinant()
        assert result["natural_count"] >= 8
        assert result["recombinant_count"] >= 3
        assert 0.0 < result["natural_mean_F"] < 1.0
        assert 0.0 < result["recombinant_mean_F"] < 1.0

    def test_compound_class_analysis_runs(self) -> None:
        stats = analyze_compound_classes()
        assert len(stats) >= 5  # At least 5 distinct classes
        for _cls_name, data in stats.items():
            assert data["count"] >= 1
            assert 0.0 < data["mean_F"] < 1.0
            assert data["mean_heterogeneity_gap"] >= -1e-12

    def test_scale_gradient_analysis_runs(self) -> None:
        result = analyze_scale_gradient()
        assert result["small_molecule_count"] >= 8
        assert result["large_molecule_count"] >= 3
        assert len(result["compounds_by_MW"]) == len(COMPOUNDS)

    def test_scale_gradient_sorted_by_mw(self) -> None:
        result = analyze_scale_gradient()
        mws = [r["MW"] for r in result["compounds_by_MW"]]
        assert mws == sorted(mws)

    def test_heterogeneity_gap_larger_for_biologics(self) -> None:
        """Biologics should have larger Δ = F − IC due to extreme channel
        heterogeneity (very high MW, very high complexity, low naturalness)."""
        result = analyze_scale_gradient()
        # Large molecules have channels with more extreme spread
        assert result["large_mean_gap"] > 0.0

    # ── Specific compound kernel checks ──────────────────────────
    def test_carvacrol_high_potency(self) -> None:
        """Carvacrol: 100% Fusarium inhibition → high potency channel."""
        carv = get_compound("Carvacrol")
        assert carv is not None
        trace = build_trace(carv)
        assert trace[1] > 0.9  # Potency channel near max

    def test_butyrate_anticancer_potency(self) -> None:
        """Butyrate: ~52% tumor reduction → moderate-high potency."""
        but = get_compound("Butyric acid (sodium butyrate)")
        assert but is not None
        trace = build_trace(but)
        assert 0.3 < trace[1] < 0.7  # ~52% of 100%

    def test_il6_in_watch_or_collapse(self) -> None:
        """IL-6 has extreme heterogeneity — should not be Stable."""
        il6 = get_compound("Interleukin-6 (IL-6)")
        assert il6 is not None
        kr = compute_bioactive_kernel(il6)
        assert kr.regime in ("Watch", "Collapse")

    def test_dupilumab_largest_mw(self) -> None:
        """Dupilumab (~147 kDa) is the largest compound in the database."""
        dups = get_compound("Dupilumab")
        assert dups is not None
        max_mw = max(c.molecular_weight for c in COMPOUNDS)
        assert dups.molecular_weight == max_mw


# ═══════════════════════════════════════════════════════════════════
#  Layer 4 — Edge Cases & Validation
# ═══════════════════════════════════════════════════════════════════


class TestBioactiveValidation:
    """Validate database integrity and edge cases."""

    def test_validate_database_conformant(self) -> None:
        result = validate_database()
        assert result["status"] == "CONFORMANT", f"Errors: {result['errors']}"
        assert result["total_compounds"] == len(COMPOUNDS)
        assert len(result["errors"]) == 0

    def test_no_duplicate_names(self) -> None:
        names = [c.name for c in COMPOUNDS]
        assert len(names) == len(set(names)), f"Duplicate names: {[n for n in names if names.count(n) > 1]}"

    def test_all_cas_numbers_unique_when_present(self) -> None:
        cas_nums = [c.cas_number for c in COMPOUNDS if c.cas_number is not None]
        assert len(cas_nums) == len(set(cas_nums))

    def test_safety_index_in_unit_interval(self) -> None:
        for c in COMPOUNDS:
            assert 0.0 <= c.safety_index <= 1.0, f"{c.name}: safety_index out of range"

    def test_proteins_have_no_bp(self) -> None:
        """Protein compounds should not have boiling points."""
        for c in COMPOUNDS:
            if c.compound_class in (
                CompoundClass.CYTOKINE,
                CompoundClass.MONOCLONAL_ANTIBODY,
                CompoundClass.RECOMBINANT_ENZYME,
            ):
                assert c.boiling_point_K is None, f"{c.name}: proteins should not have bp"

    def test_proteins_have_no_logP(self) -> None:
        """Protein compounds should not have logP."""
        for c in COMPOUNDS:
            if c.compound_class in (
                CompoundClass.CYTOKINE,
                CompoundClass.MONOCLONAL_ANTIBODY,
                CompoundClass.RECOMBINANT_ENZYME,
            ):
                assert c.logP is None, f"{c.name}: proteins should not have logP"

    def test_small_molecules_have_bp_or_mp(self) -> None:
        """Small molecules should have at least one thermal measurement."""
        for c in COMPOUNDS:
            if c.molecular_weight < 1000:
                assert c.boiling_point_K is not None or c.melting_point_K is not None, (
                    f"{c.name}: small molecule missing thermal data"
                )

    def test_blsf_articles_have_dois(self) -> None:
        for c in COMPOUNDS:
            if c.source_journal == "BLSF":
                assert c.source_article.startswith("10.3390/blsf"), f"{c.name}: BLSF article should have BLSF DOI"

    def test_biologics_articles_have_dois(self) -> None:
        for c in COMPOUNDS:
            if c.source_journal == "Biologics":
                assert c.source_article.startswith("10.3390/biologics"), (
                    f"{c.name}: Biologics article should have Biologics DOI"
                )


class TestBioactiveEdgeCases:
    """Edge case handling for the bioactive compounds kernel."""

    def test_extreme_mw_range(self) -> None:
        """Database covers ~60 Da (acetate) to ~147 kDa (dupilumab) — 2400× range."""
        mws = [c.molecular_weight for c in COMPOUNDS]
        ratio = max(mws) / min(mws)
        assert ratio > 1000, f"MW range ratio {ratio} < 1000"

    def test_all_journals_covered(self) -> None:
        """Both BLSF and Biologics journals are represented."""
        journals = {c.source_journal for c in COMPOUNDS}
        assert "BLSF" in journals
        assert "Biologics" in journals

    def test_import_from_init(self) -> None:
        """compute_bioactive_kernel accessible from __init__."""
        from closures.materials_science import compute_bioactive_kernel as fn

        assert callable(fn)
        kr = fn(COMPOUNDS[0])
        assert isinstance(kr, BioactiveKernelResult)

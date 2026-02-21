"""Tests for the photonic materials database.

Covers closures/materials_science/photonic_materials_database.py which
encodes waveguides, resonators, lasers, and metasurfaces from the MDPI
Photonics journal, mapped through the GCD kernel to Tier-1 invariants.

Test structure follows the manifold layering:
  Layer 0 — Import & construction (modules loadable, data well-formed)
  Layer 1 — Kernel identities (F + ω = 1, IC ≤ F for all entries)
  Layer 2 — Database consistency (lookups, enums, serialization)
  Layer 3 — Science consistency (spectral ordering, Q-factor hierarchy)
  Layer 4 — Edge cases & validation

References:
  Li et al. (2026)       DOI: 10.3390/photonics13020175
  Fan et al. (2026)      DOI: 10.3390/photonics13020199
  Wang et al. (2026)     DOI: 10.3390/photonics13020200
  Li, Zhang, Lu (2026)   DOI: 10.3390/photonics13020174
  Amoateng et al. (2026) DOI: 10.3390/photonics13020195
"""

from __future__ import annotations

import math

import pytest

from closures.materials_science.photonic_materials_database import (
    MATERIALS,
    ApplicationField,
    DeviceType,
    MaterialPlatform,
    PhotonicKernelResult,
    PhotonicMaterial,
    SpectralBand,
    analyze_device_types,
    analyze_platform_landscape,
    analyze_spectral_bands,
    build_trace,
    compute_all_photonic_kernels,
    compute_photonic_kernel,
)

# ═══════════════════════════════════════════════════════════════════
#  Layer 0 — Import & Construction
# ═══════════════════════════════════════════════════════════════════


class TestPhotonicImports:
    """Verify photonic materials database imports and basic structure."""

    def test_materials_tuple_nonempty(self) -> None:
        assert len(MATERIALS) >= 14

    def test_material_is_frozen_dataclass(self) -> None:
        m = MATERIALS[0]
        assert isinstance(m, PhotonicMaterial)
        with pytest.raises(AttributeError):
            m.name = "foo"  # type: ignore[misc]

    def test_all_materials_have_required_fields(self) -> None:
        for m in MATERIALS:
            assert m.name, "Missing name"
            assert m.material_formula, f"{m.name}: missing formula"
            assert isinstance(m.device_type, DeviceType)
            assert isinstance(m.platform, MaterialPlatform)
            assert isinstance(m.spectral_band, SpectralBand)
            assert isinstance(m.application, ApplicationField)
            assert m.operating_wavelength_nm > 0
            assert m.refractive_index >= 1.0
            assert m.optical_loss_dB_cm > 0 or m.optical_loss_dB_cm == 0

    def test_all_materials_have_provenance(self) -> None:
        for m in MATERIALS:
            assert m.source_article, f"{m.name}: missing source_article"
            assert m.source_journal == "Photonics"

    def test_enum_values_valid(self) -> None:
        for m in MATERIALS:
            assert m.device_type in DeviceType
            assert m.platform in MaterialPlatform
            assert m.spectral_band in SpectralBand
            assert m.application in ApplicationField

    def test_kernel_result_is_namedtuple(self) -> None:
        kr = compute_photonic_kernel(MATERIALS[0])
        assert isinstance(kr, PhotonicKernelResult)
        assert hasattr(kr, "F")
        assert hasattr(kr, "omega")
        assert hasattr(kr, "IC")

    def test_build_trace_returns_8_channels(self) -> None:
        for m in MATERIALS:
            trace = build_trace(m)
            assert len(trace) == 8, f"{m.name}: trace has {len(trace)} channels"

    def test_trace_channels_in_open_unit_interval(self) -> None:
        epsilon = 1e-8
        for m in MATERIALS:
            trace = build_trace(m)
            for i, c in enumerate(trace):
                assert epsilon <= c <= 1.0 - epsilon, f"{m.name}: channel {i} = {c} outside (ε, 1-ε)"


# ═══════════════════════════════════════════════════════════════════
#  Layer 1 — Kernel Identities (Tier-1 invariants on all materials)
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture(params=range(len(MATERIALS)), ids=[m.name for m in MATERIALS])
def kernel_result(request: pytest.FixtureRequest) -> PhotonicKernelResult:
    """Parametrized fixture yielding kernel result for each material."""
    return compute_photonic_kernel(MATERIALS[request.param])


class TestPhotonicKernelIdentities:
    """Verify Tier-1 kernel identities for every photonic material."""

    def test_duality_identity(self, kernel_result: PhotonicKernelResult) -> None:
        """F + ω = 1 (duality identity) to machine precision."""
        kr = kernel_result
        assert abs(kr.F + kr.omega - 1.0) < 1e-12, f"{kr.name}: F+ω={kr.F + kr.omega}"

    def test_integrity_bound(self, kernel_result: PhotonicKernelResult) -> None:
        """IC ≤ F (integrity bound) for all materials."""
        kr = kernel_result
        assert kr.IC <= kr.F + 1e-12, f"{kr.name}: IC={kr.IC} > F={kr.F}"

    def test_fidelity_in_unit_interval(self, kernel_result: PhotonicKernelResult) -> None:
        kr = kernel_result
        assert 0.0 <= kr.F <= 1.0, f"{kr.name}: F={kr.F}"

    def test_drift_in_unit_interval(self, kernel_result: PhotonicKernelResult) -> None:
        kr = kernel_result
        assert 0.0 <= kr.omega <= 1.0, f"{kr.name}: ω={kr.omega}"

    def test_entropy_nonnegative(self, kernel_result: PhotonicKernelResult) -> None:
        kr = kernel_result
        assert kr.S >= -1e-12, f"{kr.name}: S={kr.S}"

    def test_curvature_bounded(self, kernel_result: PhotonicKernelResult) -> None:
        kr = kernel_result
        assert 0.0 <= kr.C <= 1.0, f"{kr.name}: C={kr.C}"

    def test_ic_positive(self, kernel_result: PhotonicKernelResult) -> None:
        kr = kernel_result
        assert kr.IC > 0.0, f"{kr.name}: IC={kr.IC}"

    def test_kappa_nonpositive(self, kernel_result: PhotonicKernelResult) -> None:
        kr = kernel_result
        assert kr.kappa <= 1e-12, f"{kr.name}: κ={kr.kappa}"

    def test_ic_equals_exp_kappa(self, kernel_result: PhotonicKernelResult) -> None:
        """IC = exp(κ) (log-integrity relation)."""
        kr = kernel_result
        assert abs(kr.IC - math.exp(kr.kappa)) < 1e-12, f"{kr.name}: IC={kr.IC}, exp(κ)={math.exp(kr.kappa)}"

    def test_regime_valid(self, kernel_result: PhotonicKernelResult) -> None:
        kr = kernel_result
        assert kr.regime in ("Stable", "Watch", "Collapse")

    def test_regime_consistent_with_gates(self, kernel_result: PhotonicKernelResult) -> None:
        """Regime classification matches frozen contract gates."""
        kr = kernel_result
        if kr.omega >= 0.30:
            assert kr.regime == "Collapse"
        elif kr.omega < 0.038 and kr.F > 0.90 and kr.S < 0.15 and kr.C < 0.14:
            assert kr.regime == "Stable"
        else:
            assert kr.regime == "Watch"


class TestPhotonicBulkKernels:
    """Bulk kernel properties across all materials."""

    def test_compute_all_returns_correct_count(self) -> None:
        results = compute_all_photonic_kernels()
        assert len(results) == len(MATERIALS)

    def test_no_nan_in_bulk_results(self) -> None:
        for kr in compute_all_photonic_kernels():
            assert not math.isnan(kr.F), f"{kr.name}: F is NaN"
            assert not math.isnan(kr.omega), f"{kr.name}: ω is NaN"
            assert not math.isnan(kr.IC), f"{kr.name}: IC is NaN"
            assert not math.isnan(kr.S), f"{kr.name}: S is NaN"
            assert not math.isnan(kr.C), f"{kr.name}: C is NaN"

    def test_heterogeneity_gap_nonnegative(self) -> None:
        """Δ = F − IC ≥ 0 for all materials."""
        for kr in compute_all_photonic_kernels():
            assert kr.F - kr.IC >= -1e-12, f"{kr.name}: Δ < 0"


# ═══════════════════════════════════════════════════════════════════
#  Layer 2 — Database Consistency
# ═══════════════════════════════════════════════════════════════════


class TestPhotonicDatabaseConsistency:
    """Verify data consistency and serialization."""

    def test_no_duplicate_names(self) -> None:
        names = [m.name for m in MATERIALS]
        assert len(names) == len(set(names)), f"Duplicate names: {[n for n in names if names.count(n) > 1]}"

    def test_to_dict_roundtrip(self) -> None:
        m = MATERIALS[0]
        d = m.to_dict()
        assert d["name"] == m.name
        assert d["refractive_index"] == m.refractive_index
        assert d["device_type"] == m.device_type.value
        assert isinstance(d, dict)

    def test_all_dois_are_valid_format(self) -> None:
        for m in MATERIALS:
            assert m.source_article.startswith("10.3390/photonics"), (
                f"{m.name}: DOI '{m.source_article}' not a Photonics DOI"
            )

    def test_refractive_indices_physical(self) -> None:
        """All refractive indices ≥ 1.0 (physical requirement)."""
        for m in MATERIALS:
            assert m.refractive_index >= 1.0, f"{m.name}: n={m.refractive_index}"

    def test_wavelengths_positive(self) -> None:
        for m in MATERIALS:
            assert m.operating_wavelength_nm > 0, f"{m.name}: λ={m.operating_wavelength_nm}"

    def test_footprints_positive(self) -> None:
        for m in MATERIALS:
            assert m.footprint_um2 > 0, f"{m.name}: footprint={m.footprint_um2}"

    def test_loss_nonnegative(self) -> None:
        for m in MATERIALS:
            assert m.optical_loss_dB_cm >= 0, f"{m.name}: loss={m.optical_loss_dB_cm}"

    def test_temperatures_physical(self) -> None:
        """All temperatures > 0 K."""
        for m in MATERIALS:
            assert m.max_operating_temp_K > 0, f"{m.name}: T_max={m.max_operating_temp_K}"
            assert m.process_temp_K > 0, f"{m.name}: T_proc={m.process_temp_K}"

    def test_device_types_diverse(self) -> None:
        """Database covers multiple device types."""
        types = {m.device_type for m in MATERIALS}
        assert len(types) >= 5

    def test_spectral_bands_diverse(self) -> None:
        """Database covers multiple spectral bands."""
        bands = {m.spectral_band for m in MATERIALS}
        assert len(bands) >= 4

    def test_platforms_diverse(self) -> None:
        """Database covers multiple material platforms."""
        platforms = {m.platform for m in MATERIALS}
        assert len(platforms) >= 8


# ═══════════════════════════════════════════════════════════════════
#  Layer 3 — Science Consistency
# ═══════════════════════════════════════════════════════════════════


class TestPhotonicScience:
    """Verify scientific consistency of photonic properties and kernel behavior."""

    # ── Refractive index ordering ────────────────────────────────
    def test_silicon_highest_refractive_index(self) -> None:
        """Silicon-based materials (n≈3.5) should have the highest refractive indices."""
        si_platforms = [
            m for m in MATERIALS if m.platform in (MaterialPlatform.SILICON, MaterialPlatform.III_V_SEMICONDUCTOR)
        ]
        other = [
            m for m in MATERIALS if m.platform not in (MaterialPlatform.SILICON, MaterialPlatform.III_V_SEMICONDUCTOR)
        ]
        if si_platforms and other:
            min_hi_n = min(m.refractive_index for m in si_platforms)
            max_lo_n = max(m.refractive_index for m in other)
            assert min_hi_n > max_lo_n

    def test_silica_lower_than_silicon(self) -> None:
        """SiO₂ (n≈1.45) < Si (n≈3.48)."""
        silica = [m for m in MATERIALS if m.platform == MaterialPlatform.FUSED_SILICA]
        silicon = [m for m in MATERIALS if m.platform == MaterialPlatform.SILICON]
        if silica and silicon:
            assert silica[0].refractive_index < silicon[0].refractive_index

    # ── Quality factor hierarchy ─────────────────────────────────
    def test_wgm_resonators_highest_q(self) -> None:
        """WGM resonators (Q~10⁸) >> microdisks (Q~10³)."""
        wgm = [m for m in MATERIALS if m.quality_factor and "WGM" in m.device_type.value]
        disks = [m for m in MATERIALS if m.quality_factor and "Microdisk" in m.device_type.value]
        if wgm and disks:
            max_wgm_q = max(m.quality_factor for m in wgm)  # type: ignore[arg-type]
            max_disk_q = max(m.quality_factor for m in disks)  # type: ignore[arg-type]
            assert max_wgm_q > max_disk_q

    def test_zblan_ultrahigh_q(self) -> None:
        """ZBLAN fluoride has Q ~ 5.4×10⁸ (highest in database)."""
        zblan = [m for m in MATERIALS if "ZBLAN" in m.name]
        assert len(zblan) == 1
        assert zblan[0].quality_factor is not None
        assert zblan[0].quality_factor > 1e8

    # ── Wavelength ordering by spectral band ─────────────────────
    def test_visible_shorter_than_telecom(self) -> None:
        """Visible devices (λ < 780 nm) < telecom devices (λ > 1260 nm)."""
        vis = [m for m in MATERIALS if m.spectral_band == SpectralBand.VISIBLE]
        telecom = [m for m in MATERIALS if m.spectral_band in (SpectralBand.NEAR_IR_O, SpectralBand.NEAR_IR_C)]
        if vis and telecom:
            max_vis = max(m.operating_wavelength_nm for m in vis)
            min_tel = min(m.operating_wavelength_nm for m in telecom)
            assert max_vis < min_tel

    def test_mid_ir_longer_than_telecom(self) -> None:
        """Mid-IR devices (λ > 2000 nm) > telecom devices (λ < 1600 nm)."""
        midir = [m for m in MATERIALS if m.spectral_band == SpectralBand.MID_IR]
        telecom = [m for m in MATERIALS if m.spectral_band == SpectralBand.NEAR_IR_C]
        if midir and telecom:
            min_mir = min(m.operating_wavelength_nm for m in midir)
            max_tel = max(m.operating_wavelength_nm for m in telecom)
            assert min_mir > max_tel

    def test_thz_longest_wavelength(self) -> None:
        """THz device has the longest wavelength in database."""
        thz = [m for m in MATERIALS if m.spectral_band == SpectralBand.TERAHERTZ]
        assert len(thz) >= 1
        max_wl = max(m.operating_wavelength_nm for m in MATERIALS)
        assert thz[0].operating_wavelength_nm == max_wl

    # ── Loss ordering ────────────────────────────────────────────
    def test_silica_wgm_ultralow_loss(self) -> None:
        """Silica WGM microsphere has among the lowest losses."""
        silica_wgm = [m for m in MATERIALS if "Silica WGM" in m.name]
        assert len(silica_wgm) == 1
        assert silica_wgm[0].optical_loss_dB_cm <= 0.01

    def test_perovskite_higher_loss_than_silica(self) -> None:
        """Perovskite has higher optical loss than fused silica."""
        perovskite = [m for m in MATERIALS if m.platform == MaterialPlatform.PEROVSKITE]
        silica = [m for m in MATERIALS if m.platform == MaterialPlatform.FUSED_SILICA]
        if perovskite and silica:
            assert perovskite[0].optical_loss_dB_cm > silica[0].optical_loss_dB_cm

    # ── CMOS compatibility ───────────────────────────────────────
    def test_cmos_compatible_materials(self) -> None:
        """Si and SiN platforms should be CMOS compatible."""
        cmos = [m for m in MATERIALS if m.cmos_compatible]
        platforms = {m.platform for m in cmos}
        assert MaterialPlatform.SILICON in platforms or MaterialPlatform.SILICON_NITRIDE in platforms

    def test_iii_v_not_cmos(self) -> None:
        """III-V semiconductors are not CMOS compatible."""
        iii_v = [m for m in MATERIALS if m.platform == MaterialPlatform.III_V_SEMICONDUCTOR]
        for m in iii_v:
            assert not m.cmos_compatible

    # ── Kernel behavior patterns ─────────────────────────────────
    def test_device_type_analysis_runs(self) -> None:
        stats = analyze_device_types()
        assert len(stats) >= 4
        for _dtype, data in stats.items():
            assert data["count"] >= 1
            assert 0.0 < data["mean_F"] < 1.0
            assert data["mean_heterogeneity_gap"] >= -1e-12

    def test_spectral_band_analysis_runs(self) -> None:
        stats = analyze_spectral_bands()
        assert len(stats) >= 3
        for _band, data in stats.items():
            assert data["count"] >= 1
            assert 0.0 < data["mean_F"] < 1.0

    def test_platform_landscape_runs(self) -> None:
        result = analyze_platform_landscape()
        assert result["n_materials"] == len(MATERIALS)
        assert result["n_platforms"] >= 8
        assert len(result["landscape"]) == len(MATERIALS)

    def test_platform_landscape_sorted_by_ic_f(self) -> None:
        """Platform landscape sorted descending by IC/F ratio."""
        result = analyze_platform_landscape()
        ratios = [r["IC_F_ratio"] for r in result["landscape"]]
        assert ratios == sorted(ratios, reverse=True)


# ═══════════════════════════════════════════════════════════════════
#  Layer 4 — Edge Cases & Validation
# ═══════════════════════════════════════════════════════════════════


class TestPhotonicEdgeCases:
    """Edge case handling and database validation."""

    def test_wavelength_range_spans_orders_of_magnitude(self) -> None:
        """Database covers visible (450 nm) to THz (100,000 nm) — ~220× range."""
        wls = [m.operating_wavelength_nm for m in MATERIALS]
        ratio = max(wls) / min(wls)
        assert ratio > 100, f"Wavelength range ratio {ratio} < 100"

    def test_q_factor_range_spans_orders_of_magnitude(self) -> None:
        """Q factors span ~50 (metasurface) to ~5×10⁸ (ZBLAN) — 10⁷ range."""
        qs = [m.quality_factor for m in MATERIALS if m.quality_factor is not None]
        assert len(qs) >= 10
        ratio = max(qs) / min(qs)
        assert ratio > 1e4, f"Q factor range ratio {ratio} < 10⁴"

    def test_loss_range_covers_orders_of_magnitude(self) -> None:
        """Losses span 0.0005 to 10 dB/cm — 10⁴ range."""
        losses = [m.optical_loss_dB_cm for m in MATERIALS]
        ratio = max(losses) / min(losses)
        assert ratio > 100, f"Loss range ratio {ratio} < 100"

    def test_extreme_channel_does_not_crash(self) -> None:
        """THz device with extreme wavelength still produces valid kernel."""
        thz = next(m for m in MATERIALS if m.spectral_band == SpectralBand.TERAHERTZ)
        kr = compute_photonic_kernel(thz)
        assert not math.isnan(kr.F)
        assert abs(kr.F + kr.omega - 1.0) < 1e-12
        assert kr.IC <= kr.F + 1e-12

    def test_submicron_device_does_not_crash(self) -> None:
        """Perovskite submicron sphere has tiny footprint — still valid kernel."""
        perov = [m for m in MATERIALS if m.platform == MaterialPlatform.PEROVSKITE]
        if perov:
            kr = compute_photonic_kernel(perov[0])
            assert not math.isnan(kr.F)
            assert kr.IC > 0

    def test_none_quality_factor_handled(self) -> None:
        """Materials with Q=None should produce valid trace."""
        no_q = [m for m in MATERIALS if m.quality_factor is None]
        assert len(no_q) >= 1, "Expected at least one material without Q factor"
        for m in no_q:
            trace = build_trace(m)
            assert len(trace) == 8
            assert trace[3] == 0.5  # Default Q channel value

    def test_import_from_init(self) -> None:
        """compute_photonic_kernel accessible from __init__."""
        from closures.materials_science import compute_photonic_kernel as fn

        assert callable(fn)
        kr = fn(MATERIALS[0])
        assert isinstance(kr, PhotonicKernelResult)

    def test_all_spectral_bands_represented(self) -> None:
        """Database includes visible, near-IR, mid-IR, and THz devices."""
        bands = {m.spectral_band for m in MATERIALS}
        assert SpectralBand.VISIBLE in bands
        assert SpectralBand.NEAR_IR_C in bands
        assert SpectralBand.MID_IR in bands
        assert SpectralBand.TERAHERTZ in bands

    def test_gain_and_loss_efficiency_handled(self) -> None:
        """Both positive (gain) and negative (loss) efficiency values handled."""
        gains = [m for m in MATERIALS if m.efficiency_dB is not None and m.efficiency_dB > 0]
        losses = [m for m in MATERIALS if m.efficiency_dB is not None and m.efficiency_dB < 0]
        assert len(gains) >= 1, "Expected at least one device with gain"
        assert len(losses) >= 1, "Expected at least one device with loss"

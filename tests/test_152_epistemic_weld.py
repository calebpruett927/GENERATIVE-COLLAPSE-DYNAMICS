"""
Tests for epistemic_weld.py — Seam Epistemology Module

Tests the RETURN / GESTURE / DISSOLUTION trichotomy, positional illusion
quantification, epistemic trace assessment, and seam epistemology.

These tests verify the formal epistemic structure from "The Seam of Reality"
(Paulus, 2025; DOI: 10.5281/zenodo.17619502) as implemented in the codebase.
"""

from __future__ import annotations

import pytest

from umcp.epistemic_weld import (
    VALID_VERDICTS,
    VERDICT_DESCRIPTIONS,
    EpistemicTraceMetadata,
    EpistemicVerdict,
    GestureReason,
    PositionalIllusion,
    assess_epistemic_trace,
    assess_seam_epistemology,
    classify_epistemic_act,
    diagnose_gesture,
    quantify_positional_illusion,
)
from umcp.frozen_contract import (
    EPSILON,
    P_EXPONENT,
    TOL_SEAM,
    NonconformanceType,
    Regime,
)

# =============================================================================
# EPISTEMIC VERDICT — Core Trichotomy
# =============================================================================


class TestEpistemicVerdict:
    """Test the RETURN / GESTURE / DISSOLUTION trichotomy."""

    def test_verdict_enum_values(self) -> None:
        """Verify enum values match protocol names."""
        assert EpistemicVerdict.RETURN.value == "return"
        assert EpistemicVerdict.GESTURE.value == "gesture"
        assert EpistemicVerdict.DISSOLUTION.value == "dissolution"

    def test_verdict_is_exhaustive(self) -> None:
        """Verify exactly three verdicts exist."""
        assert len(EpistemicVerdict) == 3

    def test_valid_verdicts_frozenset(self) -> None:
        """Verify VALID_VERDICTS contains all verdicts."""
        assert frozenset(EpistemicVerdict) == VALID_VERDICTS
        assert len(VALID_VERDICTS) == 3

    def test_verdict_descriptions_complete(self) -> None:
        """Every verdict has a human-readable description."""
        for v in EpistemicVerdict:
            assert v in VERDICT_DESCRIPTIONS
            assert len(VERDICT_DESCRIPTIONS[v]) > 20


# =============================================================================
# CLASSIFY EPISTEMIC ACT — The Central Function
# =============================================================================


class TestClassifyEpistemicAct:
    """Test the central classification function."""

    # --- RETURN cases ---

    def test_return_stable_seam_pass(self) -> None:
        """STABLE regime + seam pass + finite τ_R → RETURN."""
        verdict, reasons = classify_epistemic_act(seam_pass=True, tau_R=1.85, regime=Regime.STABLE)
        assert verdict == EpistemicVerdict.RETURN
        assert reasons == []

    def test_return_watch_seam_pass(self) -> None:
        """WATCH regime + seam pass + finite τ_R → RETURN."""
        verdict, reasons = classify_epistemic_act(seam_pass=True, tau_R=5.0, regime=Regime.WATCH)
        assert verdict == EpistemicVerdict.RETURN
        assert reasons == []

    def test_return_critical_seam_pass(self) -> None:
        """CRITICAL regime + seam pass → RETURN (critical is overlay)."""
        verdict, reasons = classify_epistemic_act(seam_pass=True, tau_R=10.0, regime=Regime.CRITICAL)
        assert verdict == EpistemicVerdict.RETURN
        assert reasons == []

    # --- GESTURE cases ---

    def test_gesture_seam_fail(self) -> None:
        """Seam failure → GESTURE."""
        verdict, reasons = classify_epistemic_act(
            seam_pass=False,
            tau_R=1.85,
            regime=Regime.STABLE,
            seam_failures=["|s|=0.010000 > tol_seam=0.005"],
        )
        assert verdict == EpistemicVerdict.GESTURE
        assert GestureReason.SEAM_RESIDUAL_EXCEEDED in reasons

    def test_gesture_infinite_tau_R(self) -> None:
        """τ_R = ∞ → GESTURE with NO_FINITE_RETURN."""
        verdict, reasons = classify_epistemic_act(
            seam_pass=False,
            tau_R=float("inf"),
            regime=Regime.STABLE,
            seam_failures=["τ_R=inf is not finite (INF_REC)"],
        )
        assert verdict == EpistemicVerdict.GESTURE
        assert GestureReason.NO_FINITE_RETURN in reasons

    def test_gesture_identity_mismatch(self) -> None:
        """Identity check failure → GESTURE with IDENTITY_MISMATCH."""
        verdict, reasons = classify_epistemic_act(
            seam_pass=False,
            tau_R=1.85,
            regime=Regime.STABLE,
            seam_failures=["|I_ratio - exp(Δκ)|=0.100000 >= tol_exp=1e-06"],
        )
        assert verdict == EpistemicVerdict.GESTURE
        assert GestureReason.IDENTITY_MISMATCH in reasons

    def test_gesture_multiple_failures(self) -> None:
        """Multiple failures → GESTURE with multiple reasons."""
        verdict, reasons = classify_epistemic_act(
            seam_pass=False,
            tau_R=float("inf"),
            regime=Regime.STABLE,
            seam_failures=[
                "|s|=0.010 > tol_seam=0.005",
                "τ_R=inf is not finite (INF_REC)",
            ],
        )
        assert verdict == EpistemicVerdict.GESTURE
        assert len(reasons) >= 2

    def test_gesture_generic_seam_fail(self) -> None:
        """Generic seam failure with no failure list → GESTURE."""
        verdict, reasons = classify_epistemic_act(seam_pass=False, tau_R=1.85, regime=Regime.STABLE)
        assert verdict == EpistemicVerdict.GESTURE
        assert len(reasons) >= 1

    def test_gesture_seam_pass_but_infinite_tau_R(self) -> None:
        """Even with seam_pass=True, infinite τ_R → GESTURE."""
        verdict, reasons = classify_epistemic_act(seam_pass=True, tau_R=float("inf"), regime=Regime.STABLE)
        assert verdict == EpistemicVerdict.GESTURE
        assert GestureReason.NO_FINITE_RETURN in reasons

    # --- DISSOLUTION cases ---

    def test_dissolution_collapse_regime(self) -> None:
        """COLLAPSE regime → DISSOLUTION regardless of seam."""
        verdict, reasons = classify_epistemic_act(seam_pass=True, tau_R=1.0, regime=Regime.COLLAPSE)
        assert verdict == EpistemicVerdict.DISSOLUTION
        assert reasons == []

    def test_dissolution_collapse_seam_fail(self) -> None:
        """COLLAPSE + seam fail → still DISSOLUTION."""
        verdict, reasons = classify_epistemic_act(seam_pass=False, tau_R=float("inf"), regime=Regime.COLLAPSE)
        assert verdict == EpistemicVerdict.DISSOLUTION
        assert reasons == []

    def test_dissolution_overrides_gesture(self) -> None:
        """COLLAPSE regime overrides what would be GESTURE."""
        verdict, _ = classify_epistemic_act(
            seam_pass=False,
            tau_R=float("inf"),
            regime=Regime.COLLAPSE,
            seam_failures=["everything failed"],
        )
        assert verdict == EpistemicVerdict.DISSOLUTION


# =============================================================================
# POSITIONAL ILLUSION — Observation Cost Quantification
# =============================================================================


class TestPositionalIllusion:
    """Test positional illusion quantification."""

    def test_stable_low_cost(self) -> None:
        """At STABLE ω=0.031, observation cost should be small."""
        pi = quantify_positional_illusion(0.031)
        assert pi.gamma > 0
        assert pi.gamma < 1e-3  # Very small at stable drift
        assert pi.illusion_severity < 0.1  # Affordable

    def test_watch_moderate_cost(self) -> None:
        """At WATCH ω=0.15, cost is moderate."""
        pi = quantify_positional_illusion(0.15)
        assert pi.gamma > 1e-4
        assert pi.illusion_severity > 0.001

    def test_near_collapse_high_cost(self) -> None:
        """Near COLLAPSE ω=0.29, cost approaches seam budget."""
        pi = quantify_positional_illusion(0.29)
        assert pi.gamma > 0.01
        assert pi.illusion_severity > 1.0  # Exceeds budget

    def test_multiple_observations_scale_linearly(self) -> None:
        """N observations cost exactly N × Γ(ω)."""
        pi1 = quantify_positional_illusion(0.10, n_observations=1)
        pi10 = quantify_positional_illusion(0.10, n_observations=10)
        assert abs(pi10.total_cost - 10 * pi1.total_cost) < 1e-12

    def test_zero_drift_zero_cost(self) -> None:
        """At ω=0, Γ(0) = 0 — no drift, no observation cost."""
        pi = quantify_positional_illusion(0.0)
        assert pi.gamma == 0.0
        assert pi.total_cost == 0.0
        assert pi.illusion_severity == 0.0

    def test_gamma_formula_matches_frozen_contract(self) -> None:
        """Γ(ω) = ω^p / (1 - ω + ε) matches frozen_contract.gamma_omega."""
        omega = 0.05
        expected = omega**P_EXPONENT / (1.0 - omega + EPSILON)
        pi = quantify_positional_illusion(omega)
        assert abs(pi.gamma - expected) < 1e-15

    def test_budget_fraction_is_ratio_to_tolerance(self) -> None:
        """Budget fraction = total_cost / tol_seam."""
        pi = quantify_positional_illusion(0.10, n_observations=3)
        expected = pi.total_cost / TOL_SEAM
        assert abs(pi.budget_fraction - expected) < 1e-15

    def test_result_is_namedtuple(self) -> None:
        """PositionalIllusion is a NamedTuple with correct fields."""
        pi = quantify_positional_illusion(0.05)
        assert isinstance(pi, PositionalIllusion)
        assert hasattr(pi, "gamma")
        assert hasattr(pi, "n_observations")
        assert hasattr(pi, "total_cost")
        assert hasattr(pi, "budget_fraction")
        assert hasattr(pi, "illusion_severity")


# =============================================================================
# EPISTEMIC TRACE METADATA
# =============================================================================


class TestEpistemicTraceMetadata:
    """Test epistemic trace metadata assessment."""

    def test_basic_assessment(self) -> None:
        """Basic trace assessment produces metadata."""
        meta = assess_epistemic_trace(
            n_components=10,
            n_timesteps=100,
            seam_pass=True,
            tau_R=1.85,
            regime=Regime.STABLE,
        )
        assert meta.n_components == 10
        assert meta.n_timesteps == 100
        assert meta.verdict == EpistemicVerdict.RETURN

    def test_gesture_trace(self) -> None:
        """Trace with seam failure → GESTURE verdict."""
        meta = assess_epistemic_trace(
            n_components=5,
            n_timesteps=50,
            seam_pass=False,
            tau_R=float("inf"),
            regime=Regime.STABLE,
        )
        assert meta.verdict == EpistemicVerdict.GESTURE

    def test_dissolution_trace(self) -> None:
        """Trace in COLLAPSE → DISSOLUTION verdict."""
        meta = assess_epistemic_trace(
            n_components=5,
            n_timesteps=50,
            seam_pass=True,
            tau_R=1.0,
            regime=Regime.COLLAPSE,
        )
        assert meta.verdict == EpistemicVerdict.DISSOLUTION

    def test_clipped_fraction_computed(self) -> None:
        """Clipped fraction = n_clipped / n_components."""
        meta = assess_epistemic_trace(
            n_components=10,
            n_timesteps=50,
            n_clipped=3,
            seam_pass=True,
            tau_R=1.0,
            regime=Regime.STABLE,
        )
        assert abs(meta.clipped_fraction - 0.3) < 1e-10

    def test_zero_components_no_divide_by_zero(self) -> None:
        """Zero components doesn't cause division by zero."""
        meta = assess_epistemic_trace(
            n_components=0,
            n_timesteps=50,
        )
        assert meta.clipped_fraction == 0.0

    def test_to_dict_serialization(self) -> None:
        """Metadata serializes to dict correctly."""
        meta = assess_epistemic_trace(
            n_components=5,
            n_timesteps=20,
            seam_pass=True,
            tau_R=2.0,
            regime=Regime.STABLE,
        )
        d = meta.to_dict()
        assert d["n_components"] == 5
        assert d["n_timesteps"] == 20
        assert d["verdict"] == "return"
        assert "epsilon_floor" in d

    def test_default_verdict_is_gesture(self) -> None:
        """Default verdict (unverified trace) is GESTURE."""
        meta = EpistemicTraceMetadata(n_components=5, n_timesteps=10)
        assert meta.verdict == EpistemicVerdict.GESTURE


# =============================================================================
# SEAM EPISTEMOLOGY — Complete Assessment
# =============================================================================


class TestSeamEpistemology:
    """Test complete epistemic seam assessment."""

    def test_return_assessment(self) -> None:
        """Passing seam → RETURN with is_real=True."""
        epi = assess_seam_epistemology(
            seam_pass=True,
            seam_failures=[],
            seam_residual=0.0,
            seam_budget=1.697,
            tau_R=1.85,
            omega=0.031,
            regime=Regime.STABLE,
        )
        assert epi.verdict == EpistemicVerdict.RETURN
        assert epi.is_real is True
        assert epi.earned_credit is True
        assert epi.reasons == []

    def test_gesture_assessment(self) -> None:
        """Failing seam → GESTURE with is_real=False."""
        epi = assess_seam_epistemology(
            seam_pass=False,
            seam_failures=["|s|=0.01 > tol_seam=0.005"],
            seam_residual=0.01,
            seam_budget=1.0,
            tau_R=1.0,
            omega=0.05,
            regime=Regime.STABLE,
        )
        assert epi.verdict == EpistemicVerdict.GESTURE
        assert epi.is_real is False
        assert epi.earned_credit is False
        assert len(epi.reasons) > 0

    def test_dissolution_assessment(self) -> None:
        """COLLAPSE regime → DISSOLUTION."""
        epi = assess_seam_epistemology(
            seam_pass=True,
            seam_failures=[],
            seam_residual=0.0,
            seam_budget=0.5,
            tau_R=1.0,
            omega=0.35,
            regime=Regime.COLLAPSE,
        )
        assert epi.verdict == EpistemicVerdict.DISSOLUTION
        assert epi.is_real is False

    def test_illusion_computed_by_default(self) -> None:
        """Positional illusion is computed when compute_illusion=True."""
        epi = assess_seam_epistemology(
            seam_pass=True,
            seam_failures=[],
            seam_residual=0.0,
            seam_budget=1.0,
            tau_R=1.0,
            omega=0.05,
            regime=Regime.STABLE,
        )
        assert epi.illusion is not None
        assert epi.illusion.gamma > 0

    def test_illusion_skipped_when_disabled(self) -> None:
        """Positional illusion skipped when compute_illusion=False."""
        epi = assess_seam_epistemology(
            seam_pass=True,
            seam_failures=[],
            seam_residual=0.0,
            seam_budget=1.0,
            tau_R=1.0,
            omega=0.05,
            regime=Regime.STABLE,
            compute_illusion=False,
        )
        assert epi.illusion is None

    def test_to_dict_serialization(self) -> None:
        """SeamEpistemology serializes to dict."""
        epi = assess_seam_epistemology(
            seam_pass=True,
            seam_failures=[],
            seam_residual=0.001,
            seam_budget=1.697,
            tau_R=1.85,
            omega=0.031,
            regime=Regime.STABLE,
        )
        d = epi.to_dict()
        assert d["verdict"] == "return"
        assert d["seam_residual"] == 0.001
        assert d["omega"] == 0.031
        assert "illusion" in d
        assert d["illusion"]["gamma"] > 0

    def test_hud_values_from_paper(self) -> None:
        """Verify HUD values from 'The Seam of Reality' paper.

        ω=0.031, τ_R=1.850, R=1.000, Δκ=1.697, s=0.000
        All should produce RETURN verdict.
        """
        epi = assess_seam_epistemology(
            seam_pass=True,
            seam_failures=[],
            seam_residual=0.0,
            seam_budget=1.697,
            tau_R=1.85,
            omega=0.031,
            regime=Regime.STABLE,
        )
        assert epi.verdict == EpistemicVerdict.RETURN
        assert epi.is_real is True
        assert abs(epi.seam_residual) <= TOL_SEAM
        assert epi.tau_R == 1.85
        assert epi.omega == 0.031


# =============================================================================
# GESTURE DIAGNOSTICS
# =============================================================================


class TestGestureDiagnostics:
    """Test gesture diagnostic output."""

    def test_infinite_tau_R_diagnosis(self) -> None:
        """INF_REC diagnosis produces meaningful commentary."""
        diag = diagnose_gesture(
            seam_residual=0.01,
            tau_R=float("inf"),
            omega=0.05,
            regime=Regime.STABLE,
        )
        assert diag["return_status"] == "INF_REC"
        assert "return_commentary" in diag
        assert "no path" not in diag.get("dissolution_commentary", "").lower()

    def test_collapse_dissolution_commentary(self) -> None:
        """COLLAPSE regime produces dissolution commentary."""
        diag = diagnose_gesture(
            seam_residual=0.1,
            tau_R=float("inf"),
            omega=0.35,
            regime=Regime.COLLAPSE,
        )
        assert "dissolution_commentary" in diag
        assert "boundary condition" in diag["dissolution_commentary"]

    def test_watch_commentary(self) -> None:
        """WATCH regime produces watch commentary."""
        diag = diagnose_gesture(
            seam_residual=0.01,
            tau_R=2.0,
            omega=0.15,
            regime=Regime.WATCH,
        )
        assert "watch_commentary" in diag
        assert "curvature" in diag["watch_commentary"].lower()

    def test_observation_cost_included(self) -> None:
        """Observation cost (Γ(ω)) always included in diagnostics."""
        diag = diagnose_gesture(
            seam_residual=0.001,
            tau_R=1.0,
            omega=0.05,
            regime=Regime.STABLE,
        )
        assert "observation_cost_gamma" in diag
        assert diag["observation_cost_gamma"] > 0
        assert "positional_illusion_note" in diag

    def test_near_return_detection(self) -> None:
        """Residual just above tolerance → near_return=True."""
        diag = diagnose_gesture(
            seam_residual=0.006,
            tau_R=1.0,
            omega=0.03,
            regime=Regime.STABLE,
        )
        assert diag["near_return"] is True
        assert diag["residual_excess"] == pytest.approx(0.001, abs=1e-10)

    def test_far_from_return_detection(self) -> None:
        """Residual far above tolerance → near_return=False."""
        diag = diagnose_gesture(
            seam_residual=0.1,
            tau_R=1.0,
            omega=0.10,
            regime=Regime.STABLE,
        )
        assert diag["near_return"] is False


# =============================================================================
# NONCONFORMANCE TYPE — GESTURE INTEGRATION
# =============================================================================


class TestNonconformanceGesture:
    """Test that GESTURE is properly integrated in NonconformanceType."""

    def test_gesture_in_enum(self) -> None:
        """GESTURE exists in NonconformanceType."""
        assert hasattr(NonconformanceType, "GESTURE")
        assert NonconformanceType.GESTURE.value == "gesture"

    def test_gesture_distinct_from_no_return(self) -> None:
        """GESTURE and NO_RETURN are distinct failure modes."""
        assert NonconformanceType.GESTURE != NonconformanceType.NO_RETURN

    def test_all_nonconformance_types(self) -> None:
        """All expected nonconformance types exist."""
        expected = {
            "SEAM_FAILURE",
            "NO_RETURN",
            "GESTURE",
            "TIER0_FAILURE",
            "CLOSURE_FAILURE",
            "SYMBOL_FAILURE",
            "DIAGNOSTIC_MISUSE",
        }
        actual = {member.name for member in NonconformanceType}
        assert expected == actual


# =============================================================================
# EDGE CASES AND INVARIANTS
# =============================================================================


class TestEdgeCases:
    """Test edge cases and invariants of the epistemic module."""

    def test_verdict_exhaustive_over_regime_seam_combinations(self) -> None:
        """Every regime × seam combination produces a valid verdict."""
        for regime in [Regime.STABLE, Regime.WATCH, Regime.COLLAPSE, Regime.CRITICAL]:
            for seam_pass in [True, False]:
                for tau_R in [1.0, float("inf")]:
                    verdict, _ = classify_epistemic_act(seam_pass=seam_pass, tau_R=tau_R, regime=regime)
                    assert verdict in VALID_VERDICTS

    def test_return_requires_both_seam_pass_and_finite_tau_R(self) -> None:
        """RETURN requires BOTH seam_pass=True AND finite τ_R."""
        # Miss seam
        v1, _ = classify_epistemic_act(seam_pass=False, tau_R=1.0, regime=Regime.STABLE)
        assert v1 != EpistemicVerdict.RETURN

        # Miss tau_R
        v2, _ = classify_epistemic_act(seam_pass=True, tau_R=float("inf"), regime=Regime.STABLE)
        assert v2 != EpistemicVerdict.RETURN

        # Both → RETURN
        v3, _ = classify_epistemic_act(seam_pass=True, tau_R=1.0, regime=Regime.STABLE)
        assert v3 == EpistemicVerdict.RETURN

    def test_collapse_always_dissolution(self) -> None:
        """COLLAPSE regime ALWAYS produces DISSOLUTION."""
        for seam_pass in [True, False]:
            for tau_R in [0.1, 1.0, 100.0, float("inf")]:
                verdict, _ = classify_epistemic_act(seam_pass=seam_pass, tau_R=tau_R, regime=Regime.COLLAPSE)
                assert verdict == EpistemicVerdict.DISSOLUTION

    def test_positional_illusion_monotone_in_omega(self) -> None:
        """Illusion severity is monotonically increasing in ω."""
        severities = []
        for omega in [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.29]:
            pi = quantify_positional_illusion(omega)
            severities.append(pi.illusion_severity)
        for i in range(1, len(severities)):
            assert severities[i] > severities[i - 1]

    def test_seam_epistemology_is_frozen(self) -> None:
        """SeamEpistemology is a frozen dataclass."""
        epi = assess_seam_epistemology(
            seam_pass=True,
            seam_failures=[],
            seam_residual=0.0,
            seam_budget=1.0,
            tau_R=1.0,
            omega=0.05,
            regime=Regime.STABLE,
        )
        with pytest.raises(AttributeError):
            epi.verdict = EpistemicVerdict.GESTURE  # type: ignore[misc]

    def test_epistemic_trace_metadata_is_frozen(self) -> None:
        """EpistemicTraceMetadata is a frozen dataclass."""
        meta = assess_epistemic_trace(
            n_components=5,
            n_timesteps=10,
            seam_pass=True,
            tau_R=1.0,
            regime=Regime.STABLE,
        )
        with pytest.raises(AttributeError):
            meta.n_components = 99  # type: ignore[misc]


# =============================================================================
# GESTURE REASON ENUM
# =============================================================================


class TestGestureReason:
    """Test GestureReason enum completeness."""

    def test_gesture_reasons_exist(self) -> None:
        """All expected gesture reasons exist."""
        expected = {
            "SEAM_RESIDUAL_EXCEEDED",
            "NO_FINITE_RETURN",
            "IDENTITY_MISMATCH",
            "FROZEN_PARAMETER_DRIFT",
            "TIER0_INCOMPLETE",
        }
        actual = {member.name for member in GestureReason}
        assert expected == actual

    def test_gesture_reasons_have_string_values(self) -> None:
        """Each reason has a meaningful string value."""
        for r in GestureReason:
            assert isinstance(r.value, str)
            assert len(r.value) > 5

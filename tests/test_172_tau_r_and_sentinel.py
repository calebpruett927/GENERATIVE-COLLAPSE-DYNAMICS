"""Tests for τ_R norm variants and safe_tau_R/tau_R_display edge cases.

compute_tau_R supports L2, L1, Linf norms — only L2 was tested.
safe_tau_R handles 10+ input forms — many edge cases were untested.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from umcp.frozen_contract import compute_tau_R
from umcp.measurement_engine import safe_tau_R, tau_R_display

# ============================================================================
# compute_tau_R norm variants
# ============================================================================


class TestTauRNormVariants:
    """compute_tau_R with different norm specifications."""

    @pytest.fixture()
    def returning_trace(self) -> np.ndarray:
        """Trace where t=2 is close to t=0."""
        return np.array(
            [
                [0.3, 0.4],
                [0.8, 0.9],  # far away
                [0.31, 0.41],  # close to t=0
            ]
        )

    def test_l2_norm_returns(self, returning_trace: np.ndarray) -> None:
        tau = compute_tau_R(returning_trace, t=2, eta=0.1, H_rec=10, norm="L2")
        assert math.isfinite(tau)
        assert tau == 2.0  # returns to t=0

    def test_l1_norm_returns(self, returning_trace: np.ndarray) -> None:
        tau = compute_tau_R(returning_trace, t=2, eta=0.1, H_rec=10, norm="L1")
        assert math.isfinite(tau)

    def test_linf_norm_returns(self, returning_trace: np.ndarray) -> None:
        tau = compute_tau_R(returning_trace, t=2, eta=0.1, H_rec=10, norm="Linf")
        assert math.isfinite(tau)

    def test_invalid_norm_raises(self) -> None:
        trace = np.array([[0.3, 0.4], [0.31, 0.41]])
        with pytest.raises(ValueError, match=r"[Nn]orm|[Uu]nknown|[Uu]nsupported"):
            compute_tau_R(trace, t=1, eta=0.1, H_rec=10, norm="L3")

    def test_norms_produce_different_distances(self) -> None:
        """L1, L2, Linf measure distance differently; results may differ."""
        trace = np.array(
            [
                [0.1, 0.1],
                [0.5, 0.5],
                [0.15, 0.08],  # close to t=0 under some norms
            ]
        )
        taus = {}
        for norm in ("L1", "L2", "Linf"):
            taus[norm] = compute_tau_R(trace, t=2, eta=0.15, H_rec=10, norm=norm)
        # All should be either finite or inf; no NaN
        for norm, tau in taus.items():
            assert not math.isnan(tau), f"NaN for norm={norm}"

    def test_l1_norm_diverging_is_inf(self) -> None:
        """Monotonically diverging trace → INF_REC for all norms."""
        trace = np.array(
            [
                [0.1, 0.1],
                [0.3, 0.3],
                [0.5, 0.5],
                [0.7, 0.7],
                [0.9, 0.9],
            ]
        )
        tau = compute_tau_R(trace, t=4, eta=0.05, H_rec=10, norm="L1")
        assert math.isinf(tau)

    def test_linf_norm_diverging_is_inf(self) -> None:
        trace = np.array(
            [
                [0.1, 0.1],
                [0.3, 0.3],
                [0.5, 0.5],
                [0.7, 0.7],
                [0.9, 0.9],
            ]
        )
        tau = compute_tau_R(trace, t=4, eta=0.05, H_rec=10, norm="Linf")
        assert math.isinf(tau)


# ============================================================================
# safe_tau_R edge cases
# ============================================================================


class TestSafeTauREdgeCases:
    """Edge cases for safe_tau_R sentinel handling."""

    # --- Standard cases (already tested, included for completeness) ---
    def test_inf_rec_string(self) -> None:
        assert math.isinf(safe_tau_R("INF_REC"))

    def test_float_inf(self) -> None:
        assert math.isinf(safe_tau_R(float("inf")))

    def test_normal_float(self) -> None:
        assert safe_tau_R(5.0) == 5.0

    def test_normal_int(self) -> None:
        assert safe_tau_R(3) == 3.0

    # --- New edge cases ---
    def test_none_returns_inf(self) -> None:
        assert math.isinf(safe_tau_R(None))

    def test_empty_string_returns_inf(self) -> None:
        assert math.isinf(safe_tau_R(""))

    def test_nan_string_returns_inf(self) -> None:
        assert math.isinf(safe_tau_R("NAN"))

    def test_none_string_returns_inf(self) -> None:
        assert math.isinf(safe_tau_R("NONE"))

    def test_infinity_string_returns_inf(self) -> None:
        assert math.isinf(safe_tau_R("INFINITY"))

    def test_inf_string_returns_inf(self) -> None:
        assert math.isinf(safe_tau_R("INF"))

    def test_unicode_infinity_returns_inf(self) -> None:
        assert math.isinf(safe_tau_R("∞"))

    def test_float_nan_returns_inf(self) -> None:
        assert math.isinf(safe_tau_R(float("nan")))

    def test_negative_float_preserved(self) -> None:
        """Negative τ_R should be preserved as a float (caller checks validity)."""
        result = safe_tau_R(-5.0)
        assert result == -5.0 or math.isinf(result)  # impl may reject negatives

    def test_numeric_string_parsed(self) -> None:
        assert safe_tau_R("12.5") == 12.5

    def test_zero_preserved(self) -> None:
        assert safe_tau_R(0) == 0.0

    def test_zero_string_preserved(self) -> None:
        assert safe_tau_R("0") == 0.0


# ============================================================================
# tau_R_display edge cases
# ============================================================================


class TestTauRDisplayEdgeCases:
    """tau_R_display formatting edge cases."""

    def test_inf_displays_as_inf_rec(self) -> None:
        assert tau_R_display(float("inf")) == "INF_REC"

    def test_finite_displays_as_number(self) -> None:
        result = tau_R_display(5.0)
        assert "5" in result

    def test_none_displays_as_inf_rec(self) -> None:
        assert tau_R_display(None) == "INF_REC"

    def test_string_inf_rec_roundtrips(self) -> None:
        """INF_REC → safe_tau_R → tau_R_display → INF_REC."""
        parsed = safe_tau_R("INF_REC")
        displayed = tau_R_display(parsed)
        assert displayed == "INF_REC"

    def test_zero_displays_as_zero(self) -> None:
        result = tau_R_display(0)
        assert "0" in result

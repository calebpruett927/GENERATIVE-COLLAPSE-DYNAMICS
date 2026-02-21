"""
Tests for nuclear_physics/double_sided_collapse.py closure module.
"""

from __future__ import annotations

import pytest


class TestDoubleSidedCollapse:
    """Unit tests for compute_double_sided."""

    def test_iron_peak(self):
        from closures.nuclear_physics.double_sided_collapse import compute_double_sided

        result = compute_double_sided(Z=28, A=62, BE_per_A=8.7945)
        assert result.side == "Peak"
        assert result.convergence_direction == "≡Fe"
        assert result.regime == "AtPeak"

    def test_light_nucleus(self):
        from closures.nuclear_physics.double_sided_collapse import compute_double_sided

        result = compute_double_sided(Z=2, A=4, BE_per_A=7.0739)
        assert result.side == "Light"
        assert result.convergence_direction == "→Fe"
        assert result.regime in ("Convergent", "Distant")

    def test_heavy_nucleus(self):
        from closures.nuclear_physics.double_sided_collapse import compute_double_sided

        result = compute_double_sided(Z=92, A=238, BE_per_A=7.5701)
        assert result.side == "Heavy"
        assert result.convergence_direction == "←Fe"

    def test_near_peak(self):
        from closures.nuclear_physics.double_sided_collapse import compute_double_sided

        result = compute_double_sided(Z=26, A=56, BE_per_A=8.79)
        assert result.regime in ("AtPeak", "NearPeak")

    def test_omega_eff_bounds(self):
        from closures.nuclear_physics.double_sided_collapse import compute_double_sided

        result = compute_double_sided(Z=1, A=1, BE_per_A=0.0)
        assert result.omega_eff >= 0.0

    @pytest.mark.parametrize(
        "Z,A,BE_per_A,expected_side",
        [
            (1, 2, 1.112, "Light"),
            (2, 4, 7.074, "Light"),
            (6, 12, 7.680, "Light"),
            (26, 56, 8.790, "Light"),
            (28, 62, 8.795, "Peak"),
            (50, 120, 8.505, "Heavy"),
            (82, 208, 7.868, "Heavy"),
            (92, 238, 7.570, "Heavy"),
        ],
    )
    def test_nuclear_chart(self, Z, A, BE_per_A, expected_side):
        from closures.nuclear_physics.double_sided_collapse import compute_double_sided

        result = compute_double_sided(Z=Z, A=A, BE_per_A=BE_per_A)
        assert result.side == expected_side
        assert result.A == A

    def test_result_fields(self):
        from closures.nuclear_physics.double_sided_collapse import compute_double_sided

        result = compute_double_sided(Z=26, A=56, BE_per_A=8.79)
        assert hasattr(result, "A")
        assert hasattr(result, "BE_per_A")
        assert hasattr(result, "signed_distance")
        assert hasattr(result, "abs_distance")
        assert hasattr(result, "side")
        assert hasattr(result, "convergence_direction")
        assert hasattr(result, "omega_eff")
        assert hasattr(result, "regime")

    @pytest.mark.parametrize(
        "BE_per_A,expected_regime",
        [
            (8.7945, "AtPeak"),
            (8.75, "NearPeak"),
            (8.5, "Convergent"),
            (7.0, "Distant"),
        ],
    )
    def test_regime_classification(self, BE_per_A, expected_regime):
        from closures.nuclear_physics.double_sided_collapse import compute_double_sided

        result = compute_double_sided(Z=28, A=62, BE_per_A=BE_per_A)
        assert result.regime == expected_regime

    def test_signed_distance_symmetry(self):
        from closures.nuclear_physics.double_sided_collapse import compute_double_sided

        light = compute_double_sided(Z=2, A=4, BE_per_A=7.074)
        heavy = compute_double_sided(Z=92, A=238, BE_per_A=7.570)
        # Both below peak → both negative signed_distance
        assert light.signed_distance < 0
        assert heavy.signed_distance < 0

    def test_abs_distance_positive(self):
        from closures.nuclear_physics.double_sided_collapse import compute_double_sided

        result = compute_double_sided(Z=1, A=2, BE_per_A=1.112)
        assert result.abs_distance >= 0.0

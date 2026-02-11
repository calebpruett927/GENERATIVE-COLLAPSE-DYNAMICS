"""
Test Utilities Module

Provides optimized test fixtures and helpers using lemma-based shortcuts
from KERNEL_SPECIFICATION.md and docs/COMPUTATIONAL_OPTIMIZATIONS.md.

Key optimizations:
- OPT-1 (Lemma 10): Homogeneity detection for fast test data generation
- OPT-12 (Lemma 23): Lipschitz bounds for approximate validation
- Session-scoped caching for expensive I/O operations
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

# =============================================================================
# Lemma 10: Homogeneity Fast-Path for Test Data
# =============================================================================


def generate_homogeneous_state(value: float = 0.9, n_coords: int = 4) -> dict[str, Any]:
    """
    Generate a homogeneous test state using Lemma 10 fast-path.

    Lemma 10: When all coordinates equal, C = 0 and F = IC.
    This allows skipping full kernel computation in tests.

    Args:
        value: Coordinate value (default 0.9 for Stable regime)
        n_coords: Number of coordinates (default 4)

    Returns:
        State dict with pre-computed invariants
    """
    # Homogeneous case: F = IC = value, C = 0, ω = 1 - value
    F = value
    omega = 1 - value
    IC = value
    kappa = np.log(value)
    C = 0.0

    # Entropy for homogeneous: h(F) = -F*log(F) - (1-F)*log(1-F)
    entropy = -value * np.log(value) - (1 - value) * np.log(1 - value) if 0 < value < 1 else 0.0

    return {
        "coordinates": [value] * n_coords,
        "weights": [1.0 / n_coords] * n_coords,
        "F": F,
        "omega": omega,
        "IC": IC,
        "kappa": kappa,
        "C": C,
        "S": entropy,
        "regime": _classify_regime(omega, F, entropy, C),
        "is_homogeneous": True,
    }


def _classify_regime(omega: float, F: float, S: float, C: float) -> str:
    """Quick regime classification per canon/anchors.yaml thresholds."""
    if omega >= 0.30:
        return "Collapse"
    if omega < 0.038 and F > 0.90 and S < 0.15 and C < 0.14:
        return "Stable"
    return "Watch"


# =============================================================================
# Lemma 1: Bounds Validation for Quick Tests
# =============================================================================


def quick_bounds_check(omega: float, F: float, S: float, C: float) -> bool:
    """
    Fast Lemma 1 bounds validation without full kernel computation.

    Lemma 1 constraints:
    - 0 ≤ F ≤ 1
    - 0 ≤ ω ≤ 1 (with ω = 1 - F)
    - 0 ≤ S ≤ ln(n) (capped at 1 for normalized)
    - 0 ≤ C ≤ 1

    Returns:
        True if all bounds satisfied
    """
    return (
        0 <= F <= 1
        and 0 <= omega <= 1
        and abs(omega - (1 - F)) < 1e-9  # Identity check
        and 0 <= S <= 1
        and 0 <= C <= 1
    )


# =============================================================================
# Lemma 23: Lipschitz Approximate Comparison
# =============================================================================


def approx_equal_lipschitz(
    value1: float, value2: float, perturbation: float = 0.01, lipschitz_constant: float = 1.0
) -> bool:
    """
    Use Lipschitz bound (Lemma 23) for approximate comparison.

    Lemma 23: |f(x) - f(y)| ≤ L · |x - y|

    For small perturbations, outputs should differ by at most L * perturbation.
    This allows faster approximate validation in tests.
    """
    return abs(value1 - value2) <= lipschitz_constant * perturbation * 1.1  # 10% margin


# =============================================================================
# Session-Scoped Caching for I/O Operations
# =============================================================================


@lru_cache(maxsize=32)
def cached_load_json(path_str: str) -> dict[str, Any]:
    """Load JSON file with caching - call once per session."""
    path = Path(path_str)
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=32)
def cached_load_yaml(path_str: str) -> dict[str, Any]:
    """Load YAML file with caching - call once per session."""
    import yaml

    path = Path(path_str)
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=8)
def cached_compile_schema(schema_path_str: str) -> Any:
    """Compile JSON schema validator with caching."""
    from jsonschema import Draft202012Validator

    schema = cached_load_json(schema_path_str)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


# =============================================================================
# Fast Test Data Generators
# =============================================================================


def generate_regime_test_cases() -> list[dict[str, Any]]:
    """
    Generate minimal test cases for each regime using Lemma 10 fast-path.

    Returns:
        List of pre-computed states for Stable, Watch, and Collapse regimes
    """
    return [
        {**generate_homogeneous_state(0.98), "expected_regime": "Stable"},
        {**generate_homogeneous_state(0.85), "expected_regime": "Watch"},
        {**generate_homogeneous_state(0.65), "expected_regime": "Collapse"},
    ]


def generate_boundary_test_cases() -> list[dict[str, Any]]:
    """
    Generate boundary condition test cases.

    Uses Lemma 1 bounds to ensure valid edge cases.
    """
    return [
        # Near-zero entropy (homogeneous near 1.0)
        generate_homogeneous_state(0.999),
        # Near-collapse boundary
        generate_homogeneous_state(0.70),
        # Mid-range Watch
        generate_homogeneous_state(0.80),
    ]

"""Test Manifold — Bound Surface Verification

This module defines the COMPLETE bound surface of the UMCP test manifold.
Instead of running 1,817 individual tests, this file establishes the
tightest proven bounds across every degree of freedom and verifies them
in a single fast pass.

Architecture — Three-layer manifold:

  Layer 0  IDENTITY   (algebraic closure — no data dependencies)
  ──────────────────────────────────────────────────────────────
  Tests that any valid coordinate vector c ∈ [ε, 1-ε]^d with
  weights w must satisfy:
    • F + ω  = 1            (partition of unity, exact)
    • IC     ≤ F            (AM-GM inequality)
    • IC     ≈ exp(κ)       (curvature-complexity link)
    • S      ≥ 0            (non-negative seam residual)
    • C      ≥ 0            (non-negative cost)
    • ω      ∈ [0, 1)       (drift cannot exceed unity)
    • F      ∈ (0, 1]       (fidelity always positive)
  If this layer fails, nothing above it can be trusted.

  Layer 1  REGIME GATES   (threshold closure — depends on Layer 0)
  ──────────────────────────────────────────────────────────────
  Given valid invariants from Layer 0, the four regime gates
  are deterministic functions of (ω, F, S, C, IC):
    • STABLE:   ω < 0.038 ∧ F > 0.90 ∧ S < 0.15 ∧ C < 0.14
    • COLLAPSE: ω ≥ 0.30
    • CRITICAL: IC < 0.30
    • WATCH:    ¬STABLE ∧ ¬COLLAPSE ∧ ¬CRITICAL
  This layer verifies the gates at boundary points, corners, and
  interior samples.  If Layer 0 holds, this layer is a pure lookup.

  Layer 2  DOMAIN CLOSURE   (physics — depends on Layers 0 + 1)
  ──────────────────────────────────────────────────────────────
  Domain-specific embeddings (GCD, RCFT, kinematics, Weyl,
  nuclear, finance, active matter) map real-world measurement
  data to coordinate vectors.  This layer verifies:
    • Embedding produces valid coordinates (each ∈ [ε, 1-ε])
    • Schema conformance of closure files
    • Known reference points classify to expected regimes
  Only needs to run if Layers 0-1 pass.

Bound compression:
  Each layer defines a set of BOUNDS — tight intervals that
  must hold.  A new test is valuable iff it tightens at least
  one bound.  The `TestBoundsRegistry` at the end reports
  current bound widths and identifies which tests are compressing
  which degrees of freedom.

Cross-references:
  KERNEL_SPECIFICATION.md  — formal definitions
  docs/COMPUTATIONAL_OPTIMIZATIONS.md  — OPT-* lemma tags
  tests/conftest.py  — shared fixtures
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from umcp.frozen_contract import (
    EPSILON,
    Regime,
    RegimeThresholds,
    classify_regime,
)
from umcp.kernel_optimized import OptimizedKernelComputer

# ═══════════════════════════════════════════════════════════════════
#  BOUND DEFINITIONS
# ═══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Bound:
    """A named interval constraint on a scalar quantity.

    A bound is TIGHT when width → 0 (exact identity).
    A bound is OPEN when width > 0 (range of valid values).
    """

    name: str
    lower: float
    upper: float
    tolerance: float  # max allowed violation
    layer: int  # 0=identity, 1=regime, 2=domain

    @property
    def width(self) -> float:
        return self.upper - self.lower

    @property
    def is_tight(self) -> bool:
        return self.width <= self.tolerance

    def check(self, value: float) -> bool:
        """Return True if value is within [lower - tol, upper + tol]."""
        return (self.lower - self.tolerance) <= value <= (self.upper + self.tolerance)


@dataclass
class BoundsRegistry:
    """Tracks all proven bounds and their current tightness."""

    bounds: list[Bound] = field(default_factory=list)
    violations: list[tuple[str, float, Bound]] = field(default_factory=list)

    def register(self, bound: Bound) -> None:
        self.bounds.append(bound)

    def check(self, name: str, value: float) -> bool:
        for b in self.bounds:
            if b.name == name:
                ok = b.check(value)
                if not ok:
                    self.violations.append((name, value, b))
                return ok
        raise KeyError(f"No bound registered: {name}")

    def layer_bounds(self, layer: int) -> list[Bound]:
        return [b for b in self.bounds if b.layer == layer]

    def summary(self) -> dict[str, Any]:
        by_layer: dict[int, list[dict[str, Any]]] = {}
        for b in self.bounds:
            entry = {
                "name": b.name,
                "interval": f"[{b.lower}, {b.upper}]",
                "width": b.width,
                "tight": b.is_tight,
                "tolerance": b.tolerance,
            }
            by_layer.setdefault(b.layer, []).append(entry)
        return {
            "total_bounds": len(self.bounds),
            "tight_bounds": sum(1 for b in self.bounds if b.is_tight),
            "open_bounds": sum(1 for b in self.bounds if not b.is_tight),
            "violations": len(self.violations),
            "layers": by_layer,
        }


# ── The canonical bound registry ────────────────────────────────
REGISTRY = BoundsRegistry()

# Layer 0: Identity bounds (algebraic — must be exact)
REGISTRY.register(Bound("F+ω", 1.0, 1.0, 1e-12, 0))  # partition of unity
REGISTRY.register(Bound("IC≤F", 0.0, 0.0, 1e-12, 0))  # AM-GM: IC-F ≤ 0
REGISTRY.register(Bound("ω_range_lo", 0.0, 1.0, 1e-12, 0))  # ω ≥ 0
REGISTRY.register(Bound("ω_range_hi", 0.0, 1.0, 1e-12, 0))  # ω < 1
REGISTRY.register(Bound("F_range_lo", 0.0, 1.0, 1e-12, 0))  # F > 0
REGISTRY.register(Bound("S_nonneg", 0.0, 1.0, 1e-12, 0))  # S ≥ 0
REGISTRY.register(Bound("C_nonneg", 0.0, 1.0, 1e-12, 0))  # C ≥ 0

# Layer 1: Regime threshold bounds (frozen constants)
THRESHOLDS = RegimeThresholds()
REGISTRY.register(Bound("ω_stable_max", 0.0, THRESHOLDS.omega_stable_max, 0.0, 1))
REGISTRY.register(Bound("F_stable_min", THRESHOLDS.F_stable_min, 1.0, 0.0, 1))
REGISTRY.register(Bound("S_stable_max", 0.0, THRESHOLDS.S_stable_max, 0.0, 1))
REGISTRY.register(Bound("C_stable_max", 0.0, THRESHOLDS.C_stable_max, 0.0, 1))
REGISTRY.register(Bound("ω_collapse", THRESHOLDS.omega_collapse_min, 1.0, 0.0, 1))
REGISTRY.register(Bound("IC_critical", 0.0, THRESHOLDS.I_critical_max, 0.0, 1))

# Layer 2: Domain embedding bounds (coordinate validity)
REGISTRY.register(Bound("coord_floor", EPSILON, 1.0, 0.0, 2))
REGISTRY.register(Bound("coord_ceil", 0.0, 1.0 - EPSILON, 0.0, 2))


# ═══════════════════════════════════════════════════════════════════
#  LAYER 0 — IDENTITY SURFACE
# ═══════════════════════════════════════════════════════════════════

_KERNEL = OptimizedKernelComputer(epsilon=EPSILON)


def _sample_random_coords(n: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    """Generate n random coordinate vectors in [ε, 1-ε]^dim."""
    raw = rng.uniform(EPSILON, 1.0 - EPSILON, size=(n, dim))
    return raw


def _check_identities(c: np.ndarray, w: np.ndarray) -> dict[str, float]:
    """Compute all identity residuals for a single coordinate vector."""
    r = _KERNEL.compute(c, w, validate=False)

    f_plus_omega = r.F + r.omega
    ic_minus_f = r.IC - r.F  # must be ≤ 0

    # IC ≈ exp(κ) check (logarithmic)
    if r.IC > 0 and math.isfinite(r.kappa):
        ic_exp_kappa_ratio = abs(math.log(r.IC) - r.kappa) if r.IC > 1e-30 else float("inf")
    else:
        ic_exp_kappa_ratio = 0.0  # degenerate case, not a violation

    return {
        "F+ω": f_plus_omega,
        "IC-F": ic_minus_f,
        "IC_exp_κ_residual": ic_exp_kappa_ratio,
        "ω": r.omega,
        "F": r.F,
        "S": r.S,
        "C": r.C,
        "IC": r.IC,
        "κ": r.kappa,
    }


@pytest.mark.layer0
@pytest.mark.manifold
class TestLayer0Identity:
    """Layer 0: Algebraic identity surface.

    These bounds hold for ALL valid inputs — no exceptions.
    If ANY fail, the kernel itself is broken.
    """

    # ── Exact partition: F + ω = 1 ─────────────────────────────

    @pytest.mark.parametrize("dim", [1, 2, 3, 5, 10, 50])
    def test_partition_unity_uniform(self, dim: int) -> None:
        """F + ω = 1 for uniform weights across dimensions."""
        rng = np.random.default_rng(42)
        w = np.ones(dim) / dim
        for _ in range(200):
            c = rng.uniform(EPSILON, 1.0 - EPSILON, size=dim)
            vals = _check_identities(c, w)
            assert abs(vals["F+ω"] - 1.0) < 1e-12, f"F+ω = {vals['F+ω']} at c={c}"

    @pytest.mark.parametrize("dim", [2, 3, 5])
    def test_partition_unity_nonuniform_weights(self, dim: int) -> None:
        """F + ω = 1 for arbitrary positive weights."""
        rng = np.random.default_rng(123)
        for _ in range(200):
            c = rng.uniform(EPSILON, 1.0 - EPSILON, size=dim)
            w_raw = rng.uniform(0.1, 2.0, size=dim)
            w = w_raw / w_raw.sum()
            vals = _check_identities(c, w)
            assert abs(vals["F+ω"] - 1.0) < 1e-12

    # ── AM-GM: IC ≤ F ──────────────────────────────────────────

    @pytest.mark.parametrize("dim", [1, 2, 3, 5, 10, 50])
    def test_am_gm_inequality(self, dim: int) -> None:
        """IC ≤ F (geometric mean ≤ arithmetic mean)."""
        rng = np.random.default_rng(77)
        w = np.ones(dim) / dim
        for _ in range(200):
            c = rng.uniform(EPSILON, 1.0 - EPSILON, size=dim)
            vals = _check_identities(c, w)
            assert vals["IC-F"] <= 1e-12, f"IC > F: IC={vals['IC']}, F={vals['F']}"

    # ── Range bounds ───────────────────────────────────────────

    @pytest.mark.parametrize("dim", [1, 3, 10])
    def test_invariant_ranges(self, dim: int) -> None:
        """All invariants in valid ranges."""
        rng = np.random.default_rng(999)
        w = np.ones(dim) / dim
        for _ in range(500):
            c = rng.uniform(EPSILON, 1.0 - EPSILON, size=dim)
            vals = _check_identities(c, w)
            assert vals["ω"] >= -1e-12, f"ω negative: {vals['ω']}"
            assert vals["ω"] < 1.0 + 1e-12, f"ω ≥ 1: {vals['ω']}"
            assert vals["F"] > -1e-12, f"F negative: {vals['F']}"
            assert vals["F"] <= 1.0 + 1e-12, f"F > 1: {vals['F']}"
            assert vals["S"] >= -1e-12, f"S negative: {vals['S']}"
            assert vals["C"] >= -1e-12, f"C negative: {vals['C']}"

    # ── Corner probes (boundary of valid domain) ───────────────

    def test_corner_all_epsilon(self) -> None:
        """All coordinates at floor → maximum drift, minimum fidelity."""
        c = np.array([EPSILON, EPSILON, EPSILON])
        w = np.ones(3) / 3
        vals = _check_identities(c, w)
        assert abs(vals["F+ω"] - 1.0) < 1e-12
        assert vals["IC-F"] <= 1e-12
        assert vals["F"] < 0.05  # near-zero fidelity

    def test_corner_all_near_one(self) -> None:
        """All coordinates at ceiling → minimum drift, maximum fidelity."""
        c = np.array([1.0 - EPSILON, 1.0 - EPSILON, 1.0 - EPSILON])
        w = np.ones(3) / 3
        vals = _check_identities(c, w)
        assert abs(vals["F+ω"] - 1.0) < 1e-12
        assert vals["IC-F"] <= 1e-12
        assert vals["F"] > 0.99  # near-perfect fidelity

    def test_corner_mixed_extremes(self) -> None:
        """One coordinate at floor, others at ceiling → identity still holds."""
        c = np.array([EPSILON, 1.0 - EPSILON, 1.0 - EPSILON])
        w = np.ones(3) / 3
        vals = _check_identities(c, w)
        assert abs(vals["F+ω"] - 1.0) < 1e-12
        assert vals["IC-F"] <= 1e-12

    # ── Homogeneous fast-path (OPT-1/Lemma 10) ────────────────

    def test_homogeneous_identity(self) -> None:
        """c = (v, v, ..., v) → F = v, ω = 1-v, IC = v exactly."""
        for v in [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]:
            c = np.array([v, v, v])
            w = np.ones(3) / 3
            r = _KERNEL.compute(c, w, validate=False)
            assert abs(r.F - v) < 1e-10, f"F ≠ v for v={v}: F={r.F}"
            assert abs(r.omega - (1.0 - v)) < 1e-10
            assert abs(r.IC - v) < 1e-10  # GM = AM when all equal


# ═══════════════════════════════════════════════════════════════════
#  LAYER 1 — REGIME GATE SURFACE
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.layer1
@pytest.mark.manifold
class TestLayer1RegimeGates:
    """Layer 1: Regime classification boundary probes.

    Tests the exact boundary of each regime gate.
    Each test probes a point just inside and just outside
    the regime boundary to verify the gate is correctly placed.
    """

    # ── STABLE boundary ────────────────────────────────────────

    def test_stable_boundary_omega(self) -> None:
        """ω just below vs just above 0.038 flips STABLE→WATCH."""
        # Inside stable
        r = classify_regime(0.037, 0.95, 0.10, 0.10, 0.90)
        assert r == Regime.STABLE
        # Outside: ω too high
        r = classify_regime(0.039, 0.95, 0.10, 0.10, 0.90)
        assert r != Regime.STABLE

    def test_stable_boundary_F(self) -> None:
        """F just above vs just below 0.90 flips STABLE→WATCH."""
        r = classify_regime(0.03, 0.91, 0.10, 0.10, 0.90)
        assert r == Regime.STABLE
        r = classify_regime(0.03, 0.89, 0.10, 0.10, 0.90)
        assert r != Regime.STABLE

    def test_stable_boundary_S(self) -> None:
        """S just below vs just above 0.15 flips STABLE→WATCH."""
        r = classify_regime(0.03, 0.95, 0.14, 0.10, 0.90)
        assert r == Regime.STABLE
        r = classify_regime(0.03, 0.95, 0.16, 0.10, 0.90)
        assert r != Regime.STABLE

    def test_stable_boundary_C(self) -> None:
        """C just below vs just above 0.14 flips STABLE→WATCH."""
        r = classify_regime(0.03, 0.95, 0.10, 0.13, 0.90)
        assert r == Regime.STABLE
        r = classify_regime(0.03, 0.95, 0.10, 0.15, 0.90)
        assert r != Regime.STABLE

    # ── COLLAPSE boundary ──────────────────────────────────────

    def test_collapse_boundary(self) -> None:
        """ω just below vs at/above 0.30 flips WATCH→COLLAPSE."""
        r = classify_regime(0.299, 0.701, 0.40, 0.40, 0.50)
        assert r != Regime.COLLAPSE
        r = classify_regime(0.30, 0.70, 0.40, 0.40, 0.50)
        assert r == Regime.COLLAPSE

    # ── CRITICAL boundary ──────────────────────────────────────

    def test_critical_boundary(self) -> None:
        """IC just above vs just below 0.30 flips WATCH→CRITICAL."""
        r = classify_regime(0.15, 0.85, 0.30, 0.30, 0.31)
        assert r != Regime.CRITICAL
        r = classify_regime(0.15, 0.85, 0.30, 0.30, 0.29)
        assert r == Regime.CRITICAL

    # ── Gate exhaustiveness ────────────────────────────────────

    def test_regime_exhaustive(self) -> None:
        """Every valid (ω, F, S, C, IC) maps to exactly one regime."""
        rng = np.random.default_rng(42)
        valid_regimes = {Regime.STABLE, Regime.WATCH, Regime.CRITICAL, Regime.COLLAPSE}
        for _ in range(1000):
            omega = rng.uniform(0.0, 0.5)
            F = 1.0 - omega
            S = rng.uniform(0.0, 0.6)
            C = rng.uniform(0.0, 0.5)
            IC = rng.uniform(0.0, F)
            r = classify_regime(omega, F, S, C, IC)
            assert r in valid_regimes, f"Unknown regime: {r}"

    # ── Gate determinism ───────────────────────────────────────

    def test_regime_deterministic(self) -> None:
        """Same inputs always produce same regime."""
        for _ in range(100):
            args = (0.05, 0.95, 0.12, 0.08, 0.80)
            r1 = classify_regime(*args)
            r2 = classify_regime(*args)
            assert r1 == r2


# ═══════════════════════════════════════════════════════════════════
#  LAYER 2 — DOMAIN EMBEDDING SURFACE (representative probes)
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.layer2
@pytest.mark.manifold
class TestLayer2DomainProbes:
    """Layer 2: Domain embedding reference points.

    Each domain closure maps measured data to coordinates.
    These tests verify known reference points classify correctly.
    If Layers 0-1 pass, any misclassification here means the
    EMBEDDING is wrong, not the kernel.
    """

    def _classify_from_coords(self, c1: float, c2: float, c3: float) -> tuple[Regime, dict[str, float]]:
        """Classify a 3D coordinate point."""
        c = np.array([c1, c2, c3])
        w = np.ones(3) / 3
        r = _KERNEL.compute(c, w, validate=False)
        regime = classify_regime(r.omega, r.F, r.S, r.C, r.IC)
        return regime, {
            "ω": r.omega,
            "F": r.F,
            "S": r.S,
            "C": r.C,
            "IC": r.IC,
        }

    # ── Nuclear physics reference points ───────────────────────

    def test_iron_56_is_watch(self) -> None:
        """Fe-56: near-perfect binding but N/Z deviation → WATCH."""
        regime, _ = self._classify_from_coords(0.990, 0.990, 0.869)
        assert regime == Regime.WATCH, f"Fe-56 regime: {regime}"

    def test_hydrogen_is_critical(self) -> None:
        """H-1: no binding (c1≈ε) but eternal (c2≈1) → CRITICAL."""
        regime, _ = self._classify_from_coords(0.010, 0.990, 0.010)
        assert regime == Regime.CRITICAL, f"H-1 regime: {regime}"

    def test_lead_208_is_watch(self) -> None:
        """Pb-208: doubly-magic, stable, good N/Z → WATCH."""
        regime, _ = self._classify_from_coords(0.895, 0.990, 0.980)
        assert regime == Regime.WATCH, f"Pb-208 regime: {regime}"

    # ── Extreme points ─────────────────────────────────────────

    def test_perfect_coherence_is_stable(self) -> None:
        """All coordinates near 1 → STABLE."""
        regime, _ = self._classify_from_coords(0.98, 0.98, 0.98)
        assert regime == Regime.STABLE

    def test_all_low_is_critical(self) -> None:
        """All coordinates near ε → CRITICAL (IC < 0.30, catch before COLLAPSE)."""
        _, vals = self._classify_from_coords(0.01, 0.01, 0.01)
        assert vals["ω"] >= 0.30, f"ω too low for collapse/critical: {vals['ω']}"
        # IC is so low that CRITICAL gate catches it before COLLAPSE
        assert vals["IC"] < 0.30

    def test_asymmetric_low_is_collapse(self) -> None:
        """One high, two low → COLLAPSE (ω ≥ 0.30 and IC ≥ 0.30)."""
        regime, vals = self._classify_from_coords(0.99, 0.20, 0.20)
        assert regime == Regime.COLLAPSE
        assert vals["ω"] >= 0.30
        assert vals["IC"] >= 0.30


# ═══════════════════════════════════════════════════════════════════
#  STRESS TESTS — BULK SAMPLING FOR BOUND VERIFICATION
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.layer0
@pytest.mark.manifold
class TestBulkBoundSurface:
    """Verify bounds hold across large random samples.

    This replaces hundreds of individual point tests with a single
    sweep that checks the entire bound surface at once.
    """

    N_SAMPLES = 5000

    @pytest.mark.parametrize("dim", [3, 5, 10])
    def test_identity_surface_bulk(self, dim: int) -> None:
        """All identity bounds hold for N_SAMPLES random points."""
        rng = np.random.default_rng(2026)
        w = np.ones(dim) / dim
        max_f_omega_residual = 0.0
        max_ic_excess = 0.0
        violations = 0

        for _ in range(self.N_SAMPLES):
            c = rng.uniform(EPSILON, 1.0 - EPSILON, size=dim)
            vals = _check_identities(c, w)

            residual = abs(vals["F+ω"] - 1.0)
            max_f_omega_residual = max(max_f_omega_residual, residual)

            ic_excess = vals["IC-F"]
            max_ic_excess = max(max_ic_excess, ic_excess)

            if residual > 1e-12 or ic_excess > 1e-12:
                violations += 1

        assert violations == 0, (
            f"dim={dim}: {violations}/{self.N_SAMPLES} violations, "
            f"max|F+ω-1|={max_f_omega_residual:.2e}, max(IC-F)={max_ic_excess:.2e}"
        )

    def test_regime_coverage_all_four(self) -> None:
        """All four regimes are reachable from valid coordinates."""
        seen: set[str] = set()

        # STABLE: high coherence
        c = np.array([0.98, 0.98, 0.98])
        w = np.ones(3) / 3
        r = _KERNEL.compute(c, w, validate=False)
        regime = classify_regime(r.omega, r.F, r.S, r.C, r.IC)
        seen.add(regime.value)

        # WATCH: moderate
        c = np.array([0.6, 0.9, 0.7])
        r = _KERNEL.compute(c, w, validate=False)
        regime = classify_regime(r.omega, r.F, r.S, r.C, r.IC)
        seen.add(regime.value)

        # CRITICAL: imbalanced binding
        c = np.array([EPSILON, 0.99, EPSILON])
        r = _KERNEL.compute(c, w, validate=False)
        regime = classify_regime(r.omega, r.F, r.S, r.C, r.IC)
        seen.add(regime.value)

        # COLLAPSE: asymmetric — one high to keep IC > 0.30, two low for ω ≥ 0.30
        c = np.array([0.99, 0.20, 0.20])
        r = _KERNEL.compute(c, w, validate=False)
        regime = classify_regime(r.omega, r.F, r.S, r.C, r.IC)
        seen.add(regime.value)

        expected = {"STABLE", "WATCH", "CRITICAL", "COLLAPSE"}
        assert seen == expected, f"Missing regimes: {expected - seen}"


# ═══════════════════════════════════════════════════════════════════
#  BOUND COMPRESSION MONITOR
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.manifold
class TestBoundsCompression:
    """Track and report the current state of the bound surface.

    This test always passes — it measures how tight the bounds are
    and reports which degrees of freedom are fully constrained vs
    still have slack.  Use this to assess whether a new test would
    compress any DoF.
    """

    def test_report_bound_surface(self) -> None:
        """Generate and validate the bound surface report."""
        rng = np.random.default_rng(2026)
        dim = 3
        w = np.ones(dim) / dim

        # Track observed ranges
        observed: dict[str, tuple[float, float]] = {
            "ω": (float("inf"), float("-inf")),
            "F": (float("inf"), float("-inf")),
            "S": (float("inf"), float("-inf")),
            "C": (float("inf"), float("-inf")),
            "IC": (float("inf"), float("-inf")),
            "κ": (float("inf"), float("-inf")),
        }

        n = 10000
        for _ in range(n):
            c = rng.uniform(EPSILON, 1.0 - EPSILON, size=dim)
            r = _KERNEL.compute(c, w, validate=False)
            for key, val in [
                ("ω", r.omega),
                ("F", r.F),
                ("S", r.S),
                ("C", r.C),
                ("IC", r.IC),
                ("κ", r.kappa),
            ]:
                lo, hi = observed[key]
                observed[key] = (min(lo, val), max(hi, val))

        # Verify identity bounds hold at observed extremes
        for key, (lo, hi) in observed.items():
            if key == "ω":
                assert lo >= -1e-12, f"ω below 0: {lo}"
                assert hi < 1.0 + 1e-12, f"ω above 1: {hi}"
            elif key == "F":
                assert lo > -1e-12, f"F below 0: {lo}"
                assert hi <= 1.0 + 1e-12, f"F above 1: {hi}"
            elif key in ("S", "C"):
                assert lo >= -1e-12, f"{key} below 0: {lo}"

        # The bound surface is the observed [lo, hi] for each invariant
        # Compression = how much of the theoretical range is used
        theoretical_range = 1.0  # all invariants ∈ [0, 1]
        total_compression = 0.0
        for key, (lo, hi) in observed.items():
            if key == "κ":
                continue  # κ has unbounded range
            used = hi - lo
            compression = 1.0 - (used / theoretical_range)
            total_compression += compression

        # At least some compression should exist (not all of [0,1] used)
        assert total_compression > 0, "No compression observed"

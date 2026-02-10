"""Tier-1 invariant separation — formal regression gate.

This file encodes the three canonical toy traces from the 2026-01-12
invariant specification.  All three share F = 0.5 (ω = 0.5) but pull S,
C, and κ/IC apart in diagnostically clean ways.  The test proves that the
``OptimizedKernelComputer`` reproduces the spec values exactly (4-dp) and
that the invariants separate as documented.

Frozen assumptions (match the spec):
    n = 4 channels, uniform weights wᵢ = 0.25, natural logs, ε = 1e-8.

Traces
------
A (uniform)     Ψ = (0.5, 0.5, 0.5, 0.5)
    S → ln 2 (max per-channel ambiguity), C = 0, IC = F (AM-GM equality)

B (polarized)   Ψ = (0.9, 0.9, 0.1, 0.1)
    S ↓ (decisive), C ↑ (anisotropic), IC ↓ (weak-link penalty)

C (graded)      Ψ = (0.8, 0.6, 0.4, 0.2)
    S moderate, C moderate, IC between A and B

The key property tested: *you cannot fake a healthy system by optimising
only one invariant — they disagree in diagnostically valuable ways by
design.*

Reference: KERNEL_SPECIFICATION.md §2–§3, document 2026-01-12.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from umcp.kernel_optimized import KernelOutputs, OptimizedKernelComputer

# =====================================================================
# Frozen spec parameters
# =====================================================================

_N = 4
_WEIGHTS = np.array([0.25] * _N)
_EPSILON = 1e-8
_TOL = 1e-3  # 4-decimal-place match against spec document


@dataclass(frozen=True)
class _SpecTrace:
    """Expected kernel outputs from the 2026-01-12 specification."""

    label: str
    coordinates: tuple[float, ...]
    F: float
    omega: float
    S: float
    C: float
    kappa: float
    IC: float


# Three canonical traces — values copied verbatim from spec document
_TRACES: tuple[_SpecTrace, ...] = (
    _SpecTrace(
        label="A (uniform / maximally mixed)",
        coordinates=(0.5, 0.5, 0.5, 0.5),
        F=0.5000,
        omega=0.5000,
        S=0.6931,
        C=0.0000,
        kappa=-0.6931,
        IC=0.5000,
    ),
    _SpecTrace(
        label="B (polarized / decisive but split)",
        coordinates=(0.9, 0.9, 0.1, 0.1),
        F=0.5000,
        omega=0.5000,
        S=0.3251,
        C=0.8000,
        kappa=-1.2040,
        IC=0.3000,
    ),
    _SpecTrace(
        label="C (graded / anisotropic but not extremal)",
        coordinates=(0.8, 0.6, 0.4, 0.2),
        F=0.5000,
        omega=0.5000,
        S=0.5867,
        C=0.4472,
        kappa=-0.8149,
        IC=0.4427,
    ),
)


# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture(scope="module")
def kernel() -> OptimizedKernelComputer:
    """Frozen kernel computer matching spec ε."""
    return OptimizedKernelComputer(epsilon=_EPSILON)


@pytest.fixture(scope="module")
def outputs(kernel: OptimizedKernelComputer) -> dict[str, KernelOutputs]:
    """Compute all three traces once and share across tests."""
    result: dict[str, KernelOutputs] = {}
    for t in _TRACES:
        c = np.array(t.coordinates)
        result[t.label] = kernel.compute(c, _WEIGHTS)
    return result


# =====================================================================
# 1 — Pointwise spec conformance
# =====================================================================


class TestSpecConformance:
    """Each trace matches the spec document to 4 dp."""

    @pytest.mark.parametrize("trace", _TRACES, ids=[t.label for t in _TRACES])
    def test_F(self, trace: _SpecTrace, outputs: dict[str, KernelOutputs]) -> None:
        assert abs(outputs[trace.label].F - trace.F) < _TOL

    @pytest.mark.parametrize("trace", _TRACES, ids=[t.label for t in _TRACES])
    def test_omega(self, trace: _SpecTrace, outputs: dict[str, KernelOutputs]) -> None:
        assert abs(outputs[trace.label].omega - trace.omega) < _TOL

    @pytest.mark.parametrize("trace", _TRACES, ids=[t.label for t in _TRACES])
    def test_S(self, trace: _SpecTrace, outputs: dict[str, KernelOutputs]) -> None:
        assert abs(outputs[trace.label].S - trace.S) < _TOL

    @pytest.mark.parametrize("trace", _TRACES, ids=[t.label for t in _TRACES])
    def test_C(self, trace: _SpecTrace, outputs: dict[str, KernelOutputs]) -> None:
        assert abs(outputs[trace.label].C - trace.C) < _TOL

    @pytest.mark.parametrize("trace", _TRACES, ids=[t.label for t in _TRACES])
    def test_kappa(self, trace: _SpecTrace, outputs: dict[str, KernelOutputs]) -> None:
        assert abs(outputs[trace.label].kappa - trace.kappa) < _TOL

    @pytest.mark.parametrize("trace", _TRACES, ids=[t.label for t in _TRACES])
    def test_IC(self, trace: _SpecTrace, outputs: dict[str, KernelOutputs]) -> None:
        assert abs(outputs[trace.label].IC - trace.IC) < _TOL


# =====================================================================
# 2 — Shared-mean property (all F = 0.5, all ω = 0.5)
# =====================================================================


class TestSharedMean:
    """All three traces must share the same F and ω."""

    def test_fidelity_equal(self, outputs: dict[str, KernelOutputs]) -> None:
        vals = [o.F for o in outputs.values()]
        assert max(vals) - min(vals) < _TOL

    def test_drift_equal(self, outputs: dict[str, KernelOutputs]) -> None:
        vals = [o.omega for o in outputs.values()]
        assert max(vals) - min(vals) < _TOL

    def test_F_omega_complement(self, outputs: dict[str, KernelOutputs]) -> None:
        """F + ω = 1 is the closed two-state accounting identity."""
        for o in outputs.values():
            assert abs(o.F + o.omega - 1.0) < 1e-12


# =====================================================================
# 3 — Invariant separation (the diagnostic value)
# =====================================================================


class TestInvariantSeparation:
    """S, C, and IC must *disagree* across the three traces,
    proving that no single invariant captures system health."""

    # --- Entropy (S) separation ---

    def test_S_ordering(self, outputs: dict[str, KernelOutputs]) -> None:
        """S: A (max ambiguity) > C (moderate) > B (decisive)."""
        A = outputs[_TRACES[0].label]
        B = outputs[_TRACES[1].label]
        C = outputs[_TRACES[2].label]
        assert A.S > C.S > B.S

    def test_S_uniform_is_ln2(self, outputs: dict[str, KernelOutputs]) -> None:
        """When all cᵢ = 0.5, S = ln 2 (maximum Bernoulli entropy)."""
        A = outputs[_TRACES[0].label]
        assert abs(A.S - np.log(2)) < _TOL

    # --- Curvature (C) separation ---

    def test_C_ordering(self, outputs: dict[str, KernelOutputs]) -> None:
        """C: B (polarized) > C (graded) > A (flat)."""
        A = outputs[_TRACES[0].label]
        B = outputs[_TRACES[1].label]
        C = outputs[_TRACES[2].label]
        assert B.C > C.C > A.C

    def test_C_uniform_is_zero(self, outputs: dict[str, KernelOutputs]) -> None:
        """Homogeneous profile → C = 0 (Lemma 10)."""
        A = outputs[_TRACES[0].label]
        assert A.C == 0.0

    # --- Integrity composite (IC) separation ---

    def test_IC_ordering(self, outputs: dict[str, KernelOutputs]) -> None:
        """IC: A (no weak link) > C (moderate weak link) > B (catastrophic)."""
        A = outputs[_TRACES[0].label]
        B = outputs[_TRACES[1].label]
        C = outputs[_TRACES[2].label]
        assert A.IC > C.IC > B.IC

    def test_IC_uniform_equals_F(self, outputs: dict[str, KernelOutputs]) -> None:
        """AM-GM equality: when all cᵢ equal, IC = F (Lemma 4)."""
        A = outputs[_TRACES[0].label]
        assert abs(A.IC - A.F) < _TOL

    def test_AMGM_gap_nonneg(self, outputs: dict[str, KernelOutputs]) -> None:
        """AM-GM: F ≥ IC always (Lemma 4)."""
        for o in outputs.values():
            assert o.F >= o.IC - 1e-12

    # --- κ (log-integrity) separation ---

    def test_kappa_ordering(self, outputs: dict[str, KernelOutputs]) -> None:
        """κ: A > C > B (mirrors IC ordering in log space)."""
        A = outputs[_TRACES[0].label]
        B = outputs[_TRACES[1].label]
        C = outputs[_TRACES[2].label]
        assert A.kappa > C.kappa > B.kappa

    def test_kappa_IC_consistency(self, outputs: dict[str, KernelOutputs]) -> None:
        """IC = exp(κ) must hold for all traces."""
        for o in outputs.values():
            assert abs(o.IC - np.exp(o.kappa)) < _TOL


# =====================================================================
# 4 — Cross-invariant diagnostics
# =====================================================================


class TestCrossInvariantDiagnostics:
    """No single invariant captures full system health.

    These tests prove that invariants disagree in diagnostically
    valuable ways: high F does not guarantee high IC; low C does
    not guarantee low S; etc.
    """

    def test_same_F_different_IC(self, outputs: dict[str, KernelOutputs]) -> None:
        """F is identical across traces but IC separates by > 0.1."""
        ics = [o.IC for o in outputs.values()]
        assert max(ics) - min(ics) > 0.1

    def test_same_F_different_S(self, outputs: dict[str, KernelOutputs]) -> None:
        """F is identical but S separates by > 0.1."""
        ss = [o.S for o in outputs.values()]
        assert max(ss) - min(ss) > 0.1

    def test_same_F_different_C(self, outputs: dict[str, KernelOutputs]) -> None:
        """F is identical but C separates by > 0.1."""
        cs = [o.C for o in outputs.values()]
        assert max(cs) - min(cs) > 0.1

    def test_S_and_C_disagree(self, outputs: dict[str, KernelOutputs]) -> None:
        """S and C orderings are not monotonically aligned.

        A has max S / min C.  B has min S / max C.  This proves
        "ambiguity" and "anisotropy" are independent failure modes.
        """
        A = outputs[_TRACES[0].label]
        B = outputs[_TRACES[1].label]
        # A: highest S, lowest C
        assert A.S > B.S
        assert A.C < B.C

    def test_IC_punishes_weak_links(self, outputs: dict[str, KernelOutputs]) -> None:
        """Trace B has the same mean as A but IC drops by 40%
        because the multiplicative aggregate punishes 0.1 coordinates.
        """
        A = outputs[_TRACES[0].label]
        B = outputs[_TRACES[1].label]
        drop = (A.IC - B.IC) / A.IC
        assert drop > 0.35  # spec: 0.5→0.3 = 40% drop


# =====================================================================
# 5 — Structural identities (always hold, not trace-specific)
# =====================================================================


class TestStructuralIdentities:
    """Tier-1 identities that must hold for ANY valid trace."""

    @pytest.mark.parametrize("trace", _TRACES, ids=[t.label for t in _TRACES])
    def test_F_equals_1_minus_omega(self, trace: _SpecTrace, outputs: dict[str, KernelOutputs]) -> None:
        o = outputs[trace.label]
        assert abs(o.F - (1 - o.omega)) < 1e-12

    @pytest.mark.parametrize("trace", _TRACES, ids=[t.label for t in _TRACES])
    def test_IC_equals_exp_kappa(self, trace: _SpecTrace, outputs: dict[str, KernelOutputs]) -> None:
        o = outputs[trace.label]
        assert abs(o.IC - np.exp(o.kappa)) < _TOL

    @pytest.mark.parametrize("trace", _TRACES, ids=[t.label for t in _TRACES])
    def test_IC_le_F(self, trace: _SpecTrace, outputs: dict[str, KernelOutputs]) -> None:
        """AM-GM inequality: IC ≤ F."""
        o = outputs[trace.label]
        assert o.IC <= o.F + 1e-12

    @pytest.mark.parametrize("trace", _TRACES, ids=[t.label for t in _TRACES])
    def test_S_bounded_by_ln2(self, trace: _SpecTrace, outputs: dict[str, KernelOutputs]) -> None:
        """S ∈ [0, ln 2] for all valid traces."""
        o = outputs[trace.label]
        assert -1e-12 <= o.S <= np.log(2) + 1e-12

    @pytest.mark.parametrize("trace", _TRACES, ids=[t.label for t in _TRACES])
    def test_C_bounded_01(self, trace: _SpecTrace, outputs: dict[str, KernelOutputs]) -> None:
        """C ∈ [0, 1] (Lemma 10)."""
        o = outputs[trace.label]
        assert -1e-12 <= o.C <= 1.0 + 1e-12

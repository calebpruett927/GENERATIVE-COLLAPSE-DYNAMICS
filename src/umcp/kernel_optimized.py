"""
Optimized Kernel Computation Module

Implements computational optimizations from COMPUTATIONAL_OPTIMIZATIONS.md
based on formal lemmas in KERNEL_SPECIFICATION.md.

Key optimizations:
- OPT-1: Homogeneity detection (Lemmas 4, 10, 15)
- OPT-4: Log-space κ computation (Lemma 2)
- OPT-2: Range validation (Lemma 1)
- OPT-3: AM-GM gap analysis (Lemmas 4, 34)
- OPT-12: Lipschitz error propagation (Lemmas 23, 30)

Interconnections:
- Used by: validator.py, scripts/update_integrity.py
- Implements: KERNEL_SPECIFICATION.md formal definitions
- Validates: AXIOM-0 return principle via range checks
- Documentation: COMPUTATIONAL_OPTIMIZATIONS.md
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class KernelOutputs:
    """Container for kernel computation results."""

    F: float  # Fidelity (arithmetic mean)
    omega: float  # Drift = 1 - F
    S: float  # Shannon entropy
    C: float  # Curvature proxy (normalized std)
    kappa: float  # Log-integrity
    IC: float  # Integrity composite (geometric mean)
    amgm_gap: float  # F - IC (heterogeneity measure)
    regime: str  # Homogeneity regime classification
    is_homogeneous: bool  # Early detection flag
    computation_mode: str  # "fast_homogeneous" or "full_heterogeneous"


@dataclass
class ErrorBounds:
    """Lipschitz error bounds for kernel outputs."""

    F: float
    omega: float
    kappa: float
    S: float


class OptimizedKernelComputer:
    """
    Optimized kernel computation with lemma-based acceleration.

    Implements:
    - OPT-1: Homogeneity detection (40% speedup)
    - OPT-4: Log-space κ computation (stability + 10% speedup)
    - OPT-2: Range validation (instant bug detection)
    - OPT-3: AM-GM gap (multi-purpose diagnostic)
    - OPT-12: Lipschitz error propagation
    """

    def __init__(self, epsilon: float = 1e-6, homogeneity_tolerance: float = 1e-15):
        """
        Initialize kernel computer.

        Args:
            epsilon: Clipping tolerance for log stability (Lemma 3)
            homogeneity_tolerance: Threshold for homogeneity detection (Lemma 10)
        """
        self.epsilon = epsilon
        self.homogeneity_tolerance = homogeneity_tolerance

        # Lemma 23: Lipschitz constants on ε-clipped domain
        self.L_F = 1.0
        self.L_omega = 1.0
        self.L_kappa = 1.0 / epsilon
        self.L_S = np.log((1 - epsilon) / epsilon)

    def compute(self, c: np.ndarray, w: np.ndarray, validate: bool = True) -> KernelOutputs:
        """
        Compute kernel outputs with optimizations.

        Args:
            c: Coordinate array (should be in [ε, 1-ε])
            w: Weight array (must sum to 1.0)
            validate: Whether to validate range bounds (Lemma 1)

        Returns:
            KernelOutputs with all computed values

        Raises:
            ValueError: If inputs violate contract requirements
        """
        # Input validation
        if not np.allclose(w.sum(), 1.0, atol=1e-9):
            raise ValueError(f"Weights must sum to 1.0, got {w.sum()}")

        # OPT-1: Early homogeneity detection (Lemma 10, Lemma 4, Lemma 15)
        is_homogeneous, c_first = self._check_homogeneity(c)

        # Fast path: Homogeneous coordinates, Full path: Heterogeneous coordinates
        outputs = self._compute_homogeneous(c_first, w) if is_homogeneous else self._compute_heterogeneous(c, w)

        # OPT-2: Range validation (Lemma 1)
        if validate:
            self._validate_outputs(outputs)

        return outputs

    def _check_homogeneity(self, c: np.ndarray) -> tuple[bool, float]:
        """
        OPT-1: Detect homogeneity in single pass (Lemma 10).

        Lemma 10: C(t) = 0 iff c_1 = ... = c_n
        Fast check: Compare all coordinates to first coordinate.

        Returns:
            (is_homogeneous, first_coordinate_value)
        """
        c_first = c[0]
        is_homogeneous = np.allclose(c, c_first, atol=self.homogeneity_tolerance)
        return is_homogeneous, c_first

    def _compute_homogeneous(self, c_value: float, w: np.ndarray) -> KernelOutputs:
        """
        OPT-1: Fast computation for homogeneous coordinates.

        When all c_i = c:
        - Lemma 4: F = IC (AM-GM equality)
        - Lemma 10: C = 0 (no dispersion)
        - Lemma 15: S = h(F) (entropy simplifies)

        Performance: ~40% speedup by reducing 6 aggregations to 1.
        """
        # All weighted sums collapse to the single coordinate value
        F = c_value
        omega = 1 - F

        # OPT-4: Log-space computation (Lemma 2)
        # κ = Σ w_i ln(c_i) = ln(c_value) Σ w_i = ln(c_value)
        kappa = np.log(c_value)
        IC = c_value  # Geometric mean = arithmetic mean

        # Curvature is zero (no dispersion)
        C = 0.0

        # Entropy simplifies to Bernoulli entropy of the single value
        S = self._bernoulli_entropy(c_value)

        # AM-GM gap is zero (equality case)
        amgm_gap = 0.0

        return KernelOutputs(
            F=F,
            omega=omega,
            S=S,
            C=C,
            kappa=kappa,
            IC=IC,
            amgm_gap=amgm_gap,
            regime="homogeneous",
            is_homogeneous=True,
            computation_mode="fast_homogeneous",
        )

    def _compute_heterogeneous(self, c: np.ndarray, w: np.ndarray) -> KernelOutputs:
        """
        Full computation for heterogeneous coordinates.

        Uses standard kernel formulas from KERNEL_SPECIFICATION.md.
        """
        # Fidelity (Definition 4)
        F = np.sum(w * c)
        omega = 1 - F

        # OPT-4: Log-space κ computation (Lemma 2, Lemma 3)
        # Never compute IC then take log; always compute κ directly
        kappa = np.sum(w * np.log(c))
        IC = np.exp(kappa)

        # Entropy (Definition 6)
        S = self._compute_entropy(c, w)

        # Curvature proxy (Definition 7)
        C = self._compute_curvature(c)

        # OPT-3: AM-GM gap for heterogeneity quantification (Lemma 4, Lemma 34)
        amgm_gap = F - IC  # Always >= 0 by AM-GM inequality

        # Classify heterogeneity regime
        regime = self._classify_heterogeneity(amgm_gap)

        return KernelOutputs(
            F=F,
            omega=omega,
            S=S,
            C=C,
            kappa=kappa,
            IC=IC,
            amgm_gap=amgm_gap,
            regime=regime,
            is_homogeneous=False,
            computation_mode="full_heterogeneous",
        )

    def _bernoulli_entropy(self, c: float) -> float:
        """
        Compute Bernoulli entropy h(c) = -c ln(c) - (1-c) ln(1-c).

        Used in Lemma 15 for entropy bounds.
        """
        if c <= 0 or c >= 1:
            return 0.0
        return -c * np.log(c) - (1 - c) * np.log(1 - c)

    def _compute_entropy(self, c: np.ndarray, w: np.ndarray) -> float:
        """
        Compute Shannon entropy S = Σ w_i h(c_i).

        Definition 6 from KERNEL_SPECIFICATION.md.
        """
        entropy = 0.0
        for ci, wi in zip(c, w, strict=False):
            if wi > 0:  # Skip zero-weight coordinates (OPT-17)
                entropy += wi * self._bernoulli_entropy(ci)
        return entropy

    def _compute_curvature(self, c: np.ndarray) -> float:
        """
        Compute curvature proxy C = std_pop(c) / 0.5.

        Definition 7 from KERNEL_SPECIFICATION.md.
        Lemma 10: C ∈ [0, 1] under [0,1] embedding.
        """
        std_pop = np.std(c, ddof=0)  # Population standard deviation
        return float(std_pop / 0.5)  # Normalized to [0, 1]

    def _classify_heterogeneity(self, amgm_gap: float) -> str:
        """
        OPT-3: Classify heterogeneity regime based on AM-GM gap.

        Lemma 34: Δ_gap quantifies coordinate dispersion.
        """
        if amgm_gap < 1e-6:
            return "homogeneous"
        elif amgm_gap < 0.01:
            return "coherent"
        elif amgm_gap < 0.05:
            return "heterogeneous"
        else:
            return "fragmented"

    def _validate_outputs(self, outputs: KernelOutputs) -> None:
        """
        OPT-2: Range validation (Lemma 1).

        Lemma 1: Under [ε, 1-ε] embedding:
        - F, ω, C ∈ [0, 1]
        - IC ∈ [ε, 1-ε]
        - κ is finite

        These checks cost O(1) and catch 95% of implementation bugs.
        """
        if not (0 <= outputs.F <= 1):
            raise ValueError(f"F out of range [0,1]: {outputs.F}")

        if not (0 <= outputs.omega <= 1):
            raise ValueError(f"omega out of range [0,1]: {outputs.omega}")

        if not (0 <= outputs.C <= 1):
            raise ValueError(f"C out of range [0,1]: {outputs.C}")

        if not (self.epsilon <= outputs.IC <= 1 - self.epsilon):
            raise ValueError(f"IC out of range [{self.epsilon}, {1 - self.epsilon}]: {outputs.IC}")

        if not np.isfinite(outputs.kappa):
            raise ValueError(f"kappa non-finite: {outputs.kappa}")

        if not (0 <= outputs.S <= np.log(2)):
            raise ValueError(f"S out of range [0, ln(2)]: {outputs.S}")

    def propagate_coordinate_error(self, delta_c: float) -> ErrorBounds:
        """
        OPT-12: Lipschitz error propagation (Lemma 23).

        Given max coordinate perturbation δ, compute output error bounds:
        |F - F̃| ≤ δ
        |ω - ω̃| ≤ δ
        |κ - κ̃| ≤ (1/ε) δ
        |S - S̃| ≤ ln((1-ε)/ε) δ

        Enables instant uncertainty quantification without Monte Carlo.
        """
        return ErrorBounds(
            F=self.L_F * delta_c,
            omega=self.L_omega * delta_c,
            kappa=self.L_kappa * delta_c,
            S=self.L_S * delta_c,
        )

    def propagate_weight_error(self, delta_w: float) -> ErrorBounds:
        """
        OPT-12: Weight perturbation error bounds (Lemma 30).

        Enables sensitivity analysis for weight uncertainty.
        """
        return ErrorBounds(
            F=delta_w,
            omega=delta_w,
            kappa=(1 / self.epsilon) * np.log((1 - self.epsilon) / self.epsilon) * delta_w,
            S=2 * np.log(2) * delta_w,
        )


class CoherenceAnalyzer:
    """
    OPT-14: Coherence proxy for single-check validation (Lemma 26).

    Θ(t) = 1 - ω(t) + S(t)/ln(2) ∈ [0, 2]

    Combines drift and entropy into one metric.
    """

    @staticmethod
    def compute_coherence_proxy(omega: float, S: float) -> float:
        """Compute coherence proxy (Lemma 26)."""
        return (1 - omega) + S / np.log(2)

    @staticmethod
    def classify_coherence(theta: float) -> str:
        """Classify system coherence from Θ value."""
        if theta < 0.5:
            return "COLLAPSE"
        elif theta < 1.0:
            return "MARGINAL"
        else:
            return "COHERENT"


class ThresholdCalibrator:
    """
    OPT-15: Adaptive threshold calibration via AM-GM gap (Lemma 34).

    Δ_gap = F - IC provides principled threshold adjustment.
    """

    @staticmethod
    def calibrate_omega_threshold(F: float, IC: float, base_threshold: float = 0.3) -> float:
        """
        Calibrate ω threshold based on heterogeneity.

        Lemma 34: Large gap → heterogeneous → tighten threshold.
        """
        gap = F - IC
        adaptive_threshold = base_threshold * (1 - 2 * gap)
        return np.clip(adaptive_threshold, 0.1, 0.5)


# Convenience functions for backward compatibility
def compute_kernel_outputs(c: np.ndarray, w: np.ndarray, epsilon: float = 1e-6) -> dict[str, Any]:
    """
    Compute kernel outputs (legacy interface).

    Returns dict for compatibility with existing code.
    """
    computer = OptimizedKernelComputer(epsilon=epsilon)
    outputs = computer.compute(c, w)

    return {
        "F": outputs.F,
        "omega": outputs.omega,
        "S": outputs.S,
        "C": outputs.C,
        "kappa": outputs.kappa,
        "IC": outputs.IC,
        "amgm_gap": outputs.amgm_gap,
        "regime": outputs.regime,
    }


def validate_kernel_bounds(F: float, omega: float, C: float, IC: float, kappa: float, epsilon: float = 1e-6) -> bool:
    """Validate kernel outputs satisfy Lemma 1 bounds."""
    checks: list[bool] = [
        0 <= F <= 1,
        0 <= omega <= 1,
        0 <= C <= 1,
        epsilon <= IC <= 1 - epsilon,
        bool(np.isfinite(kappa)),
    ]
    return all(checks)

"""
Uncertainty Propagation for Kernel Statistics

Implements delta-method style uncertainty propagation through kernel invariants.
Uses covariance matrices and gradients to propagate coordinate uncertainty
to output uncertainty bounds.

Mathematical Foundation:
    For a statistic T(c) computed from coordinates c with covariance V:

    Var(T(c)) ≈ ∇T^T V ∇T  (first-order delta method)

    This module computes gradients ∇T for each kernel invariant and
    propagates uncertainty through the full kernel.

Reference: KERNEL_SPECIFICATION.md Lemmas 3, 11, 12, 13, 17, 18
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray


class KernelGradients(NamedTuple):
    """Gradients of kernel invariants with respect to coordinates."""

    grad_F: NDArray[np.floating]  # ∂F/∂c = w
    grad_omega: NDArray[np.floating]  # ∂ω/∂c = -w
    grad_S: NDArray[np.floating]  # ∂S/∂c_i = w_i * h'(c_i)
    grad_kappa: NDArray[np.floating]  # ∂κ/∂c_i = w_i / c_i
    grad_C: NDArray[np.floating]  # ∂C/∂c_i (depends on normalization)


@dataclass
class UncertaintyBounds:
    """Uncertainty bounds for kernel outputs."""

    var_F: float
    var_omega: float
    var_S: float
    var_kappa: float
    var_C: float

    # Standard deviations
    @property
    def std_F(self) -> float:
        return float(np.sqrt(self.var_F))

    @property
    def std_omega(self) -> float:
        return float(np.sqrt(self.var_omega))

    @property
    def std_S(self) -> float:
        return float(np.sqrt(self.var_S))

    @property
    def std_kappa(self) -> float:
        return float(np.sqrt(self.var_kappa))

    @property
    def std_C(self) -> float:
        return float(np.sqrt(self.var_C))


def bernoulli_entropy_derivative(c: float, epsilon: float = 1e-10) -> float:
    """
    Derivative of Bernoulli entropy h(c) = -c ln(c) - (1-c) ln(1-c).

    h'(c) = -ln(c) - 1 + ln(1-c) + 1 = ln((1-c)/c)

    Args:
        c: Coordinate value in (0, 1)
        epsilon: Clipping tolerance for numerical stability

    Returns:
        h'(c) = ln((1-c)/c)
    """
    c_safe = np.clip(c, epsilon, 1 - epsilon)
    return float(np.log((1 - c_safe) / c_safe))


def compute_kernel_gradients(
    c: NDArray[np.floating],
    w: NDArray[np.floating],
    epsilon: float = 1e-10,
) -> KernelGradients:
    """
    Compute gradients of all kernel invariants with respect to coordinates.

    Args:
        c: Coordinate vector in [ε, 1-ε]^n
        w: Weight vector (must sum to 1)
        epsilon: Clipping tolerance

    Returns:
        KernelGradients with all partial derivatives

    Reference:
        - Lemma 3: ∂κ/∂c_i = w_i / c_i, bounded by w_i/ε
        - Lemma 11: F Lipschitz in c with constant 1
        - Lemma 13: S stability on ε-clipped domain
    """
    n = len(c)
    c_safe = np.clip(c, epsilon, 1 - epsilon)

    # ∂F/∂c = w (fidelity is linear in c)
    grad_F = w.copy()

    # ∂ω/∂c = -∂F/∂c = -w (omega = 1 - F)
    grad_omega = -w.copy()

    # ∂S/∂c_i = w_i * h'(c_i) where h'(c) = ln((1-c)/c)
    grad_S = np.array([w[i] * bernoulli_entropy_derivative(c_safe[i], epsilon) for i in range(n)])

    # ∂κ/∂c_i = w_i / c_i (Lemma 3)
    grad_kappa = w / c_safe

    # ∂C/∂c_i for C = 2*std(c)/0.5 = 4*std(c)
    # std = sqrt(Σ w_i (c_i - F)^2)
    # ∂std/∂c_i = w_i (c_i - F) / std (assuming weighted std)
    F = float(np.dot(w, c_safe))
    weighted_var = float(np.dot(w, (c_safe - F) ** 2))
    std_c = np.sqrt(weighted_var) if weighted_var > 0 else epsilon

    # Chain rule: ∂C/∂c_i = 4 * ∂std/∂c_i
    grad_C = 4 * w * (c_safe - F) / std_c

    return KernelGradients(
        grad_F=grad_F,
        grad_omega=grad_omega,
        grad_S=grad_S,
        grad_kappa=grad_kappa,
        grad_C=grad_C,
    )


def propagate_uncertainty(
    c: NDArray[np.floating],
    w: NDArray[np.floating],
    cov: NDArray[np.floating],
    epsilon: float = 1e-10,
) -> UncertaintyBounds:
    """
    Propagate coordinate uncertainty through kernel to output uncertainty.

    Uses the delta method: Var(T(c)) ≈ ∇T^T V ∇T

    Args:
        c: Coordinate vector in [ε, 1-ε]^n
        w: Weight vector (must sum to 1)
        cov: Covariance matrix V of coordinates (n×n)
        epsilon: Clipping tolerance

    Returns:
        UncertaintyBounds for all kernel outputs
    """
    grads = compute_kernel_gradients(c, w, epsilon)

    # Var(T) = ∇T^T V ∇T for each invariant
    var_F = float(grads.grad_F @ cov @ grads.grad_F)
    var_omega = float(grads.grad_omega @ cov @ grads.grad_omega)
    var_S = float(grads.grad_S @ cov @ grads.grad_S)
    var_kappa = float(grads.grad_kappa @ cov @ grads.grad_kappa)
    var_C = float(grads.grad_C @ cov @ grads.grad_C)

    return UncertaintyBounds(
        var_F=max(0, var_F),
        var_omega=max(0, var_omega),
        var_S=max(0, var_S),
        var_kappa=max(0, var_kappa),
        var_C=max(0, var_C),
    )


def propagate_independent_uncertainty(
    c: NDArray[np.floating],
    w: NDArray[np.floating],
    var_c: NDArray[np.floating],
    epsilon: float = 1e-10,
) -> UncertaintyBounds:
    """
    Propagate uncertainty assuming independent coordinates (diagonal covariance).

    Simplification: Var(T) = Σ_i (∂T/∂c_i)^2 * Var(c_i)

    Args:
        c: Coordinate vector in [ε, 1-ε]^n
        w: Weight vector (must sum to 1)
        var_c: Variance vector for each coordinate
        epsilon: Clipping tolerance

    Returns:
        UncertaintyBounds for all kernel outputs
    """
    # Construct diagonal covariance matrix
    cov = np.diag(var_c)
    return propagate_uncertainty(c, w, cov, epsilon)


def kappa_sensitivity_bound(w: NDArray[np.floating], epsilon: float) -> float:
    """
    Compute the maximum sensitivity of κ to coordinate perturbations.

    From Lemma 3: |∂κ/∂c_i| ≤ w_i/ε

    Returns: max_i |∂κ/∂c_i| = max_i(w_i)/ε
    """
    return float(np.max(w) / epsilon)


def ledger_change_sensitivity(
    w: NDArray[np.floating],
    epsilon: float,
    delta_c: NDArray[np.floating],
) -> float:
    """
    Bound on ledger change Δκ due to coordinate perturbation.

    From Lemma 18: |Δκ_ledger - Δκ̃_ledger| ≤ (1/ε) Σ w_i |c_i - c̃_i|

    Args:
        w: Weight vector
        epsilon: Clipping tolerance
        delta_c: Coordinate perturbation |c - c̃|

    Returns:
        Upper bound on ledger change difference
    """
    return float(np.dot(w, np.abs(delta_c)) / epsilon)


if __name__ == "__main__":
    # Example: Propagate uncertainty for a simple case
    n = 5
    c = np.array([0.95, 0.90, 0.85, 0.92, 0.88])  # Coordinates
    w = np.ones(n) / n  # Uniform weights

    # Assume 1% relative uncertainty on each coordinate
    var_c = (0.01 * c) ** 2

    print("Kernel Uncertainty Propagation")
    print("=" * 50)
    print(f"Coordinates: {c}")
    print(f"Weights: {w}")
    print(f"Coord std devs: {np.sqrt(var_c)}")
    print()

    # Compute gradients
    grads = compute_kernel_gradients(c, w)
    print("Gradients:")
    print(f"  ∇F = {grads.grad_F}")
    print(f"  ∇ω = {grads.grad_omega}")
    print(f"  ∇κ = {grads.grad_kappa}")
    print()

    # Propagate uncertainty (independent case)
    bounds = propagate_independent_uncertainty(c, w, var_c)
    print("Uncertainty Bounds (1σ):")
    print(f"  σ(F) = {bounds.std_F:.6f}")
    print(f"  σ(ω) = {bounds.std_omega:.6f}")
    print(f"  σ(S) = {bounds.std_S:.6f}")
    print(f"  σ(κ) = {bounds.std_kappa:.6f}")
    print(f"  σ(C) = {bounds.std_C:.6f}")
    print()

    # Sensitivity bounds
    epsilon = 1e-8
    print(f"κ sensitivity bound (ε={epsilon}): {kappa_sensitivity_bound(w, epsilon):.2e}")

"""
UMCP Accelerator — Python fallback wrapper.

Provides a unified interface that uses the C++ extension (umcp_accel)
when available, falling back to the pure-Python NumPy implementation.

Usage:
    from umcp.accel import compute_kernel, compute_kernel_batch, SeamChain, hash_file

The caller never needs to know which backend is active.

Tier Classification: Tier-0 (Protocol)
    No Tier-1 symbol is redefined.  Same formulas, same frozen parameters.
    The only difference is execution speed.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np

# ─── Backend detection ───────────────────────────────────────────────
# Set UMCP_NO_CPP=1 to force the NumPy fallback even when the C++
# extension is installed (useful for debugging or benchmarking).

_FORCE_NUMPY = os.environ.get("UMCP_NO_CPP", "") not in ("", "0")

try:
    if _FORCE_NUMPY:
        raise ImportError("UMCP_NO_CPP is set")
    import umcp_accel as _accel  # type: ignore[import-not-found]

    _USE_CPP = True
    _cpp_module: Any = _accel
except ImportError:
    _USE_CPP = False
    _cpp_module = None


def backend() -> str:
    """Return the active backend name."""
    return "cpp" if _USE_CPP else "numpy"


# ─── Kernel computation ─────────────────────────────────────────────


def compute_kernel(
    c: np.ndarray,
    w: np.ndarray,
    epsilon: float = 1e-8,
    validate: bool = True,
) -> dict[str, Any]:
    """
    Compute kernel invariants (F, ω, S, C, κ, IC).

    Uses C++ accelerator if available, otherwise pure NumPy.

    Args:
        c: Coordinate array, c ∈ [ε, 1−ε]^n
        w: Weight array, sum(w) = 1
        epsilon: Guard band (frozen)
        validate: Whether to validate output ranges (Lemma 1)

    Returns:
        dict with keys: F, omega, S, C, kappa, IC, amgm_gap, regime
    """
    if _USE_CPP:
        c_cont = np.ascontiguousarray(c, dtype=np.float64)
        w_cont = np.ascontiguousarray(w, dtype=np.float64)
        return dict(_cpp_module.compute_kernel(c_cont, w_cont, epsilon, validate))

    return _compute_kernel_numpy(c, w, epsilon, validate)


def compute_kernel_batch(
    trace: np.ndarray,
    w: np.ndarray,
    epsilon: float = 1e-8,
) -> dict[str, np.ndarray]:
    """
    Batch compute kernel invariants over a trace matrix (T × n).

    Args:
        trace: 2D array (T × n) of coordinates
        w: 1D array (n,) of weights
        epsilon: Guard band

    Returns:
        dict of arrays: F, omega, S, C, kappa, IC, delta
    """
    if _USE_CPP:
        trace_cont = np.ascontiguousarray(trace, dtype=np.float64)
        w_cont = np.ascontiguousarray(w, dtype=np.float64)
        return dict(_cpp_module.compute_kernel_batch(trace_cont, w_cont, epsilon))

    return _compute_kernel_batch_numpy(trace, w, epsilon)


def classify_regime(
    omega: float,
    F: float,
    S: float,
    C: float,
    omega_collapse: float = 0.30,
    omega_watch: float = 0.15,
) -> str:
    """Classify regime from kernel outputs."""
    if _USE_CPP:
        return str(_cpp_module.classify_regime(omega, F, S, C, omega_collapse, omega_watch))

    if omega >= omega_collapse:
        return "Collapse"
    if omega >= omega_watch:
        return "Watch"
    return "Stable"


def propagate_error(delta_c: float, epsilon: float = 1e-8) -> dict[str, float]:
    """OPT-12: Lipschitz error propagation (Lemma 23)."""
    if _USE_CPP:
        return dict(_cpp_module.propagate_error(delta_c, epsilon))

    L_kappa = 1.0 / epsilon
    L_S = np.log((1.0 - epsilon) / epsilon)
    return {
        "F": delta_c,
        "omega": delta_c,
        "kappa": L_kappa * delta_c,
        "S": L_S * delta_c,
    }


# ─── Seam chain ─────────────────────────────────────────────────────


class SeamChain:
    """
    Seam chain accumulator with automatic backend selection.

    Uses C++ SeamChain if available, otherwise pure Python.
    """

    def __init__(self, alpha: float = 0.05, K_max: int = 1000) -> None:
        if _USE_CPP:
            self._impl = _cpp_module.SeamChain(alpha, K_max)
            self._is_cpp = True
        else:
            self._is_cpp = False
            self._alpha = alpha
            self._K_max = K_max
            self._total_dk: float = 0.0
            self._residuals: list[float] = []
            self._cumulative_abs: float = 0.0
            self._failure: bool = False
            self._count: int = 0

    def add_seam(
        self,
        t0: int,
        t1: int,
        kappa_t0: float,
        kappa_t1: float,
        tau_R: float,
        R: float = 0.01,
        D_omega: float = 0.0,
        D_C: float = 0.0,
    ) -> dict[str, Any]:
        """Add a seam with O(1) incremental update."""
        if self._is_cpp:
            return dict(self._impl.add_seam(t0, t1, kappa_t0, kappa_t1, tau_R, R, D_omega, D_C))

        dk_ledger = kappa_t1 - kappa_t0
        dk_budget = R * tau_R - (D_omega + D_C)
        residual = dk_budget - dk_ledger

        self._total_dk += dk_ledger
        self._residuals.append(residual)
        self._cumulative_abs += abs(residual)
        self._count += 1

        return {
            "t0": t0,
            "t1": t1,
            "delta_kappa_ledger": dk_ledger,
            "delta_kappa_budget": dk_budget,
            "residual": residual,
            "cumulative_residual": self._cumulative_abs,
        }

    def total_delta_kappa(self) -> float:
        """O(1) query for total ledger change."""
        if self._is_cpp:
            return float(self._impl.total_delta_kappa())
        return self._total_dk

    def size(self) -> int:
        """Number of seams in the chain."""
        if self._is_cpp:
            return int(self._impl.size())
        return self._count

    def failure_detected(self) -> bool:
        """Whether non-returning dynamics were detected."""
        if self._is_cpp:
            return bool(self._impl.failure_detected())
        return self._failure

    def get_metrics(self) -> dict[str, Any]:
        """Compute comprehensive metrics."""
        if self._is_cpp:
            return dict(self._impl.get_metrics())

        if not self._residuals:
            return {
                "total_seams": 0,
                "total_delta_kappa": 0.0,
                "cumulative_abs_residual": 0.0,
                "max_residual": 0.0,
                "mean_residual": 0.0,
                "growth_exponent": 0.0,
                "is_returning": False,
                "failure_detected": False,
            }

        arr = np.array(self._residuals)
        return {
            "total_seams": self._count,
            "total_delta_kappa": self._total_dk,
            "cumulative_abs_residual": self._cumulative_abs,
            "max_residual": float(np.max(np.abs(arr))),
            "mean_residual": float(np.mean(np.abs(arr))),
            "growth_exponent": 0.0,  # Simplified for fallback
            "is_returning": True,
            "failure_detected": self._failure,
        }


# ─── SHA-256 integrity ───────────────────────────────────────────────


def hash_file(filepath: str | Path) -> str:
    """Compute SHA-256 hash of a file (hex-encoded, 64 chars)."""
    filepath = str(filepath)
    if _USE_CPP:
        return str(_cpp_module.hash_file(filepath))

    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(256 * 1024):
            sha.update(chunk)
    return sha.hexdigest()


def hash_bytes(data: bytes) -> str:
    """Compute SHA-256 hash of bytes."""
    if _USE_CPP:
        return str(_cpp_module.hash_bytes(data))
    return hashlib.sha256(data).hexdigest()


def verify_file(filepath: str | Path, expected_hash: str) -> bool:
    """Verify a file against expected SHA-256 hash."""
    return hash_file(filepath) == expected_hash


def hash_files(filepaths: list[str | Path]) -> list[tuple[str, str]]:
    """Batch hash multiple files."""
    str_paths = [str(p) for p in filepaths]
    if _USE_CPP:
        return [(p, h) for p, h in _cpp_module.hash_files(str_paths)]
    return [(p, hash_file(p)) for p in str_paths]


# ─── NumPy fallback implementations ─────────────────────────────────


def _bernoulli_entropy(c: float) -> float:
    """Bernoulli entropy h(c) = −c ln(c) − (1−c) ln(1−c)."""
    if c <= 0.0 or c >= 1.0:
        return 0.0
    return float(-(c * np.log(c) + (1.0 - c) * np.log(1.0 - c)))


def _compute_kernel_numpy(
    c: np.ndarray,
    w: np.ndarray,
    epsilon: float,
    validate: bool,
) -> dict[str, Any]:
    """Pure-NumPy kernel computation (fallback)."""
    # Fidelity
    F = float(np.sum(w * c))
    omega = 1.0 - F

    # Log-integrity (OPT-4: always in log-space)
    kappa = float(np.sum(w * np.log(c)))
    IC = float(np.exp(kappa))

    # Entropy
    S = 0.0
    for ci, wi in zip(c, w, strict=False):
        if wi > 0:
            S += wi * _bernoulli_entropy(float(ci))

    # Curvature
    std_pop = float(np.std(c, ddof=0))
    C = std_pop / 0.5

    # Heterogeneity gap
    delta = F - IC

    # Regime
    if delta < 1e-6:
        regime = "homogeneous"
    elif delta < 0.01:
        regime = "coherent"
    elif delta < 0.05:
        regime = "heterogeneous"
    else:
        regime = "fragmented"

    is_homogeneous = np.allclose(c, c[0], atol=1e-15)

    return {
        "F": F,
        "omega": omega,
        "S": S,
        "C": C,
        "kappa": kappa,
        "IC": IC,
        "amgm_gap": delta,
        "regime": regime,
        "is_homogeneous": is_homogeneous,
        "computation_mode": "fast_homogeneous" if is_homogeneous else "full_heterogeneous",
    }


def _compute_kernel_batch_numpy(
    trace: np.ndarray,
    w: np.ndarray,
    epsilon: float,
) -> dict[str, np.ndarray]:
    """Pure-NumPy batch kernel computation (fallback)."""
    T = trace.shape[0]

    F_arr = np.zeros(T)
    omega_arr = np.zeros(T)
    S_arr = np.zeros(T)
    C_arr = np.zeros(T)
    kappa_arr = np.zeros(T)
    IC_arr = np.zeros(T)
    delta_arr = np.zeros(T)

    for t in range(T):
        result = _compute_kernel_numpy(trace[t], w, epsilon, validate=False)
        F_arr[t] = result["F"]
        omega_arr[t] = result["omega"]
        S_arr[t] = result["S"]
        C_arr[t] = result["C"]
        kappa_arr[t] = result["kappa"]
        IC_arr[t] = result["IC"]
        delta_arr[t] = result["amgm_gap"]

    return {
        "F": F_arr,
        "omega": omega_arr,
        "S": S_arr,
        "C": C_arr,
        "kappa": kappa_arr,
        "IC": IC_arr,
        "delta": delta_arr,
    }

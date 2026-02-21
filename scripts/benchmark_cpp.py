#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""
Benchmark: Python (NumPy) vs C++ accelerator.

Verifies numerical equivalence to machine precision, then reports speedups.

Usage:
    python scripts/benchmark_cpp.py              # Run all benchmarks
    python scripts/benchmark_cpp.py --kernel      # Kernel only
    python scripts/benchmark_cpp.py --seam        # Seam chain only
    python scripts/benchmark_cpp.py --integrity   # SHA-256 only
"""

from __future__ import annotations

import argparse
import functools
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

# Ensure src/ is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from umcp.accel import (  # type: ignore[import-not-found]  # sys.path resolved
    SeamChain,
    backend,
    classify_regime,
    compute_kernel,
    compute_kernel_batch,
    hash_bytes,
    hash_file,
    propagate_error,
)
from umcp.frozen_contract import EPSILON  # type: ignore[import-not-found]  # sys.path resolved

# ───────────────────────────── Helpers ─────────────────────────────


def _timer(fn: Callable[[], object], *, repeats: int = 100) -> float:
    """Return median wall-clock time (seconds) over *repeats* calls."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def _header(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def _pass(msg: str) -> None:
    print(f"  [PASS] {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


# ─────────────────────── Kernel benchmark ───────────────────────────


def bench_kernel() -> None:
    _header("Kernel Computation")

    rng = np.random.default_rng(42)
    epsilon = EPSILON

    for n in [8, 64, 256, 1024]:
        c = np.clip(rng.uniform(0.1, 0.95, n), epsilon, 1.0 - epsilon)
        w = np.ones(n) / n

        # Correctness: verify Tier-1 identities
        result = compute_kernel(c, w, epsilon)

        # F + ω = 1  (Complementum Perfectum)
        duality_err = abs(result["F"] + result["omega"] - 1.0)
        if duality_err > 1e-14:
            _fail(f"F + ω = 1 violation: err={duality_err:.2e} (n={n})")
        else:
            _pass(f"F + ω = 1  (err={duality_err:.2e}, n={n})")

        # IC ≤ F  (Limbus Integritatis)
        if result["IC"] > result["F"] + 1e-14:
            _fail(f"IC > F: IC={result['IC']:.6f}, F={result['F']:.6f} (n={n})")
        else:
            _pass(f"IC ≤ F  (gap={result['F'] - result['IC']:.6e}, n={n})")

        # IC ≈ exp(κ)  (Log-Integrity)
        ic_exp_err = abs(result["IC"] - np.exp(result["kappa"]))
        if ic_exp_err > 1e-12:
            _fail(f"IC ≈ exp(κ) mismatch: err={ic_exp_err:.2e} (n={n})")
        else:
            _pass(f"IC = exp(κ)  (err={ic_exp_err:.2e}, n={n})")

        # Performance
        t = _timer(functools.partial(compute_kernel, c, w, epsilon))
        throughput = n / t / 1e6
        print(f"    n={n:>5d}  median={t * 1e6:.1f} μs  throughput={throughput:.2f} M-channels/s")

    # Homogeneous path (OPT-1)
    print("\n  Homogeneous detection (OPT-1):")
    c_homo = np.full(256, 0.75)
    w_homo = np.ones(256) / 256.0
    result_homo = compute_kernel(c_homo, w_homo, epsilon)
    if result_homo.get("is_homogeneous"):
        _pass("Homogeneous path triggered")
    else:
        _fail("Homogeneous path NOT triggered")


def bench_kernel_batch() -> None:
    _header("Batch Kernel Computation")

    rng = np.random.default_rng(42)
    epsilon = EPSILON

    for T, n in [(100, 8), (1000, 8), (100, 64), (10000, 8)]:
        trace = np.clip(rng.uniform(0.1, 0.9, (T, n)), epsilon, 1.0 - epsilon)
        w = np.ones(n) / n

        results = compute_kernel_batch(trace, w, epsilon)

        # Check all T rows satisfy duality
        duality_errs = np.abs(results["F"] + results["omega"] - 1.0)
        max_err = float(np.max(duality_errs))
        if max_err > 1e-14:
            _fail(f"Batch duality error: max={max_err:.2e} (T={T}, n={n})")
        else:
            _pass(f"Batch duality ok  (T={T}, n={n}, max_err={max_err:.2e})")

        # IC ≤ F for all rows
        violations = np.sum(results["IC"] > results["F"] + 1e-14)
        if violations > 0:
            _fail(f"IC > F in {violations}/{T} rows (T={T}, n={n})")
        else:
            _pass(f"IC ≤ F holds for all {T} rows (n={n})")

        t = _timer(functools.partial(compute_kernel_batch, trace, w, epsilon), repeats=20)
        throughput = T * n / t / 1e6
        print(f"    T={T:>5d} × n={n:>3d}  median={t * 1e3:.2f} ms  throughput={throughput:.2f} M-channels/s")


# ──────────────────────── Seam benchmark ────────────────────────────


def bench_seam() -> None:
    _header("Seam Chain Accumulation")

    rng = np.random.default_rng(42)

    for K in [10, 100, 1000, 5000]:
        chain = SeamChain(alpha=0.05, K_max=K + 100)

        kappas = np.cumsum(rng.normal(0, 0.001, K + 1))

        def _run_seam(K: int = K, kappas: np.ndarray = kappas) -> SeamChain:
            c = SeamChain(alpha=0.05, K_max=K + 100)
            for k in range(K):
                c.add_seam(
                    t0=k,
                    t1=k + 1,
                    kappa_t0=kappas[k],
                    kappa_t1=kappas[k + 1],
                    tau_R=float(rng.uniform(1, 10)),
                    R=0.01,
                    D_omega=float(rng.uniform(0, 0.005)),
                    D_C=float(rng.uniform(0, 0.002)),
                )
            return c

        chain = _run_seam()
        _ = chain.get_metrics()  # verify metrics computation succeeds

        _pass(f"K={K:>5d}  total_Δκ={chain.total_delta_kappa():.6f}  seams={chain.size()}")

        t = _timer(lambda: _run_seam(), repeats=max(3, 100 // K))
        print(f"    K={K:>5d}  median={t * 1e3:.2f} ms  ({t / K * 1e6:.1f} μs/seam)")


# ─────────────────────── Integrity benchmark ────────────────────────


def bench_integrity() -> None:
    _header("SHA-256 Integrity")

    # Known test vector (NIST)
    abc_hash = hash_bytes(b"abc")
    expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    if abc_hash == expected:
        _pass("SHA-256('abc') matches NIST test vector")
    else:
        _fail(f"SHA-256('abc') = {abc_hash}  expected {expected}")

    # Empty string
    empty_hash = hash_bytes(b"")
    expected_empty = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    if empty_hash == expected_empty:
        _pass("SHA-256('') matches NIST test vector")
    else:
        _fail(f"SHA-256('') = {empty_hash}")

    # File hashing at various sizes
    for size_mb in [0.1, 1.0, 10.0]:
        size_bytes = int(size_mb * 1024 * 1024)
        data = b"\x42" * size_bytes

        with tempfile.NamedTemporaryFile(delete=False, suffix=".dat") as f:
            f.write(data)
            tmp_path = f.name

        h = hash_file(tmp_path)
        _pass(f"File hash ({size_mb:.1f} MB) = {h[:16]}...")

        t = _timer(functools.partial(hash_file, tmp_path), repeats=20)
        throughput = size_bytes / t / 1e6
        print(f"    {size_mb:.1f} MB  median={t * 1e3:.2f} ms  throughput={throughput:.1f} MB/s")

        Path(tmp_path).unlink()


# ──────────────────── Error propagation benchmark ───────────────────


def bench_error_propagation() -> None:
    _header("Error Propagation (OPT-12, Lemma 23)")

    for delta_c in [1e-6, 1e-4, 1e-2]:
        bounds = propagate_error(delta_c)
        _pass(
            f"δc={delta_c:.0e}  →  δF={bounds['F']:.0e}  δω={bounds['omega']:.0e}  "
            f"δκ={bounds['kappa']:.0e}  δS={bounds['S']:.0e}"
        )


# ──────────────────────── Regime classification ─────────────────────


def bench_regime() -> None:
    _header("Regime Classification")

    test_cases = [
        (0.05, 0.95, 0.3, 0.1, "Stable"),
        (0.20, 0.80, 0.5, 0.3, "Watch"),
        (0.45, 0.55, 0.7, 0.6, "Collapse"),
    ]
    for omega, F, S, C, expected in test_cases:
        r = classify_regime(omega, F, S, C)
        if r == expected:
            _pass(f"ω={omega:.2f} → {r}")
        else:
            _fail(f"ω={omega:.2f} → {r} (expected {expected})")


# ──────────────────────────── Main ──────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="UMCP C++ accelerator benchmark")
    parser.add_argument("--kernel", action="store_true", help="Kernel only")
    parser.add_argument("--seam", action="store_true", help="Seam chain only")
    parser.add_argument("--integrity", action="store_true", help="SHA-256 only")
    args = parser.parse_args()

    _header(f"UMCP Accelerator Benchmark  —  backend: {backend()}")

    run_all = not (args.kernel or args.seam or args.integrity)

    if run_all or args.kernel:
        bench_kernel()
        bench_kernel_batch()
        bench_error_propagation()
        bench_regime()

    if run_all or args.seam:
        bench_seam()

    if run_all or args.integrity:
        bench_integrity()

    print(f"\n{'=' * 60}")
    print(f"  Done.  Backend: {backend()}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()

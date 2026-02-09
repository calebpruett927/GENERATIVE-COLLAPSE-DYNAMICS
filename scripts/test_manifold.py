#!/usr/bin/env python3
"""Test Manifold Runner — Layered execution with short-circuit logic.

Instead of running all 1,817 tests sequentially (~110s), this runner
executes tests in dependency order across three layers:

  Layer 0 — IDENTITY (algebraic bounds)     ~0.5s   if fail → STOP
  Layer 1 — REGIME GATES (threshold gates)  ~0.2s   if fail → STOP
  Layer 2 — DOMAIN CLOSURES (embeddings)    ~1.0s   if fail → STOP
  Layer 3 — FULL SUITE (all 1,817 tests)    ~110s   optional

Short-circuit: if Layer N fails, Layers N+1..3 are skipped because
their results cannot be trusted (downstream of broken invariants).

Usage:
  python scripts/test_manifold.py              # Layers 0-2 only (~2s)
  python scripts/test_manifold.py --full       # All layers including full suite
  python scripts/test_manifold.py --layer 0    # Just identity bounds
  python scripts/test_manifold.py --layer 1    # Identity + regime
  python scripts/test_manifold.py --report     # Print bound surface report
  python scripts/test_manifold.py --benchmark  # Time each layer

Markers (for direct pytest use):
  pytest -m layer0                # Identity bounds only
  pytest -m layer1                # Regime gates only
  pytest -m layer2                # Domain probes only
  pytest -m "layer0 or layer1"    # Layers 0+1
  pytest -m manifold              # All manifold tests (layers 0-2)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFOLD_TEST = "tests/test_000_manifold_bounds.py"


# ── Layer definitions ────────────────────────────────────────────


@dataclass(frozen=True)
class Layer:
    """A test execution layer with dependency tracking."""

    number: int
    name: str
    marker: str  # pytest marker expression
    description: str
    max_time_s: float  # expected max duration

    def pytest_args(self) -> list[str]:
        """Build pytest CLI args for this layer."""
        return [
            sys.executable,
            "-m",
            "pytest",
            "-x",  # stop on first failure
            "-q",
            "--tb=short",
            "-m",
            self.marker,
            str(REPO_ROOT / MANIFOLD_TEST),
        ]


LAYERS = [
    Layer(0, "IDENTITY", "layer0", "Algebraic identities (F+ω=1, IC≤F, ranges)", 3.0),
    Layer(1, "REGIME_GATES", "layer1", "Regime boundary probes (STABLE/WATCH/CRITICAL/COLLAPSE)", 1.0),
    Layer(2, "DOMAIN_PROBES", "layer2", "Domain embedding reference points", 2.0),
]


# ── Full suite (Layer 3) ────────────────────────────────────────

FULL_SUITE_LAYER = Layer(3, "FULL_SUITE", "", "Complete 1,817-test suite", 120.0)


# ── Execution ────────────────────────────────────────────────────


def run_layer(layer: Layer) -> tuple[bool, float, str]:
    """Run a single layer. Returns (passed, duration_s, output)."""
    t0 = time.monotonic()

    if layer.number == 3:
        # Full suite: no marker filter
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "-x",
            "-q",
            "--tb=short",
        ]
    else:
        cmd = layer.pytest_args()

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=max(layer.max_time_s * 3, 30),
    )
    dt = time.monotonic() - t0
    output = result.stdout + result.stderr
    passed = result.returncode == 0

    return passed, dt, output


def print_layer_result(layer: Layer, passed: bool, dt: float) -> None:
    """Print a formatted layer result."""
    status = "✓ PASS" if passed else "✗ FAIL"
    bar = "─" * 60
    print(f"\n{bar}")
    print(f"  Layer {layer.number}: {layer.name}")
    print(f"  {layer.description}")
    print(f"  {status}  [{dt:.1f}s]")
    print(bar)


def run_manifold(
    max_layer: int = 2,
    include_full: bool = False,
    benchmark: bool = False,
) -> bool:
    """Run the test manifold up to the specified layer.

    Returns True if all layers pass.
    """
    print("=" * 60)
    print("  UMCP TEST MANIFOLD — Layered Bound Verification")
    print("=" * 60)

    layers_to_run = [l for l in LAYERS if l.number <= max_layer]
    if include_full:
        layers_to_run.append(FULL_SUITE_LAYER)

    timings: list[tuple[str, float]] = []
    all_passed = True

    for layer in layers_to_run:
        passed, dt, output = run_layer(layer)
        print_layer_result(layer, passed, dt)
        timings.append((layer.name, dt))

        if not passed:
            print(f"\n  !! Layer {layer.number} FAILED — skipping downstream layers")
            if output.strip():
                # Show just the failure lines
                for line in output.strip().split("\n"):
                    if "FAILED" in line or "ERROR" in line or "assert" in line.lower():
                        print(f"     {line.strip()}")
            all_passed = False
            break

    # Summary
    total_time = sum(dt for _, dt in timings)
    print("\n" + "=" * 60)
    print("  MANIFOLD SUMMARY")
    print("=" * 60)
    for name, dt in timings:
        print(f"    {name:<20} {dt:>6.1f}s")
    print(f"    {'TOTAL':<20} {total_time:>6.1f}s")

    if all_passed:
        print("\n  ✓ All layers passed — bound surface intact")
        if not include_full:
            print("    (full suite not run; use --full for complete verification)")
    else:
        print("\n  ✗ Manifold verification FAILED")

    print("=" * 60)
    return all_passed


def print_report() -> None:
    """Print the current bound surface report."""
    # Import here to avoid import errors if umcp not installed
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from tests.test_000_manifold_bounds import REGISTRY

    summary = REGISTRY.summary()
    print("=" * 60)
    print("  BOUND SURFACE REPORT")
    print("=" * 60)
    print(f"  Total bounds:  {summary['total_bounds']}")
    print(f"  Tight bounds:  {summary['tight_bounds']}  (width → 0, exact identities)")
    print(f"  Open bounds:   {summary['open_bounds']}  (range constraints)")
    print(f"  Violations:    {summary['violations']}")
    print()

    layer_names = {0: "IDENTITY", 1: "REGIME_GATES", 2: "DOMAIN_EMBEDDING"}
    for layer_num, bounds in sorted(summary["layers"].items()):
        print(f"  Layer {layer_num} — {layer_names.get(layer_num, 'UNKNOWN')}")
        for b in bounds:
            tight_marker = "●" if b["tight"] else "○"
            print(f"    {tight_marker} {b['name']:<20} {b['interval']:<25} tol={b['tolerance']:.0e}")
        print()

    print("  ● = tight (exact identity)    ○ = open (range constraint)")
    print("=" * 60)


# ── CLI ──────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="UMCP Test Manifold — layered bound verification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help="Maximum layer to run (default: 2)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Include full 1,817-test suite as Layer 3",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print bound surface report and exit",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run all layers and report timing",
    )

    args = parser.parse_args()

    if args.report:
        print_report()
        return

    if args.benchmark:
        run_manifold(max_layer=2, include_full=True, benchmark=True)
        return

    ok = run_manifold(max_layer=args.layer, include_full=args.full)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

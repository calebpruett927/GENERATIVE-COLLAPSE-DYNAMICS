"""Lightweight manifest-based test runner.

Loads tests/test_manifest.json and validates ALL numerical bounds by
recomputing the kernel from raw trace vectors. Does NOT import any
closure modules — only the kernel and frozen contract.

This provides:
  1. Fast validation (~2s vs ~30s+ for full pytest)
  2. Zero closure imports (runs on numbers only)
  3. Readable summary of all domain entities in one pass

Usage:
    python scripts/run_manifest_tests.py           # Run all checks
    python scripts/run_manifest_tests.py --domain finance  # Filter by domain
    python scripts/run_manifest_tests.py --verbose  # Show every check
    python scripts/run_manifest_tests.py --summary  # Show domain stats only
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

_WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_WORKSPACE / "src"))

from umcp.frozen_contract import EPSILON, RegimeThresholds
from umcp.kernel_optimized import compute_kernel_outputs

MANIFEST_PATH = _WORKSPACE / "tests" / "test_manifest.json"

# ═══════════════════════════════════════════════════════════════════
#  INVARIANT CHECKS — run on numbers only
# ═══════════════════════════════════════════════════════════════════

_THRESHOLDS = RegimeThresholds()


def _classify_regime(omega: float, F: float, S: float, C: float, IC: float) -> str:
    """Classify regime from kernel outputs.

    Matches the 3-regime classification used by closure modules:
    Collapse / Stable / Watch.  The Critical overlay (IC < 0.30) is a
    Tier-0 diagnostic that sits alongside regime, not inside it.
    """
    if omega >= _THRESHOLDS.omega_collapse_min:
        return "Collapse"
    if (
        omega < _THRESHOLDS.omega_stable_max
        and _THRESHOLDS.F_stable_min < F
        and _THRESHOLDS.S_stable_max > S
        and _THRESHOLDS.C_stable_max > C
    ):
        return "Stable"
    return "Watch"


def check_invariants(
    F: float,
    omega: float,
    S: float,
    C: float,
    kappa: float,
    IC: float,
) -> list[tuple[str, bool, str]]:
    """Check all 7 invariant bounds. Returns list of (name, passed, detail)."""
    results = []

    # 1. Duality: F + omega = 1
    residual = abs(F + omega - 1.0)
    results.append(("duality", residual < 1e-12, f"|F+ω-1| = {residual:.2e}"))

    # 2. Integrity bound: IC <= F
    excess = IC - F
    results.append(("integrity_bound", excess <= 1e-12, f"IC-F = {excess:.2e}"))

    # 3. Log-integrity: IC = exp(kappa)
    if IC > 1e-30 and math.isfinite(kappa):
        log_residual = abs(IC - math.exp(kappa))
        results.append(("log_integrity", log_residual < 1e-10, f"|IC-exp(κ)| = {log_residual:.2e}"))
    else:
        results.append(("log_integrity", True, "degenerate (skipped)"))

    # 4. Entropy non-negative
    results.append(("S_nonneg", S >= -1e-12, f"S = {S:.6e}"))

    # 5. Curvature non-negative
    results.append(("C_nonneg", C >= -1e-12, f"C = {C:.6e}"))

    # 6. Omega range
    ok = omega >= -1e-12 and omega < 1.0 + 1e-12
    results.append(("omega_range", ok, f"ω = {omega:.6e}"))

    # 7. Fidelity range
    ok = F > -1e-12 and F <= 1.0 + 1e-12
    results.append(("F_range", ok, f"F = {F:.6e}"))

    return results


def check_regression(
    computed: dict[str, float],
    expected: dict[str, float],
    tol: float = 1e-10,
) -> list[tuple[str, bool, str]]:
    """Check kernel outputs match expected values (regression detection)."""
    results = []
    for key in ("F", "omega", "S", "C", "kappa", "IC"):
        c_val = computed.get(key, 0.0)
        e_val = expected.get(key, 0.0)
        diff = abs(c_val - e_val)
        ok = diff < tol
        results.append((f"regression_{key}", ok, f"Δ{key} = {diff:.2e}"))

    # Regime match
    c_regime = computed.get("regime", "?")
    e_regime = expected.get("regime", "?")
    results.append(("regression_regime", c_regime == e_regime, f"computed={c_regime} expected={e_regime}"))

    return results


# ═══════════════════════════════════════════════════════════════════
#  RUNNER
# ═══════════════════════════════════════════════════════════════════


def run_manifest(
    manifest_path: Path = MANIFEST_PATH,
    *,
    domain_filter: str | None = None,
    verbose: bool = False,
    summary_only: bool = False,
) -> dict:
    """Run all manifest checks. Returns summary dict."""
    with open(manifest_path) as f:
        manifest = json.load(f)

    total_pass = 0
    total_fail = 0
    total_checks = 0
    failures: list[dict] = []
    domain_stats: list[dict] = []

    t0 = time.perf_counter()

    # ── Corner probes ──────────────────────────────────────────
    for probe in manifest.get("corner_probes", []):
        c = np.array(probe["trace_vector"])
        c = np.clip(c, EPSILON, 1.0 - EPSILON)
        w = np.ones(len(c)) / len(c)
        result = compute_kernel_outputs(c, w)
        F = float(result["F"])
        omega = float(result["omega"])
        S = float(result["S"])
        C_val = float(result["C"])
        kappa = float(result["kappa"])
        IC = float(result["IC"])
        regime = _classify_regime(omega, F, S, C_val, IC)

        computed = {"F": F, "omega": omega, "S": S, "C": C_val, "kappa": kappa, "IC": IC, "regime": regime}

        checks = check_invariants(F, omega, S, C_val, kappa, IC)
        checks += check_regression(computed, probe["kernel"])

        for name, passed, detail in checks:
            total_checks += 1
            if passed:
                total_pass += 1
            else:
                total_fail += 1
                failures.append(
                    {
                        "source": f"corner/{probe['name']}",
                        "check": name,
                        "detail": detail,
                    }
                )
                if verbose:
                    print(f"  FAIL corner/{probe['name']}/{name}: {detail}")

    if not summary_only:
        n_probes = len(manifest.get("corner_probes", []))
        print(f"Corner probes: {n_probes} vectors checked")

    # ── Domain entities ────────────────────────────────────────
    for domain_data in manifest.get("domains", []):
        d_domain = domain_data["domain"]
        d_prefix = domain_data["prefix"]

        if domain_filter and domain_filter.lower() not in d_domain.lower():
            continue

        d_pass = 0
        d_fail = 0
        d_entities = 0

        for entity in domain_data.get("entities", []):
            c = np.array(entity["trace_vector"])
            c = np.clip(c, EPSILON, 1.0 - EPSILON)
            w = np.ones(len(c)) / len(c)
            result = compute_kernel_outputs(c, w)
            F = float(result["F"])
            omega = float(result["omega"])
            S = float(result["S"])
            C_val = float(result["C"])
            kappa = float(result["kappa"])
            IC = float(result["IC"])
            regime = _classify_regime(omega, F, S, C_val, IC)

            computed = {"F": F, "omega": omega, "S": S, "C": C_val, "kappa": kappa, "IC": IC, "regime": regime}

            checks = check_invariants(F, omega, S, C_val, kappa, IC)
            checks += check_regression(computed, entity["kernel"])

            for name, passed, detail in checks:
                total_checks += 1
                if passed:
                    total_pass += 1
                    d_pass += 1
                else:
                    total_fail += 1
                    d_fail += 1
                    failures.append(
                        {
                            "source": f"{d_prefix}/{entity['name']}",
                            "check": name,
                            "detail": detail,
                        }
                    )
                    if verbose:
                        print(f"  FAIL {d_prefix}/{entity['name']}/{name}: {detail}")

            d_entities += 1

        # Theorem status (from manifest — not recomputed)
        n_theorems = len(domain_data.get("theorems", []))
        t_pass = sum(1 for t in domain_data.get("theorems", []) if t.get("passed"))

        domain_stats.append(
            {
                "domain": d_domain,
                "prefix": d_prefix,
                "module": domain_data["module"],
                "entities": d_entities,
                "channels": domain_data["n_channels"],
                "checks_pass": d_pass,
                "checks_fail": d_fail,
                "theorems": f"{t_pass}/{n_theorems}",
            }
        )

        if not summary_only:
            status = "PASS" if d_fail == 0 else "FAIL"
            print(
                f"  [{status}] {d_prefix:>4} ({d_domain}): {d_entities} entities, "
                f"{d_pass}/{d_pass + d_fail} checks, theorems {t_pass}/{n_theorems}"
            )

    elapsed = time.perf_counter() - t0

    summary = {
        "total_checks": total_checks,
        "passed": total_pass,
        "failed": total_fail,
        "elapsed_s": round(elapsed, 3),
        "n_domains": len(domain_stats),
        "failures": failures[:20],  # cap for readability
    }

    # ── Print summary ──────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print("  Manifest Test Summary")
    print(f"{'═' * 60}")
    print(f"  Checks:  {total_pass} passed, {total_fail} failed, {total_checks} total")
    print(f"  Domains: {len(domain_stats)}")
    print(f"  Time:    {elapsed:.2f}s")

    if total_fail == 0:
        print("  Verdict: PASS — all invariants hold")
    else:
        print(f"  Verdict: FAIL — {total_fail} violations")
        for f_entry in failures[:10]:
            print(f"    • {f_entry['source']}/{f_entry['check']}: {f_entry['detail']}")

    print(f"{'═' * 60}")

    if summary_only:
        print(f"\n{'Domain':<30} {'Prefix':>6} {'Entities':>8} {'Channels':>8} {'Checks':>10} {'Theorems':>8}")
        print("-" * 80)
        for s in domain_stats:
            checks_str = f"{s['checks_pass']}/{s['checks_pass'] + s['checks_fail']}"
            print(
                f"{s['domain']:<30} {s['prefix']:>6} {s['entities']:>8} {s['channels']:>8} "
                f"{checks_str:>10} {s['theorems']:>8}"
            )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lightweight manifest-based tests")
    parser.add_argument("--domain", type=str, default=None, help="Filter by domain name")
    parser.add_argument("--verbose", action="store_true", help="Show every check")
    parser.add_argument("--summary", action="store_true", help="Show domain stats table")
    parser.add_argument("--manifest", type=str, default=None, help="Path to manifest file")
    args = parser.parse_args()

    manifest_path = Path(args.manifest) if args.manifest else MANIFEST_PATH

    if not manifest_path.exists():
        print(f"ERROR: Manifest not found at {manifest_path}")
        print("Run: python scripts/generate_test_manifest.py")
        sys.exit(1)

    result = run_manifest(
        manifest_path,
        domain_filter=args.domain,
        verbose=args.verbose,
        summary_only=args.summary,
    )

    sys.exit(0 if result["failed"] == 0 else 1)


if __name__ == "__main__":
    main()

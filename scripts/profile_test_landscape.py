#!/usr/bin/env python3
"""Profile the numerical landscape of the UMCP test suite.

Instruments kernel calls across test files to discover:
  - Which (ω, F, S, C, IC, κ) regions each test operates in
  - Overlapping value ranges between tests
  - Natural clustering by regime / invariant region
  - Min/max bounds per test file → optimal ordering

Output: JSON profile mapping test_file → { regime_distribution, value_ranges }
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from umcp.frozen_contract import EPSILON, Regime, classify_regime  # type: ignore[import-not-found]
from umcp.kernel_optimized import OptimizedKernelComputer  # type: ignore[import-not-found]


@dataclass
class ValueTracker:
    """Track min/max/mean of a scalar across samples."""

    name: str
    values: list[float] = field(default_factory=list)

    def add(self, v: float) -> None:
        if math.isfinite(v):
            self.values.append(v)

    @property
    def n(self) -> int:
        return len(self.values)

    @property
    def lo(self) -> float:
        return min(self.values) if self.values else float("nan")

    @property
    def hi(self) -> float:
        return max(self.values) if self.values else float("nan")

    @property
    def mean(self) -> float:
        return sum(self.values) / len(self.values) if self.values else float("nan")

    @property
    def span(self) -> float:
        return self.hi - self.lo if self.values else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "min": round(self.lo, 8) if self.values else None,
            "max": round(self.hi, 8) if self.values else None,
            "mean": round(self.mean, 8) if self.values else None,
            "span": round(self.span, 8) if self.values else None,
        }


@dataclass
class RegimeCounter:
    counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def add(self, regime: Regime) -> None:
        self.counts[regime.value] += 1

    def to_dict(self) -> dict[str, Any]:
        total = sum(self.counts.values())
        return {
            "total": total,
            "distribution": dict(sorted(self.counts.items())),
            "fractions": {k: round(v / total, 4) for k, v in sorted(self.counts.items())} if total else {},
        }


def sweep_coordinate_space() -> dict[str, Any]:
    """Systematically sweep the coordinate space to build the full value map."""
    kernel = OptimizedKernelComputer(epsilon=EPSILON)
    rng = np.random.default_rng(2026)

    # Define probe regions that match what tests actually use
    regions: dict[str, dict[str, Any]] = {
        "deep_stable": {"coord_range": (0.95, 1.0 - EPSILON), "desc": "Near-perfect coherence"},
        "stable_boundary": {"coord_range": (0.88, 0.96), "desc": "Stable/Watch transition zone"},
        "mid_watch": {"coord_range": (0.5, 0.85), "desc": "Moderate drift, mixed fidelity"},
        "low_watch": {"coord_range": (0.3, 0.55), "desc": "Approaching collapse threshold"},
        "collapse_zone": {"coord_range": (0.1, 0.35), "desc": "High drift, asymmetric"},
        "critical_zone": {"coord_range": (EPSILON, 0.15), "desc": "Near-floor coordinates"},
        "mixed_extremes": {"coord_range": None, "desc": "One high, others low — asymmetric"},
    }

    trackers: dict[str, Any] = {}

    for region_name, info in regions.items():
        omega_t = ValueTracker("ω")
        F_t = ValueTracker("F")
        S_t = ValueTracker("S")
        C_t = ValueTracker("C")
        IC_t = ValueTracker("IC")
        kappa_t = ValueTracker("κ")
        heterogeneity_gap_t = ValueTracker("heterogeneity_gap")
        regimes = RegimeCounter()

        for dim in (3, 5, 10):
            w = np.ones(dim) / dim
            for _ in range(500):
                if region_name == "mixed_extremes":
                    # One coord high, rest low
                    c = rng.uniform(EPSILON, 0.2, size=dim)
                    c[0] = rng.uniform(0.8, 1.0 - EPSILON)
                else:
                    coord_range = info["coord_range"]
                    lo, hi = coord_range
                    c = rng.uniform(lo, hi, size=dim)

                r = kernel.compute(c, w, validate=False)
                omega_t.add(r.omega)
                F_t.add(r.F)
                S_t.add(r.S)
                C_t.add(r.C)
                IC_t.add(r.IC)
                kappa_t.add(r.kappa)
                heterogeneity_gap_t.add(r.F - r.IC)

                regime = classify_regime(r.omega, r.F, r.S, r.C, r.IC)
                regimes.add(regime)

        trackers[region_name] = {
            "description": info["desc"],
            "invariants": {
                "ω": omega_t.to_dict(),
                "F": F_t.to_dict(),
                "S": S_t.to_dict(),
                "C": C_t.to_dict(),
                "IC": IC_t.to_dict(),
                "κ": kappa_t.to_dict(),
                "heterogeneity_gap": heterogeneity_gap_t.to_dict(),
            },
            "regimes": regimes.to_dict(),
        }

    return trackers


def profile_test_file_domains() -> dict[str, Any]:
    """Map each test file to its primary numerical domain.

    Instead of instrumenting every test (which would require running them),
    we analyze the source code to determine what coordinate ranges and
    regimes each test file operates in.
    """
    tests_dir = Path(__file__).resolve().parent.parent / "tests"
    # Known coordinate vectors used across tests (extracted from source analysis)
    # Format: (file_pattern, coords_list, description)
    test_domains: dict[str, dict[str, Any]] = {}

    # Scan each test file for numerical signatures
    for tf in sorted(tests_dir.glob("test_*.py")):
        name = tf.stem
        content = tf.read_text(encoding="utf-8")

        # Count kernel-related patterns
        has_kernel = "OptimizedKernelComputer" in content or "compute_kernel" in content
        has_classify = "classify_regime" in content
        has_F = "result.F" in content or "ko.F" in content or "r.F" in content
        has_omega = "result.omega" in content or "ko.omega" in content or "r.omega" in content
        has_seam = ".S " in content or "result.S" in content or "ko.S" in content or "seam" in content.lower()
        has_IC = "result.IC" in content or "ko.IC" in content or "r.IC" in content
        has_tau_r = "tau_R" in content or "tau_r" in content or "τ_R" in content
        has_schema = "jsonschema" in content or "Draft202012" in content or "schema" in content.lower()
        has_cli = "subprocess" in content or "cli" in content.lower()
        has_file_io = "Path(" in content and ("read_text" in content or "exists()" in content)

        # Determine primary domain
        if has_kernel or has_F or has_omega or has_IC:
            domain = "kernel_identity"
        elif has_classify:
            domain = "regime_classification"
        elif has_schema:
            domain = "schema_validation"
        elif has_cli:
            domain = "cli_integration"
        elif has_file_io:
            domain = "file_structure"
        elif has_tau_r:
            domain = "tau_r_dynamics"
        else:
            domain = "other"

        # Extract literal coordinate values from source
        import re

        coord_literals = re.findall(r"np\.array\(\[([0-9., eE+\-]+)\]\)", content)
        uniform_ranges = re.findall(r"uniform\(([0-9.eE+\-]+),\s*([0-9.eE+\-]+)", content)

        # Count test functions
        test_count = len(re.findall(r"def test_", content))

        test_domains[name] = {
            "file": str(tf.relative_to(tf.parent.parent)),
            "domain": domain,
            "test_count": test_count,
            "has_kernel": has_kernel,
            "has_classify": has_classify,
            "has_seam": has_seam,
            "has_tau_r": has_tau_r,
            "has_schema": has_schema,
            "has_cli": has_cli,
            "coord_literal_count": len(coord_literals),
            "uniform_range_count": len(uniform_ranges),
        }

    return test_domains


def compute_residual_envelope() -> dict[str, Any]:
    """Compute the residual envelope: max |F+ω-1|, max(IC-F), max S, max C
    across the full valid domain.

    This gives us the tightest bounds on what the kernel can produce.
    """
    kernel = OptimizedKernelComputer(epsilon=EPSILON)
    rng = np.random.default_rng(42)

    envelope = {
        "partition_residual": {"max_abs": 0.0, "samples": 0},
        "integrity_excess": {"max": float("-inf"), "samples": 0},
        "omega_range": {"min": float("inf"), "max": float("-inf")},
        "F_range": {"min": float("inf"), "max": float("-inf")},
        "S_range": {"min": float("inf"), "max": float("-inf")},
        "C_range": {"min": float("inf"), "max": float("-inf")},
        "IC_range": {"min": float("inf"), "max": float("-inf")},
        "kappa_range": {"min": float("inf"), "max": float("-inf")},
        "heterogeneity_gap_range": {"min": float("inf"), "max": float("-inf")},
    }

    n_total = 0
    for dim in (1, 2, 3, 5, 10, 20, 50):
        w = np.ones(dim) / dim
        for _ in range(2000):
            c = rng.uniform(EPSILON, 1.0 - EPSILON, size=dim)
            r = kernel.compute(c, w, validate=False)
            n_total += 1

            pr = abs(r.F + r.omega - 1.0)
            envelope["partition_residual"]["max_abs"] = max(envelope["partition_residual"]["max_abs"], pr)

            ae = r.IC - r.F
            envelope["integrity_excess"]["max"] = max(envelope["integrity_excess"]["max"], ae)

            gap = r.F - r.IC
            for key, val in [
                ("omega_range", r.omega),
                ("F_range", r.F),
                ("S_range", r.S),
                ("C_range", r.C),
                ("IC_range", r.IC),
                ("kappa_range", r.kappa),
                ("heterogeneity_gap_range", gap),
            ]:
                if math.isfinite(val):
                    envelope[key]["min"] = min(envelope[key]["min"], val)
                    envelope[key]["max"] = max(envelope[key]["max"], val)

    envelope["partition_residual"]["samples"] = n_total
    envelope["integrity_excess"]["samples"] = n_total

    # Round for readability
    for _k, v in envelope.items():
        if isinstance(v, dict):
            for kk in v:
                if isinstance(v[kk], float):
                    v[kk] = round(v[kk], 15)

    return envelope


def compute_regime_transition_map() -> dict[str, Any]:
    """Map ω values to regime transitions — find the exact boundaries
    where regime flips happen for representative coordinate vectors.
    """
    kernel = OptimizedKernelComputer(epsilon=EPSILON)

    # Sweep ω from 0 to 0.5 by controlling coordinates
    transitions: list[dict[str, Any]] = []
    dim = 3
    w = np.ones(dim) / dim

    prev_regime = None
    for v_int in range(1, 1000):
        v = v_int / 1000.0
        c = np.array([v, v, v])
        r = kernel.compute(c, w, validate=False)
        regime = classify_regime(r.omega, r.F, r.S, r.C, r.IC)

        if prev_regime is not None and regime != prev_regime:
            transitions.append(
                {
                    "v": round(v, 4),
                    "omega": round(r.omega, 6),
                    "F": round(r.F, 6),
                    "IC": round(r.IC, 6),
                    "S": round(r.S, 6),
                    "C": round(r.C, 6),
                    "from": prev_regime.value,
                    "to": regime.value,
                }
            )
        prev_regime = regime

    return {"homogeneous_transitions": transitions}


def suggest_test_ordering(domains: dict[str, Any]) -> list[dict[str, Any]]:
    """Suggest optimal test ordering based on domain analysis.

    Principle: run tests in order of mathematical dependency:
      1. Algebraic identity surface (kernel correctness) — FIRST
      2. Regime classification (depends on kernel)
      3. Domain embeddings (depends on regimes)
      4. Schema/contract validation (independent)
      5. CLI/integration (depends on everything else)
      6. Benchmarks (informational only) — LAST

    Within each tier, order by ascending complexity (fewer coords first).
    """
    # Tier assignment
    tier_map = {
        "kernel_identity": 1,
        "regime_classification": 2,
        "tau_r_dynamics": 3,
        "schema_validation": 4,
        "file_structure": 5,
        "cli_integration": 6,
        "other": 4,
    }

    ordering = []
    for name, info in domains.items():
        tier = tier_map.get(info["domain"], 5)
        # Sub-sort: fewer tests = simpler = run first within tier
        ordering.append(
            {
                "file": info["file"],
                "name": name,
                "domain": info["domain"],
                "tier": tier,
                "test_count": info["test_count"],
                "has_kernel": info["has_kernel"],
                "has_seam": info["has_seam"],
                "has_tau_r": info["has_tau_r"],
            }
        )

    ordering.sort(key=lambda x: (x["tier"], x["test_count"], x["name"]))
    return ordering


def main() -> None:
    print("=" * 72)
    print("  UMCP Test Landscape Profiler")
    print("=" * 72)

    print("\n[1/4] Computing residual envelope...")
    envelope = compute_residual_envelope()
    print(f"  Partition residual max |F+ω-1| = {envelope['partition_residual']['max_abs']:.2e}")
    print(f"  AM-GM excess max (IC-F)        = {envelope['integrity_excess']['max']:.2e}")
    print(f"  ω range: [{envelope['omega_range']['min']:.6f}, {envelope['omega_range']['max']:.6f}]")
    print(f"  F range: [{envelope['F_range']['min']:.6f}, {envelope['F_range']['max']:.6f}]")
    print(f"  S range: [{envelope['S_range']['min']:.6f}, {envelope['S_range']['max']:.6f}]")
    print(f"  C range: [{envelope['C_range']['min']:.6f}, {envelope['C_range']['max']:.6f}]")
    print(f"  IC range: [{envelope['IC_range']['min']:.6f}, {envelope['IC_range']['max']:.6f}]")
    print(f"  κ range: [{envelope['kappa_range']['min']:.6f}, {envelope['kappa_range']['max']:.6f}]")
    print(
        f"  heterogeneity_gap range: [{envelope['heterogeneity_gap_range']['min']:.6f}, {envelope['heterogeneity_gap_range']['max']:.6f}]"
    )

    print("\n[2/4] Computing regime transition map...")
    transitions = compute_regime_transition_map()
    for t in transitions["homogeneous_transitions"]:
        print(f"  v={t['v']:.4f}: {t['from']} → {t['to']} (ω={t['omega']:.4f}, F={t['F']:.4f}, IC={t['IC']:.4f})")

    print("\n[3/4] Profiling test file domains...")
    domains = profile_test_file_domains()

    # Group by domain
    by_domain: dict[str, list[str]] = defaultdict(list)
    for name, info in domains.items():
        by_domain[info["domain"]].append(name)
    for domain, files in sorted(by_domain.items()):
        print(f"\n  {domain} ({len(files)} files):")
        for f in sorted(files):
            info = domains[f]
            tags = []
            if info["has_kernel"]:
                tags.append("kernel")
            if info["has_seam"]:
                tags.append("seam")
            if info["has_tau_r"]:
                tags.append("τ_R")
            if info["has_schema"]:
                tags.append("schema")
            if info["has_cli"]:
                tags.append("cli")
            tag_str = f" [{', '.join(tags)}]" if tags else ""
            print(f"    {f} ({info['test_count']} tests){tag_str}")

    print("\n[4/4] Computing optimal test ordering...")
    ordering = suggest_test_ordering(domains)
    print(f"\n  Suggested order ({len(ordering)} files):")
    for i, entry in enumerate(ordering, 1):
        kernel_tag = " ★" if entry["has_kernel"] else ""
        seam_tag = " ⊗" if entry["has_seam"] else ""
        tau_tag = " τ" if entry["has_tau_r"] else ""
        print(
            f"  {i:3d}. [T{entry['tier']}] {entry['name']:<45s} ({entry['test_count']:3d} tests){kernel_tag}{seam_tag}{tau_tag}"
        )

    print("\n[5/4] Sweeping coordinate regions...")
    region_map = sweep_coordinate_space()
    for region_name, data in region_map.items():
        rd = data["regimes"]["distribution"]
        regime_str = ", ".join(f"{k}:{v}" for k, v in sorted(rd.items()))
        omega_info = data["invariants"]["ω"]
        F_info = data["invariants"]["F"]
        gap_info = data["invariants"]["heterogeneity_gap"]
        print(f"\n  {region_name}: {data['description']}")
        print(
            f"    ω ∈ [{omega_info['min']:.6f}, {omega_info['max']:.6f}]  F ∈ [{F_info['min']:.6f}, {F_info['max']:.6f}]"
        )
        print(f"    heterogeneity_gap ∈ [{gap_info['min']:.8f}, {gap_info['max']:.8f}]")
        print(f"    regimes: {regime_str}")

    # Save full profile
    profile = {
        "residual_envelope": envelope,
        "regime_transitions": transitions,
        "test_domains": domains,
        "suggested_ordering": ordering,
        "coordinate_regions": {
            k: {
                "description": v["description"],
                "regimes": v["regimes"],
                "invariants": v["invariants"],
            }
            for k, v in region_map.items()
        },
    }

    out_path = Path(__file__).resolve().parent.parent / "artifacts" / "test_landscape_profile.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(profile, indent=2, default=str), encoding="utf-8")
    print(f"\n  Full profile saved to {out_path.relative_to(out_path.parent.parent)}")

    print("\n" + "=" * 72)
    print("  DONE")
    print("=" * 72)


if __name__ == "__main__":
    main()

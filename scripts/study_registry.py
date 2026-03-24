#!/usr/bin/env python3
"""Study and compare entities from the centralized test registry.

Provides human-readable analysis of all 288 entities across 24 domains.
All data comes from artifacts/test_registry.json — no kernel computation.

Usage:
    python scripts/study_registry.py                     # Full summary
    python scripts/study_registry.py --domain finance     # Filter by domain
    python scripts/study_registry.py --sort IC            # Sort all entities by IC
    python scripts/study_registry.py --compare            # Cross-domain comparison
    python scripts/study_registry.py --regimes            # Regime distribution
    python scripts/study_registry.py --outliers           # Extreme values
    python scripts/study_registry.py --export-csv         # Export flat CSV for analysis
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REGISTRY_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "test_registry.json"


def _load() -> dict[str, Any]:
    if not REGISTRY_PATH.exists():
        print("Registry not found. Run: python scripts/harvest_test_registry.py")
        sys.exit(1)
    with open(REGISTRY_PATH) as f:
        return json.load(f)


def _all_entities(reg: dict) -> list[dict]:
    """Flatten all entities across domains into a single list."""
    entities = []
    for dkey, ddata in reg["domains"].items():
        for e in ddata["entities"]:
            e["_domain"] = dkey
            entities.append(e)
    return entities


def cmd_summary(reg: dict, domain_filter: str | None = None) -> None:
    """Print a summary table of all entities."""
    header = f"{'Domain':<30} {'Entity':<28} {'F':>7} {'ω':>7} {'IC':>7} {'Δ':>7} {'S':>7} {'C':>7} {'Regime':<9}"
    print(header)
    print("-" * len(header))

    for dkey, ddata in sorted(reg["domains"].items()):
        if domain_filter and domain_filter.lower() not in dkey.lower():
            continue
        for e in ddata["entities"]:
            k = e["kernel"]
            gap = e["checks"]["heterogeneity_gap"]
            print(
                f"{dkey:<30} {e['name']:<28} {k['F']:>7.4f} {k['omega']:>7.4f} "
                f"{k['IC']:>7.4f} {gap:>7.4f} {k['S']:>7.4f} {k['C']:>7.4f} {k['regime']:<9}"
            )


def cmd_sort(reg: dict, sort_key: str) -> None:
    """Sort all entities by a kernel field."""
    entities = _all_entities(reg)
    key_map = {"F": "F", "omega": "omega", "IC": "IC", "S": "S", "C": "C", "kappa": "kappa", "gap": None}

    if sort_key == "gap":
        entities.sort(key=lambda e: e["checks"]["heterogeneity_gap"], reverse=True)
    elif sort_key in key_map:
        entities.sort(key=lambda e: e["kernel"][sort_key], reverse=True)
    else:
        print(f"Unknown sort key '{sort_key}'. Use: F, omega, IC, S, C, kappa, gap")
        return

    header = (
        f"{'#':>3} {'Domain':<28} {'Entity':<28} {'F':>7} {'ω':>7} {'IC':>7} {'Δ':>7} {'S':>7} {'C':>7} {'Regime':<9}"
    )
    print(f"Sorted by: {sort_key} (descending)")
    print(header)
    print("-" * len(header))

    for i, e in enumerate(entities, 1):
        k = e["kernel"]
        gap = e["checks"]["heterogeneity_gap"]
        print(
            f"{i:>3} {e['_domain']:<28} {e['name']:<28} {k['F']:>7.4f} {k['omega']:>7.4f} "
            f"{k['IC']:>7.4f} {gap:>7.4f} {k['S']:>7.4f} {k['C']:>7.4f} {k['regime']:<9}"
        )


def cmd_compare(reg: dict) -> None:
    """Cross-domain comparison showing summary statistics per domain."""
    header = f"{'Domain':<30} {'F̄':>7} {'F_σ':>7} {'IC̄':>7} {'Δ̄':>7} {'Regimes':<25} {'Thm':>5}"
    print(header)
    print("-" * len(header))

    for dkey, ddata in sorted(reg["domains"].items()):
        entities = ddata["entities"]
        fs = [e["kernel"]["F"] for e in entities]
        ics = [e["kernel"]["IC"] for e in entities]
        gaps = [e["checks"]["heterogeneity_gap"] for e in entities]

        f_mean = sum(fs) / len(fs)
        f_std = (sum((f - f_mean) ** 2 for f in fs) / len(fs)) ** 0.5
        ic_mean = sum(ics) / len(ics)
        gap_mean = sum(gaps) / len(gaps)

        regimes = {}
        for e in entities:
            r = e["kernel"]["regime"]
            regimes[r] = regimes.get(r, 0) + 1
        regime_str = " ".join(f"{r}:{n}" for r, n in sorted(regimes.items()))

        thm = f"{ddata['summary']['theorems_passed']}/{ddata['summary']['theorems_total']}"

        print(f"{dkey:<30} {f_mean:>7.4f} {f_std:>7.4f} {ic_mean:>7.4f} {gap_mean:>7.4f} {regime_str:<25} {thm:>5}")


def cmd_regimes(reg: dict) -> None:
    """Show regime distribution across all domains."""
    global_regimes: dict[str, list[str]] = {}
    for dkey, ddata in reg["domains"].items():
        for e in ddata["entities"]:
            r = e["kernel"]["regime"]
            global_regimes.setdefault(r, []).append(f"{dkey}:{e['name']}")

    total = sum(len(v) for v in global_regimes.values())
    print(f"Regime distribution across {total} entities:\n")
    for regime in ["Stable", "Watch", "Collapse"]:
        entities = global_regimes.get(regime, [])
        pct = len(entities) / total * 100
        print(f"  {regime:<10} {len(entities):>4} ({pct:>5.1f}%)")

    print("\nPer regime:")
    for regime in ["Stable", "Watch", "Collapse"]:
        entities = global_regimes.get(regime, [])
        if entities:
            print(f"\n  [{regime}] ({len(entities)} entities):")
            for e in sorted(entities):
                print(f"    {e}")


def cmd_outliers(reg: dict) -> None:
    """Show entities with extreme kernel values."""
    entities = _all_entities(reg)

    print("=== Extreme Values ===\n")

    for field, label in [("F", "Fidelity"), ("IC", "Integrity"), ("S", "Entropy"), ("C", "Curvature")]:
        by_field = sorted(entities, key=lambda e: e["kernel"][field])
        lo = by_field[:3]
        hi = by_field[-3:]

        print(f"  {label} ({field}):")
        print("    Lowest:  ", end="")
        print("  |  ".join(f"{e['_domain']}:{e['name']} = {e['kernel'][field]:.4f}" for e in lo))
        print("    Highest: ", end="")
        print("  |  ".join(f"{e['_domain']}:{e['name']} = {e['kernel'][field]:.4f}" for e in hi))
        print()

    # Largest heterogeneity gaps
    by_gap = sorted(entities, key=lambda e: e["checks"]["heterogeneity_gap"], reverse=True)
    print("  Heterogeneity gap (Δ = F − IC):")
    print("    Largest: ", end="")
    print("  |  ".join(f"{e['_domain']}:{e['name']} = {e['checks']['heterogeneity_gap']:.4f}" for e in by_gap[:5]))
    print("    Smallest:", end="")
    print("  |  ".join(f"{e['_domain']}:{e['name']} = {e['checks']['heterogeneity_gap']:.4f}" for e in by_gap[-5:]))


def cmd_export_csv(reg: dict) -> None:
    """Export all entity data as a flat CSV for external analysis."""
    out_path = REGISTRY_PATH.parent / "test_registry_flat.csv"
    entities = _all_entities(reg)

    fieldnames = [
        "domain",
        "entity",
        "category",
        "F",
        "omega",
        "S",
        "C",
        "kappa",
        "IC",
        "regime",
        "heterogeneity_gap",
        "duality_residual",
        "log_integrity_residual",
    ]
    # Add trace vector channels
    n_ch = max(len(e["trace_vector"]) for e in entities)
    for i in range(n_ch):
        fieldnames.append(f"c_{i}")

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for e in entities:
            row = {
                "domain": e["_domain"],
                "entity": e["name"],
                "category": e["category"],
                **{k: e["kernel"][k] for k in ["F", "omega", "S", "C", "kappa", "IC", "regime"]},
                "heterogeneity_gap": e["checks"]["heterogeneity_gap"],
                "duality_residual": e["checks"]["duality_residual"],
                "log_integrity_residual": e["checks"]["log_integrity_residual"],
            }
            for i, v in enumerate(e["trace_vector"]):
                row[f"c_{i}"] = v
            writer.writerow(row)

    print(f"Exported {len(entities)} entities to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Study and compare entities from the test registry")
    parser.add_argument("--domain", type=str, help="Filter by domain name (substring match)")
    parser.add_argument("--sort", type=str, help="Sort all entities by field: F, omega, IC, S, C, kappa, gap")
    parser.add_argument("--compare", action="store_true", help="Cross-domain comparison table")
    parser.add_argument("--regimes", action="store_true", help="Regime distribution analysis")
    parser.add_argument("--outliers", action="store_true", help="Extreme values analysis")
    parser.add_argument("--export-csv", action="store_true", help="Export flat CSV for external analysis")

    args = parser.parse_args()
    reg = _load()

    if args.sort:
        cmd_sort(reg, args.sort)
    elif args.compare:
        cmd_compare(reg)
    elif args.regimes:
        cmd_regimes(reg)
    elif args.outliers:
        cmd_outliers(reg)
    elif args.export_csv:
        cmd_export_csv(reg)
    else:
        cmd_summary(reg, args.domain)


if __name__ == "__main__":
    main()

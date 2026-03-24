#!/usr/bin/env python3
"""Harvest all domain closure entities into a centralized test registry.

Imports every 6-theorem domain closure, computes kernel outputs for all entities,
runs all theorems, and writes the results to artifacts/test_registry.json.

This produces a single file containing:
  - All entity trace vectors and kernel outputs (F, ω, S, C, κ, IC, regime)
  - All theorem results (passed/failed, key values)
  - Channel definitions and categories per domain
  - Metadata (timestamp, kernel version, frozen parameters)

Usage:
    python scripts/harvest_test_registry.py          # Generate registry
    python scripts/harvest_test_registry.py --verify  # Verify existing registry
    python scripts/harvest_test_registry.py --diff    # Show drift from registry
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from importlib import import_module
from pathlib import Path
from typing import Any

# Ensure workspace paths
_WORKSPACE = Path(__file__).resolve().parents[1]
for _p in [str(_WORKSPACE / "src"), str(_WORKSPACE)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from umcp.frozen_contract import EPSILON

REGISTRY_PATH = _WORKSPACE / "artifacts" / "test_registry.json"

# ── Domain definitions ──────────────────────────────────────────────────────
# Each entry: (domain_key, closure_module_path, prefix, categories)
# prefix is the uppercase abbreviation used in symbol names (e.g., "MM" for market_microstructure)

DOMAIN_CLOSURES: list[tuple[str, str, str, list[str]]] = [
    (
        "gravitational_wave_memory",
        "closures.spacetime_memory.gravitational_wave_memory",
        "GW",
        ["compact_binary", "transient", "continuous", "stochastic"],
    ),
    (
        "market_microstructure",
        "closures.finance.market_microstructure",
        "MM",
        ["equity", "fixed_income", "derivatives", "alternative"],
    ),
    (
        "media_coherence",
        "closures.dynamic_semiotics.media_coherence",
        "MC",
        ["linguistic", "visual", "symbolic", "embodied"],
    ),
    (
        "topological_persistence",
        "closures.continuity_theory.topological_persistence",
        "TP",
        ["surface", "manifold", "knot", "fractal"],
    ),
    (
        "attention_mechanisms",
        "closures.awareness_cognition.attention_mechanisms",
        "AM",
        ["selective", "sustained", "divided", "orienting"],
    ),
    (
        "fluid_dynamics",
        "closures.everyday_physics.fluid_dynamics",
        "FD",
        ["laminar", "transitional", "turbulent", "compressible"],
    ),
    (
        "electroweak_precision",
        "closures.standard_model.electroweak_precision",
        "EWP",
        ["asymmetry", "ratio", "w_mixing", "z_pole"],
    ),
    ("binary_star_systems", "closures.astronomy.binary_star_systems", "BS", ["compact", "eclipsing", "visual", "xray"]),
    (
        "defect_physics",
        "closures.materials_science.defect_physics",
        "DP",
        ["antisite", "color_center", "interstitial", "vacancy"],
    ),
    (
        "topological_band_structures",
        "closures.quantum_mechanics.topological_band_structures",
        "TB",
        ["semimetal", "strong_TI", "trivial", "two_d"],
    ),
    (
        "sleep_neurophysiology",
        "closures.clinical_neuroscience.sleep_neurophysiology",
        "SN",
        ["event", "nrem", "rem", "waking"],
    ),
    (
        "molecular_evolution",
        "closures.evolution.molecular_evolution",
        "ME",
        ["highly_conserved", "moderate", "rapid", "ultra_conserved"],
    ),
    ("acoustics", "closures.everyday_physics.acoustics", "AC", ["absorber", "fluid", "solid", "system"]),
    (
        "reaction_channels",
        "closures.nuclear_physics.reaction_channels",
        "RC",
        ["capture", "fission", "fusion", "spallation"],
    ),
    ("rigid_body_dynamics", "closures.kinematics.rigid_body_dynamics", "RB", ["astro", "engineering", "sport", "toy"]),
    ("volatility_surface", "closures.finance.volatility_surface", "VS", ["commodity", "equity", "fixed_income", "fx"]),
    (
        "computational_semiotics",
        "closures.dynamic_semiotics.computational_semiotics",
        "CS",
        ["algorithmic", "social_media", "translation", "generative"],
    ),
    (
        "neural_correlates",
        "closures.consciousness_coherence.neural_correlates",
        "NC",
        ["binding", "network", "gating", "modulation"],
    ),
    (
        "organizational_resilience",
        "closures.continuity_theory.organizational_resilience",
        "OR",
        ["agile", "institutional", "distributed", "specialized"],
    ),
    (
        "cosmological_memory",
        "closures.spacetime_memory.cosmological_memory",
        "CM",
        ["radiation", "structure", "relic", "dynamic"],
    ),
    (
        "developmental_neuroscience",
        "closures.clinical_neuroscience.developmental_neuroscience",
        "DN",
        ["prenatal", "childhood", "adolescent", "aging"],
    ),
    (
        "gravitational_phenomena",
        "closures.spacetime_memory.gravitational_phenomena",
        "GP",
        ["weak_field", "strong_field", "lensing", "cosmological"],
    ),
    (
        "temporal_topology",
        "closures.spacetime_memory.temporal_topology",
        "TT",
        ["descent", "circulation", "toroidal", "special"],
    ),
    ("budget_geometry", "closures.continuity_theory.budget_geometry", "BG", ["flat_plain", "ramp", "wall", "special"]),
]


def _harvest_domain(domain_key: str, module_path: str, prefix: str, categories: list[str]) -> dict[str, Any]:
    """Import a domain closure and harvest all entity/kernel/theorem data."""
    mod = import_module(module_path)

    # Resolve symbols using the prefix convention
    channels_attr = f"{prefix}_CHANNELS"
    entities_attr = f"{prefix}_ENTITIES"
    n_channels_attr = f"N_{prefix}_CHANNELS"

    channels = getattr(mod, channels_attr)
    entities = getattr(mod, entities_attr)
    n_channels = getattr(mod, n_channels_attr)

    # Compute kernel for all entities
    compute_all = mod.compute_all_entities
    kernel_results = compute_all()

    # Build entity records with trace vectors and kernel outputs
    entity_records = []
    for entity, result in zip(entities, kernel_results, strict=True):
        trace = entity.trace_vector().tolist()
        entity_records.append(
            {
                "name": result.name,
                "category": result.category,
                "trace_vector": trace,
                "kernel": {
                    "F": result.F,
                    "omega": result.omega,
                    "S": result.S,
                    "C": result.C,
                    "kappa": result.kappa,
                    "IC": result.IC,
                    "regime": result.regime,
                },
                # Derived checks (precomputed for fast validation)
                "checks": {
                    "duality_residual": abs(result.F + result.omega - 1.0),
                    "integrity_bound_satisfied": result.IC <= result.F + 1e-12,
                    "log_integrity_residual": abs(result.IC - math.exp(result.kappa)),
                    "heterogeneity_gap": result.F - result.IC,
                },
            }
        )

    # Run all theorems
    verify_all = mod.verify_all_theorems
    theorem_results = verify_all()
    theorem_records = []
    for t in theorem_results:
        record = {"name": t["name"], "passed": t["passed"]}
        # Include any extra keys the theorem provides
        for k, v in t.items():
            if k not in ("name", "passed"):
                # Convert numpy types to native Python
                if isinstance(v, (np.floating, np.integer)):
                    record[k] = float(v)
                elif isinstance(v, np.bool_):
                    record[k] = bool(v)
                elif isinstance(v, np.ndarray):
                    record[k] = v.tolist()
                else:
                    try:
                        json.dumps(v)
                        record[k] = v
                    except (TypeError, ValueError):
                        record[k] = str(v)
        theorem_records.append(record)

    return {
        "domain": domain_key,
        "module": module_path,
        "prefix": prefix,
        "channels": channels if isinstance(channels, list) else list(channels),
        "n_channels": n_channels,
        "categories": categories,
        "n_entities": len(entities),
        "entities": entity_records,
        "theorems": theorem_records,
        "summary": {
            "F_min": min(e["kernel"]["F"] for e in entity_records),
            "F_max": max(e["kernel"]["F"] for e in entity_records),
            "F_mean": sum(e["kernel"]["F"] for e in entity_records) / len(entity_records),
            "IC_min": min(e["kernel"]["IC"] for e in entity_records),
            "IC_max": max(e["kernel"]["IC"] for e in entity_records),
            "IC_mean": sum(e["kernel"]["IC"] for e in entity_records) / len(entity_records),
            "gap_min": min(e["checks"]["heterogeneity_gap"] for e in entity_records),
            "gap_max": max(e["checks"]["heterogeneity_gap"] for e in entity_records),
            "regimes": dict(
                sorted(
                    {
                        r: sum(1 for e in entity_records if e["kernel"]["regime"] == r)
                        for r in {e["kernel"]["regime"] for e in entity_records}
                    }.items()
                )
            ),
            "all_theorems_pass": all(t["passed"] for t in theorem_records),
            "theorems_passed": sum(1 for t in theorem_records if t["passed"]),
            "theorems_total": len(theorem_records),
        },
    }


def harvest_all() -> dict[str, Any]:
    """Harvest all domain closures into a single registry."""
    t0 = time.time()
    domains = {}
    errors = []

    for domain_key, module_path, prefix, categories in DOMAIN_CLOSURES:
        try:
            print(f"  Harvesting {domain_key}...", end=" ", flush=True)
            data = _harvest_domain(domain_key, module_path, prefix, categories)
            domains[domain_key] = data
            n_ent = data["n_entities"]
            n_thm = data["summary"]["theorems_passed"]
            t_thm = data["summary"]["theorems_total"]
            print(f"{n_ent} entities, {n_thm}/{t_thm} theorems")
        except Exception as e:
            errors.append({"domain": domain_key, "error": str(e)})
            print(f"ERROR: {e}")

    elapsed = time.time() - t0

    # Global summary across all domains
    all_entities = []
    for d in domains.values():
        all_entities.extend(d["entities"])

    registry = {
        "_meta": {
            "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "generator": "scripts/harvest_test_registry.py",
            "epsilon": EPSILON,
            "n_domains": len(domains),
            "n_entities_total": len(all_entities),
            "n_theorems_total": sum(d["summary"]["theorems_total"] for d in domains.values()),
            "all_pass": all(d["summary"]["all_theorems_pass"] for d in domains.values()),
            "harvest_seconds": round(elapsed, 2),
        },
        "global_summary": {
            "F_range": [
                min(e["kernel"]["F"] for e in all_entities),
                max(e["kernel"]["F"] for e in all_entities),
            ],
            "IC_range": [
                min(e["kernel"]["IC"] for e in all_entities),
                max(e["kernel"]["IC"] for e in all_entities),
            ],
            "gap_range": [
                min(e["checks"]["heterogeneity_gap"] for e in all_entities),
                max(e["checks"]["heterogeneity_gap"] for e in all_entities),
            ],
            "regime_distribution": dict(
                sorted(
                    {
                        r: sum(1 for e in all_entities if e["kernel"]["regime"] == r)
                        for r in {e["kernel"]["regime"] for e in all_entities}
                    }.items()
                )
            ),
            "max_duality_residual": max(e["checks"]["duality_residual"] for e in all_entities),
            "max_log_integrity_residual": max(e["checks"]["log_integrity_residual"] for e in all_entities),
            "all_integrity_bounds_satisfied": all(e["checks"]["integrity_bound_satisfied"] for e in all_entities),
        },
        "domains": domains,
    }

    if errors:
        registry["_errors"] = errors

    return registry


def verify_registry(registry_path: Path) -> bool:
    """Verify an existing registry against live kernel computation."""
    if not registry_path.exists():
        print(f"Registry not found: {registry_path}")
        return False

    with open(registry_path) as f:
        registry = json.load(f)

    from umcp.kernel_optimized import compute_kernel_outputs

    n_checked = 0
    n_drift = 0

    for domain_key, domain_data in registry["domains"].items():
        for entity in domain_data["entities"]:
            c = np.array(entity["trace_vector"])
            c = np.clip(c, EPSILON, 1.0 - EPSILON)
            n_ch = len(c)
            w = np.ones(n_ch) / n_ch
            result = compute_kernel_outputs(c, w)

            # Check kernel values match
            stored = entity["kernel"]
            for key in ["F", "omega", "S", "C", "kappa", "IC"]:
                live = result[key]
                saved = stored[key]
                if abs(live - saved) > 1e-10:
                    print(
                        f"  DRIFT {domain_key}/{entity['name']}.{key}: "
                        f"registry={saved:.10f} live={live:.10f} "
                        f"Δ={abs(live - saved):.2e}"
                    )
                    n_drift += 1
            n_checked += 1

    print(f"\nVerified {n_checked} entities across {len(registry['domains'])} domains")
    if n_drift:
        print(f"  {n_drift} values drifted from registry")
        return False
    else:
        print("  All values match — registry is current")
        return True


def diff_registry(registry_path: Path) -> None:
    """Show a human-readable diff between registry and live computation."""
    if not registry_path.exists():
        print(f"Registry not found: {registry_path}")
        return

    with open(registry_path) as f:
        registry = json.load(f)

    from umcp.kernel_optimized import compute_kernel_outputs

    print(f"{'Domain':<30} {'Entity':<25} {'Field':<8} {'Registry':>12} {'Live':>12} {'Δ':>12}")
    print("-" * 105)

    any_drift = False
    for domain_key, domain_data in registry["domains"].items():
        for entity in domain_data["entities"]:
            c = np.array(entity["trace_vector"])
            c = np.clip(c, EPSILON, 1.0 - EPSILON)
            n_ch = len(c)
            w = np.ones(n_ch) / n_ch
            result = compute_kernel_outputs(c, w)

            stored = entity["kernel"]
            for key in ["F", "omega", "S", "C", "kappa", "IC"]:
                live = result[key]
                saved = stored[key]
                delta = abs(live - saved)
                if delta > 1e-14:
                    print(f"{domain_key:<30} {entity['name']:<25} {key:<8} {saved:>12.8f} {live:>12.8f} {delta:>12.2e}")
                    any_drift = True

    if not any_drift:
        print("No drift detected — registry matches live computation exactly.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Harvest domain closures into centralized test registry")
    parser.add_argument("--verify", action="store_true", help="Verify existing registry against live computation")
    parser.add_argument("--diff", action="store_true", help="Show drift between registry and live computation")
    parser.add_argument("--output", type=Path, default=REGISTRY_PATH, help=f"Output path (default: {REGISTRY_PATH})")
    args = parser.parse_args()

    if args.verify:
        ok = verify_registry(args.output)
        sys.exit(0 if ok else 1)

    if args.diff:
        diff_registry(args.output)
        return

    print(f"Harvesting {len(DOMAIN_CLOSURES)} domain closures...")
    registry = harvest_all()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(registry, f, indent=2)

    meta = registry["_meta"]
    print(f"\nRegistry written to {args.output}")
    print(f"  {meta['n_domains']} domains, {meta['n_entities_total']} entities, {meta['n_theorems_total']} theorems")
    print(f"  All pass: {meta['all_pass']}")
    print(f"  Harvest time: {meta['harvest_seconds']:.1f}s")


if __name__ == "__main__":
    main()

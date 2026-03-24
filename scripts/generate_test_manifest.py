"""Generate the centralized test manifest.

Introspects all standardized domain closure modules, computes kernel outputs
for every entity, and writes the complete data to tests/test_manifest.json.

This manifest captures ALL numerical test data in one place so that:
  1. Lightweight validation can recompute kernels from raw trace vectors
  2. All test bounds and expected values are visible in a single file
  3. Cross-domain comparison and study is possible without importing modules

Usage:
    python scripts/generate_test_manifest.py          # Generate manifest
    python scripts/generate_test_manifest.py --check  # Verify existing manifest
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

_WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_WORKSPACE / "src"))
sys.path.insert(0, str(_WORKSPACE))

from umcp.frozen_contract import EPSILON, RegimeThresholds

# ═══════════════════════════════════════════════════════════════════
#  DOMAIN REGISTRY — all standardized closure modules
# ═══════════════════════════════════════════════════════════════════

DOMAIN_MODULES: list[dict[str, str]] = [
    {"module": "closures.astronomy.binary_star_systems", "prefix": "BS", "domain": "astronomy"},
    {"module": "closures.awareness_cognition.attention_mechanisms", "prefix": "AM", "domain": "awareness_cognition"},
    {
        "module": "closures.clinical_neuroscience.developmental_neuroscience",
        "prefix": "DN",
        "domain": "clinical_neuroscience",
    },
    {
        "module": "closures.clinical_neuroscience.neurotransmitter_systems",
        "prefix": "NT",
        "domain": "clinical_neuroscience",
    },
    {
        "module": "closures.clinical_neuroscience.sleep_neurophysiology",
        "prefix": "SN",
        "domain": "clinical_neuroscience",
    },
    {"module": "closures.consciousness_coherence.altered_states", "prefix": "AS", "domain": "consciousness_coherence"},
    {
        "module": "closures.consciousness_coherence.neural_correlates",
        "prefix": "NC",
        "domain": "consciousness_coherence",
    },
    {"module": "closures.continuity_theory.budget_geometry", "prefix": "BG", "domain": "continuity_theory"},
    {"module": "closures.continuity_theory.organizational_resilience", "prefix": "OR", "domain": "continuity_theory"},
    {"module": "closures.continuity_theory.topological_persistence", "prefix": "TP", "domain": "continuity_theory"},
    {"module": "closures.dynamic_semiotics.computational_semiotics", "prefix": "CS", "domain": "dynamic_semiotics"},
    {"module": "closures.dynamic_semiotics.media_coherence", "prefix": "MC", "domain": "dynamic_semiotics"},
    {"module": "closures.everyday_physics.acoustics", "prefix": "AC", "domain": "everyday_physics"},
    {"module": "closures.everyday_physics.fluid_dynamics", "prefix": "FD", "domain": "everyday_physics"},
    {"module": "closures.evolution.molecular_evolution", "prefix": "ME", "domain": "evolution"},
    {"module": "closures.finance.market_microstructure", "prefix": "MM", "domain": "finance"},
    {"module": "closures.finance.volatility_surface", "prefix": "VS", "domain": "finance"},
    {"module": "closures.kinematics.rigid_body_dynamics", "prefix": "RB", "domain": "kinematics"},
    {"module": "closures.materials_science.defect_physics", "prefix": "DP", "domain": "materials_science"},
    {"module": "closures.nuclear_physics.reaction_channels", "prefix": "RC", "domain": "nuclear_physics"},
    {"module": "closures.quantum_mechanics.photonic_confinement", "prefix": "CPM", "domain": "quantum_mechanics"},
    {"module": "closures.quantum_mechanics.topological_band_structures", "prefix": "TB", "domain": "quantum_mechanics"},
    {"module": "closures.spacetime_memory.cosmological_memory", "prefix": "CM", "domain": "spacetime_memory"},
    {"module": "closures.spacetime_memory.gravitational_phenomena", "prefix": "GP", "domain": "spacetime_memory"},
    {"module": "closures.spacetime_memory.gravitational_wave_memory", "prefix": "GW", "domain": "spacetime_memory"},
    {"module": "closures.spacetime_memory.temporal_topology", "prefix": "TT", "domain": "spacetime_memory"},
    {"module": "closures.standard_model.electroweak_precision", "prefix": "EWP", "domain": "standard_model"},
]

# ═══════════════════════════════════════════════════════════════════
#  CORNER PROBES — fixed trace vectors with known behavior
# ═══════════════════════════════════════════════════════════════════

CORNER_PROBES = [
    {"name": "all_epsilon", "c": [EPSILON, EPSILON, EPSILON], "expect_F_lt": 0.05},
    {"name": "all_near_one", "c": [1.0 - EPSILON, 1.0 - EPSILON, 1.0 - EPSILON], "expect_F_gt": 0.99},
    {"name": "mixed_extremes", "c": [EPSILON, 1.0 - EPSILON, 1.0 - EPSILON]},
    {"name": "homogeneous_0.5", "c": [0.5, 0.5, 0.5]},
    {"name": "homogeneous_0.9", "c": [0.9, 0.9, 0.9]},
    {"name": "homogeneous_0.1", "c": [0.1, 0.1, 0.1]},
    {"name": "geometric_slaughter", "c": [EPSILON, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99]},
    {"name": "perfect_coherence", "c": [0.98, 0.98, 0.98]},
    {"name": "asymmetric_low", "c": [0.99, 0.20, 0.20]},
    {"name": "nuclear_fe56", "c": [0.990, 0.990, 0.869]},
    {"name": "nuclear_h1", "c": [0.010, 0.990, 0.010]},
    {"name": "nuclear_pb208", "c": [0.895, 0.990, 0.980]},
]


# ═══════════════════════════════════════════════════════════════════
#  MANIFEST GENERATION
# ═══════════════════════════════════════════════════════════════════


def _compute_kernel_from_trace(c: list[float]) -> dict[str, float]:
    """Compute kernel outputs from a raw trace vector using the kernel directly."""
    from umcp.kernel_optimized import compute_kernel_outputs

    c_arr = np.array(c)
    c_arr = np.clip(c_arr, EPSILON, 1.0 - EPSILON)
    w = np.ones(len(c_arr)) / len(c_arr)
    result = compute_kernel_outputs(c_arr, w)
    F = float(result["F"])
    omega = float(result["omega"])
    S = float(result["S"])
    C = float(result["C"])
    kappa = float(result["kappa"])
    IC = float(result["IC"])

    thresholds = RegimeThresholds()
    if omega >= thresholds.omega_collapse_min:
        regime = "Collapse"
    elif (
        omega < thresholds.omega_stable_max
        and thresholds.F_stable_min < F
        and thresholds.S_stable_max > S
        and thresholds.C_stable_max > C
    ):
        regime = "Stable"
    else:
        regime = "Watch"

    return {
        "F": round(F, 15),
        "omega": round(omega, 15),
        "S": round(S, 15),
        "C": round(C, 15),
        "kappa": round(kappa, 15),
        "IC": round(IC, 15),
        "regime": regime,
    }


def _process_domain(spec: dict[str, str]) -> dict | None:
    """Import a domain closure module and extract all entity data."""
    mod_path = spec["module"]
    prefix = spec["prefix"]

    try:
        mod = importlib.import_module(mod_path)
    except Exception as e:
        print(f"  WARNING: Could not import {mod_path}: {e}")
        return None

    # Find the entities list
    entities_attr = f"{prefix}_ENTITIES"
    entities = getattr(mod, entities_attr, None)
    if entities is None:
        print(f"  WARNING: No {entities_attr} in {mod_path}")
        return None

    # Find channels list
    channels_attr = f"{prefix}_CHANNELS"
    channels = getattr(mod, channels_attr, None)
    n_channels_attr = f"N_{prefix}_CHANNELS"
    n_channels = getattr(mod, n_channels_attr, 8)

    # Find compute function
    compute_fn_name = f"compute_{prefix.lower()}_kernel"
    compute_fn = getattr(mod, compute_fn_name, None)

    # Find verify/prove functions (theorems)
    theorem_fns = []
    for name in sorted(dir(mod)):
        if name.startswith(f"verify_t_{prefix.lower()}_") or name.startswith(f"prove_t_{prefix.lower()}_"):
            theorem_fns.append(name)

    # Compute all entities
    entity_data = []
    compute_all_fn = getattr(mod, "compute_all_entities", None)
    all_results = compute_all_fn() if compute_all_fn else None

    for entity in entities:
        trace = entity.trace_vector().tolist()
        name = entity.name
        category = getattr(entity, "category", "unknown")

        # Compute kernel via the module's own function
        if compute_fn:
            r = compute_fn(entity)
            kernel_out = {
                "F": round(float(r.F), 15),
                "omega": round(float(r.omega), 15),
                "S": round(float(r.S), 15),
                "C": round(float(r.C), 15),
                "kappa": round(float(r.kappa), 15),
                "IC": round(float(r.IC), 15),
                "regime": r.regime,
            }
        else:
            kernel_out = _compute_kernel_from_trace(trace)

        entity_data.append(
            {
                "name": name,
                "category": category,
                "trace_vector": [round(v, 10) for v in trace],
                "kernel": kernel_out,
            }
        )

    # Run theorems
    theorem_results = []
    if all_results is not None and theorem_fns:
        for fn_name in theorem_fns:
            fn = getattr(mod, fn_name)
            try:
                result = fn(all_results)
                tag = result.get("name", fn_name)
                passed = result.get("passed", False)
                theorem_results.append({"tag": tag, "passed": bool(passed)})
            except Exception as e:
                theorem_results.append({"tag": fn_name, "passed": False, "error": str(e)})

    categories = sorted({e["category"] for e in entity_data})

    return {
        "module": mod_path,
        "domain": spec["domain"],
        "prefix": prefix,
        "n_channels": n_channels,
        "channels": list(channels) if channels else [],
        "n_entities": len(entity_data),
        "categories": categories,
        "entities": entity_data,
        "theorems": theorem_results,
    }


def _build_corner_probes() -> list[dict]:
    """Compute kernel outputs for all corner probe vectors."""
    probes = []
    for probe in CORNER_PROBES:
        kernel = _compute_kernel_from_trace(probe["c"])
        entry = {
            "name": probe["name"],
            "trace_vector": probe["c"],
            "dim": len(probe["c"]),
            "kernel": kernel,
        }
        # Copy extra expected fields
        for k, v in probe.items():
            if k.startswith("expect_"):
                entry[k] = v
        probes.append(entry)
    return probes


def generate_manifest() -> dict:
    """Generate the complete test manifest."""
    thresholds = RegimeThresholds()

    manifest = {
        "_comment": "Test Manifest — Centralized test data for lightweight validation and study",
        "_generated": datetime.now(UTC).isoformat(),
        "_generator": "scripts/generate_test_manifest.py",
        "frozen_parameters": {
            "EPSILON": EPSILON,
            "regime_gates": {
                "stable": {
                    "omega_max": thresholds.omega_stable_max,
                    "F_min": thresholds.F_stable_min,
                    "S_max": thresholds.S_stable_max,
                    "C_max": thresholds.C_stable_max,
                },
                "collapse": {"omega_min": thresholds.omega_collapse_min},
                "critical": {"IC_max": thresholds.I_critical_max},
            },
        },
        "invariant_bounds": {
            "duality": {
                "identity": "F + omega = 1",
                "tolerance": 1e-12,
                "check": "abs(F + omega - 1.0) < tolerance",
            },
            "integrity_bound": {
                "identity": "IC <= F",
                "tolerance": 1e-12,
                "check": "IC <= F + tolerance",
            },
            "log_integrity": {
                "identity": "IC = exp(kappa)",
                "tolerance": 1e-10,
                "check": "abs(IC - exp(kappa)) < tolerance",
            },
            "entropy_nonneg": {
                "identity": "S >= 0",
                "tolerance": 1e-12,
                "check": "S >= -tolerance",
            },
            "curvature_nonneg": {
                "identity": "C >= 0",
                "tolerance": 1e-12,
                "check": "C >= -tolerance",
            },
            "omega_range": {
                "identity": "0 <= omega < 1",
                "tolerance": 1e-12,
                "check": "omega >= -tolerance and omega < 1.0 + tolerance",
            },
            "fidelity_range": {
                "identity": "0 < F <= 1",
                "tolerance": 1e-12,
                "check": "F > -tolerance and F <= 1.0 + tolerance",
            },
        },
        "corner_probes": [],
        "domains": [],
    }

    # Corner probes
    print("Computing corner probes...")
    manifest["corner_probes"] = _build_corner_probes()

    # Domain closures
    total_entities = 0
    total_theorems = 0
    for spec in DOMAIN_MODULES:
        print(f"Processing {spec['module']}...")
        domain_data = _process_domain(spec)
        if domain_data:
            manifest["domains"].append(domain_data)
            total_entities += domain_data["n_entities"]
            total_theorems += len(domain_data["theorems"])

    manifest["summary"] = {
        "n_domains": len(manifest["domains"]),
        "n_entities": total_entities,
        "n_corner_probes": len(manifest["corner_probes"]),
        "n_theorems": total_theorems,
        "n_checks_per_entity": 7,
        "total_numerical_checks": total_entities * 7 + len(manifest["corner_probes"]) * 7,
    }

    return manifest


def write_manifest(manifest: dict, path: Path) -> None:
    """Write manifest to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"\nManifest written to {path}")
    print(f"  Domains: {manifest['summary']['n_domains']}")
    print(f"  Entities: {manifest['summary']['n_entities']}")
    print(f"  Corner probes: {manifest['summary']['n_corner_probes']}")
    print(f"  Theorems: {manifest['summary']['n_theorems']}")
    print(f"  Total numerical checks: {manifest['summary']['total_numerical_checks']}")


def verify_manifest(path: Path) -> bool:
    """Verify an existing manifest is valid and current."""
    if not path.exists():
        print(f"ERROR: Manifest not found at {path}")
        return False

    with open(path) as f:
        manifest = json.load(f)

    n_domains = manifest.get("summary", {}).get("n_domains", 0)
    n_entities = manifest.get("summary", {}).get("n_entities", 0)
    print(f"Manifest: {n_domains} domains, {n_entities} entities")

    # Spot-check: recompute a few entities and compare
    from umcp.kernel_optimized import compute_kernel_outputs

    mismatches = 0
    checked = 0
    for domain in manifest.get("domains", []):
        for entity in domain.get("entities", [])[:2]:  # check first 2 per domain
            c = np.array(entity["trace_vector"])
            c = np.clip(c, EPSILON, 1.0 - EPSILON)
            w = np.ones(len(c)) / len(c)
            result = compute_kernel_outputs(c, w)
            F = float(result["F"])
            expected_F = entity["kernel"]["F"]
            if abs(F - expected_F) > 1e-10:
                print(f"  MISMATCH: {domain['prefix']}/{entity['name']} F={F} expected={expected_F}")
                mismatches += 1
            checked += 1

    print(f"Spot-checked {checked} entities: {mismatches} mismatches")
    return mismatches == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate or verify test manifest")
    parser.add_argument("--check", action="store_true", help="Verify existing manifest")
    parser.add_argument("--output", type=str, default=None, help="Output path (default: tests/test_manifest.json)")
    args = parser.parse_args()

    manifest_path = Path(args.output) if args.output else _WORKSPACE / "tests" / "test_manifest.json"

    if args.check:
        ok = verify_manifest(manifest_path)
        sys.exit(0 if ok else 1)

    manifest = generate_manifest()
    write_manifest(manifest, manifest_path)


if __name__ == "__main__":
    main()

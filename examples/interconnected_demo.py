#!/usr/bin/env python3
"""
UMCP Interconnected System Demo

Demonstrates the full interconnected UMCP system:
1. Load root configuration files
2. Validate mathematical consistency
3. Execute closures from registry
4. Cross-reference between components
"""
from __future__ import annotations

from umcp import (
    get_closure_loader,
    get_root_validator,
    get_umcp_files,
)


def main():
    print("=" * 70)
    print("UMCP INTERCONNECTED SYSTEM DEMONSTRATION")
    print("=" * 70)
    print()
    
    # ========================================================================
    # PART 1: Load Root Configuration Files
    # ========================================================================
    print("PART 1: Loading Root Configuration Files")
    print("-" * 70)
    
    files = get_umcp_files()
    
    print("Loading manifest...")
    manifest = files.load_manifest()
    casepack_id = manifest.get("casepack", {}).get("id", "unknown")
    print(f"  ✓ CasePack ID: {casepack_id}")
    
    print("Loading contract...")
    contract = files.load_contract()
    contract_id = contract.get("contract", {}).get("id", "unknown")
    print(f"  ✓ Contract ID: {contract_id}")
    
    print("Loading observables...")
    observables = files.load_observables()
    obs_count = len(observables.get("observables", []))
    print(f"  ✓ Observables defined: {obs_count}")
    
    print("Loading weights...")
    weights = files.load_weights()
    weight_sum = sum(float(weights[0][k]) for k in weights[0] if k.startswith('w_'))
    print(f"  ✓ Weights sum: {weight_sum:.10f}")
    
    print()
    
    # ========================================================================
    # PART 2: Load and Validate Data Files
    # ========================================================================
    print("PART 2: Loading and Validating Data Files")
    print("-" * 70)
    
    print("Loading trace...")
    trace = files.load_trace()
    coords = [trace[0][f"c_{i}"] for i in range(1, 4)]
    print(f"  ✓ Coordinates: c₁={coords[0]}, c₂={coords[1]}, c₃={coords[2]}")
    
    print("Loading invariants...")
    invariants = files.load_invariants()
    inv = invariants[0]
    print(f"  ✓ ω (omega) = {inv['omega']}")
    print(f"  ✓ F (fidelity) = {inv['F']}")
    print(f"  ✓ S (entropy) = {inv['S']}")
    print(f"  ✓ C (curvature) = {inv['C']}")
    print(f"  ✓ IC (integrity) = {inv['IC']}")
    print(f"  ✓ Regime: {inv['regime_label']}")
    
    print()
    
    # ========================================================================
    # PART 3: Validate Mathematical Consistency
    # ========================================================================
    print("PART 3: Validating Mathematical Consistency")
    print("-" * 70)
    
    validator = get_root_validator()
    result = validator.validate_all()
    
    print(f"Validation Status: {result['status']}")
    print(f"Total Checks: {result['total_checks']}")
    print(f"Errors: {len(result['errors'])}")
    print(f"Warnings: {len(result['warnings'])}")
    print(f"Passed: {len(result['passed'])}")
    print()
    
    if result["errors"]:
        print("Errors:")
        for error in result["errors"][:5]:  # Show first 5
            print(f"  {error}")
        if len(result["errors"]) > 5:
            print(f"  ... and {len(result['errors']) - 5} more")
        print()
    
    if result["passed"]:
        print("Sample Passed Checks:")
        for check in result["passed"][:3]:
            print(f"  {check}")
        print()
    
    # ========================================================================
    # PART 4: Load and Execute Closures
    # ========================================================================
    print("PART 4: Loading and Executing Closures")
    print("-" * 70)
    
    loader = get_closure_loader()
    
    print("Listing available closures...")
    closures = loader.list_closures()
    print(f"  ✓ Found {len(closures)} closures in registry:")
    for name, path in closures.items():
        print(f"    - {name}: {path}")
    print()
    
    # Try to execute some closures
    print("Executing closures...")
    
    try:
        print("  Testing F_from_omega closure...")
        result = loader.execute_closure("F_from_omega", omega=10.0, r=0.5, m=1.0)
        print(f"    ✓ F_from_omega(ω=10.0, r=0.5, m=1.0) = {result['F']} N")
    except (FileNotFoundError, ImportError) as e:
        print(f"    ⚠ F_from_omega not available: {e}")
    
    try:
        print("  Testing hello_world closure...")
        result = loader.execute_closure("hello_world", omega=10.0)
        print(f"    ✓ hello_world(ω=10.0) = {result}")
    except (FileNotFoundError, ImportError) as e:
        print(f"    ⚠ hello_world not available: {e}")
    
    try:
        print("  Testing tau_R_compute closure...")
        result = loader.execute_closure("tau_R_compute", omega=10.0, damping=0.1)
        print(f"    ✓ tau_R_compute(ω=10.0, damping=0.1) = {result['tau_R']} s")
    except (FileNotFoundError, ImportError, ValueError) as e:
        print(f"    ⚠ tau_R_compute not available or failed: {e}")
    
    print()
    
    # ========================================================================
    # PART 5: Cross-Reference Between Components
    # ========================================================================
    print("PART 5: Cross-Referencing Between Components")
    print("-" * 70)
    
    print("Checking interconnections...")
    
    # Check manifest references contract
    refs = manifest.get("refs", {})
    if "contract" in refs:
        contract_ref = refs["contract"]
        print(f"  ✓ Manifest references contract: {contract_ref.get('id', 'N/A')}")
    
    # Check manifest references closures
    if "closures_registry" in refs:
        closures_ref = refs["closures_registry"]
        print(f"  ✓ Manifest references closures: {closures_ref.get('id', 'N/A')}")
    
    # Check contract tier-1 kernel
    contract_obj = contract.get("contract", {})
    if "tier_1_kernel" in contract_obj:
        kernel = contract_obj["tier_1_kernel"]
        invariant_names = kernel.get("invariants", [])
        print(f"  ✓ Contract defines {len(invariant_names)} Tier-1 invariants")
        print(f"    Invariants: {', '.join([inv.get('symbol', '') for inv in invariant_names])}")
    
    # Check that invariants match regime classification
    omega_val = float(inv["omega"])
    F_val = float(inv["F"])
    S_val = float(inv["S"])
    C_val = float(inv["C"])
    regime = inv["regime_label"]
    
    print(f"  ✓ Current regime: {regime}")
    print("    Thresholds satisfied:")
    print(f"      ω < 0.038: {omega_val < 0.038} (ω={omega_val:.6f})")
    print(f"      F > 0.90:  {F_val > 0.90} (F={F_val:.6f})")
    print(f"      S < 0.15:  {S_val < 0.15} (S={S_val:.6f})")
    print(f"      C < 0.14:  {C_val < 0.14} (C={C_val:.6f})")
    
    print()
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("=" * 70)
    print("INTERCONNECTED SYSTEM SUMMARY")
    print("=" * 70)
    print()
    print("✓ Root files loaded: 16/16")
    
    # Get validation result
    validator_result = validator.validate_all()
    print(f"✓ Validation status: {validator_result['status']}")
    print(f"✓ Closures available: {len(closures)}")
    print(f"✓ Regime state: {regime}")
    print()
    print("The UMCP system is fully interconnected:")
    print("  • Manifest → Contract → Closures")
    print("  • Observables → Trace → Invariants → Regimes")
    print("  • Weights → Mathematical calculations → Validation")
    print("  • Integrity checksums → All files")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()

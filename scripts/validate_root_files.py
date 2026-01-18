#!/usr/bin/env python3
"""
Validate root-level UMCP files against schemas and semantic rules.

This script performs comprehensive validation of all root-level UMCP files.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from umcp import get_umcp_files


def validate_files():
    """Validate all UMCP files."""
    
    print("🔍 UMCP Root-Level Files Validation")
    print("=" * 70)
    print()
    
    umcp = get_umcp_files()
    errors = []
    warnings = []
    
    # Check existence
    print("📁 File Existence:")
    missing = umcp.get_missing_files()
    if missing:
        for filename in missing:
            errors.append(f"Missing required file: {filename}")
            print(f"  ✗ {filename}")
    else:
        print("  ✓ All files present")
    print()
    
    if errors:
        print(f"❌ Validation failed with {len(errors)} error(s)")
        for error in errors:
            print(f"  - {error}")
        return 1
    
    # Validate manifest
    print("📋 Validating manifest.yaml:")
    try:
        manifest = umcp.load_manifest()
        assert 'casepack' in manifest, "Missing 'casepack' section"
        assert 'refs' in manifest, "Missing 'refs' section"
        assert 'artifacts' in manifest, "Missing 'artifacts' section"
        print("  ✓ Structure valid")
        print(f"    CasePack: {manifest['casepack']['id']} v{manifest['casepack']['version']}")
    except Exception as e:
        errors.append(f"manifest.yaml validation failed: {e}")
        print(f"  ✗ {e}")
    print()
    
    # Validate contract
    print("📜 Validating contract.yaml:")
    try:
        contract = umcp.load_contract()
        assert 'contract' in contract, "Missing 'contract' section"
        c = contract['contract']
        assert 'embedding' in c, "Missing 'embedding' section"
        assert 'tier_1_kernel' in c, "Missing 'tier_1_kernel' section"
        assert 'typed_censoring' in c, "Missing 'typed_censoring' section"
        print("  ✓ Structure valid")
        print(f"    Contract: {c['id']} v{c['version']}")
    except Exception as e:
        errors.append(f"contract.yaml validation failed: {e}")
        print(f"  ✗ {e}")
    print()
    
    # Validate observables
    print("🔬 Validating observables.yaml:")
    try:
        observables = umcp.load_observables()
        assert 'observables' in observables, "Missing 'observables' section"
        obs = observables['observables']
        assert 'primary' in obs, "Missing 'primary' observables"
        assert 'derived' in obs, "Missing 'derived' observables"
        print("  ✓ Structure valid")
        print(f"    Primary: {len(obs['primary'])} variables")
        print(f"    Derived: {len(obs['derived'])} variables")
    except Exception as e:
        errors.append(f"observables.yaml validation failed: {e}")
        print(f"  ✗ {e}")
    print()
    
    # Validate weights
    print("⚖️  Validating weights.csv:")
    try:
        weights = umcp.load_weights()
        assert len(weights) > 0, "No weight rows found"
        w_row = weights[0]
        w_values = [float(v) for v in w_row.values()]
        
        # Check sum to 1
        w_sum = sum(w_values)
        if abs(w_sum - 1.0) > 1e-9:
            warnings.append(f"Weights sum to {w_sum:.6f}, not 1.0")
            print(f"  ⚠ Sum = {w_sum:.6f} (expected 1.0)")
        
        # Check non-negative
        if not all(w >= 0 for w in w_values):
            errors.append("Some weights are negative")
            print("  ✗ Contains negative weights")
        else:
            print("  ✓ All weights non-negative")
            print(f"    Sum: {w_sum:.6f}")
    except Exception as e:
        errors.append(f"weights.csv validation failed: {e}")
        print(f"  ✗ {e}")
    print()
    
    # Validate trace
    print("📊 Validating derived/trace.csv:")
    try:
        trace = umcp.load_trace()
        assert len(trace) > 0, "No trace rows found"
        row = trace[0]
        
        # Check for coordinate columns
        coord_cols = [k for k in row.keys() if k.startswith('c_')]
        assert len(coord_cols) > 0, "No coordinate columns found"
        
        # Check all coordinates in [0, 1]
        coords = [float(row[c]) for c in coord_cols]
        if not all(0 <= c <= 1 for c in coords):
            errors.append("Some coordinates out of [0, 1] range")
            print("  ✗ Coordinates out of range")
        else:
            print("  ✓ All coordinates in [0, 1]")
            print(f"    Rows: {len(trace)}")
            print(f"    Coordinates: {len(coord_cols)}")
    except Exception as e:
        errors.append(f"trace.csv validation failed: {e}")
        print(f"  ✗ {e}")
    print()
    
    # Validate invariants
    print("📈 Validating outputs/invariants.csv:")
    try:
        invariants = umcp.load_invariants()
        assert len(invariants) > 0, "No invariant rows found"
        
        for i, inv in enumerate(invariants):
            omega = float(inv['omega'])
            F = float(inv['F'])
            kappa = float(inv['kappa'])
            IC = float(inv['IC'])
            
            # Check F ≈ 1 - ω
            f_error = abs(F - (1 - omega))
            if f_error > 1e-6:
                warnings.append(f"Row {i}: F ≈ 1-ω violation (error={f_error:.2e})")
            
            # Check IC ≈ exp(κ)
            import math
            ic_expected = math.exp(kappa)
            ic_error = abs(IC - ic_expected) / max(abs(ic_expected), 1e-10)
            if ic_error > 1e-3:  # Relative error threshold
                warnings.append(f"Row {i}: IC ≈ exp(κ) violation (rel_error={ic_error:.2e})")
        
        print("  ✓ Structure valid")
        print(f"    Rows: {len(invariants)}")
        if warnings:
            print(f"    ⚠ {len([w for w in warnings if 'Row' in w])} Tier-1 warnings")
    except Exception as e:
        errors.append(f"invariants.csv validation failed: {e}")
        print(f"  ✗ {e}")
    print()
    
    # Summary
    print("=" * 70)
    if errors:
        print(f"❌ Validation FAILED with {len(errors)} error(s)")
        for error in errors:
            print(f"  - {error}")
        if warnings:
            print(f"\n⚠️  {len(warnings)} warning(s):")
            for warning in warnings:
                print(f"  - {warning}")
        return 1
    elif warnings:
        print(f"✅ Validation PASSED with {len(warnings)} warning(s)")
        for warning in warnings:
            print(f"  - {warning}")
        return 0
    else:
        print("✅ Validation PASSED - All checks successful!")
        return 0


if __name__ == "__main__":
    sys.exit(validate_files())

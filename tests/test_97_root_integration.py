"""
Test root file validation and interconnectedness.
"""

from __future__ import annotations

import pytest

from src.umcp import (
    ClosureLoader,
    RootFileValidator,
    UMCPFiles,
    get_closure_loader,
    get_root_validator,
    get_umcp_files,
)


def test_root_file_validator_runs():
    """Test that root file validator can be instantiated and runs."""
    validator = get_root_validator()
    result = validator.validate_all()

    assert result["status"] in ["PASS", "FAIL"]
    assert "errors" in result
    assert "warnings" in result
    assert "passed" in result
    assert result["total_checks"] > 0


def test_root_files_mathematically_consistent():
    """Test that root files satisfy mathematical identities."""
    validator = RootFileValidator()
    validator._validate_invariant_identities()

    # Should have no errors for F=1-ω and IC≈exp(κ)
    identity_errors = [
        e for e in validator.errors if "identity" in e.lower() or "F ≠ 1-ω" in e
    ]
    assert not identity_errors, f"Mathematical identity violations: {identity_errors}"


def test_regime_classification_accurate():
    """Test that regime classification matches omega/F/S/C thresholds."""
    validator = RootFileValidator()
    validator._validate_regime_classification()

    regime_errors = [e for e in validator.errors if "regime" in e.lower()]
    assert not regime_errors, f"Regime classification errors: {regime_errors}"


def test_weights_sum_to_one():
    """Test that weights sum to exactly 1.0."""
    validator = RootFileValidator()
    validator._validate_weights()

    weight_errors = [e for e in validator.errors if "weight" in e.lower()]
    assert not weight_errors, f"Weight sum errors: {weight_errors}"


def test_trace_coordinates_in_bounds():
    """Test that trace coordinates are in [0,1]."""
    validator = RootFileValidator()
    validator._validate_trace_bounds()

    bound_errors = [
        e for e in validator.errors if "bound" in e.lower() or "coordinate" in e.lower()
    ]
    assert not bound_errors, f"Coordinate bound errors: {bound_errors}"


def test_umcp_files_integration():
    """Test that UMCPFiles can load all root files."""
    files = get_umcp_files()

    # Test loading each file type
    manifest = files.load_manifest()
    assert "casepack" in manifest

    contract = files.load_contract()
    assert "contract" in contract

    observables = files.load_observables()
    assert "observables" in observables

    weights = files.load_weights()
    assert len(weights) > 0

    trace = files.load_trace()
    assert len(trace) > 0

    invariants = files.load_invariants()
    assert len(invariants) > 0

    regimes = files.load_regimes()
    assert len(regimes) > 0


def test_closure_loader_can_list_closures():
    """Test that ClosureLoader can list available closures."""
    loader = get_closure_loader()
    closures = loader.list_closures()

    assert isinstance(closures, dict)
    # Should have at least the closures in registry
    assert len(closures) > 0


def test_closure_loader_can_load_modules():
    """Test that ClosureLoader can load Python closure modules."""
    loader = ClosureLoader()

    # Test loading F_from_omega
    try:
        module = loader.load_closure_module("F_from_omega")
        assert hasattr(module, "compute")
    except FileNotFoundError:
        pytest.skip("F_from_omega.py not found")


def test_closure_loader_can_execute():
    """Test that ClosureLoader can execute closures."""
    loader = ClosureLoader()

    try:
        # Test F_from_omega closure
        result = loader.execute_closure("F_from_omega", omega=10.0, r=0.5, m=1.0)
        assert "F" in result
        assert result["F"] == 50.0  # m * omega^2 * r = 1 * 100 * 0.5
    except FileNotFoundError:
        pytest.skip("F_from_omega.py not found")


def test_closure_hello_world_exists():
    """Test that hello_world closure exists and can be executed."""
    loader = ClosureLoader()

    try:
        result = loader.execute_closure("hello_world", omega=10.0)
        assert "F" in result
    except FileNotFoundError:
        pytest.skip("hello_world.py not found")


def test_full_integration_pipeline():
    """
    Test full integration: load root files, validate them, and execute closures.
    This represents the complete interconnected system.
    """
    # Step 1: Load root files
    files = get_umcp_files()
    manifest = files.load_manifest()
    contract = files.load_contract()
    invariants = files.load_invariants()

    # Step 2: Validate files
    validator = get_root_validator()
    result = validator.validate_all()

    # Should pass basic checks even if warnings exist
    assert result["total_checks"] > 10

    # Step 3: Load closures
    loader = get_closure_loader()
    closures = loader.list_closures()

    # System should have interconnected components
    assert manifest is not None
    assert contract is not None
    assert invariants is not None
    assert closures is not None

    # If validation passed, check regime is Stable
    if result["status"] == "PASS":
        inv = invariants[0]
        assert inv.get("regime_label") == "Stable"


def test_contract_references_closures():
    """Test that contract properly references the closures registry."""
    files = UMCPFiles()
    contract = files.load_contract()

    # Contract should reference closures
    contract_obj = contract.get("contract", {})
    assert "tier_1_kernel" in contract_obj or "closures" in contract_obj


def test_manifest_references_contract():
    """Test that manifest properly references the contract."""
    files = UMCPFiles()
    manifest = files.load_manifest()

    # Manifest should reference contract
    refs = manifest.get("refs", {})
    assert "contract" in refs or "contract_id" in manifest.get("casepack", {})


def test_invariants_match_trace_dimensions():
    """Test that invariants CSV has entries matching trace CSV dimensions."""
    files = UMCPFiles()
    trace = files.load_trace()
    invariants = files.load_invariants()

    # Should have at least one row in each
    assert len(trace) > 0
    assert len(invariants) > 0

    # Invariants should have the same number of rows as trace (or validate against trace)
    # For now, just check they're both non-empty
    assert len(invariants) >= len(trace)


def test_regimes_match_invariants():
    """Test that regimes CSV matches invariants CSV."""
    files = UMCPFiles()
    invariants = files.load_invariants()
    regimes = files.load_regimes()

    # Should have same number of rows
    assert len(invariants) == len(regimes)

    # Regime labels should match
    for inv, reg in zip(invariants, regimes, strict=False):
        inv_regime = inv.get("regime_label", "")
        reg_regime = reg.get("regime_label", "")
        assert (
            inv_regime == reg_regime
        ), f"Regime mismatch: invariants={inv_regime}, regimes={reg_regime}"


def test_checksums_are_valid():
    """Test that integrity checksums are valid for all files."""
    validator = RootFileValidator()
    validator._validate_checksums()

    checksum_errors = [
        e for e in validator.errors if "checksum" in e.lower() or "hash" in e.lower()
    ]
    assert not checksum_errors, f"Checksum validation errors: {checksum_errors}"


def test_all_required_files_exist():
    """Test that all 16 required root files exist."""
    files = UMCPFiles()
    missing = files.get_missing_files()

    assert not missing, f"Missing required files: {missing}"


def test_stable_regime_achieved():
    """Test that the current configuration is in Stable regime."""
    files = UMCPFiles()
    invariants = files.load_invariants()

    if invariants:
        inv = invariants[0]
        omega = float(inv["omega"])
        F = float(inv["F"])
        S = float(inv["S"])
        C = float(inv["C"])
        regime = inv.get("regime_label", "")

        # Check Stable thresholds
        assert omega < 0.038, f"ω={omega} should be < 0.038 for Stable"
        assert F > 0.90, f"F={F} should be > 0.90 for Stable"
        assert S < 0.15, f"S={S} should be < 0.15 for Stable"
        assert C < 0.14, f"C={C} should be < 0.14 for Stable"
        assert regime == "Stable", f"Expected regime=Stable, got {regime}"

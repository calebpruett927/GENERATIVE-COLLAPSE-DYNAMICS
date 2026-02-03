"""
Test UMCP-REF-E2E-0001 audit-ready exemplar case.

Validates that the reference E2E case demonstrates all required behaviors:
- OOR events (clip_and_flag)
- Finite returns (at least one τ_R numeric)
- Typed non-returns (INF_REC)
- IC ≈ exp(κ) consistency
- Manifest hash in SS1M receipt
- Environment metadata
- Passes both baseline and strict validation
"""

import json
import math
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
E2E_CASE = REPO_ROOT / "casepacks" / "UMCP-REF-E2E-0001"
SS1M_RECEIPT = E2E_CASE / "receipts" / "ss1m.json"
KERNEL_LEDGER = E2E_CASE / "outputs" / "kernel_ledger.csv"
PSI_TRACE = E2E_CASE / "data" / "psi_trace.csv"


def test_e2e_case_exists():
    """Verify E2E case directory structure exists."""
    assert E2E_CASE.exists(), f"E2E case not found: {E2E_CASE}"
    assert (E2E_CASE / "README.md").exists()
    assert (E2E_CASE / "contracts").exists()
    assert (E2E_CASE / "contracts" / "contract.yaml").exists()
    assert (E2E_CASE / "contracts" / "embedding.yaml").exists()
    assert (E2E_CASE / "contracts" / "return.yaml").exists()
    assert (E2E_CASE / "contracts" / "weights.yaml").exists()
    assert (E2E_CASE / "closures" / "closure_registry.yaml").exists()


def test_e2e_outputs_exist():
    """Verify pipeline outputs exist."""
    assert SS1M_RECEIPT.exists(), f"SS1M receipt not found: {SS1M_RECEIPT}"
    assert KERNEL_LEDGER.exists(), f"Kernel ledger not found: {KERNEL_LEDGER}"
    assert PSI_TRACE.exists(), f"Psi trace not found: {PSI_TRACE}"


def test_e2e_oor_events():
    """Verify at least one OOR event is present."""
    with open(PSI_TRACE) as f:
        lines = f.readlines()

    # Skip header
    data_lines = lines[1:]

    # Check for at least one OOR flag
    oor_count = 0
    for line in data_lines:
        if "True" in line:  # OOR flags are boolean True
            oor_count += 1

    assert oor_count >= 1, f"Expected at least 1 OOR event, found {oor_count}"


def test_e2e_finite_returns():
    """Verify at least one finite return is present."""
    with open(KERNEL_LEDGER) as f:
        lines = f.readlines()

    # Skip header
    data_lines = lines[1:]

    finite_count = 0
    inf_rec_count = 0

    for line in data_lines:
        parts = line.strip().split(",")
        # Columns: t,omega,F,S,C,tau_R,kappa,IC (tau_R is index 5)
        tau_R = parts[5]
        if tau_R == "INF_REC":
            inf_rec_count += 1
        else:
            # Try to parse as number
            try:
                float(tau_R)
                finite_count += 1
            except ValueError:
                pass

    assert finite_count >= 1, f"Expected at least 1 finite return, found {finite_count}"
    assert inf_rec_count >= 1, f"Expected at least 1 INF_REC, found {inf_rec_count}"
    print(f"✓ Found {finite_count} finite returns and {inf_rec_count} INF_REC instances")


def test_e2e_ss1m_receipt_structure():
    """Verify SS1M receipt has required structure."""
    with open(SS1M_RECEIPT) as f:
        receipt = json.load(f)

    assert "receipt" in receipt
    r = receipt["receipt"]

    # Check basic fields
    assert r["case_id"] == "UMCP-REF-E2E-0001"
    assert r["status"] == "CONFORMANT"

    # Check typed boundaries
    assert "typed_boundaries" in r
    tb = r["typed_boundaries"]
    assert tb["oor_count"] >= 1, "Expected at least 1 OOR event"
    assert tb["finite_return_count"] >= 1, "Expected at least 1 finite return"
    assert tb["inf_rec_count"] >= 1, "Expected at least 1 INF_REC"

    print(f"✓ OOR: {tb['oor_count']}, Finite: {tb['finite_return_count']}, INF_REC: {tb['inf_rec_count']}")


def test_e2e_manifest_hash_present():
    """Verify manifest root hash is present in SS1M receipt."""
    with open(SS1M_RECEIPT) as f:
        receipt = json.load(f)

    manifest_hash = receipt["receipt"]["manifest"]["root_sha256"]
    assert manifest_hash is not None, "Manifest hash is None"
    assert manifest_hash != "pending", "Manifest hash is still 'pending'"
    assert len(manifest_hash) == 64, f"Invalid SHA256 hash length: {len(manifest_hash)}"

    print(f"✓ Manifest hash: {manifest_hash[:16]}...")


def test_e2e_environment_metadata():
    """Verify environment metadata is present in SS1M receipt."""
    with open(SS1M_RECEIPT) as f:
        receipt = json.load(f)

    assert "environment" in receipt["receipt"], "Environment metadata missing"
    env = receipt["receipt"]["environment"]

    assert "python_version" in env, "Python version missing"
    assert "platform" in env, "Platform missing"

    print(f"✓ Environment: Python {env.get('python_version')}, {env.get('platform')}")


def test_e2e_ic_consistency():
    """Verify IC ≈ exp(κ) for all rows."""
    with open(KERNEL_LEDGER) as f:
        lines = f.readlines()

    # Skip header
    data_lines = lines[1:]

    tolerance = 1e-9
    max_error = 0.0

    for line in data_lines:
        parts = line.strip().split(",")
        kappa = float(parts[6])  # kappa is 7th column
        IC = float(parts[7])  # IC is 8th column

        expected_IC = math.exp(kappa)
        error = abs(IC - expected_IC)
        max_error = max(max_error, error)

        assert error < tolerance, f"IC consistency violation: IC={IC}, exp(κ)={expected_IC}, error={error}"

    print(f"✓ IC ≈ exp(κ) check passed (max error: {max_error:.2e})")


def test_e2e_kernel_summary_in_receipt():
    """Verify kernel summary includes IC consistency check."""
    with open(SS1M_RECEIPT) as f:
        receipt = json.load(f)

    kernel_summary = receipt["receipt"]["kernel_summary"]

    # Check IC consistency metadata
    if "ic_consistency" in kernel_summary:
        ic_check = kernel_summary["ic_consistency"]
        assert "check_passed" in ic_check
        assert ic_check["check_passed"] is True, "IC consistency check failed"
        print(f"✓ IC consistency check in receipt: PASS (error={ic_check.get('final_abs_error', 'N/A')})")
    else:
        # Legacy format - just verify final row exists
        assert "final_row" in kernel_summary
        print("⚠ IC consistency metadata not present (legacy format)")


@pytest.mark.slow
def test_e2e_baseline_validation():
    """Verify E2E case passes baseline validation."""
    result = subprocess.run(
        ["umcp", "validate", str(E2E_CASE)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    # Parse JSON output
    output = result.stdout
    try:
        # Extract JSON from output
        json_start = output.find("{")
        json_output = output[json_start:]
        validation_result = json.loads(json_output)

        # Find E2E case target
        e2e_target = None
        for target in validation_result.get("targets", []):
            if "UMCP-REF-E2E-0001" in target.get("target_path", ""):
                e2e_target = target
                break

        assert e2e_target is not None, "E2E case not found in validation results"
        assert e2e_target["run_status"] == "CONFORMANT", f"Baseline validation failed: {e2e_target}"
        assert e2e_target["counts"]["errors"] == 0, f"Expected 0 errors, got {e2e_target['counts']['errors']}"

        print(
            f"✓ Baseline validation: CONFORMANT (errors={e2e_target['counts']['errors']}, warnings={e2e_target['counts']['warnings']})"
        )
    except (json.JSONDecodeError, KeyError) as e:
        pytest.fail(f"Failed to parse validation output: {e}\n{output}")


@pytest.mark.slow
def test_e2e_strict_validation():
    """Verify E2E case passes strict validation."""
    result = subprocess.run(
        ["umcp", "validate", "--strict", str(E2E_CASE)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    # Parse JSON output
    output = result.stdout
    try:
        # Extract JSON from output (it comes after the summary line)
        lines = output.split("\n")
        json_lines = []
        in_json = False
        for line in lines:
            if line.strip().startswith("{"):
                in_json = True
            if in_json:
                json_lines.append(line)

        json_output = "\n".join(json_lines)
        validation_result = json.loads(json_output)

        # Find E2E case target
        e2e_target = None
        for target in validation_result.get("targets", []):
            if "UMCP-REF-E2E-0001" in target.get("target_path", ""):
                e2e_target = target
                break

        assert e2e_target is not None, "E2E case not found in strict validation results"
        assert e2e_target["run_status"] == "CONFORMANT", f"Strict validation failed: {e2e_target}"
        assert e2e_target["counts"]["errors"] == 0, (
            f"Expected 0 errors in strict mode, got {e2e_target['counts']['errors']}"
        )

        print(f"✓ Strict validation: CONFORMANT (errors={e2e_target['counts']['errors']})")
    except (json.JSONDecodeError, KeyError) as e:
        pytest.fail(f"Failed to parse strict validation output: {e}\n{output}")


def test_e2e_weights_normalization():
    """Verify channel weights sum to 1.0."""
    import yaml

    weights_file = E2E_CASE / "contracts" / "weights.yaml"
    with open(weights_file) as f:
        weights_doc = yaml.safe_load(f)

    channels = weights_doc["weights"]["channels"]
    weight_sum = sum(ch["weight"] for ch in channels)
    tolerance = weights_doc["weights"]["validation"]["tolerance"]

    assert abs(weight_sum - 1.0) < tolerance, f"Weights do not sum to 1.0: {weight_sum}"
    print(f"✓ Weights sum: {weight_sum} (tolerance: {tolerance})")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

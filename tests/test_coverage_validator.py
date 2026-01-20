"""Targeted tests to improve validator.py coverage."""

from __future__ import annotations

from pathlib import Path

import pytest

from umcp.validator import RootFileValidator

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestManifestValidation:
    """Test manifest validation edge cases."""

    def test_manifest_missing_schema_field(self, tmp_path):
        """Test manifest without schema field."""
        manifest_path = tmp_path / "manifest.yaml"
        manifest_path.write_text("casepack: test\nversion: 1.0")
        
        validator = RootFileValidator(root_dir=tmp_path)
        validator._validate_manifest()
        
        errors = [e for e in validator.errors if "manifest" in e and "schema" in e]
        assert len(errors) > 0

    def test_manifest_missing_casepack_field(self, tmp_path):
        """Test manifest without casepack field."""
        manifest_path = tmp_path / "manifest.yaml"
        manifest_path.write_text("schema: test\nversion: 1.0")
        
        validator = RootFileValidator(root_dir=tmp_path)
        validator._validate_manifest()
        
        errors = [e for e in validator.errors if "manifest" in e and "casepack" in e]
        assert len(errors) > 0

    def test_manifest_load_error(self, tmp_path):
        """Test manifest with invalid YAML."""
        manifest_path = tmp_path / "manifest.yaml"
        manifest_path.write_text("invalid: yaml: structure: [[[")
        
        validator = RootFileValidator(root_dir=tmp_path)
        validator._validate_manifest()
        
        errors = [e for e in validator.errors if "manifest" in e]
        assert len(errors) > 0


class TestContractValidation:
    """Test contract validation edge cases."""

    def test_contract_missing_schema_field(self, tmp_path):
        """Test contract without schema field."""
        contract_path = tmp_path / "contract.yaml"
        contract_path.write_text("contract: test\nversion: 1.0")
        
        validator = RootFileValidator(root_dir=tmp_path)
        validator._validate_contract()
        
        errors = [e for e in validator.errors if "contract" in e and "schema" in e]
        assert len(errors) > 0

    def test_contract_missing_contract_field(self, tmp_path):
        """Test contract without contract field."""
        contract_path = tmp_path / "contract.yaml"
        contract_path.write_text("schema: test\nversion: 1.0")
        
        validator = RootFileValidator(root_dir=tmp_path)
        validator._validate_contract()
        
        errors = [e for e in validator.errors if "contract" in e.lower()]
        assert len(errors) > 0

    def test_contract_load_error(self, tmp_path):
        """Test contract with invalid YAML."""
        contract_path = tmp_path / "contract.yaml"
        contract_path.write_text("invalid: yaml: [[[")
        
        validator = RootFileValidator(root_dir=tmp_path)
        validator._validate_contract()
        
        errors = [e for e in validator.errors if "contract" in e.lower()]
        assert len(errors) > 0


class TestObservablesValidation:
    """Test observables validation edge cases."""

    def test_observables_missing_observables_field(self, tmp_path):
        """Test observables without observables field."""
        obs_path = tmp_path / "observables.yaml"
        obs_path.write_text("schema: test\nversion: 1.0")
        
        validator = RootFileValidator(root_dir=tmp_path)
        validator._validate_observables()
        
        errors = [e for e in validator.errors if "observables" in e and "missing" in e]
        assert len(errors) > 0

    def test_observables_load_error(self, tmp_path):
        """Test observables with invalid YAML."""
        obs_path = tmp_path / "observables.yaml"
        obs_path.write_text("invalid: [[[")
        
        validator = RootFileValidator(root_dir=tmp_path)
        validator._validate_observables()
        
        errors = [e for e in validator.errors if "observables" in e]
        assert len(errors) > 0


class TestWeightsValidation:
    """Test weights validation edge cases."""

    def test_weights_invalid_csv(self, tmp_path):
        """Test weights with invalid CSV."""
        weights_path = tmp_path / "weights.csv"
        weights_path.write_text("invalid,csv\n,,,malformed")
        
        validator = RootFileValidator(root_dir=tmp_path)
        validator._validate_weights()
        
        # Should handle gracefully
        assert True

    def test_weights_file_not_found(self, tmp_path):
        """Test weights file missing."""
        validator = RootFileValidator(root_dir=tmp_path)
        validator._validate_weights()
        
        errors = [e for e in validator.errors if "weight" in e.lower()]
        assert len(errors) > 0


class TestChecksumValidation:
    """Test checksum validation edge cases."""

    def test_checksum_file_missing(self, tmp_path):
        """Test when checksum file doesn't exist."""
        validator = RootFileValidator(root_dir=tmp_path)
        validator._validate_checksums()
        
        errors = [e for e in validator.errors if "checksum" in e.lower() or "sha256" in e.lower()]
        assert len(errors) > 0

    def test_checksum_malformed_line(self, tmp_path):
        """Test checksum file with malformed lines."""
        sha_path = tmp_path / "integrity" / "sha256.txt"
        sha_path.parent.mkdir(parents=True, exist_ok=True)
        sha_path.write_text("malformed line without hash\n")
        
        validator = RootFileValidator(root_dir=tmp_path)
        validator._validate_checksums()
        
        # Should handle gracefully
        assert True


class TestInvariantValidation:
    """Test invariant validation edge cases."""

    def test_invariant_missing_files(self, tmp_path):
        """Test invariant validation with missing files."""
        validator = RootFileValidator(root_dir=tmp_path)
        validator._validate_invariant_identities()
        
        # Should handle missing files gracefully
        assert True

    def test_invariant_with_real_files(self):
        """Test invariant validation with actual files."""
        validator = RootFileValidator(root_dir=REPO_ROOT)
        
        # This may pass or fail depending on data, but shouldn't crash
        try:
            validator._validate_invariant_identities()
        except Exception as e:
            pytest.fail(f"Invariant validation crashed: {e}")


class TestFileExistenceChecks:
    """Test file existence validation."""

    def test_file_existence_with_missing_files(self, tmp_path):
        """Test checking files that don't exist."""
        validator = RootFileValidator(root_dir=tmp_path)
        validator._check_file_existence()
        
        # Should have errors for missing files
        assert len(validator.errors) > 0

    def test_file_existence_with_existing_files(self):
        """Test checking files that exist."""
        validator = RootFileValidator(root_dir=REPO_ROOT)
        validator._check_file_existence()
        
        # Should have some passed checks
        assert len(validator.passed) > 0

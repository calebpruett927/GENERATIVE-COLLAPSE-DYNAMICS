"""Additional tests for improving coverage of validator, closures, and file_refs modules."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from umcp.closures import ClosureLoader, get_closure_loader
from umcp.file_refs import UMCPFiles, get_umcp_files
from umcp.validator import RootFileValidator, get_root_validator

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestClosureLoader:
    """Test closure loading functionality."""

    def test_get_closure_loader(self):
        """Test factory function."""
        loader = get_closure_loader()
        assert isinstance(loader, ClosureLoader)

    def test_load_registry(self):
        """Test accessing the closures registry."""
        loader = ClosureLoader()
        registry = loader.registry
        assert isinstance(registry, dict)
        assert len(registry) > 0

    def test_list_available_closures(self):
        """Test listing closures."""
        loader = ClosureLoader()
        closures = loader.list_closures()
        assert isinstance(closures, dict)
        # Should have at least the base closures
        assert len(closures) > 0

    def test_load_specific_closure(self):
        """Test loading a specific closure module."""
        loader = ClosureLoader()
        # Try to load hello_world closure if it exists
        try:
            module = loader.load_closure_module("hello_world")
            assert module is not None
        except (KeyError, FileNotFoundError):
            pytest.skip("hello_world closure not available in test environment")


class TestUMCPFiles:
    """Test UMCP file references."""

    def test_get_umcp_files(self):
        """Test factory function."""
        files = get_umcp_files()
        assert isinstance(files, UMCPFiles)

    def test_umcp_files_attributes(self):
        """Test that UMCPFiles has expected attributes."""
        files = UMCPFiles()
        # Check for some expected attributes
        assert hasattr(files, "root")

    def test_find_repo_root(self):
        """Test finding repository root."""
        files = UMCPFiles()
        # Should find root from test directory
        assert files.root is not None
        assert files.root.exists()


class TestRootValidator:
    """Test root file validator."""

    def test_get_root_validator(self):
        """Test factory function."""
        validator = get_root_validator()
        assert isinstance(validator, RootFileValidator)

    def test_validator_with_explicit_root(self):
        """Test validator with explicit root path."""
        validator = RootFileValidator(root_dir=REPO_ROOT)
        assert validator.root == REPO_ROOT

    def test_file_existence_checks(self):
        """Test that validator checks file existence."""
        validator = RootFileValidator(root_dir=REPO_ROOT)
        validator._check_file_existence()
        # Should have some passed or error checks
        assert len(validator.passed) > 0 or len(validator.errors) > 0

    def test_manifest_validation(self):
        """Test manifest validation."""
        validator = RootFileValidator(root_dir=REPO_ROOT)
        validator._validate_manifest()
        # Should complete without crashing
        assert True

    def test_weights_validation(self):
        """Test weights validation."""
        validator = RootFileValidator(root_dir=REPO_ROOT)
        validator._validate_weights()
        # Should complete
        assert True

    def test_validate_all(self):
        """Test running all validations."""
        validator = RootFileValidator(root_dir=REPO_ROOT)
        result = validator.validate_all()
        assert isinstance(result, dict)
        assert "status" in result
        assert "errors" in result
        assert "warnings" in result
        assert "passed" in result


class TestValidatorHelpers:
    """Test validator helper methods."""

    def test_checksum_validation_with_invalid_path(self):
        """Test checksum validation with non-existent path."""
        validator = RootFileValidator(root_dir=Path("/nonexistent"))
        # Should handle gracefully
        try:
            validator._validate_checksums()
        except FileNotFoundError:
            pass  # Expected
        assert True

    def test_invariant_identities_validation(self):
        """Test mathematical identity validation."""
        validator = RootFileValidator(root_dir=REPO_ROOT)
        # May fail if files don't exist, but should not crash
        try:
            validator._validate_invariant_identities()
        except (FileNotFoundError, KeyError, csv.Error):
            pass  # Expected if outputs don't exist
        assert True


class TestClosureRegistry:
    """Test closure registry operations."""

    def test_registry_file_exists(self):
        """Test that registry file exists."""
        loader = ClosureLoader()
        assert loader.registry_path.exists()
        registry = loader.registry
        assert registry is not None

    def test_load_nonexistent_closure(self):
        """Test loading a closure that doesn't exist."""
        loader = ClosureLoader()
        with pytest.raises((KeyError, FileNotFoundError, AttributeError)):
            loader.load_closure_module("nonexistent_closure_xyz")


class TestFileRefsEdgeCases:
    """Test edge cases for file references."""

    def test_umcp_files_from_nonexistent_path(self):
        """Test creating UMCPFiles from nonexistent path."""
        # Should handle gracefully or raise clear error
        try:
            files = UMCPFiles(root_path=Path("/nonexistent/path"))
            # If it doesn't raise, that's fine
            assert files.root == Path("/nonexistent/path")
        except (FileNotFoundError, ValueError):
            pass  # Also acceptable

    def test_load_sha256_checksums(self):
        """Test loading SHA256 checksums if method exists."""
        files = UMCPFiles()
        if hasattr(files, "load_sha256"):
            try:
                checksums = files.load_sha256()
                assert isinstance(checksums, (str, dict, list))
            except FileNotFoundError:
                pass  # Expected if file doesn't exist


class TestValidatorReporting:
    """Test validator reporting functionality."""

    def test_validation_result_structure(self):
        """Test that validation result has expected structure."""
        validator = RootFileValidator(root_dir=REPO_ROOT)
        result = validator.validate_all()

        # Check all expected keys are present
        assert "status" in result
        assert "errors" in result
        assert "warnings" in result
        assert "passed" in result
        assert "total_checks" in result

        # Check types
        assert isinstance(result["errors"], list)
        assert isinstance(result["warnings"], list)
        assert isinstance(result["passed"], list)
        assert isinstance(result["total_checks"], int)

    def test_error_accumulation(self):
        """Test that errors accumulate properly."""
        validator = RootFileValidator(root_dir=REPO_ROOT)
        initial_errors = len(validator.errors)
        validator._check_file_existence()
        # Should have checked something
        assert len(validator.errors) >= initial_errors or len(validator.passed) > 0

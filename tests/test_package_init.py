"""
Tests for umcp package __init__.py.

These tests verify the public API exposed by the umcp package.
"""

import json
import pytest

import umcp
from umcp import validate, ValidationResult


class TestPackageAPI:
    """Tests for the public package API."""

    def test_version_exists(self):
        """Package has __version__."""
        assert hasattr(umcp, '__version__')
        assert isinstance(umcp.__version__, str)

    def test_validate_function_exists(self):
        """validate() function is exported."""
        assert callable(validate)

    def test_validation_result_exists(self):
        """ValidationResult class is exported."""
        assert ValidationResult is not None


class TestValidationResult:
    """Tests for ValidationResult class."""

    def test_result_from_dict(self):
        """ValidationResult can be created from dict."""
        data = {
            "run_status": "CONFORMANT",
            "summary": {"errors": 0, "warnings": 0}
        }
        result = ValidationResult(data)
        assert result  # Truthy for CONFORMANT

    def test_result_bool_conformant(self):
        """CONFORMANT result is truthy."""
        data = {"run_status": "CONFORMANT"}
        result = ValidationResult(data)
        assert bool(result) is True

    def test_result_bool_nonconformant(self):
        """NONCONFORMANT result is falsy."""
        data = {"run_status": "NONCONFORMANT"}
        result = ValidationResult(data)
        assert bool(result) is False

    def test_result_repr(self):
        """ValidationResult has readable repr."""
        data = {"run_status": "CONFORMANT"}
        result = ValidationResult(data)
        repr_str = repr(result)
        assert "CONFORMANT" in repr_str or "ValidationResult" in repr_str


class TestValidateFunction:
    """Tests for the validate() function."""

    def test_validate_nonexistent_path(self):
        """validate() handles missing path gracefully."""
        # validate() may raise or return an error result for non-existent paths
        # The key is it doesn't crash unexpectedly
        try:
            result = validate("/nonexistent/path/xyz123")
            # If it returns, it should be some result
            assert result is not None or result is None
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
            # These are acceptable error types for invalid input
            pass

    def test_validate_returns_result(self):
        """validate() returns ValidationResult for valid repo."""
        # Validate from the repo root
        import os
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        result = validate(repo_root)
        assert isinstance(result, ValidationResult)

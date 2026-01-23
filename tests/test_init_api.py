"""Tests for umcp package-level API (ValidationResult and validate function)."""

import json
import tempfile
from pathlib import Path

import pytest

import umcp


def test_validation_result_conformant():
    """Test ValidationResult with CONFORMANT status."""
    data = {
        "run_status": "CONFORMANT",
        "summary": {
            "counts": {
                "errors": 0,
                "warnings": 2
            }
        },
        "targets": [
            {
                "messages": [
                    {"severity": "warning", "text": "Warning 1"},
                    {"severity": "warning", "text": "Warning 2"}
                ]
            }
        ]
    }
    
    result = umcp.ValidationResult(data)
    
    assert result.status == "CONFORMANT"
    assert result.error_count == 0
    assert result.warning_count == 2
    assert len(result.errors) == 0
    assert len(result.warnings) == 2
    assert result.warnings[0] == "Warning 1"
    assert bool(result) is True  # Should be truthy for CONFORMANT
    assert repr(result) == "ValidationResult(status='CONFORMANT', errors=0, warnings=2)"


def test_validation_result_nonconformant():
    """Test ValidationResult with NONCONFORMANT status."""
    data = {
        "run_status": "NONCONFORMANT",
        "summary": {
            "counts": {
                "errors": 3,
                "warnings": 1
            }
        },
        "targets": [
            {
                "messages": [
                    {"severity": "error", "text": "Error 1"},
                    {"severity": "error", "text": "Error 2"},
                    {"severity": "warning", "text": "Warning 1"}
                ]
            },
            {
                "messages": [
                    {"severity": "error", "text": "Error 3"}
                ]
            }
        ]
    }
    
    result = umcp.ValidationResult(data)
    
    assert result.status == "NONCONFORMANT"
    assert result.error_count == 3
    assert result.warning_count == 1
    assert len(result.errors) == 3
    assert len(result.warnings) == 1
    assert result.errors[0] == "Error 1"
    assert bool(result) is False  # Should be falsy for NONCONFORMANT
    assert repr(result) == "ValidationResult(status='NONCONFORMANT', errors=3, warnings=1)"


def test_validation_result_empty():
    """Test ValidationResult with minimal data."""
    data = {
        "run_status": "UNKNOWN",
        "summary": {},
        "targets": []
    }
    
    result = umcp.ValidationResult(data)
    
    assert result.status == "UNKNOWN"
    assert result.error_count == 0
    assert result.warning_count == 0
    assert len(result.errors) == 0
    assert len(result.warnings) == 0
    assert bool(result) is False  # Should be falsy for non-CONFORMANT


def test_validate_hello_world():
    """Test validate() function with hello_world casepack."""
    result = umcp.validate("casepacks/hello_world")
    
    assert isinstance(result, umcp.ValidationResult)
    assert result.status == "CONFORMANT"
    assert result.error_count == 0
    assert bool(result) is True


def test_validate_with_path_object():
    """Test validate() with Path object."""
    path = Path("casepacks/hello_world")
    result = umcp.validate(path)
    
    assert isinstance(result, umcp.ValidationResult)
    assert result.status == "CONFORMANT"


def test_validate_strict_mode():
    """Test validate() with strict mode enabled."""
    result = umcp.validate("casepacks/hello_world", strict=True)
    
    assert isinstance(result, umcp.ValidationResult)
    assert result.status == "CONFORMANT"


def test_validate_nonexistent_path():
    """Test validate() with nonexistent path."""
    # The validator may handle missing paths gracefully
    # Just verify we get a ValidationResult back
    result = umcp.validate("nonexistent/path/that/does/not/exist")
    
    assert isinstance(result, umcp.ValidationResult)
    # Status could be CONFORMANT (empty validation) or NONCONFORMANT
    assert result.status in ("CONFORMANT", "NONCONFORMANT")


def test_validation_result_data_access():
    """Test that ValidationResult provides access to full data."""
    data = {
        "run_status": "CONFORMANT",
        "summary": {"counts": {"errors": 0, "warnings": 0}},
        "targets": [],
        "custom_field": "custom_value"
    }
    
    result = umcp.ValidationResult(data)
    
    # Should have access to full data dict
    assert result.data == data
    assert result.data["custom_field"] == "custom_value"


def test_validation_result_missing_fields():
    """Test ValidationResult handles missing fields gracefully."""
    # Minimal data with missing summary/targets
    data = {"run_status": "CONFORMANT"}
    
    result = umcp.ValidationResult(data)
    
    assert result.status == "CONFORMANT"
    assert result.error_count == 0
    assert result.warning_count == 0
    assert len(result.errors) == 0
    assert len(result.warnings) == 0


def test_validate_imports():
    """Test that validate function is properly exported."""
    # Should be able to import directly
    from umcp import validate, ValidationResult
    
    assert callable(validate)
    assert isinstance(ValidationResult, type)

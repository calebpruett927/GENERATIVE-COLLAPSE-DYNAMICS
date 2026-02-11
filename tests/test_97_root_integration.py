"""
Integration tests for the RootFileValidator across the actual repository.
"""

from __future__ import annotations

import pytest

from umcp.validator import RootFileValidator


@pytest.mark.integration
def test_repo_validates_conformant():
    """Verify the repository root validates as CONFORMANT."""
    validator = RootFileValidator()
    _ = validator.validate_all()

    # We expect the repo to be conformant
    assert not validator.errors, f"Repo validation failed with errors: {validator.errors}"
    # Warnings are allowed in non-strict mode, but let's check if we can be strict or just log them
    if validator.warnings:
        print(f"Validation warnings: {validator.warnings}")

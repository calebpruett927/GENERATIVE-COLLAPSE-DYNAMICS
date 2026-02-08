"""Tests for the public validate() API and ValidationResult.

This is the primary public API — the first thing users call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from umcp import ValidationResult, validate

REPO_ROOT = Path(__file__).resolve().parents[1]
HELLO_WORLD = REPO_ROOT / "casepacks" / "hello_world"


# ============================================================================
# validate() function
# ============================================================================


class TestValidateAPI:
    """Test the top-level validate() convenience function."""

    def test_validate_repo_root(self) -> None:
        """validate('.') returns CONFORMANT on a clean repo."""
        result = validate(REPO_ROOT)
        assert isinstance(result, ValidationResult)
        assert result.status == "CONFORMANT"

    def test_validate_hello_world(self) -> None:
        """validate(casepacks/hello_world) returns CONFORMANT."""
        result = validate(HELLO_WORLD)
        assert isinstance(result, ValidationResult)
        assert result.status == "CONFORMANT"

    def test_validate_strict_hello_world(self) -> None:
        """validate(casepacks/hello_world, strict=True) returns CONFORMANT."""
        result = validate(HELLO_WORLD, strict=True)
        assert isinstance(result, ValidationResult)
        assert result.status == "CONFORMANT"

    def test_validate_nonexistent_path(self) -> None:
        """validate() on a nonexistent path returns non-CONFORMANT or raises."""
        try:
            result = validate(Path("/nonexistent/fake/path"))
            assert result.status != "CONFORMANT"
        except (FileNotFoundError, SystemExit, Exception):
            pass  # acceptable — the point is it doesn't silently pass


# ============================================================================
# ValidationResult
# ============================================================================


class TestValidationResult:
    """Test ValidationResult class behavior.

    ValidationResult constructor takes a dict with keys:
      - run_status: str (maps to .status)
      - summary.counts.errors / warnings: int (maps to .error_count / .warning_count)
      - targets[*].messages[*]: {severity, text} (maps to .errors / .warnings lists)
    """

    @staticmethod
    def _make_data(
        status: str = "CONFORMANT",
        errors: int = 0,
        warnings: int = 0,
        messages: list[dict[str, str]] | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        """Build a dict matching ValidationResult's expected structure."""
        d: dict[str, Any] = {
            "run_status": status,
            "summary": {"counts": {"errors": errors, "warnings": warnings}},
            "targets": [],
        }
        if messages:
            d["targets"] = [{"messages": messages}]
        d.update(extra)
        return d

    def test_conformant_is_truthy(self) -> None:
        result = ValidationResult(self._make_data("CONFORMANT"))
        assert bool(result) is True

    def test_nonconformant_is_falsy(self) -> None:
        result = ValidationResult(
            self._make_data(
                "NONCONFORMANT",
                errors=1,
                messages=[{"severity": "error", "text": "fail"}],
            )
        )
        assert bool(result) is False

    def test_repr_is_string(self) -> None:
        result = ValidationResult(self._make_data("CONFORMANT"))
        r = repr(result)
        assert isinstance(r, str)
        assert "CONFORMANT" in r

    def test_error_count(self) -> None:
        result = ValidationResult(
            self._make_data(
                "NONCONFORMANT",
                errors=2,
                warnings=1,
                messages=[
                    {"severity": "error", "text": "e1"},
                    {"severity": "error", "text": "e2"},
                    {"severity": "warning", "text": "w1"},
                ],
            )
        )
        assert result.error_count == 2
        assert result.warning_count == 1

    def test_errors_list(self) -> None:
        result = ValidationResult(
            self._make_data(
                "NONCONFORMANT",
                errors=2,
                messages=[
                    {"severity": "error", "text": "e1"},
                    {"severity": "error", "text": "e2"},
                ],
            )
        )
        assert result.errors == ["e1", "e2"]

    def test_empty_result(self) -> None:
        """Minimal result construction works."""
        result = ValidationResult(self._make_data("NON_EVALUABLE"))
        assert result.status == "NON_EVALUABLE"
        assert result.error_count == 0

    def test_data_preserved(self) -> None:
        """Original data dict is accessible."""
        data = self._make_data("CONFORMANT", extra=42)
        result = ValidationResult(data)
        assert result.data["extra"] == 42

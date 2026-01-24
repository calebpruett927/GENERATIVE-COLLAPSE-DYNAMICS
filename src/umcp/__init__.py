"""
UMCP — Universal Measurement Contract Protocol

This package provides a contract-first validator surface for UMCP repositories and CasePacks.

Primary entry point:
- CLI: `umcp` (see src/umcp/cli.py)

Design intent:
- Keep the kernel enforcement and artifact validation portable across implementations.
- Treat contracts + closures + schemas + receipts as the minimum audit surface.

This package intentionally does not implement a full numerical “engine” for generating Ψ(t) and
Tier-1 invariants from arbitrary raw measurements yet. The current deliverable is a validator
and repo conformance toolchain.
"""
# pyright: reportPrivateUsage=false

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = [
    "DEFAULT_TZ",
    "VALIDATOR_NAME",
    "ClosureLoader",
    "RootFileValidator",
    "UMCPFiles",
    "ValidationResult",
    "__version__",
    "get_closure_loader",
    "get_root_validator",
    "get_umcp_files",
    "umcp_extensions",
    "validate",
    # Computational optimizations (KERNEL_SPECIFICATION.md Lemmas 1-34)
    "OptimizedKernelComputer",
    "SeamChainAccumulator",
    "compute_utils",
]

__version__ = "1.4.7"

VALIDATOR_NAME = "umcp-validator"
DEFAULT_TZ = "America/Chicago"

# Import utilities
from . import umcp_extensions
from . import compute_utils
from .closures import ClosureLoader, get_closure_loader
from .file_refs import UMCPFiles, get_umcp_files
from .kernel_optimized import OptimizedKernelComputer
from .seam_optimized import SeamChainAccumulator
from .validator import RootFileValidator, get_root_validator


class ValidationResult:
    """Result of a validation run.

    Attributes:
        status: "CONFORMANT" or "NONCONFORMANT"
        data: Full validation result dictionary
        errors: List of error messages
        warnings: List of warning messages
    """

    def __init__(self, data: dict[str, Any]):
        self.data = data
        self.status = data.get("run_status", "UNKNOWN")

        # Extract errors and warnings from summary
        summary = data.get("summary", {})
        counts = summary.get("counts", {})
        self.error_count = counts.get("errors", 0)
        self.warning_count = counts.get("warnings", 0)

        # Extract messages from targets
        self.errors: list[str] = []
        self.warnings: list[str] = []
        for target in data.get("targets", []):
            for msg in target.get("messages", []):
                if msg.get("severity") == "error":
                    self.errors.append(msg.get("text", ""))
                elif msg.get("severity") == "warning":
                    self.warnings.append(msg.get("text", ""))

    def __bool__(self) -> bool:
        """Returns True if validation passed (CONFORMANT)."""
        result: bool = self.status == "CONFORMANT"
        return result

    def __repr__(self) -> str:
        return f"ValidationResult(status={self.status!r}, errors={self.error_count}, warnings={self.warning_count})"


def validate(path: str | Path, strict: bool = False) -> ValidationResult:
    """Validate a UMCP casepack or repository.

    This is a convenience wrapper around the CLI validation logic.
    For full control, use the CLI: `umcp validate <path>`

    Args:
        path: Path to casepack directory or repository root
        strict: If True, enforce strict publication-grade validation

    Returns:
        ValidationResult with status, errors, and warnings

    Example:
        >>> import umcp
        >>> result = umcp.validate("casepacks/hello_world")
        >>> if result:
        ...     print("✓ CONFORMANT")
        >>> print(f"Errors: {result.error_count}, Warnings: {result.warning_count}")
    """
    import json
    import tempfile
    from argparse import Namespace

    from .cli import _cmd_validate  # pyright: ignore[reportPrivateUsage]

    # Create temporary file for output
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        output_path = f.name

    try:
        # Create args namespace mimicking CLI
        args = Namespace(path=str(path), out=output_path, strict=strict, fail_on_warning=False, verbose=False)

        # Run validation
        _cmd_validate(args)

        # Read result
        with open(output_path) as f:
            result_data = json.load(f)

        return ValidationResult(result_data)

    finally:
        # Clean up temp file
        import os

        if os.path.exists(output_path):
            os.unlink(output_path)

"""
UMCP — Universal Measurement Contract Protocol

This package provides a contract-first validator surface for UMCP repositories and CasePacks,
plus a raw measurement engine for generating Ψ(t) traces and Tier-1 kernel invariants from
arbitrary domain data.

Primary entry points:
- CLI: ``umcp`` (see src/umcp/cli.py)
- Engine: ``MeasurementEngine`` — raw data → Ψ(t) → invariants
- Calculator: ``UniversalCalculator`` — coordinate-level computation

Design intent:
- Keep the kernel enforcement and artifact validation portable across implementations.
- Treat contracts + closures + schemas + receipts as the minimum audit surface.
- Bridge raw domain data into the UMCP validation pipeline via the measurement engine.
"""
# pyright: reportPrivateUsage=false

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = [
    "DEFAULT_CONTRACT",
    "DEFAULT_TZ",
    "VALIDATOR_NAME",
    "ClosureLoader",
    "ComputationMode",
    "ContinuityLawSpec",
    "ContinuityVerdict",
    "EditionIdentity",
    "EditionTriad",
    "EmbeddingConfig",
    "EmbeddingSpec",
    "EmbeddingStrategy",
    "EngineResult",
    "FrozenContract",
    "InvariantRow",
    "KernelGradients",
    "KernelOutput",
    "MeasurementEngine",
    "OptimizedKernelComputer",
    "Regime",
    "RootFileValidator",
    "SS1mReceipt",
    "SeamChainAccumulator",
    "TraceRow",
    "UMCPFiles",
    "UncertaintyBounds",
    "UniversalCalculator",
    "UniversalResult",
    "ValidationResult",
    "WeldAnchor",
    "WeldLineage",
    "__version__",
    "accel",
    "check_seam_pass",
    "classify_regime",
    "compute_extended_triad",
    "compute_full",
    "compute_kernel",
    "compute_kernel_gradients",
    "compute_regime",
    "compute_ss1m_receipt",
    "compute_triad",
    "compute_utils",
    "continuity_law",
    "create_weld",
    "diagnose_extended",
    "diagnose_thermodynamic",
    "diagnose_thermodynamic_batch",
    "frozen_contract",
    "gamma_omega",
    "get_closure_loader",
    "get_root_validator",
    "get_umcp_files",
    "measurement_engine",
    "propagate_uncertainty",
    "safe_tau_R",
    "ss1m_triad",
    "tau_R_display",
    "tau_r_star",
    "tau_r_star_dynamics",
    "triad_to_eid12",
    "umcp_extensions",
    "uncertainty",
    "universal_calculator",
    "validate",
    "verify_continuity_law",
    "verify_triad",
    "weld_lineage",
]

__version__ = "2.1.3"

VALIDATOR_NAME = "umcp-validator"
DEFAULT_TZ = "America/Chicago"

# Import utilities
from . import (
    accel,
    compute_utils,
    continuity_law,
    frozen_contract,
    measurement_engine,
    ss1m_triad,
    tau_r_star,
    tau_r_star_dynamics,
    umcp_extensions,
    uncertainty,
    universal_calculator,
    weld_lineage,
)
from .closures import ClosureLoader, get_closure_loader
from .continuity_law import ContinuityLawSpec, ContinuityVerdict, verify_continuity_law
from .file_refs import UMCPFiles, get_umcp_files
from .frozen_contract import (
    DEFAULT_CONTRACT,
    FrozenContract,
    KernelOutput,
    Regime,
    check_seam_pass,
    classify_regime,
    compute_kernel,
    gamma_omega,
)
from .kernel_optimized import OptimizedKernelComputer
from .measurement_engine import (
    EmbeddingConfig,
    EmbeddingSpec,
    EmbeddingStrategy,
    EngineResult,
    InvariantRow,
    MeasurementEngine,
    TraceRow,
    safe_tau_R,
    tau_R_display,
)
from .seam_optimized import SeamChainAccumulator
from .ss1m_triad import EditionTriad, compute_triad, triad_to_eid12, verify_triad
from .tau_r_star import diagnose as diagnose_thermodynamic
from .tau_r_star import diagnose_invariants as diagnose_thermodynamic_batch
from .tau_r_star_dynamics import diagnose_extended
from .uncertainty import (
    KernelGradients,
    UncertaintyBounds,
    compute_kernel_gradients,
    propagate_uncertainty,
)
from .universal_calculator import (
    ComputationMode,
    UniversalCalculator,
    UniversalResult,
    compute_full,
    compute_regime,
)
from .validator import RootFileValidator, get_root_validator
from .weld_lineage import (
    EditionIdentity,
    SS1mReceipt,
    WeldAnchor,
    WeldLineage,
    compute_extended_triad,
    compute_ss1m_receipt,
    create_weld,
)


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

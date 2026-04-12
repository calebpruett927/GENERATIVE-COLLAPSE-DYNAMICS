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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from . import accel as accel
    from . import compute_utils as compute_utils
    from . import continuity_law as continuity_law
    from . import frozen_contract as frozen_contract
    from . import measurement_engine as measurement_engine
    from . import return_rope as return_rope
    from . import ss1m_triad as ss1m_triad
    from . import tau_r_star as tau_r_star
    from . import tau_r_star_dynamics as tau_r_star_dynamics
    from . import umcp_extensions as umcp_extensions
    from . import uncertainty as uncertainty
    from . import universal_calculator as universal_calculator
    from . import weld_lineage as weld_lineage
    from .closures import ClosureLoader as ClosureLoader
    from .closures import get_closure_loader as get_closure_loader
    from .cognitive_equalizer import CE_SYSTEM_PROMPT as CE_SYSTEM_PROMPT
    from .cognitive_equalizer import AequatorCognitivus as AequatorCognitivus
    from .cognitive_equalizer import CEChannels as CEChannels
    from .cognitive_equalizer import CEReport as CEReport
    from .cognitive_equalizer import CEVerdict as CEVerdict
    from .cognitive_equalizer import CognitiveEqualizer as CognitiveEqualizer
    from .continuity_law import ContinuityLawSpec as ContinuityLawSpec
    from .continuity_law import ContinuityVerdict as ContinuityVerdict
    from .continuity_law import verify_continuity_law as verify_continuity_law
    from .file_refs import UMCPFiles as UMCPFiles
    from .file_refs import get_umcp_files as get_umcp_files
    from .frozen_contract import DEFAULT_CONTRACT as DEFAULT_CONTRACT
    from .frozen_contract import FrozenContract as FrozenContract
    from .frozen_contract import KernelOutput as KernelOutput
    from .frozen_contract import Regime as Regime
    from .frozen_contract import check_seam_pass as check_seam_pass
    from .frozen_contract import classify_regime as classify_regime
    from .frozen_contract import compute_kernel as compute_kernel
    from .frozen_contract import gamma_omega as gamma_omega
    from .kernel_optimized import CostDecomposition as CostDecomposition
    from .kernel_optimized import GateMargins as GateMargins
    from .kernel_optimized import KernelDiagnostics as KernelDiagnostics
    from .kernel_optimized import OptimizedKernelComputer as OptimizedKernelComputer
    from .kernel_optimized import classify_collapse_type as classify_collapse_type
    from .kernel_optimized import diagnose as diagnose
    from .measurement_engine import EmbeddingConfig as EmbeddingConfig
    from .measurement_engine import EmbeddingSpec as EmbeddingSpec
    from .measurement_engine import EmbeddingStrategy as EmbeddingStrategy
    from .measurement_engine import EngineResult as EngineResult
    from .measurement_engine import InvariantRow as InvariantRow
    from .measurement_engine import MeasurementEngine as MeasurementEngine
    from .measurement_engine import TraceRow as TraceRow
    from .measurement_engine import safe_tau_R as safe_tau_R
    from .measurement_engine import tau_R_display as tau_R_display
    from .seam_optimized import SeamChainAccumulator as SeamChainAccumulator
    from .ss1m_triad import EditionTriad as EditionTriad
    from .ss1m_triad import compute_triad as compute_triad
    from .ss1m_triad import triad_to_eid12 as triad_to_eid12
    from .ss1m_triad import verify_triad as verify_triad
    from .tau_r_star import diagnose as diagnose_thermodynamic
    from .tau_r_star import diagnose_invariants as diagnose_thermodynamic_batch
    from .tau_r_star_dynamics import diagnose_extended as diagnose_extended
    from .uncertainty import KernelGradients as KernelGradients
    from .uncertainty import UncertaintyBounds as UncertaintyBounds
    from .uncertainty import compute_kernel_gradients as compute_kernel_gradients
    from .uncertainty import propagate_uncertainty as propagate_uncertainty
    from .universal_calculator import ComputationMode as ComputationMode
    from .universal_calculator import UniversalCalculator as UniversalCalculator
    from .universal_calculator import UniversalResult as UniversalResult
    from .universal_calculator import compute_full as compute_full
    from .universal_calculator import compute_regime as compute_regime
    from .validator import RootFileValidator as RootFileValidator
    from .validator import get_root_validator as get_root_validator
    from .weld_lineage import EditionIdentity as EditionIdentity
    from .weld_lineage import SS1mReceipt as SS1mReceipt
    from .weld_lineage import WeldAnchor as WeldAnchor
    from .weld_lineage import WeldLineage as WeldLineage
    from .weld_lineage import compute_extended_triad as compute_extended_triad
    from .weld_lineage import compute_ss1m_receipt as compute_ss1m_receipt
    from .weld_lineage import create_weld as create_weld

__all__ = [
    "CE_SYSTEM_PROMPT",
    "DEFAULT_CONTRACT",
    "DEFAULT_TZ",
    "VALIDATOR_NAME",
    "AequatorCognitivus",
    "CEChannels",
    "CEReport",
    "CEVerdict",
    "ClosureLoader",
    "CognitiveEqualizer",
    "ComputationMode",
    "ContinuityLawSpec",
    "ContinuityVerdict",
    "CostDecomposition",
    "EditionIdentity",
    "EditionTriad",
    "EmbeddingConfig",
    "EmbeddingSpec",
    "EmbeddingStrategy",
    "EngineResult",
    "FrozenContract",
    "GateMargins",
    "InvariantRow",
    "KernelDiagnostics",
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
    "classify_collapse_type",
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
    "diagnose",
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
    "return_rope",
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

__version__ = "2.3.1"

VALIDATOR_NAME = "umcp-validator"
DEFAULT_TZ = "America/Chicago"

# ---------------------------------------------------------------------------
# Lazy imports — dimensional reduction applied to the import graph.
#
# The kernel reduces 6 outputs to 3 effective DOF (F, κ, C — S is determined).
# Same principle: most consumers need 2-3 modules, not all 20.  Eager imports
# pulled ~183 modules (numpy, yaml, math, dataclasses …) on every
# ``import umcp``.  Lazy loading pays that cost only when a name is actually
# used, cutting bare-import time from ~300 ms to ~5 ms.
#
# ``__getattr__`` is the Python 3.7+ standard mechanism (PEP 562).
# ---------------------------------------------------------------------------

# Map every public name to (module_path, attribute_name | None).
# None means "import the module itself".
_LAZY_IMPORTS: dict[str, tuple[str, str | None]] = {
    # --- submodules (import umcp.X) ---
    "accel": (".accel", None),
    "compute_utils": (".compute_utils", None),
    "continuity_law": (".continuity_law", None),
    "frozen_contract": (".frozen_contract", None),
    "measurement_engine": (".measurement_engine", None),
    "return_rope": (".return_rope", None),
    "ss1m_triad": (".ss1m_triad", None),
    "tau_r_star": (".tau_r_star", None),
    "tau_r_star_dynamics": (".tau_r_star_dynamics", None),
    "umcp_extensions": (".umcp_extensions", None),
    "uncertainty": (".uncertainty", None),
    "universal_calculator": (".universal_calculator", None),
    "weld_lineage": (".weld_lineage", None),
    "cognitive_equalizer": (".cognitive_equalizer", None),
    # --- cognitive_equalizer symbols ---
    "CE_SYSTEM_PROMPT": (".cognitive_equalizer", "CE_SYSTEM_PROMPT"),
    "AequatorCognitivus": (".cognitive_equalizer", "AequatorCognitivus"),
    "CEChannels": (".cognitive_equalizer", "CEChannels"),
    "CEReport": (".cognitive_equalizer", "CEReport"),
    "CEVerdict": (".cognitive_equalizer", "CEVerdict"),
    "CognitiveEqualizer": (".cognitive_equalizer", "CognitiveEqualizer"),
    # --- closures ---
    "ClosureLoader": (".closures", "ClosureLoader"),
    "get_closure_loader": (".closures", "get_closure_loader"),
    # --- continuity_law ---
    "ContinuityLawSpec": (".continuity_law", "ContinuityLawSpec"),
    "ContinuityVerdict": (".continuity_law", "ContinuityVerdict"),
    "verify_continuity_law": (".continuity_law", "verify_continuity_law"),
    # --- file_refs ---
    "UMCPFiles": (".file_refs", "UMCPFiles"),
    "get_umcp_files": (".file_refs", "get_umcp_files"),
    # --- frozen_contract ---
    "DEFAULT_CONTRACT": (".frozen_contract", "DEFAULT_CONTRACT"),
    "FrozenContract": (".frozen_contract", "FrozenContract"),
    "KernelOutput": (".frozen_contract", "KernelOutput"),
    "Regime": (".frozen_contract", "Regime"),
    "check_seam_pass": (".frozen_contract", "check_seam_pass"),
    "classify_regime": (".frozen_contract", "classify_regime"),
    "compute_kernel": (".frozen_contract", "compute_kernel"),
    "gamma_omega": (".frozen_contract", "gamma_omega"),
    # --- kernel_optimized ---
    "CostDecomposition": (".kernel_optimized", "CostDecomposition"),
    "GateMargins": (".kernel_optimized", "GateMargins"),
    "KernelDiagnostics": (".kernel_optimized", "KernelDiagnostics"),
    "OptimizedKernelComputer": (".kernel_optimized", "OptimizedKernelComputer"),
    "classify_collapse_type": (".kernel_optimized", "classify_collapse_type"),
    "diagnose": (".kernel_optimized", "diagnose"),
    # --- measurement_engine ---
    "EmbeddingConfig": (".measurement_engine", "EmbeddingConfig"),
    "EmbeddingSpec": (".measurement_engine", "EmbeddingSpec"),
    "EmbeddingStrategy": (".measurement_engine", "EmbeddingStrategy"),
    "EngineResult": (".measurement_engine", "EngineResult"),
    "InvariantRow": (".measurement_engine", "InvariantRow"),
    "MeasurementEngine": (".measurement_engine", "MeasurementEngine"),
    "TraceRow": (".measurement_engine", "TraceRow"),
    "safe_tau_R": (".measurement_engine", "safe_tau_R"),
    "tau_R_display": (".measurement_engine", "tau_R_display"),
    # --- seam_optimized ---
    "SeamChainAccumulator": (".seam_optimized", "SeamChainAccumulator"),
    # --- ss1m_triad ---
    "EditionTriad": (".ss1m_triad", "EditionTriad"),
    "compute_triad": (".ss1m_triad", "compute_triad"),
    "triad_to_eid12": (".ss1m_triad", "triad_to_eid12"),
    "verify_triad": (".ss1m_triad", "verify_triad"),
    # --- tau_r_star ---
    "diagnose_thermodynamic": (".tau_r_star", "diagnose"),
    "diagnose_thermodynamic_batch": (".tau_r_star", "diagnose_invariants"),
    # --- tau_r_star_dynamics ---
    "diagnose_extended": (".tau_r_star_dynamics", "diagnose_extended"),
    # --- uncertainty ---
    "KernelGradients": (".uncertainty", "KernelGradients"),
    "UncertaintyBounds": (".uncertainty", "UncertaintyBounds"),
    "compute_kernel_gradients": (".uncertainty", "compute_kernel_gradients"),
    "propagate_uncertainty": (".uncertainty", "propagate_uncertainty"),
    # --- universal_calculator ---
    "ComputationMode": (".universal_calculator", "ComputationMode"),
    "UniversalCalculator": (".universal_calculator", "UniversalCalculator"),
    "UniversalResult": (".universal_calculator", "UniversalResult"),
    "compute_full": (".universal_calculator", "compute_full"),
    "compute_regime": (".universal_calculator", "compute_regime"),
    # --- validator ---
    "RootFileValidator": (".validator", "RootFileValidator"),
    "get_root_validator": (".validator", "get_root_validator"),
    # ValidationResult and validate are defined locally in __init__.py
    # --- weld_lineage ---
    "EditionIdentity": (".weld_lineage", "EditionIdentity"),
    "SS1mReceipt": (".weld_lineage", "SS1mReceipt"),
    "WeldAnchor": (".weld_lineage", "WeldAnchor"),
    "WeldLineage": (".weld_lineage", "WeldLineage"),
    "compute_extended_triad": (".weld_lineage", "compute_extended_triad"),
    "compute_ss1m_receipt": (".weld_lineage", "compute_ss1m_receipt"),
    "create_weld": (".weld_lineage", "create_weld"),
}


def __getattr__(name: str) -> Any:
    """Lazy import — PEP 562.  Resolves public names on first access."""
    entry = _LAZY_IMPORTS.get(name)
    if entry is None:
        msg = f"module 'umcp' has no attribute {name!r}"
        raise AttributeError(msg)
    mod_path, attr = entry
    import importlib

    mod = importlib.import_module(mod_path, __name__)
    obj = mod if attr is None else getattr(mod, attr)
    # Cache on the module dict so __getattr__ is not called again
    globals()[name] = obj
    return obj


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

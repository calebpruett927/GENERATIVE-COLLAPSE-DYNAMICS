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

from __future__ import annotations

__all__ = [
    "DEFAULT_TZ",
    "VALIDATOR_NAME",
    "ClosureLoader",
    "RootFileValidator",
    "UMCPFiles",
    "__version__",
    "get_closure_loader",
    "get_root_validator",
    "get_umcp_files",
]

__version__ = "1.4.0"

VALIDATOR_NAME = "umcp-validator"
DEFAULT_TZ = "America/Chicago"

# Import utilities
from .closures import ClosureLoader, get_closure_loader
from .file_refs import UMCPFiles, get_umcp_files
from .validator import RootFileValidator, get_root_validator

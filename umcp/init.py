"""
UMCP runtime package (UMCP-Metadata-Runnable-Code).

This package is the executable companion to the UMCP manuscript: it is expected to
host (or expose) the validator/runner and the supporting IO needed to execute and
verify pinned UMCP artifacts (schemas, contracts, closures, CasePacks, receipts).

Contract-first / audit-first discipline:
- Do not introduce silent defaults in code paths that affect outputs.
- Any value that affects meaning must be pinned by contract / closure registry / manifest,
  and must surface in receipts.
"""

from __future__ import annotations

__all__ = ["__version__", "get_version"]

# Keep this pinned and bumped explicitly when you tag releases.
__version__ = "0.0.0"


def get_version() -> str:
    """Return the UMCP runtime package version string."""
    return __version__

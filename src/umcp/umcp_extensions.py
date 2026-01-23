"""UMCP Extensions - Reserved for future plugin system.

This module provides a placeholder interface for a future extension/plugin system.
Currently, UMCP has two built-in core features:
  1. Continuous Ledger (automatic validation logging to ledger/return_log.csv)
  2. Contract Auto-Formatter (YAML contract validation and formatting)

Communication extensions (REST API, web UI) would be implemented here when needed.
For now, this returns empty results to maintain API compatibility.
"""

from __future__ import annotations


def list_extensions() -> list[str]:
    """Return list of available extensions.

    Currently returns empty list. Future plugin system would enumerate
    available communication extensions (HTTP API, web UI, etc.).
    """
    return []


def get_extension_info(name: str) -> dict[str, str]:
    """Get information about an extension.

    Args:
        name: Extension name to query

    Returns:
        Dictionary with extension metadata. Currently placeholder.
    """
    return {"name": name, "status": "not_implemented", "info": "Extension system reserved for future use"}

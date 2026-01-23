"""UMCP Extensions - minimal stub for extension system."""

from __future__ import annotations


def list_extensions() -> list[str]:
    """Return list of available extensions."""
    return []


def get_extension_info(name: str) -> dict[str, str]:
    """Get information about an extension."""
    return {"name": name, "info": "No extension info available"}

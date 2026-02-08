"""
Shared dependency imports for UMCP Dashboard.

Centralizes optional visualization dependency handling so every
page module can do: ``from umcp.dashboard._deps import st, pd, px, ...``
"""
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportOptionalMemberAccess=false
# pyright: reportMissingTypeStubs=false
# mypy: warn-unused-ignores=false

from __future__ import annotations

import importlib
from typing import Any

# Try to import visualization dependencies
_has_viz_deps = False

np: Any = None
pd: Any = None
px: Any = None
go: Any = None
st: Any = None
make_subplots: Any = None

try:
    np = importlib.import_module("numpy")
    pd = importlib.import_module("pandas")
    px = importlib.import_module("plotly.express")
    go = importlib.import_module("plotly.graph_objects")
    st = importlib.import_module("streamlit")
    make_subplots = importlib.import_module("plotly.subplots").make_subplots
    _has_viz_deps = True
except ImportError:
    pass

HAS_VIZ_DEPS: bool = _has_viz_deps


def _cache_data(**kwargs: Any) -> Any:
    """Safe wrapper around st.cache_data; no-op when Streamlit is unavailable."""
    if st is not None:
        return st.cache_data(**kwargs)

    # Fallback: identity decorator
    def _identity(func: Any) -> Any:
        return func

    return _identity


__all__ = [
    "HAS_VIZ_DEPS",
    "Any",
    "_cache_data",
    "go",
    "make_subplots",
    "np",
    "pd",
    "px",
    "st",
]

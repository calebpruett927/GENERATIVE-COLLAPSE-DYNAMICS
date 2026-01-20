# UMCP Python Coding Key (Beam)

**Protocol Infrastructure:**  
[GLOSSARY.md](../GLOSSARY.md) | [SYMBOL_INDEX.md](../SYMBOL_INDEX.md#file-format-encodings) | [TERM_INDEX.md](../TERM_INDEX.md)

This document is the authoritative coding and style key for all Python files in this repository.
Goal: consistent imports, consistent typing, stable CLI behavior, predictable test execution, and
audit-friendly error surfaces.

## 1. Repository invariants (do not break)

1) Package root is `src/` (set by `pyproject.toml`).
2) Importable package name is `umcp`.
3) The CLI entry point is `umcp = "umcp.cli:main"` and must always remain valid.
4) Tests use `pytest` and may use relative imports (therefore `tests/` is a package).

Required files:
- `src/umcp/__init__.py`
- `src/umcp/cli.py` (must define `main()`)
- `tests/__init__.py`
- `tests/conftest.py`

Optional but recommended:
- `src/umcp/py.typed` (ship type hints)

## 2. Universal header for Python files

All Python source files should begin with:

- `from __future__ import annotations`
- a short module docstring (one line is fine)
- imports grouped in this order:
  1) standard library
  2) third-party
  3) local package imports (`umcp.*` or relative in tests)

Example header:

```python
from __future__ import annotations

"""Short purpose statement for this module."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────
# UMCP Dev Container — Post-Create Bootstrap
# Runs once after the container is built. Installs all project deps,
# regenerates integrity checksums, and validates the repo is CONFORMANT.
# ──────────────────────────────────────────────────────────────────
set -euo pipefail

echo "══════════════════════════════════════════════════════════════"
echo "  UMCP Dev Container Bootstrap"
echo "══════════════════════════════════════════════════════════════"

# ── 1. Install project in editable mode with ALL extras ──────────
echo ""
echo "▸ [1/6] Installing UMCP with all dependencies..."
pip install --no-cache-dir -e ".[all]"

# ── 2. Regenerate integrity checksums ────────────────────────────
echo ""
echo "▸ [2/6] Regenerating integrity checksums..."
python scripts/update_integrity.py

# ── 3. Verify ruff (lint + format) ──────────────────────────────
echo ""
echo "▸ [3/6] Checking code quality (ruff)..."
if ruff check . --quiet 2>/dev/null && ruff format --check . --quiet 2>/dev/null; then
    echo "  ✓ Ruff lint + format clean"
else
    echo "  ⚠ Ruff issues detected (will be auto-fixed on save)"
fi

# ── 4. Type check (mypy, non-blocking) ──────────────────────────
echo ""
echo "▸ [4/6] Type checking (mypy, informational)..."
if mypy src/umcp --config-file=pyproject.toml 2>/dev/null; then
    echo "  ✓ mypy clean"
else
    echo "  ⚠ mypy issues detected (non-blocking, matches CI)"
fi

# ── 5. Run test suite ───────────────────────────────────────────
echo ""
echo "▸ [5/6] Running test suite..."
if python -m pytest -q --tb=short 2>/dev/null; then
    echo "  ✓ All tests passed"
else
    echo "  ⚠ Some tests failed — run 'pytest -v' for details"
fi

# ── 6. Validate repo ────────────────────────────────────────────
echo ""
echo "▸ [6/6] Validating repo (umcp validate .)..."
if umcp validate . --quiet 2>/dev/null; then
    echo "  ✓ Repo is CONFORMANT"
else
    echo "  ⚠ Validation issues — run 'umcp validate .' for details"
fi

# ── Summary ──────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  UMCP Dev Container Ready"
echo ""
echo "  Quick commands:"
echo "    pytest                          — Run test suite"
echo "    pytest -n auto                  — Parallel test suite"
echo "    umcp validate .                 — Validate repo"
echo "    umcp validate casepacks/<name>  — Validate a casepack"
echo "    python scripts/pre_commit_protocol.py  — Pre-commit gate"
echo "    umcp-api                        — Start REST API (:8000)"
echo "    umcp-dashboard                  — Start dashboard (:8501)"
echo ""
echo "  LaTeX (paper/):"
echo "    latexmk -pdf paper/<file>.tex   — Compile paper"
echo "    cd paper && pdflatex → bibtex → pdflatex × 2"
echo ""
echo "══════════════════════════════════════════════════════════════"
# verified hook runs

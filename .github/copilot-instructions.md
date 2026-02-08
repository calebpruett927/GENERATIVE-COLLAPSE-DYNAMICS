# Copilot Instructions for UMCP-Metadata-Runnable-Code

## What This Project Is

UMCP (Universal Measurement Contract Protocol) validates reproducible computational workflows against mathematical contracts. The unit of work is a **casepack** — a directory containing raw data, a contract reference, closures, and expected outputs. The validator checks schema conformance, Tier-1 kernel identities (F = 1 − ω, IC ≈ exp(κ), IC ≤ F), regime classification, and SHA256 integrity, producing a CONFORMANT/NONCONFORMANT verdict and appending to `ledger/return_log.csv`.

> **Core Axiom**: *"What Returns Through Collapse Is Real"* — Within-run: frozen causes only (no back-edges). Between-run: continuity only by return-weld (τ_R finite + seam residual within tolerance). Reality is declared by showing closure after collapse: each claim is welded to a seam. Frozen parameters (ε, p, α, λ, tol_seam) are not arbitrary constants — they are **consistent across the seam**, meaning the same rules govern both sides of every collapse-return boundary.

## Architecture

```
src/umcp/
├── cli.py              # 2500-line argparse CLI — validation engine, all subcommands
├── validator.py        # Root-file validator (16 files, checksums, math identities)
├── kernel_optimized.py # Lemma-based kernel computation (F, ω, S, C, κ, IC)
├── constants.py        # Regime enum, frozen threshold dataclass, all math constants
├── api_umcp.py         # [Optional] FastAPI REST extension (Pydantic models)
├── dashboard.py        # [Optional] Streamlit dashboard (23 pages)
├── umcp_extensions.py  # Protocol-based plugin system
└── __init__.py         # Public API: validate() convenience function, __version__
```

**Data artifacts** (not Python — never import these):
- `contracts/*.yaml` — versioned mathematical contracts (JSON Schema Draft 2020-12)
- `closures/` — Python/YAML closures organized by domain subdirs (gcd/, rcft/, kinematics/, weyl/, security/)
- `closures/registry.yaml` — central registry; must list every closure used in a run
- `casepacks/*/manifest.json` — casepack manifest referencing contract, closures, expected outputs
- `schemas/*.schema.json` — 13 JSON Schema Draft 2020-12 files validating all artifacts
- `ledger/return_log.csv` — append-only validation log

## Critical Workflows

```bash
pip install -e ".[all]"                     # Dev install (core + api + viz + dev tools)
pytest                                       # 1002+ tests (growing), ~30s
python scripts/update_integrity.py          # MUST run after changing any tracked file
umcp validate .                             # Validate entire repo
umcp validate casepacks/hello_world --strict # Validate casepack (strict = fail on warnings)
```

**⚠️ `python scripts/update_integrity.py` is mandatory** after modifying any `src/umcp/*.py`, `contracts/*.yaml`, `closures/**`, `schemas/**`, or `scripts/*.py` file. It regenerates SHA256 checksums in `integrity/checksums.sha256`. CI will fail on mismatch.

**CI pipeline** (`.github/workflows/validate.yml`): lint (ruff + mypy) → test (pytest) → validate (baseline + strict, both must return CONFORMANT).

## Pre-Commit Protocol (MANDATORY)

**Before every commit**, run the pre-commit protocol:

```bash
python scripts/pre_commit_protocol.py       # Auto-fix + validate (default)
python scripts/pre_commit_protocol.py --check  # Dry-run: report only
```

This script mirrors CI exactly and must exit 0 before committing. It:
1. Runs `ruff format` (auto-fixes formatting)
2. Runs `ruff check --fix` (auto-fixes lint issues)
3. Runs `mypy src/umcp` (reports, non-blocking)
4. Stages all changes (`git add -A`)
5. Regenerates integrity checksums
6. Runs full pytest suite
7. Runs `umcp validate .` (must be CONFORMANT)

See `COMMIT_PROTOCOL.md` for the full specification. **Never skip this step.** Every commit that reaches GitHub must pass all CI checks.

## Code Conventions

**Every source file** starts with `from __future__ import annotations` (PEP 563). Maintain this.

**Optional dependency guarding** — wrap optional imports in `try/except`, set to `None` on failure, check before use. Applied to: yaml, fastapi, streamlit, plotly, pandas, numpy. Never add required imports for optional features.

**Dataclasses** are the dominant data container. `NamedTuples` for immutable math outputs (`KernelInvariants` in `constants.py`). Pydantic `BaseModel` is API-extension only. Serialization uses explicit `.to_dict()` methods, not `dataclasses.asdict()`.

**Three-valued status**: `CONFORMANT` / `NONCONFORMANT` / `NON_EVALUABLE` — never boolean. CLI exit: 0 = CONFORMANT, 1 = NONCONFORMANT.

**`INF_REC` is a typed sentinel**: In CSV/YAML/JSON data it stays as the string `"INF_REC"`. In Python it maps to `float("inf")`. Never coerce the string to a number in data files. When τ_R = INF_REC, the seam budget is zero (no return → no credit). Continuity cannot be synthesized from structure alone — it must be measured.

**Greek letters** (`ω`, `κ`, `Ψ`, `Γ`, `τ`) appear in comments and strings. Ruff rules RUF001/002/003 are suppressed. Line length: 120 chars.

**OPT-* tags** in comments (e.g., `# OPT-1`, `# OPT-12`) reference proven lemmas in `KERNEL_SPECIFICATION.md`. These are formal math cross-references.

## Validation Data Flow

```
umcp validate <target>
  → detect type (repo | casepack | file)
  → schema validation (jsonschema Draft 2020-12)
  → semantic rule checks (validator_rules.yaml: E101, W201, ...)
  → kernel identity checks: F=1−ω, IC≈exp(κ), IC≤F (AM-GM)
  → regime: STABLE|WATCH|COLLAPSE
  → SHA256 integrity check
  → CONFORMANT → append to ledger/return_log.csv + JSON report
```

## Test Patterns

**56 test files** in `tests/`, numbered by group (`test_00_*` through `test_140_*`). Single `tests/conftest.py` provides:
- Frozen `RepoPaths` dataclass (session-scoped) with all critical paths
- `@lru_cache` helpers: `_read_file()`, `_parse_json()`, `_parse_yaml()`, `_compile_schema()`
- Convention: `test_<subject>_<behavior>()` for functions; `TestCLI*` classes with `subprocess.run` for CLI integration

## Extension System

Extensions use `typing.Protocol` (`ExtensionProtocol` requiring `name`, `version`, `description`, `check_dependencies()`). Built-in extensions (api, visualization, ledger, formatter) registered in a plain dict. CLI: `umcp-ext list|info|check|run`. API: `umcp-api` (:8000). Dashboard: `umcp-dashboard` (:8501).

## Key Files to Read First

| To understand... | Read... |
|---|---|
| Validation logic | `src/umcp/cli.py` (top + `_cmd_validate`) |
| Math identities | `src/umcp/validator.py` (`_validate_invariant_identities`) |
| Kernel computation | `src/umcp/kernel_optimized.py` |
| Constants & regimes | `src/umcp/constants.py` |
| Budget thermodynamics | `src/umcp/tau_r_star.py` (τ_R*, phase diagram, arrow of time) |
| Prediction scorecard & nuances | `KERNEL_SPECIFICATION.md` §5 (measured accuracy, outliers as limits, rederived principles, seam-derived constants) |
| Test fixtures | `tests/conftest.py` (first 100 lines) |
| Casepack structure | `casepacks/hello_world/` |
| Contract format | `contracts/UMA.INTSTACK.v1.yaml` |
| Semantic rules | `validator_rules.yaml` |
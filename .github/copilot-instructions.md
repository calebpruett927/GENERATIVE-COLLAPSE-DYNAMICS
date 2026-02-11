# Copilot Instructions for GENERATIVE-COLLAPSE-DYNAMICS

## What This Project Is

UMCP (Universal Measurement Contract Protocol) validates reproducible computational workflows against mathematical contracts. The unit of work is a **casepack** — a directory containing raw data, a contract reference, closures, and expected outputs. The validator checks schema conformance, Tier-1 kernel identities (F = 1 − ω, IC ≈ exp(κ), IC ≤ F), regime classification, and SHA256 integrity, producing a CONFORMANT/NONCONFORMANT verdict and appending to `ledger/return_log.csv`.

> **Core Axiom**: *"What Returns Through Collapse Is Real"* — Within-run: frozen causes only (no back-edges). Between-run: continuity only by return-weld (τ_R finite + seam residual within tolerance). Reality is declared by showing closure after collapse: each claim is welded to a seam. Frozen parameters (ε, p, α, λ, tol_seam) are not arbitrary constants — they are **consistent across the seam**, meaning the same rules govern both sides of every collapse-return boundary.

## Architecture

```
src/umcp/
├── cli.py                    # 2500-line argparse CLI — validation engine, all subcommands
├── validator.py              # Root-file validator (16 files, checksums, math identities)
├── kernel_optimized.py       # Lemma-based kernel computation (F, ω, S, C, κ, IC)
├── seam_optimized.py         # Optimized seam budget computation (Γ, D_C, Δκ)
├── tau_r_star.py             # τ_R* thermodynamic diagnostic (phase diagram)
├── tau_r_star_dynamics.py    # Dynamic τ_R* evolution and trajectories
├── compute_utils.py          # Vectorized utilities (OPT-17,20: coordinate clipping, bounds)
├── epistemic_weld.py         # Epistemic cost tracking (Theorem T9: observation cost)
├── insights.py               # Lessons-learned database (pattern discovery)
├── uncertainty.py            # Uncertainty propagation and error analysis
├── frozen_contract.py        # Frozen contract constants dataclass
├── api_umcp.py               # [Optional] FastAPI REST extension (Pydantic models)
├── finance_cli.py            # Finance domain CLI
├── finance_dashboard.py      # Finance Streamlit dashboard
├── universal_calculator.py   # Universal kernel calculator CLI
├── umcp_extensions.py        # Protocol-based plugin system
├── dashboard/                # [Optional] Modular Streamlit dashboard package
│   ├── __init__.py           # Main dashboard entry point (31 pages)
│   ├── pages_core.py         # Core validation pages
│   ├── pages_analysis.py     # Analysis & diagnostics pages
│   ├── pages_science.py      # Scientific domain pages
│   ├── pages_physics.py      # Physics-specific pages
│   ├── pages_interactive.py  # Interactive exploration pages
│   ├── pages_management.py   # Project management pages
│   └── pages_advanced.py     # Advanced tools & settings
├── fleet/                    # Distributed fleet-scale validation
│   ├── __init__.py           # Fleet public API
│   ├── scheduler.py          # Job scheduler (submit, route, track)
│   ├── worker.py             # Worker + WorkerPool (register, heartbeat, execute)
│   ├── queue.py              # Priority queue (DLQ, retry, backpressure)
│   ├── cache.py              # Content-addressable artifact cache
│   ├── tenant.py             # Multi-tenant isolation, quotas, namespaces
│   └── models.py             # Shared dataclass models (Job, WorkerInfo, etc.)
└── __init__.py               # Public API: validate() convenience function, __version__
```

**Data artifacts** (not Python — never import these):
- `contracts/*.yaml` — 15 versioned mathematical contracts (JSON Schema Draft 2020-12)
- `closures/` — 97 Python/YAML closures organized by domain subdirs (gcd/, rcft/, kinematics/, weyl/, security/, astronomy/, nuclear_physics/, quantum_mechanics/, atomic_physics/, materials_science/, standard_model/, finance/)
- `closures/registry.yaml` — central registry; must list every closure used in a run
- `casepacks/*/manifest.json` — 13 casepack manifests referencing contract, closures, expected outputs
- `schemas/*.schema.json` — 12 JSON Schema Draft 2020-12 files validating all artifacts
- `canon/*.yaml` — 11 canonical anchor files (domain-specific reference points)
- `ledger/return_log.csv` — append-only validation log

## Critical Workflows

```bash
pip install -e ".[all]"                     # Dev install (core + api + viz + dev tools)
pytest                                       # 1900+ tests (growing), ~70s
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

**80 test files** in `tests/`, numbered by tier and domain (`test_00_*` through `test_140_*`). Single `tests/conftest.py` provides:
- Frozen `RepoPaths` dataclass (session-scoped) with all critical paths
- `@lru_cache` helpers: `_read_file()`, `_parse_json()`, `_parse_yaml()`, `_compile_schema()`
- Convention: `test_<subject>_<behavior>()` for functions; `TestCLI*` classes with `subprocess.run` for CLI integration
- New test coverage: `test_fleet_worker.py` (Worker, WorkerPool, WorkerConfig), `test_insights.py` (PatternDatabase, InsightEngine)

## Extension System

Extensions use `typing.Protocol` (`ExtensionProtocol` requiring `name`, `version`, `description`, `check_dependencies()`). Built-in extensions (api, visualization, ledger, formatter) registered in a plain dict. CLI: `umcp-ext list|info|check|run`. API: `umcp-api` (:8000). Dashboard: `umcp-dashboard` (:8501).

## Key Files to Read First

| To understand... | Read... |
|---|---|
| Validation logic | `src/umcp/cli.py` (top + `_cmd_validate`) |
| Math identities | `src/umcp/validator.py` (`_validate_invariant_identities`) |
| Kernel computation | `src/umcp/kernel_optimized.py` |
| Seam budget closure | `src/umcp/seam_optimized.py` (Γ, D_C, Δκ) |
| Thermodynamic diagnostic | `src/umcp/tau_r_star.py` (τ_R*, phase diagram, arrow of time) |
| Epistemic cost tracking | `src/umcp/epistemic_weld.py` (Theorem T9: observation cost) |
| Lessons-learned system | `src/umcp/insights.py` (PatternDatabase, InsightEngine) |
| Fleet architecture | `src/umcp/fleet/` (Scheduler, Worker, Queue, Cache, Tenant) |
| Dashboard pages | `src/umcp/dashboard/` (31 modular pages) |
| Test fixtures | `tests/conftest.py` (first 100 lines) |
| Casepack structure | `casepacks/hello_world/` |
| Contract format | `contracts/UMA.INTSTACK.v1.yaml` |
| Semantic rules | `validator_rules.yaml` |
| Canonical anchors | `canon/` (11 domain anchor files) |

# Copilot Instructions for UMCP-Metadata-Runnable-Code

## Project Overview
UMCP is a production-grade system for creating, validating, and sharing reproducible computational workflows. It enforces mathematical contracts, tracks provenance, generates cryptographic receipts, and validates results against frozen specifications. The system is organized around the concept of a "row"—a minimal publishable unit containing observations, rules, kernel outputs, and receipts.

## Architecture & Key Components

### Core System (No External Dependencies Beyond NumPy/SciPy/YAML/JSON)
- **Core Python Files**: `umcp_autoformat.py`, `umcp_extensions.py`, core validation engine
- **Contracts**: Versioned YAML files in `contracts/` (e.g., `UMA.INTSTACK.v1.yaml`).
- **Closures**: Explicit closure sets in `closures/` with registry in `closures/registry.yaml`.
- **Casepacks**: Reference implementations and manifests in `casepacks/`.
- **Ledger**: Continuous append log in `ledger/return_log.csv`.
- **Schemas**: JSON schema definitions for contracts and closures in `schemas/`.
- **Tests**: Comprehensive suite in `tests/` (pytest, 325+ tests).
- **Validation Engine**: Pure Python validation with mathematical contract enforcement.

### Communication Extensions (Optional - Not Required for Core Validation)
- **REST API**: FastAPI server (`api_umcp.py`) for remote validation and ledger access
  - Install with: `pip install umcp[api]`
  - Run with: `umcp-api` or `uvicorn api_umcp:app --reload`
  - Endpoints: `/health`, `/latest-receipt`, `/ledger`, `/stats`, `/regime`
- **Visualization**: Streamlit dashboard for interactive exploration
  - Install with: `pip install umcp[viz]`
  - Run with: `streamlit run umcp-ext/streamlit_viz.py`
- **All Communications**: Install with `pip install umcp[communications]`

### Extensions Directory (Auto-Discovery Plugins)
- **Extensions**: Auto-discovery plugins in `umcp-ext/`
- **Extensions Command**: `umcp-ext list` to enumerate available plugins

## Developer Workflows

### Core Validation (No Extensions Required)
- **Validation**: `umcp validate` or `python scripts/update_integrity.py` (run after changes to core files).
- **Formatting**: `umcp-format --all` to auto-format contracts.
- **Testing**: `pytest`, `pytest -v`, `pytest -k "gcd"`, `pytest --cov`.
- **CI/CD**: Automated via GitHub Actions (`.github/workflows/validate.yml`).
- **Code Style**: Enforced with `ruff` and `black`.

### Optional Communication Extensions
- **API Server**: Requires `pip install umcp[api]` first
  - `umcp-api` or `uvicorn api_umcp:app --reload` (port 8000)
  - Used for: Remote validation, ledger queries, health checks
- **Visualization**: Requires `pip install umcp[viz]` first
  - Used for: Interactive exploration, receipt visualization, trend analysis
- **Extensions**: `umcp-ext list` to enumerate plugins

## Installation Options

```bash
# Core validation engine only (minimal dependencies)
pip install umcp

# With API communication layer
pip install umcp[api]

# With visualization communication layer
pip install umcp[viz]

# With all communication extensions
pip install umcp[communications]

# Development environment
pip install umcp[dev]

# Everything
pip install umcp[all]
```

## Project-Specific Conventions
- **Typed Boundary Values**: Do not coerce types (e.g., `tau_R = INF_REC` must remain typed).
- **Registry Enforcement**: Closure registry must reference all files used in a run; missing/ambiguous registry breaks conformance.
- **Custom Validation**: Add semantic rules in `validator_rules.yaml`.
- **Publication Rows**: Use tools/scripts in `scripts/` and follow formats in `PUBLICATION_INFRASTRUCTURE.md`.
- **Tier System**: See `TIER_SYSTEM.md` for interface/kernel/weld/overlay boundaries and dependency flow.
- **Kernel Specification**: Mathematical invariants and implementation bounds in `KERNEL_SPECIFICATION.md`.
- **Infrastructure Geometry**: See `INFRASTRUCTURE_GEOMETRY.md` for the three-layer geometric architecture (state space → projections → seam graph) and what the infrastructure "holds" (portable state, geometry, coordinates, partitions, continuity tests).

## Communication vs Core
**IMPORTANT**: The REST API and visualization dashboard are **communication extensions** for standard interfaces (HTTP, web UI). They are NOT required for core validation functionality. UMCP core runs entirely with CLI commands and Python imports.

## Integration Points
- **REST API** (Optional Extension): Endpoints for remote systems
- **Structured Logging**: JSON logs for ELK/Splunk/CloudWatch
- **Docker/Kubernetes**: See `docs/production_deployment.md` for deployment

## Key References
- [README.md](README.md): High-level overview, CLI commands, workflow, architecture diagrams.
- [INFRASTRUCTURE_GEOMETRY.md](INFRASTRUCTURE_GEOMETRY.md): Three-layer geometric architecture and what the infrastructure holds.
- [EXTENSION_INTEGRATION.md](EXTENSION_INTEGRATION.md): Extension system documentation.
- [CASEPACK_REFERENCE.md](CASEPACK_REFERENCE.md): CasePack structure and validation.
- [KERNEL_SPECIFICATION.md](KERNEL_SPECIFICATION.md): Formal specification and debugging guide.
- [TIER_SYSTEM.md](TIER_SYSTEM.md): System boundaries and dependency rules.
- [PUBLICATION_INFRASTRUCTURE.md](PUBLICATION_INFRASTRUCTURE.md): Publication formats and conventions.

## Example Workflow (Core Only - No Extensions)
1. Prepare data (`raw_measurements.csv`).
2. Validate with `umcp validate`.
3. Format contracts with `umcp-format --all`.
4. Run tests with `pytest`.
5. Update integrity with `python scripts/update_integrity.py`.

## Example Workflow (With API Communication Extension)
1. Install API extension: `pip install umcp[api]`
2. Start API server: `umcp-api`
3. Query remotely: `curl http://localhost:8000/health`

---
For unclear or incomplete sections, please provide feedback to improve these instructions.
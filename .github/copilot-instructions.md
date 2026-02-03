# Copilot Instructions for UMCP-Metadata-Runnable-Code

## Project Overview
UMCP is a production-grade system for creating, validating, and sharing reproducible computational workflows. It enforces mathematical contracts, tracks provenance, generates cryptographic receipts, and validates results against frozen specifications. The system is organized around the concept of a "row"—a minimal publishable unit containing observations, rules, kernel outputs, and receipts.

### Core Axiom
> **"What Returns Through Collapse Is Real"** — Reality is defined by what persists through collapse-reconstruction cycles.

### Core Principle
> **One-way dependency flow within a frozen run, with return-based canonization between runs.**

- **Within-run**: Frozen causes only. The frozen interface determines the bounded trace Ψ(t); Tier-1 invariants are pure functions of that frozen trace; Tier-2 overlays may read Tier-1 outputs but cannot alter the interface, trace, or kernel definitions. No back-edges, no retroactive tuning.
- **Between-run**: Continuity only by return-weld. New runs are canon-continuous with prior runs only if the seam returns (τ_R finite) and the weld closes (ledger–budget residual within tolerance + identity check).

### Constitutional Clauses (equivalent formulations)
- "Within-run: frozen causes only. Between-run: continuity only by return-weld."
- "Runs are deterministic under /freeze; canon is a graph whose edges require returned seam closure."
- "No back-edges inside a run; no canon claims between runs without welded return."

## Architecture & Key Components

### Core System (No External Dependencies Beyond NumPy/SciPy/YAML/JSON)
- **Core Python Files**: `src/umcp/validator.py`, `src/umcp/cli.py`, `src/umcp/umcp_extensions.py`
- **Contracts**: Versioned YAML files in `contracts/` (e.g., `UMA.INTSTACK.v1.yaml`).
- **Closures**: Explicit closure sets in `closures/` with registry in `closures/registry.yaml`.
- **Casepacks**: Reference implementations and manifests in `casepacks/`.
- **Ledger**: Continuous append log in `ledger/return_log.csv`.
- **Schemas**: JSON schema definitions for contracts and closures in `schemas/`.
- **Tests**: Comprehensive suite in `tests/` (pytest, 740+ tests).
- **Validation Engine**: Pure Python validation with mathematical contract enforcement.

### Communication Extensions (Optional - Fully Implemented)
- **REST API**: ✅ FastAPI server for remote validation and ledger access
  - Install with: `pip install umcp[api]`
  - Run with: `umcp-api` or `uvicorn umcp.api_umcp:app --reload`
  - Endpoints: `/health`, `/validate`, `/casepacks`, `/ledger`, `/contracts`, `/closures`
- **Visualization**: ✅ Streamlit dashboard for interactive exploration
  - Install with: `pip install umcp[viz]`
  - Run with: `umcp-dashboard` or `streamlit run src/umcp/dashboard.py`
  - 8 pages: Overview, Ledger, Casepacks, Contracts, Closures, Regime, Metrics, Health
- **All Communications**: Install with `pip install umcp[communications]`

### Extensions System (Fully Implemented)
- **Extension Registry**: `src/umcp/umcp_extensions.py` - complete plugin system
- **Extension CLI**: `umcp-ext list|info|check|install|run`
- **Available Extensions**: api, visualization, ledger, formatter

## Developer Workflows

### Core Validation (No Extensions Required)
- **Validation**: `umcp validate` or `python scripts/update_integrity.py` (run after changes to core files).
- **Testing**: `pytest`, `pytest -v`, `pytest -k "gcd"`, `pytest --cov`.
- **CI/CD**: Automated via GitHub Actions (`.github/workflows/validate.yml`).
- **Code Style**: Enforced with `ruff`.

### Communication Extensions
- **API Server**: `umcp-api` or `umcp-ext run api`
- **Dashboard**: `umcp-dashboard` or `umcp-ext run visualization`
- **List Extensions**: `umcp-ext list`
- **Check Dependencies**: `umcp-ext check api`

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
- **REST API** (Optional Extension): Full REST endpoints for remote systems at `http://localhost:8000`
- **Streamlit Dashboard** (Optional Extension): Interactive UI at `http://localhost:8501`
- **Structured Logging**: JSON logs for ELK/Splunk/CloudWatch
- **Docker/Kubernetes**: See `docs/production_deployment.md` for deployment

## Key References
- README.md: High-level overview, CLI commands, workflow, architecture diagrams.
- INFRASTRUCTURE_GEOMETRY.md: Three-layer geometric architecture and what the infrastructure holds.
- EXTENSION_INTEGRATION.md: Extension system documentation.
- CASEPACK_REFERENCE.md: CasePack structure and validation.
- KERNEL_SPECIFICATION.md: Formal specification and debugging guide.
- TIER_SYSTEM.md: System boundaries and dependency rules.
- PUBLICATION_INFRASTRUCTURE.md: Publication formats and conventions.

## Example Workflow (Core Only - No Extensions)
1. Prepare data (`raw_measurements.csv`).
2. Validate with `umcp validate`.
3. Run tests with `pytest`.
4. Update integrity with `python scripts/update_integrity.py`.

## Example Workflow (With API Communication Extension)
1. Install API extension: `pip install umcp[api]`
2. Start API server: `umcp-api` or `uvicorn umcp.api_umcp:app --reload`
3. Query remotely: `curl http://localhost:8000/health`
4. View docs: `http://localhost:8000/docs`

## Example Workflow (With Visualization Dashboard)
1. Install viz extension: `pip install umcp[viz]`
2. Start dashboard: `umcp-dashboard` or `streamlit run src/umcp/dashboard.py`
3. Open browser: `http://localhost:8501`

---
For unclear or incomplete sections, please provide feedback to improve these instructions.
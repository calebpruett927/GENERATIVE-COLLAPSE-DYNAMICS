# Copilot Instructions for UMCP-Metadata-Runnable-Code

## Project Overview
UMCP is a production-grade system for creating, validating, and sharing reproducible computational workflows. It enforces mathematical contracts, tracks provenance, generates cryptographic receipts, and validates results against frozen specifications. The system is organized around the concept of a "row"—a minimal publishable unit containing observations, rules, kernel outputs, and receipts.

## Architecture & Key Components
- **Core Python Files**: `api_umcp.py`, `umcp_autoformat.py`, `umcp_extensions.py`
- **Contracts**: Versioned YAML files in `contracts/` (e.g., `UMA.INTSTACK.v1.yaml`).
- **Closures**: Explicit closure sets in `closures/` with registry in `closures/registry.yaml`.
- **Casepacks**: Reference implementations and manifests in `casepacks/`.
- **Ledger**: Continuous append log in `ledger/return_log.csv`.
- **Schemas**: JSON schema definitions for contracts and closures in `schemas/`.
- **Extensions**: Auto-discovery plugins in `umcp-ext/`.
- **Tests**: Comprehensive suite in `tests/` (pytest, 325+ tests).

## Developer Workflows
- **Validation**: `umcp validate` or `python scripts/update_integrity.py` (run after changes to `api_umcp.py`).
- **API Server**: `umcp-api` or `uvicorn api_umcp:app --reload` (port 8000).
- **Extensions**: `umcp-ext list` to enumerate plugins.
- **Formatting**: `umcp-format --all` to auto-format contracts.
- **Testing**: `pytest`, `pytest -v`, `pytest -k "gcd"`, `pytest --cov`.
- **CI/CD**: Automated via GitHub Actions (`.github/workflows/validate.yml`).
- **Code Style**: Enforced with `ruff` and `black`.

## Project-Specific Conventions
- **Typed Boundary Values**: Do not coerce types (e.g., `tau_R = INF_REC` must remain typed).
- **Registry Enforcement**: Closure registry must reference all files used in a run; missing/ambiguous registry breaks conformance.
- **Custom Validation**: Add semantic rules in `validator_rules.yaml`.
- **Publication Rows**: Use tools/scripts in `scripts/` and follow formats in `PUBLICATION_INFRASTRUCTURE.md`.
- **Tier System**: See `TIER_SYSTEM.md` for interface/kernel/weld/overlay boundaries and dependency flow.
- **Kernel Specification**: Mathematical invariants and implementation bounds in `KERNEL_SPECIFICATION.md`.

## Integration Points
- **REST API**: Endpoints `/health`, `/latest-receipt`, `/ledger`, `/stats`, `/regime`.
- **Structured Logging**: JSON logs for ELK/Splunk/CloudWatch.
- **Docker/Kubernetes**: See `docs/production_deployment.md` for deployment.

## Key References
- [README.md](README.md): High-level overview, CLI commands, workflow, architecture diagrams.
- [EXTENSION_INTEGRATION.md](EXTENSION_INTEGRATION.md): Extension system documentation.
- [CASEPACK_REFERENCE.md](CASEPACK_REFERENCE.md): CasePack structure and validation.
- [KERNEL_SPECIFICATION.md](KERNEL_SPECIFICATION.md): Formal specification and debugging guide.
- [TIER_SYSTEM.md](TIER_SYSTEM.md): System boundaries and dependency rules.
- [PUBLICATION_INFRASTRUCTURE.md](PUBLICATION_INFRASTRUCTURE.md): Publication formats and conventions.

## Example Workflow
1. Prepare data (`raw_measurements.csv`).
2. Validate with `umcp validate`.
3. Format contracts with `umcp-format --all`.
4. Run tests with `pytest`.
5. Update integrity with `python scripts/update_integrity.py`.
6. Start API server with `umcp-api`.

---
For unclear or incomplete sections, please provide feedback to improve these instructions.
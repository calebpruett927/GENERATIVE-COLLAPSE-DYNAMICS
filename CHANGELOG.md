# Changelog

All notable changes to the UMCP validator and repository will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-01-18

### Added
- Initial production release of UMCP validator
- Complete CLI interface with `validate`, `run`, and `diff` commands
- Comprehensive schema validation for all UMCP artifacts:
  - Contracts (UMA.INTSTACK.v1, v1.0.1, v2)
  - Closures registry and definitions
  - Canon anchors
  - CasePack manifests, traces, invariants, and receipts
- Semantic validation rules engine with extensible YAML configuration
- 47 comprehensive test cases covering:
  - Schema validation
  - Contract and closure validation
  - CasePack validation
  - Semantic rules (regime checks, identity checks, critical overlays)
  - CLI commands and diff functionality
  - Edge cases and error handling
- Full CasePack example: `casepacks/hello_world/`
- Complete documentation:
  - README with quickstart and merge verification
  - Contract versioning guidelines (contracts/CHANGELOG.md)
  - Python coding standards
  - Validator usage guide
- CI/CD workflow with GitHub Actions
- Production-quality code with type hints and comprehensive error handling

### Features
- **Validator CLI**: Production-grade command-line interface
  - Schema and structural validation
  - Semantic rule enforcement
  - Receipt comparison and diff capabilities
  - Strict mode for publication-level validation
- **Closure System**: 4 closure implementations with registry
  - gamma (Γ forms)
  - return_domain (window-based)
  - norms (L2 with threshold)
  - curvature_neighborhood
- **Schema Library**: 10 JSON schemas for complete artifact validation
- **Provenance Tracking**: Git commit, Python version, and timestamp tracking
- **Extensible Architecture**: YAML-based rule configuration for easy customization

### Dependencies
- Python >= 3.11
- pyyaml >= 6.0.1
- jsonschema >= 4.23.0
- pytest >= 8.0.0 (dev)
- ruff >= 0.5.0 (dev)

### Known Limitations
- CasePack generation from raw measurements not yet implemented
- Numerical engine for Ψ(t) trace generation is future work
- Seam receipt validation implemented but requires contract continuity claims

### Notes
- This release focuses on validation and metadata conformance
- Runnable numerical computation will be added in future releases
- All 47 tests passing with 100% validation conformance

---

## Version History

For contract-specific changes, see [contracts/CHANGELOG.md](contracts/CHANGELOG.md).

[0.1.0]: https://github.com/calebpruett927/UMCP-Metadata-Runnable-Code/releases/tag/v0.1.0

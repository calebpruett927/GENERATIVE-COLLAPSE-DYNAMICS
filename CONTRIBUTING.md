# Contributing to UMCP

Thank you for contributing to the Universal Measurement Contract Protocol. This repository prioritizes reproducibility, auditability, and deterministic outputs. Please keep changes small, testable, and well-documented.

## Development Environment

**Requirements:**
- Python: 3.11+ (match `pyproject.toml` / CI)
- OS: Linux/macOS recommended (Windows via WSL is acceptable)

**Setup:**

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
python -m pip install -U pip
```

Install dependencies:

```bash
pip install -e ".[dev]"
```

This installs the package in editable mode with all development dependencies (pytest, mypy, ruff, pre-commit, etc.).

## Pre-commit Hooks

We use pre-commit to run formatting/linting/hygiene checks locally before commits.

**Install hooks:**

```bash
pre-commit install
```

**Run on all files:**

```bash
pre-commit run --all-files
```

The hooks will automatically:
- Format code with Ruff
- Lint code with Ruff
- Check YAML, TOML, and JSON syntax
- Detect trailing whitespace and fix line endings
- Check for accidentally committed secrets

## Code Quality Tools

### Formatting and Linting (Ruff)

**Format code:**

```bash
ruff format .
```

**Lint code:**

```bash
ruff check .           # Check for issues
ruff check . --fix     # Auto-fix issues where possible
```

**CI runs:**
```bash
ruff format --check .  # Verify formatting
ruff check .           # Verify linting
```

### Type Checking (mypy)

Run type checking:

```bash
mypy .
```

Or check only the package:

```bash
mypy src/umcp
```

**Guidelines:**
- Do not introduce broad `# type: ignore` usage
- Prefer local typing fixes or narrowly scoped module overrides
- Document any necessary type ignores with explanations

## Tests and Coverage

**Run tests:**

```bash
pytest
```

**Run tests with coverage:**

```bash
pytest --cov=umcp --cov-report=term-missing
```

**Check coverage threshold:**

```bash
pytest --cov=umcp --cov-fail-under=80
```

CI enforces a minimum **80% coverage threshold**. When adding new code paths:
- Add tests that cover both success and failure modes
- Test edge cases and error handling
- Use fixtures for test data when appropriate

## CasePacks and Validation

CasePacks are self-contained, reproducible computational units that serve as audited artifacts.

**Structure:**
- `manifest.json` or `manifest.yaml` - What was computed
- Contract snapshot - Frozen semantics
- Closures/registries - Executable code
- Data - Inputs and outputs
- Receipt - Cryptographic proof

**Validate a CasePack:**

```bash
umcp validate casepacks/<CASEPACK_ID>
```

**Validate the entire repository:**

```bash
umcp validate .
```

**Expectations:**
- CasePacks must be self-describing
- Any continuity/seam claim must be traceable to receipts and manifest hashes
- Validation should return `CONFORMANT` status with 0 errors

## Making Changes

**Workflow:**

1. Create a branch from `main`
2. Make focused changes (or a small set of related changes)
3. Add or update tests
4. Ensure all checks pass:
   ```bash
   pre-commit run --all-files
   pytest --cov=umcp --cov-fail-under=80
   mypy src/umcp
   ```
5. Open a Pull Request

### Branch Naming

Use descriptive prefixes:
- `fix/<short-description>` - Bug fixes
- `feat/<short-description>` - New features
- `chore/<short-description>` - Maintenance tasks
- `docs/<short-description>` - Documentation updates
- `test/<short-description>` - Test additions/improvements

### Commit Messages

Use clear, descriptive commits. Conventional format is preferred:

```
feat: add recursive field memory computation
fix: correct SHA256 verification for modified contracts
chore: update dependencies to latest stable versions
docs: clarify contract-first validation workflow
test: add coverage for typed censoring edge cases
```

### Pull Requests

A good PR includes:

- **Clear description** of what changed and why
- **Test coverage** for new behavior
- **Documentation updates** if behavior changes
- **Backward compatibility notes** if relevant
- **Commands/config changes** that might impact CI or users

## CI Expectations

Our CI pipeline runs:

1. **Tests** - Full pytest suite with coverage ≥80%
2. **Type checking** - mypy strict mode
3. **Linting** - Ruff format and lint checks
4. **UMCP validation** - Baseline and strict mode validation

**Common failure fixes:**

- **Coverage below 80%** - Add tests for uncovered code paths
- **Type errors** - Add type annotations or fix type mismatches
- **Formatting** - Run `ruff format .`
- **Linting** - Run `ruff check . --fix`
- **UMCP validation fails** - Check contract integrity, closure paths, and receipt validity

## Reporting Issues

When filing a bug:

1. **Include the exact command** that failed
2. **Describe expected vs actual behavior**
3. **Provide logs** and error messages
4. **Attach the smallest CasePack** that reproduces the issue (if applicable)
5. **Specify your environment** (Python version, OS, package version)

## Standards and Specifications

- **Core Axiom**: "What Returns Through Collapse Is Real" (`no_return_no_credit: true`)
- **Contracts** live in `contracts/` and define frozen mathematical specifications
- **Closures** live in `closures/` and implement computational functions
- **Schemas** live in `schemas/` and validate all YAML/JSON structures
- **Canon** lives in `canon/` and defines tier-specific anchor specifications

Key files:
- [AXIOM.md](AXIOM.md) - Core principle and philosophy
- [GLOSSARY.md](GLOSSARY.md) - Authoritative term definitions
- [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md) - Component interconnections

## License and Provenance

- This project is licensed under the MIT License
- Do not add content you do not have rights to distribute
- For third-party code, include attribution and verify license compatibility
- All contributions will be subject to the project license

## Questions?

- Check the [README.md](README.md) for quickstart guides
- Review [GLOSSARY.md](GLOSSARY.md) for term definitions
- See [docs/](docs/) for detailed documentation
- Open a GitHub issue for questions not covered by documentation

Thank you for helping make computational science more reproducible and auditable!

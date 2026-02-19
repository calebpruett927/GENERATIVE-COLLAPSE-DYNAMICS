# Contributing to UMCP

<p align="center">
  <strong>Thank you for your interest in contributing to the Universal Measurement Contract Protocol!</strong>
</p>

UMCP is a community-driven project that prioritizes **reproducibility**, **auditability**, and **deterministic outputs**. Every contribution helps advance the cause of reproducible science.

---

## 📋 Table of Contents

- [Code of Conduct](#-code-of-conduct)
- [Getting Started](#-getting-started)
- [Development Environment](#-development-environment)
- [Contribution Types](#-contribution-types)
- [Development Workflow](#-development-workflow)
- [Code Standards](#-code-standards)
- [Testing Guidelines](#-testing-guidelines)
- [Documentation Standards](#-documentation-standards)
- [Pull Request Process](#-pull-request-process)
- [CasePack Development](#-casepack-development)
- [Closure Development](#-closure-development)
- [CI/CD Pipeline](#-cicd-pipeline)
- [Architecture Guide](#-architecture-guide)
- [FAQ](#-faq)

---

## 📜 Code of Conduct

We are committed to providing a welcoming and inclusive environment. All contributors are expected to:

- ✅ Be respectful and considerate
- ✅ Accept constructive criticism gracefully
- ✅ Focus on what is best for the project
- ✅ Show empathy towards other community members
- ❌ No harassment, discrimination, or personal attacks
- ❌ No trolling or inflammatory comments

Violations may result in removal from the project.

---

## 🚀 Getting Started

### Prerequisites

| Requirement | Version | Check Command |
|-------------|---------|---------------|
| Python | 3.11+ | `python --version` |
| pip | Latest | `pip --version` |
| git | Any | `git --version` |
| OS | Linux/macOS (Windows via WSL) | - |

### Quick Setup

```bash
# 1. Fork the repository on GitHub
# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/GENERATIVE-COLLAPSE-DYNAMICS.git
cd GENERATIVE-COLLAPSE-DYNAMICS

# 3. Add upstream remote
git remote add upstream https://github.com/calebpruett927/GENERATIVE-COLLAPSE-DYNAMICS.git

# 4. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 5. Install development dependencies
pip install -e ".[dev]"

# 6. Install pre-commit hooks
pre-commit install

# 7. Verify installation
umcp health
pytest --co -q  # Should show 3,616 tests
```

---

## 🛠 Development Environment

### Project Structure Overview

```
GENERATIVE-COLLAPSE-DYNAMICS/
├── src/umcp/              # Core Python implementation
│   ├── frozen_contract.py # Canonical constants
│   ├── validator.py       # Core validation engine
│   ├── cli.py             # Command-line interface
│   ├── api_umcp.py        # REST API
│   └── dashboard.py       # Streamlit dashboard
├── tests/                 # Test suite (3,616 tests)
├── casepacks/             # Reproducible examples
├── closures/              # Computational functions
├── contracts/             # Mathematical contracts
├── canon/                 # Canonical anchors
├── schemas/               # JSON schemas
├── docs/                  # Documentation
└── pyproject.toml         # Project configuration
```

### IDE Setup

#### VS Code (Recommended)

Install these extensions:
- **Python** (ms-python.python)
- **Pylance** (ms-python.vscode-pylance)
- **Ruff** (charliermarsh.ruff)

Create `.vscode/settings.json`:

```json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
    "python.analysis.typeCheckingMode": "basic",
    "editor.formatOnSave": true,
    "[python]": {
        "editor.defaultFormatter": "charliermarsh.ruff"
    },
    "ruff.enable": true,
    "ruff.lint.run": "onSave"
}
```

#### PyCharm

1. Open the project directory
2. Configure Python interpreter: `.venv/bin/python`
3. Enable Ruff plugin
4. Set file watchers for auto-formatting

---

## 🎯 Contribution Types

### 🟢 Good First Issues (Easy)

| Type | Description | Example |
|------|-------------|---------|
| 📖 Documentation | Fix typos, improve clarity | Update README sections |
| 🧪 Tests | Add test coverage | New test for edge case |
| 🐛 Bug fixes | Fix minor issues | Correct error message |
| 💅 Style | Code formatting | Apply ruff fixes |

### 🟡 Intermediate Issues

| Type | Description | Example |
|------|-------------|---------|
| ✨ Features | New functionality | Add CLI command |
| 🔧 Refactoring | Improve code structure | Optimize function |
| 📦 CasePacks | New examples | Create tutorial casepack |
| 🔌 API | New endpoints | Add analysis endpoint |

### 🔴 Advanced Issues

| Type | Description | Example |
|------|-------------|---------|
| 🔬 Closures | New computational functions | Implement new metric |
| 📐 Frameworks | New framework tier | Add domain-specific layer |
| 🧮 Lemmas | Mathematical proofs | Derive new theorem |
| 🏗 Architecture | Core system changes | Modify validation engine |

---

## 🔄 Development Workflow

### Branch Naming Convention

```
<type>/<short-description>

Types:
- feat/     New features
- fix/      Bug fixes
- docs/     Documentation
- test/     Test additions
- refactor/ Code restructuring
- chore/    Maintenance tasks
- perf/     Performance improvements
```

Examples:
- `feat/add-correlation-endpoint`
- `fix/seam-tolerance-validation`
- `docs/update-quickstart`
- `test/add-rcft-edge-cases`

### Commit Message Format

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `test` | Adding/modifying tests |
| `refactor` | Code change (no new feature/fix) |
| `perf` | Performance improvement |
| `chore` | Maintenance tasks |
| `ci` | CI/CD changes |

Examples:

```bash
feat(api): add ledger analysis endpoint

fix(validator): correct seam tolerance check for edge case

docs(readme): update test count to 3,616

test(closures): add coverage for kinematic stability
```

### Workflow Steps

```bash
# 1. Sync with upstream
git fetch upstream
git checkout main
git merge upstream/main

# 2. Create feature branch
git checkout -b feat/your-feature

# 3. Make changes
# ... edit files ...

# 4. Run quality checks
pre-commit run --all-files
pytest
mypy src/umcp

# 5. Stage and commit
git add .
git commit -m "feat(scope): description"

# 6. Push to your fork
git push origin feat/your-feature

# 7. Open Pull Request on GitHub
```

---

## 📏 Code Standards

### Python Style Guide

We follow **PEP 8** with Ruff enforcement:

```python
# ✅ Good
def compute_kernel(
    coordinates: np.ndarray,
    weights: np.ndarray,
    tau_R: float = 5.0,
    epsilon: float = 1e-8,
) -> KernelResult:
    """Compute kernel invariants from coordinates.

    Args:
        coordinates: Coherence values in [0, 1].
        weights: Weights summing to 1.
        tau_R: Return time.
        epsilon: Guard band for numerical stability.

    Returns:
        KernelResult with omega, F, S, C, kappa, IC.

    Raises:
        ValueError: If coordinates are out of range.
    """
    if not np.all((0 <= coordinates) & (coordinates <= 1)):
        raise ValueError("Coordinates must be in [0, 1]")

    # Compute fidelity
    F = np.dot(weights, coordinates)
    omega = 1 - F

    return KernelResult(omega=omega, F=F, ...)


# ❌ Bad
def compute_kernel(c,w,t=5,e=1e-8):  # No types, poor names
    f=np.dot(w,c)  # No spaces, no docstring
    return f
```

### Type Annotations

All public functions MUST have type annotations:

```python
from typing import Any
import numpy as np
from numpy.typing import NDArray

def validate_coordinates(
    coords: NDArray[np.floating[Any]],
    strict: bool = True,
) -> tuple[bool, list[str]]:
    """Validate coordinate array."""
    errors: list[str] = []
    # ...
    return len(errors) == 0, errors
```

### Import Organization

```python
# Standard library
import os
from pathlib import Path

# Third-party
import numpy as np
import yaml
from pydantic import BaseModel

# Local
from umcp.frozen_contract import compute_kernel
from umcp.validator import ValidationResult
```

### Formatting and Linting

```bash
# Format code
ruff format .

# Check for linting issues
ruff check .

# Auto-fix linting issues
ruff check . --fix

# Type checking
mypy src/umcp
```

---

## 🧪 Testing Guidelines

### Test Structure

```
tests/
├── test_frozen_contract.py     # Module tests
├── test_validator.py           # Integration tests
├── test_extended_lemmas.py     # Mathematical lemmas
├── closures/                   # Closure-specific tests
│   └── test_kin_ref_phase.py
└── conftest.py                 # Shared fixtures
```

### Writing Tests

```python
import pytest
import numpy as np
from umcp.frozen_contract import compute_kernel, classify_regime


class TestComputeKernel:
    """Tests for compute_kernel function."""

    def test_basic_computation(self) -> None:
        """Verify basic kernel computation."""
        c = np.array([0.9, 0.85, 0.92])
        w = np.array([0.5, 0.3, 0.2])

        result = compute_kernel(c, w, tau_R=5.0)

        assert 0 <= result.omega <= 1
        assert 0 <= result.F <= 1
        assert result.omega == pytest.approx(1 - result.F)

    def test_boundary_coordinates(self) -> None:
        """Test with boundary values."""
        c = np.array([0.0, 0.5, 1.0])
        w = np.array([0.33, 0.34, 0.33])

        result = compute_kernel(c, w)

        assert result.F == pytest.approx(0.5, rel=0.01)

    def test_invalid_coordinates_raises(self) -> None:
        """Verify error on invalid input."""
        c = np.array([1.5, 0.5, 0.5])  # Out of range
        w = np.array([0.33, 0.34, 0.33])

        with pytest.raises(ValueError, match="Coordinates must be in"):
            compute_kernel(c, w)

    @pytest.mark.parametrize("coords,expected_regime", [
        (np.array([0.99, 0.98, 0.97]), "STABLE"),
        (np.array([0.80, 0.75, 0.70]), "WATCH"),
        (np.array([0.50, 0.40, 0.30]), "COLLAPSE"),
    ])
    def test_regime_classification(
        self,
        coords: np.ndarray,
        expected_regime: str,
    ) -> None:
        """Test regime classification for various inputs."""
        w = np.array([0.33, 0.34, 0.33])
        result = compute_kernel(coords, w)

        regime = classify_regime(
            omega=result.omega,
            F=result.F,
            S=result.S,
            C=result.C,
            integrity=result.IC,
        )

        assert regime.name == expected_regime
```

### Running Tests

```bash
# All tests
pytest

# Verbose output
pytest -v

# Specific file
pytest tests/test_frozen_contract.py

# Specific test
pytest -k "test_basic_computation"

# With coverage
pytest --cov=umcp --cov-report=html

# Coverage threshold (CI enforces 80%)
pytest --cov=umcp --cov-fail-under=80

# Parallel execution
pytest -n auto

# Skip slow tests
pytest -m "not slow"
```

### Test Categories (Markers)

```python
@pytest.mark.slow
def test_large_dataset():
    """This test takes a long time."""
    pass

@pytest.mark.integration
def test_end_to_end():
    """Full system test."""
    pass

@pytest.mark.api
def test_endpoint():
    """API endpoint test."""
    pass
```

---

## 📝 Documentation Standards

### Docstring Format (Google Style)

```python
def validate_casepack(
    path: str | Path,
    strict: bool = True,
    timeout: float | None = None,
) -> ValidationResult:
    """Validate a CasePack against UMCP contracts.

    Performs comprehensive validation including schema checks,
    closure verification, and seam testing.

    Args:
        path: Path to the casepack directory.
        strict: If True, treat warnings as errors.
        timeout: Maximum validation time in seconds.
            None means no timeout.

    Returns:
        ValidationResult containing:
            - status: CONFORMANT or NONCONFORMANT
            - errors: List of error messages
            - warnings: List of warning messages
            - metrics: Computed kernel invariants

    Raises:
        FileNotFoundError: If casepack path doesn't exist.
        ValidationError: If manifest is malformed.
        TimeoutError: If validation exceeds timeout.

    Example:
        >>> result = validate_casepack("casepacks/hello_world")
        >>> print(result.status)
        CONFORMANT
        >>> print(result.metrics.omega)
        0.0

    Note:
        CasePacks must contain a valid manifest.yaml file
        at the root level.

    See Also:
        - `validate_repository`: Validate entire repo
        - `ValidationResult`: Result dataclass
    """
    pass
```

### Markdown Documentation

- Use headers hierarchically (H1 → H2 → H3)
- Include code examples with syntax highlighting
- Link to related documents
- Keep line length under 100 characters
- Use tables for structured information

---

## 🔀 Pull Request Process

### Before Opening PR

- [ ] All tests pass locally (`pytest`)
- [ ] Code is formatted (`ruff format .`)
- [ ] No linting errors (`ruff check .`)
- [ ] No type errors (`mypy src/umcp`)
- [ ] Pre-commit hooks pass
- [ ] Documentation updated if needed
- [ ] Commit messages follow convention

### PR Template

```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix (non-breaking)
- [ ] New feature (non-breaking)
- [ ] Breaking change
- [ ] Documentation update

## Testing
Describe how you tested:
- [ ] New tests added
- [ ] Existing tests pass
- [ ] Manual testing performed

## Checklist
- [ ] My code follows the project style
- [ ] I have added tests for my changes
- [ ] All new and existing tests pass
- [ ] I have updated documentation
- [ ] My changes don't break backwards compatibility

## Related Issues
Fixes #123, Relates to #456
```

### Review Process

1. **Automated checks** must pass (CI/CD)
2. **Code review** by at least one maintainer
3. **Merge** after approval (squash merge preferred)

---

## 📦 CasePack Development

### CasePack Structure

```
casepacks/my_casepack/
├── manifest.yaml          # Required: metadata and config
├── contract.yaml          # Required: mathematical contract
├── raw_measurements.csv   # Required: input data
├── closures/              # Optional: local closures
│   └── my_closure.py
├── expected/              # Optional: expected outputs
│   └── kernel.json
└── README.md              # Optional: documentation
```

### Minimal manifest.yaml

```yaml
name: my_casepack
version: 1.0.0
description: Brief description of the casepack
author: Your Name
created: 2026-02-05

contract: contract.yaml
data:
  measurements: raw_measurements.csv

closures:
  - name: compute_coherence
    path: closures/my_closure.py

validation:
  seam_tolerance: 0.005
  regime_expected: STABLE
```

### Validation

```bash
# Validate your casepack
umcp validate casepacks/my_casepack

# Detailed output
umcp validate casepacks/my_casepack --verbose

# Generate report
umcp report casepacks/my_casepack
```

---

## 🔬 Closure Development

### Closure Structure

```python
"""My custom closure for computing something."""

from typing import Any
import numpy as np
from numpy.typing import NDArray


def my_closure(
    data: NDArray[np.floating[Any]],
    params: dict[str, Any],
) -> dict[str, float]:
    """Compute custom metric from data.

    Args:
        data: Input array of measurements.
        params: Configuration parameters.

    Returns:
        Dictionary with computed values.
    """
    # Implementation
    result = np.mean(data) * params.get("scale", 1.0)

    return {
        "metric": float(result),
        "count": len(data),
    }


# Required for closure registry
CLOSURE_METADATA = {
    "name": "my_closure",
    "version": "1.0.0",
    "description": "Computes custom metric",
    "inputs": ["data", "params"],
    "outputs": ["metric", "count"],
}
```

### Register in registry.yaml

```yaml
closures:
  my_closure:
    path: closures/my_closure.py
    version: 1.0.0
    framework: custom
    description: Computes custom metric
```

---

## 🔄 CI/CD Pipeline

### GitHub Actions Workflow

Our CI runs on every PR and push to main:

```yaml
# .github/workflows/validate.yml
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: pip install -e ".[dev]"

      - name: Ruff format check
        run: ruff format --check .

      - name: Ruff lint
        run: ruff check .

      - name: Type check
        run: mypy src/umcp

      - name: Run tests
        run: pytest --cov=umcp --cov-fail-under=80

      - name: UMCP validation
        run: umcp validate .
```

### Common CI Failures and Fixes

| Failure | Command | Fix |
|---------|---------|-----|
| Formatting | `ruff format --check .` | `ruff format .` |
| Linting | `ruff check .` | `ruff check . --fix` |
| Type errors | `mypy src/umcp` | Add type annotations |
| Tests fail | `pytest` | Fix failing tests |
| Coverage < 80% | `pytest --cov` | Add more tests |

---

## 🏗 Architecture Guide

### Core Components

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLI Layer                               │
│  (cli.py - 10 commands)                                        │
├─────────────────────────────────────────────────────────────────┤
│                         API Layer                               │
│  (api_umcp.py - 37+ endpoints)                                 │
├─────────────────────────────────────────────────────────────────┤
│                      Validation Engine                          │
│  (validator.py)                                                 │
├─────────────────────────────────────────────────────────────────┤
│                       Core Components                           │
│  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐         │
│  │ frozen_       │ │ uncertainty.  │ │ ss1m_triad.   │         │
│  │ contract.py   │ │ py            │ │ py            │         │
│  │ (constants)   │ │ (delta-method)│ │ (checksums)   │         │
│  └───────────────┘ └───────────────┘ └───────────────┘         │
├─────────────────────────────────────────────────────────────────┤
│                        Closures                                 │
│  ┌─────────┐ ┌───────────┐ ┌──────────┐ ┌──────────┐           │
│  │   GCD   │ │ Kinematics│ │   RCFT   │ │   WEYL   │           │
│  │(5 funcs)│ │ (6 funcs) │ │(4 funcs) │ │(5 funcs) │           │
│  └─────────┘ └───────────┘ └──────────┘ └──────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Principles

1. **Frozen Contracts**: Mathematical constants are immutable
2. **One-Way Dependencies**: No back-edges in validation flow
3. **Return-Based Canon**: Only reproduced results become canon
4. **Seam Testing**: Budget conservation: |s| ≤ 0.005
5. **Cryptographic Provenance**: SHA256 receipts for all artifacts

---

## ❓ FAQ

### Q: How do I run only my tests?

```bash
pytest tests/my_test_file.py -v
pytest -k "my_test_name"
```

### Q: My PR fails CI but passes locally?

Check Python version matches CI (3.12), and ensure all dependencies are in `pyproject.toml`.

### Q: How do I add a new CLI command?

Edit `src/umcp/cli.py` and add a new click command:

```python
@click.command()
@click.argument("name")
def mycommand(name: str) -> None:
    """My new command description."""
    click.echo(f"Hello, {name}!")
```

### Q: How do I add a new API endpoint?

Edit `src/umcp/api_umcp.py`:

```python
@app.get("/my-endpoint")
async def my_endpoint():
    """My endpoint description."""
    return {"message": "Hello"}
```

### Q: Where do I report bugs?

Open an issue on [GitHub Issues](https://github.com/calebpruett927/GENERATIVE-COLLAPSE-DYNAMICS/issues) with:
- Exact command that failed
- Expected vs actual behavior
- Python version and OS
- Minimal reproduction steps

---

## 🙏 Thank You

Your contributions make UMCP better for everyone. Every improvement, no matter how small, helps advance reproducible science.

**Questions?** Open an issue or reach out to the maintainers.

---

<div align="center">

*"What Returns Through Collapse Is Real"*

</div>

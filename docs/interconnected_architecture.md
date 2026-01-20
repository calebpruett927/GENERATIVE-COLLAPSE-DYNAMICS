# UMCP Interconnected System Architecture

**Term Definitions:** [GLOSSARY.md](../GLOSSARY.md) | [SYMBOL_INDEX.md](../SYMBOL_INDEX.md) | [TERM_INDEX.md](../TERM_INDEX.md)

## Overview

The UMCP (Universal Measurement Contract Protocol) system is designed with full interconnectedness between components. This document describes how the various parts of the system reference and depend on each other.

## Component Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│                      UMCP Repository                         │
└─────────────────────────────────────────────────────────────┘
                          │
           ┌──────────────┼──────────────┐
           ▼              ▼              ▼
    ┌──────────┐   ┌───────────┐  ┌──────────┐
    │ Manifest │   │ Contract  │  │ Closures │
    │  .yaml   │───│   .yaml   │──│ Registry │
    └──────────┘   └───────────┘  └──────────┘
           │              │              │
           └──────────────┼──────────────┘
                          ▼
              ┌───────────────────────┐
              │   Observables.yaml    │
              │   Embedding.yaml      │
              │   Return.yaml         │
              └───────────────────────┘
                          │
           ┌──────────────┼──────────────┐
           ▼              ▼              ▼
    ┌──────────┐   ┌───────────┐  ┌──────────┐
    │ Weights  │   │   Trace   │  │ Closures │
    │   .csv   │───│   .csv    │──│  .py     │
    └──────────┘   └───────────┘  └──────────┘
           │              │              │
           └──────────────┼──────────────┘
                          ▼
              ┌───────────────────────┐
              │   Invariants.csv      │
              │   Regimes.csv         │
              │   Welds.csv           │
              │   Report.txt          │
              └───────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   Integrity Files     │
              │   - sha256.txt        │
              │   - env.txt           │
              │   - code_version.txt  │
              └───────────────────────┘
```

## Interconnection Details

### 1. Manifest → Contract → Closures

**manifest.yaml** references:
- `refs.contract.id`: Contract identifier (e.g., "UMA.INTSTACK.v1")
- `refs.contract.path`: Path to contract file
- `refs.closures_registry.id`: Closures registry identifier
- `refs.closures_registry.path`: Path to closures registry

**contract.yaml** defines:
- `contract.tier_1_kernel.invariants`: List of Tier-1 invariants (ω, F, S, C, τ_R, κ, IC)
- `contract.embedding`: Embedding parameters for coordinates
- `contract.typed_censoring`: Censoring rules

**closures.yaml** provides:
- Registry ID matching manifest reference
- Paths to individual closure YAML files
- References to Python closure implementations

### 2. Observables → Trace → Invariants

**observables.yaml** defines:
- Primary observables (x₁, x₂, x₃)
- Derived observables (v₁, v₂, v₃)
- Units and domains for each observable

**derived/trace.csv** contains:
- Embedded coordinates c₁, c₂, c₃ ∈ [0,1]
- Computed from observables using embedding rules
- Includes flags (oor_flag, missing_flag)

**outputs/invariants.csv** computes:
- Tier-1 invariants from trace + weights
- `ω = Σ wᵢ(1-cᵢ)` - drift measure
- `F = 1 - ω` - fidelity (Tier-1 identity)
- `S = -Σ wᵢ[cᵢln(cᵢ) + (1-cᵢ)ln(1-cᵢ)]` - Shannon entropy
- `C = std(c)/0.5` - curvature measure
- `κ = Σ wᵢ ln(cᵢ + ε)` - kappa
- `IC = exp(κ)` - integrity check (Tier-1 identity)

### 3. Weights → Calculations

**weights.csv** provides:
- Weight coefficients (w₁, w₂, w₃)
- Constraint: Σ wᵢ = 1.0
- Used in all weighted sum calculations

### 4. Invariants → Regimes

**outputs/regimes.csv** classifies based on invariants:
- **Stable**: ω < 0.038 AND F > 0.90 AND S < 0.15 AND C < 0.14
- **Watch**: 0.038 ≤ ω < 0.30 (intermediate)
- **Collapse**: ω ≥ 0.30 (critical)

### 5. Closures Registry → Python Implementations

**closures/registry.yaml** references:
- `gamma.default.v1.yaml`
- `return_domain.window64.v1.yaml`
- `norms.l2_eta1e-3.v1.yaml`
- `curvature_neighborhood.default.v1.yaml`

**Python closures** provide computational functions:
- `F_from_omega.py`: Force from angular velocity
- `tau_R_compute.py`: Resonance time constant
- `stiffness_check.py`: Stiffness validation
- `hello_world.py`: Simple demonstration closure

### 6. Integrity Verification

**integrity/sha256.txt** contains checksums for:
- All configuration files (manifest, contract, observables, etc.)
- All data files (weights.csv, trace.csv, invariants.csv, etc.)
- Used for audit trail and tampering detection

**integrity/env.txt** records:
- Python version
- Package versions
- Environment reproducibility

**integrity/code_version.txt** tracks:
- Git commit hash
- Repository state
- Code provenance

## Programmatic Access

### Using UMCPFiles

```python
from umcp import get_umcp_files

# Load any UMCP file
files = get_umcp_files()
manifest = files.load_manifest()
contract = files.load_contract()
invariants = files.load_invariants()

# Cross-reference
contract_id = manifest["refs"]["contract"]["id"]
print(f"Using contract: {contract_id}")
```

### Using ClosureLoader

```python
from umcp import get_closure_loader

# Load and execute closures
loader = get_closure_loader()
closures = loader.list_closures()

# Execute a closure
result = loader.execute_closure("F_from_omega", omega=10.0, r=0.5, m=1.0)
print(f"Force: {result['F']} N")
```

### Using RootFileValidator

```python
from umcp import get_root_validator

# Validate entire system
validator = get_root_validator()
result = validator.validate_all()

print(f"Status: {result['status']}")
print(f"Errors: {len(result['errors'])}")
print(f"Passed: {len(result['passed'])}")
```

## Mathematical Dependencies

### Tier-1 Identities

1. **F = 1 - ω** (exact)
   - Fidelity is defined as complement of drift
   - Must be satisfied exactly (tolerance < 1e-9)

2. **IC ≈ exp(κ)** (approximate)
   - Integrity check relates to kappa
   - Tolerance < 1e-6 due to floating-point precision

### Regime Classification Rules

```python
# Stable regime
omega < 0.038 and F > 0.90 and S < 0.15 and C < 0.14

# Watch regime
0.038 <= omega < 0.30

# Collapse regime
omega >= 0.30
```

### Omega Formula

**Correct formula** (matches hello_world example):
```
ω = Σ wᵢ(1 - cᵢ)
```

This measures drift as weighted distance from upper boundary (1.0).

## Validation Chain

1. **Structural Validation**
   - All 16 required files exist
   - YAML/CSV files parse correctly
   - Required fields present

2. **Schema Validation**
   - Files conform to JSON schemas
   - Data types correct
   - Enums and constraints satisfied

3. **Mathematical Validation**
   - Tier-1 identities satisfied
   - Weights sum to 1.0
   - Coordinates in [0,1]

4. **Semantic Validation**
   - Regime classification matches thresholds
   - Continuity verified at seam boundaries
   - Invariants consistent with trace

5. **Integrity Validation**
   - SHA256 checksums match
   - Environment documented
   - Code version tracked

## Testing Interconnections

Run the integration test suite:

```bash
pytest tests/test_97_root_integration.py -v
```

Or run the demonstration script:

```bash
python examples/interconnected_demo.py
```

## Best Practices

1. **Always validate after changes**
   ```bash
   python -m umcp.validator
   ```

2. **Regenerate checksums after updates**
   ```bash
   sha256sum manifest.yaml contract.yaml ... > integrity/sha256.txt
   ```

3. **Use programmatic access for consistency**
   ```python
   from umcp import get_umcp_files
   files = get_umcp_files()
   # Always use files.load_*() methods
   ```

4. **Test closure execution before deployment**
   ```python
   from umcp import get_closure_loader
   loader = get_closure_loader()
   # Test all closures with sample data
   ```

5. **Maintain Tier-1 identities**
   - Never manually edit invariants.csv
   - Always recalculate from trace.csv
   - Verify F = 1-ω and IC ≈ exp(κ)

## Troubleshooting

### Issue: "Regime mismatch"
**Cause**: Invariants don't match thresholds  
**Fix**: Recalculate invariants from trace using correct omega formula

### Issue: "Checksum mismatch"
**Cause**: File modified without updating checksums  
**Fix**: Regenerate `integrity/sha256.txt`

### Issue: "Weights don't sum to 1.0"
**Cause**: Floating-point precision error  
**Fix**: Adjust last weight: `w_3 = 1.0 - w_1 - w_2`

### Issue: "Coordinates out of bounds"
**Cause**: Trace values outside [0,1]  
**Fix**: Check embedding parameters and censoring rules

## References

- [File Reference Documentation](file_reference.md)
- [Quick Reference](file_reference_quick.md)
- [Validator Usage](../canon/docs/validator_usage.md)
- [Python Coding Standards](python_coding_key.md)

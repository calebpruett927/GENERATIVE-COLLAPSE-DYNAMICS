# Root-Level UMCP Files Reference

This document describes the root-level UMCP configuration files and how to reference them programmatically.

## File Structure

```
/
├── manifest.yaml              # CasePack manifest and schema references
├── contract.yaml              # Contract specification
├── observables.yaml           # Observable variable definitions
├── embedding.yaml             # Embedding transformation config
├── return.yaml               # Return domain specifications
├── closures.yaml             # Closure registry references
├── weights.csv               # Weight coefficients
├── derived/                  # Derived data
│   ├── trace.csv            # Bounded trace Ψ_ε(t)
│   └── trace_meta.yaml      # Trace metadata
├── outputs/                  # Computation outputs
│   ├── invariants.csv       # Tier-1 invariants
│   ├── regimes.csv          # Regime classifications
│   ├── welds.csv            # Continuity verification
│   └── report.txt           # Validation report
└── integrity/               # Audit trail
    ├── sha256.txt           # File checksums
    ├── env.txt              # Python environment
    └── code_version.txt     # Git provenance
```

## Programmatic Access

### Using the UMCPFiles Helper

```python
from umcp import get_umcp_files

# Initialize
umcp = get_umcp_files()

# Load configuration files
manifest = umcp.load_manifest()
contract = umcp.load_contract()
observables = umcp.load_observables()
embedding = umcp.load_embedding()
return_config = umcp.load_return()
closures = umcp.load_closures()

# Load data files
weights = umcp.load_weights()
trace = umcp.load_trace()
invariants = umcp.load_invariants()
regimes = umcp.load_regimes()

# Load integrity files
checksums = umcp.load_sha256()
env_info = umcp.load_env()
version_info = umcp.load_code_version()

# Check file existence
missing = umcp.get_missing_files()
if missing:
    print(f"Missing files: {missing}")
```

### Direct Path Access

```python
from umcp import UMCPFiles
from pathlib import Path

umcp = UMCPFiles(Path("/path/to/repo"))

# Access file paths directly
manifest_path = umcp.manifest_yaml
contract_path = umcp.contract_yaml
trace_path = umcp.trace_csv

# Use paths with your own loading logic
with open(umcp.observables_yaml) as f:
    data = yaml.safe_load(f)
```

## File Descriptions

### Configuration Files (YAML)

#### manifest.yaml

CasePack manifest following `schemas/manifest.schema.json`. Contains:
- CasePack metadata (id, version, title, authors)
- References to canon anchors, contract, and closure registry
- Artifact paths (raw measurements, expected outputs)
- Run intent and notes

#### contract.yaml

Contract specification following `schemas/contract.schema.json`. Defines:
- Embedding parameters (interval, face, oor_policy, epsilon)
- Tier-1 kernel parameters (reserved symbols, weights policy, frozen parameters)
- Typed censoring rules (special values, run status enum)

#### observables.yaml

Observable variable definitions including:
- Primary observables from raw measurements
- Derived observables (computed from primary)
- Measurement metadata and quality flags

#### embedding.yaml

Embedding transformation configuration specifying:
- Coordinate transformations from raw to bounded
- OOR (out-of-range) handling policies
- Log-safety parameters

#### return.yaml

Return domain specifications including:
- Return domain generator type and parameters
- Neighborhood tolerance for return evaluation
- Return time computation method
- Censoring policy

#### closures.yaml

Closure registry references specifying:
- Gamma form (drift dissipation closure)
- Return domain generator
- Norm specifications
- Curvature neighborhood
- Computational closure implementations

### Data Files

#### weights.csv

Weight coefficients (w_1, w_2, w_3, ...) satisfying:
- Non-negative values
- Sum to 1.0
- Used in Tier-1 invariant computations

#### derived/trace.csv

Bounded trace Ψ_ε(t) containing:
- Time column (t)
- Coordinate columns (c_1, c_2, c_3, ...)
- Quality flags (oor_flag, missing_flag)
- All coordinates in [0, 1]

#### derived/trace_meta.yaml

Trace metadata including:
- Format specification (wide_psi)
- Source information
- Embedding verification
- Quality metrics
- Generation metadata

### Output Files

#### outputs/invariants.csv

Tier-1 invariants per time point:
- t: Time
- omega (ω): Drift magnitude
- F: Tier-1 invariant (F ≈ 1 - ω)
- S: Bernoulli field entropy
- C: Curvature proxy
- tau_R (τ_R): Return time
- kappa (κ): Log-geometric mean
- IC: Integrity check (IC ≈ exp(κ))
- regime_label: Stable/Watch/Collapse
- critical_overlay: Boolean flag
- IC_min: Minimum coordinate value

#### outputs/regimes.csv

Regime classifications including:
- Time points
- Regime labels with threshold checks
- Critical overlays
- Supporting invariants
- Notes

#### outputs/welds.csv

Continuity verification at seam boundaries:
- Weld IDs
- Pre/post time points
- Pre/post invariant values
- Seam distance
- Conformance status
- Notes

#### outputs/report.txt

Human-readable validation report containing:
- Summary (status, test counts)
- Schema validation results
- Trace validation
- Tier-1 invariant checks
- Regime classification verification
- Continuity (weld) verification
- Closure verification
- Integrity checks
- Conclusion

### Integrity Files

#### integrity/sha256.txt

SHA256 checksums for all tracked files. Format:

```
<hash>  <filepath>
<hash>  <filepath>
...
```

#### integrity/env.txt

Python environment documentation including:
- Python version
- Installed packages with versions
- Used for reproducibility verification

#### integrity/code_version.txt

Git provenance information:
- Commit hash
- Git describe output
- Branch name
- Commit date

## Integration Patterns

### Validation Workflow

```python
from umcp import get_umcp_files

umcp = get_umcp_files()

# 1. Load and validate configuration
manifest = umcp.load_manifest()
contract = umcp.load_contract()

# 2. Load trace data
trace = umcp.load_trace()
trace_meta = umcp.load_trace_meta()

# 3. Load computed invariants
invariants = umcp.load_invariants()

# 4. Verify against contract
for inv_row in invariants:
    omega = float(inv_row['omega'])
    F = float(inv_row['F'])

    # Check Tier-1 identity: F ≈ 1 - ω
    assert abs(F - (1 - omega)) < 1e-9
```

### Audit Trail Verification

```python
umcp = get_umcp_files()

# Verify checksums
import hashlib

checksums = {}
for line in umcp.load_sha256().strip().split('\n'):
    hash_val, filepath = line.split(None, 1)
    checksums[filepath.strip()] = hash_val

# Verify a file
def verify_file(path: Path) -> bool:
    with open(path, 'rb') as f:
        computed = hashlib.sha256(f.read()).hexdigest()
    expected = checksums.get(str(path), None)
    return computed == expected

# Check provenance
version_info = umcp.load_code_version()
print(f"Code version: {version_info}")
```

### Testing Integration

```python
import pytest
from umcp import get_umcp_files

@pytest.fixture
def umcp_files():
    return get_umcp_files()

def test_manifest_exists(umcp_files):
    manifest = umcp_files.load_manifest()
    assert manifest['schema'] == "schemas/manifest.schema.json"

def test_invariants_satisfy_tier1(umcp_files):
    invariants = umcp_files.load_invariants()
    for row in invariants:
        omega = float(row['omega'])
        F = float(row['F'])
        # F ≈ 1 - ω
        assert abs(F - (1 - omega)) < 1e-9
```

## Schema References

All YAML files should reference their corresponding schemas:

- `manifest.yaml` → `schemas/manifest.schema.json`
- `contract.yaml` → `schemas/contract.schema.json`
- `derived/trace_meta.yaml` → (internal format)
- `closures.yaml` → `schemas/closures.schema.json`

CSV files follow the schema implied by:
- `outputs/invariants.csv` → `schemas/invariants.schema.json`
- `derived/trace.csv` → `schemas/trace.psi.schema.json`

## Best Practices

1. **Always use the UMCPFiles helper** for programmatic access to ensure consistent path resolution
2. **Verify file existence** before loading with `umcp.verify_all_exist()`
3. **Check integrity** using SHA256 checksums after loading
4. **Document provenance** by referencing code_version.txt in outputs
5. **Validate against schemas** after loading YAML/JSON files
6. **Handle missing files gracefully** with appropriate error messages

## See Also

- [Production Deployment Guide](production_deployment.md)
- [Quickstart Guide](quickstart.md)
- [Python Coding Key](python_coding_key.md)
- Main README: [../README.md](../README.md)

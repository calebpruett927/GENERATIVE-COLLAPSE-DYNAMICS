# Quick Reference: UMCP File Access

## Command Line Tools

### View all files
```bash
python examples/load_umcp_files.py
```

### Validate root-level files
```bash
python scripts/validate_root_files.py
```

### Run file reference tests
```bash
pytest tests/test_96_file_references.py -v
```

## Python API

### Quick Start
```python
from umcp import get_umcp_files

umcp = get_umcp_files()
manifest = umcp.load_manifest()
```

### Load Configuration
```python
# Metadata files (YAML)
manifest = umcp.load_manifest()      # manifest.yaml
contract = umcp.load_contract()      # contract.yaml
observables = umcp.load_observables()  # observables.yaml
embedding = umcp.load_embedding()    # embedding.yaml
return_config = umcp.load_return()   # return.yaml
closures = umcp.load_closures()      # closures.yaml
```

### Load Data
```python
# Input data
weights = umcp.load_weights()        # weights.csv (list of dicts)

# Derived data
trace = umcp.load_trace()            # derived/trace.csv
trace_meta = umcp.load_trace_meta()  # derived/trace_meta.yaml

# Outputs
invariants = umcp.load_invariants()  # outputs/invariants.csv
regimes = umcp.load_regimes()        # outputs/regimes.csv
welds = umcp.load_welds()            # outputs/welds.csv
report = umcp.load_report()          # outputs/report.txt
```

### Load Integrity Files
```python
# Audit trail
checksums = umcp.load_sha256()          # integrity/sha256.txt
env_info = umcp.load_env()              # integrity/env.txt
version_info = umcp.load_code_version()  # integrity/code_version.txt
```

### File Verification
```python
# Check which files exist
status = umcp.verify_all_exist()
print(status)  # {'manifest.yaml': True, ...}

# Get list of missing files
missing = umcp.get_missing_files()
if missing:
    print(f"Missing: {missing}")
```

### Direct Path Access
```python
# Access file paths directly
manifest_path = umcp.manifest_yaml
contract_path = umcp.contract_yaml
trace_path = umcp.trace_csv

# Use with your own loading
import yaml
with open(umcp.observables_yaml) as f:
    data = yaml.safe_load(f)
```

## File Organization

```
Root Level (Configuration)
├── manifest.yaml          → umcp.load_manifest()
├── contract.yaml          → umcp.load_contract()
├── observables.yaml       → umcp.load_observables()
├── embedding.yaml         → umcp.load_embedding()
├── return.yaml            → umcp.load_return()
├── closures.yaml          → umcp.load_closures()
└── weights.csv            → umcp.load_weights()

Derived (Computed from inputs)
├── trace.csv              → umcp.load_trace()
└── trace_meta.yaml        → umcp.load_trace_meta()

Outputs (Results)
├── invariants.csv         → umcp.load_invariants()
├── regimes.csv            → umcp.load_regimes()
├── welds.csv              → umcp.load_welds()
└── report.txt             → umcp.load_report()

Integrity (Audit trail)
├── sha256.txt             → umcp.load_sha256()
├── env.txt                → umcp.load_env()
└── code_version.txt       → umcp.load_code_version()
```

## Common Patterns

### Validate Tier-1 Invariants
```python
umcp = get_umcp_files()
invariants = umcp.load_invariants()

for inv in invariants:
    omega = float(inv['omega'])
    F = float(inv['F'])
    
    # Check F ≈ 1 - ω
    assert abs(F - (1 - omega)) < 1e-9
```

### Check File Integrity
```python
import hashlib

umcp = get_umcp_files()
checksums = {}

for line in umcp.load_sha256().strip().split('\n'):
    hash_val, filepath = line.split(None, 1)
    checksums[filepath.strip()] = hash_val

def verify_file(path):
    with open(path, 'rb') as f:
        computed = hashlib.sha256(f.read()).hexdigest()
    expected = checksums.get(str(path))
    return computed == expected

# Verify manifest
assert verify_file(umcp.manifest_yaml)
```

### Load All Configuration
```python
umcp = get_umcp_files()

config = {
    'manifest': umcp.load_manifest(),
    'contract': umcp.load_contract(),
    'observables': umcp.load_observables(),
    'embedding': umcp.load_embedding(),
    'return': umcp.load_return(),
    'closures': umcp.load_closures(),
}

print(f"Loaded {len(config)} configuration files")
```

### Custom Repository Path
```python
from pathlib import Path
from umcp import UMCPFiles

# Specify custom repository root
umcp = UMCPFiles(Path("/path/to/repo"))
manifest = umcp.load_manifest()
```

## Integration with Tests

```python
import pytest
from umcp import get_umcp_files

@pytest.fixture
def umcp():
    return get_umcp_files()

def test_manifest_schema(umcp):
    manifest = umcp.load_manifest()
    assert manifest['schema'] == 'schemas/manifest.schema.json'

def test_weights_sum_to_one(umcp):
    weights = umcp.load_weights()
    w_values = [float(v) for v in weights[0].values()]
    assert abs(sum(w_values) - 1.0) < 1e-9
```

## See Also

- **Comprehensive Guide**: [docs/file_reference.md](../docs/file_reference.md)
- **Example Script**: [examples/load_umcp_files.py](../examples/load_umcp_files.py)
- **Validation Script**: [scripts/validate_root_files.py](../scripts/validate_root_files.py)
- **Tests**: [tests/test_96_file_references.py](../tests/test_96_file_references.py)
- **Main README**: [README.md](../README.md)

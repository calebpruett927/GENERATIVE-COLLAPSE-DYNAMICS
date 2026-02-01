# UMCP Quick Tutorial: From Zero to First Validation

**Estimated time**: 10 minutes  
**Prerequisites**: Python 3.11+, pip installed

This tutorial walks you through your first UMCP validation from installation to running a custom casepack.

---

## Step 1: Install UMCP (2 minutes)

```bash
# Clone the repository
git clone https://github.com/calebpruett927/UMCP-Metadata-Runnable-Code.git
cd UMCP-Metadata-Runnable-Code

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install UMCP with all features
pip install -e ".[all]"
```

**Verify installation**:
```bash
umcp health
# Should show: Status: HEALTHY
```

---

## Step 2: Run Your First Validation (1 minute)

```bash
# Validate the hello_world casepack
umcp validate casepacks/hello_world
```

**What just happened?**
- UMCP loaded a frozen contract (`UMA.INTSTACK.v1`)
- Validated kernel invariants (ω, F, S, C, τ_R, κ, IC)
- Checked regime classification (Stable/Watch/Collapse)
- Generated a cryptographic receipt

**Output**: You'll see `CONFORMANT` with 0 errors, 0 warnings.

---

## Step 3: Understand the Output (2 minutes)

```bash
# View detailed validation with verbose flag
umcp validate casepacks/hello_world --verbose
```

**Key sections**:
1. **run_status**: `CONFORMANT` (passed) or `NONCONFORMANT` (failed)
2. **validator**: Version, git commit, Python version
3. **targets**: Each casepack's validation result
4. **issues**: Errors, warnings, info messages (if any)

**JSON output**: Add `--out result.json` to save the receipt.

---

## Step 4: Explore Casepacks (2 minutes)

```bash
# List all available casepacks
umcp list casepacks

# Validate a more complex casepack
umcp validate casepacks/gcd_complete
```

**Available casepacks**:
- `hello_world`: Minimal example (good starting point)
- `gcd_complete`: Generative Collapse Dynamics (energy, flux, resonance)
- `kinematics_complete`: Physics examples (projectiles, oscillators)
- `rcft_complete`: Recursive Collapse Field Theory (fractal dimensions)

---

## Step 5: Use the Python API (2 minutes)

Create `my_first_validation.py`:

```python
import umcp
from umcp.frozen_contract import compute_kernel
import numpy as np

# Validate a casepack programmatically
result = umcp.validate("casepacks/hello_world")
print(f"Status: {'✓ PASS' if result else '✗ FAIL'}")
print(f"Errors: {result.error_count}, Warnings: {result.warning_count}")

# Compute kernel invariants directly
coherence = np.array([0.95, 0.90, 0.92])  # Your data
weights = np.array([0.5, 0.3, 0.2])       # Channel weights

kernel = compute_kernel(coherence, weights, tau_R=5.0)
print(f"\nKernel invariants:")
print(f"  Drift (ω): {kernel.omega:.4f}")
print(f"  Fidelity (F): {kernel.F:.4f}")
print(f"  Entropy (S): {kernel.S:.4f}")
print(f"  Integrity (IC): {kernel.IC:.4f}")
```

Run it:
```bash
python my_first_validation.py
```

---

## Step 6: Try the API Server (Optional, 1 minute)

```bash
# Enable dev mode (no authentication required)
export UMCP_DEV_MODE=1

# Start the API server
umcp-api
# Or: uvicorn umcp.api_umcp:app --reload
```

Open browser to `http://localhost:8000/docs` to see interactive API documentation.

**Try it**:
```bash
curl http://localhost:8000/health
curl http://localhost:8000/casepacks
```

---

## Step 7: Try the Dashboard (Optional, 1 minute)

```bash
# Start the Streamlit dashboard
umcp-dashboard
# Or: streamlit run src/umcp/dashboard.py
```

Open browser to `http://localhost:8501` to explore:
- **Overview**: System health and metrics
- **Ledger**: Historical validation log
- **Casepacks**: Browse and validate casepacks
- **Contracts**: View frozen contracts
- **Closures**: Explore closure registry

---

## Next Steps: Create Your Own Casepack

### Minimal Casepack Structure

```
my_casepack/
├── manifest.json       # Metadata and references
├── raw_measurements.csv # Your input data
├── contracts/          # Frozen contract (copy from contracts/)
└── expected/           # Expected outputs
    └── invariants.json
```

**manifest.json** (minimal):
```json
{
  "schema": "schemas/manifest.schema.json",
  "casepack": {
    "id": "my_first_casepack",
    "version": "1.0.0",
    "title": "My First Casepack",
    "description": "Learning UMCP validation",
    "created_utc": "2026-01-31T00:00:00Z",
    "timezone": "UTC",
    "authors": ["Your Name"]
  },
  "refs": {
    "contract": {
      "id": "UMA.INTSTACK.v1",
      "path": "contracts/UMA.INTSTACK.v1.yaml"
    }
  },
  "artifacts": {
    "raw_measurements": {
      "path": "raw_measurements.csv",
      "format": "csv"
    }
  }
}
```

**raw_measurements.csv** (example):
```csv
t,c_1,c_2,c_3,w_1,w_2,w_3
0,0.95,0.90,0.92,0.5,0.3,0.2
1,0.93,0.88,0.90,0.5,0.3,0.2
2,0.96,0.91,0.93,0.5,0.3,0.2
```

**Validate your casepack**:
```bash
umcp validate my_casepack/
```

---

## Key Concepts (Quick Reference)

| Concept | Meaning | Example |
|---------|---------|---------|
| **Contract** | Frozen specification of computation | `UMA.INTSTACK.v1.yaml` |
| **Casepack** | Reproducible validation unit | `casepacks/hello_world/` |
| **Kernel invariants** | Core metrics (ω, F, S, C, τ_R, κ, IC) | Computed from coherence values |
| **Return time (τ_R)** | First-hitting-time to η-neighborhood | `5.0` or `INF_REC` |
| **Regime** | System state classification | Stable, Watch, Collapse |
| **Seam** | Continuity check across transitions | `\|residual\| ≤ 0.005` |
| **Closure** | Pluggable computation extension | `energy_potential.py` |

---

## Troubleshooting

**Q: `umcp: command not found`**  
A: Activate your virtual environment: `source .venv/bin/activate`

**Q: `umcp-ext: command not found`**  
A: Reinstall in editable mode: `pip install -e .`

**Q: API returns 401 Unauthorized**  
A: Set `export UMCP_DEV_MODE=1` for local testing (no auth required)

**Q: Tests fail**  
A: Check Python version (≥3.11 required): `python --version`

**Q: Where are the docs?**  
A: Key docs:
- [README.md](README.md) - Overview and quick start
- [KERNEL_SPECIFICATION.md](KERNEL_SPECIFICATION.md) - Formal math
- [TIER_SYSTEM.md](TIER_SYSTEM.md) - Architecture
- [CASEPACK_REFERENCE.md](CASEPACK_REFERENCE.md) - Casepack format

---

## What You've Learned

✓ Installed UMCP and verified system health  
✓ Ran your first validation  
✓ Explored casepacks and outputs  
✓ Used the Python API  
✓ (Optional) Started API server and dashboard  
✓ Created a minimal casepack structure  

**Next**: Dive into [KERNEL_SPECIFICATION.md](KERNEL_SPECIFICATION.md) to understand the mathematical foundations, or explore [casepacks/gcd_complete/](casepacks/gcd_complete/) for a comprehensive example.

---

## Getting Help

- **GitHub Issues**: [Report bugs or request features](https://github.com/calebpruett927/UMCP-Metadata-Runnable-Code/issues)
- **Documentation**: Browse the 164 `.md` files in the repository
- **Tests**: See `tests/` for 777 examples of API usage
- **Examples**: All casepacks in `casepacks/` are fully documented

**Happy validating! 🎯**

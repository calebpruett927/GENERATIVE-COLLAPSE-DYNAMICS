# UMCP Publication Infrastructure

**Version**: 1.0.0  
**Status**: Protocol Foundation  
**Source**: UMCP Manuscript v1.0.0 §5.19  
**Last Updated**: 2026-01-21

---


## Overview

The **UMCP publication infrastructure** is fully unified:
- All core and extension code is under `src/umcp/`
- All dependencies are managed in a single `pyproject.toml`
- All scripts/utilities are in `scripts/`
- No `requirements.txt` or `setup.py` is needed—use `pip install -e .[production]` for all dependencies

It standardizes how results are packaged, cited, and published. It provides:

1. **Publication rows**: Standardized CSV format for weld results with canonical headers
2. **Case IDs**: Human-legible handles for published CasePacks (e.g., `CPOC-TOY-TRACE-0001`)
3. **make_pubrow.py tool**: Automated publication row generation from receipts
4. **Contract variants**: Handling parameter changes that require new contract versions
5. **Weld ID vs EID distinction**: Separating continuity evidence from artifact identity
6. **ASCII header mapping**: Fixed mapping between CSV headers and typeset symbols

---

## Publication Row Format

### Purpose

**Definition**: A publication row is a single-line CSV record that captures weld identity, integrity metrics, and cryptographic root in a standardized format suitable for:
- Manuscript captions (HUD line)
- Ledger append operations
- Cross-CasePack comparison
- Reproducibility verification

### Canonical Header

```csv
weld_id,manifest_id,kappa,I,delta_kappa,It1_over_It0,tol,residual,seed,sha256
```

### Field Definitions

| Field | Symbol | Definition | Format | Example |
|-------|--------|------------|--------|---------|
| **weld_id** | - | Seam event identifier | String | `W-2025-12-31-PHYS-COHERENCE` |
| **manifest_id** | - | CasePack identifier | String | `manifest:gcd_complete` |
| **kappa** | κ₁ | Log-integrity at t₁ | 6 decimals | `-0.123456` |
| **I** | IC₁ | Integrity composite at t₁ | 6 decimals | `0.884123` |
| **delta_kappa** | Δκ | Ledger term: κ₁ - κ₀ | 12 decimals | `-0.050000000000` |
| **It1_over_It0** | IC₁/IC₀ | Integrity ratio | 12 decimals | `0.951229424500` |
| **tol** | tol_seam | Seam tolerance | 6 decimals | `0.005000` |
| **residual** | s | Budget minus ledger | 12 decimals | `0.002000000000` |
| **seed** | - | Determinism stamp | String | `fixed_2026-01-21` or `NA` |
| **sha256** | H₀ | Root hash of integrity/sha256.txt | 64 hex chars | `9a2f83b8...c57e9` |

### Mathematical Consistency Requirements

**Rule**: Publication rows must satisfy two consistency checks:

1. **Dial consistency**: 
   ```
   |IC₁/IC₀ - exp(Δκ)| < 10⁻⁹
   ```
   The integrity ratio must equal the exponential of the ledger term within numerical tolerance.

2. **Budget closure**: 
   ```
   |s| ≤ tol_seam
   ```
   The seam residual must fall within declared tolerance.

### HUD Line Format

**Definition**: A HUD (Heads-Up Display) line is a single-line citation format for manuscript captions:

```
HUD: weld_id=W-2025-12-31-PHYS-COHERENCE | manifest=manifest:gcd_complete | I=0.884123 | kappa=-0.123456 | tol=0.005000 | residual=0.002000000000 | seed=fixed_2026-01-21 | sha256=9a2f83b8...c57e9
```

**Use case**: Copy this line directly into figure/table captions to provide reproducibility metadata.

---

## make_pubrow.py Tool

### Purpose

Automated publication row generation from SS1m receipts with:
- Auto-detection of receipt files
- Flexible field mapping (multiple naming conventions)
- Derived field computation (if IC provided but κ missing, compute κ = ln(IC))
- Integrity root hash computation
- PASS/FAIL validation

### Location

```
tools/make_pubrow.py
```

*Note: This tool is specified in the manuscript but not yet implemented in the repository. See [Future Work](#future-work) below.*

### Command Stencils

```bash
# Auto-detect receipt (searches weld/, receipts/, receipt/, integrity/)
python tools/make_pubrow.py casepacks/<CASEPACK_DIR>

# Provide receipt explicitly
python tools/make_pubrow.py casepacks/<CASEPACK_DIR> --receipt weld/ss1m.json

# Override manifest_id (rare; usually from manifest.yaml/json)
python tools/make_pubrow.py casepacks/<CASEPACK_DIR> --manifest-id manifest:2026-01-10

# Override weld_id
python tools/make_pubrow.py casepacks/<CASEPACK_DIR> --weld-id W-2026-01-21-CUSTOM

# Override seed/determinism stamp
python tools/make_pubrow.py casepacks/<CASEPACK_DIR> --seed fixed_2026-01-21

# Custom output path
python tools/make_pubrow.py casepacks/<CASEPACK_DIR> --out outputs/custom_pubrow.csv

# Skip PASS/FAIL checks (still writes row)
python tools/make_pubrow.py casepacks/<CASEPACK_DIR> --no-checks
```

### Receipt Auto-Detection

**Rule**: The tool searches for receipts in this order:

```python
for sub in ("weld", "receipts", "receipt", "integrity"):
    candidates = [
        sub / "ss1m.json",
        sub / "ss1m.yaml",
        sub / "ss1m.yml",
        sub / "ss1m.csv",
        sub / "receipt.json",
        sub / "receipt.yaml",
        sub / "receipt.yml",
        sub / "receipt.csv",
    ]
```

First match is used. If no receipt found, error is raised.

### Field Mapping (Flexible Naming)

The tool accepts multiple naming conventions for receipt fields:

| Canonical Field | Accepted Aliases |
|----------------|------------------|
| **weld_id** | `weld_id`, `weldId`, `Weld-ID`, `weld`, `seam_id`, `seamId` |
| **seed** | `seed`, `rng_seed`, `rngSeed`, `determinism`, `stamp` |
| **kappa** | `kappa`, `kappa1`, `kappa_end`, `kappa_close`, `κ`, `κ1` |
| **I** | `I`, `I1`, `IC`, `IC1`, `integrity`, `integrity1` |
| **delta_kappa** | `delta_kappa`, `Δκ`, `delta_kappa_ledger`, `Delta_kappa_ledger` |
| **It1_over_It0** | `It1_over_It0`, `ir`, `IR`, `I_ratio`, `I1_over_I0` |
| **tol** | `tol`, `tol_seam`, `tolSeam`, `tolerance`, `tol_seam_budget` |
| **residual** | `residual`, `s`, `seam_residual`, `budget_minus_ledger`, `seam_s` |

### Derived Field Computation

**Rule**: If a field is missing but can be derived, the tool computes it:

```python
# If Δκ missing but IC₁/IC₀ provided
if delta_kappa is None and It1_over_It0 is not None:
    delta_kappa = math.log(It1_over_It0)

# If IC₁/IC₀ missing but Δκ provided
if It1_over_It0 is None and delta_kappa is not None:
    It1_over_It0 = math.exp(delta_kappa)

# If κ missing but IC provided
if kappa is None and I is not None:
    kappa = math.log(I)

# If IC missing but κ provided
if I is None and kappa is not None:
    I = math.exp(kappa)
```

### Validation Checks

**Rule**: Unless `--no-checks` is specified, the tool runs:

1. **Budget closure check**:
   ```python
   if abs(residual) > tol:
       FAIL: |residual| > tol_seam
   ```

2. **Dial consistency check**:
   ```python
   dial = math.exp(delta_kappa)
   if abs(It1_over_It0 - dial) >= 1e-9:
       FAIL: |IC₁/IC₀ - exp(Δκ)| >= 10⁻⁹
   ```

### Output Format

After execution, the CasePack will contain:

```
casepacks/<CASEPACK_DIR>/
└── outputs/
    └── publication_row.csv        # Append-only; canonical header if newly created
```

Terminal output includes:

```
OK: PASS basic checks (budget closure, dial consistency)

HUD: weld_id=W-2025-12-31-PHYS-COHERENCE | manifest=manifest:gcd_complete | I=0.884123 | kappa=-0.123456 | tol=0.005000 | residual=0.002000000000 | seed=fixed_2026-01-21 | sha256=9a2f83b8...c57e9

Wrote: /workspaces/UMCP-Metadata-Runnable-Code/casepacks/gcd_complete/outputs/publication_row.csv
Receipt: /workspaces/UMCP-Metadata-Runnable-Code/casepacks/gcd_complete/weld/ss1m.json
Root sha256 = sha256(integrity/sha256.txt) = 9a2f83b8...c57e9
```

---

## Case ID Conventions

### Purpose

**Definition**: A **Case ID** is a human-legible handle for a published CasePack. It serves as:
- Manuscript citation anchor
- Directory name convention
- Cross-reference key in multi-CasePack studies

### Format Convention

```
CPOC-<DOMAIN>-<TYPE>-<NNNN>
```

Where:
- **CPOC**: CasePack Of Consequence (protocol prefix)
- **DOMAIN**: Subject area (e.g., TOY, TS, NON-EUCLIDEAN, OVERLAY)
- **TYPE**: Analysis type (e.g., TRACE, WELD, POSTERIOR)
- **NNNN**: Sequential 4-digit number (zero-padded)

### Valid Examples

From manuscript §5.19.7:

```
CPOC-TOY-TRACE-0001        # Toy example with trace analysis
CPOC-TS-WELD-0002          # Time series with weld accounting
CPOC-NON-EUCLIDEAN-0003    # Non-Euclidean geometry application
CPOC-OVERLAY-POSTERIOR-0004 # Tier-2 overlay with posterior inference
```

### Usage in File Structure

```
casepacks/
├── CPOC-TOY-TRACE-0001/
│   ├── manifest.yaml             # manifest_id: CPOC-TOY-TRACE-0001
│   ├── contract.yaml
│   ├── trace.csv
│   └── outputs/
│       └── publication_row.csv
├── CPOC-TS-WELD-0002/
│   ├── manifest.yaml             # manifest_id: CPOC-TS-WELD-0002
│   ├── welds.csv
│   └── weld/
│       └── ss1m.json
```

### Current Repository Case IDs

The repository currently uses informal naming:

```
casepacks/
├── hello_world/              # Informal: introductory example
├── gcd_complete/             # Informal: complete GCD demonstration
├── rcft_complete/            # Informal: complete RCFT demonstration
└── UMCP-REF-E2E-0001/        # Formal: reference end-to-end test
```

**Future work**: Migrate to CPOC-* convention for published examples.

---

## Contract Variants

### Purpose

**Definition**: A **contract variant** handles parameter changes that affect computation but maintain compatibility with the base contract structure.

**Rule**: If a parameter change affects kernel outputs (e.g., H_rec changes τ_R computation), it requires a contract variant.

### Variant ID Format

```
<BASE_CONTRACT_ID>+<AUTHOR>.<CHANGE_DESCRIPTION>.<VERSION>
```

**Example** (from manuscript §5.19.7):

```
Base contract:     UMA.INTSTACK.v1
Change:            H_rec from 100 to 500 (affects return typing)
Variant ID:        UMA.INTSTACK.v1+PAULUS.RETURNHORIZON500.v1
Pinning:           Contract hash recorded in manifest.yaml
```

### When to Create a Variant

| Change | Requires Variant? | Reasoning |
|--------|------------------|-----------|
| **H_rec: 100 → 500** | ✅ YES | Changes return horizon; affects τ_R computation |
| **tol_seam: 0.005 → 0.01** | ✅ YES | Changes weld PASS/FAIL outcomes |
| **ε: 1e-8 → 1e-10** | ✅ YES | Changes log-safety clipping; affects κ, IC |
| **Documentation typo fix** | ❌ NO | Does not affect computation |
| **Adding comments** | ❌ NO | Does not affect frozen parameters |

### Variant Declaration

**In contract.yaml**:

```yaml
schema: UMA.CONTRACT.v1
id: UMA.INTSTACK.v1+PAULUS.RETURNHORIZON500.v1
title: "Universal Measurement Contract (IntStack) - Extended Return Horizon Variant"
base_contract: UMA.INTSTACK.v1
variant_description: "Return horizon extended from 100 to 500 samples"
author: "Clement Paulus"
created: "2026-01-21T10:00:00-06:00"

frozen_params:
  H_rec: 500                        # Changed from base (was 100)
  eta: 0.001                        # Inherited from base
  epsilon: 1.0e-8                   # Inherited from base
  lambda: 0.2                       # Inherited from base
  tol_seam: 0.005                   # Inherited from base
  tol_id: 1.0e-9                    # Inherited from base
  # ... other params
```

**In manifest.yaml**:

```yaml
manifest_id: CPOC-EXTENDED-HORIZON-0005
contract:
  id: UMA.INTSTACK.v1+PAULUS.RETURNHORIZON500.v1
  hash: "a7f3d9e8...b2c1"           # SHA-256 of contract.yaml
  base_contract: UMA.INTSTACK.v1
```

---

## Weld ID vs EID Distinction

### Purpose

**Critical distinction**: Weld identity and artifact identity are **separate concerns** and must not be conflated.

| Concept | Purpose | Evidence Location | Example |
|---------|---------|------------------|---------|
| **Weld ID** | Seam event identity | `outputs/welds.csv`, `weld/ss1m.json` | `W-2025-12-31-PHYS-COHERENCE` |
| **EID** | Artifact structural fingerprint | Report metadata, document properties | `P=10,Eq=6,Fig=2,Tab=1` |

### Weld ID

**Definition**: A weld ID identifies a **continuity event** (seam transition from t₀ to t₁).

**Format**: `W-<DATE>-<DESCRIPTION>`

**Location**: 
- `outputs/welds.csv` (weld_id column)
- `weld/ss1m.json` (weld_id field)
- `outputs/publication_row.csv` (weld_id column)

**Purpose**: 
- Binds seam audit to frozen contract + closures
- Cites continuity claim in manuscript
- Enables ledger lookup for Δκ, s, PASS/FAIL

**Example**:

```csv
weld_id,t0,t1,dk_ledger,dk_budget,residual_s,pass
W-2025-12-31-PHYS-COHERENCE,0,1000,-0.05,-0.048,0.002,PASS
```

### EID (Edition Identity)

**Definition**: An EID is an **artifact structural fingerprint** used for document edition tracking.

**Format**: `P=<pages>,Eq=<equations>,Fig=<figures>,Tab=<tables>,List=<listings>,Box=<boxes>,Ref=<references>`

**Location**: 
- Report metadata (PDF properties, document front matter)
- Archive manifests (Zenodo, arXiv)

**Purpose**: 
- Detect structural changes between document editions
- NOT used for continuity evidence
- NOT used for weld accounting

**Example** (manuscript §5.19.7):

```
Report metadata:
  eid_counts: {P=10, Eq=6, Fig=2, Tab=1, List=0, Box=0, Ref=12}
  report_file_hash: "d3f4a9c7...e1b2"
```

### Prohibition

**Warning**: The EID is **not** used as evidence of continuity; continuity is evidenced by weld rows **only**.

**Anti-pattern** (nonconformant):

```
❌ "Continuity is maintained because the EID is unchanged."
```

**Correct pattern** (conformant):

```
✅ "Continuity is maintained as evidenced by weld W-2025-12-31-PHYS-COHERENCE
   with residual s=0.002 ≤ tol_seam=0.005 (PASS)."
```

---

## ASCII Header Mapping

### Purpose

**Definition**: CSV outputs use **ASCII-only headers** for maximum compatibility, with a **fixed mapping** to typeset symbols in manuscripts.

**Rule**: The mapping is frozen and must not be changed without a new protocol version.

### Tier-1 Invariants Mapping

From manuscript §5.19.7:

```csv
t,omega,F,S,C,tau_R,kappa,IC
```

**Fixed mapping**:

| ASCII Header | Typeset Symbol | Definition |
|--------------|---------------|------------|
| **t** | t | Time index |
| **omega** | ω | Drift |
| **F** | F | Fidelity |
| **S** | S | Entropy |
| **C** | C | Curvature |
| **tau_R** | τ_R | Return time |
| **kappa** | κ | Log-integrity |
| **IC** | IC | Integrity composite |

### Example File

**outputs/invariants.csv**:

```csv
t,omega,F,S,C,tau_R,kappa,IC,regime
1,0.08,0.92,0.11,0.10,12,-0.15,0.86,Stable
2,0.12,0.88,0.14,0.12,18,-0.22,0.80,Stable
3,0.65,0.35,0.42,0.28,INF_REC,-2.10,0.12,Collapse
```

**In manuscript**:

> At time $t=1$, we observe $\omega=0.08$, $F=0.92$, $S=0.11$, and $\tau_R=12$, 
> with regime label Stable.

### Why ASCII-Only?

1. **Compatibility**: Works with all CSV parsers (no Unicode issues)
2. **Command-line friendly**: Easy to grep, awk, sed
3. **Deterministic sorting**: No locale-dependent collation issues
4. **Git-friendly**: Clean diffs without encoding artifacts

### Symbol Index Cross-Reference

See [SYMBOL_INDEX.md](SYMBOL_INDEX.md) for the authoritative mapping table including:
- Unicode symbols (for manuscripts)
- ASCII equivalents (for CSV files)
- Approved alternatives (to avoid namespace collision)

---

## Implementation Status

### Current Support (v1.4.0)

- ✅ **Publication row CSV structure**: Defined in repository
  - Canonical header documented
  - Field definitions clear
  - Mathematical consistency requirements stated

- ✅ **ASCII header mapping**: Implemented
  - All invariants.csv files use ASCII headers
  - Fixed mapping to typeset symbols documented
  - See: [outputs/invariants.csv](casepacks/gcd_complete/outputs/invariants.csv)

- ⚠️ **Case ID convention**: Partial
  - One formal Case ID: UMCP-REF-E2E-0001
  - Other CasePacks use informal naming
  - Future: Migrate to CPOC-* convention

- ❌ **make_pubrow.py tool**: Not yet implemented
  - Specified in manuscript §5.19
  - Implementation planned

- ⚠️ **Contract variants**: Partial
  - Base contracts defined (UMA.INTSTACK.v1, GCD.INTSTACK.v1, RCFT.INTSTACK.v1)
  - Variant syntax not yet used
  - Future: Support +AUTHOR.CHANGE.VERSION format

- ⚠️ **Weld ID vs EID distinction**: Documented but not enforced
  - Weld IDs present in some CasePacks
  - EID tracking not yet implemented
  - Future: Add validation to prevent EID misuse as continuity evidence

### Future Work

- [ ] Implement `tools/make_pubrow.py` with full auto-detection
- [ ] Add publication row validation to `umcp validate` CLI
- [ ] Migrate existing CasePacks to CPOC-* naming convention
- [ ] Implement contract variant system (+AUTHOR.CHANGE.VERSION)
- [ ] Add EID tracking to report generation pipeline
- [ ] Create tests for publication row consistency checks
- [ ] Add HUD line generation to receipt tools

---

## Minimal Examples

### Example 1: Complete Publication Flow

```bash
# 1. Generate receipt (from weld accounting)
umcp validate casepacks/CPOC-TOY-TRACE-0001 --weld

# 2. Generate publication row
python tools/make_pubrow.py casepacks/CPOC-TOY-TRACE-0001

# Output:
# OK: PASS basic checks (budget closure, dial consistency)
# 
# HUD: weld_id=W-2026-01-21-TOY | manifest=CPOC-TOY-TRACE-0001 | I=0.884123 | kappa=-0.123456 | tol=0.005000 | residual=0.002000000000 | seed=fixed | sha256=9a2f83b8...c57e9
# 
# Wrote: casepacks/CPOC-TOY-TRACE-0001/outputs/publication_row.csv

# 3. Use HUD line in manuscript caption
```

**In manuscript**:

> **Figure 1**: Phase space trajectory for CPOC-TOY-TRACE-0001. 
> HUD: weld_id=W-2026-01-21-TOY | manifest=CPOC-TOY-TRACE-0001 | I=0.884123 | kappa=-0.123456 | tol=0.005000 | residual=0.002000000000 | seed=fixed | sha256=9a2f83b8...c57e9

### Example 2: Contract Variant for Extended Horizon

```yaml
# contracts/UMA.INTSTACK.v1+PAULUS.RETURNHORIZON500.v1.yaml
schema: UMA.CONTRACT.v1
id: UMA.INTSTACK.v1+PAULUS.RETURNHORIZON500.v1
title: "UMA IntStack - Extended Return Horizon (H_rec=500)"
base_contract: UMA.INTSTACK.v1
variant_description: "Return horizon extended to 500 samples for long-term stability analysis"

frozen_params:
  H_rec: 500              # Changed from 100
  eta: 0.001              # Inherited
  epsilon: 1.0e-8         # Inherited
  # ... rest inherited
```

### Example 3: Weld ID vs EID Separation

**Conformant usage**:

```yaml
# In weld/ss1m.json
{
  "weld_id": "W-2026-01-21-STABILITY-ANALYSIS",
  "manifest_id": "CPOC-TS-WELD-0002",
  "delta_kappa": -0.05,
  "residual": 0.002,
  "pass": true
}

# In report metadata (separate)
{
  "eid": "P=25,Eq=12,Fig=5,Tab=3,List=2,Box=1,Ref=48",
  "report_hash": "d3f4a9c7...e1b2",
  "note": "EID for edition tracking only; NOT continuity evidence"
}
```

**Manuscript citation**:

> Continuity is maintained across the transition (weld W-2026-01-21-STABILITY-ANALYSIS, 
> residual s=0.002 ≤ tol_seam=0.005). The report edition is P=25,Eq=12,Fig=5,Tab=3,List=2,Box=1,Ref=48 
> for archival reference.

---

**See Also**:
- [TIER_SYSTEM.md](TIER_SYSTEM.md) - Tier-1.5 weld accounting requirements
- [UHMP.md](UHMP.md) - Manifest root hash computation (sha256 field)
- [SYMBOL_INDEX.md](SYMBOL_INDEX.md) - ASCII to typeset symbol mapping
- [contracts/](contracts/) - Base contracts and variant examples
- [casepacks/](casepacks/) - Example CasePacks with publication rows

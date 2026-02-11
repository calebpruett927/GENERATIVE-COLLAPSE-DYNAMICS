# UMCP CasePack Reference Structure

**Manuscript Reference**: UMCP Manuscript v1.0.0, §5 (CasePack Examples)  
**Status**: Canonical reference format  
**Purpose**: Document the complete structure, file formats, and validation rules for UMCP CasePacks

---


## Overview

A **CasePack** is the complete publishable unit in UMCP. The entire system is unified:
- All core and extension code is under `src/umcp/`
- All dependencies are managed in a single `pyproject.toml`
- All scripts/utilities are in `scripts/`
- No `requirements.txt` or `setup.py` is needed—use `pip install -e .[production]` for all dependencies

It contains:

1. **Interface files** (contract, closures, embedding, observables, return, weights)
2. **Data** (trace.csv, measurements)
3. **Receipts** (ss1m_receipt.json, seam_receipt.json)
4. **Outputs** (kernel.json, regime.json, invariants.csv, welds.csv, publication_row.csv)
5. **Integrity ledger** (integrity/sha256.txt)
6. **Manifests** (manifest.yaml, manifest.json)
7. **Validators** (rules.yml)

**Key principle**: A CasePack is **self-contained** and **cryptographically verifiable**. All files are hashed in the integrity ledger, and receipts provide audit trails for kernel evaluation and continuity claims.

---

## 1. Manifest Files

### manifest.yaml

**Purpose**: Human-readable manifest in YAML format.

```yaml
version: Manifest-1.0
case_id: UMCP-REF-CASEPACK-0001
tz: America/Chicago
generated_at: '2026-01-12T00:00:00'
manifest_id: manifest:UMCP-REF-CASEPACK-0001
weld_id: W-2025-12-31-PHYS-COHERENCE
publication_doi: 10.5281/zenodo.18226878
notes:
- "Hashes are authoritative in integrity/sha256.txt;
  publication_row binds the ledger root (excluded from ledger)."
files:
- path: contract.yaml
- path: closures.yaml
- path: embedding.yaml
- path: observables.yaml
- path: return.yaml
- path: weights.csv
- path: trace.csv
- path: receipts/ss1m_receipt.json
- path: receipts/seam_receipt.json
- path: outputs/kernel.json
- path: outputs/regime.json
- path: outputs/invariants.csv
- path: outputs/regimes.csv
- path: outputs/welds.csv
- path: outputs/report.txt
- path: outputs/publication_row.csv
- path: integrity/sha256.txt
```

**Key fields**:
- `version`: Manifest schema version (Manifest-1.0)
- `case_id`: Unique CasePack identifier (CPOC-DOMAIN-TYPE-NNNN format)
- `tz`: Timezone for all timestamps
- `generated_at`: ISO 8601 timestamp
- `manifest_id`: Manifest root hash (UHMP identifier)
- `weld_id`: Weld identifier if continuity is claimed (W-YYYY-MM-DD-DOMAIN-TOPIC)
- `publication_doi`: DOI if published (e.g., Zenodo)
- `files`: Complete list of files included in the CasePack

---

### manifest.json

**Purpose**: Machine-readable manifest in JSON format (identical content to YAML).

```json
{
  "schema": {
    "name": "UMCP-Manifest",
    "version": "Manifest-1.0"
  },
  "case_id": "UMCP-REF-CASEPACK-0001",
  "tz": "America/Chicago",
  "generated_at": "2026-01-12T00:00:00",
  "manifest_id": "manifest:UMCP-REF-CASEPACK-0001",
  "weld_id": "W-2025-12-31-PHYS-COHERENCE",
  "publication_doi": "10.5281/zenodo.18226878",
  "notes": [
    "Hashes are authoritative in integrity/sha256.txt; publication_row binds the ledger root (excluded from ledger)."
  ],
  "files": [
    { "path": "contract.yaml" },
    { "path": "closures.yaml" },
    { "path": "embedding.yaml" },
    { "path": "observables.yaml" },
    { "path": "return.yaml" },
    { "path": "weights.csv" },
    { "path": "trace.csv" },
    { "path": "receipts/ss1m_receipt.json" },
    { "path": "receipts/seam_receipt.json" },
    { "path": "outputs/kernel.json" },
    { "path": "outputs/regime.json" },
    { "path": "outputs/invariants.csv" },
    { "path": "outputs/regimes.csv" },
    { "path": "outputs/welds.csv" },
    { "path": "outputs/report.txt" },
    { "path": "outputs/publication_row.csv" },
    { "path": "integrity/sha256.txt" }
  ]
}
```

**Rule**: Both YAML and JSON manifests MUST contain identical information. The JSON format is preferred for machine parsing.

---

## 2. Receipt Files

### receipts/ss1m_receipt.json (Minimum Audit Receipt)

**Purpose**: The **SS1m** (Snapshot-Seam-1-minimal) receipt is **required for every run**. It provides a human-checkable kernel/regime snapshot plus identity pins.

**Schema**: UMCP-SS1m v1.0

```json
{
  "schema": { "name": "UMCP-SS1m", "version": "SS1m-1.0" },
  "case_id": "UMCP-REF-CASEPACK-0001",
  "tz": "America/Chicago",
  "emitted_at": "2026-01-12T00:00:00",
  "publication_doi": "10.5281/zenodo.18226878",
  "manifest_id": "manifest:UMCP-REF-CASEPACK-0001",
  "purpose": "Minimum audit receipt emitted for every run; human-checkable kernel/regime snapshot plus identity pins.",
  
  "interface_pins": {
    "contract_id": "UMA.INTSTACK.v1",
    "contract_path": "contract.yaml",
    "embedding_id": "UMCP-EMBED-REF-0001",
    "embedding_path": "embedding.yaml",
    "closures_id": "UMCP-CLOS-REF-0001",
    "closures_path": "closures.yaml",
    "observables_path": "observables.yaml",
    "return_path": "return.yaml",
    "weights_path": "weights.csv"
  },
  
  "row_ref": {
    "t": 4,
    "F": 0.7506249999999999,
    "omega": 0.24937500000000012,
    "S": 0.4986331848245621,
    "C": 0.3062755091416877,
    "tau_R": "INF_REC",
    "kappa": -0.31022507722327575,
    "IC": 0.7332818925966906,
    "regime": "WATCH"
  },
  
  "seam": {
    "seam_claim": false,
    "weld_id": "NA",
    "notes": "Continuity claims require seam_claim=true and a PASS seam receipt."
  },
  
  "tolerances": { "tol_seam": 0.005, "tol_id": 1e-09 },
  
  "integrity": {
    "ledger_path": "integrity/sha256.txt",
    "publication_row_path": "outputs/publication_row.csv",
    "note": "Hashes are authoritative in integrity/sha256.txt; publication_row binds the ledger root (excluded from ledger)."
  },
  
  "eid": {
    "eid_counts": null,
    "note": "EID (edition fingerprint) is distinct from weld_id; this CasePack does not assert an EID."
  }
}
```

**Key sections**:

- **interface_pins**: Identity pins for all interface files (contract, closures, embedding, etc.)
- **row_ref**: Reference row snapshot showing kernel outputs (F, ω, S, C, κ, IC, τ_R) and regime
- **seam**: Continuity claim status (seam_claim=true requires PASS seam receipt)
- **tolerances**: Frozen tolerance values (tol_seam for residual, tol_id for numerical identity)
- **integrity**: Paths to integrity ledger and publication row
- **eid**: Edition fingerprint (distinct from weld_id; see PUBLICATION_INFRASTRUCTURE.md)

**Rule**: The ss1m_receipt.json is **mandatory** for all runs. It provides the minimum audit trail.

---

### receipts/seam_receipt.json (Seam/Continuity Receipt)

**Purpose**: The **Seam** receipt is **required only when making continuity claims** (weld assertions). It contains full seam accounting: ledger change, budget model, residual, and PASS/FAIL status.

**Schema**: UMCP-Seam v1.0

```json
{
  "schema": { "name": "UMCP-Seam", "version": "Seam-1.0" },
  "case_id": "UMCP-REF-CASEPACK-0001",
  "tz": "America/Chicago",
  "emitted_at": "2026-01-12T00:00:00",
  "publication_doi": "10.5281/zenodo.18226878",
  "manifest_id": "manifest:UMCP-REF-CASEPACK-0001",
  "purpose": "Continuity-only seam receipt. Required when making continuity claims (weld assertions).",
  "weld_id": "W-2025-12-31-PHYS-COHERENCE",
  
  "inputs": {
    "kappa0": -0.3207540029165972,
    "kappa1": -0.31022507722327575,
    "IC0": 0.7256017249452535,
    "IC1": 0.7332818925966906,
    "tau_R": "INF_REC"
  },
  
  "ledger": {
    "delta_kappa_ledger": 0.010528925693321478,
    "ir": 1.0105845498810198
  },
  
  "closures_used": {
    "Gamma": {
      "form": "ω^p/(1-ω+ε)",
      "p": 3,
      "epsilon": 1e-08,
      "omega_at_t1": 0.24937500000000012
    },
    "D_C": { 
      "form": "α·C", 
      "alpha": 1.0, 
      "C_at_t1": 0.3062755091416877 
    },
    "R": { 
      "form": "λ", 
      "lambda": 0.2 
    },
    "typed_rule": "τ_R=INF_REC → R·τ_R=0"
  },
  
  "budget": {
    "R_tau_R": 0.0,
    "D_omega": 0.020660256476944992,
    "D_C": 0.3062755091416877,
    "delta_kappa_budget": -0.3269357656186327
  },
  
  "residual": { 
    "s": -0.33746469131195417, 
    "tol_seam": 0.005, 
    "tol_id": 1e-09 
  },
  
  "status": {
    "pass": false,
    "reason": "FAIL because τ_R=INF_REC (no return → no seam credit)."
  },
  
  "notes": [
    "This seam receipt is included as a reference continuity audit artifact. It does not constitute a continuity claim for this CasePack."
  ]
}
```

**Key sections**:

- **inputs**: κ(t₀), κ(t₁), IC(t₀), IC(t₁), τ_R(t₁)
- **ledger**: Δκ_ledger and i_r (ratio form)
- **closures_used**: Complete specification of all closures (Γ, D_C, R) with frozen parameters
- **budget**: Budget model terms (R·τ_R, D_ω, D_C, Δκ_budget)
- **residual**: Seam residual s = Δκ_budget - Δκ_ledger, with tolerances
- **status**: PASS/FAIL with explicit reason

**Rule**: Seam receipt is **required only for continuity claims**. A PASS status requires:
1. τ_R(t₁) is finite (not INF_REC, not UNIDENTIFIABLE)
2. |s| ≤ tol_seam
3. Algebraic identity check: |i_r - exp(Δκ_ledger)| ≤ tol_id

**Typed censoring**: If τ_R = INF_REC, then R·τ_R = 0 (no return implies no seam credit), and status.pass MUST be false.

---

## 3. Integrity Ledger

### integrity/sha256.txt

**Purpose**: Per-file SHA-256 hash ledger. This is the **authoritative integrity record** for all files in the CasePack.

**Format**: `<sha256_hash>  <relative_path>`

```text
# integrity/sha256.txt
# Per-file SHA-256 ledger for UMCP-REF-CASEPACK-0001.
# Format: <sha256> <relative_path>
# Note: this ledger excludes itself and outputs/publication_row.csv (self-reference avoidance).

9c48e67bf10a8c606015ef533960f4e259a39406ee6a98d942d377588f26c046  README.md
cf593b3bc6248868715f80247167c944c4705a0fb7f5df39db3f94fec494986c  casepack.yaml
651487fd908f4b7e3e5a6fee54ff8344b212631d9de4dc34243410a7c9f3bb55  closures.yaml
6d9da4ab327d34bb886c49bec15fad5ee91d9ba84ca7739bb66a32cec207c394  contract.yaml
8444580e6e9c37e5918ec023951ad98ee27b63f425562b62c0a8539b55769976  derived/trace.csv
0b846c5d9b516607044df6557f0c19e941a75f82e13279d8300fd0b6920c36f1  embedding.yaml
a136c7bc4f53d45ee8f68b0ad5df3d60f8507a1eb66fa1f0ba20ad216d241895  expected/expected_outcome.json
2a0c533067eb0742df7ecf25d9b70b9500fc5bfba3798f9a08ebc0eb2c88dd33  inputs/provenance/PROVENANCE.md
e7336f7255044d9efd88fdeb14e52b787d708576b7878ffd7201c940dd97816e  inputs/provenance/acquisition.yaml
06b4a02b9c9127738916f0863911dc0a60b650998625ebffcc1e9ae03471d588  inputs/provenance/raw_sha256.txt
0b0cea7c29ef15f225a3bee1a0be2df749edfab104ac71ab4ef4ba87b1bad8aa  inputs/raw/README.md
c3932eeaaabee4a708ef19984b5e0c539b5db8eb7406e8fa170676d9428f2755  inputs/raw/measurements.csv
c79adcdaeccf0b217937989683c185767cf364b8be538e90cf3467d84ec9434b  manifest.json
d13c005111873051929b8aaccc97e0644b54e64cf39f570376678b48edd6d1eb  manifest.yaml
f155e734c2a3bc431db6695b51ded9be6e37e1ca49bb7f2470b57a7af2b375fd  observables.yaml
... (remaining lines) ...
```

**Rules**:

1. **Self-reference avoidance**: The ledger excludes itself and `outputs/publication_row.csv` (which contains the ledger root hash)
2. **Relative paths**: All paths are relative to the CasePack root
3. **Sorted order**: Paths are typically sorted for deterministic output
4. **Comments allowed**: Lines starting with `#` are comments
5. **Format**: Two-space separator between hash and path (standard `sha256sum` format)

**Verification**: Use `sha256sum -c integrity/sha256.txt` to verify all files.

---

## 4. Validation Rules

### validators/rules.yml

**Purpose**: Human-readable validation checklist. The **reference Python validator** lives at `validator/validate.py`, but this file provides a concise summary of key checks.

```yaml
version: 1
case_id: UMCP-REF-CASEPACK-0001

required_files:
- contract.yaml
- closures.yaml
- return.yaml
- weights.csv
- receipts/ss1m_receipt.json
- manifest.yaml
- manifest.json
- integrity/sha256.txt

checks:
- id: contract_frozen
  assert: contract.yaml:contract_id == "UMA.INTSTACK.v1"

- id: weights_sum_to_1
  assert: sum_csv(weights.csv) == 1.0

- id: ss1m_min_fields
  assert:
  - receipts/ss1m_receipt.json:schema.name == "UMCP-SS1m"
  - receipts/ss1m_receipt.json:row_ref.F is not null
  - receipts/ss1m_receipt.json:row_ref.omega is not null
  - receipts/ss1m_receipt.json:row_ref.S is not null
  - receipts/ss1m_receipt.json:row_ref.C is not null
  - receipts/ss1m_receipt.json:row_ref.tau_R is not null
  - receipts/ss1m_receipt.json:row_ref.kappa is not null
  - receipts/ss1m_receipt.json:row_ref.IC is not null

- id: typed_boundary_rule
  assert: return.yaml:typed_boundaries.rule == "τ_R=INF_REC → R·τ_R=0"

- id: seam_receipt_optional_but_strict
  when_present: receipts/seam_receipt.json
  assert:
  - receipts/seam_receipt.json:schema.name == "UMCP-Seam"
  - receipts/seam_receipt.json:weld_id is not null
  - receipts/seam_receipt.json:ledger.delta_kappa_ledger is not null
  - receipts/seam_receipt.json:ledger.ir is not null
  - receipts/seam_receipt.json:residual.s is not null
  - abs(receipts/seam_receipt.json:ledger.ir - exp(receipts/seam_receipt.json:ledger.delta_kappa_ledger)) <= receipts/seam_receipt.json:residual.tol_id
  - (receipts/seam_receipt.json:inputs.tau_R == "INF_REC") implies (receipts/seam_receipt.json:status.pass == false)

notes:
- This rules.yml is a human-readable validator checklist; the reference Python validator lives at validator/validate.py.
```

**Key checks**:

1. **contract_frozen**: Contract ID is frozen (UMA.INTSTACK.v1)
2. **weights_sum_to_1**: Weights normalize to 1.0
3. **ss1m_min_fields**: SS1m receipt contains all required kernel outputs
4. **typed_boundary_rule**: Typed censoring rule is declared
5. **seam_receipt_optional_but_strict**: If seam receipt is present, it must satisfy:
   - Schema is UMCP-Seam
   - Weld ID is not null
   - Ledger and residual fields are present
   - Algebraic identity: |i_r - exp(Δκ_ledger)| ≤ tol_id
   - Typed censoring: τ_R=INF_REC implies status.pass=false

---

## 5. Complete CasePack Structure

### Minimal CasePack (Required Files)

```
UMCP-REF-CASEPACK-0001/
├── contract.yaml                    # Frozen contract (Tier-0 interface)
├── closures.yaml                    # Closure registry (Tier-2)
├── embedding.yaml                   # Embedding specification
├── observables.yaml                 # Observable definitions
├── return.yaml                      # Return machinery specification
├── weights.csv                      # Coordinate weights (Σw_i = 1)
├── manifest.yaml                    # Human-readable manifest
├── manifest.json                    # Machine-readable manifest
├── receipts/
│   └── ss1m_receipt.json           # REQUIRED: Minimum audit receipt
├── integrity/
│   └── sha256.txt                  # REQUIRED: Integrity ledger
└── validators/
    └── rules.yml                    # Optional: Human-readable validation checklist
```

### Full CasePack (with Outputs and Seam)

```
UMCP-REF-CASEPACK-0001/
├── contract.yaml
├── closures.yaml
├── embedding.yaml
├── observables.yaml
├── return.yaml
├── weights.csv
├── trace.csv                        # Time series data
├── manifest.yaml
├── manifest.json
├── receipts/
│   ├── ss1m_receipt.json           # Minimum audit receipt
│   └── seam_receipt.json           # Seam/continuity receipt (required for continuity claims)
├── outputs/
│   ├── kernel.json                 # Full kernel outputs
│   ├── regime.json                 # Regime classification
│   ├── invariants.csv              # Time series of invariants
│   ├── regimes.csv                 # Time series of regimes
│   ├── welds.csv                   # Weld accounting
│   ├── report.txt                  # Human-readable summary
│   └── publication_row.csv         # Publication row (binds ledger root)
├── integrity/
│   └── sha256.txt                  # Integrity ledger
└── validators/
    └── rules.yml                    # Validation checklist
```

---

## 6. Receipt Schemas and Versioning

### SS1m Schema (Snapshot-Seam-1-minimal)

- **Name**: UMCP-SS1m
- **Version**: SS1m-1.0
- **Purpose**: Minimum audit receipt (required for every run)
- **Key fields**: interface_pins, row_ref, seam, tolerances, integrity, eid

### Seam Schema

- **Name**: UMCP-Seam
- **Version**: Seam-1.0
- **Purpose**: Continuity-only receipt (required for weld claims)
- **Key fields**: inputs, ledger, closures_used, budget, residual, status

**Versioning rule**: Schema versions are declared in the `schema` field of each receipt. Breaking changes require a new schema version.

---

## 7. Identity Conventions

### Manifest ID (UHMP)

Format: `manifest:<case_id>`

Example: `manifest:UMCP-REF-CASEPACK-0001`

The manifest ID is the **UHMP identity** for the CasePack. It is computed from the manifest root hash after the 5-phase UHMP mint (see [UHMP.md](UHMP.md)).

### Weld ID

Format: `W-YYYY-MM-DD-DOMAIN-TOPIC`

Example: `W-2025-12-31-PHYS-COHERENCE`

The weld ID identifies a **continuity claim**. It is distinct from the Edition ID (EID), which is an artifact fingerprint. See [PUBLICATION_INFRASTRUCTURE.md](PUBLICATION_INFRASTRUCTURE.md) for details.

### Case ID

Format: `CPOC-DOMAIN-TYPE-NNNN` (CasePack Of Continuity)

Example: `UMCP-REF-CASEPACK-0001`

The case ID is a human-readable identifier for the CasePack. See [PUBLICATION_INFRASTRUCTURE.md](PUBLICATION_INFRASTRUCTURE.md) for Case ID conventions.

---

## 8. Relationship to Other Protocol Documents

This reference structure is governed by:

- **[TIER_SYSTEM.md](../TIER_SYSTEM.md)**: Tier-0 (protocol), Tier-1 (kernel), Tier-2 (expansion)
- **[KERNEL_SPECIFICATION.md](../KERNEL_SPECIFICATION.md)**: Formal definitions for kernel outputs (F, ω, S, C, κ, IC, τ_R)
- **[UHMP.md](UHMP.md)**: 5-phase mint for manifest identity
- **[PUBLICATION_INFRASTRUCTURE.md](PUBLICATION_INFRASTRUCTURE.md)**: Publication row format, Case IDs, Weld ID vs EID
- **[FACE_POLICY.md](../FACE_POLICY.md)**: Boundary governance for clipping and OOR handling

---

## 9. Implementation Notes

### Creating a CasePack

1. **Freeze interface files** (contract, closures, embedding, observables, return, weights)
2. **Run kernel evaluation** to generate outputs
3. **Emit ss1m_receipt.json** (mandatory)
4. **If making continuity claim**: Emit seam_receipt.json with PASS/FAIL status
5. **Compute integrity ledger**: `find . -type f | grep -v integrity/sha256.txt | grep -v outputs/publication_row.csv | xargs sha256sum > integrity/sha256.txt`
6. **Generate manifests**: Create manifest.yaml and manifest.json
7. **Optional**: Create publication_row.csv (binds ledger root)
8. **Optional**: Create validators/rules.yml for human-readable checks

### Validating a CasePack

1. **Verify integrity**: `sha256sum -c integrity/sha256.txt`
2. **Check required files**: Ensure contract.yaml, closures.yaml, return.yaml, weights.csv, receipts/ss1m_receipt.json, manifest.yaml, manifest.json, integrity/sha256.txt exist
3. **Validate ss1m_receipt**: Check schema, interface_pins, row_ref fields
4. **If seam receipt present**: Validate algebraic identity (|i_r - exp(Δκ_ledger)| ≤ tol_id) and typed censoring rule
5. **Run validator**: `python validator/validate.py` (reference implementation)

### Common Pitfalls

- **Missing ss1m_receipt.json**: Every run MUST emit an ss1m receipt
- **Self-reference in integrity ledger**: integrity/sha256.txt and outputs/publication_row.csv MUST be excluded from the ledger
- **Inconsistent manifests**: manifest.yaml and manifest.json MUST contain identical information
- **Seam receipt without continuity claim**: If seam receipt is present, ss1m_receipt.json:seam.seam_claim SHOULD be true
- **Typed censoring violation**: If τ_R=INF_REC, seam receipt status.pass MUST be false

---

## 10. Reference Implementation

See `casepacks/UMCP-REF-E2E-0001/` for a complete reference CasePack with all required and optional files.

**Key files to inspect**:
- [casepacks/UMCP-REF-E2E-0001/manifest.json](casepacks/UMCP-REF-E2E-0001/manifest.json)
- [casepacks/UMCP-REF-E2E-0001/manifest/manifest.json](casepacks/UMCP-REF-E2E-0001/manifest/manifest.json)
- [casepacks/UMCP-REF-E2E-0001/receipts/ss1m.json](casepacks/UMCP-REF-E2E-0001/receipts/ss1m.json)
- [casepacks/UMCP-REF-E2E-0001/manifest/sha256sums.txt](casepacks/UMCP-REF-E2E-0001/manifest/sha256sums.txt)

---

## 11. Version Control

**Status**: This reference structure is **freeze-controlled** and versioned with the UMCP protocol.

- Schema versions are declared in receipt `schema` fields
- Breaking changes to receipt formats require new schema versions
- Manifest format changes require new Manifest schema versions

**Current Versions**:
- Manifest: Manifest-1.0
- SS1m: SS1m-1.0
- Seam: Seam-1.0

---

**Document Status**: Complete CasePack reference structure from manuscript §5  
**Last Updated**: 2026-01-21  
**Checksum**: (Recorded in integrity/sha256.txt)

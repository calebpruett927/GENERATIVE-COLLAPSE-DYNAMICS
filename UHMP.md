# Universal Hash Manifest Protocol (UHMP)

**Version**: 1.0.0  
**Status**: Protocol Foundation  
**Domain**: Digital manifest, hash seams, and remint welds  
**Axiom**: AX–0 (identity): *Only that which hashes without recursion is manifest.*

---

## Overview

**UHMP makes UMCP artifacts reproducible in the strict sense**: a verifier can (i) obtain the mint target, (ii) compute SHA-256, (iii) compare against the ledger, and (iv) determine identity-match or mismatch without relying on narrative claims, formatting interpretation, or tool-specific "export equivalence."

UHMP completes the measurement-to-manifest loop of UMCP by separating what **MUST remain stable** (the minted bytes and their root hash) from what **MAY change** (presentation, printing, repagination, or readability layers).

---

## Key Concepts (Operational Definitions)

| Term | Operational Meaning | NOT Confused With |
|------|---------------------|-------------------|
| **Mint Target** | The frozen, canonical file (PDF/A, JSON, etc.) whose SHA-256 becomes the manifest root | Draft, display copy, "final version" |
| **Neutral Placeholder** | Fixed literal string in mint target indicating intended hash location without containing the hash value (e.g., `SHA256: [PLACEHOLDER]`) | Actual hash value, temporary marker |
| **Manifest Root (H₀)** | SHA-256 digest of mint target bytes after pre-freeze completion and final export | Document hash, checksum, "fingerprint" |
| **Manifest Ledger Row** | External, authoritative record binding H₀ to contract ID, timestamp, timezone, author, and seam/audit metadata | Git commit, changelog entry |
| **Display Copy** | Optional human-readable derivative that prints H₀ inside document for convenience; NOT the mint target and MUST NOT replace it | Final version, published copy |

---

## Five-Phase Mint Protocol

UHMP proceeds in five sequential phases. Each phase has a clear input, output, and audit consequence.

### Phase 1: Pre-Freeze (Contract Preparation)

**Input**: Draft artifact with frozen UMA contract (contract ID, closures, return settings, tolerances)  
**Action**: Finalize structure, insert neutral placeholder if hash line desired  
**Output**: Structurally complete artifact explicitly open with respect to identity

```text
SHA256: [PLACEHOLDER]
```

**Contract State**: Frozen contract declared  
**Audit Outcome**: Awaiting mint

---

### Phase 2: Hash-Freeze (Mint the Manifest Root)

**Input**: Pre-frozen artifact  
**Action**: Export mint target in canonical distribution format (PDF/A, JSON), compute SHA-256  
**Output**: H₀ (manifest root) - immutable identity claim for mint target bytes

```bash
shasum -a 256 artifact.pdf
# Output: 9a2f83b88d8f7e13b21c1a59f02d9cd5735ad67bb2a0b372d76e4e2c5b6c57e9
```

**Contract State**: Mint target fixed  
**Audit Outcome**: Root H₀ minted

---

### Phase 3: Manifest Ledger Registration (Authoritative Binding)

**Input**: H₀ from Phase 2  
**Action**: Record H₀ in external ledger row binding identity to frozen contract and audit state  
**Output**: Authoritative provenance record

**Minimal JSON-shaped ledger row**:

```json
{
  "schema": "UMA.UHMP.LedgerRow.v1",
  "manifest_id": "WLD-2026-01-11-A01",
  "weld_id": "W-2025-12-31-PHYS-COHERENCE",
  "canon": {
    "pre": "10.5281/zenodo.17756705",
    "post": "10.5281/zenodo.18072852"
  },
  "sha256_root": "9a2f83b88d8f7e13b21c1a59f02d9cd5735ad67bb2a0b372d76e4e2c5b6c57e9",
  "hash_alg": "SHA-256",
  "mint_target": {
    "format": "PDF/A",
    "role": "mint_target_bytes",
    "sha256_field_policy": "placeholder_only",
    "placeholder_literal": "SHA256: [PLACEHOLDER]"
  },
  "created": "2026-01-11T21:00:00-06:00",
  "tz": "America/Chicago",
  "contract": {
    "id": "UMA.IDENTITY.v1",
    "title": "Universal Identity Contract — Manifest and Ledger Governance Layer",
    "parent": "UMA.INTSTACK.v1",
    "domain": "Digital manifest, hash seams, and remint welds",
    "axiom": "AX–0 (identity): Only that which hashes without recursion is manifest.",
    "frozen_params": {
      "embedding_range": [0, 1],
      "face_policy": "pre_clip",
      "epsilon": 1e-8,
      "lambda": 0.2,
      "tol_seam": 0.005,
      "tol_id": 1e-9,
      "oor_policy": "clip_and_flag"
    },
    "typed_censoring": {
      "tau_R_infinite_policy": "R*tau_R := 0"
    }
  },
  "author": {
    "name": "Clement Paulus",
    "role": "Declarant",
    "orcid": "0009-0000-6069-8234"
  },
  "audit": {
    "protocol": "UHMP.v1.0",
    "continuity_law": {
      "delta_kappa_ledger": 0.0,
      "ir": 1.0,
      "identity_check": {
        "abs_ir_minus_exp_delta_kappa": 0.0,
        "tol_id": 1e-9,
        "pass": true
      }
    },
    "seam_budget": {
      "residual_s": 0.0,
      "tol_seam": 0.005,
      "pass": true
    },
    "phi": "S",
    "verdict": "PASS"
  },
  "notes": [
    "This ledger row is the authoritative binding between mint-target bytes and frozen contract.",
    "Any re-export, re-pagination, metadata change, or content change constitutes a new identity and requires a new mint + new ledger row.",
    "The SHA-256 value MUST NOT be embedded into the mint target bytes (non-recursion). If printed for readability, it MUST be a derivative display copy."
  ]
}
```

**Contract State**: Provenance frozen  
**Audit Outcome**: Authoritative binding established

**Critical Rule**: The ledger row is the authoritative statement. The file is the mint target. The hash binds them.

---

### Phase 4: Manifest Citation (Optional Display, Non-Authoritative)

**Input**: Minted artifact with H₀ from ledger  
**Action**: Create display copy printing root hash for human readability (optional)  
**Output**: Display copy (derivative with its own identity)

```text
SHA256 (manifest root): 9a2f83b8...c57e9
```

**Warning**: This is a **citation only**. If you write H₀ into the artifact, the bytes change and the file becomes a different object. The minted file MUST be preserved as-is, and the display copy (if produced) MUST be treated as a derivative.

**Contract State**: Non-recursive reference  
**Audit Outcome**: Convenience only

---

### Phase 5: Print / Remint Policy (Identity Discontinuities)

**Input**: Request for modified, re-exported, or re-formatted derivative  
**Action**: Recognize as new identity, remint with new H₀  
**Output**: New ledger row with PRE→POST seam event

**Any modified derivative is a new identity and MUST be reminted**.

| Operation | PRE (root) | POST (root) | Δκ_ledger | s | Verdict |
|-----------|-----------|-------------|-----------|---|---------|
| weld | 9a2f83b8...c57e9 | 51b0e913...aa9f | 0.000 | 0.000 | PASS |

**Contract State**: New identity  
**Audit Outcome**: New root + new ledger row

---

## Verification Procedure

A verifier checks UHMP compliance by re-performing the mint computation on the received mint target:

1. **Obtain** the mint target bytes (exact file as distributed)
2. **Compute** SHA-256 locally:
   ```bash
   shasum -a 256 received_artifact.pdf
   ```
3. **Compare** computed digest against ledger's `sha256_root`
4. **Determine** identity status:
   - **Match**: Artifact is identity-consistent with ledger claim
   - **Mismatch**: Artifact is nonconformant with UHMP for that manifest ID (regardless of whether content "looks similar")

---

## Seam Law (UHMP as UMCP Weld)

UHMP identity claims are admissible within the same seam discipline as all UMCP welds:

$$
\left| e^{\Delta\kappa} - \frac{IC_{t_1}}{IC_{t_0}} \right| < 10^{-9}, \quad |s| \leq \text{tol}_{\text{seam}}
$$

For a **pure identity operation** (no semantic or metric change):
- Intended closure: Δκ = 0 and s = 0 within tolerance
- Any remint that changes bytes is, by definition, a new identity
- Must be represented as new ledger row (and, if applicable, recorded PRE→POST seam)

---

## Summary Table

| Phase | Operation | Contract State | Audit Outcome |
|-------|-----------|----------------|---------------|
| **Pre-Freeze** | Placeholder; finalize structure | Frozen contract declared | Awaiting mint |
| **Hash-Freeze** | Compute SHA-256 on final export | Mint target fixed | Root H₀ minted |
| **Ledger Registration** | Record H₀ externally | Provenance frozen | Authoritative binding |
| **Manifest Citation** | Optional display copy | Non-recursive reference | Convenience only |
| **Print / Remint** | Any re-export or edit | New identity | New root + new ledger row |

---

## Normative Rules

1. **Mint target MUST** be exported in canonical, final form (e.g., PDF/A for documents, JSON for CasePacks)
2. **Manifest root MUST** be computed over exact bytes of mint target
3. **Manifest root MUST** be recorded externally (ledger) and **MUST NOT** be "closed" by re-hashing revised artifact that prints hash inside itself
4. **Any change** to mint target bytes (content, metadata, formatting, export settings, line endings, compression) constitutes **new identity** and requires reminting

---

## Implementation in UMCP

### Existing Support

UMCP already implements partial UHMP support:

- **Manifest generation**: [casepacks/UMCP-REF-E2E-0001/generate_manifest.py](casepacks/UMCP-REF-E2E-0001/generate_manifest.py)
- **Receipt integration**: `receipts/ss1m.json` includes `manifest.root_sha256`
- **Validation**: [src/umcp/cli.py](src/umcp/cli.py) checks manifest root presence (strict mode)
- **Testing**: [tests/test_25_umcp_ref_e2e_0001.py](tests/test_25_umcp_ref_e2e_0001.py) verifies manifest root hash

### Future Work

- [ ] Full ledger row schema validation (UMA.UHMP.LedgerRow.v1)
- [ ] Automated remint workflow with PRE→POST seam tracking
- [ ] Display copy generation with embedded citations
- [ ] UHMP compliance checker CLI command
- [ ] Ledger row database for institutional deployments

---

## Closing Note

UHMP completes the measurement-to-manifest loop of UMCP by separating what **MUST remain stable** (the minted bytes and their root hash) from what **MAY change** (presentation, printing, repagination, or readability layers). 

Once minted and ledgered, an artifact can be **independently verified and cited without ambiguity**: it is either byte-identical to the claim, or it is not.

**HASH.SEAM**: Δκ = 0.000, s = 0.000 ≤ tol_seam = 0.005, φ = S, seam pass.

---

**See Also**:
- [AXIOM.md](AXIOM.md) - Core axiom and operational definitions
- [GLOSSARY.md](GLOSSARY.md) - Comprehensive term definitions
- [SYMBOL_INDEX.md](SYMBOL_INDEX.md) - Authoritative symbol table
- [contracts/](contracts/) - Frozen contract specifications

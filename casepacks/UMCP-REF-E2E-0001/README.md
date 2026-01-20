# UMCP-REF-E2E-0001

**Audit-ready reference CasePack demonstrating UMCP publication standards**

## Overview

This CasePack serves as the **audit-ready exemplar** for UMCP (Universal Measurement Contract Protocol), demonstrating complete compliance with publication-grade validation requirements. It intentionally exercises all critical behaviors: OOR handling, finite returns, typed non-returns, and strict validation gates.

**Case ID**: UMCP-REF-E2E-0001  
**Status**: CONFORMANT (baseline + strict)  
**Contract**: UMA.INTSTACK.v1  
**Timezone**: America/Chicago  
**Address**: Clement Paulus

## Canon References

- **PRE DOI**: 10.5281/zenodo.17756705 (The Episteme of Return)
- **POST DOI**: 10.5281/zenodo.18072852 (Physics of Coherence)
- **Reference ID**: W-2025-12-31-PHYS-COHERENCE

## What This Demonstrates

1. **Complete pipeline execution**: ingest → freeze → compute → regime → render → export
2. **OOR event exercised**: clip_and_flag policy at t=3 (voltage spike to 12.5V)
3. **Finite return detected**: Exact return to t=0 state at t=5 (τ_R = 5)
4. **Typed non-return**: Multiple INF_REC instances with explicit no-credit rule (R·τ_R = 0)
5. **Tier-1 kernel**: All 7 reserved symbols {ω, F, S, C, τ_R, κ, IC} computed
6. **Invariant consistency**: IC ≈ exp(κ) validated to tolerance 1e-9
7. **Regime classification**: Stable/Watch/Collapse labels based on frozen thresholds
8. **Cryptographic audit trail**: SHA256 manifest + receipt with environment metadata
9. **Strict validation compliance**: Passes publication lint gate requirements
10. **Separation of concerns**: Diagnostics computed but not used as gates

## Test Data

- **Channels**: 3 (voltage, temperature, pressure)
- **Timepoints**: 9 (t=0..8)
- **Physical range**: [0, 10] → embedded to unit hypercube [0,1]
- **OOR event**: t=3, x1=12.5V (clipped to 1.0 and flagged)
- **Return behavior**: 
  - **Finite return at t=5**: Exact match to t=0 state (8.5V, 9.2K, 9.8Pa)
  - **Non-returns (INF_REC)**: t=0,1,2,3,4,6,7,8 (8 instances)

## Frozen Contract Parameters

```yaml
a: 0, b: 1
face: pre_clip
ε: 1e-8
p: 3
α: 1.0
λ: 0.2
η: 1e-3
tol_seam: 0.005
tol_id: 1e-9
OOR: clip_and_flag
```

## How to Rerun

From the CasePack directory:

```bash
# Execute pipeline
python compute_pipeline.py

# Generate manifest
python generate_manifest.py

# Validate (from repo root) - baseline mode
cd /workspaces/UMCP-Metadata-Runnable-Code
umcp validate casepacks/UMCP-REF-E2E-0001

# Validate with strict mode (publication lint gate)
umcp validate --strict casepacks/UMCP-REF-E2E-0001
```

**Expected outcomes:**
- OOR events detected: 1
- Finite returns: 1 (at t=5)
- INF_REC count: 8
- IC ≈ exp(κ) check: PASS (error < 1e-9)
- Baseline validation: CONFORMANT (0 errors, 0 warnings)
- Strict validation: CONFORMANT (0 errors)

## Directory Structure

```
UMCP-REF-E2E-0001/
├── README.md                          # This file
├── contracts/
│   ├── contract.yaml                  # Frozen UMA.INTSTACK.v1 snapshot
│   ├── embedding.yaml                 # x(t) → Ψ(t) N_K embedding specification
│   ├── return.yaml                    # Return domain D_θ and norm ‖·‖ configuration
│   └── weights.yaml                   # Channel weights w_i (uniform: 1/3 each)
├── closures/
│   └── closure_registry.yaml          # Budget terms registry (Γ, D_C, R)
├── data/
│   ├── raw.csv                        # Raw measurements (3 channels, 9 timepoints)
│   └── psi_trace.csv                  # Ψ-coordinates after embedding + OOR handling
├── outputs/
│   ├── kernel_ledger.csv              # Tier-1 kernel: ω,F,S,C,τ_R,κ,IC
│   ├── regime.csv                     # Regime classifications (Stable/Watch/Collapse)
│   └── diagnostics.csv                # Non-gating diagnostic checks
├── receipts/
│   └── ss1m.json                      # SS1M receipt with manifest hash + environment
├── manifest/
│   ├── manifest.json                  # Complete file inventory with SHA256 hashes
│   └── sha256sums.txt                 # Standard checksum format
└── logs/
    └── run.log                        # Pipeline execution log
```

## Key Results

**Final state (t=8)**:
- ω = 0.056667
- F = 0.943333
- S = 0.215076
- C = 0.033993
- τ_R = INF_REC (t=8 did not return)
- κ = -0.058499
- IC = 0.943179 (≈ exp(κ) ✓)
- Regime: Watch

**Return at t=5**:
- τ_R = 5 (finite return detected)
- Distance from t=0: < η (1e-3)
- State match: (8.5V, 9.2K, 9.8Pa) = t=0

**Regime distribution**: Watch (8), Stable (1)  
**OOR events**: 1 (t=3)  
**Typed boundaries**: 8 × INF_REC, 1 × finite τ_R  
**IC consistency**: max error = 0.00e+00 (within 1e-9 tolerance)

## Manifest Root Hash

*(Updated after pipeline execution)*

## Changelog

### 2026-01-20 - Upgrade to Audit-Ready Exemplar

**Motivation**: Upgraded from "structurally valid baseline" to "publication-grade exemplar" that intentionally exercises all critical UMCP behaviors and passes strict validation.

**Changes**:

1. **Data modification** ([data/raw.csv](data/raw.csv)):
   - Added t=8 timepoint
   - Modified t=5 to exactly repeat t=0 state (8.5V, 9.2K, 9.8Pa)
   - **Effect**: Produces 1 finite return (τ_R=5) while maintaining 1 OOR event and 8 INF_REC instances

2. **Enhanced compute pipeline** ([compute_pipeline.py](compute_pipeline.py)):
   - Added IC ≈ exp(κ) consistency validation with per-row checking
   - Added environment metadata to SS1M receipt (Python version, platform, hostname)
   - Enhanced logging with IC consistency checks and tolerance verification
   - **Effect**: Strengthens invariant integrity guarantees and audit trail

3. **Strict validation mode** (src/umcp/cli.py):
   - Implemented `--strict` flag for publication lint gate
   - Added comprehensive CasePack structure checks (contracts/, closures/, receipts/)
   - Validates contract completeness (all UMA.INTSTACK.v1 required parameters)
   - Checks weights normalization (Σw_i = 1.0 within tolerance)
   - Validates manifest hash presence in SS1M receipt
   - Enforces environment metadata requirement
   - **Effect**: Enables publication-ready validation beyond structural correctness

4. **Updated README** (this file):
   - Clarified case purpose as audit-ready exemplar
   - Documented all demonstrated behaviors (OOR, finite return, INF_REC)
   - Added expected outcomes and validation commands
   - Removed weld/seam language (no continuity assertion made)
   - Added comprehensive changelog

**Acceptance criteria met**:
- ✅ OOR events ≥ 1: detected at t=3
- ✅ Finite returns ≥ 1: τ_R=5 at t=5
- ✅ INF_REC ≥ 1: 8 instances across t=0,1,2,3,4,6,7,8
- ✅ IC ≈ exp(κ): validated to 1e-9 tolerance
- ✅ Baseline validation: CONFORMANT (0 errors, 0 warnings)
- ✅ Strict validation: CONFORMANT when properly configured
- ✅ Manifest hash: included in ss1m.json
- ✅ Environment metadata: Python version, platform, hostname recorded

**Contract discipline maintained**:
- Pipeline ordering enforced (ingest → freeze → compute → regime → render → export)
- No Tier-1 computation before /freeze completion
- Tier-1 reserved symbols preserved: {ω, F, S, C, τ_R, κ, IC}
- Typed boundaries explicit (τ_R="INF_REC" or numeric)
- No-credit rule enforced: R·τ_R = 0 when τ_R = INF_REC
- Diagnostics remain non-gating

## Notes

- No continuity assertion made (no seam_receipt.json generated)
- All diagnostics are informational only, not used as gates
- Contract frozen before any Tier-1 computation
- No symbol capture: all Tier-1 symbols reserved
- Explicit typed boundaries: τ_R = INF_REC → R·τ_R = 0
- Budget terms declared in closure registry but not computed (no weld)

## Compliance

✅ Pipeline order enforced  
✅ Contract frozen before computation  
✅ No Tier-1 symbol capture  
✅ Typed boundaries explicit  
✅ OOR handling exercised (clip_and_flag)  
✅ Finite return detected (τ_R=5)  
✅ Non-return typed (INF_REC with no-credit rule)  
✅ IC ≈ exp(κ) validated (< 1e-9 error)  
✅ Diagnostics non-gating  
✅ Cryptographic audit trail  
✅ Environment metadata recorded  
✅ Manifest hash in receipt  
✅ Deterministic outputs  
✅ Third-party reproducible  
✅ Strict validation ready

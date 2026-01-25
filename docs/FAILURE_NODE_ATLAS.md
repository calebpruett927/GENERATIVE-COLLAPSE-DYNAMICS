# Failure Node Atlas

> **Version**: 1.0.0  
> **Specification**: [specs/failure_node_atlas_v1.yaml](../specs/failure_node_atlas_v1.yaml)  
> **Schema**: [schemas/failure_node_atlas.schema.json](../schemas/failure_node_atlas.schema.json)  
> **Implementation**: [src/umcp/preflight.py](../src/umcp/preflight.py)

## Overview

The **Failure Node Atlas** is a canonical enumeration of the recurring ways meaning can accidentally drift in computational workflows. Each "failure node" represents a named, detectable choke point where contract violation or comparability loss can occur.

The validator treats failure nodes as either:
- **ERROR** — Hard nonconformance. Outputs are inadmissible. Do not interpret.
- **WARN** — Suspicious. Interpretation allowed only with explicit warning banner.
- **INFO** — Recorded for audit trail. No enforcement action.

## Quick Reference

| ID | Name | Phase | Severity | Summary |
|----|------|-------|----------|---------|
| FN-001 | Adapter drift | B | ERROR | Embedding/adapter changes without new baseline |
| FN-002 | Bounds drift | B | ERROR | Coordinate bounds change without seam |
| FN-003 | Silent clipping | A | ERROR | OOR values clipped without flags |
| FN-004 | Undeclared smoothing | C | WARN | Hidden preprocessing not declared |
| FN-005 | Timebase drift | A | ERROR | Timestamp semantics change |
| FN-006 | Unit drift | C | WARN | Calibration/units change undeclared |
| FN-007 | Weight drift | A | ERROR | Weights change without baseline |
| FN-008 | Norm drift | B | ERROR | Distance function changes |
| FN-009 | Return settings drift | B | ERROR | η, H_rec, τ_R change without seam |
| FN-010 | Closure incompleteness | A | ERROR | Missing/inconsistent closure registry |
| FN-011 | Diagnostic laundering | A | ERROR | Tier-2 overlays used as gates |
| FN-012 | Post hoc leakage | C | ERROR | Window selection after outcomes |
| FN-013 | Rounding drift | C | WARN | Inconsistent rounding breaks recomputation |
| FN-014 | Manifest integrity | A | ERROR | Hash mismatch or missing manifest |

## Validation Phases

The preflight validator operates in three phases:

### Phase A: Structural Conformance
**Fast, deterministic checks.**

- Required files exist
- Schemas validate
- Manifest present
- Hashes match
- Weights sum to 1
- Timestamps monotone
- Required flags present

**Nodes**: FN-003, FN-005, FN-007, FN-010, FN-011, FN-014

### Phase B: Drift Conformance
**Comparability protection.**

Compares current artifacts to the frozen snapshot (`freeze/` directory) for this Run_ID:
- Adapter/embedding hashes
- Coordinate bounds
- Norm/metric definitions
- Return settings

**Nodes**: FN-001, FN-002, FN-008, FN-009

**Requires**: `freeze/` directory with baseline artifacts

### Phase C: Statistical Sentinels
**Suspicion detectors that force human review.**

- Abrupt distribution shifts in Ψ
- Sudden τ_R density change
- OOR spikes
- Unexpected autocorrelation (smoothing detection)
- Calibration-like jumps

**Nodes**: FN-004, FN-006, FN-012, FN-013

---

## Failure Node Definitions

### FN-001: Adapter Drift (Ψ Meaning Drift)

**Severity**: ERROR (Phase B)

**Definition**: N_K / Adapter_ID changes (or any embedding logic changes) while reusing the same Run_ID or while claiming comparability without a new baseline.

**Artifact Triggers**:
- `adapter.yaml` differs from frozen hash
- Adapter_ID string differs
- Embedding parameters differ

**Kernel Symptom**: Discontinuities in ω/S/C/IC unrelated to any declared intervention; distribution shifts in Ψ coordinates.

**Seam Symptom**: Weld residual s fails to close; identity check breaks intermittently; "mysterious" Δκ changes on otherwise similar runs.

**Rule**: ERROR if adapter hash differs and Run_ID not updated / no new baseline declared; WARN if adapter differs but a new baseline exists and weld not attempted.

---

### FN-002: Bounds Drift (Comparability Intervals K Change)

**Severity**: ERROR (Phase B)

**Definition**: Coordinate bounds (ℓ_i, u_i) change midstream or across runs without explicit seam/baseline.

**Artifact Triggers**:
- Bounds array differs from freeze/adapter schema
- Bounds recorded in two places disagree

**Kernel Symptom**: Abrupt changes in kernel scale; regime label flips without matching raw behavior.

**Seam Symptom**: Systematic residual bias (not noise); repeated FAIL clustered at boundary-crossing segments.

**Rule**: ERROR if bounds differ without new baseline; WARN if bounds differ with baseline but no disclosure in report header.

---

### FN-003: Silent Clipping / Missing OOR Flags

**Severity**: ERROR (Phase A)

**Definition**: Values are clipped or imputed without emitting OOR/missingness flags as part of the measured object.

**Artifact Triggers**:
- Face policy says clip, but no OOR flag columns exist
- OOR counts always zero despite boundary hits
- Missingness handled without flags

**Kernel Symptom**: "Too-stable" appearance; reduced variability; suspiciously low stress signatures.

**Seam Symptom**: False PASS risk (manufactured calm); later welds fail when compared to properly flagged runs.

**Rule**: ERROR if OOR policy requires flags and flags absent; WARN if flags exist but OOR rate is implausibly low given distribution.

---

### FN-004: Undeclared Smoothing/Filtering/Resampling of Ψ(t)

**Severity**: WARN (Phase C)

**Definition**: Smoothing, denoising, moving averages, downsampling, interpolation applied without being declared as part of Tier-0 pipeline.

**Artifact Triggers**:
- `preprocessing_steps` missing/empty but derived traces show non-physical autocorrelation
- Cadence differs from declared sampling
- Kernel computed from a file that is not the direct embedding output

**Kernel Symptom**: τ_R artificially decreases; returns become unusually dense; curvature/roughness attenuated.

**Seam Symptom**: Spurious continuity claims (return manufactured); seams may PASS locally but fail under recomputation with declared pipeline.

**Rule**: ERROR if any preprocessing is detected/declared mismatch; WARN if statistical tests indicate smoothing but metadata claims none (forces human audit).

---

### FN-005: Timebase Drift (Timestamp Semantics Change)

**Severity**: ERROR (Phase A)

**Definition**: Timezone, timestamp parsing, alignment, or indexing changes; "t" is no longer the same variable.

**Artifact Triggers**:
- Timezone mismatch between ingest header and files
- Non-monotone timestamps
- Duplicated timestamps
- Inconsistent sampling interval without disclosure

**Kernel Symptom**: Apparent regime changes aligned to calendar artifacts; sudden ω spikes at DST/time parsing edges.

**Seam Symptom**: Weld comparisons invalid because t0/t1 not comparable; FAILs cluster at timebase discontinuities.

**Rule**: ERROR for non-monotone/duplicate timestamps (unless explicitly allowed); WARN for cadence change without disclosure.

---

### FN-006: Unit Drift Before Embedding

**Severity**: WARN (Phase C)

**Definition**: Unit conversion or sensor calibration changes without update to ingest declaration.

**Artifact Triggers**:
- Units metadata differs from ingest
- Calibration version differs
- Raw ranges shift dramatically without corresponding adapter change

**Kernel Symptom**: Distribution shifts that look like drift in the world but are actually drift in measurement.

**Seam Symptom**: Residual shows structured bias; repeated failures at known calibration update points.

**Rule**: ERROR if units unspecified or conflict; WARN if calibration version changes without new baseline.

---

### FN-007: Weight Drift (w_i Changes)

**Severity**: ERROR (Phase A)

**Definition**: Weights change without being frozen/minted as a new run or without seam procedure.

**Artifact Triggers**:
- `weights.yaml` hash differs from freeze
- Weights sum ≠ 1
- Weights file missing

**Kernel Symptom**: IC/κ changes in scale and sensitivity; regime classification shifts.

**Seam Symptom**: Identity ratio ir mismatches expected exp(Δκ); residual s shifts systematically.

**Rule**: ERROR if weights differ without new baseline; ERROR if weights don't sum to 1 within tolerance; WARN if weights are present but not referenced in manifest.

---

### FN-008: Norm / Distance Function Drift (‖·‖ or Dθ Changes)

**Severity**: ERROR (Phase B)

**Definition**: Change in norm or metric generator used for return detection or kernel computation.

**Artifact Triggers**:
- Contract says one norm but code/config uses another
- Dθ closure registry hash differs

**Kernel Symptom**: τ_R distribution changes; ω sensitivity changes; "return deserts" shift.

**Seam Symptom**: PASS/FAIL becomes irreproducible across machines.

**Rule**: ERROR if norm/Dθ mismatch vs freeze; WARN if multiple norms used in one run without separation.

---

### FN-009: Return Settings Drift (η, H_rec, τ_R Definition)

**Severity**: ERROR (Phase B)

**Definition**: Return threshold η, horizon H_rec, or τ_R rule changes without seam/baseline.

**Artifact Triggers**:
- `return_settings.yaml` differs
- τ_R computed with different horizon than declared

**Kernel Symptom**: τ_R shifts sharply; censoring frequency changes; return credit effectively redefined.

**Seam Symptom**: Weld budgets inconsistent because R·τ_R meaning changes; residual s becomes nonstationary.

**Rule**: ERROR if return settings differ without new baseline; WARN if censoring rate changes drastically vs baseline (forces audit).

---

### FN-010: Closure Registry Incompleteness or Mismatch

**Severity**: ERROR (Phase A)

**Definition**: Missing or inconsistent closure definitions for any claimed seam or published kernel outputs.

**Artifact Triggers**:
- `closures.yaml` absent
- Closure fields missing
- Hashes inconsistent
- A closure referenced in text not present in registry

**Kernel Symptom**: Not recomputable; outputs become "story dependent."

**Seam Symptom**: Cannot be evaluated; or fails because terms can't be reconstructed.

**Rule**: ERROR for any continuity claim without complete closure registry; WARN for exploratory runs.

---

### FN-011: Diagnostic Laundering Into Gates

**Severity**: ERROR (Phase A)

**Definition**: Tier-2 overlays used as decision gates or presented as regime truth.

**Artifact Triggers**:
- Report labels show overlay score as gate
- Regime label derived from non-kernel symbols
- Reserved-symbol linter flags `O.*` or `J.*` in gate path

**Kernel Symptom**: May look fine, but interpretation is invalid.

**Seam Symptom**: Not directly; this is a governance failure that corrupts claims.

**Reserved Symbols**:
- Tier-2 overlay prefixes: `O.`, `O_`
- Tier-2 diagnostic prefixes: `J.`, `J_`
- Allowed Tier-1 kernel symbols: `omega`, `ω`, `S`, `C`, `F`, `IC`, `kappa`, `κ`, `tau_R`, `τ_R`

**Rule**: ERROR if any gate references non-kernel symbols; WARN if overlays appear near gate section without explicit "diagnostic only" labels.

---

### FN-012: Post Hoc Window Selection / Leakage

**Severity**: ERROR (Phase C)

**Definition**: Evaluation windows adjusted after outcomes; training/test contamination; forecasts generated after seeing future labels.

**Artifact Triggers**:
- Forecast receipts missing issuance timestamps
- Model fit window overlaps test window
- Commit timestamps after outcome window but claims "ex ante"

**Kernel Symptom**: None necessarily; this is a validity failure.

**Seam Symptom**: None; but predictive claims become inadmissible.

**Rule**: ERROR for leakage detected; WARN if receipts exist but window policy ambiguous.

---

### FN-013: Rounding / Formatting Drift That Breaks Recomputation

**Severity**: WARN (Phase C)

**Definition**: Published numbers cannot be recomputed due to inconsistent rounding rules or unit presentation drift.

**Artifact Triggers**:
- Rounding policy missing
- Different rounding across tables
- Published SS1m values not reproducible from stored high-precision values

**Kernel Symptom**: Small differences that accumulate into different regime edges.

**Seam Symptom**: False FAIL or false PASS at tolerance boundaries.

**Rule**: WARN when SS1m recomputation mismatch within a "presentation tolerance"; ERROR if mismatch exceeds declared tolerance.

---

### FN-014: Manifest/Hash Integrity Failure

**Severity**: ERROR (Phase A)

**Definition**: Published artifacts aren't the ones computed; bytes changed post-compute.

**Artifact Triggers**:
- SHA256 mismatch
- Manifest missing
- Missing environment pin

**Kernel Symptom**: May match locally but cannot be trusted externally.

**Seam Symptom**: Receipts untrustworthy because referenced artifacts aren't stable.

**Rule**: ERROR on any hash mismatch or missing manifest for a claimed result.

---

## CLI Usage

### Run Preflight Validation

```bash
# Run preflight and output report
umcp preflight

# Verbose output
umcp preflight --verbose

# With explicit freeze directory
umcp preflight --freeze-dir ./freeze

# Output to specific location
umcp preflight --output ./preflight_report.json
```

### Exit Codes

| Code | Status | Meaning |
|------|--------|---------|
| 0 | PASS | No hits above INFO. Outputs admissible. |
| 1 | WARN | Interpretation allowed only with explicit warning banner. |
| 2 | ERROR | Nonconformant. Do not interpret outputs. |

### CI Integration

```yaml
# Example GitHub Actions workflow
- name: Run preflight validation
  run: umcp preflight
  # Will fail CI if exit code is 2 (ERROR)
  # Will warn if exit code is 1 (WARN)
```

---

## Preflight Report Format

The preflight validator produces a JSON report:

```json
{
  "run_id": "RUN-20260125-120000",
  "status": "ERROR",
  "created_utc": "2026-01-25T12:00:00+00:00",
  "validator": {
    "version": "1.0.0",
    "atlas_version": "1.0.0"
  },
  "hits": [
    {
      "node_id": "FN-001",
      "severity": "ERROR",
      "phase": "B",
      "evidence": {
        "expected_adapter_hash": "abc123...",
        "found_adapter_hash": "def456..."
      },
      "action": "Mint new baseline Run_ID or declare seam; do not interpret outputs as comparable."
    }
  ],
  "summary": {
    "error_count": 1,
    "warn_count": 2,
    "info_count": 0
  }
}
```

---

## CasePack Requirements

Any published claim **must** include the preflight report (or a hash pointer to it) in the CasePack manifest:

```yaml
# casepacks/<Run_ID>/manifest.yaml
preflight:
  report: "preflight/preflight_report.json"
  status: "PASS"
  atlas_version: "1.0.0"
```

---

## Reviewer Checklist

When reviewing a CasePack:

1. **Check preflight status** — If not PASS, review all hits
2. **FN-010 or FN-014 present?** — STOP. Do not proceed until resolved.
3. **ERROR hits?** — Outputs are nonconformant and inadmissible
4. **WARN hits?** — Require explicit justification and warning banner
5. **Verify freeze directory** — Phase B checks require baseline artifacts

---

## Creating a Freeze Baseline

To enable Phase B drift detection, create a freeze directory:

```bash
# Create freeze directory
mkdir -p freeze

# Freeze critical artifacts
sha256sum embedding.yaml > freeze/embedding.yaml.sha256
sha256sum return.yaml > freeze/return.yaml.sha256
sha256sum closures/norms.l2_eta1e-3.v1.yaml > freeze/norms_closure.sha256

# Freeze bounds as JSON
python -c "
import yaml, json
with open('observables.yaml') as f:
    obs = yaml.safe_load(f)
with open('freeze/bounds.json', 'w') as f:
    json.dump(obs.get('bounds', {}), f)
"
```

---

## Cross-References

- [KERNEL_SPECIFICATION.md](../KERNEL_SPECIFICATION.md) — Mathematical invariants
- [TIER_SYSTEM.md](../TIER_SYSTEM.md) — Interface/kernel/weld/overlay boundaries
- [INFRASTRUCTURE_GEOMETRY.md](../INFRASTRUCTURE_GEOMETRY.md) — Three-layer architecture
- [PROTOCOL_REFERENCE.md](../PROTOCOL_REFERENCE.md) — Protocol specification
- [CASEPACK_REFERENCE.md](../CASEPACK_REFERENCE.md) — CasePack structure

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-25 | Initial release with 14 failure nodes |

# KIN.REF.PHASE Oscillator CasePack

## Overview

This CasePack provides a **canonical, executable test vector** for the `KIN.REF.PHASE` closure—a deterministic phase-anchor selector for oscillatory motion in the UMCP kinematics framework.

**Purpose**: Pin down phase-anchor selection by deterministic evidence, not prose. This CasePack is audit-ready with:
- Manifest + SHA256 hashes
- Frozen configuration anchors
- Deterministic test vectors
- Complete pytest coverage

## KIN.REF.PHASE Closure

The `KIN.REF.PHASE` closure implements deterministic reference-phase anchor selection:

### Phase Mapping φ(u)

Given normalized phase-space coordinates (x, v) ∈ [0,1]², compute:

```
x' = 2x - 1  (center to [-1,1])
v' = 2v - 1  (center to [-1,1])
φ = atan2(v', x')  (wrapped to [0, 2π))
```

### Circular Distance Δφ(a, b)

```
Δφ(a, b) = min(|a - b|, 2π - |a - b|)
```

Result is in [0, π].

### Return-Domain Generator D_θ(t)

```
D_W(t) = {u : max(0, t - W) ≤ u ≤ t - 1}
D_{W,δ}(t) = {u ∈ D_W(t) : (t - u) ≥ δ}
```

Parameters (FROZEN):
- W = 20 (window size in samples)
- δ = 3 (debounce lag in samples)

### Eligibility ℰ(u)

An index u is eligible if:
1. u ∈ D_{W,δ}(t) (within window and past debounce)
2. (No additional constraints in this minimal test vector)

### Selector Rule

Given current index t with phase φ(t):

1. Compute eligible set: ℰ(t) = D_{W,δ}(t)
2. If ℰ(t) is empty: **undefined** (EMPTY_ELIGIBLE)
3. For each u ∈ ℰ(t), compute Δφ(φ(t), φ(u))
4. Find minimum Δφ_min = min{Δφ(φ(t), φ(u)) : u ∈ ℰ(t)}
5. If Δφ_min > δφ_max: **undefined** (PHASE_MISMATCH)
6. Otherwise, select anchor via tie-breakers:
   - (i) Minimize Δφ
   - (ii) If tied: choose most recent u (largest u)
   - (iii) If still tied: minimize d(Ψ(t), Ψ(u)) where d is Euclidean distance in phase space

### Frozen Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| δφ_max | 0.5235987756 (π/6) | Maximum phase mismatch (30°) |
| W | 20 | Window size (samples) |
| δ | 3 | Debounce lag (samples) |

## Edge Cases Included

This test vector includes three critical edge cases:

### 1. Tie-Breaker Case (rows 15-16)
- Both rows have φ = 0.2140606836
- Forces tie-breaker selection: choose most recent u
- Expected: row 16 selects anchor_u = 3 (not an earlier index)

### 2. Empty Eligible Case (rows 0-2)
- Startup phase: t < δ, so D_{W,δ}(t) = ∅
- Expected: undefined_reason = EMPTY_ELIGIBLE
- Typed censor entry in censor_expected.json

### 3. Phase Mismatch Case (rows 19-20)
- φ = 3.926990817 (≈225°, opposite quadrant)
- All eligible anchors have Δφ > δφ_max
- Expected: undefined_reason = PHASE_MISMATCH
- Typed censor entry in censor_expected.json

## Dataset

The `raw_measurements.csv` contains 31 rows based on the harmonic oscillator scenario from `casepacks/kinematics_complete`, augmented with:
- Identical phase rows for tie-breaker testing
- Phase-inverted rows for mismatch testing

Columns:
- `t`: Time in seconds
- `x`: Normalized position [0,1]
- `v`: Normalized velocity [0,1]
- `scenario`: Always "harmonic" for this test vector

## Validation

### Generate Expected Outputs

```bash
python scripts/generate_kin_ref_phase_expected.py
```

This script:
1. Loads raw_measurements.csv
2. Computes φ for each row
3. Applies D_θ + ℰ + selector + δφ_max rules
4. Outputs expected/ref_phase_expected.csv and expected/censor_expected.json

### Run Tests

```bash
pytest tests/closures/test_kin_ref_phase.py -v
```

Tests verify:
- All defined anchors match expected
- Tie-breaker case selects most recent u
- Empty eligible case produces EMPTY_ELIGIBLE censor
- Phase mismatch case produces PHASE_MISMATCH censor
- Determinism: two runs produce identical output

### Validate CasePack

```bash
umcp validate casepacks/kin_ref_phase_oscillator
```

Expected: CONFORMANT with manifest hash and frozen_config_sha256.

## Files

```
kin_ref_phase_oscillator/
├── manifest.json           # CasePack manifest with artifact hashes
├── README.md               # This file
├── observables.yaml        # Observable definitions
├── embedding.yaml          # Normalization and frozen parameters
├── raw_measurements.csv    # Oscillator dataset with edge cases
├── closures/
│   ├── closure_registry.yaml  # Closure registration
│   └── kin_ref_phase.py       # KIN.REF.PHASE implementation
└── expected/
    ├── ref_phase_expected.csv  # Expected anchor selections
    └── censor_expected.json    # Expected typed censors
```

## References

- [KINEMATICS_SPECIFICATION.md](../../KINEMATICS_SPECIFICATION.md) - Full kinematics spec
- [TIER_SYSTEM.md](../../TIER_SYSTEM.md) - Tier classification
- [closures/kinematics/phase_space_return.py](../../closures/kinematics/phase_space_return.py) - Related return computation

---

*KIN.REF.PHASE v1.0.0 — Deterministic Phase-Anchor Selection*

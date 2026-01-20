# hello_world

**Minimal test case for UMCP validation**

## Overview

This is the simplest possible CasePack demonstrating UMCP validation. It contains a single timepoint with near-maximum coherence values, designed to validate the structural framework without complex dynamics.

**Case ID**: hello_world  
**Status**: CONFORMANT  
**Contract**: UMA.INTSTACK.v1  
**Timezone**: America/Chicago

## Test Data

- **Channels**: 3 (x1_si, x2_si, x3_si)
- **Timepoints**: 1 (t=0)
- **Values**: 9.9 across all channels (→ Ψ = 0.99)
- **Physical range**: [0, 10] → embedded to [0, 1]

## Structure

```
hello_world/
├── README.md                    # This file
├── contracts/
│   ├── contract.yaml            # Frozen UMA.INTSTACK.v1 snapshot
│   ├── embedding.yaml           # x(t) → Ψ(t) specification
│   ├── return.yaml              # Return domain configuration
│   └── weights.yaml             # Uniform channel weights
├── closures/
│   └── closure_registry.yaml   # Budget terms registry
├── raw_measurements.csv         # Input data
├── expected/
│   ├── psi.csv                  # Expected Ψ-coordinates
│   ├── invariants.json          # Expected Tier-1 kernel
│   └── ss1m_receipt.json        # Expected receipt
├── manifest.json                # File list + checksums
└── generate_expected.py         # Output generator

```

## How to Validate

From the repo root:

```bash
# Baseline validation
umcp validate casepacks/hello_world

# Strict validation
umcp validate --strict casepacks/hello_world
```

## Expected Results

- **Status**: CONFORMANT
- **OOR events**: 0 (all values in range)
- **Finite returns**: 0 (single timepoint)
- **τ_R**: INF_REC (no return possible)

## Notes

- Test case for validator framework verification
- All diagnostics informational only
- No weld assertion (test case)
- Contract frozen, all Tier-1 symbols reserved

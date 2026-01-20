# gcd_complete

**Complete test case for GCD (Generalized Collapse Dynamics) framework**

## Overview

This CasePack demonstrates the complete GCD Tier-1 framework validation. It exercises the full pipeline with all 4 GCD closures and validates the complete contract specification.

**Case ID**: gcd_complete  
**Status**: CONFORMANT  
**Contract**: UMA.INTSTACK.v1 (GCD extensions)  
**Timezone**: America/Chicago

## Test Data

- **Channels**: 3 (x1_si, x2_si, x3_si)
- **Timepoints**: Multiple
- **Physical range**: [0, 10] → embedded to [0, 1]
- **Closures**: 4 GCD closures (energy, collapse, flux, resonance)

## Structure

```
gcd_complete/
├── README.md                    # This file
├── contracts/
│   ├── contract.yaml            # Frozen UMA.INTSTACK.v1 + GCD
│   ├── embedding.yaml           # x(t) → Ψ(t) specification
│   ├── return.yaml              # Return domain configuration
│   └── weights.yaml             # Channel weights
├── closures/
│   └── closure_registry.yaml   # GCD closure registry
├── raw_measurements.csv         # Input data
├── expected/
│   ├── psi.csv                  # Expected Ψ-coordinates
│   ├── invariants.json          # Expected Tier-1 + GCD outputs
│   └── ss1m_receipt.json        # Expected receipt
└── manifest.json                # File list + checksums
```

## GCD Closures

1. **Energy** (E): Scalar energy functional
2. **Collapse** (Γ): Instability amplification (p=3)
3. **Flux** (Φ): Cross-channel flow dynamics
4. **Resonance** (R): Oscillatory coupling patterns

## How to Validate

From the repo root:

```bash
# Baseline validation
umcp validate casepacks/gcd_complete

# Strict validation
umcp validate --strict casepacks/gcd_complete
```

## Expected Results

- **Status**: CONFORMANT
- **All GCD closures**: Validated against expected outputs
- **Contract compliance**: Full GCD.INTSTACK.v1 conformance

## Notes

- Test case for complete GCD framework
- All 4 closures computed and validated
- Contract frozen with GCD extensions
- Demonstrates Tier-1 kernel + GCD overlay

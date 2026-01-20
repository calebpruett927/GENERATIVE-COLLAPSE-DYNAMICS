# rcft_complete

**Complete test case for RCFT (Recursive Collapse Field Theory) Tier-2 framework**

**Protocol Resources:**  
[Glossary](../../GLOSSARY.md#tier-2-rcft-overlay-extensions) | [Symbol Index](../../SYMBOL_INDEX.md#tier-2-rcft-extension-symbols) | [Term Index](../../TERM_INDEX.md#r)

## Overview

This CasePack demonstrates the complete RCFT Tier-2 framework validation, including all GCD Tier-1 closures plus 3 RCFT Tier-2 closures. It validates the full tier hierarchy and augmentation model.

**Case ID**: rcft_complete  
**Status**: CONFORMANT  
**Contract**: RCFT.INTSTACK.v1 (extends GCD.INTSTACK.v1)  
**Timezone**: America/Chicago

## Test Data

- **Channels**: 3 (x1_si, x2_si, x3_si)
- **Timepoints**: Multiple (zero-entropy state example)
- **Physical range**: [0, 10] → embedded to [0, 1]
- **Closures**: 7 total (4 GCD + 3 RCFT)

## Structure

```
rcft_complete/
├── README.md                    # This file
├── contracts/
│   ├── contract.yaml            # Frozen RCFT.INTSTACK.v1
│   ├── embedding.yaml           # x(t) → Ψ(t) specification
│   ├── return.yaml              # Return domain configuration
│   └── weights.yaml             # Channel weights
├── closures/
│   └── closure_registry.yaml   # RCFT closure registry (GCD + RCFT)
├── raw_measurements.csv         # Input data (zero-entropy)
├── expected/
│   ├── psi.csv                  # Expected Ψ-coordinates
│   ├── invariants.json          # Expected Tier-1 + GCD + RCFT outputs
│   └── ss1m_receipt.json        # Expected receipt
└── manifest.json                # File list + checksums
```

## RCFT Tier-2 Closures

**GCD Tier-1** (inherited, frozen):
1. **Energy** (E): Scalar energy functional
2. **Collapse** (Γ): Instability amplification
3. **Flux** (Φ): Cross-channel flow
4. **Resonance** (R): Oscillatory coupling

**RCFT Tier-2** (augmentation):
5. **Fractal Dimension** (D_f): Trajectory complexity via box-counting
6. **Recursive Field** (Ψ_r): Exponential memory decay analysis
7. **Resonance Pattern** (λ_p, Θ): FFT-based oscillatory structure

## Tier Hierarchy

- **Tier-1**: Frozen GCD invariants {ω, F, S, C, τ_R, κ, IC}
- **Tier-2**: RCFT augmentation (no override of Tier-1)
- **Augmentation model**: RCFT adds geometric/topological analysis on top of GCD

## How to Validate

From the repo root:

```bash
# Baseline validation
umcp validate casepacks/rcft_complete

# Strict validation
umcp validate --strict casepacks/rcft_complete
```

## Expected Results

- **Status**: CONFORMANT
- **All 7 closures**: Validated against expected outputs
- **Contract compliance**: Full RCFT.INTSTACK.v1 conformance
- **Tier hierarchy**: All GCD Tier-1 frozen, RCFT Tier-2 augmented

## Notes

- Test case for complete RCFT Tier-2 framework
- Demonstrates zero-entropy state (all equal values)
- All 7 closures (4 GCD + 3 RCFT) computed
- Contract frozen with RCFT extensions
- Tier-2 augments but does not override Tier-1

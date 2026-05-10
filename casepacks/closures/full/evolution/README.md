# Evolution Kernel Casepack

**5 representative organisms mapped through the GCD kernel**

## Overview

This CasePack exercises the evolution domain closure with 5 organisms
spanning the tree of life, demonstrating that Tier-1 kernel identities
hold across all biological kingdoms.

**Case ID**: evolution_kernel
**Status**: CONFORMANT
**Contract**: UMA.INTSTACK.v1
**Canon Anchors**: UMCP.CANON.EVO.v1

## Organisms

| # | Organism | Kingdom | F | IC | Regime | Strategy |
|---|----------|---------|---|---|--------|----------|
| 0 | E. coli | Monera | 0.590 | 0.376 | Collapse | Adapted Specialist |
| 1 | Drosophila | Animalia | 0.513 | 0.470 | Collapse | Vulnerable Specialist |
| 2 | Quercus (oak) | Plantae | 0.513 | 0.387 | Collapse | Vulnerable Specialist |
| 3 | Latimeria | Animalia | 0.419 | 0.359 | Collapse | Resilient Ancient |
| 4 | Homo sapiens | Animalia | 0.654 | 0.318 | Collapse | Vulnerable Specialist |

## Tier-1 Identities

All 5 organisms verify:
- **F + ω = 1** (duality identity) — exact to machine precision
- **IC ≤ F** (integrity bound) — holds in every row
- **IC = exp(κ)** (log-integritas) — exact to 1e-9

## Structure

```
evolution_kernel/
├── README.md                    # This file
├── manifest.json                # Casepack manifest
├── raw_measurements.csv         # 5 organisms × 8 channels
├── contracts/
│   └── contract.yaml            # UMA.INTSTACK.v1 frozen snapshot
├── closures/
│   └── closure_registry.yaml    # Evolution closure registry
└── expected/
    └── invariants.json          # Expected Tier-1 invariants
```

## Key Insights

- All organisms are in **Collapse regime** — this reflects genuine channel
  heterogeneity (organisms have strongly varying trait profiles), not failure
- **Homo sapiens** has the highest F (0.654) but lowest IC (0.318) due to
  lineage_persistence = 0.001 — the geometric mean penalizes recency
- **Latimeria** (coelacanth) demonstrates "living fossil" pattern: low F but
  IC/F ratio (0.86) is highest — channels are uniformly modest
- **E. coli** has extreme channel heterogeneity: behavioral_complexity ≈ ε
  while genetic_diversity = 0.92

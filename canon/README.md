# UMCP Canon Standard

The **canon** directory contains the authoritative specifications for UMCP's mathematical framework. These files define frozen identifiers, invariants, and defaults that form the immutable foundation of the system.

## Canon Structure

```
canon/
├── README.md              # This file
├── anchors.yaml           # UMCP.CANON.v1 - Base framework
├── gcd_anchors.yaml       # UMCP.GCD.v1 - Tier-1 framework (GCD)
├── rcft_anchors.yaml      # UMCP.RCFT.v1 - Tier-2 overlay (RCFT)
└── docs/
    └── validator_usage.md # Validation guidance
```

## Tier Hierarchy

| Tier | Canon File | Framework | Description |
|------|------------|-----------|-------------|
| Base | [anchors.yaml](anchors.yaml) | UMCP.CANON.v1 | Base framework: hygiene rules, DOI anchors, contract defaults |
| 1 | [gcd_anchors.yaml](gcd_anchors.yaml) | UMCP.GCD.v1 | Generative Collapse Dynamics: frozen invariants (ω, F, S, C, τ_R, κ, IC) |
| 2 | [rcft_anchors.yaml](rcft_anchors.yaml) | UMCP.RCFT.v1 | Recursive Collapse Field Theory: overlay extensions (D_f, Ψ_r, λ_p, Θ) |

## Key Principles

### Tier Discipline
- **Tier-1 symbols are frozen**: Domain programmes cannot redefine ω, F, S, C, τ_R, κ, IC
- **Tier-2 augments, never overrides**: RCFT adds metrics but cannot change GCD definitions
- **No improvisation**: All terms must trace to canonical definitions

### Immutability
- Semantic changes require version bump (e.g., `UMCP.CANON.v1` → `UMCP.CANON.v2`)
- Changes must be logged in provenance
- Hash-sealed references enforce integrity

## Canon Files

### anchors.yaml (UMCP.CANON.v1)

The base canon anchor defines:
- **Hygiene rules**: Weld definition, EID definition, tier discipline
- **DOI anchors**: PRE and POST publication references
- **Contract defaults**: Embedding interval, face policy, frozen parameters
- **Regime classifications**: Stable, Watch, Collapse thresholds
- **Artifact requirements**: Receipt fields, EID structure

**Schema**: [schemas/canon.anchors.schema.json](../schemas/canon.anchors.schema.json)

### gcd_anchors.yaml (UMCP.GCD.v1)

The Tier-1 GCD anchor defines:
- **Axioms**: AX-0 (Collapse is generative), AX-1 (Boundary defines interior), AX-2 (Entropy measures determinacy)
- **Reserved symbols**: ω, F, S, C, τ_R, κ, IC, IC_min, I
- **Mathematical identities**: F = 1 - ω, IC ≈ exp(κ), κ = ln(ω)
- **Tolerances**: tol_seam = 0.005, tol_identity = 1e-9
- **Typed censoring**: INF_REC, UNIDENTIFIABLE

### rcft_anchors.yaml (UMCP.RCFT.v1)

The Tier-2 RCFT overlay defines:
- **Extension symbols**: D_f (fractal dimension), Ψ_r (recursive field), λ_p (pattern wavelength), Θ (phase angle)
- **Fractal regime classification**: Smooth, Wrinkled, Turbulent
- **Recursive regime classification**: Dormant, Active, Resonant
- **Computational notes**: Box-counting, FFT analysis, series convergence

## Interconnections

```
manifest.yaml
    └── refs.canon_anchors → canon/anchors.yaml
                                 ├── relates → canon/gcd_anchors.yaml (Tier-1)
                                 └── relates → canon/rcft_anchors.yaml (Tier-2)

contracts/GCD.INTSTACK.v1.yaml
    └── refs.canonical_anchor → canon/gcd_anchors.yaml

contracts/RCFT.INTSTACK.v1.yaml
    └── refs.canonical_anchor → canon/rcft_anchors.yaml

casepacks/gcd_complete/manifest.json
    └── refs.canon_anchors → canon/gcd_anchors.yaml

casepacks/rcft_complete/manifest.json
    └── refs.canon_anchors → canon/rcft_anchors.yaml
```

## Validation

The UMCP validator enforces canon compliance:

```bash
# Validate full repo against canon
umcp validate .

# Strict mode (publication-grade)
umcp validate . --strict
```

Canon validation checks:
- ✅ Canon anchor files exist and match schema
- ✅ All Tier-1 symbols are properly defined
- ✅ Mathematical identities hold within tolerance
- ✅ Regime thresholds are consistent
- ✅ No Tier-2 overlay redefines Tier-1 symbols

## Usage in Code

```python
import yaml
from pathlib import Path

# Load canon anchors
canon_path = Path("canon/anchors.yaml")
with open(canon_path) as f:
    canon = yaml.safe_load(f)

# Access frozen parameters
frozen = canon["umcp_canon"]["contract_defaults"]["tier_1_kernel"]["frozen_parameters"]
print(f"p = {frozen['p']}, alpha = {frozen['alpha']}")

# Check regime thresholds
regimes = canon["umcp_canon"]["regimes"]
print(f"Stable: ω < {regimes['stable']['omega_lt']}")
print(f"Collapse: ω ≥ {regimes['collapse']['omega_gte']}")
```

## References

- [GLOSSARY.md](../GLOSSARY.md) — Complete term definitions
- [SYMBOL_INDEX.md](../SYMBOL_INDEX.md) — Symbol lookup table
- [TIER_SYSTEM.md](../TIER_SYSTEM.md) — Tier hierarchy documentation
- [AXIOM.md](../AXIOM.md) — GCD axiomatic foundation
- [KERNEL_SPECIFICATION.md](../KERNEL_SPECIFICATION.md) — Formal invariant specification

## Provenance

| File | Canon ID | Version | Created |
|------|----------|---------|---------|
| anchors.yaml | UMCP.CANON.v1 | 1.0.0 | 2026-01-18 |
| gcd_anchors.yaml | UMCP.GCD.v1 | 1.0.0 | 2026-01-18 |
| rcft_anchors.yaml | UMCP.RCFT.v1 | 1.0.0 | 2026-01-18 |

---

*Last updated: 2026-01-24*

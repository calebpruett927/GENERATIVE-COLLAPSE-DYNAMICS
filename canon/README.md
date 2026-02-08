# UMCP Canon Standard

The **canon** directory contains the authoritative specifications for UMCP's mathematical framework. These files define frozen identifiers, invariants, and defaults that form the immutable foundation of the system.

## Canon Structure

```
canon/
├── README.md              # This file
├── anchors.yaml           # UMCP.CANON.v1 - Base framework
├── gcd_anchors.yaml       # UMCP.GCD.v1 - Tier-1 invariant structure (GCD)
├── rcft_anchors.yaml      # UMCP.RCFT.v1 - Tier-2 domain expansion (RCFT)
├── kin_anchors.yaml       # UMCP.KIN.v1 - Tier-1.5 diagnostic (Kinematics)
├── astro_anchors.yaml     # UMCP.ASTRO.v1 - Tier-2 domain expansion (Astronomy)
├── weyl_anchors.yaml      # UMCP.WEYL.v1 - Weyl geometry extension
└── docs/
    └── validator_usage.md # Validation guidance
```

## Tier Hierarchy

| Tier | Canon File | Framework | Description |
|------|------------|-----------|-------------|
| Base | [anchors.yaml](anchors.yaml) | UMCP.CANON.v1 | Base framework: hygiene rules, DOI anchors, contract defaults |
| 1 | [gcd_anchors.yaml](gcd_anchors.yaml) | UMCP.GCD.v1 | Invariant structure: frozen structural identities (ω, F, S, C, τ_R, κ, IC) |
| 1.5 | [kin_anchors.yaml](kin_anchors.yaml) | UMCP.KIN.v1 | Diagnostic: kinematics control experiments (x, v, a, E_kin, τ_kin, K_stability) |
| 2 | [astro_anchors.yaml](astro_anchors.yaml) | UMCP.ASTRO.v1 | Domain expansion: stellar luminosity, orbital mechanics, spectral analysis |
| 2 | [weyl_anchors.yaml](weyl_anchors.yaml) | UMCP.WEYL.v1 | Domain expansion: conformal symmetry, curvature invariants |
| 2 | [rcft_anchors.yaml](rcft_anchors.yaml) | UMCP.RCFT.v1 | Domain expansion: recursive collapse field theory (D_f, Ψ_r, λ_p, Θ) |

## Key Principles

### Tier Discipline
- **Tier-1 structural invariants are frozen**: Domain expansions cannot redefine ω, F, S, C, τ_R, κ, IC
- **Tier-2 expands, never overrides**: Domain closures map domain observables into Tier-1 invariants but cannot alter structural identities
- **Tier-1.5 diagnostics confirm**: Kinematics and PHYS-04 verify the translation layer works
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

### kin_anchors.yaml (UMCP.KIN.v1)

The Tier-1 Kinematics extension defines:
- **Axioms**: KIN-AX-0 (Phase Space Closure), KIN-AX-1 (Return Time Finiteness), KIN-AX-2 (Conservation Closure), KIN-AX-3 (Stability Finiteness)
- **Reserved symbols**: x, v, a, θ, ω_rot, α, p, E_kin, E_pot, E_mech, τ_kin, K_stability
- **Mathematical identities**: E_kin = ½mv², p = mv, E_mech = E_kin + E_pot
- **Phase space**: Γ_kin = {(x, v) : x ∈ [0,1], v ∈ [0,1]}
- **Return time**: τ_kin - phase space return detection
- **Stability index**: K_stability ∈ [0, 1]

### rcft_anchors.yaml (UMCP.RCFT.v1)

The Tier-2 RCFT overlay defines:
- **Extension symbols**: D_f (fractal dimension), Ψ_r (recursive field), λ_p (pattern wavelength), Θ (phase angle)
- **Fractal regime classification**: Smooth, Wrinkled, Turbulent
- **Recursive regime classification**: Dormant, Active, Resonant
- **Computational notes**: Box-counting, FFT analysis, series convergence

### astro_anchors.yaml (UMCP.ASTRO.v1)

The Tier-1 Astronomy extension defines:
- **Axioms**: AX-A0 (Inverse square law anchors distance), AX-A1 (Mass determines stellar fate), AX-A2 (Spectral class encodes temperature), AX-A3 (Kepler's laws govern bound orbits)
- **Reserved symbols**: L, M_star, T_eff, R_star, lambda_peak, P_orb, a_semi, e_orb, v_rot, M_virial, pi_arcsec, z_cosmo
- **Mathematical identities**: L = 4πR²σT⁴, λ_peak = b/T, P² = (4π²/GM)a³, μ = m − M, 2⟨KE⟩ + ⟨PE⟩ = 0
- **Regime classifications**: Luminosity deviation, spectral fit, orbital stability, distance confidence
- **Closures**: stellar_luminosity, orbital_mechanics, spectral_analysis, distance_ladder, gravitational_dynamics, stellar_evolution

### weyl_anchors.yaml (UMCP.WEYL.v1)

The Tier-1 Weyl geometry extension defines:
- **Extension symbols**: W_μν (Weyl tensor), C_inv (conformal invariant), R_s (Ricci scalar)
- **Conformal symmetry**: Weyl curvature invariants for geometric analysis
- **Regime classification**: Geometric stability thresholds

## Interconnections

```
manifest.yaml
    └── refs.canon_anchors → canon/anchors.yaml
                                 ├── relates → canon/gcd_anchors.yaml (Tier-1)
                                 ├── relates → canon/kin_anchors.yaml (Tier-1)
                                 └── relates → canon/rcft_anchors.yaml (Tier-2)

contracts/GCD.INTSTACK.v1.yaml
    └── refs.canonical_anchor → canon/gcd_anchors.yaml

contracts/KIN.INTSTACK.v1.yaml
    └── refs.canonical_anchor → canon/kin_anchors.yaml

contracts/RCFT.INTSTACK.v1.yaml
    └── refs.canonical_anchor → canon/rcft_anchors.yaml

casepacks/gcd_complete/manifest.json
    └── refs.canon_anchors → canon/gcd_anchors.yaml

casepacks/kinematics_complete/manifest.json
    └── refs.canon_anchors → canon/kin_anchors.yaml

casepacks/rcft_complete/manifest.json
    └── refs.canon_anchors → canon/rcft_anchors.yaml

contracts/ASTRO.INTSTACK.v1.yaml
    └── refs.canonical_anchor → canon/astro_anchors.yaml

contracts/WEYL.INTSTACK.v1.yaml
    └── refs.canonical_anchor → canon/weyl_anchors.yaml

casepacks/astronomy_complete/manifest.json
    └── refs.canon_anchors → canon/astro_anchors.yaml

casepacks/weyl_des_y3/manifest.json
    └── refs.canon_anchors → canon/weyl_anchors.yaml
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
- ✅ No Tier-2 domain expansion redefines Tier-1 structural invariants

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
| kin_anchors.yaml | UMCP.KIN.v1 | 1.0.0 | 2026-01-18 |
| rcft_anchors.yaml | UMCP.RCFT.v1 | 1.0.0 | 2026-01-18 |
| astro_anchors.yaml | UMCP.ASTRO.v1 | 1.0.0 | 2026-02-07 |
| weyl_anchors.yaml | UMCP.WEYL.v1 | 1.0.0 | 2026-01-18 |

---

*Last updated: 2026-02-07*

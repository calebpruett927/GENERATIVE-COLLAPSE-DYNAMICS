# Weyl Evolution DES Y3 CasePack

**Contract**: WEYL.INTSTACK.v1  
**Source**: Nature Communications 15:9295 (2024)  
**Status**: Reference Implementation

---

## Overview

This CasePack implements the Weyl evolution framework for testing modified gravity using Dark Energy Survey Year 3 data. It demonstrates how cosmological observables map to UMCP's geometric infrastructure.

## Core Axiom Realization

**AX-W0: Reference anchor defines deviation**

> The high-redshift anchor z* where General Relativity holds defines the meaning of gravitational deviation at lower redshift. This is the cosmological realization of AX-1 ("Boundary defines interior").

| Axiom | Cosmological Form | UMCP Analog |
|-------|-------------------|-------------|
| AX-W0 | Σ(z*) = 1 (GR at z*) | Return domain D_θ |
| AX-1 | z* boundary → Σ(z) interior | Boundary defines interior |
| AX-2 | σ8 uncertainty | Entropy S |

## Key Results

### σ8 Tension (Collapse Potential Analog)

| Method | σ8(z=0) | Uncertainty |
|--------|---------|-------------|
| From cosmological parameters | 0.849 | ±0.030 |
| From ĥJ measurements | 0.743 | ±0.039 |
| **Tension** | **14.3%** | — |

This 14% discrepancy is the cosmological analog of **collapse potential Φ_collapse**.

### Modified Gravity Σ₀ (Drift Analog)

| g(z) Model | Σ₀ | Uncertainty | χ²_red |
|------------|-----|-------------|--------|
| Standard (Ω_Λ) | 0.24 | ±0.10 | 1.1 |
| Constant | 0.13 | ±0.06 | 1.1 |
| Exponential | 0.027 | ±0.013 | 0.8 |
| **ΛCDM (Σ=1)** | **0.0** | — | **2.1** |

The deviation Σ₀ = 0.24 maps to **drift ω ≈ 0.24** from the GR ideal (Σ = 1).

### ĥJ Measurements (Primary Coordinates)

Effective lens bin redshifts: z̄ = [0.295, 0.467, 0.626, 0.771]

| Bin | z_eff | ĥJ (CMB) | σ |
|-----|-------|----------|---|
| z1 | 0.295 | 0.326 | ±0.020 |
| z2 | 0.467 | 0.332 | ±0.015 |
| z3 | 0.626 | 0.387 | ±0.026 |
| z4 | 0.771 | 0.354 | ±0.033 |

## UMCP Mapping

```
Cosmology                          UMCP
─────────────────────────────────────────────────
ĥJ(z_i)                    →      c_i(t) coordinates
Σ₀ deviation               →      ω (drift from ideal)
σ8 tension                 →      Φ_collapse (collapse potential)
χ²_red fit quality         →      seam residual s
CMB prior                  →      frozen closure
z* = 10 anchor             →      D_θ(t) return domain
Σ(z) = 1 + Σ₀g(z)          →      regime classification gate
```

## Directory Structure

```
weyl_des_y3/
├── README.md                    # This file
├── manifest.json                # CasePack manifest
├── observables.yaml             # All observational data
├── data/
│   ├── hJ_measurements.csv      # ĥJ per bin
│   ├── sigma8_comparison.csv    # σ8 tension data
│   └── Sigma0_fits.csv          # Modified gravity fits
├── expected/
│   ├── invariants.json          # Expected UMCP invariants
│   ├── regime_classification.json
│   └── sigma_mapping.json       # Σ → UMCP mapping
└── closures/
    └── closure_registry.yaml    # Local closure overrides
```

## Mathematical Framework

### Weyl Potential (Eq. 1)
$$\Psi_W \equiv \frac{\Phi + \Psi}{2}$$

### Weyl Transfer Function (Eq. 2)
$$T_{\Psi_W}(k, z) = \frac{H^2(z) J(k,z)}{H^2(z^*) D_1(z^*)} \cdot \sqrt{\frac{B(k,z)}{B(k,z^*)}} \cdot T_{\Psi_W}(k, z^*)$$

### Weyl Evolution Proxy (Eq. 4)
$$\hat{h}_J(z) \equiv J(z) \frac{\sigma_8(z)}{D_1(z)} = J(z) \frac{\sigma_8(z^*)}{D_1(z^*)}$$

### ĥJ to Σ Mapping (Eq. 12)
$$\hat{h}_J(z_i) = \Omega_m(z_i) \cdot \frac{D_1(z_i)}{D_1(z^*)} \cdot \sigma_8(z^*) \cdot \Sigma(z_i)$$

### Σ Parametrization (Eq. 13)
$$\Sigma(z) = 1 + \Sigma_0 \cdot g(z)$$

## Regime Classification

Based on Σ₀ amplitude:

| Regime | Condition | UMCP Analog |
|--------|-----------|-------------|
| GR_consistent | \|Σ₀\| < 0.1 | Stable |
| Tension | 0.1 ≤ \|Σ₀\| < 0.3 | Watch |
| Modified_gravity | \|Σ₀\| ≥ 0.3 | Collapse |

**DES Y3 Result**: Σ₀ = 0.24 → **Tension regime** (borderline modified gravity)

## Usage

```python
from closures.weyl import (
    compute_Sigma,
    Sigma_to_UMCP_invariants,
    DES_Y3_DATA,
    compute_des_y3_background,
)

# Load DES Y3 data
hJ_cmb = DES_Y3_DATA["hJ_cmb"]["mean"]
z_bins = DES_Y3_DATA["z_bins"]

# Compute UMCP invariant mapping
mapping = Sigma_to_UMCP_invariants(
    Sigma_0=0.24,
    chi2_red_Sigma=1.1,
    chi2_red_LCDM=2.1,
)

print(f"ω_analog = {mapping['omega_analog']:.3f}")  # 0.240
print(f"regime = {mapping['regime']}")  # Watch
print(f"χ² improvement = {mapping['chi2_improvement']:.1%}")  # 47.6%
```

## Validation

Run validation:
```bash
umcp validate casepacks/weyl_des_y3
```

Run closure self-tests:
```bash
python closures/weyl/sigma_evolution.py
python closures/weyl/cosmology_background.py
```

## References

1. **Nature Communications 15:9295 (2024)**: "Model-independent test of gravity with a network of galaxy surveys"
2. **DES Collaboration**: Dark Energy Survey Year 3 results
3. **UMCP Kernel Specification**: Mathematical foundations
4. **INFRASTRUCTURE_GEOMETRY.md**: Three-layer geometric architecture

---

*This CasePack demonstrates that cosmological gravity testing follows the same contract discipline as any UMCP validation: freeze the reference (z*), compute deviations (Σ₀), and produce auditable receipts (χ², regime).*

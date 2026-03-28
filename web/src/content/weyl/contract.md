---
title: Contract — WEYL Cosmology
description: Frozen contract for WEYL Cosmology
domain: weyl
pageType: contract
---

# Contract — WEYL Cosmology

> The contract defines the rules *before* evidence. All thresholds, embedding parameters, and reserved symbols are frozen here.

## Identity

| Field | Value |
|-------|-------|
| **Contract ID** | `WEYL.INTSTACK.v1` |
| **Version** | 1.0.0 |
| **Parent Contract** | `GCD.INTSTACK.v1` |
| **Tier Level** | 2 |

## Embedding Configuration

| Parameter | Value |
|-----------|-------|
| `interval` | `[0.0, 1.0]` |
| `face` | `pre_clip` |
| `oor_policy` | `clip_and_flag` |
| `epsilon` | `1e-08` |

## Reserved Symbols (Tier-1)

These symbols are frozen within a run. Any Tier-2 code that redefines them is automatic nonconformance.

**GCD Kernel Invariants** (inherited):

- `omega`
- `F`
- `S`
- `C`
- `tau_R`
- `kappa`
- `IC`
- `IC_min`

**Domain-Specific Symbols**:

- `hJ`
- `hb`
- `z_eff`
- `Sigma`
- `Sigma_0`
- `sigma8`
- `D1`
- `H_z`
- `chi`
- `Omega_m`
- `C_ell`
- `T_Weyl`
- `J`
- `chi2_red`

## Frozen Parameters

*Consistent across the seam — same rules both sides of every collapse-return boundary.*

| Parameter | Value |
|-----------|-------|
| `p` | `3` |
| `alpha` | `1.0` |
| `lambda` | `0.2` |
| `eta` | `0.001` |
| `alpha_energy` | `1.0` |
| `beta_energy` | `0.5` |
| `tau_0` | `10.0` |
| `C_crit` | `0.2` |
| `z_star` | `1100.0` |
| `z_anchor_matter_era` | `10.0` |
| `ell_limber_threshold` | `200` |
| `sigma8_planck` | `0.811` |
| `Omega_m_0` | `0.315` |
| `h_0` | `0.674` |
| `n_s` | `0.965` |
| `z_bin_1` | `0.295` |
| `z_bin_2` | `0.467` |
| `z_bin_3` | `0.626` |
| `z_bin_4` | `0.771` |

**Weights Policy**: `nonnegative_sum_to_one`

---

*Contract frozen by the Headless Contract Gateway (HCG) · Domain: weyl · UMCP v2.3.0*

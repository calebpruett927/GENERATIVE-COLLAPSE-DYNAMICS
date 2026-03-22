---
title: Contract — Recursive Collapse Field Theory
description: Frozen contract for Recursive Collapse Field Theory
domain: rcft
pageType: contract
---

# Contract — Recursive Collapse Field Theory

> The contract defines the rules *before* evidence. All thresholds, embedding parameters, and reserved symbols are frozen here.

## Identity

| Field | Value |
|-------|-------|
| **Contract ID** | `RCFT.INTSTACK.v1` |
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

- `I`
- `E_potential`
- `Phi_collapse`
- `Phi_gen`
- `R`
- `D_fractal`
- `Psi_recursive`
- `lambda_pattern`
- `Theta_phase`

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
| `alpha_decay` | `0.8` |
| `max_recursion_depth` | `100` |
| `n_fft_points` | `512` |
| `eps_min_factor` | `0.01` |

**Weights Policy**: `nonnegative_sum_to_one`

---

*Contract frozen by the Headless Contract Gateway (HCG) · Domain: rcft · UMCP v2.2.5*

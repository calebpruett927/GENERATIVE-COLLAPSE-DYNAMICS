---
title: Contract — Generative Collapse Dynamics
description: Frozen contract for Generative Collapse Dynamics
domain: gcd
pageType: contract
---

# Contract — Generative Collapse Dynamics

> The contract defines the rules *before* evidence. All thresholds, embedding parameters, and reserved symbols are frozen here.

## Identity

| Field | Value |
|-------|-------|
| **Contract ID** | `GCD.INTSTACK.v1` |
| **Version** | 1.0.0 |
| **Parent Contract** | `UMA.INTSTACK.v1` |
| **Tier Level** | 1 |

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

- `F`
- `S`
- `C`
- `IC`
- `IC_min`
- `omega`
- `tau_R`
- `kappa`

**Domain-Specific Symbols**:

- `ω`
- `τ_R`
- `κ`
- `E_potential`
- `Φ_collapse`
- `Φ_gen`
- `R`
- `I`

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

**Weights Policy**: `nonnegative_sum_to_one`

---

*Contract frozen by the Headless Contract Gateway (HCG) · Domain: gcd · UMCP v2.2.5*

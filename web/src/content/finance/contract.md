---
title: Contract — Finance
description: Frozen contract for Finance
domain: finance
pageType: contract
---

# Contract — Finance

> The contract defines the rules *before* evidence. All thresholds, embedding parameters, and reserved symbols are frozen here.

## Identity

| Field | Value |
|-------|-------|
| **Contract ID** | `FINANCE.INTSTACK.v1` |
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

- `revenue_performance`
- `expense_control`
- `gross_margin`
- `cashflow_health`

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
| `n_channels` | `4` |

**Weights Policy**: `nonnegative_sum_to_one`

---

*Contract frozen by the Headless Contract Gateway (HCG) · Domain: finance · UMCP v2.3.0*

---
title: Contract — Standard Model
description: Frozen contract for Standard Model
domain: standard_model
pageType: contract
---

# Contract — Standard Model

> The contract defines the rules *before* evidence. All thresholds, embedding parameters, and reserved symbols are frozen here.

## Identity

| Field | Value |
|-------|-------|
| **Contract ID** | `SM.INTSTACK.v1` |
| **Version** | 1.0.0 |
| **Parent Contract** | `GCD.INTSTACK.v1` |
| **Tier Level** | 2 |

## Embedding Configuration

| Parameter | Value |
|-----------|-------|
| `interval` | `[0.0, 1.0]` |
| `face` | `pre_clip` |
| `epsilon` | `1e-08` |
| `channels` | 8 (mass_log, spin_norm, charge_norm, color, weak_isospin, lepton_num, baryon_num, generation) |
| `weights` | Equal: w_i = 1/8 |

## Reserved Symbols (Tier-1)

**GCD Kernel Invariants** (inherited):

- `F`, `S`, `C`, `IC`, `IC_min`, `omega`, `tau_R`, `kappa`

**Domain-Specific Symbols**:

- `mass_GeV` — particle mass
- `charge_e` — electric charge in units of e
- `spin` — intrinsic spin
- `generation` — fermion generation (1, 2, 3)
- `alpha_s` — strong coupling constant
- `alpha_em` — electromagnetic coupling
- `sin2_theta_W` — weak mixing angle
- `Q_GeV` — momentum transfer scale
- `R_predicted` — R-ratio prediction
- `sigma_point_pb` — point cross section
- `v_GeV` — Higgs VEV (246.22 GeV)
- `V_matrix` — CKM matrix
- `J_CP` — Jarlskog invariant

## Frozen Parameters (PDG 2024)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `alpha_s_MZ` | 0.1179 | Strong coupling at Z mass |
| `alpha_em_0` | 7.2973525693 × 10⁻³ | EM coupling (1/137.036) |
| `sin2_theta_W_0` | 0.23122 | Weak mixing angle |
| `M_Z_GeV` | 91.1876 | Z boson mass |
| `M_W_GeV` | 80.3692 | W boson mass |
| `M_H_GeV` | 125.25 | Higgs boson mass |
| `v_GeV` | 246.22 | Higgs VEV |

**CKM Wolfenstein Parameters**:
- λ = 0.22650, A = 0.790, ρ̄ = 0.141, η̄ = 0.357

## Tolerances

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `tol_seam` | 0.005 | Seam residual |
| `tol_coupling` | 0.01 | 1% running couplings |
| `tol_R_ratio` | 0.05 | 5% R-ratio |
| `tol_mass_prediction` | 0.01 | 1% mass predictions |
| `tol_unitarity` | 1.0 × 10⁻⁶ | CKM unitarity |

---

*Contract frozen by the Headless Contract Gateway (HCG) · Domain: standard_model · UMCP v2.3.0*

*No contract found for this domain.*

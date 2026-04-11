---
title: Contract — Atomic Physics
description: Frozen contract for Atomic Physics
domain: atomic_physics
pageType: contract
---

# Contract — Atomic Physics

> The contract defines the rules *before* evidence. All thresholds, embedding parameters, and reserved symbols are frozen here.

## Identity

| Field | Value |
|-------|-------|
| **Contract ID** | `ATOM.INTSTACK.v1` |
| **Version** | 1.0.0 |
| **Parent Contract** | `GCD.INTSTACK.v1` |
| **Tier Level** | 2 |

## Embedding Configuration

| Parameter | Value |
|-----------|-------|
| `interval` | `[0.0, 1.0]` |
| `face` | `pre_clip` |
| `epsilon` | `1e-08` |
| `channels` | 8 (property kernel) / 12 (cross-scale) |

## Reserved Symbols (Tier-1)

**GCD Kernel Invariants** (inherited):

- `F`, `S`, `C`, `IC`, `IC_min`, `omega`, `tau_R`, `kappa`

**Domain-Specific Symbols**:

- `Z` — atomic number
- `Z_eff` — effective nuclear charge
- `n_eff` — effective principal quantum number
- `IE_predicted_eV` — predicted ionization energy
- `IE_measured_eV` — measured ionization energy
- `Psi_IE` — ionization energy fidelity
- `lambda_nm` — spectral wavelength
- `E_transition_eV` — transition energy
- `shell_completeness` — electron shell fill fraction
- `E_fine_eV` — fine structure splitting
- `delta_E_zeeman_eV` — Zeeman splitting
- `delta_E_stark_eV` — Stark splitting
- `g_lande` — Landé g-factor

## Frozen Parameters (NIST CODATA 2022)

| Parameter | Value | Unit |
|-----------|-------|------|
| `R_inf` | 1.0973731568160 × 10⁷ | m⁻¹ (Rydberg constant) |
| `a_0` | 5.29177210903 × 10⁻¹¹ | m (Bohr radius) |
| `alpha_fs` | 7.2973525693 × 10⁻³ | — (fine-structure constant) |
| `mu_B` | 5.7883818060 × 10⁻⁵ | eV/T (Bohr magneton) |
| `E_H` | 13.605693122994 | eV (hydrogen ground state) |

## Atomic Physics Axioms

- **AX-AT0**: Shell closure is an attractor (noble gases Z = 2, 10, 18, 36, 54, 86)
- **AX-AT1**: Rydberg formula exact for hydrogen (Z = 1)
- **AX-AT2**: Fine structure scales as (Zα)²

## Tolerances

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `tol_seam` | 0.005 | Seam residual |
| `tol_IE` | 0.10 | 10% ionization energy prediction |
| `tol_spectral` | 0.001 | 0.1% wavelength tolerance |
| `tol_fine_structure` | 0.05 | 5% fine-structure splitting |

---

*Contract frozen by the Headless Contract Gateway (HCG) · Domain: atomic_physics · UMCP v2.3.1*

*No contract found for this domain.*

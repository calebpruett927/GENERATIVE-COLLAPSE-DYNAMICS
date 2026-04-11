---
title: Contract — Quantum Mechanics
description: Frozen contract for Quantum Mechanics
domain: quantum_mechanics
pageType: contract
---

# Contract — Quantum Mechanics

> The contract defines the rules *before* evidence. All thresholds, embedding parameters, and reserved symbols are frozen here.

## Identity

| Field | Value |
|-------|-------|
| **Contract ID** | `QM.INTSTACK.v1` |
| **Version** | 1.0.0 |
| **Parent Contract** | `GCD.INTSTACK.v1` |
| **Tier Level** | 2 |

## Embedding Configuration

| Parameter | Value |
|-----------|-------|
| `interval` | `[0.0, 1.0]` |
| `face` | `pre_clip` |
| `epsilon` | `1e-08` |

## Reserved Symbols (Tier-1)

**GCD Kernel Invariants** (inherited):

- `F`, `S`, `C`, `IC`, `IC_min`, `omega`, `tau_R`, `kappa`

**Domain-Specific Observables** (22 channels):

- **Core**: `psi`, `P_born`, `fidelity_state`, `delta_P`, `purity`
- **Entanglement**: `concurrence`, `S_vN`, `bell_parameter`, `negativity`
- **Tunneling**: `T_coeff`, `kappa_barrier`, `barrier_width`, `V_barrier`, `E_particle`
- **Harmonic Oscillator**: `n_quanta`, `E_n`, `delta_E`, `coherent_alpha`, `squeeze_r`
- **Spin**: `S_z`, `S_total`, `g_factor`

## Frozen Physical Constants

| Parameter | Value | Unit |
|-----------|-------|------|
| `hbar` | 1.054571817 × 10⁻³⁴ | J·s |
| `m_electron` | 9.1093837015 × 10⁻³¹ | kg |
| `m_proton` | 1.67262192 × 10⁻²⁷ | kg |
| `mu_B` | 9.2740100783 × 10⁻²⁴ | J/T (Bohr magneton) |
| `a_bohr` | 5.29177210903 × 10⁻¹¹ | m (Bohr radius) |
| `bell_classical_limit` | 2.0 | — |
| `bell_quantum_limit` | 2.8284 (2√2) | — |

## Tolerances

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `tol_seam` | 0.005 | Seam residual |
| `tol_born` | 0.001 | Born rule probability |
| `tol_fidelity` | 0.005 | State fidelity |
| `tol_heisenberg` | 0.01 | Δx·Δp / (ℏ/2) ≥ 1 |
| `tol_bell` | 0.01 | Bell parameter |
| `tol_tunneling` | 0.02 | Tunneling coefficient |

## Closure Coverage (13 modules)

- `entanglement.py` — Wootters concurrence, von Neumann entropy, CHSH-Bell
- `wavefunction_collapse.py` — Born rule, state projection
- `tunneling.py` — Barrier transmission
- `harmonic_oscillator.py` — Coherent/squeezed states
- `spin_measurement.py` — Stern-Gerlach, Larmor precession
- `uncertainty_principle.py` — Heisenberg bound
- `double_slit_interference.py` — Wave-particle duality
- `quantum_dimer_model.py` — Yan et al. 2022
- `fqhe_bilayer_graphene.py` — Kim et al. 2026
- `ters_near_field.py` — Tip-enhanced Raman
- `atom_dot_mi_transition.py` — Atom-dot transitions
- `muon_laser_decay.py` — Muon decay analysis
- `topological_band_structures.py` — Band topology

---

*Contract frozen by the Headless Contract Gateway (HCG) · Domain: quantum_mechanics · UMCP v2.3.1*

*No contract found for this domain.*

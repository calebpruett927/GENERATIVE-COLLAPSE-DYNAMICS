# Contracts changelog

This changelog documents *additions* of new contract versions. Contracts are immutable once released; do not edit old entries, append new ones.

## KIN.INTSTACK

### KIN.INTSTACK.v1 (contract id: `KIN.INTSTACK.v1`)
- File: `contracts/KIN.INTSTACK.v1.yaml`
- Contract version field: `1.0.0`
- Parent: `UMA.INTSTACK.v1`
- Purpose:
  - Physics-based kinematics extension for motion analysis
  - Reserve kinematics symbols: x, v, a, θ, ω_rot, α, p, E_kin, E_pot, E_mech, τ_kin, K_stability
  - Define reference scales: L_ref, v_ref, a_ref, m_ref, t_ref
  - Pin kinematic return detection: eta_position, eta_velocity, H_rec_kin
  - Define stability thresholds: v_stable_max, a_stable_max, E_stable_max
- Closures: linear_kinematics, rotational_kinematics, energy_mechanics, momentum_dynamics, phase_space_return, kinematic_stability
- CasePack: `casepacks/kinematics_complete/`
- Canon Anchor: `canon/kin_anchors.yaml`
- Documentation: `KINEMATICS_SPECIFICATION.md`

- Contract version field: `1.0.0`
- Purpose:
  - Freeze embedding interval, face, OOR policy, epsilon
  - Reserve Tier-1 kernel symbol set
  - Pin typed-censoring enums used across receipts and validators
  - Pin seam tolerances (`tol_seam`, `tol_id`) and frozen parameters (`p`, `alpha`, `lambda`, `eta`)

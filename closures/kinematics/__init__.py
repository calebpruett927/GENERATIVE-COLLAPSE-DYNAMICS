"""
UMCP Kinematics Closures Package

This package provides KIN-Domain Closures for the UMCP framework.
These are DOMAIN CLOSURES (not UMCP Tier-1 kernel invariants).
They emit bounded Ψ observables with typed infinities and frozen tolerances.

CONFORMANCE STATEMENT:
All closures do not redefine or repurpose Tier-1 kernel symbols;
Tier-1 symbols (ω, F, S, C, τ_R, κ, IC) are computed only by the
UMCP kernel from Ψ. Kinematics closures produce Ψ components and
overlays only.

τ_kin is a kinematics overlay diagnostic and does NOT define or
substitute for kernel τ_R.

KIN-Domain Closures:
- Linear kinematics (position, velocity, acceleration)
- Rotational kinematics (angular motion, torque, angular momentum)
- Energy mechanics (kinetic, potential, mechanical energy, work, power)
- Momentum dynamics (linear momentum, impulse, collisions)
- Phase space return (τ_kin computation in (x,v) space)
- Kinematic stability (stability indices and regime classification)

All quantities are normalized to [0,1] via bounded embedding with OOR flags.
Ψ(t) = (x̃, ṽ, ã, Ẽ_kin, p̃) ∈ [0,1]^5 with frozen weights.

Contract: KIN.INTSTACK.v1
Parent: UMA.INTSTACK.v1

Usage:
    from closures.kinematics import (
        compute_linear_kinematics,
        compute_rotational_kinematics,
        compute_mechanical_energy,
        compute_linear_momentum,
        compute_kinematic_return,
        compute_kinematic_stability,
    )

    # Linear kinematics
    result = compute_linear_kinematics(x=0.5, v=0.3, a=0.1)

    # Phase space return
    tau = compute_kinematic_return(x_series, v_series)

See KINEMATICS_SPECIFICATION.md for full mathematical definitions.
"""

from closures.kinematics.energy_mechanics import (
    compute_kinetic_energy,
    compute_mechanical_energy,
    compute_potential_energy,
    compute_power,
    compute_work,
    verify_energy_conservation,
    verify_work_energy_theorem,
)
from closures.kinematics.kinematic_stability import (
    classify_motion_regime,
    compute_kinematic_budget,
    compute_kinematic_stability,
    compute_stability_margin,
    compute_stability_trend,
)
from closures.kinematics.linear_kinematics import (
    compute_linear_kinematics,
    compute_trajectory,
    verify_kinematic_consistency,
)
from closures.kinematics.momentum_dynamics import (
    compute_collision_1d,
    compute_impulse,
    compute_linear_momentum,
    compute_momentum_flux,
    verify_momentum_conservation,
)
from closures.kinematics.phase_space_return import (
    compute_kinematic_return,
    compute_lyapunov_estimate,
    compute_phase_distance,
    compute_phase_trajectory,
    detect_oscillation,
)
from closures.kinematics.rotational_kinematics import (
    compute_centripetal,
    compute_rotational_kinematics,
    compute_rotational_trajectory,
)

__all__ = [
    "classify_motion_regime",
    "compute_centripetal",
    "compute_collision_1d",
    "compute_impulse",
    "compute_kinematic_budget",
    "compute_kinematic_return",
    # Kinematic stability
    "compute_kinematic_stability",
    # Energy mechanics
    "compute_kinetic_energy",
    # Linear kinematics
    "compute_linear_kinematics",
    # Momentum dynamics
    "compute_linear_momentum",
    "compute_lyapunov_estimate",
    "compute_mechanical_energy",
    "compute_momentum_flux",
    # Phase space return
    "compute_phase_distance",
    "compute_phase_trajectory",
    "compute_potential_energy",
    "compute_power",
    # Rotational kinematics
    "compute_rotational_kinematics",
    "compute_rotational_trajectory",
    "compute_stability_margin",
    "compute_stability_trend",
    "compute_trajectory",
    "compute_work",
    "detect_oscillation",
    "verify_energy_conservation",
    "verify_kinematic_consistency",
    "verify_momentum_conservation",
    "verify_work_energy_theorem",
]

__version__ = "1.0.0"
__contract__ = "KIN.INTSTACK.v1"

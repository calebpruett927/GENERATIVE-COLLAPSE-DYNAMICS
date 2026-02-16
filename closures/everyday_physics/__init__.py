"""Everyday Physics Closures — Bridging Particle Physics to Daily Experience

Provides closures mapping macroscopic physical observables to UMCP Tier-1
invariants, demonstrating that the same minimal structure (F + ω = 1,
IC ≤ F, IC = exp(κ)) governs phenomena from thermodynamics to optics.

Cross-scale bridge:
    subatomic (SM)  →  atomic (periodic kernel)  →  everyday  →  stellar  →  cosmic
    This domain occupies the "everyday" node, connecting:
      - Thermodynamics:    atomic properties → heat + phase transitions
      - Electromagnetism:  electron structure → circuits + fields
      - Optics:            photon + atomic transitions → light + color
      - Wave phenomena:    quantum wave-particle → classical waves

Cross-references:
    Contract:  contracts/UMA.INTSTACK.v1.yaml
    Canon:     canon/anchors.yaml
    Registry:  closures/registry.yaml (extensions.everyday_physics)
"""

from __future__ import annotations

from closures.everyday_physics.electromagnetism import compute_electromagnetic_material
from closures.everyday_physics.optics import compute_optical_material
from closures.everyday_physics.thermodynamics import compute_thermal_material
from closures.everyday_physics.wave_phenomena import compute_wave_system

__all__ = [
    "compute_electromagnetic_material",
    "compute_optical_material",
    "compute_thermal_material",
    "compute_wave_system",
]

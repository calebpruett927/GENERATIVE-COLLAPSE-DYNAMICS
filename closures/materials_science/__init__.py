"""Materials Science Closures — MATL.INTSTACK.v1

Derives emergent material properties from fundamental particle interactions
through the RCFT (Recursive Collapse Field Theory) universality framework.

The bridge from atomic physics → materials science is the collapse of individual
atomic observables (ionization energy, electron configuration, fine structure)
into collective material phases.  RCFT's universality class (c_eff = 1/p,
scaling relations, partition functions) governs the phase transitions,
elastic response, and thermal behavior that define material identity.

Key derivation chain:
  Atomic IE (Z, Z_eff, n_eff)  ──→  Cohesive energy per atom
  Electron config (valence, shell)  ──→  Band character (metal/insulator/semiconductor)
  RCFT universality (ν, γ, α, β)  ──→  Phase transition scaling
  RCFT partition function Z(β)  ──→  Debye thermodynamics
  RCFT attractor basins  ──→  Structural stability & polymorphism
  SM coupling running α(Q)  ──→  Bonding character at material energy scales

Five closures:
  1. cohesive_energy — Atomic binding → bulk cohesive energy via Madelung/Born-Mayer
  2. phase_transition — RCFT critical exponents → structural/magnetic phase transitions
  3. elastic_moduli — Interatomic potential curvature → bulk/shear/Young's moduli
  4. band_structure — Electron config + Bloch periodicity → band gaps & conductor class
  5. debye_thermal — RCFT partition function → Debye model thermodynamics
  6. magnetic_properties — Zeeman/Stark → bulk magnetism via Weiss + RCFT
  7. bcs_superconductivity — RCFT partition condensation → BCS pairing
  8. surface_catalysis — Broken-bond model + d-band theory → surface energetics

Cross-references:
    Contract:  contracts/MATL.INTSTACK.v1.yaml
    Canon:     canon/matl_anchors.yaml
    Registry:  closures/registry.yaml (extensions.materials_science)
    Atomic:    closures/atomic_physics/ (IE, electron_config, fine_structure)
    RCFT:      closures/rcft/ (universality_class, partition_function, attractor_basin)
    SM:        closures/standard_model/ (coupling_constants, symmetry_breaking)
"""

from __future__ import annotations

from closures.materials_science.band_structure import compute_band_structure
from closures.materials_science.bcs_superconductivity import (
    compute_bcs_superconductivity,
)
from closures.materials_science.cohesive_energy import compute_cohesive_energy
from closures.materials_science.debye_thermal import compute_debye_thermal
from closures.materials_science.elastic_moduli import compute_elastic_moduli
from closures.materials_science.magnetic_properties import (
    compute_magnetic_properties,
)
from closures.materials_science.phase_transition import compute_phase_transition
from closures.materials_science.surface_catalysis import compute_surface_catalysis

__all__ = [
    "compute_band_structure",
    "compute_bcs_superconductivity",
    "compute_cohesive_energy",
    "compute_debye_thermal",
    "compute_elastic_moduli",
    "compute_magnetic_properties",
    "compute_phase_transition",
    "compute_surface_catalysis",
]

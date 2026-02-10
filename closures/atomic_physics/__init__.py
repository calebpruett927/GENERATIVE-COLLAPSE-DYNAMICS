"""Atomic Physics Closures — ATOM.INTSTACK.v1

Provides closures for atomic structure, ionization energies, spectral lines,
electron configuration, and fine structure analysis — mapping to UMCP Tier-1 invariants.

Cross-references:
  Contract:  contracts/ATOM.INTSTACK.v1.yaml
  Canon:     canon/atom_anchors.yaml
  Registry:  closures/registry.yaml (extensions.atomic_physics)
"""

from __future__ import annotations

from closures.atomic_physics.electron_config import compute_electron_config
from closures.atomic_physics.fine_structure import compute_fine_structure
from closures.atomic_physics.ionization_energy import compute_ionization
from closures.atomic_physics.selection_rules import compute_selection_rules
from closures.atomic_physics.spectral_lines import compute_spectral_lines
from closures.atomic_physics.zeeman_stark import compute_zeeman_stark

__all__ = [
    "compute_electron_config",
    "compute_fine_structure",
    "compute_ionization",
    "compute_selection_rules",
    "compute_spectral_lines",
    "compute_zeeman_stark",
]

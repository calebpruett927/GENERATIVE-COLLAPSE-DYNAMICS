"""Nuclear Physics Closures — NUC.INTSTACK.v1

Provides closures for nuclear binding, decay dynamics, shell structure,
fissility assessment, decay chains, and the double-sided collapse overlay.

Cross-references:
  Contract:  contracts/NUC.INTSTACK.v1.yaml
  Canon:     canon/nuc_anchors.yaml
  Registry:  closures/registry.yaml (extensions.nuclear_physics)
"""

from __future__ import annotations

from closures.nuclear_physics.alpha_decay import compute_alpha_decay
from closures.nuclear_physics.decay_chain import compute_decay_chain
from closures.nuclear_physics.double_sided_collapse import compute_double_sided
from closures.nuclear_physics.fissility import compute_fissility
from closures.nuclear_physics.nuclide_binding import compute_binding
from closures.nuclear_physics.shell_structure import compute_shell

__all__ = [
    "compute_alpha_decay",
    "compute_binding",
    "compute_decay_chain",
    "compute_double_sided",
    "compute_fissility",
    "compute_shell",
]

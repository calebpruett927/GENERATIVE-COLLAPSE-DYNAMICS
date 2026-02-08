"""RCFT Closures — RCFT.INTSTACK.v1

Recursive Collapse Field Theory extensions providing Tier-2 overlays
for fractal dimension, recursive fields, attractor basins, and
resonance patterns.

Cross-references:
    Contract:  contracts/RCFT.INTSTACK.v1.yaml
    Canon:     canon/rcft_anchors.yaml
    Registry:  closures/registry.yaml (extensions.rcft)
"""

from __future__ import annotations

from closures.rcft.attractor_basin import compute_attractor_basin
from closures.rcft.fractal_dimension import compute_fractal_dimension
from closures.rcft.recursive_field import compute_recursive_field
from closures.rcft.resonance_pattern import compute_resonance_pattern

__all__ = [
    "compute_attractor_basin",
    "compute_fractal_dimension",
    "compute_recursive_field",
    "compute_resonance_pattern",
]

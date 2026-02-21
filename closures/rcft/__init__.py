"""RCFT Closures — RCFT.INTSTACK.v1

Recursive Collapse Field Theory extensions providing Tier-2 overlays
for fractal dimension, recursive fields, attractor basins, resonance
patterns, information geometry, universality classification, and
collapse grammar.

Cross-references:
    Contract:  contracts/RCFT.INTSTACK.v1.yaml
    Canon:     canon/rcft_anchors.yaml
    Registry:  closures/registry.yaml (extensions.rcft)
"""

from __future__ import annotations

from closures.rcft.attractor_basin import compute_attractor_basin
from closures.rcft.coherence_pipeline_closure import (
    CoherenceDerivation,
    derive_coherence,
    verify_coherence_is_derived,
)
from closures.rcft.collapse_grammar import diagnose_grammar
from closures.rcft.fractal_dimension import compute_fractal_dimension
from closures.rcft.information_geometry import (
    compute_efficiency,
    compute_geodesic_budget_cost,
    fisher_distance_1d,
    fisher_distance_weighted,
    fisher_geodesic,
    verify_fano_fisher_duality,
)
from closures.rcft.recursive_field import compute_recursive_field
from closures.rcft.resonance_pattern import compute_resonance_pattern
from closures.rcft.universality_class import (
    compute_central_charge,
    compute_critical_exponents,
    compute_partition_function,
    verify_scaling_relations,
)

__all__ = [
    "CoherenceDerivation",
    "compute_attractor_basin",
    "compute_central_charge",
    "compute_critical_exponents",
    "compute_efficiency",
    "compute_fractal_dimension",
    "compute_geodesic_budget_cost",
    "compute_partition_function",
    "compute_recursive_field",
    "compute_resonance_pattern",
    "derive_coherence",
    "diagnose_grammar",
    "fisher_distance_1d",
    "fisher_distance_weighted",
    "fisher_geodesic",
    "verify_coherence_is_derived",
    "verify_fano_fisher_duality",
    "verify_scaling_relations",
]

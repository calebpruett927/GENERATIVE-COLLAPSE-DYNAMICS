"""Standard Model closures for UMCP — SM.INTSTACK.v1

Ten closures mapping Standard Model observables to UMCP Tier-1 invariants:
  1. particle_catalog            — Full SM particle table with mass/charge/spin embedding
  2. coupling_constants          — Running couplings α_s(Q²), α_em(Q²), G_F
  3. cross_sections              — σ(e⁺e⁻→hadrons), R-ratio, Drell-Yan
  4. symmetry_breaking           — Higgs mechanism, VEV, mass generation
  5. ckm_mixing                  — CKM matrix, unitarity triangle, CP violation
  6. pmns_mixing                 — PMNS lepton mixing, neutrino oscillations, complementarity
  7. subatomic_kernel            — 31 particles (17 fundamental + 14 composite)
  8. particle_physics_formalism  — 10 proven theorems (74 tests, duality exact)
  9. neutrino_oscillation        — DUNE/LBNF oscillation, MSW matter effects, T11+T12 (13 tests)

Cross-references:
    Contract:  contracts/SM.INTSTACK.v1.yaml
    Canon:     canon/sm_anchors.yaml
    Registry:  closures/registry.yaml (extensions.standard_model)
"""

from __future__ import annotations

from closures.standard_model.ckm_mixing import compute_ckm_mixing
from closures.standard_model.coupling_constants import compute_running_coupling
from closures.standard_model.cross_sections import compute_cross_section
from closures.standard_model.neutrino_oscillation import (
    compute_dune_prediction,
    compute_oscillation_point,
    compute_oscillation_sweep,
    oscillation_probability_matter,
    oscillation_probability_vacuum,
)
from closures.standard_model.particle_catalog import get_particle, list_particles
from closures.standard_model.pmns_mixing import (
    compute_mixing_comparison,
    compute_pmns_mixing,
)
from closures.standard_model.subatomic_kernel import (
    compute_all_composite,
    compute_all_fundamental,
)
from closures.standard_model.symmetry_breaking import compute_higgs_mechanism

__all__ = [
    "compute_all_composite",
    "compute_all_fundamental",
    "compute_ckm_mixing",
    "compute_cross_section",
    "compute_dune_prediction",
    "compute_higgs_mechanism",
    "compute_mixing_comparison",
    "compute_oscillation_point",
    "compute_oscillation_sweep",
    "compute_pmns_mixing",
    "compute_running_coupling",
    "get_particle",
    "list_particles",
    "oscillation_probability_matter",
    "oscillation_probability_vacuum",
]

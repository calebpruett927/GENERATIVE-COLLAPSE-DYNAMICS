"""Standard Model closures for UMCP — SM.INTSTACK.v1

Eight closures mapping Standard Model observables to UMCP Tier-1 invariants:
  1. particle_catalog            — Full SM particle table with mass/charge/spin embedding
  2. coupling_constants          — Running couplings α_s(Q²), α_em(Q²), G_F
  3. cross_sections              — σ(e⁺e⁻→hadrons), R-ratio, Drell-Yan
  4. symmetry_breaking           — Higgs mechanism, VEV, mass generation
  5. ckm_mixing                  — CKM matrix, unitarity triangle, CP violation
  6. subatomic_kernel            — 31 particles (17 fundamental + 14 composite)
  7. particle_physics_formalism  — 10 proven theorems (74 tests, duality exact)

Cross-references:
    Contract:  contracts/SM.INTSTACK.v1.yaml
    Canon:     canon/sm_anchors.yaml
    Registry:  closures/registry.yaml (extensions.standard_model)
"""

from __future__ import annotations

from closures.standard_model.ckm_mixing import compute_ckm_mixing
from closures.standard_model.coupling_constants import compute_running_coupling
from closures.standard_model.cross_sections import compute_cross_section
from closures.standard_model.particle_catalog import get_particle, list_particles
from closures.standard_model.subatomic_kernel import (
    compute_all_composites,
    compute_all_fundamentals,
)
from closures.standard_model.symmetry_breaking import compute_higgs_mechanism

__all__ = [
    "compute_all_composites",
    "compute_all_fundamentals",
    "compute_ckm_mixing",
    "compute_cross_section",
    "compute_higgs_mechanism",
    "compute_running_coupling",
    "get_particle",
    "list_particles",
]

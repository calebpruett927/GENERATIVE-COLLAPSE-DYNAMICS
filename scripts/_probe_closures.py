"""Temporary script to probe closure module interfaces."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODULES = [
    "closures.astronomy.binary_star_systems",
    "closures.astronomy.long_period_radio_transients",
    "closures.awareness_cognition.attention_mechanisms",
    "closures.clinical_neuroscience.developmental_neuroscience",
    "closures.clinical_neuroscience.neurotransmitter_systems",
    "closures.clinical_neuroscience.sleep_neurophysiology",
    "closures.consciousness_coherence.altered_states",
    "closures.consciousness_coherence.neural_correlates",
    "closures.continuity_theory.budget_geometry",
    "closures.continuity_theory.organizational_resilience",
    "closures.continuity_theory.topological_persistence",
    "closures.dynamic_semiotics.computational_semiotics",
    "closures.dynamic_semiotics.media_coherence",
    "closures.everyday_physics.acoustics",
    "closures.everyday_physics.fluid_dynamics",
    "closures.evolution.molecular_evolution",
    "closures.finance.market_microstructure",
    "closures.finance.volatility_surface",
    "closures.kinematics.rigid_body_dynamics",
    "closures.materials_science.defect_physics",
    "closures.nuclear_physics.reaction_channels",
    "closures.quantum_mechanics.photonic_confinement",
    "closures.quantum_mechanics.topological_band_structures",
    "closures.spacetime_memory.cosmological_memory",
    "closures.spacetime_memory.gravitational_phenomena",
    "closures.spacetime_memory.gravitational_wave_memory",
    "closures.spacetime_memory.temporal_topology",
    "closures.standard_model.electroweak_precision",
]

for mod_path in MODULES:
    try:
        mod = importlib.import_module(mod_path)
    except Exception as e:
        print(f"{mod_path}: IMPORT ERROR: {e}")
        continue

    entities_name = None
    channels_name = None
    compute_name = None
    n_channels_name = None
    verify_fns = []

    for name in dir(mod):
        if name.endswith("_ENTITIES") and name[0].isupper():
            entities_name = name
        if name.endswith("_CHANNELS") and name.startswith("N_"):
            n_channels_name = name
        elif name.endswith("_CHANNELS") and name[0].isupper():
            channels_name = name
        if name.startswith("compute_") and name.endswith("_kernel"):
            compute_name = name
        if name.startswith("verify_t_"):
            verify_fns.append(name)

    prefix = entities_name.replace("_ENTITIES", "") if entities_name else "?"
    n_ent = len(getattr(mod, entities_name)) if entities_name else 0
    n_ch = getattr(mod, n_channels_name) if n_channels_name else 0
    n_theorems = len(verify_fns)
    print(
        f"{mod_path}: prefix={prefix}, entities={n_ent}, channels={n_ch}, compute={compute_name}, theorems={n_theorems}"
    )

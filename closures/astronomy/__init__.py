"""Astronomy Closures — ASTRO.INTSTACK.v1

Provides closures for stellar luminosity, distance ladder, spectral
analysis, stellar evolution, orbital mechanics, gravitational dynamics,
and cosmological parameter analysis (Planck 2018).

Cross-references:
    Contract:  contracts/ASTRO.INTSTACK.v1.yaml
    Canon:     canon/astro_anchors.yaml
    Registry:  closures/registry.yaml (extensions.astronomy)
"""

from __future__ import annotations

from closures.astronomy.cosmology import compute_all_cosmological_epochs, compute_cosmological_epoch
from closures.astronomy.distance_ladder import compute_distance_ladder
from closures.astronomy.gravitational_dynamics import compute_gravitational_dynamics
from closures.astronomy.long_period_radio_transients import (
    build_lpt_catalog,
    compute_all_lpt_kernels,
    compute_lpt_kernel,
)
from closures.astronomy.long_period_radio_transients import (
    run_full_analysis as run_lpt_analysis,
)
from closures.astronomy.orbital_mechanics import compute_orbital_mechanics
from closures.astronomy.spectral_analysis import compute_spectral_analysis
from closures.astronomy.stellar_evolution import compute_stellar_evolution
from closures.astronomy.stellar_luminosity import compute_stellar_luminosity

__all__ = [
    "build_lpt_catalog",
    "compute_all_cosmological_epochs",
    "compute_all_lpt_kernels",
    "compute_cosmological_epoch",
    "compute_distance_ladder",
    "compute_gravitational_dynamics",
    "compute_lpt_kernel",
    "compute_orbital_mechanics",
    "compute_spectral_analysis",
    "compute_stellar_evolution",
    "compute_stellar_luminosity",
    "run_lpt_analysis",
]

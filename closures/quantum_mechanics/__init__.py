"""Quantum Mechanics closures for UMCP — QM.INTSTACK.v1.

Six closures mapping quantum observables to UMCP Tier-1 invariants:
  1. wavefunction_collapse — Born rule, state fidelity, purity
  2. entanglement — concurrence, Bell parameter, von Neumann entropy
  3. tunneling — barrier transmission, decay constant
  4. harmonic_oscillator — energy quantization, coherent states
  5. spin_measurement — Stern-Gerlach, Zeeman, Larmor
  6. uncertainty_principle — Heisenberg bounds

Cross-references:
    Contract:  contracts/QM.INTSTACK.v1.yaml
    Canon:     canon/qm_anchors.yaml
    Registry:  closures/registry.yaml (extensions.quantum_mechanics)
"""

from __future__ import annotations

from closures.quantum_mechanics.entanglement import compute_entanglement
from closures.quantum_mechanics.harmonic_oscillator import compute_harmonic_oscillator
from closures.quantum_mechanics.spin_measurement import compute_spin_measurement
from closures.quantum_mechanics.tunneling import compute_tunneling
from closures.quantum_mechanics.uncertainty_principle import compute_uncertainty
from closures.quantum_mechanics.wavefunction_collapse import compute_wavefunction_collapse

__all__ = [
    "compute_entanglement",
    "compute_harmonic_oscillator",
    "compute_spin_measurement",
    "compute_tunneling",
    "compute_uncertainty",
    "compute_wavefunction_collapse",
]

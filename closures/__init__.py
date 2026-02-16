"""
UMCP Closures Package

This package contains all closure implementations for the UMCP validation framework.
Closures are frozen computational rules that extend or customize kernel behavior
while maintaining auditability and reproducibility.

Available Subpackages:
    - gcd: Generative Collapse Dynamics closures
    - rcft: Recursive Collapse Field Theory closures
    - kinematics: Kinematics domain closures
    - security: Security validation closures
    - weyl: WEYL cosmology closures (modified gravity)
    - astronomy: Stellar classification, HR diagram
    - nuclear_physics: Binding energy, decay chains
    - quantum_mechanics: Wavefunction, entanglement
    - finance: Portfolio continuity, market coherence
    - atomic_physics: Periodic kernel, cross-scale, Tier-1 proof (118 elements)
    - materials_science: Element database (118 elements, 18 fields)
    - standard_model: Subatomic kernel (31 particles), 10 proven theorems
    - everyday_physics: Thermodynamics, electromagnetism, optics, wave phenomena

See closures/registry.yaml for the complete closure registry.
"""

from pathlib import Path

# Closure package root
CLOSURES_ROOT = Path(__file__).parent

# Available closure domains (13 total)
CLOSURE_DOMAINS = [
    "gcd",
    "rcft",
    "kinematics",
    "security",
    "weyl",
    "astronomy",
    "nuclear_physics",
    "quantum_mechanics",
    "finance",
    "atomic_physics",
    "materials_science",
    "standard_model",
    "everyday_physics",
]

__all__ = ["CLOSURES_ROOT", "CLOSURE_DOMAINS"]

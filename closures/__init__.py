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

See closures/registry.yaml for the complete closure registry.
"""

from pathlib import Path

# Closure package root
CLOSURES_ROOT = Path(__file__).parent

# Available closure domains
CLOSURE_DOMAINS = ["gcd", "rcft", "kinematics", "security", "weyl"]

__all__ = ["CLOSURES_ROOT", "CLOSURE_DOMAINS"]

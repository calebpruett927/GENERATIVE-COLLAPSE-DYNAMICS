#!/usr/bin/env python3
"""
GCD Closure: Field Resonance
Detects resonance patterns in collapse fields per Axiom AX-1 ("boundary defines interior").

Formula:
    R = (1 - |ω|) · (1 - S) · exp(-C / C_crit)

Where:
    ω = Drift measure
    S = Entropy (disorder measure)
    C = Curvature measure
    C_crit = Critical curvature threshold (default=0.2)

Physical interpretation:
    - Low drift + low entropy → high resonance (system coherent)
    - High curvature → dampened resonance (boundary distortion)
    - R quantifies how well boundary conditions resonate with interior dynamics
    - R ≈ 1: Perfect resonance (boundary-interior alignment)
    - R ≈ 0: No resonance (boundary-interior decoupled)

Resonance regime thresholds:
    - Decoupled: R < 0.3
    - Partial: 0.3 ≤ R < 0.7
    - Coherent: R ≥ 0.7
"""

import math


def compute_field_resonance(omega: float, S: float, C: float, C_crit: float = 0.2) -> dict[str, float]:
    """
    Compute field resonance from GCD Tier-1 invariants.

    Args:
        omega: Drift measure (|ω| ≤ 1)
        S: Entropy (0 ≤ S ≤ 1)
        C: Curvature measure (C ≥ 0)
        C_crit: Critical curvature threshold (default=0.2)

    Returns:
        Dictionary with:
            - resonance: Total field resonance (0 ≤ R ≤ 1)
            - coherence_factor: (1-|ω|) drift coherence
            - order_factor: (1-S) entropy order
            - curvature_damping: exp(-C/C_crit) boundary damping
            - regime: Resonance regime classification

    Raises:
        ValueError: If inputs violate constraints
    """
    # Input validation
    if abs(omega) > 1:
        raise ValueError(f"Drift |ω| must be ≤ 1, got {abs(omega)}")
    if not (0 <= S <= 1):
        raise ValueError(f"Entropy S must be in [0,1], got {S}")
    if C < 0:
        raise ValueError(f"Curvature C must be ≥ 0, got {C}")
    if C_crit <= 0:
        raise ValueError(f"Critical curvature C_crit must be > 0, got {C_crit}")

    # Compute components
    coherence_factor = 1.0 - abs(omega)
    order_factor = 1.0 - S
    curvature_damping = math.exp(-C / C_crit)

    # Total resonance
    resonance = coherence_factor * order_factor * curvature_damping

    # Classify regime
    if resonance < 0.3:
        regime = "Decoupled"
    elif resonance < 0.7:
        regime = "Partial"
    else:
        regime = "Coherent"

    return {
        "resonance": resonance,
        "coherence_factor": coherence_factor,
        "order_factor": order_factor,
        "curvature_damping": curvature_damping,
        "regime": regime,
    }


def main():
    """Test field resonance computation across regime spectrum."""
    print("=" * 70)
    print("GCD CLOSURE TEST: Field Resonance")
    print("=" * 70)
    print()

    # Test cases spanning regime spectrum
    test_cases = [
        {
            "name": "Zero Entropy (ω=0, S=0, C=0, perfect resonance)",
            "omega": 0.0,
            "S": 0.0,
            "C": 0.0,
        },
        {
            "name": "Stable Regime (low drift, low entropy, low curvature)",
            "omega": 0.01,
            "S": 0.056,
            "C": 0.05,
        },
        {
            "name": "Watch Regime (moderate drift, moderate entropy)",
            "omega": 0.15,
            "S": 0.20,
            "C": 0.15,
        },
        {
            "name": "Collapse Boundary (high drift, high entropy, high curvature)",
            "omega": 0.30,
            "S": 0.50,
            "C": 0.50,
        },
    ]

    for tc in test_cases:
        result = compute_field_resonance(omega=tc["omega"], S=tc["S"], C=tc["C"])

        print(f"{tc['name']}")
        print(f"  Inputs: ω={tc['omega']:.3f}, S={tc['S']:.3f}, C={tc['C']:.2f}")
        print(f"  R = {result['resonance']:.6f}")
        print("  Components:")
        print(f"    - Coherence (1-|ω|): {result['coherence_factor']:.3f}")
        print(f"    - Order (1-S): {result['order_factor']:.3f}")
        print(f"    - Curvature damping: {result['curvature_damping']:.6f}")
        print(f"  Regime: {result['regime']}")
        print()


if __name__ == "__main__":
    main()

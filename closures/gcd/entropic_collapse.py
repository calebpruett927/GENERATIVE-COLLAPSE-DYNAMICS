#!/usr/bin/env python3
"""
GCD Closure: Entropic Collapse
Computes entropy-driven collapse dynamics per Axiom AX-0 ("collapse is generative").

Formula:
    Φ_collapse = S · (1 - F) · exp(-τ_R / τ_0)

Where:
    S = Entropy (disorder measure)
    F = Fidelity (coherence measure)
    τ_R = Relaxation timescale
    τ_0 = Reference timescale (default=10.0)

Physical interpretation:
    - High entropy + low fidelity → strong collapse tendency
    - Long relaxation times → dampened collapse (system resists)
    - Φ_collapse quantifies generative potential released by collapse

Regime thresholds:
    - Minimal: Φ_collapse < 0.01
    - Active: 0.01 ≤ Φ_collapse < 0.1
    - Critical: Φ_collapse ≥ 0.1
"""


def compute_entropic_collapse(S: float, F: float, tau_R: float, tau_0: float = 10.0) -> dict[str, float]:
    """
    Compute entropic collapse potential from GCD Tier-1 invariants.

    Args:
        S: Entropy (0 ≤ S ≤ 1)
        F: Fidelity (0 ≤ F ≤ 1)
        tau_R: Relaxation timescale (τ_R > 0)
        tau_0: Reference timescale (default=10.0)

    Returns:
        Dictionary with:
            - phi_collapse: Total entropic collapse potential
            - S_contribution: Entropy contribution (S)
            - F_contribution: Anti-fidelity contribution (1-F)
            - tau_damping: Temporal damping factor (exp(-τ_R/τ_0))
            - regime: Collapse regime classification

    Raises:
        ValueError: If inputs violate constraints
    """
    # Input validation
    if not (0 <= S <= 1):
        raise ValueError(f"Entropy S must be in [0,1], got {S}")
    if not (0 <= F <= 1):
        raise ValueError(f"Fidelity F must be in [0,1], got {F}")
    if tau_R <= 0:
        raise ValueError(f"Relaxation timescale τ_R must be positive, got {tau_R}")
    if tau_0 <= 0:
        raise ValueError(f"Reference timescale τ_0 must be positive, got {tau_0}")

    # Compute components
    import math

    S_contribution = S
    F_contribution = 1.0 - F
    tau_damping = math.exp(-tau_R / tau_0)

    # Total collapse potential
    phi_collapse = S * (1.0 - F) * tau_damping

    # Classify regime
    if phi_collapse < 0.01:
        regime = "Minimal"
    elif phi_collapse < 0.1:
        regime = "Active"
    else:
        regime = "Critical"

    return {
        "phi_collapse": phi_collapse,
        "S_contribution": S_contribution,
        "F_contribution": F_contribution,
        "tau_damping": tau_damping,
        "regime": regime,
    }


def main():
    """Test entropic collapse computation across regime spectrum."""
    print("=" * 70)
    print("GCD CLOSURE TEST: Entropic Collapse")
    print("=" * 70)
    print()

    # Test cases spanning regime spectrum
    test_cases = [
        {
            "name": "Zero Entropy (S=0, deterministic precision)",
            "S": 0.0,
            "F": 1.0,
            "tau_R": 1.0,
        },
        {
            "name": "Stable Regime (low entropy, high fidelity)",
            "S": 0.056,
            "F": 0.99,
            "tau_R": 5.0,
        },
        {
            "name": "Watch Regime (moderate entropy, moderate fidelity)",
            "S": 0.20,
            "F": 0.85,
            "tau_R": 15.0,
        },
        {
            "name": "Collapse Boundary (high entropy, low fidelity)",
            "S": 0.50,
            "F": 0.50,
            "tau_R": 50.0,
        },
    ]

    for tc in test_cases:
        result = compute_entropic_collapse(S=tc["S"], F=tc["F"], tau_R=tc["tau_R"])

        print(f"{tc['name']}")
        print(f"  Inputs: S={tc['S']:.3f}, F={tc['F']:.3f}, τ_R={tc['tau_R']:.1f}")
        print(f"  Φ_collapse = {result['phi_collapse']:.6f}")
        print("  Components:")
        print(f"    - Entropy contribution: {result['S_contribution']:.3f}")
        print(f"    - Anti-fidelity (1-F): {result['F_contribution']:.3f}")
        print(f"    - Temporal damping: {result['tau_damping']:.6f}")
        print(f"  Regime: {result['regime']}")
        print()


if __name__ == "__main__":
    main()

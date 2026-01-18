"""
Resonance Time Constant Closure
Computes the resonance time constant from angular velocity and damping.
"""
from __future__ import annotations


def compute(omega: float, damping: float) -> dict:
    """
    Compute resonance time constant.
    
    tau_R = 1 / (damping * omega)
    
    Args:
        omega: Angular velocity in rad/s
        damping: Damping ratio (dimensionless)
        
    Returns:
        Dict with computed tau_R in seconds
    """
    if omega <= 0 or damping <= 0:
        raise ValueError("omega and damping must be positive")
    
    tau_R = 1.0 / (damping * omega)
    return {"tau_R": tau_R}


if __name__ == "__main__":
    # Example: 10 rad/s, 0.1 damping ratio
    result = compute(10.0, 0.1)
    print(f"Result: {result}")  # tau_R = 1 / (0.1 * 10) = 1.0 s
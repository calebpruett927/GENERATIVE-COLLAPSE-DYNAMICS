"""
Force from Angular Velocity Closure
Computes centripetal force from angular velocity, radius, and mass.
"""
from __future__ import annotations


def compute(omega: float, r: float, m: float) -> dict:
    """
    Compute centripetal force from angular velocity.
    
    F = m * omega^2 * r
    
    Args:
        omega: Angular velocity in rad/s
        r: Radius in meters
        m: Mass in kg
        
    Returns:
        Dict with computed force F in Newtons
    """
    F = m * (omega ** 2) * r
    return {"F": F}


if __name__ == "__main__":
    # Example: 10 rad/s, 0.5 m radius, 1 kg mass
    result = compute(10.0, 0.5, 1.0)
    print(f"Result: {result}")  # F = 1 * 100 * 0.5 = 50 N
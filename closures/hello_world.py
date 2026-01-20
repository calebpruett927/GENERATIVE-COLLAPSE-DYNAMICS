"""
Hello World Closure
Simple demonstration closure for UMCP validation.
"""

from __future__ import annotations


def compute(omega: float) -> dict:
    """
    Simple hello world computation.

    Args:
        omega: Angular velocity in rad/s

    Returns:
        Dict with computed force F
    """
    # Simple demonstration: F = omega (placeholder)
    F = omega
    return {"F": F}


if __name__ == "__main__":
    result = compute(10.0)
    print(f"Result: {result}")

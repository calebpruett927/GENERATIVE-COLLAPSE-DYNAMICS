"""Entanglement Closure — QM.INTSTACK.v1

Computes entanglement measures for bipartite quantum systems:
  - Concurrence (Wootters):  C ∈ [0,1], 0 = separable, 1 = maximally entangled
  - von Neumann entropy:  S_vN = -Tr(ρ log ρ) = H(½(1+√(1-C²)))
  - Bell-CHSH parameter:  S_CHSH ∈ [0, 2√2], > 2 violates Bell inequality
  - Negativity:  N = (||ρ^{T_A}||₁ - 1)/2

UMCP integration:
  ω = 1 - concurrence                      (low entanglement → high drift)
  F = concurrence                           (entanglement = fidelity to correlated state)
  C = bell_parameter / bell_quantum_limit   (CHSH as coupling proxy)
  S = S_vN / log(2)                         (normalized entropy)

Regime classification (entanglement_strength):
  Separable:  concurrence < 0.1
  Weak:       0.1 ≤ concurrence < 0.4
  Strong:     0.4 ≤ concurrence < 0.8
  Maximal:    concurrence ≥ 0.8

Cross-references:
  Contract:  contracts/QM.INTSTACK.v1.yaml
  Canon:     canon/qm_anchors.yaml
  Registry:  closures/registry.yaml (extensions.quantum_mechanics)
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Any, NamedTuple


class EntanglementRegime(StrEnum):
    """Regime based on entanglement strength."""

    SEPARABLE = "Separable"
    WEAK = "Weak"
    STRONG = "Strong"
    MAXIMAL = "Maximal"


class EntanglementResult(NamedTuple):
    """Result of entanglement computation."""

    concurrence: float  # Wootters concurrence [0,1]
    S_vN: float  # von Neumann entropy (bits)
    bell_parameter: float  # CHSH parameter
    negativity: float  # Entanglement negativity
    regime: str  # Regime classification


# ── Constants ────────────────────────────────────────────────────
BELL_CLASSICAL_LIMIT = 2.0
BELL_QUANTUM_LIMIT = 2.0 * math.sqrt(2.0)  # 2√2 ≈ 2.8284, Tsirelson bound

# Regime thresholds
THRESH_SEPARABLE = 0.1
THRESH_WEAK = 0.4
THRESH_STRONG = 0.8


def _classify_regime(concurrence: float) -> EntanglementRegime:
    """Classify entanglement strength regime."""
    if concurrence < THRESH_SEPARABLE:
        return EntanglementRegime.SEPARABLE
    if concurrence < THRESH_WEAK:
        return EntanglementRegime.WEAK
    if concurrence < THRESH_STRONG:
        return EntanglementRegime.STRONG
    return EntanglementRegime.MAXIMAL


def _concurrence_from_eigenvalues(eigenvalues: list[float]) -> float:
    """Compute concurrence from density matrix eigenvalues of ρ·ρ̃.

    For a 2-qubit system with eigenvalues λ₁ ≥ λ₂ ≥ λ₃ ≥ λ₄:
      C = max(0, √λ₁ - √λ₂ - √λ₃ - √λ₄)

    If only 2 eigenvalues provided (reduced density matrix),
    approximate concurrence from purity.
    """
    if len(eigenvalues) >= 4:
        sq = sorted([math.sqrt(abs(ev)) for ev in eigenvalues], reverse=True)
        return max(0.0, sq[0] - sq[1] - sq[2] - sq[3])
    if len(eigenvalues) == 2:
        # For 2-qubit: C = 2 * |λ₁ - λ₂| when eigenvalues are of reduced ρ
        # More precisely: C = 2√(λ₁·λ₂) for maximally entangled subspace
        p1, p2 = abs(eigenvalues[0]), abs(eigenvalues[1])
        return min(1.0, 2.0 * math.sqrt(p1 * p2))
    return 0.0


def _von_neumann_entropy(eigenvalues: list[float]) -> float:
    """Compute von Neumann entropy S = -Σ λᵢ log₂ λᵢ.

    Returns entropy in bits. For a pure state: S = 0.
    For maximally mixed 2-qubit: S = 1 bit.
    """
    s = 0.0
    for ev in eigenvalues:
        if ev > 1e-15:
            s -= ev * math.log2(ev)
    return s


def _negativity_from_eigenvalues(eigenvalues: list[float]) -> float:
    """Compute negativity from partial-transpose eigenvalues.

    N = (||ρ^{T_A}||₁ - 1) / 2 = Σ |min(0, λᵢ)|
    Positive eigenvalues contribute 0; negative eigenvalues = entanglement.
    """
    return sum(abs(ev) for ev in eigenvalues if ev < 0)


def compute_entanglement(
    rho_eigenvalues: list[float],
    bell_correlations: list[float] | None = None,
) -> dict[str, Any]:
    """Compute entanglement measures for UMCP validation.

    Parameters
    ----------
    rho_eigenvalues : list[float]
        Eigenvalues of the density matrix (or ρ·ρ̃ for concurrence).
        For 2-qubit: 4 eigenvalues of ρ·ρ̃.
        For reduced ρ: 2 eigenvalues of reduced density matrix.
    bell_correlations : list[float] | None
        If provided, [E(a,b), E(a,b'), E(a',b), E(a',b')] correlators
        for CHSH computation: S = |E(a,b) + E(a,b') + E(a',b) - E(a',b')|

    Returns
    -------
    dict with keys: concurrence, S_vN, bell_parameter, negativity, regime
    """
    # Concurrence
    conc = _concurrence_from_eigenvalues(rho_eigenvalues)

    # von Neumann entropy (use reduced-system eigenvalues)
    if len(rho_eigenvalues) == 2:
        s_vn = _von_neumann_entropy(rho_eigenvalues)
    else:
        # For 4 eigenvalues, compute reduced ρ entropy from concurrence
        # S(ρ_A) = H(½(1+√(1-C²))) where H is binary entropy
        x = 0.5 * (1.0 + math.sqrt(max(0.0, 1.0 - conc**2)))
        if 0 < x < 1:
            s_vn = -x * math.log2(x) - (1 - x) * math.log2(1 - x)
        elif x <= 0 or x >= 1:
            s_vn = 0.0
        else:
            s_vn = 0.0

    # Bell-CHSH parameter
    if bell_correlations and len(bell_correlations) >= 4:
        e_ab, e_ab2, e_a2b, e_a2b2 = bell_correlations[:4]
        bell_param = abs(e_ab + e_ab2 + e_a2b - e_a2b2)
    else:
        # Estimate from concurrence: maximal violation → S = 2√2·C for pure states
        bell_param = min(BELL_QUANTUM_LIMIT, 2.0 * math.sqrt(1.0 + conc**2))

    # Negativity (approximate from concurrence for 2-qubit systems)
    # For pure states: N = C/2
    neg = conc / 2.0

    regime = _classify_regime(conc)

    return {
        "concurrence": round(conc, 6),
        "S_vN": round(s_vn, 6),
        "bell_parameter": round(bell_param, 6),
        "negativity": round(neg, 6),
        "regime": regime.value,
    }


# ── Self-test ────────────────────────────────────────────────────
if __name__ == "__main__":
    # Maximally entangled Bell state |Φ⁺⟩: ρ̃ eigenvalues = [1, 0, 0, 0]
    result = compute_entanglement(
        rho_eigenvalues=[1.0, 0.0, 0.0, 0.0],
    )
    print(
        f"Bell |Φ⁺⟩: C={result['concurrence']:.4f} S_vN={result['S_vN']:.4f} "
        f"Bell={result['bell_parameter']:.4f} N={result['negativity']:.4f} "
        f"regime={result['regime']}"
    )
    assert result["regime"] == "Maximal"

    # Separable state: ρ eigenvalues = [0.5, 0.5, 0.0, 0.0]
    result = compute_entanglement(
        rho_eigenvalues=[0.25, 0.25, 0.25, 0.25],
    )
    print(
        f"Separable: C={result['concurrence']:.4f} S_vN={result['S_vN']:.4f} "
        f"Bell={result['bell_parameter']:.4f} N={result['negativity']:.4f} "
        f"regime={result['regime']}"
    )
    assert result["regime"] == "Separable"

    # Partially entangled: reduced ρ eigenvalues [0.85, 0.15]
    result = compute_entanglement(
        rho_eigenvalues=[0.85, 0.15],
    )
    print(
        f"Partial:   C={result['concurrence']:.4f} S_vN={result['S_vN']:.4f} "
        f"Bell={result['bell_parameter']:.4f} N={result['negativity']:.4f} "
        f"regime={result['regime']}"
    )

    # Bell test with explicit CHSH correlations (maximal violation)
    result = compute_entanglement(
        rho_eigenvalues=[1.0, 0.0, 0.0, 0.0],
        bell_correlations=[0.7071, 0.7071, 0.7071, -0.7071],
    )
    print(f"CHSH max:  C={result['concurrence']:.4f} S_CHSH={result['bell_parameter']:.4f} regime={result['regime']}")
    assert result["bell_parameter"] > 2.0, "Bell violation expected"

    print("✓ entanglement self-test passed")

# Mathematical Architecture

**Version**: 1.0.0  
**Status**: Reference Documentation  
**Prerequisites**: [KERNEL_SPECIFICATION.md](KERNEL_SPECIFICATION.md), [INFRASTRUCTURE_GEOMETRY.md](INFRASTRUCTURE_GEOMETRY.md)  
**See Also**: [src/umcp/tau_r_star.py](src/umcp/tau_r_star.py) for the τ_R* thermodynamic diagnostic — phase classification, Fisher Information geometry, budget surplus/deficit, trapping analysis, and testable predictions derived from the budget identity  
**See Also**: [KERNEL_SPECIFICATION.md §5](KERNEL_SPECIFICATION.md) for measured prediction scorecard, outlier analysis, rederived principle comparisons, and seam-derived constants

---

## Overview

This document describes the explicit and implicit matrix and prime structures underlying UMCP. These structures are not decorative—they are the algebraic machinery that makes the protocol auditable, portable, and verifiable.

---

## Part I: Matrix Structures

### 1. The Trace Matrix (Primary Data Object)

At runtime, the bounded trace Ψ(t) ∈ [0,1]ⁿ across a window is naturally represented as a time×channel array:

```
Ψ = | c₁(0)  c₂(0)  ⋯  cₙ(0) |
    | c₁(1)  c₂(1)  ⋯  cₙ(1) |
    |   ⋮      ⋮    ⋱    ⋮   |
    | c₁(T)  c₂(T)  ⋯  cₙ(T) |
```

**Dimensions**: T×n (time steps × channels)

**Operations**:
- Row extraction: Ψ[t,:] = state at time t
- Column extraction: Ψ[:,i] = channel i time series
- Weighted row sum: F(t) = Ψ[t,:] · w

---

### 2. Diagonal Weight Matrix W = diag(w)

The weight vector w implicitly defines a diagonal matrix:

```
W = | w₁  0   ⋯   0  |
    | 0   w₂  ⋯   0  |
    | ⋮    ⋮   ⋱   ⋮  |
    | 0   0   ⋯   wₙ |
```

**Properties**:
- Σwᵢ = 1 (normalization constraint)
- wᵢ ≥ 0 (non-negativity)
- det(W) = ∏wᵢ (product of weights)

**Usage**:
- Fidelity: F = 1ᵀWc = Σwᵢcᵢ
- Weighted inner product: ⟨x,y⟩_w = xᵀWy
- Log-integrity: κ = Σwᵢln(cᵢ) = 1ᵀW·ln(c)

---

### 3. Covariance Matrix V = Cov(c)

For uncertainty propagation via delta-method:

```
Var(T(c)) ≈ ∇Tᵀ V ∇T
```

**Diagonal Case** (independent channels):
```
V = diag(σ₁², σ₂², ..., σₙ²)
```

**Full Case** (correlated channels):
```
V = | Cov(c₁,c₁)  Cov(c₁,c₂)  ⋯  Cov(c₁,cₙ) |
    | Cov(c₂,c₁)  Cov(c₂,c₂)  ⋯  Cov(c₂,cₙ) |
    |     ⋮           ⋮       ⋱       ⋮      |
    | Cov(cₙ,c₁)  Cov(cₙ,c₂)  ⋯  Cov(cₙ,cₙ) |
```

**Implementation**: See `src/umcp/uncertainty.py`

---

### 4. Hessian Matrix H (Second-Order Sensitivity)

For nonlinear budget models:

```
H = | ∂²Δκ/∂θ₁²      ∂²Δκ/∂θ₁∂θ₂  ⋯ |
    | ∂²Δκ/∂θ₂∂θ₁   ∂²Δκ/∂θ₂²    ⋯ |
    |      ⋮              ⋮        ⋱ |
```

**Key Property**: For affine budgets (UMA.INTSTACK), H = 0.

**Robustness Expansion**:
```
ΔT ≈ ∇Tᵀδθ + ½δθᵀHδθ + O(‖δθ‖³)
```

---

### 5. Implied Metric Matrix (from ‖·‖₂)

When τ_R uses Euclidean norm:

```
‖Ψ(t) - Ψ(s)‖₂² = (Ψ(t) - Ψ(s))ᵀ I (Ψ(t) - Ψ(s))
```

The identity matrix I is the metric tensor for Euclidean geometry. Alternative metrics would require explicit declaration.

---

### 6. Stokes Rotation Matrix (Polarimetry)

For linear polarization rotation:

```
R(θ) = | cos(2θ)   sin(2θ) |
       | -sin(2θ)  cos(2θ) |
```

Acting on (Q, U) Stokes parameters. The factor of 2 reflects polarization's headless vector nature.

---

## Part II: Prime Number Structures

### 1. mod 97: The Verification Prime

**Usage**: SS1m edition triads (C₁, C₂, C₃)

**Formulas**:
```
C₁ = (P + F) mod 97
C₂ = (P + 2F + 3T + 5E + 7R) mod 97
C₃ = (P × F + T) mod 97
```

**Why 97?**
- Largest 2-digit prime → human-readable checksums
- Z₉₇ is a field → all arithmetic operations well-defined
- Prime coefficients {2,3,5,7} are coprime to 97 → linear independence

**Implementation**: See `src/umcp/ss1m_triad.py`

---

### 2. mod 32: Encoding (Non-Prime)

**Usage**: Crockford Base32 for compact EID strings

**Properties**:
- 32 = 2⁵ → 5 bits per character
- Not prime → no field structure
- Chosen for efficiency, not verification

---

### 3. mod 2: Boolean Foundation

**Usage**: Discretio (distinction) in Collapse Metric Protocol

```
Discretio(a,b) = (a + b) mod 2 = a ⊕ b (XOR)
```

**Properties**:
- Z₂ is the smallest field
- Connects continuous Ψ∈[0,1] to discrete {0,1} via thresholding

---

### 4. Prime Exponent p = 3

**Usage**: Cubic drift penalty in closure

```
Γ(ω) = ω³
```

**Properties**:
- Odd power → preserves sign
- Superlinear → punishes large drifts more than proportionally
- Prime exponent → no factorization into smaller powers

---

### 5. Prime Period p = 7 (Recurrence)

**Usage**: Collapse Algebra recurrence exemplar

**Significance**:
- No proper subperiods (7 is prime)
- Z₇ is a field
- Primitive roots exist → can generate all residues

---

## Part III: Algebraic Invariants

### The Fundamental Accounting Identity

Seam certification requires:

```
Δκ_budget = R · τ_R - (D_ω + D_C)
s = Δκ_budget - Δκ_ledger
PASS ⟺ |s| ≤ tol_seam
```

**Matrix Form**:
```
| R  |   | τ_R |
|-1  | · |  1  | = Δκ_budget
|-1  |   |  1  |
```

This is a linear system that must balance.

---

### The Projection Structure

Kernel computation as projection:

```
| F     |       | Σwᵢcᵢ                    |
| ω     |       | 1 - F                     |
| S     | = P(Ψ,w) = | Σwᵢh(cᵢ)           |
| C     |       | 2σ(c)/0.5                 |
| κ     |       | Σwᵢln(cᵢ)                |
```

---

### AM-GM as Matrix Inequality

The fundamental inequality IC ≤ F is:

```
exp(wᵀ ln(c)) ≤ wᵀc
```

Equality iff c is constant (homogeneous coordinates).

---

## Part IV: Implementation Summary

| Structure | Module | Purpose |
|-----------|--------|---------|
| Trace matrix | Runtime data | Primary data object |
| Weight matrix W | `kernel_optimized.py` | Implicit in all weighted operations |
| Covariance V | `uncertainty.py` | Uncertainty propagation |
| Gradients ∇ | `uncertainty.py` | Sensitivity analysis |
| Triad checksums | `ss1m_triad.py` | Edition verification |
| Base32 encoding | `ss1m_triad.py` | Compact EID format |

---

## References

- [KERNEL_SPECIFICATION.md](KERNEL_SPECIFICATION.md) — Formal definitions and lemmas
- [INFRASTRUCTURE_GEOMETRY.md](INFRASTRUCTURE_GEOMETRY.md) — Three-layer architecture
- [COMPUTATIONAL_OPTIMIZATIONS.md](COMPUTATIONAL_OPTIMIZATIONS.md) — Optimization lemmas
- [src/umcp/ss1m_triad.py](src/umcp/ss1m_triad.py) — Triad checksum implementation
- [src/umcp/uncertainty.py](src/umcp/uncertainty.py) — Uncertainty propagation

---

**Algebraic completeness**: The matrix + prime structures form a closed algebra where every verification operation has an inverse or adjoint. This is why the framework can detect subtle violations—the algebra doesn't admit "narrative rescue."

# Casepack: quantum_mechanics_complete

## Overview

Comprehensive quantum mechanics casepack covering the six foundational pillars
of QM, each mapped to UMCP invariants through the Generative Collapse Dynamics
(GCD) framework.

**Quantum mechanics is the original archetype of GCD**: wavefunction collapse
upon measurement is the literal instantiation of the core axiom — *"Collapse
is generative."* The indefinite superposition resolves into a definite outcome,
generating observable reality from quantum potentiality.

## Contract

- **ID**: `QM.INTSTACK.v1`
- **Parent**: `GCD.INTSTACK.v1` (Tier-1)
- **Canon**: `canon/qm_anchors.yaml`

## Subdomains & Closures

| # | Closure | Physics | QM → GCD Mapping |
|---|---------|---------|-------------------|
| 1 | `wavefunction_collapse` | Born rule P = \|⟨φ\|ψ⟩\|² | δP → ω (drift), fidelity → F |
| 2 | `entanglement` | Bell-CHSH, concurrence | concurrence → C (coupling) |
| 3 | `tunneling` | T ≈ exp(−2κL) | transmission → F, κ → kappa |
| 4 | `harmonic_oscillator` | E_n = ℏω(n + ½) | energy deviation → ω |
| 5 | `spin_measurement` | Stern-Gerlach, Zeeman | spin fidelity → F |
| 6 | `uncertainty_principle` | Δx·Δp ≥ ℏ/2 | R < 1 → NONCONFORMANT |

## Experiments (30 total)

### Wavefunction Collapse (QM01–QM05)
- **QM01**: Perfect spin-½ measurement (|↑⟩ → 100% up)
- **QM02**: 50/50 superposition with small statistical deviation
- **QM03**: Three-state equal superposition
- **QM04**: Decoherent measurement (significant Born deviation)
- **QM05**: Four-state highly decoherent

### Entanglement (QM06–QM10)
- **QM06**: Bell state |Φ⁺⟩ (maximal entanglement, CHSH = 2√2)
- **QM07**: Bell state |Ψ⁻⟩ (maximal, alternate correlations)
- **QM08**: Partially entangled (reduced ρ eigenvalues 0.85/0.15)
- **QM09**: Separable state (product state, no entanglement)
- **QM10**: Werner mixed state (partial entanglement + noise)

### Quantum Tunneling (QM11–QM15)
- **QM11**: Electron through thin barrier (0.1 nm, moderate T)
- **QM12**: Electron through medium barrier (0.3 nm, suppressed)
- **QM13**: Electron through thick barrier (1.0 nm, opaque)
- **QM14**: Alpha decay proxy (proton mass, high barrier)
- **QM15**: Classically allowed passage (E > V₀)

### Harmonic Oscillator (QM16–QM20)
- **QM16**: Ground state n=0, ℏω=0.5 eV (exact)
- **QM17**: First excited n=1 (exact)
- **QM18**: n=3 with slight perturbation
- **QM19**: Coherent state |α⟩ with α=2.236
- **QM20**: Squeezed vacuum state (r=0.5)

### Spin Measurement (QM21–QM25)
- **QM21**: Electron spin-up in 1T field (perfect)
- **QM22**: Electron spin-down in 1T field (perfect)
- **QM23**: Spin-1 particle, m=0, in 5T field
- **QM24**: Proton spin-up in 3T field (g=5.5857)
- **QM25**: Noisy spin-½ measurement (imperfect fidelity)

### Heisenberg Uncertainty (QM26–QM30)
- **QM26**: Hydrogen 1s ground state (near minimum)
- **QM27**: Thermal electron (moderate uncertainty)
- **QM28**: Near-minimum uncertainty wavepacket
- **QM29**: Classical ball (macroscopic, R >> 1)
- **QM30**: Squeezed light quadrature

## GCD Connections

The deepest connection between QM and GCD lies in the **measurement problem**:
1. **Before measurement**: the system is in superposition — indeterminate
2. **Measurement (collapse)**: the boundary (apparatus) interacts with the interior (quantum state)
3. **After measurement**: a definite outcome is generated — *generative collapse*

This maps exactly to the GCD axioms:
- **AX-0 (Collapse is generative)**: Measurement generates classical outcomes
- **AX-1 (Boundary defines interior)**: Measurement basis defines possible outcomes
- **AX-2 (Entropy measures determinacy)**: von Neumann entropy S_vN measures mixedness

## Running

```bash
# Generate expected outputs
python casepacks/quantum_mechanics_complete/generate_expected.py

# Validate
umcp validate casepacks/quantum_mechanics_complete
```

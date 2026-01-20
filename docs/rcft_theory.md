# Recursive Collapse Field Theory (RCFT) - Tier-2 Overlay

**Quick Reference:** [GLOSSARY.md](../GLOSSARY.md#tier-2-rcft-overlay-extensions) | [SYMBOL_INDEX.md](../SYMBOL_INDEX.md#tier-2-rcft-extension-symbols) | [canon/rcft_anchors.yaml](../canon/rcft_anchors.yaml)

## Overview

RCFT is a Tier-2 overlay that extends the Generative Collapse Dynamics (GCD) framework with geometric and topological analysis capabilities. It operates under the principle of **augmentation without override**: all GCD Tier-1 invariants remain frozen and unchanged, while RCFT adds three new dimensions of analysis.

## Theoretical Foundation

### Three Core Principles

**P-RCFT-0: Augmentation, Never Override**  
RCFT adds analytical dimensions without modifying GCD's mathematical foundation. All 13 Tier-1 symbols (`ω`, `F`, `S`, `C`, `τ_R`, `κ`, `IC`, `IC_min`, `I`, `E_potential`, `Φ_collapse`, `Φ_gen`, `R`) remain frozen.

**P-RCFT-1: Recursion Reveals Hidden Structure**  
Collapse events exhibit self-similar patterns across scales. RCFT quantifies these recursive structures through fractal dimension and field memory analysis.

**P-RCFT-2: Fields Carry Collapse Memory**  
The collapse field `Ψ` encodes the history of prior collapse events. Recursive analysis reveals how past collapses influence future dynamics through exponentially-decaying field memory.

## RCFT Metrics

### 1. Fractal Dimension (`D_fractal`)

**Purpose**: Quantify the geometric complexity of collapse trajectories.

**Method**: Box-counting algorithm on trajectory embedded in (ω, S, C) space.

**Formula**:
$$D_f = \lim_{\epsilon \to 0} \frac{\log N(\epsilon)}{\log(1/\epsilon)}$$

where $N(\epsilon)$ is the number of boxes of size $\epsilon$ needed to cover the trajectory.

**Regimes**:
- **Smooth** ($D_f < 1.2$): Near-deterministic, low-complexity trajectories (e.g., zero entropy states)
- **Wrinkled** ($1.2 \leq D_f < 1.8$): Moderate complexity with some recursive structure
- **Turbulent** ($D_f \geq 1.8$): High complexity, space-filling behavior

**Interpretation**: Higher fractal dimensions indicate more complex collapse dynamics. A point trajectory (zero entropy) has $D_f = 0$, a smooth curve has $D_f \approx 1$, and chaotic attractors can approach $D_f = 2$ or higher.

**Mathematical Identity**:
$$1 \leq D_f \leq 3$$

The fractal dimension is bounded by the 3D embedding space (ω, S, C).

### 2. Recursive Field (`Ψ_recursive`)

**Purpose**: Measure the cumulative memory of past collapse events.

**Method**: Exponential decay summation over collapse history.

**Formula**:
$$\Psi_r = \sum_{n=1}^{\infty} \alpha^n \cdot \Psi_n$$

where:
- $\Psi_n = \sqrt{S_n^2 + C_n^2} \cdot (1 - F_n)$ is the field strength at timestep $n$
- $\alpha \in (0, 1)$ is the exponential decay factor (default: 0.8)
- Convergence guaranteed for $0 < \alpha < 1$

**Regimes**:
- **Dormant** ($\Psi_r < 0.1$): Minimal field memory, recent collapses negligible
- **Active** ($0.1 \leq \Psi_r < 1.0$): Significant memory, past influences present
- **Resonant** ($\Psi_r \geq 1.0$): Strong memory, past collapses dominate dynamics

**Interpretation**: $\Psi_r$ quantifies how much "momentum" the system has from prior collapses. High values indicate persistent effects that cannot be ignored in downstream analysis.

**Mathematical Identity**:
$$\Psi_r = \sum_{n=1}^{\infty} \alpha^n \cdot \Psi_n \quad \text{converges for } 0 < \alpha < 1$$

### 3. Resonance Pattern (`λ_pattern`, `Θ_phase`)

**Purpose**: Identify oscillatory structures and phase coherence in field dynamics.

**Method**: Fast Fourier Transform (FFT) analysis of field time series.

**Formulas**:
- **Wavelength**: $\lambda_p = \frac{2\pi}{k_{dominant}}$ where $k_{dominant}$ is the dominant frequency
- **Phase**: $\Theta = \arctan\left(\frac{\text{Im}(F_{dominant})}{\text{Re}(F_{dominant})}\right)$

**Pattern Types**:
- **Standing** (phase variance $< 0.1$): Coherent, stationary oscillation (constant fields → infinite wavelength)
- **Mixed** ($0.1 \leq$ phase variance $\leq 0.5$): Combination of standing and traveling waves
- **Traveling** (phase variance $> 0.5$): Propagating oscillations, low phase coherence

**Interpretation**: 
- $\lambda_p$ measures the characteristic length scale of oscillations. Infinite wavelength indicates constant fields.
- $\Theta$ measures the phase angle. Low phase variance indicates coherent oscillations (standing waves).

**Mathematical Identities**:
- $\Theta \in [0, 2\pi)$ (phase periodicity)
- $\lambda_p > 0$ (or infinite for constant fields)

## Tier Hierarchy

```
UMCP (Tier-0: Core canonical symbols)
  ↓
GCD (Tier-1: 13 frozen symbols + 4 closures)
  ↓
RCFT (Tier-2: 4 new symbols + 3 closures)
```

**Frozen Tier-1 Symbols** (cannot be modified by RCFT):
- Core: `ω`, `F`, `S`, `C`, `τ_R`, `κ`, `IC`, `IC_min`, `I`
- GCD: `E_potential`, `Φ_collapse`, `Φ_gen`, `R`

**New Tier-2 Symbols** (can be extended by future Tier-3):
- `D_fractal`: Fractal dimension
- `Psi_recursive`: Recursive field strength
- `lambda_pattern`: Resonance wavelength
- `Theta_phase`: Phase angle

## Use Cases

### When to Use RCFT

1. **Trajectory Analysis**: Quantify geometric complexity of collapse paths
2. **Memory Effects**: Measure how past collapses influence current state
3. **Oscillation Detection**: Identify periodic or quasi-periodic behavior
4. **Regime Classification**: Extend GCD regime analysis with topological metrics

### When NOT to Use RCFT

1. **Simple Zero-Entropy States**: RCFT metrics trivialize (D_f=0, Ψ_r=0, λ=∞)
2. **Single-Timestep Analysis**: RCFT requires time series data (recursive field, patterns)
3. **Pure GCD Analysis**: If only GCD metrics needed, use GCD directly (RCFT adds overhead)

## Integration with GCD

RCFT is designed for seamless integration:

- **Backward Compatible**: All GCD functionality preserved
- **Optional Enhancement**: Can be used alongside or instead of pure GCD
- **Consistent Validation**: Uses same contract schema, seam receipt structure
- **Registry Integration**: RCFT closures registered alongside GCD closures

## Computational Notes

### Fractal Dimension
- **Input**: Trajectory points (N × 3 array of [ω, S, C])
- **Algorithm**: Box-counting with logarithmic box size sampling
- **Complexity**: O(N · log(1/ε_min)) where ε_min is minimum box size
- **Minimum Points**: At least 10 points recommended for stable estimation

### Recursive Field
- **Input**: Time series of (S, C, F) values
- **Algorithm**: Exponential decay summation with convergence check
- **Complexity**: O(N) where N is series length (can truncate at α^N < tolerance)
- **Maximum Depth**: Default 100 iterations (configurable)

### Resonance Pattern
- **Input**: Field time series (length N)
- **Algorithm**: FFT + peak detection
- **Complexity**: O(N log N) via FFT
- **Recommended Length**: Power of 2 (default 512 points) for optimal FFT performance

## Validation

RCFT includes comprehensive validation:

- **Canon**: 297-line specification in `canon/rcft_anchors.yaml`
- **Contract**: Complete Tier-2 contract in `contracts/RCFT.INTSTACK.v1.yaml`
- **Casepack**: Example with zero-entropy state showing all 7 closures
- **Tests**: 56 tests covering canon, closures, contract, tier layering

All 221 tests in the suite pass (100% success rate), including 142 original tests (backward compatibility) and 56 new RCFT tests.

## References

- **Canon**: [canon/rcft_anchors.yaml](../canon/rcft_anchors.yaml)
- **Contract**: [contracts/RCFT.INTSTACK.v1.yaml](../contracts/RCFT.INTSTACK.v1.yaml)
- **Closures**:
  - [closures/rcft/fractal_dimension.py](../closures/rcft/fractal_dimension.py)
  - [closures/rcft/recursive_field.py](../closures/rcft/recursive_field.py)
  - [closures/rcft/resonance_pattern.py](../closures/rcft/resonance_pattern.py)
- **Examples**: [casepacks/rcft_complete/](../casepacks/rcft_complete/)
- **Tests**:
  - [tests/test_110_rcft_canon.py](../tests/test_110_rcft_canon.py)
  - [tests/test_111_rcft_closures.py](../tests/test_111_rcft_closures.py)
  - [tests/test_112_rcft_contract.py](../tests/test_112_rcft_contract.py)
  - [tests/test_113_rcft_tier2_layering.py](../tests/test_113_rcft_tier2_layering.py)

## Future Extensions

RCFT is designed to support future Tier-3 overlays:

- Tier-2 symbols (`D_fractal`, `Psi_recursive`, `lambda_pattern`, `Theta_phase`) can be extended (but not overridden)
- Additional closures can be added at Tier-3
- New regime classifications can augment RCFT regimes
- Mathematical identities can be extended with Tier-3 constraints

The principle remains: **augment, never override**.

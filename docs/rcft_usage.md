# RCFT Usage Guide

This guide demonstrates how to use the Recursive Collapse Field Theory (RCFT) closures in practice.

## Quick Start

### Prerequisites

```bash
# Install required packages
pip install numpy scipy
```

### Basic Example: Zero Entropy State

```python
import numpy as np
from closures.rcft.fractal_dimension import compute_fractal_dimension
from closures.rcft.recursive_field import compute_recursive_field_from_energy
from closures.rcft.resonance_pattern import compute_resonance_pattern

# Zero entropy state (perfect fidelity)
c = 0.99999999
omega = 0.0
F = 1.0
S = 0.0
C = 0.0

# Fractal dimension of point trajectory
trajectory = np.array([[omega, S, C]])  # Single point
result_fractal = compute_fractal_dimension(trajectory)

print(f"Fractal Dimension: {result_fractal['D_fractal']:.6f}")
print(f"Regime: {result_fractal['regime']}")
# Output: D_fractal = 0.0 (point), Regime = "Smooth"

# Recursive field from energy series
E_series = [0.0]  # Zero energy
result_recursive = compute_recursive_field_from_energy(E_series, alpha_decay=0.8)

print(f"Recursive Field: {result_recursive['Psi_recursive']:.6f}")
print(f"Regime: {result_recursive['regime']}")
# Output: Psi_recursive = 0.0, Regime = "Dormant"

# Resonance pattern from constant field
field_series = np.ones(512)  # Constant field
dt = 1.0
result_pattern = compute_resonance_pattern(field_series, dt)

print(f"Wavelength: {result_pattern['lambda_pattern']}")
print(f"Phase: {result_pattern['Theta_phase']:.6f}")
print(f"Pattern Type: {result_pattern['pattern_type']}")
# Output: lambda = inf, Theta = 0.0, Pattern = "Standing"
```

## Advanced Examples

### 1. Fractal Dimension: Analyzing Trajectory Complexity

```python
import numpy as np
from closures.rcft.fractal_dimension import compute_fractal_dimension, compute_trajectory_from_invariants

# Generate trajectory from GCD invariants
# Linear drift: omega increases linearly
invariants = {
    "omega": np.linspace(0, 0.1, 20),
    "S": np.zeros(20),
    "C": np.zeros(20)
}
trajectory = compute_trajectory_from_invariants(invariants)

# Compute fractal dimension
result = compute_fractal_dimension(trajectory)

print(f"Fractal Dimension: {result['D_fractal']:.4f}")
print(f"Regime: {result['regime']}")
print(f"R-squared (fit quality): {result['r_squared']:.4f}")
print(f"Number of box sizes: {len(result['box_counts'])}")

# Expected: D_f ≈ 1.0 (smooth line), Regime = "Smooth"
```

### 2. Recursive Field: Quantifying Collapse Memory

```python
import numpy as np
from closures.rcft.recursive_field import compute_recursive_field

# Simulate entropy decay over time
N = 50
S_series = 0.5 * np.exp(-np.arange(N) / 10)  # Exponential decay
C_series = np.zeros(N)
F_series = 1 - 0.5 * np.exp(-np.arange(N) / 10)

# Compute recursive field strength
result = compute_recursive_field(
    S_series=S_series,
    C_series=C_series,
    F_series=F_series,
    alpha_decay=0.8
)

print(f"Recursive Field Strength: {result['Psi_recursive']:.4f}")
print(f"Regime: {result['regime']}")
print(f"Convergence Achieved: {result['convergence_achieved']}")
print(f"First 5 contributions: {result['contributions'][:5]}")

# Interpretation:
# - High Psi_r indicates strong memory effects
# - Contributions show how each timestep influences total
```

### 3. Resonance Pattern: Detecting Oscillations

```python
import numpy as np
from closures.rcft.resonance_pattern import compute_resonance_pattern, compute_multi_field_resonance

# Generate sinusoidal oscillation
N = 512
t = np.linspace(0, 10 * np.pi, N)
field_series = np.sin(t)
dt = t[1] - t[0]

# Analyze resonance pattern
result = compute_resonance_pattern(field_series, dt)

print(f"Wavelength: {result['lambda_pattern']:.4f}")
print(f"Phase: {result['Theta_phase']:.4f} rad")
print(f"Pattern Type: {result['pattern_type']}")
print(f"Phase Coherence: {result['phase_coherence']:.4f}")

# Expected: lambda ≈ 2π, Pattern = "Standing" (coherent oscillation)

# Multi-field analysis
omega_series = np.sin(t)
S_series = np.cos(t)
C_series = 0.5 * np.sin(2 * t)

result_multi = compute_multi_field_resonance(
    omega_series=omega_series,
    S_series=S_series,
    C_series=C_series,
    dt=dt
)

print(f"Omega wavelength: {result_multi['omega_lambda']:.4f}")
print(f"S wavelength: {result_multi['S_lambda']:.4f}")
print(f"C wavelength: {result_multi['C_lambda']:.4f}")
print(f"Phase alignment: {result_multi['phase_alignment']:.4f}")
```

## Integration with GCD Closures

### Full Pipeline: GCD + RCFT

```python
import numpy as np
from closures.gcd.energy_potential import compute_energy_potential
from closures.gcd.entropic_collapse import compute_entropic_collapse
from closures.gcd.generative_flux import compute_generative_flux
from closures.gcd.field_resonance import compute_field_resonance
from closures.rcft.fractal_dimension import compute_fractal_dimension, compute_trajectory_from_invariants
from closures.rcft.recursive_field import compute_recursive_field
from closures.rcft.resonance_pattern import compute_resonance_pattern

# Simulate time series of GCD invariants
N = 100
omega = np.random.normal(0, 0.01, N)
S = np.random.uniform(0, 0.1, N)
C = np.random.uniform(0, 0.05, N)
F = 1 - omega
tau_R = np.full(N, 10.0)
kappa = np.log(1 / (1 - np.abs(omega)))
IC = np.exp(kappa)

# Run all GCD closures (example for first timestep)
energy = compute_energy_potential(omega[0], S[0], C[0])
collapse = compute_entropic_collapse(S[0], F[0], tau_R[0])
flux = compute_generative_flux(kappa[0], IC[0], C[0])
resonance = compute_field_resonance(omega[0], S[0], C[0])

print("=== GCD Metrics (t=0) ===")
print(f"Energy: {energy['E_potential']:.6f} ({energy['regime']})")
print(f"Collapse: {collapse['phi_collapse']:.6f} ({collapse['regime']})")
print(f"Flux: {flux['phi_gen']:.6f} ({flux['regime']})")
print(f"Resonance: {resonance['resonance']:.6f} ({resonance['regime']})")

# Run all RCFT closures
invariants = {"omega": omega, "S": S, "C": C}
trajectory = compute_trajectory_from_invariants(invariants)

fractal = compute_fractal_dimension(trajectory)
recursive = compute_recursive_field(S, C, F, alpha_decay=0.8)
pattern = compute_resonance_pattern(omega, dt=1.0)

print("\n=== RCFT Metrics ===")
print(f"Fractal Dimension: {fractal['D_fractal']:.4f} ({fractal['regime']})")
print(f"Recursive Field: {recursive['Psi_recursive']:.4f} ({recursive['regime']})")
print(f"Wavelength: {pattern['lambda_pattern']:.4f} ({pattern['pattern_type']})")
```

## Parameter Tuning

### Fractal Dimension

```python
# Control box size sampling
result = compute_fractal_dimension(
    trajectory=trajectory,
    eps_values=None,  # Auto-generate (recommended)
    eps_min_factor=0.01,  # Minimum box size = 1% of trajectory extent
    n_epsilon=20  # Number of box sizes to sample
)

# For noisy data, use fewer box sizes
result_noisy = compute_fractal_dimension(trajectory, n_epsilon=10)
```

### Recursive Field

```python
# Adjust decay rate
result_fast_decay = compute_recursive_field(
    S, C, F,
    alpha_decay=0.5  # Faster decay (less memory)
)

result_slow_decay = compute_recursive_field(
    S, C, F,
    alpha_decay=0.95,  # Slower decay (more memory)
    max_depth=200  # Allow more iterations
)

# Energy-based variant (auto-computes Psi from E)
from closures.rcft.recursive_field import compute_recursive_field_from_energy

result_energy = compute_recursive_field_from_energy(
    E_series=[0.01, 0.02, 0.03],  # Energy time series
    alpha_decay=0.8
)
```

### Resonance Pattern

```python
# Control FFT parameters
result = compute_resonance_pattern(
    field_series=omega,
    dt=0.1,  # Time step (affects frequency resolution)
    n_fft=1024  # FFT size (must be power of 2)
)

# For longer series, use larger FFT
result_hires = compute_resonance_pattern(omega_long, dt=0.01, n_fft=2048)
```

## Error Handling

All RCFT closures include comprehensive error handling:

```python
from closures.rcft.fractal_dimension import compute_fractal_dimension

# Too few points
try:
    result = compute_fractal_dimension(np.array([[0, 0, 0]]))
except ValueError as e:
    print(f"Error: {e}")
    # Output: "Trajectory must have at least 2 points"

# Invalid decay parameter
from closures.rcft.recursive_field import compute_recursive_field

try:
    result = compute_recursive_field(S, C, F, alpha_decay=1.5)
except ValueError as e:
    print(f"Error: {e}")
    # Output: "alpha_decay must be in (0, 1) for convergence"
```

## Regime Interpretation

### Fractal Regimes

| Regime | D_fractal Range | Interpretation | Example |
|--------|----------------|----------------|---------|
| **Smooth** | < 1.2 | Low complexity, near-deterministic | Zero entropy, linear drift |
| **Wrinkled** | 1.2 - 1.8 | Moderate complexity | Noisy random walk |
| **Turbulent** | ≥ 1.8 | High complexity, chaotic | Space-filling attractor |

### Recursive Regimes

| Regime | Ψ_r Range | Interpretation | Action |
|--------|-----------|----------------|--------|
| **Dormant** | < 0.1 | Minimal memory | Safe to ignore history |
| **Active** | 0.1 - 1.0 | Significant memory | Consider past collapses |
| **Resonant** | ≥ 1.0 | Strong memory | Past dominates present |

### Pattern Types

| Type | Phase Variance | Interpretation | Example |
|------|---------------|----------------|---------|
| **Standing** | < 0.1 | Coherent oscillation | Constant field, pure sine wave |
| **Mixed** | 0.1 - 0.5 | Partial coherence | Multi-harmonic signal |
| **Traveling** | > 0.5 | Low coherence | Noisy, aperiodic signal |

## Performance Tips

1. **Trajectory sampling**: For large datasets, subsample trajectory points to ~1000 for fractal dimension
2. **Recursive depth**: Default 100 iterations sufficient for α ≤ 0.9
3. **FFT size**: Use power-of-2 for optimal performance (512, 1024, 2048)
4. **Parallel computation**: RCFT closures are independent, can be parallelized

## Common Pitfalls

1. **Single-point trajectory**: Fractal dimension requires at least 2 points
2. **Empty series**: Recursive field requires at least 1 timestep
3. **Constant field**: Pattern analysis returns λ = infinity (not an error)
4. **Convergence warnings**: Recursive field may not converge if α too close to 1

## References

- Theory: [docs/rcft_theory.md](rcft_theory.md)
- Canon: [canon/rcft_anchors.yaml](../canon/rcft_anchors.yaml)
- Contract: [contracts/RCFT.INTSTACK.v1.yaml](../contracts/RCFT.INTSTACK.v1.yaml)
- Examples: [casepacks/rcft_complete/](../casepacks/rcft_complete/)

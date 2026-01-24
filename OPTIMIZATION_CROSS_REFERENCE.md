# Optimization Cross-Reference Map

**Status**: Integration roadmap for computational optimizations  
**Created**: 2026-01-24  
**Purpose**: Document where and how to integrate the new optimization modules

---

## Module Summary

| Module | Location | Purpose | Optimizations |
|--------|----------|---------|---------------|
| `OptimizedKernelComputer` | [src/umcp/kernel_optimized.py](src/umcp/kernel_optimized.py) | Fast kernel computation | OPT-1,2,3,4,12,14,15 |
| `SeamChainAccumulator` | [src/umcp/seam_optimized.py](src/umcp/seam_optimized.py) | Efficient seam accounting | OPT-10,11 |
| `compute_utils` | [src/umcp/compute_utils.py](src/umcp/compute_utils.py) | Shared utilities | OPT-17,20 |
| `OptimizedReturnComputer` | [closures/tau_R_optimized.py](closures/tau_R_optimized.py) | Fast return time | OPT-7,8,9 |

---

## 🔴 HIGH PRIORITY: Core System Integration

### 1. validator.py - Root File Validator

**Current Location**: [src/umcp/validator.py](src/umcp/validator.py)

| Function | Current Implementation | Optimization Opportunity |
|----------|----------------------|-------------------------|
| `_validate_weights()` (L195-216) | Manual weight sum check | Use `compute_utils.normalize_weights()` with validation |
| `_validate_trace_bounds()` (L217-238) | Manual coordinate range check | Use `compute_utils.clip_coordinates()` for diagnostics |
| `_validate_invariant_identities()` (L239-274) | Manual F=1-ω, IC=exp(κ) check | Use `kernel_optimized.validate_kernel_bounds()` |

**Integration Points**:
```python
# In validator.py imports:
from umcp.compute_utils import validate_inputs, clip_coordinates
from umcp.kernel_optimized import validate_kernel_bounds

# In _validate_weights():
validation = validate_inputs(c_array, w_array)
if not validation["valid"]:
    self.errors.append(f"✗ {validation['errors']}")

# In _validate_invariant_identities():
bounds_valid = validate_kernel_bounds(F, omega, C, IC, kappa, epsilon=1e-6)
```

---

### 2. cli.py - Main Validation CLI

**Current Location**: [src/umcp/cli.py](src/umcp/cli.py)

| Function | Lines | Current Implementation | Optimization Opportunity |
|----------|-------|----------------------|-------------------------|
| Tier-1 identity checks | L685-810 | Manual F vs 1-ω check | Use `OptimizedKernelComputer` for AM-GM gap analysis (OPT-3) |
| IC ≈ exp(κ) check | L750-810 | Manual exp(κ) computation | Use log-space κ validation (OPT-4) |
| Weight validation | Various | Manual sum check | Use `compute_utils.normalize_weights()` |

**Integration Points**:
```python
# In cli.py imports:
from umcp.kernel_optimized import OptimizedKernelComputer, validate_kernel_bounds
from umcp.compute_utils import validate_inputs, prune_zero_weights

# Create global computer instance for reuse:
_kernel_computer = OptimizedKernelComputer(epsilon=1e-6)

# In tier1 identity check functions:
outputs = _kernel_computer.compute(c_array, w_array, validate=True)
if outputs.amgm_gap > tolerance:
    # Flag AM-GM violation
```

---

### 3. __init__.py - Module Exports

**Current Location**: [src/umcp/__init__.py](src/umcp/__init__.py)

**Add exports for optimization modules**:
```python
# In __all__:
__all__ = [
    ...
    "OptimizedKernelComputer",
    "SeamChainAccumulator", 
    "compute_utils",
]

# New imports:
from .kernel_optimized import OptimizedKernelComputer
from .seam_optimized import SeamChainAccumulator
from . import compute_utils
```

---

## 🟠 MEDIUM PRIORITY: Closure Upgrades

### 4. closures/tau_R_compute.py → tau_R_optimized.py

**Action**: Deprecate simple `tau_R_compute.py` in favor of `tau_R_optimized.py`

**Current**: Simple formula τ_R = 1/(damping × ω)
**Optimized**: Full return time computation with margin-based early exit, caching

**Backward Compatibility Wrapper**:
```python
# In tau_R_compute.py - add wrapper:
def compute(omega: float, damping: float) -> dict:
    """Legacy interface - delegates to optimized version for simple cases."""
    # For simple cases, keep existing implementation
    # For trace-based computation, use OptimizedReturnComputer
    ...
```

---

### 5. GCD Closures Integration

| Closure | File | Integration |
|---------|------|-------------|
| `entropic_collapse.py` | [closures/gcd/entropic_collapse.py](closures/gcd/entropic_collapse.py) | Add `validate_kernel_bounds()` for S,F inputs |
| `momentum_flux.py` | [closures/gcd/momentum_flux.py](closures/gcd/momentum_flux.py) | Use `BatchProcessor` for κ,C,ω series validation |
| `field_resonance.py` | [closures/gcd/field_resonance.py](closures/gcd/field_resonance.py) | Add homogeneity detection (OPT-1) |
| `generative_flux.py` | [closures/gcd/generative_flux.py](closures/gcd/generative_flux.py) | Use `prune_zero_weights()` for sparse inputs |
| `energy_potential.py` | [closures/gcd/energy_potential.py](closures/gcd/energy_potential.py) | Add Lipschitz bounds (OPT-12) |

**Example for entropic_collapse.py**:
```python
# Add at top:
import sys
sys.path.insert(0, "/path/to/src")
from umcp.kernel_optimized import validate_kernel_bounds

# In compute_entropic_collapse():
if not validate_kernel_bounds(F=F, S=S, epsilon=1e-6):
    raise ValueError("Input validation failed per Lemma 1")
```

---

### 6. RCFT Closures Integration

| Closure | File | Integration |
|---------|------|-------------|
| `attractor_basin.py` | [closures/rcft/attractor_basin.py](closures/rcft/attractor_basin.py) | Use `SeamChainAccumulator` for phase transitions |
| `recursive_field.py` | [closures/rcft/recursive_field.py](closures/rcft/recursive_field.py) | Add homogeneity check for uniform field detection |
| `resonance_pattern.py` | [closures/rcft/resonance_pattern.py](closures/rcft/resonance_pattern.py) | Use `BatchProcessor.compute_batch_statistics()` |
| `fractal_dimension.py` | [closures/rcft/fractal_dimension.py](closures/rcft/fractal_dimension.py) | Add OPT-20 vectorized operations |

---

## 🟡 LOW PRIORITY: Examples & Documentation

### 7. examples/interconnected_demo.py

**Add optimization demonstration section**:
```python
# ========================================================================
# PART 5: Optimized Kernel Computation
# ========================================================================
print("PART 5: Optimized Kernel Computation")
print("-" * 70)

from umcp.kernel_optimized import OptimizedKernelComputer
import numpy as np
import time

computer = OptimizedKernelComputer(epsilon=1e-6)
c = np.array([float(trace[0][f"c_{i}"]) for i in range(1, 4)])
w = np.array([float(weights[0][f"w_{i}"]) for i in range(1, 4)])

# Time comparison
start = time.time()
for _ in range(1000):
    outputs = computer.compute(c, w)
optimized_time = time.time() - start

print(f"  ✓ Optimized computation: {optimized_time:.4f}s for 1000 runs")
print(f"  ✓ Homogeneous: {outputs.is_homogeneous}")
print(f"  ✓ AM-GM gap: {outputs.amgm_gap:.6f}")
```

---

### 8. Registry Updates

**closures/registry.yaml** - Add optimized closures:
```yaml
tau_R_optimized:
  path: tau_R_optimized.py
  version: 1.0.0
  tier: 1.5
  optimizations: [OPT-7, OPT-8, OPT-9]
  replaces: tau_R_compute.py
  notes: "Optimized return time with caching and early exit"
```

---

## Implementation Checklist

### Phase 1: Core Integration (HIGH PRIORITY)
- [ ] Update `src/umcp/__init__.py` with new exports
- [ ] Integrate `compute_utils` into `validator.py`
- [ ] Add `kernel_optimized` validation to `cli.py` Tier-1 checks
- [ ] Update docstrings with optimization references

### Phase 2: Closure Upgrades (MEDIUM PRIORITY)
- [ ] Add deprecation notice to `tau_R_compute.py`
- [ ] Integrate validation into GCD closures
- [ ] Add batch processing to RCFT closures
- [ ] Update `closures/registry.yaml`

### Phase 3: Examples & Tests (LOW PRIORITY)
- [ ] Add optimization demo to `interconnected_demo.py`
- [ ] Update `load_umcp_files.py` with validation examples
- [ ] Add benchmark tests comparing optimized vs baseline

---

## Performance Impact Summary

| Integration Point | Expected Speedup | Memory Impact |
|-------------------|------------------|---------------|
| validator.py weight check | 20% | Negligible |
| cli.py Tier-1 identity | 30-40% (homogeneous cases) | +O(n) cache |
| GCD closure validation | 10-15% | Negligible |
| RCFT batch processing | 5x | +O(T) |
| Return time (tau_R) | 30-50% | +O(T²) cache |

---

## References

- [COMPUTATIONAL_OPTIMIZATIONS.md](COMPUTATIONAL_OPTIMIZATIONS.md): Full optimization catalog (21 items)
- [KERNEL_SPECIFICATION.md](KERNEL_SPECIFICATION.md): Formal lemmas (1-34)
- [OPTIMIZATION_INTEGRATION_GUIDE.md](OPTIMIZATION_INTEGRATION_GUIDE.md): Usage examples
- [tests/test_computational_optimizations.py](tests/test_computational_optimizations.py): Test suite (27 tests)

# UMCP C++ Accelerator (`umcp_cpp`)

**Purpose**: High-performance C++ implementations of computationally intensive
UMCP kernel operations, exposed to Python via pybind11.

> *Trans suturam congelatum* — Same rules both sides of every collapse-return
> boundary. The C++ implementations produce **bit-identical** results to the
> Python reference under IEEE 754 double precision.

## Why C++?

| Operation | Python (NumPy) | C++ | Speedup Target |
|-----------|---------------|-----|----------------|
| **Kernel computation** (F, ω, S, C, κ, IC) | ~15 μs/vector | ~0.3 μs/vector | 50× |
| **Batch kernel** (10,000 trace rows) | ~150 ms | ~3 ms | 50× |
| **Seam chain accumulation** (1,000 seams) | ~12 ms | ~0.15 ms | 80× |
| **SHA-256 integrity** (100 files) | ~45 ms | ~8 ms | 5× |
| **Tier-1 proof sweep** (10,162 tests) | ~8 s | ~0.2 s | 40× |

These are the operations where C++ excels: tight inner loops over numeric arrays
with no allocation, branch-free arithmetic, SIMD vectorization, and cache-local
memory access patterns. Python's dynamic typing and interpreter overhead dominate
at this scale.

## Architecture

```
umcp_cpp/
├── README.md                     # This file
├── CMakeLists.txt                # Build system (CMake 3.16+)
├── include/
│   └── umcp/
│       ├── kernel.hpp            # Kernel invariant computation (F, ω, S, C, κ, IC)
│       ├── seam.hpp              # Seam chain accumulation and residual monitoring
│       └── integrity.hpp         # SHA-256 file integrity (optional OpenSSL)
├── src/
│   ├── kernel.cpp                # Kernel implementation
│   ├── seam.cpp                  # Seam implementation
│   └── integrity.cpp             # SHA-256 implementation
├── bindings/
│   └── py_umcp.cpp               # pybind11 Python bindings
└── tests/
    └── test_kernel.cpp           # Catch2 unit tests
```

## Tier Classification

This module is **Tier-0 (Protocol)** — it implements the same computation as
`kernel_optimized.py` and `seam_optimized.py` with identical frozen parameters.
It does NOT redefine any Tier-1 symbol (F, ω, S, C, κ, IC, τ_R, regime).

The C++ code:
- Receives the same inputs (c ∈ [ε, 1−ε]^n, w ∈ Δ^n)
- Computes the same formulas (Definitions 4–7, Lemmas 1–4)
- Returns the same outputs (KernelOutputs struct)
- Enforces the same range validation (Lemma 1)

**No frozen parameter is redefined.** All values (ε, p, tol_seam) come from
the Python-side frozen contract.

## Building

### Prerequisites

- C++17 compiler (GCC 9+, Clang 10+, MSVC 2019+)
- CMake 3.16+
- pybind11 (auto-fetched by CMake or `pip install pybind11`)
- Python 3.11+ with NumPy headers

### Build Steps

```bash
# From repo root
pip install pybind11 cmake
cd src/umcp_cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Install the Python module
cp umcp_accel*.so ../../umcp/

# Or use pip (once pyproject.toml is configured)
pip install -e ".[cpp]"
```

### Running C++ Tests

```bash
cd build
ctest --output-on-failure
```

## Python Integration

The C++ extension is **optional**. The Python wrapper auto-detects availability:

```python
from umcp.accel import compute_kernel

# Automatically uses C++ if available, falls back to NumPy
result = compute_kernel(c, w, epsilon=1e-8)
```

## Verification

The C++ kernel must pass the same 10,162 Tier-1 identity checks as the
Python kernel. The benchmark script verifies bit-level agreement:

```bash
python scripts/benchmark_cpp.py --verify --iterations 100000
```

## Key Differences: Why C++ Excels Here

| Aspect | Python/NumPy | C++ |
|--------|-------------|-----|
| **Loop overhead** | ~100ns per Python iteration | ~0.3ns per iteration |
| **Log/exp calls** | NumPy ufunc dispatch overhead | Direct `std::log`/`std::exp`, inlined |
| **Memory** | Array allocation per operation | Stack-allocated, zero-alloc inner loop |
| **Branching** | Dynamic dispatch through interpreter | Branch prediction, speculative execution |
| **SIMD** | Limited (NumPy uses BLAS for linear algebra, not custom kernels) | Auto-vectorized by compiler (-O3 -march=native) |
| **Cache** | Scattered object headers, GC pressure | Contiguous doubles, cache-line aligned |

The kernel computation is a textbook case for C++ acceleration: a small,
deterministic, numerically intensive function called millions of times with
no I/O, no allocation, and no dynamic dispatch needed.

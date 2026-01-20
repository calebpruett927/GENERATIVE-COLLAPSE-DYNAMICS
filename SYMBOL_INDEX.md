# UMCP Symbol Index

**Fast lookup table for reserved symbols**  
**Version:** 1.0.0  
**Last Updated:** 2026-01-20

This index provides rapid symbol lookup with Unicode, ASCII, and file encodings. Required for preventing symbol capture and ensuring protocol reproducibility.

See also:
- [Glossary](GLOSSARY.md) - Complete structured definitions
- [Term Index](TERM_INDEX.md) - Alphabetical term lookup
- [Canon Anchors](canon/) - Machine-readable specifications

---

## Quick Reference Table

| Symbol | ASCII | Tier | Domain | Definition | File Location |
|--------|-------|------|--------|------------|---------------|
| **Ψ(t)** | Psi(t) | Tier-0 | [0,1]^n | Bounded trace (embedded state) | [canon/anchors.yaml](canon/anchors.yaml) |
| **ω** | omega | Tier-1 | [0,1] | Drift (collapse proximity) | [canon/gcd_anchors.yaml](canon/gcd_anchors.yaml) |
| **F** | F | Tier-1 | [0,1] | Fidelity (1 - ω) | [canon/gcd_anchors.yaml](canon/gcd_anchors.yaml) |
| **S** | S | Tier-1 | [0,S_max] | Entropy (Shannon functional) | [canon/gcd_anchors.yaml](canon/gcd_anchors.yaml) |
| **C** | C | Tier-1 | [0,1] | Curvature (dispersion proxy) | [canon/gcd_anchors.yaml](canon/gcd_anchors.yaml) |
| **τ_R** | tau_R | Tier-1 | ℕ∪{∞_rec} | Return time (re-entry delay) | [canon/gcd_anchors.yaml](canon/gcd_anchors.yaml) |
| **κ** | kappa | Tier-1 | (-∞,0] | Log-integrity (ln of IC) | [canon/gcd_anchors.yaml](canon/gcd_anchors.yaml) |
| **IC** | IC | Tier-1 | (0,1] | Integrity composite (exp(κ)) | [canon/gcd_anchors.yaml](canon/gcd_anchors.yaml) |
| **D_f** | D_fractal | Tier-2 | [1,3] | Fractal dimension (RCFT) | [canon/rcft_anchors.yaml](canon/rcft_anchors.yaml) |
| **Ψ_r** | Psi_recursive | Tier-2 | [0,∞) | Recursive field strength (RCFT) | [canon/rcft_anchors.yaml](canon/rcft_anchors.yaml) |
| **λ_p** | lambda_pattern | Tier-2 | (0,∞] | Resonance wavelength (RCFT) | [canon/rcft_anchors.yaml](canon/rcft_anchors.yaml) |
| **Θ** | Theta_phase | Tier-2 | [0,2π) | Phase angle (RCFT) | [canon/rcft_anchors.yaml](canon/rcft_anchors.yaml) |
| **∞_rec** | INF_REC | Boundary | typed | No-return typed state | [contracts/UMA.INTSTACK.v1.yaml](contracts/UMA.INTSTACK.v1.yaml) |
| **Δκ** | Delta_kappa | Tier-1.5 | ℝ | Ledger/budget delta (seam) | [contracts/GCD.INTSTACK.v1.yaml](contracts/GCD.INTSTACK.v1.yaml) |

---

## Tier-0: Interface Symbols

### Measurement and Embedding

| Symbol | ASCII | Definition | Files |
|--------|-------|------------|-------|
| **x(t)** | x(t) | Observable (unitful measurement) | `observables.yaml` in CasePacks |
| **Ψ(t)** | Psi(t), trace | Bounded trace [0,1]^n | `derived/trace.csv` |
| **c_i(t)** | c_i(t), component_i | Trace component (bounded scalar) | `derived/trace.csv` columns |
| **w_i** | w_i, weight_i | Component weight (normalized) | `weights.csv` |
| **N_K** | N_K, embedding | Normalization/embedding map | `embedding.yaml` |
| **ε** | epsilon, eps | Log-safety clipping threshold | `contract.yaml` (epsilon field) |

**Not to be confused with:**
- **Ψ_r** (recursive field, Tier-2) - uses same base letter but different subscript
- **ε** machine epsilon (2.22e-16) - protocol ε is typically 1e-10

---

## Tier-1: GCD Reserved Symbols

### Core Invariants

| Symbol | ASCII | Identity/Formula | Regime Thresholds |
|--------|-------|------------------|-------------------|
| **ω** | omega | ω = 1 - F | Stable: ω < 0.038<br>Collapse: ω ≥ 0.30 |
| **F** | F | F = Σ w_i c_i | Collapse: F < 0.75 |
| **S** | S | S = -Σ w_i [c_i ln(c_i) + ...] | Collapse: S > 0.15 |
| **C** | C | C = √(Σ w_i (c_i - c̄)²) | Collapse: C > 0.14 |
| **κ** | kappa | κ = ln(IC) = Σ w_i ln(c_i,ε) | Used in seam ledger |
| **IC** | IC | IC = exp(κ) | Critical: IC_min < threshold |

**Mathematical Identities** (exact):
```
F = 1 - ω           (definition)
κ = ln(IC)          (exact)
IC = exp(κ)         (exact inverse)
```

**File Locations:**
- Definitions: [canon/gcd_anchors.yaml](canon/gcd_anchors.yaml) → reserved_symbols
- Thresholds: [contracts/GCD.INTSTACK.v1.yaml](contracts/GCD.INTSTACK.v1.yaml) → regime_gates
- Computation: [src/umcp/validator.py](src/umcp/validator.py)
- Outputs: `outputs/invariants.csv` in all CasePacks

### Return Machinery

| Symbol | ASCII | Definition | Typed States |
|--------|-------|------------|--------------|
| **τ_R** | tau_R | Return time to domain D_θ | ∞_rec (no return)<br>ℕ (finite return) |
| **η** | eta | Return tolerance (distance threshold) | Frozen per contract |
| **H_rec** | H_rec | Return horizon (search window) | Frozen per contract |
| **D_θ(t)** | D_theta(t) | Return domain generator | Declared closure |
| **∥·∥** | norm | Distance metric in Ψ-space | L2 default |
| **U_θ(t)** | U_theta(t) | Return candidate set | {u : ∥Ψ(t)-Ψ(u)∥≤η} |

**Typed Boundary Convention:**
```
τ_R = ∞_rec   when U_θ(t) = ∅
     (INF_REC in files)
```

**File Locations:**
- Definition: [canon/gcd_anchors.yaml](canon/gcd_anchors.yaml) → return_machinery
- Implementation: [closures/tau_R_compute.py](closures/tau_R_compute.py)
- Domain spec: [closures/return_domain.window64.v1.yaml](closures/return_domain.window64.v1.yaml)
- Tests: [tests/test_70_contract_closures.py](tests/test_70_contract_closures.py)

---

## Tier-1.5: Seam and Weld Symbols

### Continuity Accounting

| Symbol | ASCII | Formula | Gate Use |
|--------|-------|---------|----------|
| **Δκ_ledger** | Delta_kappa_ledger | κ_1 - κ_0 | Identity (exact) |
| **Δκ_budget** | Delta_kappa_budget | R·τ_R - (D_ω + D_C) | Closure-dependent |
| **s** | s, residual | Δκ_budget - Δκ_ledger | PASS/FAIL check |
| **ir** | ir, integrity_ratio | IC_1 / IC_0 | Identity check |
| **R** | R | Return-credit rate | Closure estimator |
| **D_ω** | D_omega | Drift dissipation via Γ(ω) | Closure: gamma |
| **D_C** | D_C | Curvature dissipation | Closure: alphaC |

**Weld PASS Conditions** (all must hold):
```
1. τ_R is finite (not ∞_rec)
2. |s| ≤ tol_seam
3. |ir - exp(Δκ_ledger)| ≤ tol_id
```

**File Locations:**
- Ledger/budget: [contracts/GCD.INTSTACK.v1.yaml](contracts/GCD.INTSTACK.v1.yaml) → weld_accounting
- Closures: [closures/registry.yaml](closures/registry.yaml)
- Gamma(ω): [closures/gamma.default.v1.yaml](closures/gamma.default.v1.yaml)
- Outputs: `outputs/welds.csv` in CasePacks with seams

---

## Tier-2: RCFT Extension Symbols

### Geometric and Recursive Metrics

| Symbol | ASCII | Formula | Regime Classification |
|--------|-------|---------|----------------------|
| **D_f** | D_fractal | lim(ε→0) log(N(ε))/log(1/ε) | Smooth: D_f < 1.2<br>Wrinkled: 1.2-1.8<br>Turbulent: D_f ≥ 1.8 |
| **Ψ_r** | Psi_recursive | Σ α^n · Ψ_n | Dormant: Ψ_r < 0.1<br>Active: 0.1-1.0<br>Resonant: Ψ_r ≥ 1.0 |
| **λ_p** | lambda_pattern | 2π / k_dominant | Wavelength of oscillation |
| **Θ** | Theta_phase | arctan(Im/Re) | Phase coherence |

**Tier-2 Principle:** These symbols augment but do not override Tier-1 invariants. All 13 GCD symbols remain frozen.

**File Locations:**
- Definitions: [canon/rcft_anchors.yaml](canon/rcft_anchors.yaml) → tier_2_extensions
- Theory: [docs/rcft_theory.md](docs/rcft_theory.md)
- Implementations: [closures/rcft/](closures/rcft/)
- CasePacks: [casepacks/rcft_complete/](casepacks/rcft_complete/)
- Tests: [tests/test_110_rcft_canon.py](tests/test_110_rcft_canon.py)

---

## Symbol Collision Prevention

### Common Conflicts (Must Disambiguate)

| Reserved | Collision | Correct Usage |
|----------|-----------|---------------|
| **ω** (drift) | Ω (angular frequency) | Use **Ω_freq** or **Omega_freq** for frequency |
| **C** (curvature) | C (capacitance) | Use **C_cap** for capacitance |
| **κ** (log-integrity) | κ (curvature in geometry) | Context: UMCP κ is always log-integrity |
| **S** (entropy) | S_th (thermodynamic) | Use **S_th** for thermodynamic entropy (Tier-2 mapping) |
| **IC** (integrity) | I (information content) | Use **I_info** for external information metrics |
| **Ψ(t)** (trace) | Ψ_r (recursive) | Different subscripts; check tier context |
| **τ_R** (return time) | τ (general time) | Use **τ** only with subscript or context |
| **λ_p** (pattern) | λ (eigenvalue) | Use **λ_eig** for eigenvalues |

**Enforcement:** Symbol capture without disambiguation is NONCONFORMANT.

**See:** [GLOSSARY.md](GLOSSARY.md) - "Not to be confused with" sections for each term

---

## File Format Encodings

### CSV Column Names

```csv
# Tier-1 invariants (outputs/invariants.csv)
t, omega, F, S, C, tau_R, kappa, IC

# Trace components (derived/trace.csv)
t, c1, c2, c3, ..., cn

# Weld accounting (outputs/welds.csv)
weld_id, t0, t1, Delta_kappa_ledger, Delta_kappa_budget, residual, status
```

### YAML Field Names

```yaml
# Tier tags in canon anchors
reserved_symbols:
  - symbol: omega      # ASCII identifier
    latex: ω           # Unicode/LaTeX
    tier: 1            # Tier classification
    
# Typed boundary states
return_time: INF_REC   # Not a number; typed state
```

### Python Variable Names

```python
# Prefer ASCII with underscores
omega = 1 - F          # Not: ω (Unicode identifier)
tau_R = compute_return_time()
Delta_kappa_ledger = kappa_1 - kappa_0

# Subscripts as suffixes
c_i = trace_components[i]
w_i = weights[i]
Psi_r = recursive_field_strength
```

---

## Cross-Reference Map

### By Analysis Task

| Task | Primary Symbols | Files |
|------|----------------|-------|
| Regime classification | ω, F, S, C | [contracts/GCD.INTSTACK.v1.yaml](contracts/GCD.INTSTACK.v1.yaml) → regime_gates |
| Weld evaluation | Δκ, τ_R, R, s | [contracts/GCD.INTSTACK.v1.yaml](contracts/GCD.INTSTACK.v1.yaml) → weld_gate |
| Integrity tracking | κ, IC | All `outputs/invariants.csv` |
| RCFT overlay | D_f, Ψ_r, λ_p | [casepacks/rcft_complete/](casepacks/rcft_complete/) |
| Trace embedding | Ψ(t), c_i, w_i | `embedding.yaml`, `derived/trace.csv` |

### By Document Type

| Document | Symbol Coverage | Link |
|----------|----------------|------|
| Canon anchors | All reserved (authoritative) | [canon/](canon/) |
| Contracts | Thresholds, frozen parameters | [contracts/](contracts/) |
| Glossary | Definitions with context | [GLOSSARY.md](GLOSSARY.md) |
| Theory docs | Mathematical foundations | [docs/rcft_theory.md](docs/rcft_theory.md) |
| Tests | Symbol validation | [tests/test_100_gcd_canon.py](tests/test_100_gcd_canon.py) |

---

## Implementation Quick Reference

### Python: Load Symbol Definitions

```python
# From canon anchors (machine-readable)
import yaml
from pathlib import Path

gcd_canon = yaml.safe_load(Path("canon/gcd_anchors.yaml").read_text())
symbols = gcd_canon["tier_1_invariants"]["reserved_symbols"]

for sym in symbols:
    print(f"{sym['latex']} ({sym['symbol']}) - {sym['description']}")
```

### Validation: Check Symbol Use

```bash
# Validate symbol usage across repository
umcp validate .

# Check for symbol capture (collision)
grep -r "\\omega" . --include="*.py" | grep -v "# omega"
```

### Schema: Symbol Validation

See [schemas/canon.anchors.schema.json](schemas/canon.anchors.schema.json) for JSON Schema validation of symbol definitions.

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-20 | Initial comprehensive symbol index |

**Next:** Add Tier-3 extensions when introduced (none currently defined)

---

## Related Resources

- **[GLOSSARY.md](GLOSSARY.md)** - Complete term definitions with context
- **[TERM_INDEX.md](TERM_INDEX.md)** - Alphabetical term index
- **[AXIOM.md](AXIOM.md)** - Core axiom and tier hierarchy
- **[canon/](canon/)** - Machine-readable symbol specifications
- **[contracts/](contracts/)** - Frozen parameters and thresholds
- **[CHANGELOG.md](CHANGELOG.md)** - Symbol evolution and deprecations

---

**Protocol Compliance:** Every symbol used in computation must appear in this index with unambiguous tier, encoding, and definition. Unlisted symbols are NONCONFORMANT.

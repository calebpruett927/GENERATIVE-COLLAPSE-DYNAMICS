# UMCP Protocol Reference Guide

**Quick Navigation:** [Glossary](GLOSSARY.md) | [Symbol Index](SYMBOL_INDEX.md) | [Term Index](TERM_INDEX.md)

## Purpose

This reference guide provides fast pathways to all protocol infrastructure, ensuring readers can locate definitions, gates, receipts, failure modes, and examples without ambiguity.

**Protocol Rule:** If a term or symbol is used but cannot be located quickly in the glossary and at least one index, it is not acceptable protocol writing.

---

## Core Protocol Infrastructure

### 1. **Glossary System** (Required Infrastructure)

| Resource | Purpose | Link |
|----------|---------|------|
| **Glossary** | Structured definitions for all terms | [GLOSSARY.md](GLOSSARY.md) |
| **Symbol Index** | Unicode/ASCII symbol lookup table | [SYMBOL_INDEX.md](SYMBOL_INDEX.md) |
| **Term Index** | Alphabetical term cross-reference | [TERM_INDEX.md](TERM_INDEX.md) |
| **Glossary Schema** | JSON Schema validation | [schemas/glossary.schema.json](schemas/glossary.schema.json) |

**Prevents:** Symbol capture, reader imports, ambiguous definitions  
**Enables:** Self-service lookup, reproducibility, dispute resolution

---

## Quick Lookup by Category

### Tier-0: Interface and Measurement

| Term | Symbol | Definition | Files |
|------|--------|------------|-------|
| Observable | x(t) | Unitful measurement | [GLOSSARY.md](GLOSSARY.md#observable-xt) |
| Bounded Trace | Ψ(t) | Embedded state [0,1]^n | [GLOSSARY.md](GLOSSARY.md#bounded-trace-psi-ψt) |
| Embedding | N_K | Observable → trace map | [GLOSSARY.md](GLOSSARY.md#embedding--normalization-n_k) |
| Weights | w_i | Component importance | [GLOSSARY.md](GLOSSARY.md#weights-w_i) |
| Epsilon-clipping | ε | Log-safety threshold | [GLOSSARY.md](GLOSSARY.md#epsilon-clipping--log-safety-ε) |

**Canonical Definitions:** [canon/anchors.yaml](canon/anchors.yaml)

### Tier-1: GCD Reserved Symbols

| Symbol | ASCII | Formula | Thresholds |
|--------|-------|---------|------------|
| **ω** | omega | ω = 1 - F | Collapse: ω ≥ 0.30 |
| **F** | F | F = Σ w_i c_i | Collapse: F < 0.75 |
| **S** | S | Shannon entropy | Collapse: S > 0.15 |
| **C** | C | Dispersion proxy | Collapse: C > 0.14 |
| **τ_R** | tau_R | Return time | Typed: ℕ∪{∞_rec} |
| **κ** | kappa | κ = ln(IC) | Seam ledger |
| **IC** | IC | IC = exp(κ) | Critical threshold |

**Canonical Definitions:** [canon/gcd_anchors.yaml](canon/gcd_anchors.yaml)  
**Glossary Section:** [GLOSSARY.md#tier-1-reserved-symbols-gcd-framework](GLOSSARY.md#tier-1-reserved-symbols-gcd-framework)

### Tier-0: Seam and Weld

| Term | Symbol | Formula | Gate Use |
|------|--------|---------|----------|
| Ledger Delta | Δκ_ledger | κ_1 - κ_0 | Identity (exact) |
| Budget Delta | Δκ_budget | R·τ_R - (D_ω + D_C) | Closure-dependent |
| Residual | s | Δκ_budget - Δκ_ledger | PASS/FAIL check |
| Weld Gate | PASS/FAIL | Finite τ_R + tolerances | Binding decision |

**Canonical Definitions:** [contracts/GCD.INTSTACK.v1.yaml](contracts/GCD.INTSTACK.v1.yaml)  
**Glossary Section:** [GLOSSARY.md#tier-0-seam-and-weld-calculus](GLOSSARY.md#tier-0-seam-and-weld-calculus)

### Tier-2: RCFT Extensions

| Symbol | ASCII | Domain | Regimes |
|--------|-------|--------|---------|
| **D_f** | D_fractal | [1, 3] | Smooth < 1.2, Turbulent ≥ 1.8 |
| **Ψ_r** | Psi_recursive | [0, ∞) | Dormant < 0.1, Resonant ≥ 1.0 |
| **λ_p** | lambda_pattern | (0, ∞] | Wavelength (FFT) |
| **Θ** | Theta_phase | [0, 2π) | Phase coherence |

**Canonical Definitions:** [canon/rcft_anchors.yaml](canon/rcft_anchors.yaml)  
**Glossary Section:** [GLOSSARY.md#tier-2-rcft-overlay-extensions](GLOSSARY.md#tier-2-rcft-overlay-extensions)

---

## File Location Map

### By Document Type

| Type | Files | Purpose |
|------|-------|---------|
| **Canon Anchors** | [canon/](canon/) | Tier-tagged authoritative symbols |
| **Contracts** | [contracts/](contracts/) | Frozen parameters and thresholds |
| **Closures** | [closures/](closures/) | Auxiliary specifications |
| **Schemas** | [schemas/](schemas/) | JSON Schema validation |
| **CasePacks** | [casepacks/](casepacks/) | Reproducible bundles |
| **Documentation** | [docs/](docs/) | Theory, usage, deployment |
| **Tests** | [tests/](tests/) | Validation and compliance |

### By Task

| Task | Primary Resources |
|------|------------------|
| **Term lookup** | [GLOSSARY.md](GLOSSARY.md), [TERM_INDEX.md](TERM_INDEX.md) |
| **Symbol lookup** | [SYMBOL_INDEX.md](SYMBOL_INDEX.md) |
| **Regime classification** | [contracts/GCD.INTSTACK.v1.yaml](contracts/GCD.INTSTACK.v1.yaml) → regime_gates |
| **Weld evaluation** | [contracts/GCD.INTSTACK.v1.yaml](contracts/GCD.INTSTACK.v1.yaml) → weld_gate |
| **RCFT overlay** | [docs/rcft_theory.md](docs/rcft_theory.md), [canon/rcft_anchors.yaml](canon/rcft_anchors.yaml) |
| **Create CasePack** | [docs/quickstart.md](docs/quickstart.md) |
| **Validate** | `umcp validate`, [src/umcp/validator.py](src/umcp/validator.py) |

---

## Symbol Collision Prevention

**Rule:** Reserved symbols must not be redefined. Collisions must be disambiguated.

| Reserved | Collision | Correct Usage |
|----------|-----------|---------------|
| **ω** (drift) | Ω (frequency) | Use **Ω_freq** or **Omega_freq** |
| **C** (curvature) | C (capacitance) | Use **C_cap** for capacitance |
| **κ** (log-integrity) | κ (geometry) | Context: UMCP κ is always log-integrity |
| **S** (entropy) | S_th (thermodynamic) | Use **S_th** for thermodynamic entropy |
| **IC** (integrity) | I (information) | Use **I_info** for information content |
| **Ψ(t)** (trace) | Ψ_r (recursive) | Different subscripts; check tier |

**See:** [SYMBOL_INDEX.md#symbol-collision-prevention](SYMBOL_INDEX.md#symbol-collision-prevention)

---

## Gates vs Diagnostics

**Critical Distinction:** Gates decide, diagnostics inform.

### Gates (Binding Authority)
- **Regime Gates:** Stable / Watch / Collapse
- **Weld Gate:** PASS / FAIL
- **Source:** Tier-1 invariants + frozen thresholds
- **Output:** Categorical labels used downstream

### Diagnostics (Insight Only)
- **Purpose:** Explain behavior, assess sensitivity
- **Authority:** Cannot override gates or assign labels
- **Tier:** Tier-2 (subordinate to kernel)
- **Examples:** Equator residuals, sensitivity analysis

**Definition:** [GLOSSARY.md#diagnostic-vs-gate](GLOSSARY.md#diagnostic-vs-gate)

---

## Conformance Status

| Status | Meaning | Treatment |
|--------|---------|-----------|
| **CONFORMANT** | All protocol requirements met | Results valid for claims |
| **NONCONFORMANT** | Structural violation | Mark "do not interpret" |
| **FAIL (Weld)** | Valid outcome within protocol | Continuity not established |

**Not confused:** FAIL (valid outcome) ≠ NONCONFORMANT (protocol violation)

**Definition:** [GLOSSARY.md#nonconformance](GLOSSARY.md#nonconformance)

---

## Common Lookup Pathways

### From Deeper Files to Glossary

**Canon Anchors:**
- [canon/gcd_anchors.yaml](canon/gcd_anchors.yaml) → header links to [GLOSSARY.md](GLOSSARY.md#tier-1-reserved-symbols-gcd-framework)
- [canon/rcft_anchors.yaml](canon/rcft_anchors.yaml) → header links to [GLOSSARY.md](GLOSSARY.md#tier-2-rcft-overlay-extensions)

**Contracts:**
- [contracts/README.md](contracts/README.md) → links to [GLOSSARY.md](GLOSSARY.md), [SYMBOL_INDEX.md](SYMBOL_INDEX.md)
- Individual contracts → terms link to glossary entries

**Closures:**
- [closures/README.md](closures/README.md) → links to [GLOSSARY.md#closure](GLOSSARY.md#closure)
- Individual closures → symbol usage references index

**CasePacks:**
- [casepacks/hello_world/README.md](casepacks/hello_world/README.md) → protocol resources header
- [casepacks/gcd_complete/README.md](casepacks/gcd_complete/README.md) → GCD glossary section
- [casepacks/rcft_complete/README.md](casepacks/rcft_complete/README.md) → RCFT glossary section

**Documentation:**
- [docs/rcft_theory.md](docs/rcft_theory.md) → quick reference header
- [docs/interconnected_architecture.md](docs/interconnected_architecture.md) → term definitions header
- [docs/python_coding_key.md](docs/python_coding_key.md) → symbol encodings

**Root Files:**
- [README.md](README.md) → quick access section
- [AXIOM.md](AXIOM.md) → protocol infrastructure header

### From Glossary to Implementations

Each glossary entry contains:
- **Where defined:** Canonical definition files
- **Where used:** Implementation locations
- **Examples:** Worked examples in CasePacks

Example pathway:
1. Reader encounters term "curvature proxy"
2. Looks up in [TERM_INDEX.md#c](TERM_INDEX.md#c)
3. Finds [GLOSSARY.md#curvature-proxy-ct](GLOSSARY.md#curvature-proxy-ct)
4. Follows "Where defined" to [canon/gcd_anchors.yaml](canon/gcd_anchors.yaml)
5. Follows "Where used" to [casepacks/hello_world/](casepacks/hello_world/)

---

## Worked Examples

| Example | Demonstrates | Location |
|---------|-------------|----------|
| **hello_world** | Minimal validation | [casepacks/hello_world/](casepacks/hello_world/) |
| **gcd_complete** | Full GCD framework | [casepacks/gcd_complete/](casepacks/gcd_complete/) |
| **rcft_complete** | Tier-2 RCFT overlay | [casepacks/rcft_complete/](casepacks/rcft_complete/) |
| **UMCP-REF-E2E-0001** | End-to-end reference | [casepacks/UMCP-REF-E2E-0001/](casepacks/UMCP-REF-E2E-0001/) |

---

## Failure Modes and Diagnostics

### Protocol Violations (NONCONFORMANT)
- Missing freeze
- Symbol capture without disambiguation
- Post-hoc artifact edits
- Diagnostic-as-gate usage
- Missing closure registry

**Reference:** [GLOSSARY.md#nonconformance](GLOSSARY.md#nonconformance)

### Valid Negative Outcomes
- Weld FAIL (no continuity established)
- Collapse regime (ω ≥ 0.30)
- Critical severity tag (min IC below threshold)
- Typed boundary state (τ_R = ∞_rec)

**Reference:** [GLOSSARY.md#pass--fail-weld-gate](GLOSSARY.md#pass--fail-weld-gate)

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-20 | Initial comprehensive reference guide |

---

## Related Resources

**Core Axiom:**
- [AXIOM.md](AXIOM.md) - "What Returns Through Collapse Is Real"

**System Architecture:**
- [docs/interconnected_architecture.md](docs/interconnected_architecture.md)
- [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md)

**Theory and Implementation:**
- [docs/rcft_theory.md](docs/rcft_theory.md) - RCFT mathematical foundations
- [docs/python_coding_key.md](docs/python_coding_key.md) - Implementation patterns

**Deployment:**
- [docs/production_deployment.md](docs/production_deployment.md)
- [QUICKSTART_EXTENSIONS.md](QUICKSTART_EXTENSIONS.md)

**Change Tracking:**
- [CHANGELOG.md](CHANGELOG.md) - Version history and deprecations

---

**Maintenance:** Update this guide when new tiers, symbols, or glossary entries are added.

**Last Updated:** 2026-01-20 | **Version:** 1.0.0

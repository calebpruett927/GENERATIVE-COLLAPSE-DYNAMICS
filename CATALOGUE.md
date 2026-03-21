# CATALOGUE — Master Index of All Formal Objects

> *Omnia numerantur; nihil latet.* — Everything is counted; nothing is hidden.

**Version**: UMCP v2.2.4 · **Last updated**: 2026-03-19

This document is the **single authoritative index** of every named symbol, constant, identity, lemma, definition, theorem, class, schema, contract, and structural concept in the GENERATIVE-COLLAPSE-DYNAMICS repository. Each entry carries:

- **Tag**: Unique identifier (e.g., `K-F`, `L-12`, `I-A2`, `T0-SeamRecord`)
- **Tier**: 1 (kernel), 0 (protocol), or 2 (expansion)
- **Lineage**: Where it derives from — either `AXIOM-0` (independent) or a parent tag
- **Source**: File where it is defined or implemented

---

## Table of Contents

- [CATALOGUE — Master Index of All Formal Objects](#catalogue--master-index-of-all-formal-objects)
  - [Table of Contents](#table-of-contents)
  - [Tier-1 — The Kernel](#tier-1--the-kernel)
    - [1.1 Kernel Outputs](#11-kernel-outputs)
    - [1.2 Frozen Parameters](#12-frozen-parameters)
    - [1.3 Algebraic Identities](#13-algebraic-identities)
    - [1.4 Statistical Constraint](#14-statistical-constraint)
    - [1.5 Structural Constants](#15-structural-constants)
    - [1.6 Regime Gates](#16-regime-gates)
    - [1.7 Rank Classification](#17-rank-classification)
    - [1.8 Formal Definitions](#18-formal-definitions)
    - [1.9 Lemmas](#19-lemmas)
      - [Core Lemmas (L1–L34)](#core-lemmas-l1l34)
      - [Extended / Empirical Lemmas (L35–L46)](#extended--empirical-lemmas-l35l46)
    - [1.10 Structural Identities — Level A (Foundation)](#110-structural-identities--level-a-foundation)
    - [1.11 Structural Identities — Level B (Structure)](#111-structural-identities--level-b-structure)
    - [1.12 Structural Identities — Level C (Skeleton)](#112-structural-identities--level-c-skeleton)
    - [1.13 Structural Identities — Level D (Convergence)](#113-structural-identities--level-d-convergence)
    - [1.14 Identity Connection Clusters](#114-identity-connection-clusters)
    - [1.15 Derivation Tree](#115-derivation-tree)
  - [Tier-0 — Protocol](#tier-0--protocol)
    - [2.1 Kernel Engine](#21-kernel-engine)
    - [2.2 Seam Budget](#22-seam-budget)
    - [2.3 Thermodynamic Diagnostic](#23-thermodynamic-diagnostic)
    - [2.4 Epistemic Weld](#24-epistemic-weld)
    - [2.5 Measurement Engine](#25-measurement-engine)
    - [2.6 Validator](#26-validator)
    - [2.7 Utilities](#27-utilities)
    - [2.8 Edition Fingerprint](#28-edition-fingerprint)
    - [2.9 Schemas](#29-schemas)
    - [2.10 Contracts](#210-contracts)
    - [2.11 OPT Tags](#211-opt-tags)
    - [2.12 Spine Stops](#212-spine-stops)
    - [2.13 Five Words](#213-five-words)
    - [2.14 Typed Outcomes](#214-typed-outcomes)
    - [2.15 Three-Valued Verdicts](#215-three-valued-verdicts)
    - [2.16 Cognitive Equalizer Mechanisms](#216-cognitive-equalizer-mechanisms)
    - [2.17 Lexicon Latinum](#217-lexicon-latinum)
  - [Tier-2 — Expansion Space](#tier-2--expansion-space)
    - [3.1 Standard Model](#31-standard-model)
      - [Trace Vectors](#trace-vectors)
      - [Particle Physics Theorems (T1–T10)](#particle-physics-theorems-t1t10)
      - [Matter Genesis Theorems (T-MG-1 through T-MG-10)](#matter-genesis-theorems-t-mg-1-through-t-mg-10)
      - [Particle-Matter Map Theorems (T-PM-1 through T-PM-8)](#particle-matter-map-theorems-t-pm-1-through-t-pm-8)
      - [Neutrino Oscillation Theorems (T11–T12)](#neutrino-oscillation-theorems-t11t12)
      - [Additional Standard Model Files](#additional-standard-model-files)
    - [3.2 Atomic Physics](#32-atomic-physics)
    - [3.3 Quantum Mechanics](#33-quantum-mechanics)
      - [FQHE Bilayer Graphene (T-FQHE-1 through T-FQHE-7)](#fqhe-bilayer-graphene-t-fqhe-1-through-t-fqhe-7)
      - [Quantum Dimer Model (T-QDM series)](#quantum-dimer-model-t-qdm-series)
      - [TERS Near-Field Spectroscopy (T-TERS-1 through T-TERS-7)](#ters-near-field-spectroscopy-t-ters-1-through-t-ters-7)
      - [Atom–Dot Metal-Insulator Transition (T-ADOT-1 through T-ADOT-7)](#atomdot-metalinsulator-transition-t-adot-1-through-t-adot-7)
      - [Muon Laser Decay (T-MLD-1 through T-MLD-7)](#muon-laser-decay-t-mld-1-through-t-mld-7)
      - [Double-Slit Interference (T-DSE-1 through T-DSE-7)](#double-slit-interference-t-dse-1-through-t-dse-7)
      - [Additional QM Files](#additional-qm-files)
    - [3.4 Nuclear Physics](#34-nuclear-physics)
      - [QGP/RHIC Theorems (T-QGP-1 through T-QGP-10)](#qgprhic-theorems-t-qgp-1-through-t-qgp-10)
    - [3.5 GCD Closures](#35-gcd-closures)
      - [Kernel Structural Theorems (T-KS-1 through T-KS-7)](#kernel-structural-theorems-t-ks-1-through-t-ks-7)
    - [3.6 RCFT Closures](#36-rcft-closures)
      - [Information Geometry Theorems](#information-geometry-theorems)
      - [Quincke Rollers (T-QR-1 through T-QR-8)](#quincke-rollers-t-qr-1-through-t-qr-8)
    - [3.7 Consciousness Coherence](#37-consciousness-coherence)
    - [3.8 Awareness-Cognition](#38-awareness-cognition)
    - [3.9 Evolution](#39-evolution)
    - [3.10 Dynamic Semiotics](#310-dynamic-semiotics)
    - [3.11 Astronomy](#311-astronomy)
      - [Stellar Ages Cosmology (T-SC-1 through T-SC-10)](#stellar-ages-cosmology-t-sc-1-through-t-sc-10)
    - [3.12 Everyday Physics](#312-everyday-physics)
      - [Epistemic Coherence (T-EC-1 through T-EC-7)](#epistemic-coherence-t-ec-1-through-t-ec-7)
    - [3.13 Finance](#313-finance)
    - [3.14 Security](#314-security)
    - [3.15 Kinematics](#315-kinematics)
    - [3.16 Weyl Cosmology](#316-weyl-cosmology)
    - [3.17 Materials Science](#317-materials-science)
    - [3.18 Continuity Theory](#318-continuity-theory)
      - [Budget Geometry Theorems (T-BG-1 through T-BG-6)](#budget-geometry-theorems-t-bg-1-through-t-bg-6)
    - [3.19 Clinical Neuroscience](#319-clinical-neuroscience)
    - [3.20 Spacetime Memory](#320-spacetime-memory)
      - [Gravitational Phenomena Theorems (T-GP-1 through T-GP-6)](#gravitational-phenomena-theorems-t-gp-1-through-t-gp-6)
      - [Temporal Topology Theorems (T-TT-1 through T-TT-6)](#temporal-topology-theorems-t-tt-1-through-t-tt-6)
  - [Cross-References](#cross-references)
    - [4.1 Lineage Chains](#41-lineage-chains)
    - [4.2 Cross-Tier Dependencies](#42-cross-tier-dependencies)
  - [Summary Counts](#summary-counts)

---

## Tier-1 — The Kernel

> *Tier-1 is the mathematical function K: [0,1]ⁿ × Δⁿ → (F, ω, S, C, κ, IC) and everything provable about it. NEVER mutable within a run.*

### 1.1 Kernel Outputs

The kernel has **4 primitive equations** and **2 derived values**, with **3 effective degrees of freedom** (F, κ, C — S is asymptotically determined by F and C).

| Tag | Symbol | Name | Formula | Range | Status | Lineage |
|-----|--------|------|---------|-------|--------|---------|
| `K-F` | **F** | Fidelity | F = Σ wᵢcᵢ | [0, 1] | Primitive | AXIOM-0 |
| `K-κ` | **κ** | Log-integrity | κ = Σ wᵢ ln(cᵢ,ε) | ≤ 0 | Primitive | AXIOM-0 |
| `K-S` | **S** | Bernoulli field entropy | S = −Σ wᵢ[cᵢ ln cᵢ + (1−cᵢ) ln(1−cᵢ)] | ≥ 0 | Primitive (computed, not free) | AXIOM-0 |
| `K-C` | **C** | Curvature | C = stddev(cᵢ)/0.5 | [0, 1] | Primitive (independent) | AXIOM-0 |
| `K-ω` | **ω** | Drift | ω = 1 − F | [0, 1] | Derived from K-F | K-F |
| `K-IC` | **IC** | Integrity composite | IC = exp(κ) | (0, 1] | Derived from K-κ | K-κ |

**Source**: [KERNEL_SPECIFICATION.md](KERNEL_SPECIFICATION.md) §1, [src/umcp/kernel_optimized.py](src/umcp/kernel_optimized.py)

---

### 1.2 Frozen Parameters

> *Trans suturam congelatum* — Consistent across the seam. Seam-derived, not prescribed.

| Tag | Symbol | Name | Value | Role | Lineage | Source |
|-----|--------|------|-------|------|---------|--------|
| `FP-ε` | ε | Guard band | 10⁻⁸ | ε-clamp: no channel fully dies | Seam-derived | `frozen_contract.EPSILON` |
| `FP-p` | p | Drift cost exponent | 3 | Unique integer: Cardano root x³+x−1=0 | Seam-derived (I-D7) | `frozen_contract.P_EXPONENT` |
| `FP-α` | α | Curvature cost coefficient | 1.0 | D_C = α·C (unit coupling) | Seam-derived | `frozen_contract.ALPHA` |
| `FP-λ` | λ | Auxiliary coefficient | 0.2 | Return-rate adaptation speed | Seam-derived | `frozen_contract.LAMBDA` |
| `FP-tol` | tol_seam | Seam tolerance | 0.005 | \|s\| ≤ tol for PASS | Seam-derived | `frozen_contract.TOL_SEAM` |
| `FP-c*` | c* | Self-dual fixed point | 0.7822 | Maximizes S + κ per channel | Computed: ln((1−c)/c)+1/c=0 | `frozen_contract.C_STAR` |
| `FP-ωt` | ω_trap | Trapping threshold | 0.6823 | Γ(ω_trap) = α exactly | Computed: x³+x−1=0 | `frozen_contract.OMEGA_TRAP` |
| `FP-ct` | c_trap | Channel trapping threshold | 0.3177 | c_trap = 1 − ω_trap | Derived from FP-ωt | `frozen_contract.C_TRAP` |

**Source**: [src/umcp/frozen_contract.py](src/umcp/frozen_contract.py)

---

### 1.3 Algebraic Identities

These hold **exactly**, by construction, for all valid inputs.

| Tag | Name | Formula | Proof | Lineage |
|-----|------|---------|-------|---------|
| `AI-1` | Duality identity | F + ω = 1 | sin²θ + cos²θ = 1 in Fisher coords | AXIOM-0, K-F, K-ω |
| `AI-2` | Integrity bound | IC ≤ F | Solvability condition: c₁,₂ = F ± √(F²−IC²) | AXIOM-0, K-F, K-IC |
| `AI-3` | Log-integrity relation | IC = exp(κ) | Definition link between K-κ and K-IC | AXIOM-0, K-κ, K-IC |

**Source**: [KERNEL_SPECIFICATION.md](KERNEL_SPECIFICATION.md), verified in [closures/atomic_physics/tier1_proof.py](closures/atomic_physics/tier1_proof.py) (10,162 tests, 0 failures)

---

### 1.4 Statistical Constraint

| Tag | Name | Formula | Status | Lineage |
|-----|------|---------|--------|---------|
| `SC-1` | Entropy determination | S ≈ f(F, C); corr(C,S) → −1 as n → ∞ | Asymptotic (CLT) | K-S, K-F, K-C |

This reduces 6 kernel outputs to **3 effective degrees of freedom**: F, κ, C.

---

### 1.5 Structural Constants

Five points partition the Bernoulli manifold (identity I-D11).

| Tag | Name | Value | Fisher Angle θ/π | Role | Lineage |
|-----|------|-------|:----------------:|------|---------|
| `SC-ε` | Guard boundary | 10⁻⁸ | ≈ 0 | No channel fully dies | FP-ε |
| `SC-ct` | Trapping threshold | 0.3177 | 0.1908 | Γ(ω_trap) = 1 exactly | FP-ct, I-D7 |
| `SC-eq` | Equator | 0.5000 | 0.2500 | Max entropy, S + κ = 0 | AXIOM-0 |
| `SC-c*` | Self-dual point | 0.7822 | 0.3538 | Maximizes S + κ per channel | FP-c*, I-C1 |
| `SC-1ε` | Perfect fidelity boundary | ≈ 1 | ≈ 0.5 | Upper clamp | FP-ε |

The geodesic path ε → c_trap → ½ → c* → 1−ε sums to exactly π (identity I-D11).

---

### 1.6 Regime Gates

| Tag | Regime | Condition | Fisher Space % | Lineage |
|-----|--------|-----------|:--------------:|---------|
| `RG-S` | **Stable** | ω < 0.038 ∧ F > 0.90 ∧ S < 0.15 ∧ C < 0.14 | 12.5% | FP-tol, I-B11 |
| `RG-W` | **Watch** | 0.038 ≤ ω < 0.30 (or Stable gates not all met) | 24.4% | FP-tol, I-B11 |
| `RG-C` | **Collapse** | ω ≥ 0.30 | 63.1% | FP-tol, I-B11 |
| `RG-X` | **Critical** | IC < 0.30 (severity overlay) | — | K-IC |

Stable is conjunctive (all four gates). Critical is an overlay, not a regime.

**Source**: `frozen_contract.RegimeThresholds`, `frozen_contract.classify_regime()`

---

### 1.7 Rank Classification

Rank is a property of the trace vector — measured, not chosen (*gradus non eligitur; mensuratur*).

| Tag | Rank | DOF | Condition | Key Property | Lineage |
|-----|:----:|:---:|-----------|--------------|---------|
| `RK-1` | 1 | 1 | All cᵢ = c₀ (homogeneous) | IC = F, C = 0, Δ = 0 | Def-16, Def-19 |
| `RK-2` | 2 | 2 | Effective 2-channel structure | C = g(F, κ) determined | Def-16, Def-18 |
| `RK-3` | 3 | 3 | General heterogeneous (n ≥ 3) | F, κ, C mutually independent | Def-16, Def-17 |

Rank-1 ⊂ Rank-2 ⊂ Rank-3. Almost all real-world systems are rank-3.

**Source**: [KERNEL_SPECIFICATION.md](KERNEL_SPECIFICATION.md) §4c (Definitions 16–19)

---

### 1.8 Formal Definitions

| Tag | ID | Name | Statement (abbreviated) | Lineage |
|-----|-----|------|------------------------|---------|
| `Def-7` | Def 7 | Curvature Proxy | C(t) = std_pop({cᵢ})/0.5 ∈ [0,1] | AXIOM-0 |
| `Def-8` | Def 8 | Log-Integrity / IC | κ(t) = Σwᵢ ln cᵢ; IC(t) = exp(κ(t)) | AXIOM-0 |
| `Def-9` | Def 9 | Return-Domain Generator | D_θ(t): lookback window → candidate set U_θ(t) | AXIOM-0 |
| `Def-10` | Def 10 | Return Time τ_R | τ_R(t) = min{t−u : u ∈ U_θ(t)}; typed outcomes: ∞_rec, UNIDENTIFIABLE | Def-9 |
| `Def-11` | Def 11 | Seam Accounting | Δκ_ledger = κ(t₁)−κ(t₀); Δκ_budget = R·τ_R − (D_ω+D_C); residual s = budget−ledger | Def-10 |
| `Def-12` | Def 12 | Frozen Closure Registry | Closures pinned per run: unique names, Tier ∈ {1,2}, declared before evidence | AXIOM-0 |
| `Def-13` | Def 13 | Weld Gate | PASS requires: finite τ_R ∧ \|s\| ≤ tol ∧ identity check passes | Def-10, Def-11 |
| `Def-14` | Def 14 | Contract Equivalence | Same experiment: identical (contract, closures, sources) | Def-12 |
| `Def-16` | Def 16 | Effective Rank | rank(c,w) = number of independent kernel constraints | AI-1, AI-2, SC-1 |
| `Def-17` | Def 17 | Rank-3 (General) | 3 DOF: F, κ, C mutually independent when n ≥ 3 heterogeneous | Def-16 |
| `Def-18` | Def 18 | Rank-2 (Two-Channel) | 2 DOF: n=2 or effective bimodal; C = g(F,κ) determined | Def-16, I-A6 |
| `Def-19` | Def 19 | Rank-1 (Homogeneous) | 1 DOF: all cᵢ = c₀; IC = F, C = 0, Δ = 0 | Def-16 |

**Source**: [KERNEL_SPECIFICATION.md](KERNEL_SPECIFICATION.md) §1–§4c

---

### 1.9 Lemmas

#### Core Lemmas (L1–L34)

| Tag | ID | Name | Statement (abbreviated) | Lineage |
|-----|-----|------|------------------------|---------|
| `L-1` | L1 | F Range | F ∈ [ε, 1−ε] when cᵢ ∈ [ε, 1−ε], wᵢ > 0 | K-F, FP-ε |
| `L-2` | L2 | ω Range | ω ∈ [ε, 1−ε] | L-1, AI-1 |
| `L-3` | L3 | κ Bounds | κ ∈ [ln ε, ln(1−ε)] ≈ [−18.42, −10⁻⁸] | K-κ, FP-ε |
| `L-4` | L4 | IC = Geometric Mean | IC = exp(Σwᵢ ln cᵢ) — weighted geometric mean of cᵢ | K-IC, AI-3 |
| `L-5` | L5 | S = ln 2 iff Equator | S(t) = ln 2 ⟺ all cᵢ = ½ | K-S |
| `L-6` | L6 | IC Sensitivity | ∂IC/∂cᵢ = (wᵢ/cᵢ)·IC — single dead channel kills IC | L-4 |
| `L-7` | L7 | IC ≤ F | Weighted AM-GM on ε-clamped domain | AI-2, L-4 |
| `L-8` | L8 | Regime Well-Defined | Regime function is total: every valid (ω,F,S,C) ↦ exactly one label | RG-S through RG-C |
| `L-9` | L9 | Stable ⊂ (low ω, high F) | Stable requires ω < 0.038, F > 0.90 | RG-S |
| `L-10` | L10 | Lipschitz F | \|ΔF\| ≤ max(wᵢ)·‖Δc‖_∞ | K-F |
| `L-11` | L11 | Lipschitz κ | \|Δκ\| ≤ max(wᵢ/cᵢ)·‖Δc‖_∞ | K-κ, FP-ε |
| `L-12` | L12 | C Monotone in Spread | Wider cᵢ spread → larger C (for fixed mean) | K-C |
| `L-13` | L13 | S Lipschitz | \|ΔS\| ≤ max(wᵢ)·\|ln((1−c_min)/c_min)\|·‖Δc‖_∞ | K-S, FP-ε |
| `L-14` | L14 | F Monotone Up | F is non-decreasing in each cᵢ | K-F |
| `L-15` | L15 | κ Monotone Up | κ is non-decreasing in each cᵢ (on ε-clamped domain) | K-κ, FP-ε |
| `L-16` | L16 | IC Monotone Up | IC is non-decreasing in each cᵢ | L-15, AI-3 |
| `L-17` | L17 | ε-Clipping Bounded | \|F_clipped − F_raw\| ≤ max(wᵢ)·ε | K-F, FP-ε |
| `L-18` | L18 | κ Clipping Bounded | \|κ_clipped − κ_raw\| ≤ max(wᵢ)·\|ln(ε/c_raw)\| | K-κ, FP-ε |
| `L-19` | L19 | Stable ⇒ Low Budget | Stable regime ⇒ Γ(ω) < 10⁻⁵ | RG-S, FP-p |
| `L-20` | L20 | Seam Composition | Sequential seam deltas telescope: Σᵢ Δκᵢ = κ(t_n) − κ(t₀) | Def-11 |
| `L-21` | L21 | Residual Telescoping | Cumulative residual = Σ budget − Σ ledger = total_budget − total_ledger | L-20, Def-11 |
| `L-22` | L22 | Γ(ω) Convex | Γ is convex on [0, 1−ε) for p ≥ 2 | FP-p |
| `L-23` | L23 | Γ Pole at ω=1 | lim_{ω→1} Γ(ω) = +∞ but Γ(1−ε) = (1−ε)³/ε < ∞ | FP-p, FP-ε |
| `L-24` | L24 | Stable Attractor | Near-stable states require O(ε²) perturbation to leave Stable | RG-S, L-10, L-11 |
| `L-25` | L25 | Return Monotonicity | If τ_R(t) < ∞ and ω decreases, then τ_R can only decrease or stay constant | Def-10 |
| `L-26` | L26 | Weld Strengthening | Adding valid closures can only tighten the gate, not loosen it | Def-12, Def-13 |
| `L-27` | L27 | Residual Growth Bound | \|cumulative_residual\| ≤ K · tol_seam for K seams if each \|sᵢ\| ≤ tol_seam | L-21, FP-tol |
| `L-28` | L28 | Closure Minimality | Removing a redundant closure does not change the weld verdict | Def-12, Def-13 |
| `L-29` | L29 | Contract Monotonicity | Tightening a contract (stricter thresholds) preserves prior PASS verdicts if data unchanged | Def-14 |
| `L-30` | L30 | IC Collapse Cascade | If one cᵢ → ε while others stay fixed, IC → ε^(wᵢ)·ICrest | L-6, FP-ε |
| `L-31` | L31 | Zero-Weight Pruning | Channels with wᵢ = 0 can be removed without changing any kernel output | K-F, K-κ, K-S, K-C |
| `L-32` | L32 | Weight Lipschitz | \|ΔF\| ≤ max(cᵢ)·‖Δw‖₁ for F; similar for κ, S | K-F |
| `L-33` | L33 | IC/F Ratio Bound | IC/F ∈ [ε^(1−min wᵢ)/F, 1] — bounded away from zero by guard band | AI-2, FP-ε |
| `L-34` | L34 | Entropy-Curvature Coupling | In the homogeneous limit (C→0): S → h(F), IC → F | SC-1, AI-2 |

#### Extended / Empirical Lemmas (L35–L46)

| Tag | ID | Name | Statement (abbreviated) | Lineage |
|-----|-----|------|------------------------|---------|
| `L-35` | L35 | Return-Collapse Duality | If τ_R is finite and Type-I (drift-dominated): τ_R ≈ D_C | Def-10, Def-11 |
| `L-36` | L36 | Generative Flux Bound | Φ_gen = Φ_collapse·(1−S) ≤ Γ(ω)·(1−S) | K-S, L-22 |
| `L-37` | L37 | Unitarity-Horizon Transition | Δκ ≈ 0.1 marks the boundary where cumulative return cost exceeds local return credit | Def-11, FP-tol |
| `L-38` | L38 | Universal Horizon Deficit | Cross-domain mean IC at horizon deficit ≈ 0.947 | K-IC (empirical) |
| `L-39` | L39 | Super-Exponential Convergence | IC convergence to F is faster than exponential as C → 0 | AI-2, I-B3 |
| `L-40` | L40 | Stable Attractor Radius | Stable regime is an attractor: IC/F > 0.95, ω < 0.05 basin width | RG-S, L-24 |
| `L-41` | L41 | Entropy-Integrity Anti-Correlation | S + κ ≤ ln 2; equality at c = ½; max(S+κ) = ln 2 − ½ at c* | K-S, K-κ, I-D1 |
| `L-42` | L42 | Coherence-Entropy Product | IC·exp(S) ≤ 1; equality iff uniform trace at c = ½ | AI-3, K-S |
| `L-43` | L43 | RCFT Convergence | Recursive application of kernel to its own outputs converges to a fixed point | K-F, K-IC |
| `L-44` | L44 | Fractal Return Scaling | Return distances scale as power law in recursion depth | L-43 |
| `L-45` | L45 | Seam Residual Algebra | Seam composition forms a monoid: associative with identity, verified to 5.55×10⁻¹⁷ | L-20, I-C8 |
| `L-46` | L46 | Weld Closure Composition | Composed welds satisfy: PASS₁ ∧ PASS₂ ⇒ PASS₁₂ if residuals are within tolerance | Def-13, L-27 |
| `L-47` | L47 | Geometric Slaughter (Cross-Domain) | One dead channel (cₖ → ε) kills IC while F stays healthy: IC/F → ε^(wₖ) · (IC_rest/F). Observed in 8/20 domains at phase boundaries (confinement, EWSB, cortical lesion). The drop is exactly IC_dead = ε^(1/n)·c₀^((n-1)/n) for uniform traces (cf. T-KS-1). | L-6, L-30, AI-2 |

**Source**: [KERNEL_SPECIFICATION.md](KERNEL_SPECIFICATION.md) §4 (L1–L34), §4b (L35–L46), cross-domain (L47)

---

### 1.10 Structural Identities — Level A (Foundation)

> *A ligat.* — A binds. 1 step from the kernel: direct from K, not derivable from any other identity. These are the irreducible roots.

**Classification**: The 44 structural identities are organized by **derivation distance from Axiom-0**:
- **Level A** (7): 1 step — direct from kernel definitions, mutually independent
- **Level B** (15): 2 steps — first derivations from A
- **Level C** (11): 3 steps — combinations and specializations of B
- **Level D** (11): 4 steps — convergences and cross-connections of C

Verification: `scripts/deep_diagnostic.py`, `scripts/cross_domain_bridge.py`, `scripts/cross_domain_bridge_phase2.py`, `scripts/identity_verification.py`, `scripts/identity_deep_probes.py`

| Tag | Name | Formula | Meaning | Derivation from K | Legacy |
|-----|------|---------|---------|-------------------|--------|
| `I-A1` | Duality | F + ω = 1; Σwᵢ[sin²θᵢ+cos²θᵢ] = 1 | Duality identity in Fisher coordinates | Substitution: ω := 1−F | B1 |
| `I-A2` | Integrity Bound | IC ≤ F; equality iff all cᵢ equal | Solvability condition for trace recovery | Jensen: ln concave → κ ≤ ln F → exp(κ) ≤ F | B2 |
| `I-A3` | Fano-Fisher | h″(c) = −g_F(c) = −1/[c(1−c)] | Entropy curvature IS the Fisher metric | Differentiation: 2nd derivative of Bernoulli entropy | N5 |
| `I-A4` | Fisher Flatness | g_F(θ) = 1 (flat manifold) | All structure from embedding, not curvature | Differentiation: Fisher metric of Bernoulli in θ = arcsin√c | D1 |
| `I-A5` | κ as Log-Sine | κ = Σwᵢ ln(sin²θᵢ) = 2Σwᵢ ln\|sin θᵢ\| | Log-integrity is logarithmic sine on the sphere | Substitution: c = sin²θ into κ-definition | D3 |
| `I-A6` | Rank-2 Closed Form | IC = √(F²−C²/4) for n=2 | Exact analytical solution for 2-channel | Algebra: specialize definitions to 2 channels | N3 |
| `I-A7` | Integral Conservation | ∫₀¹ [h(c)+ln(c)] dc = −½ | Average S+κ is exactly −½ | Integration: direct on h(c)+ln(c) | E4 |

**What Level A establishes**: The kernel's **constraints** (A1, A2), its **geometry** (A3, A4, A5), its **exact solution** at rank-2 (A6), and its **integral average** (A7). Every other identity descends from these 7 roots.

---

### 1.11 Structural Identities — Level B (Structure)

> *B format.* — B shapes. 2 steps from K: each requires one Level A identity + one additional operation. This is where outputs start talking to each other.

| Tag | Name | Formula | Meaning | Derived from | Legacy |
|-----|------|---------|---------|-------------|--------|
| `I-B1` | Coupling Function | f(θ) = S(θ)+κ(θ) = 2cos²θ·ln(tan θ) | S and κ are projections of one function (< 10⁻¹⁶) | I-A4 + I-A5: write S+κ in flat Fisher coords | D2 |
| `I-B2` | Solvability | c₁,₂ = F ± √(F²−IC²); real iff IC ≤ F | Integrity bound IS the solvability condition | I-A6: invert rank-2 formula | D4 |
| `I-B3` | Log-Integrity Correction | κ = ln F − C²/(8F²) + O(C⁴) | Taylor: heterogeneity correction always negative | I-A6: Taylor-expand rank-2 IC | N8 |
| `I-B4` | Jensen Entropy Bound | S ≤ h(F); equality iff C = 0 | Entropy companion to integrity bound | I-A3: h concave → Jensen | N10 |
| `I-B5` | Curvature = Fisher + Pole | f″(c) = −1/[c(1−c)] − 1/c² | Coupling curvature decomposes cleanly | I-A3: h″ = −g_F, add (ln c)″ = −1/c² | B8 |
| `I-B6` | Asymptotic IC-Curvature | IC² ≈ F² − β_n·C²; β₂=¼ (exact), β_∞≈0.30 | IC² deviation scales linearly with C² | I-A2 + I-A6: generalize rank-2 to asymptotic | N7 |
| `I-B7` | Fisher-Entropy Integral | ∫₀¹ g_F(c)·S(c) dc = π²/3 = 2ζ(2) | Kernel geometry tied to Basel constant | I-A3 + I-A7: integration by parts (h″ = −g_F) | N1 |
| `I-B8` | Coupling Centroid | ∫₀¹ (S+κ)·c dc = 0 | Coupling function has zero c-weighted centroid | I-A7: c-weighted moment of coupling integral | N2 |
| `I-B9` | Gap Taylor Expansion | Δ ≈ σ²/(2F) = C²/(8F) | Leading-order heterogeneity gap approximation | I-A6: leading Taylor of gap F − √(F²−C²/4) | N15 |
| `I-B10` | Equator Quintuple | At c=½: S=ln 2, S+κ=0, h′=0, g_F=4, θ=π/4 | Five properties converge simultaneously | I-A1 through I-A7: evaluate all at c = ½ | N4 |
| `I-B11` | Regime Partition | Collapse 63% / Watch 24% / Stable 12.5% | Stability is rare — 87.5% outside | I-A1: ω = 1−F + frozen gates → volume computation | D7 |
| `I-B12` | IC Democracy | CV of IC-drop ≈ 7×10⁻⁴ upon channel kill | Any channel kill drops IC by same amount | I-A2: differentiate IC → ∂IC/∂cᵢ equal | B12 |
| `I-B13` | Fisher Volume | Z = −Σwᵢ ln[cᵢ(1−cᵢ)] = Σwᵢ ln[4·g_F(cᵢ)] | Partition-function analog is log-Fisher-volume | I-A4: g_F → log-Fisher-volume | B4 |
| `I-B14` | Fisher-Geometric Scaling | Δ and Var(arcsin(√c)) co-vanish and co-rise | Heterogeneity gap tracks Fisher angle variance | I-A2 + I-A4: gap tracks Fisher angle variance | E7 |
| `I-B15` | Low-Rank Closures | 5 diagnostics → 4 effective dimensions (PCA) | Closure algebra is lower-dimensional | I-A1 + I-A2 + I-A6: constraint-counting on closures | B9 |

**What Level B establishes**: The **coupling function** (B1), **solvability** (B2), **perturbation theory** (B3), the **dual bound** (B4), **curvature decomposition** (B5), the **spectral integrals** (B7, B8), the **regime landscape** (B11), and the synthesis point at the **equator** (B10).

---

### 1.12 Structural Identities — Level C (Skeleton)

> *C invenit.* — C discovers. 3 steps from K: combinations of B identities that reveal the manifold's skeleton — its special points, composition rules, and complete moment theory.

| Tag | Name | Formula | Meaning | Derived from | Legacy |
|-----|------|---------|---------|-------------|--------|
| `I-C1` | Logistic Self-Duality | c\* = σ(1/c\*) where σ(x) = 1/(1+e⁻ˣ); c\* ≈ 0.7822 | Coupling peak is the logistic-reciprocal fixed point | I-B1: optimize coupling f → logistic fixed point | E1 |
| `I-C2` | Reflection Formula | f(θ)+f(π/2−θ) = 2ln(tan θ)cos(2θ) | Equator θ=π/4 is double zero; bridges c\* and c_trap | I-B1: reflection symmetry of coupling | N16 |
| `I-C3` | Composition Law | IC₁₂ = √(IC₁·IC₂); F₁₂ = (F₁+F₂)/2 | IC geometric, F arithmetic | I-B2: compose paired kernel evaluations | B3 |
| `I-C4` | Self-Dual Maximum | f(c\*) = 2cos²θ\*·ln(tan θ\*) | Coupling peak in Fisher coordinates | I-B1: evaluate coupling at peak | B6 |
| `I-C5` | Cubic Trapping | Γ(ω_trap) = 1; ω_trap: x³+x−1=0 (Cardano) | Below c_trap budget cannot close without R > Γ | I-B11: budget threshold from regime structure | B7 |
| `I-C6` | Cost Elasticity | ε_Γ = ω·Γ′/Γ = ω(3−2ω)/(1−ω); → 4 near pole | Effective critical exponent for budget blowup | I-B5: differentiate budget cost function | B11 |
| `I-C7` | Sandwich Theorem | IC ≤ F ∧ S ≤ h(F); both exact iff C = 0 | Kernel sandwiched: integrity below, entropy above | I-B4 + I-A2: combine dual bounds | N14 |
| `I-C8` | Dimension Collapse | 8-channel kernel: ℝ⁸ → ℝ⁴ | Kernel halves dimensionality | I-B12 + I-B6: constraint counting on 8 channels | D8 |
| `I-C9` | Budget Cost Crossover | ∃ ω_cross ≈ 0.58: Γ(ω) = E[D_C\|ω] | Below: curvature dominates; above: drift dominates | I-B3 + I-C5: where Γ(ω) = E[D_C\|ω] | N9 |
| `I-C10` | Moment Family | μ_n = [(n+1)H_{n+1}−(n+2)]/[(n+1)²(n+2)] | General closed form with harmonic numbers | I-B7 + I-B8: generalize integrals to all moments | N11 |
| `I-C11` | Entropy Moment Table | ∫₀¹ cⁿ·S dc = 1/(n+2)² + H_{n+1}/(n+1) − H_{n+2}/(n+2) | Rational closed form for entropy moments | I-B7: specialize Fisher-entropy integral | N13 |

**What Level C establishes**: The **critical point** c\* (C1), **reflection symmetry** (C2), the **composition algebra** (C3), the **budget threshold** (C5), and the **full spectral family** (C10, C11).

---

### 1.13 Structural Identities — Level D (Convergence)

> *D componit.* — D composes. 4 steps from K: convergences and cross-connections where the system's special points, thresholds, and algebras reference each other.

| Tag | Name | Formula | Meaning | Derived from | Legacy |
|-----|------|---------|---------|-------------|--------|
| `I-D1` | Coupling Maximum | max(S+κ) = (1−c\*)/c\* = exp(−1/c\*) ≈ 0.278 | Coupling peak value: odds ratio = exponential at c\* | I-C1: evaluate coupling at fixed point | E2 |
| `I-D2` | Log-Odds Reciprocal | ln(c\*/(1−c\*)) = 1/c\* | At coupling max, log-odds equals reciprocal | I-C1: log-odds at fixed point | E3 |
| `I-D3` | Triple Peak | (1−c\*)/c\* = exp(−1/c\*) = (S+κ)\|_{c\*} | Three quantities converge at coupling peak | I-C1 + I-D1 + I-D2: synthesis | N6 |
| `I-D4` | Curvature Decomposition | f″(c) = −g_F(c) − 1/c²; 78.2% Fisher / 21.8% pole | Numerical decomposition at c\* | I-C1 + I-B5: evaluate curvature decomposition at c\* | E5 |
| `I-D5` | Structural Gap | ln 2 − max(S+κ) ≈ 0.415; 59.8% of ln 2 | Gap between equator maximum and coupling peak | I-D1 + I-B10: gap between equator and peak | E6 |
| `I-D6` | Omega Hierarchy | ω_stable < ω\* < ω_collapse < ω_trap (0.038 < 0.218 < 0.300 < 0.682) | Coupling peak in Watch regime; thresholds in strict order | I-C1 + I-C5 + I-B11: ordering of all thresholds | E8 |
| `I-D7` | p=3 Cardano | ω_trap is root of x³+x−1=0; discriminant Δ=−31 < 0 | p=3 is the UNIQUE integer with Cardano structure | I-C5: analyze trapping equation → unique integer | D5 |
| `I-D8` | Composition Invariance | Δ(n identical copies) = Δ(1 copy) exactly | Heterogeneity gap invariant under replication | I-C3: compose identical copies → gap unchanged | D6 |
| `I-D9` | Gap Composition | Δ₁₂ = (Δ₁+Δ₂)/2 + (√IC₁−√IC₂)²/2 | Hellinger-like correction; gap grows for unequal IC | I-C3: compose UNequal copies → Hellinger correction | N12 |
| `I-D10` | Budget Conservation | Δκ = R·τ_R − (D_ω+D_C); \|Δκ\| ≤ tol_seam | Seam budget must reconcile | I-C3 + I-C5: composition + threshold → seam algebra | B5 |
| `I-D11` | Geodesic Partition | {ε, c_trap, ½, c\*, 1−ε} partition [0, π] in Fisher space | Structural constants partition the half-circle | I-C1 + I-C5 + I-B10: all constants in Fisher space | B10 |

**What Level D establishes**: The **convergence phenomena** (D1–D5), the **complete threshold ordering** (D6), the **Cardano uniqueness** of p=3 (D7), the **Hellinger composition algebra** (D8, D9), and the **geodesic partition** of the manifold (D11).

---

### 1.14 Identity Connection Clusters

The 44 identities (A: 7, B: 15, C: 11, D: 11) form a network with 6 computationally verified clusters. Cluster members now reference the derivation-level tags.

| Tag | Cluster | Members | Structural Insight | Lineage |
|-----|---------|---------|-------------------|---------|
| `CC-1` | Equator Web | I-C1, I-B10, I-C2, I-D6 | c=½ is quintuple fixed point; entropy-integrity cancellation | Levels B–D |
| `CC-2` | Dual Bounding Pair | I-A2, I-B4 | IC ≤ F below, S ≤ h(F) above; both exact iff C=0 | Levels A–B |
| `CC-3` | Perturbation Chain | I-A6 → I-B3 → I-A2 | Integrity bound proved from kernel's Taylor structure | Levels A–B |
| `CC-4` | Composition Algebra | I-D8, I-D9, I-C8 | Monoid structure with Hellinger-like gap correction | Levels C–D |
| `CC-5` | Fixed-Point Triangle | I-D1/I-D2, I-D3, I-B10 | Manifold skeleton: equator + c\* + c_trap | Levels B–D |
| `CC-6` | Spectral Family | I-A7, I-B7, I-B8, I-C10 | f = S+κ spectrally complete; ∫g_F·S dc = π²/3 | Levels A–C |

**Source**: `scripts/identity_connections.py`

---

### 1.15 Derivation Tree

> *A ligat, B format, C invenit, D componit.* — A binds, B shapes, C discovers, D composes.

The 7 + 15 + 11 + 11 = 44 identities form a tree of depth 4 from Axiom-0. Legacy IDs (E/B/D/N series) are retained in the Legacy column for backward compatibility.

```
AXIOM-0 → KERNEL K
              │
    ┌─────────┼──────────┬──────────┬──────────┬──────────┬────────────┐
    ▼         ▼          ▼          ▼          ▼          ▼            ▼
   A1        A2         A3         A4         A5         A6           A7
 (duality) (bound)  (h″=-g_F)  (flat)    (κ sinθ)  (rank-2)     (∫=-½)
    │         │         │       ╲  │          │          │            │
    ▼         │         ▼        ╲ │          │          ▼            ▼
  B11        │        B4         B1◄─────────┘         B2       B7, B8
 (regime)    │       (S≤h)    (coupling)               │            │
    │        │         │          │                     ▼            ▼
    ▼        ▼         ▼          ├──────────┐         C3      C10, C11
   C5    B6,B12    C7(◄A2)       ▼          ▼       (compose)  (moments)
 (trap)    │      (sandwich)    C1         C2          │
    │      ▼                 (fixed pt)  (reflect)     ├─────────┐
    ▼      │                     │                     ▼         ▼
   D7      │                  D1, D2                  D8        D9
 (p=3)     │                     │                 (invariant) (Hellinger)
           │                     ▼
    ▲      │                   D3, D5
    │      │                 (convergences)
    │      │                     │
    │      └─────────────────────▼
    └──────────────────────────►D6, D11
                              (ordering + geodesic partition)
```

| Level | Latin | Count | Role |
|:-----:|:-----:|:-----:|------|
| **A** | *ligat* | 7 | **Foundation**: constraints, geometry, exact solution, integral average |
| **B** | *format* | 15 | **Structure**: outputs interconnect; coupling, perturbation, spectra, regimes |
| **C** | *invenit* | 11 | **Skeleton**: critical point, reflection, composition, complete moments |
| **D** | *componit* | 11 | **Convergence**: special points, thresholds, algebras cross-reference |

**Legacy → Derivation Tag Cross-Reference** (for backward compatibility):

| Legacy | → New | Legacy | → New | Legacy | → New | Legacy | → New |
|--------|-------|--------|-------|--------|-------|--------|-------|
| E1 | I-C1 | B1 | I-A1 | D1 | I-A4 | N1 | I-B7 |
| E2 | I-D1 | B2 | I-A2 | D2 | I-B1 | N2 | I-B8 |
| E3 | I-D2 | B3 | I-C3 | D3 | I-A5 | N3 | I-A6 |
| E4 | I-A7 | B4 | I-B13 | D4 | I-B2 | N4 | I-B10 |
| E5 | I-D4 | B5 | I-D10 | D5 | I-D7 | N5 | I-A3 |
| E6 | I-D5 | B6 | I-C4 | D6 | I-D8 | N6 | I-D3 |
| E7 | I-B14 | B7 | I-C5 | D7 | I-B11 | N7 | I-B6 |
| E8 | I-D6 | B8 | I-B5 | D8 | I-C8 | N8 | I-B3 |
| — | — | B9 | I-B15 | — | — | N9 | I-C9 |
| — | — | B10 | I-D11 | — | — | N10 | I-B4 |
| — | — | B11 | I-C6 | — | — | N11 | I-C10 |
| — | — | B12 | I-B12 | — | — | N12 | I-D9 |
| — | — | — | — | — | — | N13 | I-C11 |
| — | — | — | — | — | — | N14 | I-C7 |
| — | — | — | — | — | — | N15 | I-B9 |
| — | — | — | — | — | — | N16 | I-C2 |

---

## Tier-0 — Protocol

> *Tier-0 is the operational machinery: code that implements Tier-1 plus embedding, gates, seam calculus, contracts, schemas, and three-valued verdicts.*

### 2.1 Kernel Engine

**Source**: [src/umcp/kernel_optimized.py](src/umcp/kernel_optimized.py)

| Tag | Object | Type | Purpose |
|-----|--------|------|---------|
| `T0-KernelOutputs` | `KernelOutputs` | dataclass | F, ω, S, C, κ, IC, heterogeneity_gap, regime, is_homogeneous, computation_mode |
| `T0-ErrorBounds` | `ErrorBounds` | dataclass | Lipschitz error bounds for F, ω, κ, S |
| `T0-OptKernel` | `OptimizedKernelComputer` | class | Main kernel engine with OPT-1/2/3/4/12 |
| `T0-compute` | `compute()` | method | Core: c, w → (F, ω, S, C, κ, IC) |
| `T0-valBounds` | `validate_kernel_bounds()` | function | Identity checks for validator |

---

### 2.2 Seam Budget

**Source**: [src/umcp/seam_optimized.py](src/umcp/seam_optimized.py)

| Tag | Object | Type | Purpose |
|-----|--------|------|---------|
| `T0-SeamRecord` | `SeamRecord` | dataclass | t0, t1, κ values, τ_R, Δκ_ledger, Δκ_budget, residual, cumulative |
| `T0-SeamMetrics` | `SeamChainMetrics` | dataclass | total_seams, total_Δκ, cumulative_abs_residual, max/mean residual, growth_exponent |
| `T0-SeamAccum` | `SeamChainAccumulator` | class | Incremental ledger + residual monitoring (OPT-10/11) |
| `T0-SeamComp` | `SeamCompositionAnalyzer` | class | Validates L-20 composition law |

**Formulas implemented**:
- Δκ_ledger = κ(t₁) − κ(t₀)
- Δκ_budget = R·τ_R − (D_ω + D_C)
- residual s = budget − ledger
- D_ω = Γ(ω) = ω^p/(1−ω+ε)
- D_C = α·C

---

### 2.3 Thermodynamic Diagnostic

**Source**: [src/umcp/tau_r_star.py](src/umcp/tau_r_star.py)

| Tag | Object | Type | Purpose |
|-----|--------|------|---------|
| `T0-Phase` | `ThermodynamicPhase` | enum | SURPLUS, DEFICIT, FREE_RETURN, TRAPPED, POLE |
| `T0-Dominance` | `DominanceTerm` | enum | DRIFT, CURVATURE, MEMORY |
| `T0-TauResult` | `TauRStarResult` | NamedTuple | τ_R*, γ, D_C, Δκ, R, numerator |
| `T0-ThermDiag` | `ThermodynamicDiagnostic` | dataclass | ~25 fields: full diagnostic |
| `T0-tauRStar` | `compute_tau_R_star()` | function | τ_R* = (Γ(ω)+αC+Δκ)/R |
| `T0-Rcrit` | `compute_R_critical()` | function | R_crit = numerator/tol_seam |
| `T0-trapped` | `is_trapped()` | function | Check Γ(ω) ≥ α (Thm T3) |
| `T0-diagnose` | `diagnose()` | function | Full Tier-2 diagnostic pipeline |

**Internal theorems**: Def T1–T6, Thm T1–T9, Foundational F1–F5

---

### 2.4 Epistemic Weld

**Source**: [src/umcp/epistemic_weld.py](src/umcp/epistemic_weld.py)

| Tag | Object | Type | Purpose |
|-----|--------|------|---------|
| `T0-EpVerdict` | `EpistemicVerdict` | enum | RETURN, GESTURE, DISSOLUTION |
| `T0-GestReason` | `GestureReason` | enum | SEAM_RESIDUAL_EXCEEDED, NO_FINITE_RETURN, IDENTITY_MISMATCH, FROZEN_PARAMETER_DRIFT, TIER0_INCOMPLETE |
| `T0-PosIllusion` | `PositionalIllusion` | NamedTuple | Zeno-analog cost: total_cost = N × Γ(ω) |
| `T0-EpTrace` | `EpistemicTraceMetadata` | dataclass | n_components, n_timesteps, ε_floor, clipped info, verdict |
| `T0-SeamEp` | `SeamEpistemology` | dataclass | verdict, reasons, seam_residual, seam_budget, τ_R, regime, ω, illusion |
| `T0-classifyEp` | `classify_epistemic_act()` | function | Core: seam_pass + τ_R + regime → verdict + reasons |
| `T0-posIllusion` | `quantify_positional_illusion()` | function | Theorem T9: observation cost |

---

### 2.5 Measurement Engine

**Source**: [src/umcp/measurement_engine.py](src/umcp/measurement_engine.py)

| Tag | Object | Type | Purpose |
|-----|--------|------|---------|
| `T0-EmbedStrat` | `EmbeddingStrategy` | enum | LINEAR_SCALE, MIN_MAX, MAX_NORM, ZSCORE_SIGMOID |
| `T0-TraceRow` | `TraceRow` | dataclass | t, c, oor, miss |
| `T0-InvRow` | `InvariantRow` | dataclass | t, ω, F, S, C, τ_R, κ, IC, regime, critical_overlay |
| `T0-EngResult` | `EngineResult` | dataclass | trace, invariants, weights, n_dims, n_timesteps, diagnostics |
| `T0-MeasEngine` | `MeasurementEngine` | class | Raw data → Ψ(t) → invariants pipeline |

**Pipeline**: raw data → embedding (normalize [0,1]) → clip [ε,1−ε] → Ψ(t) trace → kernel → τ_R → regime → artifacts

---

### 2.6 Validator

**Source**: [src/umcp/validator.py](src/umcp/validator.py)

| Tag | Object | Type | Purpose |
|-----|--------|------|---------|
| `T0-RootVal` | `RootFileValidator` | class | Validates 16 required UMCP root files |
| `T0-valAll` | `validate_all()` | method | Full validation pipeline |
| `T0-valIdent` | `_validate_invariant_identities()` | method | Checks F=1−ω, IC≈exp(κ), IC≤F |
| `T0-valChk` | `_validate_checksums()` | method | SHA-256 integrity check |

**16 required files**: manifest.yaml, contract.yaml, observables.yaml, embedding.yaml, return.yaml, closures.yaml, weights.csv, derived/trace.csv, derived/trace_meta.yaml, outputs/invariants.csv, outputs/regimes.csv, outputs/welds.csv, outputs/report.txt, integrity/sha256.txt, integrity/env.txt, integrity/code_version.txt

---

### 2.7 Utilities

**Source**: [src/umcp/compute_utils.py](src/umcp/compute_utils.py), [src/umcp/uncertainty.py](src/umcp/uncertainty.py)

| Tag | Object | Type | Purpose | Source |
|-----|--------|------|---------|--------|
| `T0-PruneResult` | `PruningResult` | dataclass | Zero-weight channel removal (L-31) | compute_utils.py |
| `T0-ClipResult` | `ClippingResult` | dataclass | ε-clipping to [ε, 1−ε] (L-17) | compute_utils.py |
| `T0-KernGrad` | `KernelGradients` | NamedTuple | ∂F/∂c, ∂ω/∂c, ∂S/∂c, ∂κ/∂c, ∂C/∂c | uncertainty.py |
| `T0-UncBounds` | `UncertaintyBounds` | dataclass | Uncertainty propagation (delta-method) | uncertainty.py |

---

### 2.8 Edition Fingerprint

**Source**: [src/umcp/ss1m_triad.py](src/umcp/ss1m_triad.py)

| Tag | Object | Type | Purpose |
|-----|--------|------|---------|
| `T0-Triad` | `EditionTriad` | dataclass | (C1, C2, C3) edition fingerprint |
| `T0-TriadCounts` | `EditionCounts` | NamedTuple | pages, figures, tables, equations, references |
| `T0-computeTriad` | `compute_triad()` | function | C1 = (P+F+T+E+R) mod 97, etc. |

---

### 2.9 Schemas

17 JSON Schema (Draft 2020-12) files in `schemas/`:

| Tag | Schema | Validates |
|-----|--------|-----------|
| `SCH-01` | canon.anchors.schema.json | Canon anchor files |
| `SCH-02` | canon.domain_anchors.schema.json | Domain-specific anchors |
| `SCH-03` | canon.schema.json | Canon structure |
| `SCH-04` | closures.schema.json | Closure declarations |
| `SCH-05` | closures_registry.schema.json | Closure registry |
| `SCH-06` | contract.schema.json | Contract YAML |
| `SCH-07` | embedding.schema.json | Embedding config |
| `SCH-08` | failure_node_atlas.schema.json | Failure node atlas |
| `SCH-09` | glossary.schema.json | Glossary |
| `SCH-10` | invariants.schema.json | Kernel invariants output |
| `SCH-11` | manifest.schema.json | Manifest YAML |
| `SCH-12` | receipt.ss1m.schema.json | SS1M receipt |
| `SCH-13` | return.schema.json | Return config |
| `SCH-14` | trace.psi.schema.json | Ψ(t) trace CSV |
| `SCH-15` | validator.result.schema.json | Validation result |
| `SCH-16` | validator.rules.schema.json | Validation rules |
| `SCH-17` | weights.schema.json | Weight vector |

---

### 2.10 Contracts

13 domain contracts in `contracts/`:

| Tag | Contract | Domain |
|-----|----------|--------|
| `CON-UMA` | UMA.INTSTACK.v1.yaml | Universal (meta-contract) |
| `CON-GCD` | GCD.INTSTACK.v1.yaml | Generative Collapse Dynamics |
| `CON-RCFT` | RCFT.INTSTACK.v1.yaml | Recursive Collapse Field Theory |
| `CON-SM` | SM.INTSTACK.v1.yaml | Standard Model |
| `CON-QM` | QM.INTSTACK.v1.yaml | Quantum Mechanics |
| `CON-NUC` | NUC.INTSTACK.v1.yaml | Nuclear Physics |
| `CON-ATOM` | ATOM.INTSTACK.v1.yaml | Atomic Physics |
| `CON-ASTRO` | ASTRO.INTSTACK.v1.yaml | Astronomy |
| `CON-KIN` | KIN.INTSTACK.v1.yaml | Kinematics |
| `CON-WEYL` | WEYL.INTSTACK.v1.yaml | Weyl Cosmology |
| `CON-FIN` | FINANCE.INTSTACK.v1.yaml | Finance |
| `CON-MATL` | MATL.INTSTACK.v1.yaml | Materials Science |
| `CON-SEC` | SECURITY.INTSTACK.v1.yaml | Security |

---

### 2.11 OPT Tags

OPT-* tags in code reference proven lemmas from KERNEL_SPECIFICATION.md.

| Tag | Reference | Lemma(s) | Purpose |
|-----|-----------|----------|---------|
| `OPT-1` | kernel_optimized.py | L-34 | Homogeneity detection (40% speedup for rank-1) |
| `OPT-2` | kernel_optimized.py | L-1, L-2 | Range validation for F, ω |
| `OPT-3` | kernel_optimized.py | L-7 | Heterogeneity gap Δ = F − IC |
| `OPT-4` | kernel_optimized.py | L-3 | Log-space κ computation |
| `OPT-10` | seam_optimized.py | L-20 | Incremental ledger (seam telescoping) |
| `OPT-11` | seam_optimized.py | L-27 | Residual growth monitoring |
| `OPT-12` | kernel_optimized.py | L-10, L-11 | Lipschitz error propagation |
| `OPT-17` | compute_utils.py | L-31 | Zero-weight channel pruning |
| `OPT-20` | compute_utils.py | L-17 | Coordinate clipping bounds |

---

### 2.12 Spine Stops

The fixed five-stop discourse structure. Every claim passes through these.

| Tag | Stop | Role | Grammatical Function |
|-----|------|------|---------------------|
| `SP-1` | **Contract** | Define before evidence | Declares rules before the sentence is written |
| `SP-2` | **Canon** | Tell the story (five words) | The narrative body |
| `SP-3` | **Closures** | Publish thresholds | Grammar rules: stance must change when crossed |
| `SP-4` | **Integrity Ledger** | Debit/credit reconciliation | Proof the sentence is well-formed |
| `SP-5` | **Stance** | Read verdict from gates | Derived, never asserted |

**Governance** (punctuate the spine, not part of it):
- **Manifest** (*manifestum*): Provenance — binds artifacts to time, tools, checksums
- **Weld** (*sutura*): Continuity across change — named anchor, pre/post tests, κ-continuity

---

### 2.13 Five Words

The minimal prose interface for narrating the Canon (SP-2).

| Tag | Word | Latin | Ledger Role | Operational Meaning |
|-----|------|-------|-------------|---------------------|
| `FW-1` | **Drift** | *derivatio* | Debit D_ω | What moved — salient change relative to Contract |
| `FW-2` | **Fidelity** | *fidelitas* | — | What persisted — structure that survived |
| `FW-3` | **Roughness** | *curvatura* | Debit D_C | Where/why it was bumpy — friction, confound |
| `FW-4` | **Return** | *reditus* | Credit R·τ_R | Credible re-entry to legitimacy |
| `FW-5` | **Integrity** | *integritas* | Verdict | Does it hang together — never asserted, always derived |

---

### 2.14 Typed Outcomes

| Tag | Value | Python Repr | Meaning |
|-----|-------|-------------|---------|
| `TO-inf` | ∞_rec (INF_REC) | `float("inf")` | No return — permanent detention; zero budget credit |
| `TO-uid` | UNIDENTIFIABLE | sentinel | Return domain empty — cannot determine τ_R |
| `TO-oor` | ⊥_oor | sentinel | Out-of-range — domain/typing violation |

**Rule**: INF_REC stays as typed string `"INF_REC"` in data files (CSV/YAML/JSON). Never silently coerced to a number.

---

### 2.15 Three-Valued Verdicts

| Tag | Verdict | CLI Exit | Meaning |
|-----|---------|:--------:|---------|
| `TV-C` | CONFORMANT | 0 | Passes all checks |
| `TV-N` | NONCONFORMANT | 1 | Fails one or more checks |
| `TV-E` | NON_EVALUABLE | — | Insufficient data for verdict |

*Numquam binarius; tertia via semper patet.* — Never boolean; the third way is always open.

---

### 2.16 Cognitive Equalizer Mechanisms

> *Non agens mensurat, sed structura.* — Not the agent measures, but the structure.

| Tag | Mechanism | What It Externalizes | Lineage |
|-----|-----------|---------------------|---------|
| `CE-1` | Frozen Contract | Threshold selection → frozen parameters | FP-* |
| `CE-2` | The Spine | Methodology → five mandatory stops | SP-1 through SP-5 |
| `CE-3` | Five Words | Vocabulary → operationally defined terms | FW-1 through FW-5 |
| `CE-4` | Regime Gates | Conclusions → three-valued verdicts from gates | RG-S through RG-X |
| `CE-5` | Integrity Ledger | Judgment → debit/credit reconciliation | Def-11 |
| `CE-6` | Orientation | Calibration → re-derivation (same numbers = same understanding) | AXIOM-0 |

---

### 2.17 Lexicon Latinum

13 canonical Latin terms from LIBER_COLLAPSUS.tex.

| Tag | Latin | Symbol | Operational Seed |
|-----|-------|--------|-----------------|
| `LL-01` | Fidelitas | F | *quid supersit post collapsum* — what survives collapse |
| `LL-02` | Derivatio | ω | *quantum collapsu deperdatur* — measured departure from fidelity |
| `LL-03` | Entropia | S | *incertitudo campi collapsus* — Bernoulli field entropy |
| `LL-04` | Curvatura | C | *coniunctio cum gradibus libertatis* — coupling to uncontrolled DOF |
| `LL-05` | Log-Integritas | κ | *sensibilitas logarithmica* — logarithmic sensitivity of coherence |
| `LL-06` | Integritas Composita | IC | *cohaerentia multiplicativa* — multiplicative coherence |
| `LL-07` | Moratio Reditus | τ_R | *tempus reentrandi* — detention before re-entry |
| `LL-08` | Auditus | — | Hearing/audit: the ledger hears everything |
| `LL-09` | Casus | — | Fall/case/occasion: collapse as generative event |
| `LL-10` | Limbus Integritatis | IC ≤ F | Threshold where integrity approaches fidelity |
| `LL-11` | Complementum Perfectum | F+ω=1 | *tertia via nulla* — no third possibility |
| `LL-12` | Trans Suturam Congelatum | ε, p, tol | Same rules both sides of every boundary |
| `LL-13` | Aequator Cognitivus | — | *structura mensurat, non agens* — same data + same contract → same verdict |

**Source**: [MANIFESTUM_LATINUM.md](MANIFESTUM_LATINUM.md), `.github/copilot-instructions.md`

---

## Tier-2 — Expansion Space

> *Tier-2 closures choose which real-world quantities become channels. Freely extensible; validated through Tier-0 against Tier-1.*

### 3.1 Standard Model

**Path**: `closures/standard_model/` · **Entities**: 31 particles (17 fundamental + 14 composite) · **Contract**: CON-SM

#### Trace Vectors

| Kernel | Channels (8) | Source |
|--------|-------------|--------|
| Fundamental | mass_log, charge_abs, spin_norm, color_dof, weak_T3, hypercharge, generation, stability | subatomic_kernel.py |
| Composite | mass_log, charge_abs, spin_norm, valence, strangeness, heavy_flavor, stability, +1 | subatomic_kernel.py |
| Cross-scale (6 levels) | varies per scale | particle_matter_map.py |

#### Particle Physics Theorems (T1–T10)

| Tag | ID | Name | Tests | Key Result |
|-----|-----|------|:-----:|------------|
| `T2-PP-1` | T1 | Spin-Statistics | 12 | ⟨F⟩_fermion(0.615) > ⟨F⟩_boson(0.421) |
| `T2-PP-2` | T2 | Generation Monotonicity | 5 | Gen1 < Gen2 < Gen3 in F |
| `T2-PP-3` | T3 | Confinement as IC Collapse | 19 | IC drops 98.1% quarks → hadrons |
| `T2-PP-4` | T4 | Mass-Kernel Log Mapping | 5 | 13.2 OOM → F∈[0.37,0.73] |
| `T2-PP-5` | T5 | Charge Quantization | 5 | IC_neutral/IC_charged = 0.020 |
| `T2-PP-6` | T6 | Cross-Scale Universality | 6 | composite < atom < fundamental in F |
| `T2-PP-7` | T7 | Symmetry Breaking | 5 | EWSB amplifies gen spread |
| `T2-PP-8` | T8 | CKM Unitarity | 5 | CKM rows pass Tier-1; J_CP=3.0e-5 |
| `T2-PP-9` | T9 | Running Coupling Flow | 6 | α_s monotone for Q ≥ 10 GeV |
| `T2-PP-10` | T10 | Nuclear Binding Curve | 6 | r(BE/A,Δ) = −0.41; peak at Cr/Fe |

**Source**: `closures/standard_model/particle_physics_formalism.py` (74/74 subtests)

#### Matter Genesis Theorems (T-MG-1 through T-MG-10)

| Tag | ID | Name | Key Result |
|-----|-----|------|------------|
| `T2-MG-1` | T-MG-1 | Higgs Mass Generation | Mass generation via VEV = 246.22 GeV |
| `T2-MG-2` | T-MG-2 | Color Confinement Cost | Confinement as IC cliff |
| `T2-MG-3` | T-MG-3 | Binding Mass Deficit | Nuclear binding as mass loss |
| `T2-MG-4` | T-MG-4 | Proton-Neutron Duality | p/n kernel patterns |
| `T2-MG-5` | T-MG-5 | Shell Closure Stability | Magic number IC peaks |
| `T2-MG-6` | T-MG-6 | Electron Config Order | Aufbau principle in kernel |
| `T2-MG-7` | T-MG-7 | Covalent Bond Coherence | Molecular bonding patterns |
| `T2-MG-8` | T-MG-8 | Mass Hierarchy Bridge | Cross-scale mass ordering |
| `T2-MG-9` | T-MG-9 | Material Property Ladder | Bulk property emergence |
| `T2-MG-10` | T-MG-10 | Universal Tier-1 | All pass F+ω=1, IC≤F, IC=exp(κ) |

**Source**: `closures/standard_model/matter_genesis.py` (99 entities, 7 acts)

#### Particle-Matter Map Theorems (T-PM-1 through T-PM-8)

| Tag | ID | Name |
|-----|-----|------|
| `T2-PM-1` | T-PM-1 | Confinement Cliff |
| `T2-PM-2` | T-PM-2 | Nuclear Restoration |
| `T2-PM-3` | T-PM-3 | Shell Amplification |
| `T2-PM-4` | T-PM-4 | Periodic Modulation |
| `T2-PM-5` | T-PM-5 | Molecular Emergence |
| `T2-PM-6` | T-PM-6 | Bulk Averaging |
| `T2-PM-7` | T-PM-7 | Scale Non-Monotonicity |
| `T2-PM-8` | T-PM-8 | Tier-1 Universal |

**Source**: `closures/standard_model/particle_matter_map.py` (6-scale cross-scale kernel)

#### Additional Standard Model Files

| File | Purpose |
|------|---------|
| ckm_mixing.py | CKM matrix, Wolfenstein (λ=0.2257, A=0.814), Jarlskog J_CP |
| coupling_constants.py | Running α_s(Q²), α_em(Q²), 1-loop RGE, α_s(M_Z)=0.1180 |
| cross_sections.py | σ(e⁺e⁻→hadrons), R-ratio, point cross section |
| symmetry_breaking.py | Higgs VEV=246.22 GeV, Yukawa mass generation |
| pmns_mixing.py | PMNS matrix, leptonic mixing angles |

#### Neutrino Oscillation Theorems (T11–T12)

**Channels (3)**: P_mu_to_e, P_mu_to_mu, P_mu_to_tau (flavor probabilities, equal weights 1/3)

| Tag | ID | Name |
|-----|-----|------|
| `T2-ν-11` | T11 | Neutrino Oscillation as Periodic Channel Drift |
| `T2-ν-12` | T12 | Matter-Enhanced Mixing as Regime Transition |

**Source**: `closures/standard_model/neutrino_oscillation.py` (DUNE/LBNF: 1285 km baseline, 2.5 GeV peak)
**Lineage**: AXIOM-0 → K-F → PMNS matrix → flavor probability trace → return at L/E = 4π/Δm²

---

### 3.2 Atomic Physics

**Path**: `closures/atomic_physics/` · **Entities**: 118 elements · **Contract**: CON-ATOM

| Kernel | Channels | Source |
|--------|----------|--------|
| Periodic (8ch) | atomic_mass, electronegativity, atomic_radius, ionization_energy, electron_affinity, melting_point, boiling_point, density | periodic_kernel.py |
| Cross-scale (12ch) | Z_norm, N_over_Z, binding_per_nucleon, magic_proximity, valence_electrons, block_ord, EN, radius_inv, IE, EA, T_melt, density_log | cross_scale_kernel.py |

**Tier-1 Proof**: `tier1_proof.py` — 10,162 tests, 0 failures (F+ω=1, IC≤F, IC=exp(κ))

| File | Purpose |
|------|---------|
| electron_config.py | Shell filling analysis |
| fine_structure.py | α = 1/137 |
| ionization_energy.py | All 118 elements |
| spectral_lines.py | Emission / absorption |
| selection_rules.py | Δl = ±1 |
| zeeman_stark.py | Field splitting effects |
| recursive_instantiation.py | Structural self-similarity |

---

### 3.3 Quantum Mechanics

**Path**: `closures/quantum_mechanics/` · **Contract**: CON-QM

#### FQHE Bilayer Graphene (T-FQHE-1 through T-FQHE-7)

**Channels (8)**: filling_fraction, quasiparticle_charge, charge_fundamental, flux_periodicity, visibility, topological_order, edge_complexity, statistics_type

| Tag | ID | Name |
|-----|-----|------|
| `T2-FQHE-1` | T-FQHE-1 | Topological Order as Fidelity Separation |
| `T2-FQHE-2` | T-FQHE-2 | Charge Fractionalization as Heterogeneity Gap |
| `T2-FQHE-3` | T-FQHE-3 | Non-Abelian Ambiguity as IC Sensitivity |
| `T2-FQHE-4` | T-FQHE-4 | Hole-Conjugate Anomaly as Channel Inversion |
| `T2-FQHE-5` | T-FQHE-5 | Visibility as Coherence Proxy |
| `T2-FQHE-6` | T-FQHE-6 | e* = ν_LL Universality |
| `T2-FQHE-7` | T-FQHE-7 | Cross-Scale Bridge |

**Source**: `closures/quantum_mechanics/fqhe_bilayer_graphene.py`

#### Quantum Dimer Model (T-QDM series)

**Channels (8)**: dimer_filling, topological_order, string_coherence, symmetry_preservation, spectral_gap, fractionalization, vison_momentum, phase_stability

**Source**: `closures/quantum_mechanics/quantum_dimer_model.py`

#### TERS Near-Field Spectroscopy (T-TERS-1 through T-TERS-7)

**Channels (8)**: polarizability_zz, field_gradient, displacement_norm, screening_factor, mode_projection, self_fraction, periodicity_fidelity, binding_sensitivity

| Tag | ID | Name |
|-----|-----|------|
| `T2-TERS-1` | T-TERS-1 | Self/Cross Decomposition as Heterogeneity Gap |
| `T2-TERS-2` | T-TERS-2 | Screening-Induced Sign Reversal as Seam Event |
| `T2-TERS-3` | T-TERS-3 | Linear Regime as ε-Controlled Sensitivity |
| `T2-TERS-4` | T-TERS-4 | Ground-State Neglect as Positional Illusion Bound |
| `T2-TERS-5` | T-TERS-5 | Periodicity Requirement as Frozen Contract Consistency |
| `T2-TERS-6` | T-TERS-6 | Binding Distance Sensitivity as κ Finite-Change Bound |
| `T2-TERS-7` | T-TERS-7 | Mode-Dependent Screening as Channel Projection Theorem |

**Source**: `closures/quantum_mechanics/ters_near_field.py`

#### Atom–Dot Metal-Insulator Transition (T-ADOT-1 through T-ADOT-7)

**Channels (8)**: metallic_stability, tunneling_norm, conductance_log, g_ratio_lowT, coupling_ratio, dot_area_norm, gap_closure, thermal_freedom

| Tag | ID | Name |
|-----|-----|------|
| `T2-ADOT-1` | T-ADOT-1 | MI Transition as Collapse Event |
| `T2-ADOT-2` | T-ADOT-2 | Conductance-Fidelity Ordering |
| `T2-ADOT-3` | T-ADOT-3 | Temperature-Driven Return Trajectory |
| `T2-ADOT-4` | T-ADOT-4 | Activation Energy as κ Sensitivity |
| `T2-ADOT-5` | T-ADOT-5 | Extended Hubbard Heterogeneity |
| `T2-ADOT-6` | T-ADOT-6 | Mott Gap as Seam Budget Ceiling |
| `T2-ADOT-7` | T-ADOT-7 | Cross-Scale Universality |

**Source**: `closures/quantum_mechanics/atom_dot_mi_transition.py`

#### Muon Laser Decay (T-MLD-1 through T-MLD-7)

**Channels (8)**: rate_modification, background_ratio_log, field_coupling_log, energy_param_log, pulse_duration_log, omega_log, signal_purity, cos2_interference

| Tag | ID | Name |
|-----|-----|------|
| `T2-MLD-1` | T-MLD-1 | Tier-1 Kernel Identities |
| `T2-MLD-2` | T-MLD-2 | Rate Suppression Monotonicity |
| `T2-MLD-3` | T-MLD-3 | 50% Floor Universality |
| `T2-MLD-4` | T-MLD-4 | Parameter Utilization Orders F |
| `T2-MLD-5` | T-MLD-5 | IC Killed by Weakest Channel |
| `T2-MLD-6` | T-MLD-6 | Perturbative Limit from Kernel |
| `T2-MLD-7` | T-MLD-7 | Interference–Balance Anticorrelation |

**Source**: `closures/quantum_mechanics/muon_laser_decay.py`

#### Double-Slit Interference (T-DSE-1 through T-DSE-7)

**Channels (8)**: coherence_visibility, path_distinguishability, slit_availability, phase_coherence, spatial_coherence, source_preparation, wavelength_resolution, environmental_isolation

| Tag | ID | Name |
|-----|-----|------|
| `T2-DSE-1` | T-DSE-1 | Tier-1 Kernel Identities |
| `T2-DSE-2` | T-DSE-2 | Complementarity as Channel Anticorrelation |
| `T2-DSE-3` | T-DSE-3 | Complementarity Cliff |
| `T2-DSE-4` | T-DSE-4 | Quantum Eraser Lifts IC Above Cliff |
| `T2-DSE-5` | T-DSE-5 | Classical Limit as Maximum Channel Death |
| `T2-DSE-6` | T-DSE-6 | Delayed Choice Invariance |
| `T2-DSE-7` | T-DSE-7 | Partial Measurement Transcends Both Extremes |

**Source**: `closures/quantum_mechanics/double_slit_interference.py`

#### Additional QM Files

wavefunction_collapse.py, entanglement.py, tunneling.py, harmonic_oscillator.py, spin_measurement.py, uncertainty_principle.py

---

### 3.4 Nuclear Physics

**Path**: `closures/nuclear_physics/` · **Contract**: CON-NUC

#### QGP/RHIC Theorems (T-QGP-1 through T-QGP-10)

**Channels (8)**: temperature_frac, baryochem_frac, energy_density_norm, collectivity, opacity, strangeness_eq, multiplicity_norm, deconfinement

| Tag | ID | Name |
|-----|-----|------|
| `T2-QGP-1` | T-QGP-1 | Perfect Liquid |
| `T2-QGP-2` | T-QGP-2 | Centrality Ordering |
| `T2-QGP-3` | T-QGP-3 | BES Energy Ordering |
| `T2-QGP-4` | T-QGP-4 | Strangeness Equilibration |
| `T2-QGP-5` | T-QGP-5 | Reconfinement Gap Jump |
| `T2-QGP-6` | T-QGP-6 | Flow-Opacity Structure |
| `T2-QGP-7` | T-QGP-7 | Chemical Freeze-out Curve |
| `T2-QGP-8` | T-QGP-8 | Reconfinement Cliff |
| `T2-QGP-9` | T-QGP-9 | Reference Discrimination |
| `T2-QGP-10` | T-QGP-10 | Universal Tier-1 |

**Source**: `closures/nuclear_physics/qgp_rhic.py`

Additional files: nuclide_binding.py, shell_structure.py, alpha_decay.py, decay_chain.py, fissility.py, periodic_table.py, element_data.py, double_sided_collapse.py

---

### 3.5 GCD Closures

**Path**: `closures/gcd/` · **Contract**: CON-GCD

#### Kernel Structural Theorems (T-KS-1 through T-KS-7)

| Tag | ID | Name |
|-----|-----|------|
| `T2-KS-1` | T-KS-1 | Dimensionality Fragility Law |
| `T2-KS-2` | T-KS-2 | Positional Democracy of Slaughter |
| `T2-KS-3` | T-KS-3 | Weight-Induced Fragility Hierarchy |
| `T2-KS-4` | T-KS-4 | Monitoring Paradox Quantified |
| `T2-KS-5` | T-KS-5 | Approximation Boundary |
| `T2-KS-6` | T-KS-6 | U-Curve of Degradation |
| `T2-KS-7` | T-KS-7 | p=3 Unification Web |

**Source**: `closures/gcd/kernel_structural_theorems.py`

Additional files: energy_potential.py, entropic_collapse.py, field_resonance.py, generative_flux.py, momentum_flux.py, universal_regime_calibration.py

---

### 3.6 RCFT Closures

**Path**: `closures/rcft/` · **Contract**: CON-RCFT

#### Information Geometry Theorems

| Tag | ID | Name | Formula |
|-----|-----|------|---------|
| `T2-IG-17` | T17 | Fisher Geodesic Distance | d_F(c₁,c₂) = 2\|arcsin(√c₁)−arcsin(√c₂)\| |
| `T2-IG-18` | T18 | Geodesic Parametrization | c(t) = sin²(θ₁+t(θ₂−θ₁)) |
| `T2-IG-19` | T19 | Fano-Fisher Duality | h″(c) = −g_F(c) |
| `T2-IG-22` | T22 | Thermodynamic Efficiency Ratio | η = Δκ_useful/Δκ_total |

**Source**: `closures/rcft/information_geometry.py`

#### Quincke Rollers (T-QR-1 through T-QR-8)

**Channels (8)**: electric_driving, rolling_speed, velocity_coherence, magnetic_saturation, alignment_speed, chain_fraction, orientational_order, rotational_regularity

| Tag | ID | Name |
|-----|-----|------|
| `T2-QR-1` | T-QR-1 | Quincke Threshold Cliff |
| `T2-QR-2` | T-QR-2 | Magnetic Chain Restoration |
| `T2-QR-3` | T-QR-3 | Reversible Assembly-Disassembly |
| `T2-QR-4` | T-QR-4 | Vortex as Collective Coherence Peak |
| `T2-QR-5` | T-QR-5 | Anomalous Dimer IC Collapse |
| `T2-QR-6` | T-QR-6 | E-Field Monotonicity |
| `T2-QR-7` | T-QR-7 | Teleoperation as Fidelity Channel |
| `T2-QR-8` | T-QR-8 | Tier-1 Universal Compliance |

**Source**: `closures/rcft/quincke_rollers.py`

Additional files: active_matter.py, attractor_basin.py, coherence_pipeline_closure.py, collapse_grammar.py, fractal_dimension.py, recursive_field.py, resonance_pattern.py, universality_class.py

---

### 3.7 Consciousness Coherence

**Path**: `closures/consciousness_coherence/` · **Entities**: 20 coherence-candidate systems

**Channels (8)**: harmonic_ratio, recursive_depth, return_fidelity, spectral_coherence, phase_stability, information_density, temporal_persistence, cross_scale_coupling

| Tag | ID | Name |
|-----|-----|------|
| `T2-CC-1` | T-CC-1 | Harmonic Non-Privilege |
| `T2-CC-2` | T-CC-2 | Recursion-Return Dissociation |
| `T2-CC-3` | T-CC-3 | Universal Instability |
| `T2-CC-4` | T-CC-4 | Geometric Slaughter |
| `T2-CC-5` | T-CC-5 | Tuning Invariance |
| `T2-CC-6` | T-CC-6 | Mathematical Supremacy |
| `T2-CC-7` | T-CC-7 | Heterogeneity Gap Ordering |

**Source**: `closures/consciousness_coherence/consciousness_theorems.py`

---

### 3.8 Awareness-Cognition

**Path**: `closures/awareness_cognition/` · **Entities**: 34 organisms across phylogeny

**Channels (10 — split domain)**:
- AWARENESS (5): mirror_recognition, metacognitive_accuracy, planning_horizon, symbolic_depth, social_cognition
- APTITUDE (5): sensory_acuity, motor_precision, environmental_tolerance, reproductive_output, somatic_resilience

| Tag | ID | Name |
|-----|-----|------|
| `T2-AW-1` | T-AW-1 | Awareness-Aptitude Inversion |
| `T2-AW-2` | T-AW-2 | Universal Instability |
| `T2-AW-3` | T-AW-3 | Geometric Slaughter Bottleneck |
| `T2-AW-4` | T-AW-4 | Sensitivity Formula |
| `T2-AW-5` | T-AW-5 | Cross-Domain Isomorphism |
| `T2-AW-6` | T-AW-6 | Cost of Awareness |
| `T2-AW-7` | T-AW-7 | Human Development Trajectory |
| `T2-AW-8` | T-AW-8 | Binding Gate Transition |
| `T2-AW-9` | T-AW-9 | Cross-Domain Bridge |
| `T2-AW-10` | T-AW-10 | Formal Bounds |

**Source**: `closures/awareness_cognition/awareness_theorems.py`

---

### 3.9 Evolution

**Path**: `closures/evolution/` · **Entities**: 40 organisms (evolution), 20 species (brain)

| Kernel | Channels | Source |
|--------|----------|--------|
| Evolution (8ch) | genetic_diversity, morphological_fitness, reproductive_success, metabolic_efficiency, immune_competence, environmental_breadth, behavioral_complexity, lineage_persistence | evolution_kernel.py |
| Brain (10ch) | encephalization_quotient, cortical_neuron_count, prefrontal_ratio, synaptic_density, connectivity_index, metabolic_investment, plasticity_window, language_architecture, temporal_integration, social_cognition | brain_kernel.py |

Additional files: deep_implications.py, evolutionary_dynamics.py, mapping.py, recursive_evolution.py, species_catalog.py

---

### 3.10 Dynamic Semiotics

**Path**: `closures/dynamic_semiotics/` · **Entities**: 30 sign systems

**Channels (8)**: sign_repertoire, interpretant_depth, ground_stability, translation_fidelity, semiotic_density, indexical_coupling, iconic_persistence, symbolic_recursion

**Source**: `closures/dynamic_semiotics/semiotic_kernel.py`

See also: [SEMIOTIC_CONVERGENCE.md](SEMIOTIC_CONVERGENCE.md) — Peirce correspondence

---

### 3.11 Astronomy

**Path**: `closures/astronomy/` · **Contract**: CON-ASTRO

#### Stellar Ages Cosmology (T-SC-1 through T-SC-10)

**Channels (8)**: age_frac, mass_norm, metallicity_norm, alpha_norm, teff_norm, logg_norm, av_norm, precision

| Tag | ID | Name |
|-----|-----|------|
| `T2-SC-1` | T-SC-1 | Selection Funnel |
| `T2-SC-2` | T-SC-2 | Age-Mass Anticorrelation |
| `T2-SC-3` | T-SC-3 | Metallicity Bias |
| `T2-SC-4` | T-SC-4 | Contamination Detection |
| `T2-SC-5` | T-SC-5 | Hubble Tension Probe |
| `T2-SC-6` | T-SC-6 | Golden vs Final |
| `T2-SC-7` | T-SC-7 | Systematic Budget |
| `T2-SC-8` | T-SC-8 | Cosmological Lower Bound |
| `T2-SC-9` | T-SC-9 | Formation Delay |
| `T2-SC-10` | T-SC-10 | Universal Tier-1 |

**Source**: `closures/astronomy/stellar_ages_cosmology.py` (Tomasetti et al. 2026, A&A 707, A111)

Additional files: cosmology.py, distance_ladder.py, gravitational_dynamics.py, orbital_mechanics.py, spectral_analysis.py, stellar_evolution.py, stellar_luminosity.py

---

### 3.12 Everyday Physics

**Path**: `closures/everyday_physics/`

#### Epistemic Coherence (T-EC-1 through T-EC-7)

**Channels (8)**: pattern_recognition, narrative_coherence, predictive_accuracy, causal_mechanism, reproducibility, falsifiability, evidential_convergence, institutional_scrutiny

| Tag | ID | Name |
|-----|-----|------|
| `T2-EC-1` | T-EC-1 | Tier-1 Kernel Identities |
| `T2-EC-2` | T-EC-2 | Persistence-Integrity Decoupling |
| `T2-EC-3` | T-EC-3 | Channel-Death Dominance |
| `T2-EC-4` | T-EC-4 | Evidence-Type Hierarchy |
| `T2-EC-5` | T-EC-5 | Paradigm Shift as Heterogeneity-Gap Event |
| `T2-EC-6` | T-EC-6 | Folk Knowledge Partial-Fidelity |
| `T2-EC-7` | T-EC-7 | Institutional Amplification |

**Source**: `closures/everyday_physics/epistemic_coherence.py`

Additional files: thermodynamics.py, electromagnetism.py, optics.py, wave_phenomena.py

---

### 3.13 Finance

**Path**: `closures/finance/` · **Contract**: CON-FIN

Embedding-focused domain. Classes: `FinanceTargets`, `FinanceRecord`, `EmbeddedFinance`. Functions: `embed_finance()`. No formal theorem series.

---

### 3.14 Security

**Path**: `closures/security/` · **Contract**: CON-SEC

Infrastructure-heavy domain (17+ files). Functions: `compute_trust_fidelity()`, `classify_trust_status()`. YAML configs: identity_verification.v1.yaml, privacy_rules.v1.yaml, threat_patterns.v1.yaml, trust_baseline.v1.yaml. No formal theorem series.

---

### 3.15 Kinematics

**Path**: `closures/kinematics/` · **Contract**: CON-KIN

Phase-space focused. Functions: `compute_rotational_kinematics()`, `compute_phase_distance()`, `compute_kinematic_return()`, `compute_stability_margin()`, `classify_motion_regime()`, `compute_kinematic_budget()`. No formal theorem numbering.

---

### 3.16 Weyl Cosmology

**Path**: `closures/weyl/` · **Contract**: CON-WEYL

Cosmological transfer functions. Classes: `LimberRegime`, `GGLResult`, `WeylRegime`, `WeylTransferResult`. Functions: `classify_limber_regime()`, `compute_weyl_transfer()`, `weyl_return_domain()`. No formal theorem series.

---

### 3.17 Materials Science

**Path**: `closures/materials_science/` · **Contract**: CON-MATL · **Entities**: 118 elements (18 fields each)

`Element` dataclass (18 fields), 118-element database. Additional modules: band_structure.py, bcs_superconductivity.py, bioactive_compounds_database.py, cohesive_energy.py, crystal_morphology_database.py, debye_thermal.py, elastic_moduli.py, magnetic_properties.py, phase_transition.py, photonic_materials_database.py, surface_catalysis.py. No formal theorem series.

---

### 3.18 Continuity Theory

**Path**: `closures/continuity_theory/` · **Channels**: 1 (the n=1 degenerate limit)

Butzbach scalar continuity coefficient embedded as degenerate kernel case. Demonstrates what the kernel cannot detect without channel decomposition. Classes: `ProtocolDeclaration`, `ContinuityTrial`. Foundational/comparative role.

#### Budget Geometry Theorems (T-BG-1 through T-BG-6)

**Entities**: 12 budget surface locations · **Channels (8)**: budget surface geometry channels

| Tag | ID | Name |
|-----|-----|------|
| `T2-BG-1` | T-BG-1 | Flat Plain Highest F |
| `T2-BG-2` | T-BG-2 | Gaussian Flatness Universal |
| `T2-BG-3` | T-BG-3 | F Monotonic Flat→Ramp→Wall |
| `T2-BG-4` | T-BG-4 | Equator Max Agent Balance |
| `T2-BG-5` | T-BG-5 | Pole/Singularity Max ω |
| `T2-BG-6` | T-BG-6 | Gap Monotonic by Terrain |

**Source**: `closures/continuity_theory/budget_geometry.py`
**Lineage**: AXIOM-0 → frozen_contract → kernel_optimized → budget_geometry
**Key insight**: Flat plain entities have highest F — gentle slope preserves coherence; heterogeneity gap increases from flat_plain to wall — divergent geometry at high ω

---

### 3.19 Clinical Neuroscience

**Path**: `closures/clinical_neuroscience/` · **Entities**: 35 neurocognitive states · **Contract**: CON-CN

**Categories (7)**: Healthy, Altered States, Disorders of Consciousness (DOC), Neurodegenerative, Psychiatric, Developmental, TBI

#### Neurocognitive Theorems (T-CN-1 through T-CN-10)

**Channels (10)**: cortical_complexity, default_mode_integrity, global_integration, oscillatory_hierarchy, neuroplasticity_capacity, structural_connectivity, neurotransmitter_tone, metabolic_efficiency, autonomic_regulation, neuroimmune_status

| Tag | ID | Name |
|-----|-----|------|
| `T2-CN-1` | T-CN-1 | Consciousness Gradient |
| `T2-CN-2` | T-CN-2 | DMN Collapse Signature |
| `T2-CN-3` | T-CN-3 | Entropic Brain Prediction |
| `T2-CN-4` | T-CN-4 | Anesthesia as Collapse |
| `T2-CN-5` | T-CN-5 | Formal Tier-1 Compliance |
| `T2-CN-6` | T-CN-6 | Sleep-Wake Regime Cycle |
| `T2-CN-7` | T-CN-7 | Healthy Category Supremacy |
| `T2-CN-8` | T-CN-8 | Psychiatric Channel Dispersion |
| `T2-CN-9` | T-CN-9 | DOC Hierarchy |
| `T2-CN-10` | T-CN-10 | Developmental Plasticity |

**Source**: `closures/clinical_neuroscience/neurocognitive_kernel.py`, `closures/clinical_neuroscience/neurocognitive_theorems.py`
**Lineage**: AXIOM-0 → frozen_contract → kernel_optimized → neurocognitive_kernel → neurocognitive_theorems
**Cross-refs**: Extends T-AW (§3.8 Awareness-Cognition) and T-CC (§3.7 Consciousness Coherence) into clinical domain

---

### 3.20 Spacetime Memory

**Path**: `closures/spacetime_memory/` · **Entities**: 40 spacetime objects · **Contract**: CON-ST

**Categories (9)**: Subatomic, Nuclear/Atomic, Stellar, Planetary, Diffuse, Composite, Cognitive, Biological, Boundary

#### Spacetime Theorems (T-ST-1 through T-ST-10)

**Channels (8)**: coherence_persistence, cycle_return_rate, well_depth_norm, gradient_strength, tidal_symmetry, trajectory_closure, circulation_area, heterogeneity_profile

| Tag | ID | Name |
|-----|-----|------|
| `T2-ST-1` | T-ST-1 | Gravity Is Budget Gradient |
| `T2-ST-2` | T-ST-2 | Always Attractive |
| `T2-ST-3` | T-ST-3 | Cubic Onset (Weakest Force) |
| `T2-ST-4` | T-ST-4 | Time Dilation Near Wells |
| `T2-ST-5` | T-ST-5 | Gradient Universality |
| `T2-ST-6` | T-ST-6 | Memory Wells From Iteration |
| `T2-ST-7` | T-ST-7 | Lensing From Heterogeneity |
| `T2-ST-8` | T-ST-8 | Arrow of Time From Asymmetry |
| `T2-ST-9` | T-ST-9 | Intrinsic Flatness (K = 0) |
| `T2-ST-10` | T-ST-10 | Cross-Scale Consistency |

**Source**: `closures/spacetime_memory/spacetime_kernel.py`, `closures/spacetime_memory/spacetime_theorems.py`
**Lineage**: AXIOM-0 → frozen_contract → kernel_optimized → spacetime_kernel → spacetime_theorems
**Key insight**: GR translation — mass as accumulated |κ|, curvature as budget surface geometry, geodesic as least-budget path, event horizon as ω=1 pole

#### Gravitational Phenomena Theorems (T-GP-1 through T-GP-6)

**Entities**: 12 gravitational systems · **Channels (8)**: gravitational phenomena channels

| Tag | ID | Name |
|-----|-----|------|
| `T2-GP-1` | T-GP-1 | Strong Field Highest ω |
| `T2-GP-2` | T-GP-2 | κ Always Negative |
| `T2-GP-3` | T-GP-3 | Weak Field Highest F |
| `T2-GP-4` | T-GP-4 | Event Horizon Lowest IC |
| `T2-GP-5` | T-GP-5 | Lensing Uniformity→Gap |
| `T2-GP-6` | T-GP-6 | Range Coverage Universal |

**Source**: `closures/spacetime_memory/gravitational_phenomena.py`
**Lineage**: AXIOM-0 → frozen_contract → kernel_optimized → gravitational_phenomena
**Key insight**: Strong field entities have highest ω — extreme gravity approaches budget pole; κ < 0 universally — gravity is always attractive

#### Temporal Topology Theorems (T-TT-1 through T-TT-6)

**Entities**: 12 collapse-return circulations · **Channels (8)**: temporal topology channels

| Tag | ID | Name |
|-----|-----|------|
| `T2-TT-1` | T-TT-1 | Arrow of Time Asymmetry |
| `T2-TT-2` | T-TT-2 | Near-Horizon Max ω |
| `T2-TT-3` | T-TT-3 | Stagnation Highest F |
| `T2-TT-4` | T-TT-4 | Descent F Monotonic |
| `T2-TT-5` | T-TT-5 | Equatorial Max Proximity |
| `T2-TT-6` | T-TT-6 | All Loops Nonzero |

**Source**: `closures/spacetime_memory/temporal_topology.py`
**Lineage**: AXIOM-0 → frozen_contract → kernel_optimized → temporal_topology
**Key insight**: Descent coherence exceeds ascent coherence — arrow of time is structural; all entities have nonzero loop completeness — time is cyclic, not linear

---

## Cross-References

### 4.1 Lineage Chains

Major derivation chains showing how objects flow from Axiom-0:

```
AXIOM-0
├── K-F (Fidelity) ──→ K-ω (Drift) ──→ AI-1 (F+ω=1) ──→ I-A1 (Duality)
├── K-κ (Log-integrity) ──→ K-IC (Integrity) ──→ AI-3 (IC=exp(κ))
│   └── AI-2 (IC≤F) ──→ I-A2 (Integrity bound) ──→ I-A6 (Rank-2 closed form)
│       └── I-B3 (Taylor correction) ──→ I-B2 (Solvability)
│           └── CC-3 (Perturbation Chain cluster)
├── K-S (Entropy) ──→ I-B4 (S≤h(F)) ──→ CC-2 (Dual Bounding, with I-A2)
├── K-C (Curvature) ──→ SC-1 (S≈f(F,C)) ──→ 3 effective DOF
├── I-A4 (Fisher flatness) ──→ I-B1 (f = 2cos²θ·ln(tanθ))
│   └── I-C1 (c*) ──→ I-D1/D2 (coupling max) ──→ I-D3 (triple peak) ──→ CC-5
├── FP-p (p=3) ──→ I-D7 (Cardano) ──→ FP-ωt ──→ FP-ct ──→ I-C5 (trapping)
├── I-A7 (∫f dc=−½) ──→ I-C10 (moment family) ──→ I-B7 (Fisher-entropy=π²/3) ──→ CC-6
└── I-C3 (composition) ──→ I-D8 (Δ invariance) ──→ I-D9 (gap composition) ──→ CC-4
```

### 4.2 Cross-Tier Dependencies

```
Tier-1 (Kernel)          Tier-0 (Protocol)              Tier-2 (Domains)
───────────────          ──────────────────              ────────────────
K-F, K-ω, K-S       →   T0-OptKernel.compute()     →   T2-PP-1 (Spin-Statistics uses F)
K-κ, K-IC            →   T0-valIdent                →   T2-PP-3 (Confinement: IC cliff)
AI-1, AI-2, AI-3    →   T0-valIdent._validate_*    →   tier1_proof.py (10,162 tests)
Def-11               →   T0-SeamAccum               →   T2-QGP-5 (Reconfinement gap)
Def-13               →   T0-classifyEp              →   T0-EpVerdict (RETURN/GESTURE)
FP-ε, FP-p, FP-tol  →   frozen_contract.py         →   All 20 domains
RG-S/W/C/X           →   classify_regime()          →   Domain regime analysis
L-6, L-30            →   IC sensitivity checks      →   T2-CC-4 (Geometric slaughter)
I-C3                  →   composition analyzer       →   T2-PM-* (cross-scale)
K-F, K-κ             →   T0-OptKernel.compute()     →   T2-CN-1 (Consciousness gradient)
K-IC, L-6            →   IC sensitivity              →   T2-CN-2 (DMN collapse signature)
AI-1, AI-2           →   T0-valIdent                →   T2-ST-5 (Gradient universality)
Def-11               →   T0-SeamAccum               →   T2-ST-9 (Intrinsic flatness)
```

**One-Way Rule**: Within a frozen run, Tier-1 → Tier-0 → Tier-2. No back-edges.

---

## Summary Counts

| Category | Count |
|----------|------:|
| **Tier-1** | |
| Kernel outputs | 6 |
| Frozen parameters | 8 |
| Algebraic identities | 3 |
| Statistical constraints | 1 |
| Structural constants | 5 |
| Regime gates | 4 |
| Rank types | 3 |
| Formal definitions | 14 |
| Lemmas (core + extended) | 46 |
| Structural identities (A+B+C+D) | 7 + 15 + 11 + 11 = 44 |
| Connection clusters | 6 |
| **Tier-1 total** | **136** |
| | |
| **Tier-0** | |
| Classes / dataclasses | 26 |
| Functions | ~35 |
| OPT tags | 9 |
| Schemas | 17 |
| Contracts | 13 |
| Spine stops | 5 |
| Five words | 5 |
| Typed outcomes | 3 |
| Three-valued verdicts | 3 |
| Equalizer mechanisms | 6 |
| Latin terms | 13 |
| **Tier-0 total** | **~135** |
| | |
| **Tier-2** | |
| Domain closures | 20 |
| Closure source files | ~140 |
| Named theorems (T-XX-N format) | 171 |
| Distinct trace vector specs | 17 |
| Total channels across specs | ~143 |
| Entity catalogs | 10 (≈565 entities total) |
| **Tier-2 total** | **~348** |
| | |
| **GRAND TOTAL** | **~619 tagged objects** |

---

*Omnia per spinam transeunt; omnia in catalogo numerantur.*
— Everything passes through the spine; everything is counted in the catalogue.

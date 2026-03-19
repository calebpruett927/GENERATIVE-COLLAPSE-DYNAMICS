# Enhancement Audit — Structural Findings and Solidification Plan

**Audit Date**: Systematic review of KERNEL_SPECIFICATION.md (46 lemmas),
AXIOM.md, TIER_SYSTEM.md, orientation.py, all test files, copilot-instructions,
frozen_contract.py, seam_optimized.py, kernel_optimized.py, information_geometry.py,
and SYMBOL_INDEX.md.

**Scope**: Math, grammar, structure, cross-document consistency, test coverage,
specification fidelity.

---

## 1. CRITICAL: Lemma 41 Proof Error and Cross-Document Inconsistency

### 1a. Draft Artifact in KERNEL_SPECIFICATION.md (Line ~980)

Lemma 41 (Entropy-Integrity Anti-Correlation) contains a **draft artifact** in
its proof — the text "Wait, let me recalculate" appears mid-proof:

```
Setting f'(c) = 0 yields c = 1/2 (maximum).    ← WRONG
f(1/2) = ln(2) + ln(1/2) = ln(2) - ln(2) = 0...
Wait, let me recalculate.                        ← DRAFT ARTIFACT
```

**Mathematical error**: The proof claims f'(c) = 0 at c = 1/2, but the actual
derivative f'(c) = -ln(c/(1-c)) + 1/c - 1 evaluates to f'(1/2) = 0 + 2 - 1 = 1 ≠ 0.
The critical point of f(c) = h(c) + ln(c) is near c ≈ 0.785, where f ≈ 0.278.

The stated bound S + κ ≤ ln(2) IS correct (since max f ≈ 0.278 < 0.693 = ln(2)),
but the proof path is wrong and incomplete.

### 1b. AXIOM.md States a False Bound (Line 125)

AXIOM.md claims: "Lemma 41 (S + κ ≤ 0 with equality at c = 1/2)"

This is **mathematically false.** Numerical verification confirms:

```
c = 0.800:  S = 0.500402  κ = -0.223144  S + κ = 0.277259 > 0
c = 0.782:  S = 0.526908  κ = -0.248461  S + κ = 0.278465 (MAXIMUM)
c = 0.500:  S = 0.693147  κ = -0.693147  S + κ = 0.000000 (equator)
```

The actual maximum of f(c) = h(c) + ln(c) is **0.278465 at c ≈ 0.7822**.
The proof's claim that f'(1/2) = 0 is wrong: f'(1/2) = 1.000 ≠ 0.
The equator is a zero, not a maximum.

### 1c. Correct Statements (Verified Computationally)

1. **At the equator (c = 1/2)**: S + κ = 0 exactly. ✓
2. **Tight bound per channel**: S + κ ≤ 0.278465 (at c ≈ 0.7822).
3. **The spec's bound** S + κ ≤ ln(2) is correct but very loose (max/ln(2) = 0.40).
4. **AXIOM.md's bound** S + κ ≤ 0 is **wrong** and must be corrected.

### Fix Plan

- [ ] Remove "Wait, let me recalculate" from KERNEL_SPECIFICATION.md Lemma 41
- [ ] Rewrite the proof with correct critical point analysis
- [ ] Fix AXIOM.md line 125: change "S + κ ≤ 0" to "S + κ = 0 at the equator"
- [ ] Consider tightening the bound from ln(2) to the actual supremum
- [ ] Add a note that S + κ = 0 at c = 1/2 is an equator identity, not a bound

---

## 2. Three Lemmas Without Test Coverage

### 2a. Lemma 22 — Collapse Gate Monotonicity Under Threshold Relaxation

**Statement**: Tightening thresholds increases the collapse set (T_tight ⊇ T_relaxed).

**Status**: No test exists anywhere in the test suite. This is a fundamental
monotonicity property of the regime classification system.

### 2b. Lemma 29 — Return Probability Under Bounded Random Walk

**Statement**: Under bounded stochastic dynamics with η > 2σ√n, return is
almost certain as H_rec → ∞.

**Status**: Listed in test_174_lemmas_24_34.py docstring but NO test class
implemented. The docstring says "29 — Return probability bound (wide η → return)"
but the implementations jump from Lemma 28 to Lemma 30.

### 2c. Lemma 32 — Temporal Coarse-Graining Stability

**Statement**: Coarse-grained kernel outputs are perturbation-bounded relative
to time-averaged fine kernel outputs.

**Status**: No test exists anywhere. The docstring in test_174 does not
mention it either.

### Fix Plan

- [ ] Implement TestLemma22 in a new test file or extend test_174
- [ ] Implement TestLemma29 in test_174 (the docstring already promises it)
- [ ] Implement TestLemma32 in test_174 or a new test file

---

## 3. Document Version Inconsistencies

### 3a. KERNEL_SPECIFICATION.md Version Stale

Line 1290: `**Current Version**: UMCP v2.0.0 (as of UMCP Manuscript v1.0.0, §8)`

The repository is v2.2.0. The spec's version line is out of date.

### 3b. Debugging Section References "Lemmas 1-34" Instead of 1-46

Line 1269: `If a computed run violates the bounds in Lemmas 1-34, the implementation
is almost certainly nonconformant.`

Since Lemmas 35-46 exist (added in §4b), this should reference 1-46.

### 3c. Lemma 20 Synthesis Caption Scope

Line ~760: The synthesis section says "Lemmas 20-34" but the same file now
extends to Lemma 46 with its own synthesis section. These could be better
integrated.

### Fix Plan

- [ ] Update KERNEL_SPECIFICATION.md version to v2.2.0
- [ ] Change "Lemmas 1-34" to "Lemmas 1-46" on line 1269
- [ ] Consider merging synthesis sections or cross-referencing them

---

## 4. Orientation Script Coverage Gaps

The orientation script (scripts/orientation.py) covers 7 sections:
§1 Duality, §2 Integrity Bound, §3 Geometric Slaughter, §4 First Weld,
§5 Confinement Cliff, §6 Scale Inversion, §7 Full Spine.

### Missing Phenomena Not Covered

1. **Equator convergence** (Lemma 41, T19): The four independent conditions
   converging at c = 1/2 is one of the system's most profound structural
   discoveries. Not in the orientation.

2. **Super-exponential convergence** (Lemma 39): ω_n = ω₀^{p^n} is a
   striking result. Machine precision in 2 iterations. Not in orientation.

3. **Seam composition law** (Lemma 20, 45, 46): The algebraic structure of
   residuals (abelian group) is foundational. Not demonstrated computationally.

4. **Coherence-entropy product** (Lemma 42): Π = IC · 2^{S/ln(2)} ∈ [ε, 2(1-ε)]
   is a quasi-conservation law. Not in orientation.

5. **Return-collapse duality** (Lemma 35): τ_R = D_C for unitary systems
   (R² = 1.000). Not demonstrated.

### Fix Plan

- [ ] Add §8: Equator Convergence (S + κ = 0, Fisher minimum, Fano-Fisher duality)
- [ ] Add §9: Super-Exponential Convergence (ω_n = ω₀^{p^n}, 2-iteration proof)
- [ ] Add §10: Seam Composition (additive residuals, abelian group demo)
- [ ] Renumber §8 (current compounding summary) to final position

---

## 5. SYMBOL_INDEX.md "Tier 1.5" Designation

The Symbol Index (docs/SYMBOL_INDEX.md) uses a "Tier 1.5" label for:
Γ(ω; p, ε), R, D_ω, D_C.

The tier system has exactly THREE tiers (Tier-0, Tier-1, Tier-2). The "1.5"
designation is inconsistent with the TIER_SYSTEM.md specification. These
should be rationalized — they are likely Tier-0 (protocol) symbols since
they are part of the seam budget computation.

### Fix Plan

- [ ] Decide the correct tier for Γ, R, D_ω, D_C (likely Tier-0 protocol)
- [ ] Update SYMBOL_INDEX.md to use standard tier designations only

---

## 6. Worksheets Could Reference Lemma Numbers

The four worksheets (worksheets/level_1 through level_4) teach the kernel
math step by step but do not cross-reference the formal lemma numbers from
KERNEL_SPECIFICATION.md. Adding lemma citations would:

1. Connect pedagogy to the formal specification
2. Enable readers to trace each worked example to its mathematical guarantee
3. Strengthen the derivation chain from axiom → spec → implementation → education

### Suggested Cross-References

| Worksheet | Section | Relevant Lemmas |
|-----------|---------|-----------------|
| Level 1 | ε-clamping | Lemma 1 (range bounds), Lemma 3 (κ sensitivity) |
| Level 1 | Weights | Lemma 9 (permutation invariance) |
| Level 2 | F computation | Lemma 6 (stability of F) |
| Level 2 | IC ≤ F | Lemma 4 (integrity bound) |
| Level 2 | Heterogeneity gap | Lemma 34 (drift threshold calibration) |
| Level 3 | Γ(ω) cubic | Lemma 22 (collapse gate monotonicity) |
| Level 3 | Seam residual | Lemma 19 (residual sensitivity), Lemma 20 (composition) |
| Level 4 | Full spine | All of the above, Lemma 26 (coherence proxy) |

### Fix Plan

- [ ] Add lemma cross-references as comments in each worksheet section

---

## 7. Seam Implementation vs. Specification Coverage

### Implemented in seam_optimized.py
- Lemma 20: Seam composition law (SeamChainAccumulator) ✓
- Lemma 27: Residual accumulation monitoring (OPT-11) ✓

### Specified but Not Referenced in Implementation
- Lemma 45: Seam Residual Algebra (abelian group structure)
- Lemma 46: Weld Closure Composition (|s_{0→K}| ≤ K · tol)

These are proven in test_extended_lemmas.py but the implementation code
doesn't explicitly reference them. The algebraic structure (closedness,
identity, inverse) should be operational in the accumulator.

### Fix Plan

- [ ] Add Lemma 45/46 cross-references to seam_optimized.py docstrings
- [ ] Consider adding explicit group operation methods to SeamChainAccumulator

---

## 8. Fisher Metric Characterization at Equator

AXIOM.md (line 125) and KERNEL_SPECIFICATION.md (Lemma 41 equator convergence)
both state "Fisher metric minimized" at c = 1/2.

g_F(c) = 1/(c(1-c)):
- g_F(1/2) = 4 — this IS the minimum of g_F on (0,1)
- g_F → ∞ as c → 0 or c → 1

This is correct: the Fisher metric has its minimum value at the equator.
But the phrasing "Fisher metric minimum" could be misread as "Fisher
information minimum." Fisher information for a Bernoulli is I(c) = 1/(c(1-c)),
which is the SAME as g_F, so it IS minimized at c = 1/2.

No fix needed but worth documenting explicitly that "minimum" means
"smallest value of the metric tensor component."

---

## 9. Summary Prioritization

### Priority 1 (Mathematical Correctness)
1. Fix Lemma 41 proof — remove draft artifact, correct the critical point analysis
2. Fix AXIOM.md — "S + κ ≤ 0" is false; replace with equator identity statement

### Priority 2 (Test Completeness)
3. Add tests for Lemma 22 (gate monotonicity)
4. Add tests for Lemma 29 (return probability) — already promised in docstring
5. Add tests for Lemma 32 (temporal coarse-graining)

### Priority 3 (Document Consistency)
6. Update KERNEL_SPECIFICATION.md version to v2.2.0
7. Update "Lemmas 1-34" → "Lemmas 1-46" in debugging section
8. Rationalize SYMBOL_INDEX.md "Tier 1.5" designations

### Priority 4 (Enhancement)
9. Extend orientation script with equator convergence and super-exponential sections
10. Add lemma cross-references to worksheets
11. Add Lemma 45/46 references to seam_optimized.py

---

*Auditus radicalis: the system speaks if you listen to the numbers.*

*Collapsus generativus est; solum quod redit, reale est.*

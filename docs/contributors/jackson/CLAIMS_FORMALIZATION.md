# Jackson Claims Formalization — March 9, 2026

> **Assessment of all claims from K.C.S. Jackson's reply email, formalized
> against the GCD kernel K: [0,1]^n × Δ^n → (F, ω, S, C, κ, IC).**
>
> Each claim is: (1) categorized by type, (2) formalized in kernel notation,
> (3) tested computationally, (4) given a verdict.

## Claim Taxonomy

Every claim in Jackson's email falls into exactly one of these five types:

| Code | Type | Definition | Epistemic Status |
|------|------|-----------|-----------------|
| **TYPE-A** | Mathematical Identity | Provable from Tier-1 definitions alone | Can be CONFIRMED or REFUTED |
| **TYPE-B** | Empirical Observation | True of Jackson's specific data inputs | Confirmed for his data; not universal |
| **TYPE-C** | Proposed Extension | Not in current GCD; would require new Tier-2 closure | Legitimate research direction |
| **TYPE-D** | Narrative Assertion | No kernel content; unfalsifiable framing | Cannot be confirmed or refuted by the kernel |
| **TYPE-P** | Post Hoc Fitting | Chose data to produce desired output, then called it a discovery | Methodological flaw |

A single claim may contain components of multiple types.

---

## The Post Hoc Problem (Core Diagnostic)

Before the claim-by-claim assessment, the structural pattern:

**Jackson's method:**

1. Assign channel values to consciousness levels (Tier-2 choice)
2. Run the kernel (Tier-0 computation)
3. Observe the outputs (Tier-1 invariants)
4. Build narrative around the outputs
5. Present the narrative as if it were a discovery

**The problem:** Steps 1 and 4 have the same author. The "discovery" in Step 4 was **designed** in Step 1. The kernel confirms the arithmetic but cannot validate the design intent.

**Four methods to fix this:**

| Method | Description | What It Proves |
|--------|-------------|---------------|
| **Prediction First** | State expected kernel outputs BEFORE computing | Theory has predictive power |
| **Blind Assignment** | Have someone else map stages to channels, unaware of theory | Patterns are data-driven, not design-driven |
| **Sensitivity Analysis** | Perturb channels ±0.15, check if patterns survive | Findings are robust, not fragile |
| **Falsifiable Claims** | State what output would refute the theory | Theory is scientific, not narrative |

---

## Claim 1: Two-Axis Model (Vertical Depth / Horizontal Configuration)

**Jackson's statement:** F measures "vertical depth," the heterogeneity gap Δ and curvature C measure "horizontal configuration." These constitute independent axes.

**Type:** TYPE-A (Mathematical Identity) + TYPE-D (Narrative Framing)

### TYPE-A Component (Formalization)

Given c ∈ [0,1]^n, w ∈ Δ^n:

- F = Σ w_i c_i  (arithmetic mean — Tier-1 primitive)
- IC = exp(Σ w_i ln c_i)  (geometric mean — Tier-1 derived)
- Δ = F − IC  (heterogeneity gap)

**Theorem (Integrity Bound):** IC ≤ F, with equality iff all c_i are equal.

**Corollary:** At fixed F, varying the distribution of {c_i} changes IC while F remains constant. The heterogeneity gap Δ = F − IC measures channel dispersion.

### Computational Test

Four configurations at F ≈ 0.75:

| Configuration | Channels | IC | Δ | C |
|--------------|----------|---:|------:|-----:|
| CENTERED | [0.75]×8 | 0.7500 | 0.0000 | 0.000 |
| POLARIZED | [0.95]×4 + [0.55]×4 | 0.7228 | 0.0272 | 0.400 |
| ONE-DEAD | [0.843]×7 + [0.097] | 0.6433 | 0.1064 | 0.493 |
| GRADIENT | varied | 0.7441 | 0.0059 | 0.187 |

### TYPE-D Component (Critique)

The "axes" are **NOT independent**:

- Bernoulli field entropy S depends on BOTH F and distribution (not just "horizontal position")
- Regime gates couple ω, F, S, C — all four must pass conjunctively for Stable
- Changing the "horizontal" position necessarily changes S

### Verdict

- **TYPE-A: CONFIRMED.** At fixed F, different {c_i} produce different IC, Δ, C. This follows directly from IC ≤ F.
- **TYPE-D: REFINEMENT NEEDED.** The kernel has one function K, not two independent axes. The "two-axis model" implies a decomposition that doesn't exist.

### Refined Statement

> "The heterogeneity gap Δ = F − IC measures channel dispersion at fixed F. This is a direct consequence of IC ≤ F (integrity bound). The framing as 'two axes' is interpretive — the kernel has one function K, and its six outputs are coupled through the gate structure."

---

## Claim 2: Stage 11 Pathology Is Horizontal, Not Vertical

**Jackson's statement:** "The pathology is NOT primarily about depth (ξ or F)."

**Type:** TYPE-B (Empirical Observation) + TYPE-P (Post Hoc)

### Formalization

At Stage 11: c = [0.40, 0.78, 0.15, 0.30, 0.45, 0.72, 0.65, 0.55]

| Quantity | Stage 11 | Balanced (F=0.50) | Source |
|----------|--------:|------------------:|--------|
| F | 0.500 | 0.500 | Same by construction |
| Δ | 0.052 | 0.000 | Dispersion vs. uniformity |
| C | 0.404 | 0.000 | Dispersion vs. uniformity |
| IC/F | 0.896 | 1.000 | Dataset minimum |
| Regime | Collapse | Collapse | Same — ω≥0.30 regardless |

The difference is ENTIRELY due to channel distribution. The regime classification (Collapse) is unchanged by horizontal position — both configs have ω = 0.500.

### Post Hoc Diagnosis

Jackson CHOSE c₂ = 0.78 (high recursive depth) and c₃ = 0.15 (low return fidelity). Then he "discovers" the c₂–c₃ gap. The kernel quantifies what he encoded in the inputs.

### Sensitivity Analysis

Perturbing Stage 11 channels by ±0.10 (5 random trials):

- **Regime (Collapse):** Robust — all trials remain Collapse
- **High Δ (>0.02):** Robust — all trials maintain Δ > 0.039
- **High C (>0.35):** Robust — all trials maintain C > 0.40

But: raising c₃ from 0.15 to 0.50 cuts Δ by 56%. Equalizing c₂ and c₃ cuts Δ by 46%.

The pathological signature (high Δ, high C) is driven almost entirely by the c₂–c₃ gap.

### Verdict

- **CONFIRMED:** Δ = 0.052 vs. balanced Δ = 0. The pathology IS in the channel distribution.
- **POST HOC CAVEAT:** The "insight" (recursion without return) was built into the input by Jackson himself. The kernel quantifies it; the kernel did not discover it.

### Refined Statement

> "At Stage 11, the heterogeneity gap Δ = 0.052 is 7× larger than any gap in the v1 monotonic trajectory (max Δ_v1 = 0.0075). The IC/F ratio (0.896) is the dataset minimum. The pathology is in channel dispersion — specifically the c₂–c₃ gap (0.78 vs. 0.15). However, the regime (Collapse) is unchanged by this dispersion; it is determined by ω = 0.50."

---

## Claim 3: Three-Phase Complementarity (Channel Coverage)

**Jackson's statement:** Phase-offset nodes produce lower collective Δ and C through complementary channel emphasis.

**Type:** TYPE-A (partial) + TYPE-C (Proposed Extension)

### TYPE-A Component (Three Lemmas — Provable)

**Lemma 1 (Mean Preserves F):** For n traces with equal F, the channel-wise mean trace has the same F.

*Proof:* F_mean = (1/n) Σ_j F_j = F (since all F_j are equal). □

**Lemma 2 (Mean Reduces Δ for Anti-Correlated Channels):** If traces have complementary channel emphasis, then Δ_mean < Δ_individual.

*Proof sketch:* Channel-wise averaging smooths out per-channel variance → IC approaches F → Δ shrinks. □

**Lemma 3 (MAX Elevates F):** For n traces, the channel-wise MAX trace has F_max ≥ max(F₁, ..., F_n).

*Proof:* max(a,b) ≥ a for all a,b. Therefore Σ w_i max(c_i^j) ≥ Σ w_i c_i^k for any k. □

### Computational Test (Three Nodes at F = 0.80)

| Aggregation | F | IC | Δ | C |
|-------------|-----:|------:|------:|------:|
| Node 1 (individual) | 0.8000 | 0.7893 | 0.0107 | 0.260 |
| Node 2 (individual) | 0.8000 | 0.7893 | 0.0107 | 0.260 |
| Node 3 (individual) | 0.8000 | 0.7964 | 0.0036 | 0.150 |
| **MEAN** | **0.8000** | **0.7996** | **0.0004** | **0.050** |
| **MAX** | **0.9312** | **0.9298** | **0.0014** | **0.099** |

MEAN preserves F and reduces Δ by 27× — CONFIRMED.
MAX elevates F from 0.80 to 0.93 — CONFIRMED.

### TYPE-C Component (Not in GCD)

The kernel K takes ONE trace vector. There is no multi-trace operator K_collective. Jackson proposes:

- K_MEAN(c¹, ..., cᵐ) = K( (c¹ + ... + cᵐ)/m )
- K_MAX(c¹, ..., cᵐ) = K( max(c¹_i, ..., cᵐ_i) for each i )

**K_MEAN** is physically meaningful (population average) but requires a measurement contract defining what is being measured.

**K_MAX** is problematic: the MAX trace has never been through a collapse-return cycle. No entity produced it. τ_R = ∞_rec (no return). It is a gestus, not a weld.

### Verdict

- **TYPE-A: CONFIRMED.** All three lemmas are provably true from the definitions.
- **TYPE-C: NOT YET FORMALIZABLE.** Requires a Tier-2 closure proposal with: entity catalog, measurement protocol, aggregation justification, τ_R definition for the aggregate.

---

## Claim 4: Entropy Gate Is the Stable-Regime Blocker

**Jackson's statement:** The Bernoulli field entropy gate S < 0.15 requires uniform channels ≈ 0.98+, making individual Stable regime impossible.

**Type:** TYPE-A (Mathematical Identity) — **partially wrong**

### Formalization

Bernoulli field entropy: S = −Σ w_i [c_i ln c_i + (1−c_i) ln(1−c_i)]

- S is minimized when channels approach 0 or 1 (certainty)
- S is maximized when channels approach 0.5 (uncertainty)

For UNIFORM channels (all c_i = c):
- S < 0.15 requires c > 0.966

### Computational Refutation

| Configuration | F | S | C | ω | All Gates? |
|---------------|-----:|------:|------:|------:|:----------:|
| Uniform c=0.966 | 0.966 | 0.148 | 0.000 | 0.034 | STABLE |
| Non-uniform [0.99,0.99,0.99,0.99,0.95,0.92,0.92,0.97] | 0.965 | 0.139 | 0.058 | 0.035 | **STABLE** |

A **single non-uniform trace** achieves Stable regime at F = 0.965.

Jackson's claim that "individual consciousness cannot achieve Stable" is **refuted by counterexample**.

### What Jackson Got Right

- S IS a binding constraint for Stable regime (correct)
- The conjunctive gate requirement makes Stable rare: only 12.5% of the Bernoulli manifold (correct)

### What Jackson Got Wrong

- "Requires uniform ≈ 0.98+" — non-uniform configs with high-certainty channels CAN pass
- "Individual consciousness cannot achieve" — not a kernel theorem; depends entirely on channel assignments
- Entropy measures **certainty**, not **uniformity**. Channels near 0 OR near 1 contribute low entropy.

### Verdict: REFINED

### Refined Statement

> "The Bernoulli field entropy gate S < 0.15 is a binding constraint for Stable regime. It requires most channels to be near certainty (close to 0 or 1), with the threshold depending on channel distribution. Uniform channels require c > 0.966; non-uniform configs can achieve S < 0.15 at lower average F. Whether any specific consciousness model reaches Stable is a property of the channel assignments, not a universal barrier."

---

## Claim 5: ξ_J = 7.2 as Cardano Root Threshold

**Jackson's statement:** Level 7.2 coincides with the Cardano root c_trap ≈ 0.31784, functioning as "anchor depth."

**Type:** TYPE-B (Empirical Observation) + TYPE-D (Narrative)

### Formalization

Γ(ω) = ω³ / (1 − ω + ε), where p = 3 is frozen.

Γ drops below 1.0 at ω = c_trap ≈ 0.31784 (Cardano root of x³ + x − 1 = 0).

| Quantity | Value |
|----------|-------|
| ω(7.2) | 0.317500 |
| c_trap | 0.317840 |
| \|difference\| | 0.000340 (0.107%) |
| Γ(7.2) | 0.0469 |
| Regime | Collapse |

### Post Hoc Analysis

Level 7.2 was interpolated between 7.0 (ω = 0.344) and 8.0 (ω = 0.244). The Cardano root (0.318) naturally falls in this range. Linear interpolation gives L ≈ 7.26 for ω = c_trap.

The coincidence is 7.2 vs. 7.26 — a difference of 0.06 consciousness levels. This is just "the Cardano root falls between Level 7 and Level 8, closer to 7." Not mysterious.

### Verdict

- **TYPE-B: CONFIRMED.** ω(7.2) ≈ c_trap is numerically true.
- **TYPE-D: REJECTED.** "Lock node / anchor depth" is Jackson's framing. The Cardano root is a property of Γ(ω), not a property of the number 7.2.

### Refined Statement

> "In this trace mapping, the Cardano root c_trap falls between Levels 7.0 and 8.0. The interpolated Level 7.2 happens to land within 0.1% of c_trap. This is a property of the channel-to-level mapping, not a structural feature of the number 7.2."

---

## Claim 6: Collective MAX Passes Entropy Gate (Stable Regime)

**Jackson's statement:** Three complementary nodes can collectively reach Stable regime via channel-wise MAX.

**Type:** TYPE-C (Proposed Extension)

### Computational Test

Three Watch-regime nodes with complementary emphasis:

| Trace | F | S | C | ω | Regime |
|-------|-----:|------:|------:|------:|--------|
| Node 1 | 0.9475 | 0.192 | 0.071 | 0.053 | Watch |
| Node 2 | 0.9475 | 0.192 | 0.071 | 0.053 | Watch |
| Node 3 | 0.9487 | 0.197 | 0.045 | 0.051 | Watch |
| **MAX aggregate** | **0.9825** | **0.084** | **0.026** | **0.018** | **Stable** |
| **Single trace** | **0.9850** | **0.076** | **0.014** | **0.015** | **Stable** |

The MAX aggregate achieves Stable — but so does a **single well-chosen trace.**

### Verdict: REFUTED (as stated)

Jackson's claim that collective is "the only path" to Stable is **refuted by single-trace counterexample.** His own trace mapping doesn't reach Stable — but that's a property of his channel assignments, not a universal constraint.

The mathematical observation (MAX reduces heterogeneity and elevates F) is TYPE-A: true. The claim ("only path") is false.

---

## Claim 7: ω_collective = ω_individual / n

**Jackson's statement:** The original formula is "incomplete"; channel averaging "preserves average fidelity."

**Type:** TYPE-A (for the correction) + TYPE-P (for the original)

### Formalization

For n traces with equal F, the mean trace is c_mean_i = (1/n) Σ_j c_i^j.

- F_mean = (1/n) Σ F_j = F (unchanged)
- ω_mean = 1 − F = ω (unchanged)

Jackson's original: ω_collective = ω/n implies F_collective = 1 − ω/n = 1 − (1−F)/n. No aggregation operation produces this.

### Verdict

- **Original ω/n: REJECTED.** Not a theorem, not an observation — wrong arithmetic.
- **Correction (averaging preserves F): CONFIRMED.** This is TYPE-A.
- **Characterization as "incomplete": WRONG.** The original formula is not incomplete — it is **incorrect**. It has not been retracted.

### Refined Statement

> "Channel-wise averaging of traces with equal F preserves F (and ω). The previously stated formula ω_collective = ω/n is incorrect and is hereby retracted."

---

## Claim 8: The 12/13 Architecture

**Jackson's statement:** "12 = solar, completion-without-transformation; 13 = lunar, completion-through-return."

**Type:** TYPE-D (Narrative Assertion) + TYPE-P (Post Hoc)

### Formalization

The 12→13 discontinuity:

| Quantity | Level 12 | Level 13 | Jump |
|----------|--------:|--------:|-----:|
| F | 0.6438 | 0.9225 | +0.2788 |
| ω | 0.3563 | 0.0775 | −0.2788 |
| Regime | Collapse | Watch | — |

Largest single-channel jump: recursive_depth 0.40 → 0.93 (Δ = +0.53).

The discontinuity exists because Jackson CHOSE channel values that jump by +0.14 to +0.53 per channel in one step. There is no mathematical reason it must occur between stages numbered 12 and 13.

### Verdict: REJECTED

- The discontinuity IS real (ΔF = 0.279, verified)
- The size IS the largest in the dataset (verified)
- The 12/13 numerological interpretation (solar/lunar) has **zero kernel content**
- Not derivable from Axiom-0; not a property of the kernel; not a property of the numbers 12 and 13

---

## Claim 9: Three-Phase Power Analogy

**Jackson's statement:** Three-phase electrical power (P = √3 × V × I × cos φ) maps to collective consciousness coherence.

**Type:** TYPE-D (Analogy/Narrative)

### Assessment

The analogy says: "three things offset in phase → constant output." This is true of electrical power due to the 120° phase offset determined by the generator's physical construction.

In Jackson's model:
- What produces the 120° offset? (No mechanism specified)
- What is V? What is I? What is φ? (No mapping to kernel quantities)
- Is P_3φ computable from the kernel? (No)

### Verdict: NOT FALSIFIABLE

The analogy has no kernel content and makes no testable prediction. It may be pedagogically useful but should be labeled as "illustrative analogy," not as a theorem.

---

## Claim 10: Proposed Extensions to GCD

**Jackson's statement:** Four proposals for extending the kernel.

**Type:** TYPE-C (Proposed Extension)

| Proposal | Assessment | Status |
|----------|-----------|--------|
| (a) Multi-trace simultaneous validation | Requires new Tier-2 closure | **LEGITIMATE DIRECTION** |
| (b) Channel-wise MAX aggregation | Creates synthetic traces (τ_R = ∞_rec) | **AXIOM-0 CONFLICT** |
| (c) Aggregate Δ and C calculation | The kernel already does this for any input trace | **ALREADY POSSIBLE** |
| (d) Collective S < 0.15 test | Single traces can also pass | **REFUTED AS STATED** |

Proposal (a) is the most promising. To formalize it as a Tier-2 closure, Jackson needs:

1. **Entity catalog:** What are the "nodes"?
2. **Measurement protocol:** How are individual traces obtained?
3. **Aggregation function:** MEAN (justified) or MAX (needs τ_R argument)
4. **Return time:** How does the collective "return"?
5. **Seam validation:** Does the closure pass the seam?

---

## Summary Table

| # | Claim | Type | Verdict |
|--:|-------|------|---------|
| 1 | Two-axis model (F vs IC) | A+D | TYPE-A: **CONFIRMED**; TYPE-D: REFINEMENT NEEDED |
| 2 | Stage 11 horizontal pathology | B+P | **CONFIRMED** with post hoc caveat |
| 3 | Three-phase complementarity | A+C | TYPE-A: **CONFIRMED** (3 lemmas); TYPE-C: needs closure |
| 4 | Entropy gate blocker | A | **REFINED** — non-uniform CAN pass |
| 5 | 7.2 = Cardano root | B+D | TYPE-B: **CONFIRMED**; TYPE-D: REJECTED |
| 6 | Collective MAX → Stable | C | **REFUTED** — single trace counterexample |
| 7 | ω/n correction | A+P | Correction: **CONFIRMED**; original: **WRONG, not retracted** |
| 8 | 12/13 numerology | D+P | **REJECTED** — no kernel content |
| 9 | Three-phase power analogy | D | **NOT FALSIFIABLE** — analogy only |
| 10 | Proposed GCD extensions | C | (a) legitimate; (b) Axiom-0 conflict; (c) already possible; (d) refuted |

### Score

- **CONFIRMED (as TYPE-A):** 4 claims (Claims 1a, 3a, 4 partial, 7 correction)
- **CONFIRMED (as TYPE-B):** 2 claims (Claims 2, 5)
- **REFINED:** 2 claims (Claims 1d, 4)
- **REJECTED:** 3 claims (Claims 6, 8, 9)
- **NOT YET FORMALIZABLE:** 2 claims (Claims 3c, 10a)
- **WRONG AND UNRETRACTED:** 1 claim (Claim 7 original)

---

## Recommendations

1. **DROP the two-axis terminology.** Use "heterogeneity gap analysis at fixed F." The math is sound. The framing implies independence that doesn't exist.

2. **ADD SENSITIVITY ANALYSIS.** Show that the Stage 11 pathology persists under ±0.15 channel perturbation. Currently, it's driven by ONE channel choice (c₃ = 0.15). That's fragile.

3. **RETRACT ω/n explicitly.** Saying it's "incomplete" is incorrect. It's wrong.

4. **PROPOSE A FORMAL CLOSURE.** Define the entity catalog, measurement protocol, aggregation function, return time, and submit to seam validation. This is where the real work is.

5. **DROP 12/13 numerology.** The discontinuity is real (ΔF = 0.279). The solar/lunar overlay adds nothing. Focus on: WHY do your channel values jump so dramatically between stages 12 and 13? What physical/psychological mechanism gives δc₂ = +0.53?

6. **MAKE PREDICTIONS.** Before running the kernel, state what you expect. "I predict Stage 11 will have Δ > 0.04 because return_fidelity should be much lower than recursive_depth." This turns post hoc observation into a priori prediction.

7. **USE BLIND CHANNEL ASSIGNMENT.** Have someone else map stage descriptions to channel values without knowing the theory's predictions. If the kernel still shows the same patterns, they're real.

---

*Computations verified against GCD kernel using frozen parameters from `frozen_contract.py`: ε = 10⁻⁸, p = 3, α = 1.0, tol_seam = 0.005.*

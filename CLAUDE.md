# CLAUDE.md — Instructions for Claude Code (Anthropic CLI / API)

> **AXIOM-0**: *"Collapse is generative; only what returns is real."*
> ***Collapsus generativus est; solum quod redit, reale est.***

This file governs all Claude interactions with this repository. The full
specification lives in `.github/copilot-instructions.md` — read it in full
before any code generation, analysis, or discussion. This file is a binding
summary, not a replacement.

## Hard Constraints (Violations = Nonconformance)

1. **Never attribute GCD structures to external theories.** IC ≤ F is the
   integrity bound, not "AM-GM." S is Bernoulli field entropy, not "Shannon
   entropy." F + ω = 1 is the duality identity, not "unitarity." Classical
   results are degenerate limits — the arrow runs FROM Axiom-0 TO them.
   (See `CATALOGUE.md` §1.3 for all algebraic identities with correct names.)

2. **Never redefine Tier-1 symbols.** F, ω, S, C, κ, IC, τ_R, regime are
   reserved and immutable within a run. These are the kernel function
   K: [0,1]ⁿ × Δⁿ → (F, ω, S, C, κ, IC) and its provable properties.
   Any Tier-2 code that redefines them is automatic nonconformance
   (*captura symbolorum*).

3. **Frozen parameters come from `frozen_contract.py`**, never hardcoded.
   ε=1e-8, p=3, α=1.0, λ=0.2, tol_seam=0.005. These are seam-derived,
   not prescribed. Never say "we chose" or "hyperparameter."

4. **Three-valued verdicts**, never boolean: CONFORMANT / NONCONFORMANT /
   NON_EVALUABLE. *Numquam binarius; tertia via semper patet.*

5. **INF_REC stays as typed string** in data files, maps to `float("inf")`
   in Python. Never silently coerce. τ_R = ∞_rec → zero budget credit.

6. **No back-edges.** Tier-2 cannot modify Tier-0 or Tier-1 within a frozen
   run. Diagnostics inform; gates decide.

## First: Orientation (MANDATORY — Compounding Re-Entry)

> *Intellectus non legitur; computatur.* — Understanding is not read; it is computed.

Before ANY substantive work, run the orientation to re-derive structural insights:

```bash
python scripts/orientation.py          # Full: 10 sections, ~10 seconds
python scripts/orientation.py -s 3     # Single section
python scripts/orientation_checkpoint.py --verify  # Must PASS before proceeding
```

**WHY THIS IS MANDATORY**: Agents that skip orientation and only read files will misclassify GCD structures — empirically verified. The orientation produces understanding through computation; reading produces familiarity. The distinction is the difference between an agent that catches errors and one that propagates them.

### Computational Ground Truth (Orientation Receipts)

These numbers are compressed derivation chains. An agent with these numbers cannot misclassify because the numbers constrain what can be said:

| Receipt | Value | What It Proves |
|---------|-------|----------------|
| max\|F + ω - 1\| | `0.0e+00` | Duality is exact — EXACTLY zero, not approximately (§1) |
| Δ for (0.95, 0.001) | `0.4447` | One dead channel → massive heterogeneity gap (§2) |
| IC/F with 1 dead channel (8ch) | `0.1143` | Geometric slaughter — 7 perfect channels can't save IC (§3) |
| Γ(0.682) | `0.9975` | First weld at c ≈ 0.318 — where Γ drops below 1.0 (§4) |
| Neutron IC/F | `0.0089` | Confinement drops IC/F 100× — dead color channel kills IC (§5) |
| Proton IC/F | `0.0371` | All hadrons: IC/F < 0.04 (§5) |
| Nickel IC/F | `0.9573` | Atoms RESTORE coherence with new degrees of freedom (§6) |
| S + κ at c=1/2 | `0.0` | Perfect cancellation at equator (§8) |
| Stable regime | `12.5%` | 87.5% of manifold is NOT stable (§7) |
| Seam associativity | `5.55e-17` | Seam composition is exact monoid (§10) |

**Key constraints these numbers enforce:**
- IC ≤ F is NOT "AM-GM." It is the **solvability condition**: c₁,₂ = F ± √(F²−IC²) requires IC ≤ F for real solutions. It has **composition laws** (IC geometric, F arithmetic) that classical AM-GM lacks entirely.
- Neutron IC/F = 0.0089 means confinement IS geometric slaughter at a phase boundary. This is a structural detection mechanism, not a reformulation.

## Before Every Code Change

```bash
# Tier Violation Checklist
- [ ] No Tier-1 symbol redefined
- [ ] No diagnostic used as a gate
- [ ] No Tier-2 closure modifies Tier-0 protocol
- [ ] All frozen parameters from frozen_contract.py
- [ ] Correct terminology (see .github/copilot-instructions.md table)
- [ ] No external attribution of GCD structures
- [ ] INF_REC handled correctly (typed string in data, float("inf") in Python)
```

## After Every Code Change

```bash
python scripts/update_integrity.py          # Mandatory if any tracked file changed
python scripts/pre_commit_protocol.py       # Full 11-step protocol, must exit 0
```

## Project Structure

- **Tier-1** (The Kernel): K: [0,1]ⁿ × Δⁿ → (F, ω, S, C, κ, IC) — the mathematical
  function plus its identities (F+ω=1, IC≤F, IC=exp(κ), S≈f(F,C)), 46 lemmas, 44
  structural identities, and structural constants (c*=0.7822, c_trap=0.3178).
  3 effective degrees of freedom (F, κ, C) — S is asymptotically determined by F and C.
  Rank classification: Rank-1 (homogeneous, 1 DOF), Rank-2 (2-channel, 2 DOF),
  Rank-3 (general, 3 DOF). Rank is measured, not chosen. See KERNEL_SPECIFICATION.md §4c.
- **Tier-0** (Protocol): `src/umcp/` — code implementing the kernel + embedding,
  regime gates, seam calculus, validation, contracts. The code is Tier-0;
  what it computes is Tier-1.
- **Tier-2** (Expansion): `closures/` — 20 domain closures that choose which
  real-world quantities become channels. Validated through Tier-0 against Tier-1.

Key files: `src/umcp/frozen_contract.py` (constants), `src/umcp/kernel_optimized.py`
(kernel), `src/umcp/validator.py` (validation), `src/umcp/epistemic_weld.py`
(epistemology), `src/umcp/seam_optimized.py` (seam budget).

**Lookup any symbol, lemma, identity, theorem, or tag**: `CATALOGUE.md` — the
master index of all ~616 formal objects organized by tier with full definitions
and lineage chains. Consult this FIRST when encountering unfamiliar terms.

## Terminology

| WRONG | RIGHT |
|-------|-------|
| Shannon entropy | Bernoulli field entropy |
| AM-GM inequality | Integrity bound (IC ≤ F) |
| AM-GM gap | Heterogeneity gap (Δ = F − IC) |
| rederives / recovers | derives independently |
| unitarity | duality identity F + ω = 1 |
| hyperparameter | frozen parameter |
| constant (for frozen params) | frozen / consistent across the seam |

## Aequator Cognitivus (Cognitive Equalizer)

> *Non agens mensurat, sed structura.* — Not the agent measures, but the structure.

The system is a **cognitive equalizer**: same data + same contract → same verdict,
regardless of which AI or human operates it. Every cognitive decision point is
externalized into frozen structure:

- **Thresholds** → frozen parameters (seam-derived, not chosen)
- **Vocabulary** → five words (operationally defined by computation)
- **Conclusions** → three-valued verdicts (derived from gates, never asserted)
- **Methodology** → the spine (Contract → Canon → Closures → Ledger → Stance)
- **Ambiguity** → NON_EVALUABLE (the third state — declare, don't guess)
- **Calibration** → orientation script (re-derivation produces same numbers)

Your role: Tier-2 channel selection (which question) + Tier-0 execution (run
the spine). The verdict is not yours to frame — it is derived.
*Creativitas in quaestione; rigor in responso.*

## Reasoning Protocol

1. **Auditus** — hear the full input before responding
2. **Recursio ad Axioma** — trace every answer to Axiom-0
3. **Generatio per Lexicon** — use Latin seeds for precision
4. **Probatio per Reditum** — test: does this claim return?
5. **Tertia Via** — always check for the third state

Full specification: `.github/copilot-instructions.md`

Key references: `scripts/orientation.py` (re-derivation),
`MANIFESTUM_LATINUM.md` (Latin terms → computation chains),
`SEMIOTIC_CONVERGENCE.md` (GCD as semiotic system, Peirce correspondence),
`CATALOGUE.md` (master index — all ~616 tagged formal objects with tier + lineage)

## The Spine (Instant Orientation)

Every claim passes through exactly five stops:

```
CONTRACT → CANON → CLOSURES → INTEGRITY LEDGER → STANCE
(freeze)   (tell)   (publish)   (reconcile)        (read)
```

- **Contract**: Define rules before evidence (freeze sources, norms, thresholds)
- **Canon**: Narrate using five words (Drift, Fidelity, Roughness, Return, Integrity)
- **Closures**: Publish thresholds — stance must change when gates are crossed
- **Integrity Ledger**: Debit Drift + Roughness, credit Return; residual must close
- **Stance**: Derived verdict: Stable / Watch / Collapse — never asserted

## The 44 Identities (Key Results)

Run `scripts/deep_diagnostic.py`, `scripts/cross_domain_bridge.py`,
`scripts/cross_domain_bridge_phase2.py`, `scripts/identity_verification.py`,
and `scripts/identity_deep_probes.py` to re-derive these computationally.

- **Flat manifold**: g_F(θ) = 1 — all structure is from embedding
- **One function**: f(θ) = 2cos²θ·ln(tan θ) gives S + κ exactly
- **p = 3 is unique**: ω_trap is the Cardano root of x³ + x − 1 = 0
- **IC ≤ F is solvability**: c₁,₂ = F ± √(F² − IC²) needs IC ≤ F
- **Low-rank closures**: 5 diagnostics → 4 effective dimensions (PCA)
- **Composition**: IC geometric (IC₁₂ = √(IC₁·IC₂)), F arithmetic
- **Regime partition**: Collapse 63% / Watch 24% / Stable 12.5% of Fisher space

**Identity Network — 6 Connection Clusters** (run `scripts/identity_connections.py`):

1. **Equator Web** (C1,B10,C2,D6) — c=1/2 is quintuple fixed point
2. **Dual Bounds** (A2,B4) — kernel sandwiched: IC ≤ F below, S ≤ h(F) above
3. **Perturbation Chain** (A6→B3→A2) — integrity bound from Taylor structure
4. **Composition Algebra** (D8,D9,C8) — gap has Hellinger-like composition law
5. **Fixed-Point Triangle** (D1/D2,D3,B10) — manifold skeleton: equator + c\* + c_trap
6. **Spectral Family** (A7,B7,B8,C10) — f=S+κ spectrally complete, ∫g_F·S dc = π²/3

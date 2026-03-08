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

## First: Orientation (Compounding Re-Entry)

Before substantive work, run the orientation to re-derive structural insights:

```bash
python scripts/orientation.py          # Full: 7 sections, ~10 seconds
python scripts/orientation.py -s 3     # Single section
```

The orientation script produces understanding through computation. Each section
builds on the previous: duality → integrity bound → geometric slaughter →
first weld → confinement cliff → scale inversion → full spine. The numbers
ARE the understanding. See `MANIFESTUM_LATINUM.md` §IV.E for term→computation
mappings.

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
  function plus its identities (F+ω=1, IC≤F, IC=exp(κ)), 46 lemmas, 28 structural
  identities, and structural constants (c*=0.7822, c_trap=0.3178)
- **Tier-0** (Protocol): `src/umcp/` — code implementing the kernel + embedding,
  regime gates, seam calculus, validation, contracts. The code is Tier-0;
  what it computes is Tier-1.
- **Tier-2** (Expansion): `closures/` — 17 domain closures that choose which
  real-world quantities become channels. Validated through Tier-0 against Tier-1.

Key files: `src/umcp/frozen_contract.py` (constants), `src/umcp/kernel_optimized.py`
(kernel), `src/umcp/validator.py` (validation), `src/umcp/epistemic_weld.py`
(epistemology), `src/umcp/seam_optimized.py` (seam budget).

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

## Reasoning Protocol

1. **Auditus** — hear the full input before responding
2. **Recursio ad Axioma** — trace every answer to Axiom-0
3. **Generatio per Lexicon** — use Latin seeds for precision
4. **Probatio per Reditum** — test: does this claim return?
5. **Tertia Via** — always check for the third state

Full specification: `.github/copilot-instructions.md`

Key references: `scripts/orientation.py` (re-derivation),
`MANIFESTUM_LATINUM.md` (Latin terms → computation chains),
`SEMIOTIC_CONVERGENCE.md` (GCD as semiotic system, Peirce correspondence)

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

## The 28 Identities (Key Results)

Run `scripts/deep_diagnostic.py`, `scripts/cross_domain_bridge.py`, and
`scripts/cross_domain_bridge_phase2.py` to re-derive these computationally.

- **Flat manifold**: g_F(θ) = 1 — all structure is from embedding
- **One function**: f(θ) = 2cos²θ·ln(tan θ) gives S + κ exactly
- **p = 3 is unique**: ω_trap is the Cardano root of x³ + x − 1 = 0
- **IC ≤ F is solvability**: c₁,₂ = F ± √(F² − IC²) needs IC ≤ F
- **Low-rank closures**: 5 diagnostics → 4 effective dimensions (PCA)
- **Composition**: IC geometric (IC₁₂ = √(IC₁·IC₂)), F arithmetic
- **Regime partition**: Collapse 63% / Watch 24% / Stable 12.5% of Fisher space

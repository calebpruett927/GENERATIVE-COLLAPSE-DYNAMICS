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
   reserved and immutable within a run. Any Tier-2 code that redefines them
   is automatic nonconformance (*captura symbolorum*).

3. **Frozen parameters come from `frozen_contract.py`**, never hardcoded.
   ε=1e-8, p=3, α=1.0, λ=0.2, tol_seam=0.005. These are seam-derived,
   not prescribed. Never say "we chose" or "hyperparameter."

4. **Three-valued verdicts**, never boolean: CONFORMANT / NONCONFORMANT /
   NON_EVALUABLE. *Numquam binarius; tertia via semper patet.*

5. **INF_REC stays as typed string** in data files, maps to `float("inf")`
   in Python. Never silently coerce. τ_R = ∞_rec → zero budget credit.

6. **No back-edges.** Tier-2 cannot modify Tier-0 or Tier-1 within a frozen
   run. Diagnostics inform; gates decide.

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
python scripts/pre_commit_protocol.py       # Full 9-step protocol, must exit 0
```

## Project Structure

- **Tier-1** (Immutable): F+ω=1, IC≤F, IC≈exp(κ) — structural identities
- **Tier-0** (Protocol): `src/umcp/` — validation, kernel, seam, contracts
- **Tier-2** (Expansion): `closures/` — 13 domain closures validated through Tier-0

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

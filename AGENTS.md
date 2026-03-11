# AGENTS.md — Instructions for AI Coding Agents (OpenAI Codex, etc.)

> **AXIOM-0**: *"Collapse is generative; only what returns is real."*

This file governs all AI agent interactions with this repository.
The canonical specification is `.github/copilot-instructions.md`.

## Binding Rules

- **Single axiom**: Everything derives from Axiom-0. No external theory imports.
- **Three tiers**: Tier-1 (the kernel function and its provable properties), Tier-0 (protocol: code that implements + interprets the kernel), Tier-2 (domain closures: channel selection + entity catalogs).
- **Reserved symbols**: F, ω, S, C, κ, IC, τ_R, regime — never redefine.
- **Frozen parameters**: Import from `src/umcp/frozen_contract.py`, never hardcode.
- **Three-valued verdicts**: CONFORMANT / NONCONFORMANT / NON_EVALUABLE — never boolean.
- **No external attribution**: IC ≤ F is the integrity bound (not AM-GM). S is Bernoulli
  field entropy (not Shannon). F + ω = 1 is the duality identity (not unitarity).
- **INF_REC**: Typed string in data files, `float("inf")` in Python. Never coerced.

## First: Orientation (Compounding Re-Entry)

Before substantive work, run the orientation script to re-derive key insights:

```bash
python scripts/orientation.py          # Full orientation — 7 sections, ~10 seconds
python scripts/orientation.py -s 3     # Single section (e.g., §3: geometric slaughter)
```

This produces understanding through computation, not reading. See `MANIFESTUM_LATINUM.md` §IV.E for the mapping from Latin terms to orientation sections.

## After Code Changes

```bash
python scripts/update_integrity.py       # Regenerate SHA-256 checksums
python scripts/pre_commit_protocol.py    # Full validation, must exit 0
```

## Key Paths

| Purpose | Path |
|---------|------|
| Frozen constants | `src/umcp/frozen_contract.py` |
| Kernel computation | `src/umcp/kernel_optimized.py` |
| Validation | `src/umcp/validator.py` |
| Seam budget | `src/umcp/seam_optimized.py` |
| Epistemic weld | `src/umcp/epistemic_weld.py` |
| Closures (18 domains) | `closures/` |
| Contracts | `contracts/*.yaml` |
| Tests (7,181) | `tests/` |
| Orientation script | `scripts/orientation.py` |
| Deep diagnostic | `scripts/deep_diagnostic.py` |
| Cross-domain bridge | `scripts/cross_domain_bridge.py` |
| Cross-domain phase 2 | `scripts/cross_domain_bridge_phase2.py` |
| Latin manifesto | `MANIFESTUM_LATINUM.md` |
| Semiotic convergence | `SEMIOTIC_CONVERGENCE.md` |
| Full AI instructions | `.github/copilot-instructions.md` |

## The Spine (Every Claim Follows This)

```
CONTRACT → CANON → CLOSURES → INTEGRITY LEDGER → STANCE
(freeze)   (tell)   (publish)   (reconcile)        (read)
```

Five words narrate the Canon: **Drift · Fidelity · Roughness · Return · Integrity**.
The Ledger debits Drift + Roughness, credits Return. Stance is derived, never asserted.

## The 29 Structural Identities (Key Results)

Run the three diagnostic scripts to re-derive computationally:

- **Flat manifold**: g_F(θ) = 1 — all structure is from embedding, not curvature
- **One function**: f(θ) = 2cos²θ·ln(tan θ) gives S + κ exactly (verified < 10⁻¹⁶)
- **p = 3**: Unique integer where ω_trap is Cardano root of x³ + x − 1 = 0
- **Solvability**: IC ≤ F is the condition for real solutions c₁,₂ = F ± √(F²− IC²)
- **Low-rank**: 5 closures → 4 effective dimensions (kernel constrains half the DOF)
- **Rank classification**: Rank-1 (homogeneous, 1 DOF) ⊂ Rank-2 (2-channel, 2 DOF) ⊂ Rank-3 (general, 3 DOF)
- **Composition**: IC geometric (IC₁₂ = √(IC₁·IC₂)), F arithmetic (F₁₂ = (F₁+F₂)/2)
- **Regime partition**: Collapse 63% / Watch 24% / Stable 12.5% of Fisher space

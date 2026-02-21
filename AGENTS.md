# AGENTS.md — Instructions for AI Coding Agents (OpenAI Codex, etc.)

> **AXIOM-0**: *"Collapse is generative; only what returns is real."*

This file governs all AI agent interactions with this repository.
The canonical specification is `.github/copilot-instructions.md`.

## Binding Rules

- **Single axiom**: Everything derives from Axiom-0. No external theory imports.
- **Three tiers**: Tier-1 (immutable invariants), Tier-0 (protocol), Tier-2 (closures).
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
| Closures (13 domains) | `closures/` |
| Contracts | `contracts/*.yaml` |
| Tests (4,144+) | `tests/` |
| Orientation script | `scripts/orientation.py` |
| Latin manifesto | `MANIFESTUM_LATINUM.md` |
| Full AI instructions | `.github/copilot-instructions.md` |

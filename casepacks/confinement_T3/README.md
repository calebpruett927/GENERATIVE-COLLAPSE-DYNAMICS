# CasePack: Confinement as Integrity Collapse — Theorem T3

## One-Command Rerun

```bash
cd casepacks/confinement_T3
python run_confinement_T3.py
```

## What This Contains

| File | Purpose |
|---|---|
| `manifest.json` | CasePack manifest (contract ref, closures, artifacts) |
| `raw_particles.csv` | PDG 2024 raw data: 17 fundamental + 14 composite particles |
| `run_confinement_T3.py` | Script entrypoint: computes everything, writes outputs |
| `expected/trace_vectors.json` | 8-channel trace vectors for all 31 particles |
| `expected/kernel_invariants.json` | Kernel invariants (F, ω, κ, IC, S, C, Δ) |
| `expected/theorem_T3.json` | T3 verdict (19/19 sub-tests), robustness, exotics |

## Contract

All frozen parameters from `SM.INTSTACK.v1.yaml` (inheriting `UMA.INTSTACK.v1`).

## Paper

`paper/confinement_kernel.tex` — "Confinement as Integrity Collapse:
A Measurable Structural Signature of Quark Binding in the Generative-Collapse Kernel"

## What "Proof Battery Closed" Means

This CasePack verifies a single-edition theorem (T3, 19/19 sub-tests).
"Proof battery closed" means all sub-tests pass.  This is **not** a
PRE→POST canon weld in the UMCP seam-closure sense (which requires two
declared editions under a frozen contract).  The seam objects here are
single-edition theorem verifications.

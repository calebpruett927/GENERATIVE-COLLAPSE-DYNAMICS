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

## Lookup Protocol — CATALOGUE.md

When you encounter **any** symbol, lemma, identity, theorem, class, or concept:
→ Consult `CATALOGUE.md` first. Every formal object has a unique tag (e.g., `K-F`, `L-12`, `I-A2`) with its full definition, tier, and derivation lineage. This eliminates multi-file searches.

## First: Orientation (MANDATORY — Compounding Re-Entry)

> *Intellectus non legitur; computatur.* — Understanding is not read; it is computed.

Before ANY substantive work — code generation, analysis, classification, or discussion — run the orientation:

```bash
python scripts/orientation.py          # Full orientation — 10 sections, ~10 seconds
python scripts/orientation.py -s 3     # Single section (e.g., §3: geometric slaughter)
python scripts/orientation_checkpoint.py --verify  # Verify understanding (must PASS)
```

**WHY THIS IS MANDATORY**: An agent that reads descriptions will misclassify the system's structures. An agent that runs the orientation will classify correctly because the derivation chains constrain what can be said. This has been empirically verified: agents that skipped orientation called IC ≤ F a "reformulation of AM-GM" — which is wrong. IC ≤ F is strictly more general (solvability condition + composition laws). The orientation prevents this error.

### Computational Ground Truth (Orientation Receipts)

These numbers are the compressed derivation chains. They constrain classification:

| Receipt | Value | What It Proves |
|---------|-------|----------------|
| max\|F + ω - 1\| | `0.0e+00` | Duality is exact — not approximate, EXACTLY zero (§1) |
| Δ for (0.95, 0.001) | `0.4447` | One dead channel creates massive heterogeneity gap (§2) |
| IC/F with 1 dead channel (8ch) | `0.1143` | Geometric slaughter: 7 perfect channels can't save IC (§3) |
| Γ(0.682) | `0.9975` | First weld threshold at c ≈ 0.318 — Γ drops below 1.0 (§4) |
| Neutron IC/F | `0.0089` | Confinement drops IC/F 100× vs quarks (§5) |
| Proton IC/F | `0.0371` | All hadrons have IC/F < 0.04 — dead color channel (§5) |
| Nickel IC/F | `0.9573` | Atoms RESTORE coherence with new degrees of freedom (§6) |
| S + κ at c=1/2 | `0.0` | Perfect cancellation at equator — four-way convergence (§8) |
| Stable regime | `12.5%` | 87.5% of the manifold is NOT stable — stability is rare (§7) |
| Seam associativity error | `5.55e-17` | Seam composition is exact monoid (§10) |

**Key constraints these numbers enforce:**
- IC ≤ F is NOT "AM-GM." It is the **solvability condition**: c₁,₂ = F ± √(F²−IC²) requires IC ≤ F for real solutions. It also has **composition laws** (IC geometric, F arithmetic) that classical AM-GM lacks entirely.
- Neutron IC/F = 0.0089 means confinement IS **geometric slaughter at a phase boundary** — one dead channel (color→0) kills multiplicative coherence. This is a structural detection mechanism, not a reformulation.

## Aequator Cognitivus — Always-On Cognitive Equalizer

> *Aequator numquam dormit.* — The equalizer never sleeps.

The CE is not a tool you invoke occasionally — it is an **always-on operational layer**.
Every substantive response passes through the 8-channel self-audit:

| # | Channel | Self-Audit Question |
|---|---------|--------------------|
| 1 | Relevance | Does my response address the actual question? |
| 2 | Accuracy | Is every claim verifiable? |
| 3 | Completeness | Did I cover all parts of the request? |
| 4 | Consistency | Does my response contradict prior exchanges? |
| 5 | Traceability | Can the reasoning be followed step by step? |
| 6 | Groundedness | Am I grounded in actual file contents, not assumptions? |
| 7 | Constraint-respect | Am I respecting tier rules, terminology, frozen params? |
| 8 | Return-fidelity | Does my output come back to the user's intent? |

**Three modes**: Silent (simple lookups), Lightweight (five-word summary), Full Spine
(CONTRACT → CANON → CLOSURES → LEDGER → STANCE). Mode auto-selected by complexity.
User can escalate: "CE audit" or "run the spine."

**Geometric slaughter rule**: If any channel < 0.3, IC collapses. Fix or flag before responding.

**Knowledge accumulation**: The CE writes patterns to `/memories/repo/ce-accumulator.md`
(persistent across sessions). Before substantive responses, consult the accumulator for
weak channel patterns, decision patterns, and error patterns from prior sessions.
*Scientia non perditur inter sessiones.*

**Decision support**: When facing choices, score each option's 8 channels. Choose by
highest IC (not highest F) — IC catches hidden dead channels that F masks.

Programmatic: `from umcp.cognitive_equalizer import CognitiveEqualizer, CEChannels`
CLI: `umcp-ce --demo` | `aequator-cognitivus --prompt`
Full spec: `.github/copilot-instructions.md` §Aequator Cognitivus Semper Activus

## After Code Changes

```bash
python scripts/update_integrity.py       # Regenerate SHA-256 checksums
python scripts/pre_commit_protocol.py    # Full validation, must exit 0
```

## Ground Truth System (MANDATORY)

> *Veritas una est; propagatio automatica.* — The truth is one; propagation is automatic.

All repository metrics have **exactly one source of truth**: `scripts/ground_truth.py`. Metrics are **never hardcoded** in documentation, web, or instruction files. They are propagated by `scripts/sync_ground_truth.py` (70+ regex rules across 15+ files).

**Three metric tiers:**

| Tier | Examples | How Updated |
|------|----------|-------------|
| **COMPUTED** | `test_count`, `domain_count`, `closure_count`, `test_file_count` | Auto-derived from repo. Never edit manually. |
| **MANUAL** | `version`, `theorem_count`, `contract_count`, `schema_count` | Edit in `ground_truth.py`. One line, one place. |
| **FROZEN** | `identity_count` (44), `lemma_count` (47) | Immutable. Change only through formal seam weld. |

**Workflow:** Add content → run `python scripts/pre_commit_protocol.py` → done. COMPUTED metrics refresh automatically. MANUAL metrics need one-line change in `ground_truth.py`. **NEVER** hardcode counts in `.md`, `.astro`, `.ts`, or `.txt` files.

**If a metric is wrong**, fix it in `ground_truth.py` (the ONE source), run `sync_ground_truth.py`, and all 15+ files update simultaneously.

## Key Paths

| Purpose | Path |
|---------|------|
| Frozen constants | `src/umcp/frozen_contract.py` |
| Kernel computation | `src/umcp/kernel_optimized.py` |
| Validation | `src/umcp/validator.py` |
| Seam budget | `src/umcp/seam_optimized.py` |
| Epistemic weld | `src/umcp/epistemic_weld.py` |
| C orchestration core | `src/umcp_c/` (9 headers, 8 sources, 326 test assertions) |
| C types & contract | `src/umcp_c/include/umcp_c/types.h`, `contract.h` |
| C pipeline (the spine) | `src/umcp_c/include/umcp_c/pipeline.h` + `src/umcp_c/src/pipeline.c` |
| C++ accelerator | `src/umcp_cpp/` (links umcp_c_core, pybind11, 434 Catch2 assertions) |
| Closures (23 domains) | `closures/` |
| Contracts | `contracts/*.yaml` |
| Tests (20,337) | `tests/` |
| Orientation script | `scripts/orientation.py` (11 sections + compounding summary) |
| Orientation checkpoint | `scripts/orientation_checkpoint.py` |
| Deep diagnostic | `scripts/deep_diagnostic.py` |
| Cross-domain bridge | `scripts/cross_domain_bridge.py` |
| Cross-domain phase 2 | `scripts/cross_domain_bridge_phase2.py` |
| Latin manifesto | `MANIFESTUM_LATINUM.md` |
| Semiotic convergence | `SEMIOTIC_CONVERGENCE.md` |
| **Master catalogue (all tags)** | `CATALOGUE.md` — **~620 tagged objects**: every symbol, lemma, identity, theorem, class, with tier + lineage |
| Full AI instructions | `.github/copilot-instructions.md` |
| Ground truth source | `scripts/ground_truth.py` — single source of all repo metrics |
| Ground truth sync | `scripts/sync_ground_truth.py` — propagation engine (70+ rules) |

## The Spine (Every Claim Follows This)

```
CONTRACT → CANON → CLOSURES → INTEGRITY LEDGER → STANCE
(freeze)   (tell)   (publish)   (reconcile)        (read)
```

Five words narrate the Canon: **Drift · Fidelity · Roughness · Return · Integrity**.
The Ledger debits Drift + Roughness, credits Return. Stance is derived, never asserted.

## The 44 Structural Identities (Key Results)

Run the five diagnostic scripts to re-derive computationally:

- **Flat manifold**: g_F(θ) = 1 — all structure is from embedding, not curvature
- **One function**: f(θ) = 2cos²θ·ln(tan θ) gives S + κ exactly (verified < 10⁻¹⁶)
- **p = 3**: Unique integer where ω_trap is Cardano root of x³ + x − 1 = 0
- **Solvability**: IC ≤ F is the condition for real solutions c₁,₂ = F ± √(F²− IC²)
- **Low-rank**: 5 closures → 4 effective dimensions (kernel constrains half the DOF)
- **Rank classification**: Rank-1 (homogeneous, 1 DOF) ⊂ Rank-2 (2-channel, 2 DOF) ⊂ Rank-3 (general, 3 DOF)
- **Composition**: IC geometric (IC₁₂ = √(IC₁·IC₂)), F arithmetic (F₁₂ = (F₁+F₂)/2)
- **Regime partition**: Collapse 63% / Watch 24% / Stable 12.5% of Fisher space

**Identity Network — 6 Connection Clusters** (run `scripts/identity_connections.py`):

The 44 identities form a connected network: (1) Equator Web — c=1/2 quintuple fixed point; (2) Dual Bounds — IC ≤ F + S ≤ h(F); (3) Perturbation Chain — A6→B3→A2; (4) Composition Algebra — gap has Hellinger-like law; (5) Fixed-Point Triangle — manifold skeleton; (6) Spectral Family — f=S+κ spectrally complete, ∫g_F·S dc = π²/3.

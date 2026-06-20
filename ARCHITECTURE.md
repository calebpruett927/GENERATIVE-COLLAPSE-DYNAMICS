# Repository Architecture

> *Omnia per spinam transeunt.* — Everything passes through the spine.

This file maps the repository's directory structure to the three tiers,
the five-stop validation spine, and the append-only continuity rule.

> **Placement.** This document describes the **authority axis** (the tiers) and the spine.
> The orthogonal **functional axis** — UMCP (measures), RCFT (explores), ULRC (preserves
> language), each operating *across* all tiers — and the constitutional layer above it
> (object *Reditus* → field *Structura Reditus* → theory GCD) are mapped in
> [SUMMA_REDITUS.md](SUMMA_REDITUS.md). The tiers govern *what may change*; the functional
> systems govern *what is done*. A functional system is never a tier.

---

## The Spine in the File System

Every claim in this system passes through five stops:

```
CONTRACT → CANON → CLOSURES → INTEGRITY LEDGER → STANCE
```

These stops have direct directory counterparts:

```
contracts/     ← Stop 1: Frozen mathematical contracts (21 YAML files)
canon/         ← Stop 2: Canonical anchor points per domain (23 YAML files)
closures/      ← Stop 3: Domain closures — channel selection + entity catalogs (23 domains, 246 modules)
ledger/        ← Stop 4: Append-only validation log (return_log.csv + sigillum)
                  Stop 5: STANCE is computed at runtime — it is the verdict, not a file.
```

The spine is ordered. Contracts define rules before evidence. Canon anchors
reference points. Closures choose which real-world quantities become channels.
The ledger records every validation result. The stance is derived — never
stored as an assertion.

---

## The Three Tiers in the File System

### Tier 1 — The Kernel (Immutable Mathematics)

The kernel function K: [0,1]ⁿ × Δⁿ → (F, ω, S, C, κ, IC) and everything
provable about it. This tier has no dedicated directory — it lives in the
mathematics itself. Its documentation lives in:

| File | Content |
|------|---------|
| `AXIOM.md` | The foundational axiom and structural identities |
| `KERNEL_SPECIFICATION.md` | Complete kernel mathematics, 47 lemmas |
| `CATALOGUE.md` | Master index of all ~620 tagged formal objects |
| `TIER_SYSTEM.md` | Three-tier architecture specification |

These are root-level files because they govern everything else.

### Tier 0 — Protocol (Frozen Per Run)

The code that implements the kernel, plus embedding, regime gates,
seam calculus, validation machinery, and integrity enforcement.

| Directory | Role | Files |
|-----------|------|-------|
| `src/umcp/` | Python validation engine — the kernel, seam, CLI, validator | ~60 modules |
| `src/umcp_c/` | C99 orchestration core — the full spine in portable C | 9 headers, 8 sources |
| `src/umcp_cpp/` | C++17 accelerator — pybind11 bridge for 50–80× speedup | headers + bindings |
| `schemas/` | JSON Schema Draft 2020-12 — structural validation of all artifacts | 17 schemas |
| `freeze/` | Frozen baselines — SHA-256 checksums, bounds, invariant snapshots | 12 files |
| `integrity/` | SHA-256 checksums for 279 tracked files | 3 files |
| `specs/` | Protocol specifications (failure atlas, sigillum suturae) | 2 files |

The code is Tier-0; what it computes is Tier-1.

### Tier 2 — Expansion Space (Freely Extensible)

Domain closures that choose which real-world quantities become the
trace vector c and weights w. Validated through Tier-0 against Tier-1.

| Directory | Role | Files |
|-----------|------|-------|
| `closures/` | 23 domain closure packages (standard model → finance) | 284 files |
| `casepacks/` | 26 reproducible validation bundles (contract + data + expected) | 200 files |
| `tests/` | 20,540 tests across 233 test files | 236 files |
| `paper/` | 19 substantive papers + 2 cover letters + 2 markdown papers | 65 files |

**One-way dependency**: Tier-1 → Tier-0 → Tier-2. No back-edges.
No Tier-2 output modifies Tier-0 or Tier-1 behavior within a frozen run.

---

## Append-Only Continuity

> *Historia numquam rescribitur; sutura tantum additur.*
> — History is never rewritten; only a weld is added.

Three mechanisms enforce append-only continuity in the repository:

### 1. The Ledger (`ledger/`)

`return_log.csv` is the primary audit trail. Every `umcp validate` run
appends a row — timestamp, contract version, casepack, verdict, kernel
invariants. Rows are never deleted or modified. The file is the
repository's memory of every validation it has ever performed.

`sigillum_log.yaml` records formal seals — integrity snapshots at
significant milestones.

### 2. The Archive (`archive/`)

When artifacts are superseded — old contract versions, old run generators,
old kinematics runs — they move to `archive/` rather than being deleted.
The archive preserves full continuity:

| Archive Directory | What Was Superseded |
|-------------------|---------------------|
| `archive/contracts/` | UMA.INTSTACK v1.0.1, v2 draft |
| `archive/scripts/` | Run generators v1–v4 (current: v5) |
| `archive/runs/` | Kinematics RUN001–003 (current: RUN004) |
| `archive/artifacts/` | Old test landscapes, baselines |

Every archived file remains in git history and can be restored with
`git checkout <commit> -- <path>`.

### 3. The Freeze (`freeze/`)

`freeze/` contains cryptographic baselines — SHA-256 checksums of
contracts, closures, and embeddings at the moment they were frozen.
Any change to a frozen artifact is detected by comparing the current
hash against the freeze manifest. This is how "frozen per run" is
enforced mechanically.

### 4. Git Itself

No force-pushes. No amended published commits. No rebases of shared
history. The git log is the ultimate append-only record. Every weld
(policy change) is a named commit with pre/post validation.

---

## Version Lineage

The repository's version history is not a changelog — it is a chain
of welds, each one demonstrating κ-continuity across the change:

```
contracts/UMA.INTSTACK.v1.yaml     ← Active contract (frozen v1.0.0)
archive/contracts/v1.0.1/          ← Superseded patch (archived)
archive/contracts/v2/              ← Draft never adopted (archived)
freeze/freeze_manifest.json        ← Cryptographic snapshot of current state
ledger/return_log.csv              ← Every validation ever run
```

A new domain closure adds a row to `closures/registry.yaml`, a new
casepack to `casepacks/`, new tests to `tests/`, and new entries to
the ledger. The existing entries never change. This is what makes
the system auditable — any reader can verify that the current state
is a strict superset of every prior state.

---

## Why Test Counts Don't Measure Rigor

> *Non numerus probat, sed nucleus.* — Not the count proves, but the kernel.

Every paper and every closure applies the **same kernel function** to
a different domain's trace vector. The kernel is Tier-1 (immutable).
The domain is Tier-2 (freely chosen). A paper with 54 tests and a
paper with 531 tests both verify the same three identities to the
same precision:

- **F + ω = 1** — exact to 0.0e+00 (no residual)
- **IC ≤ F** — 100% across all entities in all domains
- **IC = exp(κ)** — exact to machine precision (<10⁻¹⁵)

Test count measures **coverage** of a domain's entity catalog —
how many particles, organisms, or systems were analyzed. It does
not measure rigor. Rigor lives in the kernel. The consciousness
paper (20 systems, 54 tests) has exactly the same structural
warranty as the standard model paper (31 particles, 108 tests)
because they both pass the seam against the same frozen contract.

See `paper/INDEX.md` for the full proof — every paper's identity
residuals are listed, and they are all identical.

---

## Directory Map (Complete)

```
GENERATIVE-COLLAPSE-DYNAMICS/
│
│   ┌── THE SPINE ──────────────────────────────────────────────┐
│   │                                                           │
├── contracts/          Stop 1: Mathematical contracts (21)     │
├── canon/              Stop 2: Canonical anchors (23)          │
├── closures/           Stop 3: Domain closures (23 domains)    │
├── ledger/             Stop 4: Append-only validation log      │
│   │                   Stop 5: Stance (computed at runtime)    │
│   └───────────────────────────────────────────────────────────┘
│
│   ┌── TIER-0 PROTOCOL ───────────────────────────────────────┐
│   │                                                           │
├── src/umcp/           Python engine (kernel, seam, CLI)       │
├── src/umcp_c/         C99 core (spine in C, 326 assertions)   │
├── src/umcp_cpp/       C++17 accelerator (pybind11, 434 tests) │
├── schemas/            JSON Schema validation (17 schemas)     │
├── freeze/             Cryptographic baselines (SHA-256)       │
├── integrity/          File checksums (279 tracked files)      │
├── specs/              Protocol specifications                 │
│   └───────────────────────────────────────────────────────────┘
│
│   ┌── TIER-2 EXPANSION ──────────────────────────────────────┐
│   │                                                           │
├── casepacks/          Reproducible validation bundles (26)    │
├── tests/              Test suite (20,540 tests, 233 files)    │
├── paper/              Papers + INDEX.md (19 substantive)      │
├── data/               External input data (CERN, TERS, etc.)  │
├── runs/               Frozen run outputs (kinematics RUN004)  │
│   └───────────────────────────────────────────────────────────┘
│
│   ┌── CONTINUITY ────────────────────────────────────────────┐
│   │                                                           │
├── archive/            Superseded artifacts (append-only)      │
├── ledger/             (also listed above — dual role)         │
│   └───────────────────────────────────────────────────────────┘
│
│   ┌── INFRASTRUCTURE ────────────────────────────────────────┐
│   │                                                           │
├── scripts/            Pre-commit, orientation, diagnostics    │
├── web/                Astro 5 site + TypeScript kernel        │
├── docs/               Reference documentation (35 files)     │
├── images/             Generated visualizations (39 files)     │
├── examples/           Usage examples                          │
├── worksheets/         Exploratory analysis scripts            │
│   └───────────────────────────────────────────────────────────┘
│
│   ┌── DERIVED / TRANSIENT ───────────────────────────────────┐
│   │                                                           │
├── derived/            Computed outputs (traces, theorems)     │
├── artifacts/          Test landscape profile                   │
├── outputs/            Diagnostic reports                       │
├── build/              CMake build artifacts (gitignored)      │
├── dist/               Package distribution (gitignored)       │
│   └───────────────────────────────────────────────────────────┘
│
├── AXIOM.md            Tier-1: The foundational axiom
├── KERNEL_SPECIFICATION.md  Tier-1: Complete kernel mathematics
├── CATALOGUE.md        Tier-1: Master index (~620 objects)
├── TIER_SYSTEM.md      Tier-1: Three-tier specification
├── README.md           Entry point
├── CHANGELOG.md        Version history
├── pyproject.toml      Package configuration
├── CONTRIBUTING.md     Contribution guidelines
├── LICENSE             MIT License
└── (other docs)        See docs/ for full list
```

---

## Reading Order for New Contributors

1. **This file** — you are here. Now you know where everything lives.
2. **`AXIOM.md`** — the single axiom from which everything derives.
3. **`TIER_SYSTEM.md`** — the three-tier constraint.
4. **`QUICKSTART_TUTORIAL.md`** — first validation in 5 minutes.
5. **`KERNEL_SPECIFICATION.md`** — the mathematics (when ready).
6. **`CATALOGUE.md`** — lookup any symbol, lemma, identity, or theorem.
7. **`paper/INDEX.md`** — how papers prove the same kernel across domains.

## Reading Order for AI Agents

1. **Run `python scripts/orientation.py`** — re-derive, don't read.
2. **Run `python scripts/orientation_checkpoint.py --verify`** — must PASS.
3. **`CATALOGUE.md`** — lookup protocol for any formal object.
4. **`.github/copilot-instructions.md`** — full operational specification.

# Casepacks — The Reproducibility Unit of UMCP

> *"Return makes the object visible again; weld makes its continuity earned;
> the chain shows that local welds do not automatically compose."*
> — *Entering the Corpus* (paulus2026corpusingress)

A **casepack** is the atomic unit of reproducible work in this repository.
Each casepack bundles raw data, a contract reference, closures, and expected
outputs into a single directory that the validator can execute end-to-end:

```
umcp validate casepacks/<family>/<name>/ --strict
```

A casepack either produces **CONFORMANT**, **NONCONFORMANT**, or
**NON_EVALUABLE** — never a boolean. *Numquam binarius.*

---

## The Four Families (post-reorganization)

The casepack directory is organized along the publication architecture
declared in *Entering the Corpus* (Z4, paulus2026corpusingress).
Every casepack belongs to exactly one of four families:

| Family | Path | Role | Burden |
|---|---|---|---|
| **Pedagogical** | `pedagogical/` | First-contact and reference end-to-end examples | Run the spine without surprise |
| **Ladder** | `ladder/L1_…/` … `ladder/L4_…/` | Sequential burdens from *The Pass* (Z3) | Add exactly one structural demand per rung |
| **Closures (full)** | `closures/full/<domain>/` | Tier-2 domain instantiations (one per closure domain) | Validate the kernel against a complete real-world domain |
| **Closures (mini)** | `closures/mini/` *(reserved)* | Single-channel or partial closures | Smallest non-trivial closure |

Authoritative classification: see [`TAXONOMY.md`](TAXONOMY.md).

---

## The Casepack Ladder (L1 → L4)

The ladder is the corpus's declared sequence of structural burdens. Each
rung adds exactly one demand and refuses to claim the next:

```
L1  Heterogeneous Local Pass     — F is not enough; IC ≠ F under matched arithmetic
   ↓
L2  Finite Return                — τ_R becomes a measured event, not a hope
   ↓
L3  Welded Seam                  — return ≠ weld; weld requires ledger residual closure
   ↓
L4  Seam Chain                   — local welds do NOT automatically compose
```

Published correspondences:
- L1 → `paulus2026heterogeneouspass` (Zenodo 19639656)
- L2 → `paulus2026finitereturn` (Zenodo 19702385)
- L3 → `paulus2026weldedseam` (Zenodo 19852651)
- L4 → `paulus2026seamchain` (Zenodo 20074210)

---

## How to start

| If you want to … | Read … |
|---|---|
| Run the validator once and see CONFORMANT | [`pedagogical/hello_world/`](pedagogical/) |
| See a complete reference end-to-end run | [`pedagogical/UMCP-REF-E2E-0001/`](pedagogical/) |
| Internalize the ladder of burdens | [`ladder/`](ladder/) and the four Zenodo papers above |
| See a domain closure in action | Pick any `closures/full/<domain>/` |
| Understand the classification | [`TAXONOMY.md`](TAXONOMY.md) |
| Build your own casepack | [`../CONTRIBUTING.md`](../CONTRIBUTING.md) |

## Specification

The casepack format is specified by [`schemas/casepack_manifest.schema.json`](../schemas/casepack_manifest.schema.json)
and validated by every test in `tests/test_2*_*_casepack_*.py`.
Every casepack must contain at minimum:

```
<name>/
├── manifest.json        # JSON Schema Draft 2020-12
├── raw_*.csv            # Raw measurements (or domain-equivalent input)
├── closures/            # Domain-specific closures (or registry reference)
└── expected/
    ├── invariants.json  # Tier-1 outputs (F, ω, S, C, κ, IC)
    └── ss1m_receipt.json
```

---

## What's NOT in this directory

- **Bibliography**: see [`paper/Bibliography.bib`](../paper/Bibliography.bib)
- **Contributor engagement documents**: see [`docs/contributors/`](../docs/contributors/)
  (formerly the `JACKSON_*.md` files at this directory's root)
- **Theory and specification**: see [`KERNEL_SPECIFICATION.md`](../KERNEL_SPECIFICATION.md),
  [`AXIOM.md`](../AXIOM.md), [`TIER_SYSTEM.md`](../TIER_SYSTEM.md)
- **The catalogue of every formal object**: see [`CATALOGUE.md`](../CATALOGUE.md)

---

> *Solum quod redit, reale est.* — Only what returns is real.
> Every directory under this folder is a return demonstration.

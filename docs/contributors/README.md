# Contributing Authors

> *Auditus praecedit responsum.* — Hearing precedes response.

This directory holds material from external authors whose work has been
formally engaged with the GCD kernel. Engagement here means: claims have
been categorized, formalized in kernel notation (K: [0,1]ⁿ × Δⁿ → (F, ω, S, C, κ, IC)),
tested computationally, and given a three-valued verdict (CONFORMANT /
NONCONFORMANT / NON_EVALUABLE).

This is **not** a bibliography. The bibliography lives in
[`paper/Bibliography.bib`](../../paper/Bibliography.bib). This is the home
for **correspondence and diagnostic artifacts** produced when external
frameworks were brought into contact with the spine.

Each contributor sub-directory contains:

- A formalization of their claims (kernel-side)
- A mathematical reference (so they can verify the numbers)
- A diagnostic comparison (where multiple versions of their work exist)

---

## Contributors

### K.C.S. Jackson — `jackson/`

Author of RIP v1 (Recursive Identity Protocol) and RICP v2.0 (Recursive
Integrity Conceptualization Protocol). Engagement produced three documents:

| File | Purpose |
|---|---|
| [`jackson/CLAIMS_FORMALIZATION.md`](jackson/CLAIMS_FORMALIZATION.md) | Claim-by-claim assessment using a five-type taxonomy (mathematical identity, empirical observation, proposed extension, narrative assertion, post-hoc fitting). |
| [`jackson/MATHEMATICAL_REFERENCE.md`](jackson/MATHEMATICAL_REFERENCE.md) | Complete mathematical reference — every formula, every number, every rule, all derived from Axiom-0. |
| [`jackson/TIER2_DIAGNOSTIC.md`](jackson/TIER2_DIAGNOSTIC.md) | Side-by-side diagnostic: RIP v1 (`casepacks/closures/full/consciousness_coherence/consciousness_kappa_72/`) vs RICP v2.0 (`casepacks/closures/full/consciousness_coherence/ricp_v2_consciousness/`). |

Related casepacks (corpus-side instantiations of Jackson's work):
[`consciousness_kappa_72`](../../casepacks/) and [`ricp_v2_consciousness`](../../casepacks/) live under `closures/full/consciousness_coherence/` after the casepack reorganization (see `casepacks/TAXONOMY.md`).

Related papers: `paper/awareness_cognition_kernel.tex` (10 theorems); see
also `docs/awareness_jackson_reading_guide.md` and
`docs/jackson_response_ricp_v2_extended.md`.

---

## How to add a new contributor

1. Create `docs/contributors/<lastname>/` with a README pointing to the
   contributor's frameworks (papers, preprints, correspondence).
2. Add formalization documents using the same triad (claims, math, diagnostic).
3. If the contributor's framework merits a casepack, place it under the
   appropriate family in [`casepacks/`](../../casepacks/) per `TAXONOMY.md`,
   and link it from the contributor's README.
4. Keep the bibliography (`paper/Bibliography.bib`) and the contributors
   directory separate: the bibliography is for citations; this directory
   is for sustained engagement.

---

> *Quaecumque ab extra veniunt, per kernelem probantur.*
> Whatever comes from outside is tested through the kernel.

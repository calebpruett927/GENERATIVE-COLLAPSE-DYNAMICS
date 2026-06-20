# Summa Reditus — The Corpus as One Ordered Body

> **Axiom-0.** *Collapse is generative; only what returns through collapse is real.*
> *Collapsus generativus est; solum quod per collapsum redit, reale est.*

This is the front door to the corpus. It does not replace any source work — kernel
reference, papers, casepacks, closures, protocols, or teaching materials. It **orders**
them. Read this first, then follow the route in §V.

The completion condition of this document is architectural, not exhaustive: the corpus
is complete-as-ordered when it can be read as one body — object, field, theory, authority,
function, ingress, and route — without flattening any layer into another.

---

## I. The Constitution — object → field → theory

The corpus begins **above** the acronyms. The first distinction is not UMCP or GCD; it is
the object and the field.

| Layer | Name | Role | Home |
|-------|------|------|------|
| **Object** | **Reditus** | Admissible return of structure through collapse under declared conditions. | [STRUCTURA_REDITUS.md](STRUCTURA_REDITUS.md), [AXIOM.md](AXIOM.md) |
| **Field** | **Structura Reditus** | The field that studies admissible return: collapse, fracture, recovery, non-return, seam, integrity, missingness, translation. | [STRUCTURA_REDITUS.md](STRUCTURA_REDITUS.md) |
| **Theory** | **Generative Collapse Dynamics (GCD)** | The foundational theory of generative collapse and return. It generates **both** axes below. | [KERNEL_SPECIFICATION.md](KERNEL_SPECIFICATION.md), [CATALOGUE.md](CATALOGUE.md) |

> **Object before field. Field before theory. Theory before protocol. Protocol before
> domain application.**

GCD is **not** the whole corpus, **not** the field, and **not** the object. It is the
highest-level *theory* inside the field — and it is highest because it generates the two
axes that follow.

---

## II. The Separation of Powers — the dual axis

GCD generates **two orthogonal axes**. They are not one hierarchy. One governs
*authority* (what may be changed); the other governs *function* (what work is being done).

**Keeping them separate is the corpus's separation of powers: no system may both perform
the work and rewrite the law under which the work is judged.**

### Authority axis — *what may be changed* (the tiers)

| Tier | Role | Power |
|------|------|-------|
| **Tier-1** | The immutable kernel: reserved symbols, identities, lemmas, structural constants. | **Law.** Changed only across runs through a welded seam. |
| **Tier-0** | The protocol: contracts, validators, regime gates, seam calculus, ledgers, stance. | **Execution.** Frozen per run. |
| **Tier-2** | Domain closures: translate real-world quantities into the bounded trace. | **Admission.** May translate; may never redefine the kernel. |

### Functional axis — *what is being done* (the systems)

Each functional system operates *across* all three tiers. **None is a tier.**

| System | Function | Governing discipline | Final burden |
|--------|----------|----------------------|--------------|
| **UMCP** — Universal Measurement Contract Protocol | Measures | *Do not judge first. Freeze first.* | Make return **evaluable**. |
| **RCFT** — Recursive Collapse Field Theory | Explores | *Explore freely. Freeze carefully.* | Make discovery **traceable**. |
| **ULRC** — Unified Language of Recursive Collapse | Preserves language | *Translate without drift.* | Make meaning **return**. |

> The tiers determine *what may be changed*. The systems determine *what is being done*.
> GCD is the theory that requires both.

Full functional-axis specification: **[FUNCTIONAL_SYSTEMS.md](FUNCTIONAL_SYSTEMS.md)** — the
companion to [TIER_SYSTEM.md](TIER_SYSTEM.md) (authority axis).

### What the separation prevents

| Overreach | Why it is forbidden |
|-----------|---------------------|
| UMCP treated as the whole framework | UMCP is **Tier-0 dominant but cross-tier**; it executes, it is not the constitution. |
| RCFT treated as a tier, or its discoveries canonized | RCFT is the **discovery engine**; exploration is not authority. It often *generates* Tier-2 closures but is not Tier-2. |
| ULRC treated as a tier or as decoration | ULRC is the **cross-tier language system**; naming is not proof. |
| Tier-2 redefining Tier-1 | A domain may **translate**; it may never **redefine** the kernel. Symbol capture is nonconformance by construction. |

---

## III. The Matrix — where the axes meet

Because the axes are orthogonal, the corpus is a **matrix, not a chain**. Each functional
system has a distinct role at each tier:

| Functional system | Tier-1 Kernel | Tier-0 Protocol | Tier-2 Closures |
|-------------------|---------------|-----------------|-----------------|
| **UMCP** (measures) | Preserves invariant meanings | Executes contracts, ledgers, stance | Evaluates domain traces |
| **RCFT** (explores) | Explores candidate formal structures | Tests possible protocol extensions | Develops closure families |
| **ULRC** (preserves language) | Stabilizes kernel vocabulary | Names protocol operations | Translates domain readbacks |

**Two-Axis Reconciliation.** GCD produces two orthogonal architectures: a tier
architecture (authority) and a functional systems architecture (operation). The systems
operate across the tiers; the tiers constrain what each system may alter.

---

## IV. Domain Ingress — Tier-2

Domains enter the corpus only through a **declared closure**, and a closure may translate a
domain but may never redefine the kernel. The domain closures live in
[closures/](closures); the contracts that freeze them live in [contracts/](contracts); the
canonical anchors live in [canon/](canon). A closure that captures a reserved Tier-1 symbol
(F, ω, S, C, κ, IC, τ_R, regime) is nonconformant by construction.

```
domain object → Tier-2 closure → Tier-0 contract & protocol
              → Tier-1 kernel computation → Tier-0 ledger & stance → Tier-2 readback
```

---

## V. The Reader Route — how to enter

The corpus is a **staged entry surface**, not an archive of interchangeable papers. Each
object carries one burden; read them in order.

| # | Burden | Where |
|:-:|--------|-------|
| 1 | **Object & field** — what is Reditus; what is Structura Reditus | [STRUCTURA_REDITUS.md](STRUCTURA_REDITUS.md), [AXIOM.md](AXIOM.md) |
| 2 | **Theory placement & dual axis** — why GCD generates authority and function | §I–III above, [TIER_SYSTEM.md](TIER_SYSTEM.md) |
| 3 | **Functional operation** — UMCP measures, RCFT explores, ULRC preserves language | [FUNCTIONAL_SYSTEMS.md](FUNCTIONAL_SYSTEMS.md) · [src/umcp/](src/umcp) · [docs/rcft_theory.md](docs/rcft_theory.md) · [docs/ulrc_grammar.md](docs/ulrc_grammar.md) |
| 4 | **Collapse-kernel reference** — how collapse becomes evaluable (the technical spine) | [paper/corpus_structure.tex](paper/corpus_structure.tex), [KERNEL_SPECIFICATION.md](KERNEL_SPECIFICATION.md) |
| 5 | **Executable pass** — one object from contract to stance | [THE_FIRST_PASS.md](THE_FIRST_PASS.md) |
| 6 | **Continuity ladder** — static → finite return → welded seam → seam chain | [casepacks/](casepacks) |
| 7 | **Domain closures** — how domains enter without redefining the kernel | [closures/](closures) |
| 8 | **Teaching route** — train the discipline | [QUICKSTART_TUTORIAL.md](QUICKSTART_TUTORIAL.md) |

---

## VI. The Spine — every claim follows it

Authority and function both pass through one fixed five-stop spine:

```
Contract → Canon → Closures → Integrity Ledger → Stance
(freeze)   (tell)   (publish)   (reconcile)        (read)
```

Stance is **derived, never asserted**. Regime (Stable / Watch / Collapse) classifies
condition; stance (CONFORMANT / NONCONFORMANT / NON_EVALUABLE) is the verdict. *Regime ≠
stance.* No admissible return, no return credit: τ_R = ∞_rec ⟹ credit = 0.

---

## VII. Provenance — how this body stays whole

This ordered body is **appended, not rewritten**. *Historia numquam rescribitur; sutura
tantum additur.* Source works remain witnesses under their own burdens — the kernel
reference is the collapse-kernel spine, not the field constitution; the papers are
witnesses, not the kernel; a closure is ingress, not law. The corpus is navigable because
these roles remain distinct.

> Reditus is the object. Structura Reditus is the field. Generative Collapse Dynamics is
> the foundational theory. The tiers govern authority. UMCP measures, RCFT explores, ULRC
> preserves language. Tier-2 closures translate domains without redefining the kernel.

*Auctor Reditus — is qui reditum generat et structuram per reditum definit.*

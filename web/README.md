# GCD Kernel — Web Site

Interactive web reference for the Generative Collapse Dynamics (GCD) framework.
Live at **[calebpruett927.github.io/GENERATIVE-COLLAPSE-DYNAMICS](https://calebpruett927.github.io/GENERATIVE-COLLAPSE-DYNAMICS/)**.

## Stack

- **[Astro 5](https://astro.build/)** — static site generator
- **[Tailwind CSS 3.4](https://tailwindcss.com/)** — utility-first CSS with custom `kernel` palette
- **TypeScript** — kernel computation (direct port of `src/umcp_c/src/kernel.c`)
- **Canvas 2D API** — all charts and visualizations (no external chart libraries)
- **[Vitest](https://vitest.dev/)** — test framework (48 kernel tests)

No React, Vue, or other UI framework. No server-side computation — everything runs client-side.

## Quick Start

```bash
cd web
npm install
npm run dev        # Local dev server at http://localhost:4321
npm run build      # Production build → dist/
npm run test       # Run 48 kernel tests
```

## Scripts

| Command | Purpose |
|---------|---------|
| `npm run dev` | Start Astro dev server with hot reload |
| `npm run build` | Production build to `dist/` |
| `npm run preview` | Preview the production build locally |
| `npm run test` | Run Vitest kernel test suite (48 tests) |
| `npm run test:watch` | Run tests in watch mode |
| `npm run generate` | Generate domain content from HCG builder |
| `npm run generate:domain` | Generate a single domain |
| `npm run full-build` | Generate all domains + build |

## Architecture

```
web/
├── public/            Static assets (favicon, robots.txt, sitemap.xml)
├── src/
│   ├── components/    9 interactive Astro components (calculators, charts)
│   ├── content/       Generated domain content (23 domains via HCG)
│   ├── layouts/       5 layouts (Base, Domain, Index, Subpage, Casepack)
│   ├── lib/
│   │   ├── kernel.ts    Tier-1 kernel: F, ω, S, C, κ, IC computation
│   │   ├── constants.ts Frozen contract parameters + Rosetta lenses
│   │   └── chart.ts     HiDPI canvas chart utilities
│   └── pages/         21 static pages + dynamic domain routes
├── tests/
│   └── kernel.test.ts 48 tests: identities, regime, seam, edge cases
├── astro.config.mjs   Site config (GitHub Pages base path)
├── tailwind.config.mjs Custom kernel palette + regime colors
├── vitest.config.ts   Test configuration
└── package.json
```

## Pages (143 total)

### Static Pages (21)

| Page | Type | Description |
|------|------|-------------|
| **index** | Navigation | Domain network overview, tools grid |
| **calculator** | Tool | 5-mode Structure Explorer (compute, sweep, phase, identity, seam) |
| **regime** | Tool | 2D regime phase diagram with crosshair query |
| **diagnostics** | Tool | τ_R* thermodynamic diagnostic heatmap |
| **precision** | Tool | 15-decimal identity verification sweep |
| **formulas** | Tool | Expression evaluator with live variable sweeps |
| **seam-budget** | Tool | Γ(ω), D_C, Δκ budget breakdown with presets |
| **identities** | Reference | 44 structural identities with filters and search |
| **reference** | Reference | Cross-reference index (symbols, frozen params, regime, lenses) |
| **scale-ladder** | Reference | Cross-scale coherence: quark → hadron → atom |
| **geometry** | Corpus | Flat manifold, 44 identities, rank theorem, composition algebra |
| **grammar** | Corpus | Five-word vocabulary, Latin type system, discourse spine |
| **epistemology** | Corpus | Episteme of Return — observation cost (T9), reproducibility, cognitive equalizer |
| **philosophy** | Corpus | Camus/Nietzsche/Jung/Sartre convergences with kernel computation |
| **rosetta** | Reference | Six-lens Rosetta translator with live scenarios |
| **ledger** | Reference | 20-domain overview with kernel metrics |
| **orientation** | Reference | 10-section structural re-derivation |
| **about** | Info | What is GCD, the kernel, the spine, tier system |
| **papers** | Info | Published papers catalog |
| **404** | Error | Not found page |

### Dynamic Pages (~122)

Each of the 21 closure domains generates:
- `/domain/` — Domain index with regime badge, kernel panel, related graph
- `/domain/theorems/` — Domain theorems
- `/domain/entities/` — Domain entities
- `/domain/contract/` — Domain contract
- `/domain/casepacks/<slug>/` — Individual casepacks

## TypeScript Kernel

The kernel at `src/lib/kernel.ts` is a faithful port of the C99 implementation. It exports:

- `computeKernel(c, w?)` — Six Tier-1 invariants from trace vector
- `classifyRegime(result)` — Four-gate regime classification
- `computeSeamBudget(ω, C, R, τ_R)` — Seam budget with Γ(ω) cost
- `verifyIdentities(result)` — Three algebraic identity checks
- `computeTauRStar(ω, C)` — Thermodynamic return diagnostic
- `gammaOmega(ω)` — Drift cost Γ(ω) = ω³/(1−ω+ε)
- `sweepChannel()`, `sweepHomogeneous()`, `heatmap2D()` — Analysis functions
- `PRESETS` — Six canonical trace presets (perfect, stable, watch, collapse, slaughter, neutron)

All frozen parameters (`EPSILON`, `P_EXPONENT`, `ALPHA`, `TOL_SEAM`) come from `constants.ts`, mirroring `src/umcp/frozen_contract.py`.

## Testing

```bash
npm run test        # 48 tests, ~700ms
```

Tests cover:
- **Duality identity** (F + ω = 1) — exact across 10K random traces
- **Integrity bound** (IC ≤ F) — verified across 10K random traces
- **Log-integrity relation** (IC = exp(κ)) — verified across 10K random traces
- **Geometric slaughter** — dead channel kills IC while F stays healthy
- **Bernoulli field entropy** — boundary, symmetry, equator convergence
- **Regime classification** — all three regimes + Critical overlay
- **Seam budget** — ∞_rec handling, cost functions, residual verification
- **Edge cases** — empty trace, single channel, guard-band clamping

## Adding a New Domain

1. Add the closure in `closures/<domain>/`
2. Run `npm run generate` (HCG builder creates `src/content/<domain>/index.md`)
3. The dynamic route `[domain]/index.astro` picks it up automatically
4. Run `npm run build` to verify

## Custom Theme

The `kernel` color palette in `tailwind.config.mjs`:

| Token | Hex | Usage |
|-------|-----|-------|
| `kernel-950` | `#020617` | Page background |
| `kernel-100` | `#f1f5f9` | Primary text |
| `kernel-400` | `#94a3b8` | Nav text |
| `kernel-700` | `#334155` | Borders, separators |
| `stable` | `#22c55e` | Stable regime (green) |
| `watch` | `#f59e0b` | Watch regime (amber) |
| `collapse` | `#ef4444` | Collapse regime (red) |
| `critical` | `#991b1b` | Critical overlay (deep red) |

## Deployment

Deployed to GitHub Pages via the repository. The `astro.config.mjs` sets:

```js
site: 'https://calebpruett927.github.io'
base: '/GENERATIVE-COLLAPSE-DYNAMICS/'
```

Build output goes to `dist/` with directory format (`/page/index.html`).

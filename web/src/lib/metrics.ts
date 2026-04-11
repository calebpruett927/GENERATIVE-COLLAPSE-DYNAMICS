/**
 * Public Metrics — Single Source of Truth
 *
 * Every public-facing number displayed on the web layer MUST be imported
 * from this file. Hard-coding metrics in page templates is a conformance
 * violation — drift between pages is how overclaims happen.
 *
 * To update a metric: change it HERE, then run `npm test` to verify
 * that the new value matches the canonical Python source.
 */

/* ─── Repository-Scale Metrics ──────────────────────────────────── */

/** Total collected pytest items (pytest --collect-only | grep "::" | wc -l). */
export const TEST_COUNT = '20,221';
export const TEST_COUNT_RAW = 20_221;

/** Number of closure domains in closures/. */
export const DOMAIN_COUNT = '23';
export const DOMAIN_COUNT_RAW = 23;

/** Structural identities derived from Axiom-0. */
export const IDENTITY_COUNT = '44';
export const IDENTITY_COUNT_RAW = 44;

/** Proven lemmas (OPT-* tagged in kernel code). */
export const LEMMA_COUNT = '47';
export const LEMMA_COUNT_RAW = 47;

/** Closure modules across all domains. */
export const CLOSURE_COUNT = '245';
export const CLOSURE_COUNT_RAW = 245;

/** Proven theorems across all domain closures. */
export const THEOREM_COUNT = '746';
export const THEOREM_COUNT_RAW = 746;

/** Implementation languages (C99, C++17, Python). */
export const LANGUAGE_COUNT = '3';
export const LANGUAGE_COUNT_RAW = 3;

/** Test files (numbered test_000 through test_338). */
export const TEST_FILE_COUNT = '231';
export const TEST_FILE_COUNT_RAW = 231;

/** Base test functions before parametrization. */
export const BASE_TEST_COUNT = '~800';

/** Average parametrization expansion factor. */
export const PARAMETRIZE_FACTOR = '~18×';

/** C99 orchestration core lines. */
export const C_LINES = '~1,900';

/** C kernel test assertions. */
export const C_ASSERTIONS = '326';
export const C_ASSERTIONS_RAW = 326;

/** C++ Catch2 test assertions. */
export const CPP_ASSERTIONS = '434';
export const CPP_ASSERTIONS_RAW = 434;

/** C++ kernel speedup factor. */
export const CPP_SPEEDUP = '~50×';

/* ─── Fisher Space Partition (Orientation §7) ───────────────────── */

export const REGIME_STABLE_PCT = '12.5%';
export const REGIME_WATCH_PCT = '24.4%';
export const REGIME_COLLAPSE_PCT = '63.1%';

/* ─── Convenience "At a Glance" array for page templates ────────── */

export interface GlanceMetric {
  n: string;
  label: string;
  icon: string;
}

export const AT_A_GLANCE: GlanceMetric[] = [
  { n: TEST_COUNT, label: 'Tests', icon: '✓' },
  { n: DOMAIN_COUNT, label: 'Domains', icon: '◈' },
  { n: IDENTITY_COUNT, label: 'Identities', icon: '≡' },
  { n: LEMMA_COUNT, label: 'Lemmas', icon: '∴' },
  { n: CLOSURE_COUNT, label: 'Closures', icon: '⊕' },
  { n: LANGUAGE_COUNT, label: 'Languages', icon: '⟨⟩' },
];

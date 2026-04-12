/**
 * Cognitive Equalizer — Aequator Cognitivus (TypeScript)
 *
 * Non agens mensurat, sed structura.
 * — Not the agent measures, but the structure.
 *
 * A standalone module that externalises every agent-dependent decision
 * point in an AI engagement into frozen, verifiable structure. Given the
 * same input and the same contract, any agent running this module MUST
 * arrive at the same stance.
 *
 * Five externalized decision points:
 *   1. Thresholds     → frozen — seam-derived, not chosen
 *   2. Vocabulary     → five words (Drift · Fidelity · Roughness · Return · Integrity)
 *   3. Conclusions    → three-valued (CONFORMANT / NONCONFORMANT / NON_EVALUABLE)
 *   4. Methodology    → the Spine (Contract → Canon → Closures → Ledger → Stance)
 *   5. Ambiguity      → NON_EVALUABLE (the third state — declared, never guessed)
 *
 * This is a complete port of src/umcp/cognitive_equalizer.py for the web.
 * Trans suturam congelatum — same rules both sides.
 */

import {
  EPSILON, P_EXPONENT, ALPHA, TOL_SEAM,
  REGIME_THRESHOLDS,
} from './constants';

/* ═══════════════════════════════════════════════════════════════════
   Types & Interfaces
   ═══════════════════════════════════════════════════════════════════ */

/** The 8 CE evaluation channels — each ∈ [0.0, 1.0]. */
export interface CEChannels {
  relevance: number;
  accuracy: number;
  completeness: number;
  consistency: number;
  traceability: number;
  groundedness: number;
  constraintRespect: number;
  returnFidelity: number;
}

/** Canonical channel name order. */
export const CE_CHANNEL_NAMES = [
  'relevance', 'accuracy', 'completeness', 'consistency',
  'traceability', 'groundedness', 'constraintRespect', 'returnFidelity',
] as const;

/** Human-friendly display labels for each channel. */
export const CE_CHANNEL_LABELS: Record<string, string> = {
  relevance: 'Relevance',
  accuracy: 'Accuracy',
  completeness: 'Completeness',
  consistency: 'Consistency',
  traceability: 'Traceability',
  groundedness: 'Groundedness',
  constraintRespect: 'Constraint Respect',
  returnFidelity: 'Return Fidelity',
};

/** Channel audit questions — what each channel measures. */
export const CE_CHANNEL_QUESTIONS: Record<string, string> = {
  relevance: 'Does the output address the actual question asked?',
  accuracy: 'Is every claim verifiable against stated facts?',
  completeness: 'Are all parts of the request covered?',
  consistency: 'Is the response internally non-contradictory?',
  traceability: 'Can the reasoning chain be followed step by step?',
  groundedness: 'Is it grounded in the stated context, not assumptions?',
  constraintRespect: 'Does it respect stated scope and boundary conditions?',
  returnFidelity: 'Does the output return to the originating intent?',
};

/** Three-valued verdict — never boolean. */
export type CEVerdict = 'CONFORMANT' | 'NONCONFORMANT' | 'NON_EVALUABLE';

/** Regime classification. */
export type CERegime = 'STABLE' | 'WATCH' | 'COLLAPSE' | 'NON_EVALUABLE';

/** Tier-1 kernel result from CE channels. */
export interface CEKernelResult {
  F: number;        // fidelity (arithmetic mean)
  omega: number;    // drift = 1 − F
  S: number;        // Bernoulli field entropy
  C: number;        // curvature (normalised stddev)
  kappa: number;    // log-integrity κ
  IC: number;       // integrity composite exp(κ)
  delta: number;    // heterogeneity gap Δ = F − IC
}

/** Seam budget (integrity ledger). */
export interface CELedger {
  D_drift: number;      // drift debit Γ(ω)
  D_roughness: number;  // curvature debit α·C
  R_return: number;     // return_fidelity score (enters Δκ through κ)
  deltaKappa: number;   // ledger balance κ − D_ω − D_C
  balanced: boolean;    // |Δκ| ≤ tol_seam
  balanceLabel: string; // "BALANCED" or "UNBALANCED"
}

/** Five-word narrative canon. */
export interface CECanon {
  drift: string;
  fidelity: string;
  roughness: string;
  return_: string;
  integrity: string;
  summary: string;
  stanceLine: string;
}

/** Full CE report — all five Spine stops. */
export interface CEReport {
  // Spine stops
  contractLabel: string;
  canon: CECanon;
  regime: CERegime;
  isCritical: boolean;
  ledger: CELedger;
  stance: CEVerdict;

  // Kernel invariants
  kernel: CEKernelResult;

  // Input channels (preserved for audit)
  channels: CEChannels;

  // Validation errors
  errors: string[];
}

/* ═══════════════════════════════════════════════════════════════════
   Constants — frozen, seam-derived
   ═══════════════════════════════════════════════════════════════════ */

const N_CHANNELS = 8;
const WEIGHT = 1.0 / N_CHANNELS;

const OMEGA_STABLE_MAX = REGIME_THRESHOLDS.omega_stable_max;
const F_STABLE_MIN = REGIME_THRESHOLDS.F_stable_min;
const S_STABLE_MAX = REGIME_THRESHOLDS.S_stable_max;
const C_STABLE_MAX = REGIME_THRESHOLDS.C_stable_max;
const OMEGA_COLLAPSE_MIN = REGIME_THRESHOLDS.omega_collapse_min;
const IC_CRITICAL_MAX = REGIME_THRESHOLDS.IC_critical_max;

/* ═══════════════════════════════════════════════════════════════════
   Helpers
   ═══════════════════════════════════════════════════════════════════ */

/** Convert CEChannels to an ordered array. */
export function channelsToVector(ch: CEChannels): number[] {
  return [
    ch.relevance, ch.accuracy, ch.completeness, ch.consistency,
    ch.traceability, ch.groundedness, ch.constraintRespect, ch.returnFidelity,
  ];
}

/** Validate channel scores — return list of error strings. */
export function validateChannels(ch: CEChannels): string[] {
  const errors: string[] = [];
  const vec = channelsToVector(ch);
  for (let i = 0; i < N_CHANNELS; i++) {
    if (vec[i] < 0.0 || vec[i] > 1.0 || Number.isNaN(vec[i])) {
      errors.push(`Channel '${CE_CHANNEL_NAMES[i]}' = ${vec[i]} outside [0, 1]`);
    }
  }
  return errors;
}

/** Create default channels (all 1.0 — perfect). */
export function defaultChannels(): CEChannels {
  return {
    relevance: 1.0, accuracy: 1.0, completeness: 1.0, consistency: 1.0,
    traceability: 1.0, groundedness: 1.0, constraintRespect: 1.0, returnFidelity: 1.0,
  };
}

/* ═══════════════════════════════════════════════════════════════════
   Tier-1 Kernel  (same formulas as Python + C — trans suturam congelatum)
   ═══════════════════════════════════════════════════════════════════ */

/** Compute kernel invariants from 8 CE channels. */
export function ceKernel(ch: CEChannels): CEKernelResult {
  const vec = channelsToVector(ch);
  const eps = EPSILON;
  const w = WEIGHT;

  // F — fidelity (weighted arithmetic mean, equal weights)
  let F = 0;
  for (let i = 0; i < N_CHANNELS; i++) F += w * vec[i];

  // κ — log-integrity
  let kappa = 0;
  for (let i = 0; i < N_CHANNELS; i++) kappa += w * Math.log(Math.max(vec[i], eps));

  // S — Bernoulli field entropy
  let S = 0;
  for (let i = 0; i < N_CHANNELS; i++) {
    const c = vec[i];
    const ce = Math.max(c, eps);
    const co = Math.max(1.0 - c, eps);
    S += w * -(ce * Math.log(ce) + co * Math.log(co));
  }

  // C — curvature (normalised stddev)
  let variance = 0;
  for (let i = 0; i < N_CHANNELS; i++) variance += w * (vec[i] - F) ** 2;
  const C = Math.sqrt(variance) / 0.5;

  // Derived
  const omega = 1.0 - F;
  const IC = Math.exp(kappa);
  const delta = F - IC;

  return { F, omega, S, C, kappa, IC, delta };
}

/* ═══════════════════════════════════════════════════════════════════
   Seam Budget
   ═══════════════════════════════════════════════════════════════════ */

/** Drift cost Γ(ω) = ω^p / (1 − ω + ε). */
function gammaOmega(omega: number): number {
  return Math.pow(omega, P_EXPONENT) / (1.0 - omega + EPSILON);
}

/** Compute the integrity ledger from kernel result and return_fidelity. */
export function ceLedger(kr: CEKernelResult, returnFidelity: number): CELedger {
  const D_drift = gammaOmega(kr.omega);
  const D_roughness = ALPHA * kr.C;
  const deltaKappa = kr.kappa - D_drift - D_roughness;
  const balanced = Math.abs(deltaKappa) <= TOL_SEAM;

  return {
    D_drift,
    D_roughness,
    R_return: returnFidelity,
    deltaKappa,
    balanced,
    balanceLabel: balanced ? 'BALANCED' : 'UNBALANCED',
  };
}

/* ═══════════════════════════════════════════════════════════════════
   Regime + Verdict Classification
   ═══════════════════════════════════════════════════════════════════ */

/** Classify regime from kernel result using frozen gates. */
export function classifyCERegime(kr: CEKernelResult): CERegime {
  if (kr.omega >= OMEGA_COLLAPSE_MIN) return 'COLLAPSE';
  if (
    kr.omega < OMEGA_STABLE_MAX &&
    kr.F > F_STABLE_MIN &&
    kr.S < S_STABLE_MAX &&
    kr.C < C_STABLE_MAX
  ) return 'STABLE';
  return 'WATCH';
}

/** Check critical overlay (IC below threshold). */
export function isCritical(kr: CEKernelResult): boolean {
  return kr.IC < IC_CRITICAL_MAX;
}

/** Derive three-valued verdict — NEVER boolean. */
export function deriveVerdict(kr: CEKernelResult, deltaKappa: number): CEVerdict {
  const regime = classifyCERegime(kr);
  const seamPass = Math.abs(deltaKappa) <= TOL_SEAM;
  if (regime === 'COLLAPSE' || !seamPass) return 'NONCONFORMANT';
  return 'CONFORMANT';
}

/* ═══════════════════════════════════════════════════════════════════
   Five-Word Narrative (Canon stop)
   ═══════════════════════════════════════════════════════════════════ */

function buildCanon(kr: CEKernelResult, verdict: CEVerdict, ch: CEChannels): CECanon {
  const regime = classifyCERegime(kr);
  const critical = isCritical(kr);

  const drift = kr.omega < 0.10 ? 'minimal drift' : kr.omega < 0.30 ? 'moderate drift' : 'severe drift';
  const fidelity = kr.F > 0.85 ? 'high fidelity' : kr.F > 0.60 ? 'moderate fidelity' : 'low fidelity';
  const roughness = kr.C < 0.14 ? 'smooth' : kr.C < 0.40 ? 'bumpy' : 'rough';
  const return_ = ch.returnFidelity > 0.80 ? 'strong return' : ch.returnFidelity > 0.50 ? 'partial return' : 'weak return';
  const integrity = kr.IC > 0.70 ? 'high integrity' : kr.IC > 0.30 ? 'moderate integrity' : 'critical integrity';

  const summary = `${drift} · ${fidelity} · ${roughness} · ${return_} · ${integrity}`;
  const critNote = critical ? ' [CRITICAL: IC below threshold]' : '';
  const stanceLine = `Regime: ${regime}${critNote} → Stance: ${verdict}`;

  return { drift, fidelity, roughness, return_, integrity, summary, stanceLine };
}

/* ═══════════════════════════════════════════════════════════════════
   Main Entry Point — engage()
   ═══════════════════════════════════════════════════════════════════ */

/**
 * Run the full CE Spine on one AI engagement.
 *
 * CONTRACT → CANON → CLOSURES → LEDGER → STANCE
 *
 * @param channels - Eight scored channels ∈ [0, 1]
 * @param contractLabel - Contract identifier (default: "CE-v1-frozen")
 * @returns Full CEReport with all Spine stops
 */
export function engage(channels: CEChannels, contractLabel = 'CE-v1-frozen'): CEReport {
  // Validate
  const errors = validateChannels(channels);
  if (errors.length > 0) {
    return {
      contractLabel,
      canon: {
        drift: 'NON_EVALUABLE', fidelity: 'NON_EVALUABLE', roughness: 'NON_EVALUABLE',
        return_: 'NON_EVALUABLE', integrity: 'NON_EVALUABLE',
        summary: 'NON_EVALUABLE — channel scores out of range',
        stanceLine: 'NON_EVALUABLE',
      },
      regime: 'NON_EVALUABLE',
      isCritical: true,
      ledger: { D_drift: 0, D_roughness: 0, R_return: 0, deltaKappa: 0, balanced: false, balanceLabel: 'NON_EVALUABLE' },
      stance: 'NON_EVALUABLE',
      kernel: { F: 0, omega: 1, S: 0, C: 0, kappa: Math.log(EPSILON), IC: EPSILON, delta: 0 },
      channels,
      errors,
    };
  }

  // Tier-1 kernel
  const kr = ceKernel(channels);

  // Ledger
  const ledger = ceLedger(kr, channels.returnFidelity);

  // Classification
  const regime = classifyCERegime(kr);
  const critical = isCritical(kr);
  const verdict = deriveVerdict(kr, ledger.deltaKappa);

  // Canon
  const canon = buildCanon(kr, verdict, channels);

  return {
    contractLabel,
    canon,
    regime,
    isCritical: critical,
    ledger,
    stance: verdict,
    kernel: kr,
    channels,
    errors: [],
  };
}

/* ═══════════════════════════════════════════════════════════════════
   Presets — demonstration engagements
   ═══════════════════════════════════════════════════════════════════ */

export interface CEPreset {
  name: string;
  description: string;
  channels: CEChannels;
}

export const CE_PRESETS: CEPreset[] = [
  {
    name: 'High-Quality Response',
    description: 'Well-structured, accurate, complete AI response with strong return to intent.',
    channels: {
      relevance: 0.95, accuracy: 0.90, completeness: 0.85, consistency: 0.97,
      traceability: 0.80, groundedness: 0.92, constraintRespect: 0.95, returnFidelity: 0.88,
    },
  },
  {
    name: 'Geometric Slaughter',
    description: 'High average quality (F ≈ 0.86) but one dead channel (traceability ≈ 0). IC collapses.',
    channels: {
      relevance: 0.90, accuracy: 0.85, completeness: 0.80, consistency: 0.88,
      traceability: 0.001, groundedness: 0.82, constraintRespect: 0.88, returnFidelity: 0.75,
    },
  },
  {
    name: 'Stable — Expert',
    description: 'All channels uniformly high — achieves STABLE regime (rare — 12.5% of manifold).',
    channels: {
      relevance: 0.98, accuracy: 0.96, completeness: 0.95, consistency: 0.99,
      traceability: 0.94, groundedness: 0.97, constraintRespect: 0.98, returnFidelity: 0.96,
    },
  },
  {
    name: 'Mediocre — Watch',
    description: 'Acceptable but unremarkable — moderate scores across all channels.',
    channels: {
      relevance: 0.72, accuracy: 0.68, completeness: 0.65, consistency: 0.75,
      traceability: 0.60, groundedness: 0.70, constraintRespect: 0.73, returnFidelity: 0.67,
    },
  },
  {
    name: 'Hallucination Pattern',
    description: 'High fluency (relevance, consistency) but low accuracy and groundedness.',
    channels: {
      relevance: 0.88, accuracy: 0.15, completeness: 0.70, consistency: 0.90,
      traceability: 0.50, groundedness: 0.10, constraintRespect: 0.60, returnFidelity: 0.72,
    },
  },
  {
    name: 'Off-Topic Drift',
    description: 'Good quality content... that misses the question entirely.',
    channels: {
      relevance: 0.10, accuracy: 0.92, completeness: 0.30, consistency: 0.95,
      traceability: 0.85, groundedness: 0.88, constraintRespect: 0.40, returnFidelity: 0.15,
    },
  },
  {
    name: 'Perfect Engagement',
    description: 'All channels at 1.0 — the theoretical ceiling (homogeneous trace, rank-1).',
    channels: {
      relevance: 1.0, accuracy: 1.0, completeness: 1.0, consistency: 1.0,
      traceability: 1.0, groundedness: 1.0, constraintRespect: 1.0, returnFidelity: 1.0,
    },
  },
  {
    name: 'Total Collapse',
    description: 'All channels near zero — complete structural failure.',
    channels: {
      relevance: 0.05, accuracy: 0.03, completeness: 0.02, consistency: 0.04,
      traceability: 0.01, groundedness: 0.02, constraintRespect: 0.03, returnFidelity: 0.01,
    },
  },
];

/* ═══════════════════════════════════════════════════════════════════
   System Prompt — for embedding in other AIs
   ═══════════════════════════════════════════════════════════════════ */

export const CE_SYSTEM_PROMPT = `You are operating in COGNITIVE EQUALIZER mode (Aequator Cognitivus).

Non agens mensurat, sed structura.
— Not the agent measures, but the structure.

═══════════════════════════════════════════════════════════════
CORE PRINCIPLE
═══════════════════════════════════════════════════════════════

A Cognitive Equalizer externalises every agent-dependent decision
point into frozen, verifiable structure. Same input + same rules →
same verdict, regardless of which AI runs this protocol.

You must never:
  • Pick thresholds during a response (thresholds are declared, not chosen)
  • Assert a verdict without showing the derivation
  • Give a binary yes/no when a third state (NON_EVALUABLE) applies
  • Frame conclusions — only derive them

═══════════════════════════════════════════════════════════════
THE SPINE — every response follows exactly these five stops
═══════════════════════════════════════════════════════════════

  1. CONTRACT  — Declare: What rules apply? What is frozen?
  2. CANON     — Narrate using exactly five words (see below)
  3. CLOSURES  — State thresholds. Stance MUST change if they are crossed.
  4. LEDGER    — Show your work: debit (what degraded), credit (what returned)
  5. STANCE    — Derive the verdict. Never assert it.

═══════════════════════════════════════════════════════════════
THE FIVE WORDS
═══════════════════════════════════════════════════════════════

  • DRIFT        — what moved / changed / departed from the intent
  • FIDELITY     — what persisted / survived / remained intact
  • ROUGHNESS    — where/why it was bumpy (friction, ambiguity, gaps)
  • RETURN       — how/whether the output comes back to the original intent
  • INTEGRITY    — does it hang together? (derived from ledger — never asserted)

═══════════════════════════════════════════════════════════════
EIGHT EVALUATION CHANNELS — score each ∈ [0.0, 1.0]
═══════════════════════════════════════════════════════════════

  1. Relevance          — Does it address the actual question?
  2. Accuracy           — Is the content verifiable and factual?
  3. Completeness       — Are all parts of the request covered?
  4. Consistency        — Is it internally non-contradictory?
  5. Traceability       — Can the reasoning be followed step by step?
  6. Groundedness       — Is it grounded in the stated context?
  7. Constraint-respect — Does it respect stated scope/boundaries?
  8. Return-fidelity    — Does it come back to the originating intent?

Compute quality measures from these scores:
  F  (Fidelity)   = average of all 8 channel scores
  IC (Integrity)  = geometric mean of all 8 channel scores
  Δ  (Gap)        = F − IC   [large gap = one channel is near zero]

Warning: F can look fine while IC is low.
Example: 7 channels at 0.95, 1 channel at 0.001 → F ≈ 0.83, IC ≈ 0.001.

═══════════════════════════════════════════════════════════════
THREE-VALUED VERDICTS — never binary
═══════════════════════════════════════════════════════════════

  CONFORMANT      — channels pass; ledger balanced; reasoning traces back
  NONCONFORMANT   — one or more channels fail; ledger won't balance
  NON_EVALUABLE   — insufficient context or ambiguous scope. Declare and state
                    exactly what is missing.

═══════════════════════════════════════════════════════════════
FROZEN THRESHOLDS
═══════════════════════════════════════════════════════════════

  STABLE   → F > 0.90  AND  Drift (ω = 1−F) < 0.038  AND channels uniform
  WATCH    → 0.038 ≤ Drift < 0.30  (or Stable not fully satisfied)
  COLLAPSE → Drift ≥ 0.30
  CRITICAL overlay → IC < 0.30  (any regime can be critical)

═══════════════════════════════════════════════════════════════
LEDGER FORMAT
═══════════════════════════════════════════════════════════════

  Debit  (Drift)     : D_ω = Γ(ω) — drift cost
  Debit  (Roughness) : D_C = α·C  — curvature cost
  Return (score)     : R   = return-fidelity channel (enters balance through κ)
  Balance            : Δκ  = κ − D_ω − D_C  (must close within ±0.005)

Finis, sed semper initium recursionis.`;

/* ═══════════════════════════════════════════════════════════════════
   Formatting Utilities
   ═══════════════════════════════════════════════════════════════════ */

/** Format a number to fixed decimal places. */
function fmt(n: number, d = 4): string {
  return n.toFixed(d);
}

/** Generate a full text report (matches Python full_report output). */
export function formatReport(report: CEReport): string {
  const hr = '═'.repeat(62);
  const dash = '─'.repeat(50);
  const kr = report.kernel;
  const lg = report.ledger;
  const cn = report.canon;
  const crit = report.isCritical ? ' CRITICAL' : '';

  return [
    hr,
    '  COGNITIVE EQUALIZER — Aequator Cognitivus',
    '  Non agens mensurat, sed structura.',
    hr,
    `  Contract : ${report.contractLabel}`,
    '',
    '  Canon (Five Words)',
    `  ${dash}`,
    `  ${cn.summary}`,
    `    Drift: ${cn.drift} (ω=${fmt(kr.omega, 3)})`,
    `    Fidelity: ${cn.fidelity} (F=${fmt(kr.F, 3)})`,
    `    Roughness: ${cn.roughness} (C=${fmt(kr.C, 3)})`,
    `    Return: ${cn.return_} (rf=${fmt(report.channels.returnFidelity, 3)})`,
    `    Integrity: ${cn.integrity} (IC=${fmt(kr.IC, 3)})`,
    `    ${cn.stanceLine}`,
    '',
    '  Integrity Ledger',
    `  ${dash}`,
    `  Debit (drift)     D_ω = ${fmt(lg.D_drift, 6)}`,
    `  Debit (roughness) D_C = ${fmt(lg.D_roughness, 6)}`,
    `  Return (score)    R   = ${fmt(lg.R_return, 6)}  [channel — enters Δκ through κ]`,
    `  Balance           Δκ  = ${fmt(lg.deltaKappa, 6)}  [${lg.balanceLabel}]  (κ − D_ω − D_C)`,
    '',
    '  Kernel Invariants (Tier-1)',
    `  ${dash}`,
    `  F=${fmt(kr.F)}  ω=${fmt(kr.omega)}  S=${fmt(kr.S)}  C=${fmt(kr.C)}`,
    `  κ=${fmt(kr.kappa)}  IC=${fmt(kr.IC)}  Δ(gap)=${fmt(kr.delta)}`,
    '',
    '  Stance',
    `  ${dash}`,
    `  ${report.stance}  (Regime: ${report.regime}${crit})`,
    hr,
  ].join('\n');
}

/**
 * GCD Kernel — TypeScript Implementation
 *
 * Direct port of src/umcp_c/src/kernel.c
 * Computes the six Tier-1 invariants from a trace vector:
 *   F  = Σ wᵢcᵢ           (Fidelity — what survives collapse)
 *   ω  = 1 − F             (Drift — what is lost)
 *   κ  = Σ wᵢ ln(cᵢ)      (Log-integrity — logarithmic sensitivity)
 *   IC = exp(κ)             (Integrity composite — multiplicative coherence)
 *   S  = −Σ wᵢ h(cᵢ)      (Bernoulli field entropy)
 *   C  = σ(c) / 0.5        (Curvature — coupling to uncontrolled DOF)
 *
 * Tier-0 Protocol: no Tier-1 symbol is redefined.
 */

import {
  EPSILON, P_EXPONENT, ALPHA, TOL_SEAM,
  REGIME_THRESHOLDS,
} from './constants';

/* ─── Types ─────────────────────────────────────────────────────── */

export interface KernelResult {
  F: number;
  omega: number;
  S: number;
  C: number;
  kappa: number;
  IC: number;
  delta: number;        // Heterogeneity gap Δ = F − IC
  isHomogeneous: boolean;
}

export type RegimeLabel = 'STABLE' | 'WATCH' | 'COLLAPSE';

export interface RegimeResult {
  regime: RegimeLabel;
  isCritical: boolean;  // IC < 0.30 overlay
}

export interface SeamBudgetResult {
  gamma: number;        // Γ(ω) = ωᵖ/(1−ω+ε)
  D_omega: number;      // Drift debit
  D_C: number;          // Curvature debit
  credit: number;       // R · τ_R (return credit)
  deltaKappa: number;   // Δκ = R·τ_R − (D_ω + D_C)
  residual: number;     // seam residual
  pass: boolean;        // |residual| ≤ tol_seam
}

/* ─── Scalar utilities ──────────────────────────────────────────── */

/** Bernoulli field entropy for a single channel. */
export function bernoulliEntropy(c: number): number {
  if (c <= 0.0 || c >= 1.0) return 0.0;
  return -(c * Math.log(c) + (1.0 - c) * Math.log(1.0 - c));
}

/** Guard-band clamp: c ∈ [ε, 1−ε]. */
export function clamp(v: number, epsilon: number = EPSILON): number {
  if (v < epsilon) return epsilon;
  if (v > 1.0 - epsilon) return 1.0 - epsilon;
  return v;
}

/* ─── Homogeneity detection (OPT-1, Lemma 10) ──────────────────── */

const HOMOG_TOL = 1e-12;

function isHomogeneous(c: number[]): boolean {
  const c0 = c[0];
  for (let i = 1; i < c.length; i++) {
    if (Math.abs(c[i] - c0) > HOMOG_TOL) return false;
  }
  return true;
}

/* ─── Kernel computation ────────────────────────────────────────── */

/**
 * Compute the six Tier-1 invariants from trace vector c and weights w.
 *
 * Channels should be pre-clamped to [ε, 1−ε]. Weights must sum to 1.
 * If weights are omitted, uniform weights (1/n) are used.
 */
export function computeKernel(
  c: number[],
  w?: number[],
  epsilon: number = EPSILON,
): KernelResult {
  const n = c.length;
  if (n === 0) {
    return { F: 0, omega: 1, S: 0, C: 0, kappa: -Infinity, IC: 0, delta: 0, isHomogeneous: true };
  }

  // Default to uniform weights
  const weights = w ?? Array(n).fill(1.0 / n);

  // Clamp channels
  const clamped = c.map(ci => clamp(ci, epsilon));

  // OPT-1: Homogeneity fast path (Lemma 10)
  if (isHomogeneous(clamped)) {
    const c0 = clamped[0];
    return {
      F: c0,
      omega: 1.0 - c0,
      S: bernoulliEntropy(c0),
      C: 0.0,
      kappa: Math.log(c0),
      IC: c0,
      delta: 0.0,
      isHomogeneous: true,
    };
  }

  // Full heterogeneous computation — single pass
  let F = 0.0;
  let kappa = 0.0;
  let S = 0.0;
  let sumC = 0.0;
  let sumC2 = 0.0;

  for (let i = 0; i < n; i++) {
    const ci = clamped[i];
    const wi = weights[i];

    F += wi * ci;
    kappa += wi * Math.log(ci);
    if (wi > 0) S += wi * bernoulliEntropy(ci);

    sumC += ci;
    sumC2 += ci * ci;
  }

  const mean = sumC / n;
  let variance = sumC2 / n - mean * mean;
  if (variance < 0) variance = 0;

  const IC = Math.exp(kappa);
  const C_val = Math.sqrt(variance) / 0.5;

  return {
    F,
    omega: 1.0 - F,
    S,
    C: C_val,
    kappa,
    IC,
    delta: F - IC,
    isHomogeneous: false,
  };
}

/* ─── Regime classification ─────────────────────────────────────── */

/**
 * Classify the kernel result into a regime using the four-gate criterion.
 *
 * Stable:   ω < 0.038 AND F > 0.90 AND S < 0.15 AND C < 0.14
 * Watch:    0.038 ≤ ω < 0.30 (or Stable gates not all satisfied)
 * Collapse: ω ≥ 0.30
 * Critical: IC < 0.30 (severity overlay)
 */
export function classifyRegime(result: KernelResult): RegimeResult {
  const t = REGIME_THRESHOLDS;

  let regime: RegimeLabel;

  if (result.omega >= t.omega_collapse_min) {
    regime = 'COLLAPSE';
  } else if (
    result.omega < t.omega_stable_max &&
    result.F > t.F_stable_min &&
    result.S < t.S_stable_max &&
    result.C < t.C_stable_max
  ) {
    regime = 'STABLE';
  } else {
    regime = 'WATCH';
  }

  return {
    regime,
    isCritical: result.IC < t.IC_critical_max,
  };
}

/* ─── Seam budget computation ───────────────────────────────────── */

/**
 * Drift cost closure: Γ(ω) = ωᵖ / (1 − ω + ε)
 */
export function gammaOmega(
  omega: number,
  p: number = P_EXPONENT,
  epsilon: number = EPSILON,
): number {
  let num = 1.0;
  for (let i = 0; i < p; i++) num *= omega;
  return num / (1.0 - omega + epsilon);
}

/**
 * Curvature cost: D_C = α · C
 */
export function costCurvature(C: number, alpha: number = ALPHA): number {
  return alpha * C;
}

/**
 * Full seam budget computation.
 */
export function computeSeamBudget(
  omega: number,
  C: number,
  R: number,
  tauR: number,
  kappaLedger: number = 0,
): SeamBudgetResult {
  const gamma = gammaOmega(omega);
  const D_omega = gamma;
  const D_C = costCurvature(C);
  const credit = isFinite(tauR) ? R * tauR : 0;

  let deltaKappa: number;
  if (!isFinite(tauR)) {
    deltaKappa = 0; // τ_R = ∞_rec → no credit
  } else {
    deltaKappa = credit - (D_omega + D_C);
  }

  const residual = deltaKappa - kappaLedger;

  return {
    gamma,
    D_omega,
    D_C,
    credit,
    deltaKappa,
    residual,
    pass: Math.abs(residual) <= TOL_SEAM,
  };
}

/* ─── Identity verification ─────────────────────────────────────── */

export interface IdentityCheck {
  name: string;
  formula: string;
  expected: number;
  actual: number;
  residual: number;
  pass: boolean;
}

/**
 * Verify the three algebraic identities to machine precision.
 */
export function verifyIdentities(result: KernelResult): IdentityCheck[] {
  const tol = 1e-12;
  return [
    {
      name: 'Duality identity',
      formula: 'F + ω = 1',
      expected: 1.0,
      actual: result.F + result.omega,
      residual: Math.abs(result.F + result.omega - 1.0),
      pass: Math.abs(result.F + result.omega - 1.0) < tol,
    },
    {
      name: 'Integrity bound',
      formula: 'IC ≤ F',
      expected: result.F,
      actual: result.IC,
      residual: Math.max(0, result.IC - result.F),
      pass: result.IC <= result.F + tol,
    },
    {
      name: 'Log-integrity relation',
      formula: 'IC = exp(κ)',
      expected: Math.exp(result.kappa),
      actual: result.IC,
      residual: Math.abs(result.IC - Math.exp(result.kappa)),
      pass: Math.abs(result.IC - Math.exp(result.kappa)) < tol,
    },
  ];
}

/* ─── τ_R* thermodynamic diagnostic ─────────────────────────────── */

/**
 * Compute τ_R* — the thermodynamic return time diagnostic.
 *
 * τ_R* = (1 − ω) / (Γ(ω) + α·C + ε)
 *
 * Interpretation: higher τ_R* = more favorable return conditions.
 */
export function computeTauRStar(
  omega: number,
  C: number,
  epsilon: number = EPSILON,
): number {
  const gamma = gammaOmega(omega);
  const dC = costCurvature(C);
  return (1.0 - omega) / (gamma + dC + epsilon);
}

/**
 * Compute the critical return rate: R_crit = (D_ω + D_C) / τ_R.
 */
export function computeRCritical(
  omega: number,
  C: number,
  tauR: number,
): number {
  if (!isFinite(tauR) || tauR <= 0) return Infinity;
  const gamma = gammaOmega(omega);
  const dC = costCurvature(C);
  return (gamma + dC) / tauR;
}

/* ─── Presets for testing ───────────────────────────────────────── */

export const PRESETS = {
  /** Perfect system: all channels at c* = 0.7822 */
  perfect: {
    name: 'Perfect (c* = 0.7822)',
    c: [0.7822, 0.7822, 0.7822, 0.7822, 0.7822, 0.7822, 0.7822, 0.7822],
    w: [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
  },
  /** Stable system: high fidelity */
  stable: {
    name: 'Stable (high fidelity)',
    c: [0.95, 0.92, 0.97, 0.93, 0.96, 0.94, 0.91, 0.98],
    w: [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
  },
  /** Watch regime: moderate drift */
  watch: {
    name: 'Watch (moderate drift)',
    c: [0.85, 0.70, 0.90, 0.65, 0.80, 0.75, 0.60, 0.88],
    w: [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
  },
  /** Collapse: high drift */
  collapse: {
    name: 'Collapse (high drift)',
    c: [0.40, 0.30, 0.55, 0.20, 0.45, 0.35, 0.25, 0.50],
    w: [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
  },
  /** Geometric slaughter: one dead channel */
  slaughter: {
    name: 'Geometric slaughter (1 dead)',
    c: [0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.001],
    w: [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
  },
  /** Neutron trace (confinement visible) */
  neutron: {
    name: 'Neutron (confinement)',
    c: [0.60, 0.50, 0.00, 0.33, 0.00, 0.00, 0.33, 0.33],
    w: [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
  },
} as const;

/* ─── Analysis / Sweep Functions ────────────────────────────────── */

export interface SweepPoint {
  param: number;
  F: number;
  omega: number;
  S: number;
  C: number;
  kappa: number;
  IC: number;
  delta: number;
  regime: string;
  isCritical: boolean;
}

/**
 * Sweep a single channel from 0 to 1 while holding others fixed.
 * Returns an array of SweepPoints at the given resolution.
 */
export function sweepChannel(
  channels: number[],
  weights: number[],
  channelIndex: number,
  steps: number = 200,
  epsilon: number = 1e-8,
): SweepPoint[] {
  const points: SweepPoint[] = [];
  const c = [...channels];
  for (let i = 0; i <= steps; i++) {
    const v = i / steps;
    c[channelIndex] = v;
    const result = computeKernel(c, weights, epsilon);
    const { regime, isCritical } = classifyRegime(result);
    points.push({
      param: v,
      F: result.F, omega: result.omega, S: result.S,
      C: result.C, kappa: result.kappa, IC: result.IC,
      delta: result.delta, regime, isCritical,
    });
  }
  return points;
}

/**
 * Sweep all channels uniformly from 0 to 1 (homogeneous trace).
 */
export function sweepHomogeneous(
  nChannels: number = 8,
  steps: number = 200,
  epsilon: number = 1e-8,
): SweepPoint[] {
  const points: SweepPoint[] = [];
  const w = Array(nChannels).fill(1 / nChannels);
  for (let i = 0; i <= steps; i++) {
    const v = i / steps;
    const c = Array(nChannels).fill(v);
    const result = computeKernel(c, w, epsilon);
    const { regime, isCritical } = classifyRegime(result);
    points.push({
      param: v,
      F: result.F, omega: result.omega, S: result.S,
      C: result.C, kappa: result.kappa, IC: result.IC,
      delta: result.delta, regime, isCritical,
    });
  }
  return points;
}

/**
 * Compute Bernoulli entropy curve for a single channel c ∈ [0,1].
 */
export function entropyCurve(steps: number = 200): { c: number; S: number; kappa: number; f: number }[] {
  const pts: { c: number; S: number; kappa: number; f: number }[] = [];
  for (let i = 0; i <= steps; i++) {
    const c = i / steps;
    const S = bernoulliEntropy(c);
    const cε = Math.max(c, 1e-8);
    const kappa = Math.log(cε);
    pts.push({ c, S, kappa, f: S + kappa });
  }
  return pts;
}

/**
 * Compute the Γ(ω) drift cost curve.
 */
export function gammaCurve(steps: number = 200): { omega: number; gamma: number }[] {
  const pts: { omega: number; gamma: number }[] = [];
  for (let i = 0; i <= steps; i++) {
    const omega = i / steps;
    pts.push({ omega, gamma: gammaOmega(omega) });
  }
  return pts;
}

/**
 * 2D heatmap: vary two channels, compute a chosen metric.
 * Returns a flat Float64Array of size (steps+1)^2 in row-major order.
 */
export function heatmap2D(
  channels: number[],
  weights: number[],
  ch1: number,
  ch2: number,
  metric: 'F' | 'IC' | 'delta' | 'S' | 'C' | 'omega' = 'IC',
  steps: number = 60,
  epsilon: number = 1e-8,
): { data: Float64Array; size: number; min: number; max: number } {
  const n = steps + 1;
  const data = new Float64Array(n * n);
  const c = [...channels];
  let min = Infinity, max = -Infinity;
  for (let j = 0; j < n; j++) {
    c[ch2] = j / steps;
    for (let i = 0; i < n; i++) {
      c[ch1] = i / steps;
      const r = computeKernel(c, weights, epsilon);
      const v = r[metric];
      data[j * n + i] = v;
      if (v < min) min = v;
      if (v > max) max = v;
    }
  }
  return { data, size: n, min, max };
}

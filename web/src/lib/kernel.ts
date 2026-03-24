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
  REGIME_THRESHOLDS, C_STAR, C_TRAP,
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

/* ─── Domain Presets (Cross-Domain Entity Library) ──────────────── */

export interface DomainPreset {
  name: string;
  domain: string;
  c: number[];
  w: number[];
  description: string;
}

export const DOMAIN_PRESETS: DomainPreset[] = [
  // Standard Model — Subatomic
  { name: 'Electron', domain: 'standard_model', description: 'Lightest charged lepton — stable, high fidelity',
    c: [0.37, 1.0, 0.67, 0.0, 0.75, 1.0, 0.0, 0.33], w: Array(8).fill(0.125) },
  { name: 'Top Quark', domain: 'standard_model', description: 'Heaviest quark — mass dominates trace',
    c: [0.73, 0.50, 0.44, 1.0, 0.75, 0.0, 0.33, 1.0], w: Array(8).fill(0.125) },
  { name: 'Higgs Boson', domain: 'standard_model', description: 'Scalar boson — spin=0 channel is distinctive',
    c: [0.72, 0.0, 0.0, 0.0, 0.50, 0.0, 0.0, 0.0], w: Array(8).fill(0.125) },
  { name: 'Proton', domain: 'standard_model', description: 'Composite hadron — confinement kills IC via dead color channel',
    c: [0.60, 0.50, 1e-8, 1.0, 1e-8, 0.0, 0.33, 0.33], w: Array(8).fill(0.125) },
  // Atomic Physics
  { name: 'Iron (Fe)', domain: 'atomic_physics', description: 'Peak nuclear binding energy — balanced channels',
    c: [0.77, 0.82, 0.71, 0.88, 0.78, 0.76, 0.79, 0.65], w: Array(8).fill(0.125) },
  { name: 'Helium (He)', domain: 'atomic_physics', description: 'Noble gas — magic number nucleus, high IC',
    c: [0.25, 0.95, 0.0, 0.01, 0.98, 0.01, 0.32, 0.0], w: Array(8).fill(0.125) },
  { name: 'Uranium (U)', domain: 'atomic_physics', description: 'Heaviest natural element — radioactive, IC stressed',
    c: [0.93, 0.42, 0.55, 0.92, 0.38, 0.82, 0.90, 0.75], w: Array(8).fill(0.125) },
  // Nuclear Physics
  { name: 'Deuterium', domain: 'nuclear_physics', description: 'Simplest composite nucleus — p+n binding',
    c: [0.30, 0.95, 0.50, 0.45, 0.70, 0.60, 0.55, 0.80], w: Array(8).fill(0.125) },
  { name: 'QGP (RHIC)', domain: 'nuclear_physics', description: 'Quark-gluon plasma — deconfined state, all channels stressed',
    c: [0.35, 0.20, 0.45, 0.30, 0.15, 0.25, 0.40, 0.10], w: Array(8).fill(0.125) },
  // Astronomy
  { name: 'Sun (Main Seq.)', domain: 'astronomy', description: 'G-type main sequence star — stable hydrogen burning',
    c: [0.92, 0.85, 0.88, 0.90, 0.87, 0.93, 0.86, 0.91], w: Array(8).fill(0.125) },
  { name: 'Neutron Star', domain: 'astronomy', description: 'Collapsed stellar remnant — extreme density channel',
    c: [0.95, 0.10, 0.98, 0.05, 0.90, 0.02, 0.88, 0.30], w: Array(8).fill(0.125) },
  // Cosmology (Weyl)
  { name: 'Dark Energy (Λ)', domain: 'weyl', description: 'Cosmological constant — dominates late-time expansion',
    c: [0.68, 0.99, 0.05, 0.95, 0.10, 0.98, 0.15, 0.90], w: Array(8).fill(0.125) },
  // Finance
  { name: 'S&P 500 (Bull)', domain: 'finance', description: 'Bull market conditions — high fidelity, low drift',
    c: [0.92, 0.88, 0.85, 0.90, 0.87, 0.93, 0.80, 0.91], w: Array(8).fill(0.125) },
  { name: 'Black Monday', domain: 'finance', description: '1987 crash — extreme drift, IC near zero',
    c: [0.15, 0.10, 0.20, 0.05, 0.30, 0.12, 0.08, 0.25], w: Array(8).fill(0.125) },
  // Consciousness / Neuroscience
  { name: 'Waking State', domain: 'consciousness_coherence', description: 'Normal waking consciousness — high coherence across cortical channels',
    c: [0.88, 0.85, 0.82, 0.90, 0.86, 0.84, 0.87, 0.83], w: Array(8).fill(0.125) },
  { name: 'Deep Sleep (N3)', domain: 'consciousness_coherence', description: 'Slow-wave sleep — reduced channel variability, high synchrony',
    c: [0.70, 0.72, 0.71, 0.69, 0.73, 0.70, 0.68, 0.71], w: Array(8).fill(0.125) },
  { name: 'Seizure', domain: 'clinical_neuroscience', description: 'Epileptic seizure — hypersynchrony destroys channel diversity',
    c: [0.95, 0.94, 0.96, 0.95, 0.93, 0.95, 0.94, 0.01], w: Array(8).fill(0.125) },
  // Evolution
  { name: 'E. coli', domain: 'evolution', description: 'Prokaryote — simple, high-fidelity genome replication',
    c: [0.95, 0.30, 0.10, 0.85, 0.20, 0.40, 0.90, 0.50], w: Array(8).fill(0.125) },
  { name: 'Human Brain', domain: 'evolution', description: 'Highest complexity — 10-channel brain kernel compressed to 8',
    c: [0.85, 0.80, 0.75, 0.90, 0.70, 0.82, 0.78, 0.88], w: Array(8).fill(0.125) },
  // Kinematics
  { name: 'Free Fall', domain: 'kinematics', description: 'Gravitational free fall — clean phase space trajectory',
    c: [0.98, 0.95, 0.90, 0.97, 0.92, 0.96, 0.85, 0.93], w: Array(8).fill(0.125) },
  // Semiotics
  { name: 'Natural Language', domain: 'dynamic_semiotics', description: 'Living language — moderate drift with strong return',
    c: [0.80, 0.75, 0.70, 0.85, 0.65, 0.78, 0.72, 0.82], w: Array(8).fill(0.125) },
  { name: 'Dead Language', domain: 'dynamic_semiotics', description: 'Extinct sign system — no return channel',
    c: [0.90, 0.10, 0.85, 0.05, 0.92, 0.08, 0.88, 0.03], w: Array(8).fill(0.125) },
  // Quantum Mechanics
  { name: 'Bell Singlet', domain: 'quantum_mechanics', description: 'Maximally entangled pair — extreme correlation',
    c: [0.50, 0.50, 0.50, 0.50, 1.0, 0.50, 0.50, 0.50], w: Array(8).fill(0.125) },
  // Spacetime Memory
  { name: 'GW150914', domain: 'spacetime_memory', description: 'First gravitational wave detection — chirp signal',
    c: [0.75, 0.80, 0.70, 0.65, 0.85, 0.60, 0.78, 0.72], w: Array(8).fill(0.125) },
  // Corner probes
  { name: 'All Zero (ε-floor)', domain: 'corner', description: 'All channels at guard band — maximum collapse',
    c: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], w: Array(8).fill(0.125) },
  { name: 'All One (1−ε ceil)', domain: 'corner', description: 'All channels at ceiling — maximum fidelity',
    c: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], w: Array(8).fill(0.125) },
  { name: 'Equator (c=0.5)', domain: 'corner', description: 'All channels at 0.5 — S+κ=0 convergence point',
    c: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], w: Array(8).fill(0.125) },
  { name: 'Self-Dual (c*)', domain: 'corner', description: 'All channels at c*=0.7822 — maximizes S+κ per channel',
    c: [0.7822, 0.7822, 0.7822, 0.7822, 0.7822, 0.7822, 0.7822, 0.7822], w: Array(8).fill(0.125) },
  { name: 'Drift Trap (c_trap)', domain: 'corner', description: 'All channels at c_trap=0.3177 — Cardano root',
    c: [0.3177, 0.3177, 0.3177, 0.3177, 0.3177, 0.3177, 0.3177, 0.3177], w: Array(8).fill(0.125) },
];

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
/* ─── Fisher Geometry ───────────────────────────────────────────── */

export interface FisherResult {
  theta: number;          // Fisher coordinate θ = arccos(√F)
  sinTheta: number;       // sin θ (drift component)
  cosTheta: number;       // cos θ (fidelity component)
  metricG: number;        // g_F(θ) = 1 (flat manifold)
  fisherInfo: number;     // Fisher information I_F = Var(c)/(2c̄)
  regimeAngle: number;    // θ normalized to [0, π/2]
}

/**
 * Compute Fisher-geometric coordinates from a kernel result.
 *
 * The Bernoulli manifold is flat in Fisher coordinates: g_F(θ) = 1.
 * All structure comes from embedding, not intrinsic curvature.
 */
export function fisherCoordinates(result: KernelResult): FisherResult {
  const Fclamped = Math.max(EPSILON, Math.min(1 - EPSILON, result.F));
  const theta = Math.acos(Math.sqrt(Fclamped));
  const cosTheta = Math.sqrt(Fclamped);
  const sinTheta = Math.sqrt(1 - Fclamped);
  // Fisher information from heterogeneity
  const fisherInfo = result.C > 0 ? (result.C * 0.5) ** 2 / (2 * Fclamped) : 0;
  return {
    theta,
    sinTheta,
    cosTheta,
    metricG: 1.0,  // flat manifold — always 1
    fisherInfo,
    regimeAngle: theta,
  };
}

/* ─── Rank Classification ───────────────────────────────────────── */

export type RankLabel = 1 | 2 | 3;

export interface RankResult {
  rank: RankLabel;
  description: string;
  effectiveDOF: number;
  isHomogeneous: boolean;
  nDistinct: number;
}

/**
 * Classify the rank of a trace vector.
 *
 * Rank-1: All cᵢ = c₀ (homogeneous). IC = F, C = 0. 1 DOF.
 * Rank-2: Effective 2-channel structure. C = g(F, κ). 2 DOF.
 * Rank-3: General heterogeneous (n ≥ 3 distinct values). 3 DOF.
 */
export function classifyRank(c: number[], tolerance: number = 1e-6): RankResult {
  if (c.length === 0) return { rank: 1, description: 'Empty (trivial)', effectiveDOF: 0, isHomogeneous: true, nDistinct: 0 };

  // Count distinct values
  const sorted = [...c].sort((a, b) => a - b);
  let nDistinct = 1;
  for (let i = 1; i < sorted.length; i++) {
    if (Math.abs(sorted[i] - sorted[i - 1]) > tolerance) nDistinct++;
  }

  if (nDistinct === 1) {
    return { rank: 1, description: 'Rank-1: Homogeneous (all channels equal)', effectiveDOF: 1, isHomogeneous: true, nDistinct };
  }
  if (nDistinct === 2) {
    return { rank: 2, description: 'Rank-2: Two-channel structure', effectiveDOF: 2, isHomogeneous: false, nDistinct };
  }
  return { rank: 3, description: 'Rank-3: General heterogeneous', effectiveDOF: 3, isHomogeneous: false, nDistinct };
}

/* ─── Composition Algebra ───────────────────────────────────────── */

export interface CompositionResult {
  F_composed: number;      // (F₁ + F₂) / 2 — arithmetic
  IC_composed: number;     // √(IC₁ · IC₂) — geometric
  delta_composed: number;  // Composed heterogeneity gap
  gap_predicted: number;   // Predicted gap from composition law
  hellinger_correction: number;
  omega_composed: number;
  regime: RegimeLabel;
  isCritical: boolean;
}

/**
 * Compose two kernel results using the GCD composition algebra.
 *
 * F composes arithmetically: F₁₂ = (F₁ + F₂) / 2
 * IC composes geometrically: IC₁₂ = √(IC₁ · IC₂)
 * Gap composition: Δ₁₂ = (Δ₁ + Δ₂)/2 + (√IC₁ − √IC₂)²/2
 */
export function composeKernels(r1: KernelResult, r2: KernelResult): CompositionResult {
  const F_composed = (r1.F + r2.F) / 2;
  const IC_composed = Math.sqrt(Math.max(0, r1.IC * r2.IC));
  const omega_composed = 1 - F_composed;
  const delta_composed = F_composed - IC_composed;

  // Hellinger-like correction term
  const hellinger_correction = (Math.sqrt(Math.max(0, r1.IC)) - Math.sqrt(Math.max(0, r2.IC))) ** 2 / 2;
  const gap_predicted = (r1.delta + r2.delta) / 2 + hellinger_correction;

  // Classify composed regime
  const composedResult: KernelResult = {
    F: F_composed, omega: omega_composed, S: (r1.S + r2.S) / 2,
    C: (r1.C + r2.C) / 2, kappa: Math.log(Math.max(EPSILON, IC_composed)),
    IC: IC_composed, delta: delta_composed, isHomogeneous: false,
  };
  const { regime, isCritical } = classifyRegime(composedResult);

  return {
    F_composed, IC_composed, delta_composed, gap_predicted,
    hellinger_correction, omega_composed, regime, isCritical,
  };
}

/* ─── Orientation Receipts ──────────────────────────────────────── */

export interface OrientationReceipt {
  section: number;
  name: string;
  description: string;
  value: number;
  expected: string;
  pass: boolean;
}

/**
 * Compute the 10 orientation receipts from scripts/orientation.py.
 * These are compressed derivation chains that constrain classification.
 */
export function computeOrientationReceipts(): OrientationReceipt[] {
  const receipts: OrientationReceipt[] = [];

  // §1: Duality — F + ω = 1 exactly across 10K traces
  let maxDualityResidual = 0;
  const rng = mulberry32Kernel(42);
  for (let i = 0; i < 10000; i++) {
    const n = 2 + Math.floor(rng() * 15);
    const c = Array.from({ length: n }, () => rng());
    const r = computeKernel(c);
    const res = Math.abs(r.F + r.omega - 1.0);
    if (res > maxDualityResidual) maxDualityResidual = res;
  }
  receipts.push({ section: 1, name: 'Duality', description: 'max|F + ω − 1| across 10K traces',
    value: maxDualityResidual, expected: '0.0e+00', pass: maxDualityResidual < 1e-12 });

  // §2: Integrity bound — Δ for (0.95, 0.001)
  const r2 = computeKernel([0.95, 0.001]);
  receipts.push({ section: 2, name: 'Integrity Bound', description: 'Δ for (0.95, 0.001)',
    value: r2.delta, expected: '~0.4447', pass: r2.delta > 0.4 && r2.delta < 0.5 });

  // §3: Geometric slaughter — IC/F with 1 dead channel (8ch)
  const c3 = [0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 1e-8];
  const w3 = Array(8).fill(0.125);
  const r3 = computeKernel(c3, w3);
  const icF3 = r3.IC / r3.F;
  receipts.push({ section: 3, name: 'Geometric Slaughter', description: 'IC/F with 1 dead channel (8ch)',
    value: icF3, expected: '~0.1143', pass: icF3 > 0.05 && icF3 < 0.15 });

  // §4: First weld — Γ(0.682)
  const g4 = gammaOmega(0.682);
  receipts.push({ section: 4, name: 'First Weld', description: 'Γ(0.682)',
    value: g4, expected: '~0.9975', pass: g4 > 0.95 && g4 < 1.05 });

  // §5a: Confinement cliff — Neutron IC/F
  const neutronC = [0.60, 0.50, 1e-8, 0.33, 1e-8, 1e-8, 0.33, 0.33];
  const neutronW = Array(8).fill(0.125);
  const rN = computeKernel(neutronC, neutronW);
  const neutronICF = rN.IC / rN.F;
  receipts.push({ section: 5, name: 'Confinement Cliff (Neutron)', description: 'Neutron IC/F',
    value: neutronICF, expected: '~0.0089', pass: neutronICF < 0.05 });

  // §5b: Proton IC/F
  const protonC = [0.60, 0.50, 1e-8, 1.0, 1e-8, 0.0, 0.33, 0.33];
  const protonW = Array(8).fill(0.125);
  const rP = computeKernel(protonC, protonW);
  const protonICF = rP.IC / rP.F;
  receipts.push({ section: 5, name: 'Confinement Cliff (Proton)', description: 'Proton IC/F',
    value: protonICF, expected: '~0.0371', pass: protonICF < 0.05 });

  // §6: Scale inversion — Nickel IC/F (atomic-scale restores coherence)
  // Nickel: mass_log~0.78, IE~0.80, EN~0.72, density~0.85, mp~0.75, bp~0.72, radius~0.80, ea~0.70
  const nickelC = [0.78, 0.80, 0.72, 0.85, 0.75, 0.72, 0.80, 0.70];
  const nickelW = Array(8).fill(0.125);
  const rNi = computeKernel(nickelC, nickelW);
  const nickelICF = rNi.IC / rNi.F;
  receipts.push({ section: 6, name: 'Scale Inversion', description: 'Nickel IC/F (atoms restore coherence)',
    value: nickelICF, expected: '~0.9573', pass: nickelICF > 0.90 });

  // §8: Equator convergence — S + κ at c = 1/2
  const r8 = computeKernel([0.5, 0.5, 0.5, 0.5]);
  const equatorSum = r8.S + r8.kappa;
  receipts.push({ section: 8, name: 'Equator Convergence', description: 'S + κ at c = 1/2',
    value: equatorSum, expected: '0.0', pass: Math.abs(equatorSum) < 1e-10 });

  // §10: Seam associativity — additive budget chain (monoid)
  // Seam budgets compose additively: Δκ_total = Σ Δκ_i
  // Addition of reals is associative: (a+b)+c = a+(b+c)
  const budA = computeSeamBudget(0.10, 0.20, 1.0, 1.0);
  const budB = computeSeamBudget(0.20, 0.30, 1.0, 1.0);
  const budC = computeSeamBudget(0.35, 0.15, 1.0, 1.0);
  const dk_a = budA.deltaKappa, dk_b = budB.deltaKappa, dk_c = budC.deltaKappa;
  const left_assoc = (dk_a + dk_b) + dk_c;
  const right_assoc = dk_a + (dk_b + dk_c);
  const assocError = Math.abs(left_assoc - right_assoc);
  receipts.push({ section: 10, name: 'Seam Associativity', description: 'Budget chain associativity error',
    value: assocError, expected: '<1e-15', pass: assocError < 1e-14 });

  return receipts;
}

/* ─── Fixed Point Analysis ──────────────────────────────────────── */

export interface FixedPointResult {
  name: string;
  latin: string;
  c_value: number;
  F: number;
  omega: number;
  S: number;
  kappa: number;
  IC: number;
  SpluskKappa: number;  // S + κ — should be 0 at equator
  significance: string;
}

/**
 * Analyze the three canonical fixed points of the Bernoulli manifold.
 *
 * c* = 0.7822  — Logistic self-dual: maximizes S + κ per channel
 * c_trap = 0.3177 — Drift trap: Cardano root of x³+x−1=0
 * c = 0.5 — Equator: S + κ = 0 (four-way convergence)
 */
export function analyzeFixedPoints(): FixedPointResult[] {
  const points = [
    { name: 'Equator', latin: 'Aequator', c: 0.5, sig: 'S + κ = 0 (four-way convergence). Quintuple fixed point.' },
    { name: 'Self-dual (c*)', latin: 'Punctum Reflexivum', c: C_STAR, sig: 'Maximizes S + κ per channel. Logistic self-dual fixed point.' },
    { name: 'Drift trap (c_trap)', latin: 'Laqueus Derivationis', c: C_TRAP, sig: 'Cardano root of x³+x−1=0. Where Γ first drops below 1.0.' },
  ];

  return points.map(p => {
    const r = computeKernel([p.c, p.c, p.c, p.c]);
    return {
      name: p.name,
      latin: p.latin,
      c_value: p.c,
      F: r.F,
      omega: r.omega,
      S: r.S,
      kappa: r.kappa,
      IC: r.IC,
      SpluskKappa: r.S + r.kappa,
      significance: p.sig,
    };
  });
}

/* ─── τ_R* Surface (2D heatmap) ─────────────────────────────────── */

/**
 * Generate a 2D τ_R* surface over the (ω, C) plane.
 * Returns a flat Float64Array in row-major order.
 */
export function tauRStarSurface(
  steps: number = 60,
): { data: Float64Array; size: number; min: number; max: number } {
  const n = steps + 1;
  const data = new Float64Array(n * n);
  let min = Infinity, max = -Infinity;
  for (let j = 0; j < n; j++) {
    const C = j / steps;
    for (let i = 0; i < n; i++) {
      const omega = i / steps;
      const v = computeTauRStar(omega, C);
      // Cap at 100 for visualization (avoids infinity at origin)
      const capped = Math.min(v, 100);
      data[j * n + i] = capped;
      if (capped < min) min = capped;
      if (capped > max) max = capped;
    }
  }
  return { data, size: n, min, max };
}

/* ─── Extended Identity Proofs ──────────────────────────────────── */

export interface ExtendedIdentityCheck extends IdentityCheck {
  section: string;
  description: string;
}

/**
 * Extended identity verification: 10 structural identities.
 * Goes beyond the basic 3 to verify deeper algebraic properties.
 */
export function verifyExtendedIdentities(result: KernelResult, c: number[], w?: number[]): ExtendedIdentityCheck[] {
  const tol = 1e-10;
  const checks: ExtendedIdentityCheck[] = [];

  // 1. Duality: F + ω = 1
  checks.push({
    section: 'A1', name: 'Duality identity', formula: 'F + ω = 1',
    expected: 1.0, actual: result.F + result.omega,
    residual: Math.abs(result.F + result.omega - 1.0),
    pass: Math.abs(result.F + result.omega - 1.0) < tol,
    description: 'Complementum perfectum — no third possibility',
  });

  // 2. Integrity bound: IC ≤ F
  checks.push({
    section: 'A2', name: 'Integrity bound', formula: 'IC ≤ F',
    expected: result.F, actual: result.IC,
    residual: Math.max(0, result.IC - result.F),
    pass: result.IC <= result.F + tol,
    description: 'Solvability condition for trace recovery',
  });

  // 3. Log-integrity: IC = exp(κ)
  const expKappa = Math.exp(result.kappa);
  checks.push({
    section: 'A3', name: 'Log-integrity', formula: 'IC = exp(κ)',
    expected: expKappa, actual: result.IC,
    residual: Math.abs(result.IC - expKappa),
    pass: Math.abs(result.IC - expKappa) < tol,
    description: 'Link between multiplicative and additive coherence',
  });

  // 4. Heterogeneity gap: Δ = F − IC ≥ 0
  checks.push({
    section: 'A4', name: 'Gap non-negativity', formula: 'Δ = F − IC ≥ 0',
    expected: 0, actual: result.delta,
    residual: Math.max(0, -result.delta),
    pass: result.delta >= -tol,
    description: 'Heterogeneity gap is always non-negative',
  });

  // 5. Entropy non-negativity: S ≥ 0
  checks.push({
    section: 'A5', name: 'Entropy non-negativity', formula: 'S ≥ 0',
    expected: 0, actual: result.S,
    residual: Math.max(0, -result.S),
    pass: result.S >= -tol,
    description: 'Bernoulli field entropy is non-negative',
  });

  // 6. Curvature bounds: 0 ≤ C ≤ 1
  checks.push({
    section: 'A6', name: 'Curvature bounds', formula: '0 ≤ C ≤ 1',
    expected: 0.5, actual: result.C,
    residual: Math.max(0, -result.C, result.C - 1),
    pass: result.C >= -tol && result.C <= 1 + tol,
    description: 'Curvature is normalized to [0,1]',
  });

  // 7. Drift bounds: 0 ≤ ω ≤ 1
  checks.push({
    section: 'A7', name: 'Drift bounds', formula: '0 ≤ ω ≤ 1',
    expected: 0.5, actual: result.omega,
    residual: Math.max(0, -result.omega, result.omega - 1),
    pass: result.omega >= -tol && result.omega <= 1 + tol,
    description: 'Drift is bounded by the duality identity',
  });

  // 8. κ ≤ 0 (log-integrity is non-positive)
  checks.push({
    section: 'A8', name: 'κ non-positive', formula: 'κ ≤ 0',
    expected: 0, actual: result.kappa,
    residual: Math.max(0, result.kappa),
    pass: result.kappa <= tol,
    description: 'Log-integrity: ln(c) ≤ 0 for c ∈ (0,1]',
  });

  // 9. Homogeneity check: if C ≈ 0 then IC ≈ F (rank-1)
  if (result.C < 1e-6) {
    checks.push({
      section: 'B1', name: 'Rank-1 identity', formula: 'C ≈ 0 ⟹ IC ≈ F',
      expected: result.F, actual: result.IC,
      residual: Math.abs(result.F - result.IC),
      pass: Math.abs(result.F - result.IC) < 0.01,
      description: 'Homogeneous trace: geometric = arithmetic mean',
    });
  }

  // 10. Solvability test: for 2-channel, c₁,₂ = F ± √(F²−IC²) are real
  if (c.length === 2) {
    const discriminant = result.F * result.F - result.IC * result.IC;
    checks.push({
      section: 'B2', name: 'Solvability condition', formula: 'F² − IC² ≥ 0',
      expected: 0, actual: discriminant,
      residual: Math.max(0, -discriminant),
      pass: discriminant >= -tol,
      description: 'Trace recovery: F² − IC² ≥ 0 for real solutions',
    });

    // Verify recovered channels match originals
    if (discriminant >= 0) {
      const sqrtD = Math.sqrt(discriminant);
      const c1_recovered = result.F + sqrtD;
      const c2_recovered = result.F - sqrtD;
      const n = c.length;
      const ww = w ?? Array(n).fill(1 / n);
      const cClamped = c.map(ci => clamp(ci));
      const cSorted = [...cClamped].sort((a, b) => b - a);
      const rSorted = [c1_recovered, c2_recovered].sort((a, b) => b - a);
      const recoverError = Math.abs(cSorted[0] - rSorted[0]) + Math.abs(cSorted[1] - rSorted[1]);
      checks.push({
        section: 'B3', name: 'Trace recovery', formula: 'cᵢ = F ± √(F²−IC²)',
        expected: 0, actual: recoverError,
        residual: recoverError,
        pass: recoverError < 0.01,
        description: 'Recovered channels match original (2-channel case)',
      });
    }
  }

  return checks;
}

/* ─── Regime Partition Statistics ────────────────────────────────── */

export interface RegimeStats {
  stable: number;
  watch: number;
  collapse: number;
  critical: number;
  total: number;
  stablePct: number;
  watchPct: number;
  collapsePct: number;
  criticalPct: number;
}

/**
 * Estimate regime partition of Fisher space via Monte Carlo sampling.
 * Orientation §7: Stable ≈ 12.5%, Watch ≈ 24.4%, Collapse ≈ 63.1%.
 */
export function estimateRegimePartition(
  nChannels: number = 8,
  nSamples: number = 10000,
  seed: number = 42,
): RegimeStats {
  const rng = mulberry32Kernel(seed);
  let stable = 0, watch = 0, collapse = 0, critical = 0;
  const w = Array(nChannels).fill(1 / nChannels);
  for (let i = 0; i < nSamples; i++) {
    const c = Array.from({ length: nChannels }, () => rng());
    const result = computeKernel(c, w);
    const { regime, isCritical } = classifyRegime(result);
    if (regime === 'STABLE') stable++;
    else if (regime === 'WATCH') watch++;
    else collapse++;
    if (isCritical) critical++;
  }
  return {
    stable, watch, collapse, critical, total: nSamples,
    stablePct: (stable / nSamples) * 100,
    watchPct: (watch / nSamples) * 100,
    collapsePct: (collapse / nSamples) * 100,
    criticalPct: (critical / nSamples) * 100,
  };
}

/* ─── Batch Analysis ────────────────────────────────────────────── */

export interface BatchEntry {
  name: string;
  result: KernelResult;
  regime: RegimeResult;
  rank: RankResult;
  fisher: FisherResult;
  identityPass: boolean;
}

/**
 * Compute kernel results for multiple trace vectors in batch.
 */
export function batchCompute(
  entries: Array<{ name: string; c: number[]; w?: number[] }>,
): BatchEntry[] {
  return entries.map(e => {
    const result = computeKernel(e.c, e.w);
    const regime = classifyRegime(result);
    const rank = classifyRank(e.c);
    const fisher = fisherCoordinates(result);
    const ids = verifyIdentities(result);
    return {
      name: e.name,
      result, regime, rank, fisher,
      identityPass: ids.every(id => id.pass),
    };
  });
}

/* ─── Internal PRNG (deterministic) ─────────────────────────────── */

function mulberry32Kernel(seed: number): () => number {
  let s = seed | 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

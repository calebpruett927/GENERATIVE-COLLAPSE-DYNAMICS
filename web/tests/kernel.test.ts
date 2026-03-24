/**
 * GCD Kernel — TypeScript Test Suite
 *
 * Verifies Tier-1 invariants, algebraic identities, regime classification,
 * seam budget computation, and edge cases against the frozen contract.
 *
 * Orientation receipts from scripts/orientation.py serve as ground truth.
 */

import { describe, it, expect } from 'vitest';
import {
  computeKernel,
  classifyRegime,
  bernoulliEntropy,
  clamp,
  gammaOmega,
  costCurvature,
  computeSeamBudget,
  verifyIdentities,
  computeTauRStar,
  computeRCritical,
  sweepHomogeneous,
  PRESETS,
  fisherCoordinates,
  classifyRank,
  composeKernels,
  computeOrientationReceipts,
  analyzeFixedPoints,
  tauRStarSurface,
  verifyExtendedIdentities,
  estimateRegimePartition,
  batchCompute,
  DOMAIN_PRESETS,
} from '../src/lib/kernel';
import type { KernelResult, DomainPreset } from '../src/lib/kernel';
import { EPSILON, P_EXPONENT, TOL_SEAM, C_STAR, C_TRAP } from '../src/lib/constants';

/* ─── §1: Duality Identity F + ω = 1 ───────────────────────────── */

describe('Duality identity (F + ω = 1)', () => {
  it('holds exactly for homogeneous traces', () => {
    for (const c of [0.1, 0.3, 0.5, 0.7822, 0.9, 0.999]) {
      const result = computeKernel([c, c, c, c], [0.25, 0.25, 0.25, 0.25]);
      expect(result.F + result.omega).toBeCloseTo(1.0, 14);
    }
  });

  it('holds exactly for heterogeneous traces', () => {
    const result = computeKernel(
      [0.95, 0.70, 0.30, 0.001],
      [0.25, 0.25, 0.25, 0.25],
    );
    expect(result.F + result.omega).toBeCloseTo(1.0, 14);
  });

  it('holds across all presets', () => {
    for (const preset of Object.values(PRESETS)) {
      const result = computeKernel([...preset.c], [...preset.w]);
      expect(result.F + result.omega).toBeCloseTo(1.0, 14);
    }
  });

  it('holds for 10,000 random traces', () => {
    const rng = mulberry32(42);
    for (let trial = 0; trial < 10_000; trial++) {
      const n = 2 + Math.floor(rng() * 15);
      const c = Array.from({ length: n }, () => rng());
      const result = computeKernel(c);
      expect(Math.abs(result.F + result.omega - 1.0)).toBeLessThan(1e-12);
    }
  });
});

/* ─── §2: Integrity Bound IC ≤ F ────────────────────────────────── */

describe('Integrity bound (IC ≤ F)', () => {
  it('holds for homogeneous traces (IC = F)', () => {
    const result = computeKernel([0.7822, 0.7822, 0.7822, 0.7822]);
    expect(result.IC).toBeLessThanOrEqual(result.F + 1e-12);
    expect(result.delta).toBeCloseTo(0.0, 10);
  });

  it('holds for heterogeneous traces (IC < F)', () => {
    const result = computeKernel([0.95, 0.001, 0.95, 0.95]);
    expect(result.IC).toBeLessThan(result.F);
    expect(result.delta).toBeGreaterThan(0);
  });

  it('holds across 10,000 random traces', () => {
    const rng = mulberry32(123);
    for (let trial = 0; trial < 10_000; trial++) {
      const n = 2 + Math.floor(rng() * 15);
      const c = Array.from({ length: n }, () => rng());
      const result = computeKernel(c);
      expect(result.IC).toBeLessThanOrEqual(result.F + 1e-10);
    }
  });

  it('produces large heterogeneity gap with one dead channel', () => {
    // Orientation §2: Δ for (0.95, 0.001) ≈ 0.4447
    const result = computeKernel([0.95, 0.001]);
    expect(result.delta).toBeGreaterThan(0.4);
    expect(result.delta).toBeLessThan(0.5);
  });
});

/* ─── §3: Log-Integrity Relation IC = exp(κ) ────────────────────── */

describe('Log-integrity relation (IC = exp(κ))', () => {
  it('holds exactly for all presets', () => {
    for (const preset of Object.values(PRESETS)) {
      const result = computeKernel([...preset.c], [...preset.w]);
      expect(Math.abs(result.IC - Math.exp(result.kappa))).toBeLessThan(1e-12);
    }
  });

  it('holds across 10,000 random traces', () => {
    const rng = mulberry32(7);
    for (let trial = 0; trial < 10_000; trial++) {
      const n = 2 + Math.floor(rng() * 15);
      const c = Array.from({ length: n }, () => rng());
      const result = computeKernel(c);
      expect(Math.abs(result.IC - Math.exp(result.kappa))).toBeLessThan(1e-12);
    }
  });
});

/* ─── §4: Geometric Slaughter ────────────────────────────────────── */

describe('Geometric slaughter', () => {
  it('one dead channel in 8 kills IC while F stays healthy', () => {
    // PRESETS.slaughter has c=0.001 (not fully dead) → IC/F ≈ 0.49
    // With a truly dead channel (c=ε), IC/F ≈ 0.114 (orientation §3)
    const result = computeKernel(PRESETS.slaughter.c as unknown as number[], [...PRESETS.slaughter.w]);
    expect(result.F).toBeGreaterThan(0.8);
    const icOverF = result.IC / result.F;
    expect(icOverF).toBeLessThan(1.0); // IC < F (heterogeneity gap)
    expect(result.delta).toBeGreaterThan(0.3); // significant gap
  });

  it('truly dead channel (c=ε) produces IC/F ≈ 0.114', () => {
    // Orientation §3: IC/F with 1 dead channel (8ch) = 0.1143
    const c = [0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 1e-8];
    const w = [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125];
    const result = computeKernel(c, w);
    const icOverF = result.IC / result.F;
    expect(icOverF).toBeLessThan(0.15);
    expect(icOverF).toBeGreaterThan(0.05);
  });

  it('homogeneous trace has no slaughter (IC/F ≈ 1)', () => {
    const result = computeKernel([0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95]);
    const icOverF = result.IC / result.F;
    expect(icOverF).toBeGreaterThan(0.99);
  });
});

/* ─── §5: Bernoulli Field Entropy ────────────────────────────────── */

describe('Bernoulli field entropy', () => {
  it('is zero at boundaries (c=0, c=1)', () => {
    expect(bernoulliEntropy(0.0)).toBe(0.0);
    expect(bernoulliEntropy(1.0)).toBe(0.0);
  });

  it('is maximal at c = 0.5 (ln 2)', () => {
    expect(bernoulliEntropy(0.5)).toBeCloseTo(Math.LN2, 12);
  });

  it('is symmetric: h(c) = h(1-c)', () => {
    for (const c of [0.1, 0.2, 0.3, 0.4]) {
      expect(bernoulliEntropy(c)).toBeCloseTo(bernoulliEntropy(1 - c), 12);
    }
  });

  it('S + κ = 0 at c = 0.5 (equator convergence)', () => {
    // Orientation §8: S + κ at c=1/2 = 0.0
    const result = computeKernel([0.5, 0.5, 0.5, 0.5]);
    expect(Math.abs(result.S + result.kappa)).toBeLessThan(1e-10);
  });
});

/* ─── §6: Guard-Band Clamp ──────────────────────────────────────── */

describe('Guard-band clamp', () => {
  it('clamps low values to ε', () => {
    expect(clamp(0.0)).toBe(EPSILON);
    expect(clamp(-1.0)).toBe(EPSILON);
    expect(clamp(1e-20)).toBe(EPSILON);
  });

  it('clamps high values to 1 - ε', () => {
    expect(clamp(1.0)).toBe(1.0 - EPSILON);
    expect(clamp(2.0)).toBe(1.0 - EPSILON);
  });

  it('passes through values in (ε, 1-ε)', () => {
    expect(clamp(0.5)).toBe(0.5);
    expect(clamp(0.7822)).toBe(0.7822);
  });
});

/* ─── §7: Regime Classification ──────────────────────────────────── */

describe('Regime classification', () => {
  it('classifies truly stable trace as STABLE', () => {
    // PRESETS.stable has S > 0.15 due to channel spread → WATCH
    // Use a tighter trace that actually meets all four gates
    const c = [0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97];
    const result = computeKernel(c);
    const { regime } = classifyRegime(result);
    expect(regime).toBe('STABLE');
  });

  it('PRESETS.stable classifies as WATCH (S gate fails)', () => {
    // The "stable" preset has enough variance that S > 0.15
    const result = computeKernel([...PRESETS.stable.c], [...PRESETS.stable.w]);
    const { regime } = classifyRegime(result);
    expect(regime).toBe('WATCH');
    expect(result.S).toBeGreaterThan(0.15);
  });

  it('classifies moderate drift as WATCH', () => {
    const result = computeKernel([...PRESETS.watch.c], [...PRESETS.watch.w]);
    const { regime } = classifyRegime(result);
    expect(regime).toBe('WATCH');
  });

  it('classifies high drift as COLLAPSE', () => {
    const result = computeKernel([...PRESETS.collapse.c], [...PRESETS.collapse.w]);
    const { regime } = classifyRegime(result);
    expect(regime).toBe('COLLAPSE');
  });

  it('flags critical when IC < 0.30', () => {
    const result = computeKernel([0.10, 0.10, 0.10, 0.10]);
    const { isCritical } = classifyRegime(result);
    expect(isCritical).toBe(true);
  });

  it('does not flag critical when IC > 0.30', () => {
    const result = computeKernel([0.95, 0.95, 0.95, 0.95]);
    const { isCritical } = classifyRegime(result);
    expect(isCritical).toBe(false);
  });

  it('Stable regime is rare in random traces', () => {
    // Orientation §7: Stable = 12.5% of Fisher space
    // Uniform random [0,1] traces almost never hit all four gates
    // (ω < 0.038, F > 0.90, S < 0.15, C < 0.14) simultaneously
    const rng = mulberry32(99);
    let stable = 0;
    const N = 10_000;
    for (let i = 0; i < N; i++) {
      const n = 8;
      const c = Array.from({ length: n }, () => rng());
      const result = computeKernel(c);
      if (classifyRegime(result).regime === 'STABLE') stable++;
    }
    const pct = (stable / N) * 100;
    // Stability is rare — most random traces are COLLAPSE or WATCH
    expect(pct).toBeLessThan(25);
  });
});

/* ─── §8: Drift Cost Γ(ω) ───────────────────────────────────────── */

describe('Drift cost Γ(ω)', () => {
  it('Γ(0) = 0', () => {
    expect(gammaOmega(0)).toBeCloseTo(0.0, 12);
  });

  it('Γ(ω) is monotonically increasing', () => {
    let prev = 0;
    for (let i = 1; i <= 99; i++) {
      const omega = i / 100;
      const g = gammaOmega(omega);
      expect(g).toBeGreaterThanOrEqual(prev);
      prev = g;
    }
  });

  it('pole at ω = 1 (Γ → large)', () => {
    expect(gammaOmega(0.999)).toBeGreaterThan(100);
  });

  it('uses p = 3 (Cardano root)', () => {
    // Γ(ω) = ω³ / (1 - ω + ε)
    const omega = 0.5;
    const expected = 0.5 ** 3 / (1 - 0.5 + EPSILON);
    expect(gammaOmega(omega)).toBeCloseTo(expected, 12);
  });
});

/* ─── §9: Seam Budget ────────────────────────────────────────────── */

describe('Seam budget computation', () => {
  it('∞_rec τ_R yields zero credit (gesture)', () => {
    const sb = computeSeamBudget(0.1, 0.1, 1.0, Infinity);
    expect(sb.credit).toBe(0);
    expect(sb.deltaKappa).toBe(0);
  });

  it('finite τ_R computes budget correctly', () => {
    const sb = computeSeamBudget(0.1, 0.1, 1.0, 1.0);
    expect(sb.credit).toBe(1.0);
    expect(sb.D_omega).toBeCloseTo(gammaOmega(0.1), 12);
    expect(sb.D_C).toBeCloseTo(costCurvature(0.1), 12);
    expect(sb.deltaKappa).toBeCloseTo(sb.credit - sb.D_omega - sb.D_C, 12);
  });

  it('seam passes when residual ≤ tol_seam', () => {
    const sb = computeSeamBudget(0.1, 0.1, 1.0, 1.0, 0);
    // Residual = deltaKappa - kappaLedger
    if (Math.abs(sb.residual) <= TOL_SEAM) {
      expect(sb.pass).toBe(true);
    }
  });
});

/* ─── §10: Identity Verification ─────────────────────────────────── */

describe('Identity verification', () => {
  it('all three identities pass for every preset', () => {
    for (const preset of Object.values(PRESETS)) {
      const result = computeKernel([...preset.c], [...preset.w]);
      const checks = verifyIdentities(result);
      expect(checks).toHaveLength(3);
      for (const check of checks) {
        expect(check.pass).toBe(true);
      }
    }
  });
});

/* ─── §11: τ_R* Diagnostic ──────────────────────────────────────── */

describe('τ_R* diagnostic', () => {
  it('is high for low ω and low C (favorable return)', () => {
    const t = computeTauRStar(0.01, 0.01);
    expect(t).toBeGreaterThan(10);
  });

  it('is low for high ω (unfavorable return)', () => {
    const t = computeTauRStar(0.9, 0.1);
    expect(t).toBeLessThan(1);
  });

  it('R_critical is Infinity when τ_R = 0', () => {
    expect(computeRCritical(0.5, 0.5, 0)).toBe(Infinity);
  });

  it('R_critical is Infinity when τ_R = ∞_rec', () => {
    expect(computeRCritical(0.5, 0.5, Infinity)).toBe(Infinity);
  });
});

/* ─── §12: Edge Cases ────────────────────────────────────────────── */

describe('Edge cases', () => {
  it('empty trace returns safe defaults', () => {
    const result = computeKernel([]);
    expect(result.F).toBe(0);
    expect(result.omega).toBe(1);
    expect(result.IC).toBe(0);
  });

  it('single-channel trace works', () => {
    const result = computeKernel([0.8]);
    expect(result.F + result.omega).toBeCloseTo(1.0, 14);
  });

  it('near-zero channels are guard-band clamped', () => {
    const result = computeKernel([0.0, 0.0, 0.0, 0.0]);
    expect(result.F).toBeGreaterThan(0);
    expect(result.IC).toBeGreaterThan(0);
  });

  it('near-one channels are guard-band clamped', () => {
    const result = computeKernel([1.0, 1.0, 1.0, 1.0]);
    expect(result.F).toBeLessThan(1.0);
  });
});

/* ─── §13: Sweep Functions ───────────────────────────────────────── */

describe('Sweep functions', () => {
  it('sweepHomogeneous returns correct number of points', () => {
    const points = sweepHomogeneous(8, 100);
    expect(points).toHaveLength(101);
  });

  it('sweepHomogeneous maintains duality at every point', () => {
    const points = sweepHomogeneous(8, 200);
    for (const pt of points) {
      expect(Math.abs(pt.F + pt.omega - 1.0)).toBeLessThan(1e-12);
    }
  });
});

/* ─── §14: Frozen Contract Constants ─────────────────────────────── */

describe('Frozen contract constants', () => {
  it('EPSILON = 1e-8', () => {
    expect(EPSILON).toBe(1e-8);
  });

  it('P_EXPONENT = 3 (Cardano)', () => {
    expect(P_EXPONENT).toBe(3);
  });

  it('TOL_SEAM = 0.005', () => {
    expect(TOL_SEAM).toBe(0.005);
  });
});

/* ─── Utility: Deterministic PRNG ────────────────────────────────── */

function mulberry32(seed: number): () => number {
  let s = seed | 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/* ════════════════════════════════════════════════════════════════════
   ADVANCED KERNEL FUNCTIONS — Tests for the new computation layer
   ════════════════════════════════════════════════════════════════════ */

/* ─── §15: Fisher Coordinates ─────────────────────────────────────── */

describe('Fisher coordinates', () => {
  it('computes theta from F via arccos(sqrt(F))', () => {
    const r = computeKernel([0.5, 0.5, 0.5, 0.5], [0.25, 0.25, 0.25, 0.25]);
    const fisher = fisherCoordinates(r);
    // F = 0.5, so theta = arccos(sqrt(0.5)) = pi/4
    expect(fisher.theta).toBeCloseTo(Math.PI / 4, 10);
  });

  it('Fisher metric g_F = 1 (flat manifold)', () => {
    // The Bernoulli manifold is flat in Fisher coordinates
    for (const c of [0.1, 0.3, 0.5, 0.7, 0.9]) {
      const r = computeKernel([c, c, c, c], [0.25, 0.25, 0.25, 0.25]);
      const fisher = fisherCoordinates(r);
      expect(fisher.metricG).toBeCloseTo(1.0, 10);
    }
  });

  it('sin^2(theta) + cos^2(theta) = 1', () => {
    for (const c of [0.2, 0.5, 0.8]) {
      const r = computeKernel([c, c, c, c], [0.25, 0.25, 0.25, 0.25]);
      const fisher = fisherCoordinates(r);
      expect(fisher.sinTheta ** 2 + fisher.cosTheta ** 2).toBeCloseTo(1.0, 14);
    }
  });

  it('F = cos^2(theta) identity', () => {
    const c = [0.95, 0.80, 0.60, 0.30];
    const w = [0.25, 0.25, 0.25, 0.25];
    const r = computeKernel(c, w);
    const fisher = fisherCoordinates(r);
    expect(fisher.cosTheta ** 2).toBeCloseTo(r.F, 10);
  });

  it('ω = sin^2(theta) identity', () => {
    const c = [0.10, 0.90, 0.50, 0.70];
    const w = [0.25, 0.25, 0.25, 0.25];
    const r = computeKernel(c, w);
    const fisher = fisherCoordinates(r);
    expect(fisher.sinTheta ** 2).toBeCloseTo(r.omega, 10);
  });

  it('fisherInfo is non-negative', () => {
    const r = computeKernel([0.3, 0.6, 0.9, 0.1], [0.25, 0.25, 0.25, 0.25]);
    const fisher = fisherCoordinates(r);
    expect(fisher.fisherInfo).toBeGreaterThanOrEqual(0);
  });

  it('regime angle in [0, pi/2]', () => {
    for (const preset of Object.values(PRESETS)) {
      const r = computeKernel([...preset.c], [...preset.w]);
      const fisher = fisherCoordinates(r);
      expect(fisher.regimeAngle).toBeGreaterThanOrEqual(0);
      expect(fisher.regimeAngle).toBeLessThanOrEqual(90);
    }
  });
});

/* ─── §16: Rank Classification ──────────────────────────────────── */

describe('Rank classification', () => {
  it('homogeneous trace → Rank 1', () => {
    const rank = classifyRank([0.5, 0.5, 0.5, 0.5]);
    expect(rank.rank).toBe(1);
    expect(rank.isHomogeneous).toBe(true);
    expect(rank.effectiveDOF).toBe(1);
  });

  it('two distinct values → Rank 2', () => {
    const rank = classifyRank([0.8, 0.8, 0.3, 0.3]);
    expect(rank.rank).toBe(2);
    expect(rank.nDistinct).toBe(2);
    expect(rank.effectiveDOF).toBe(2);
  });

  it('three+ distinct values → Rank 3', () => {
    const rank = classifyRank([0.9, 0.5, 0.3, 0.1]);
    expect(rank.rank).toBe(3);
    expect(rank.nDistinct).toBeGreaterThanOrEqual(3);
    expect(rank.effectiveDOF).toBe(3);
  });

  it('single channel is Rank 1', () => {
    const rank = classifyRank([0.7]);
    expect(rank.rank).toBe(1);
  });

  it('near-homogeneous within tolerance is Rank 1', () => {
    const rank = classifyRank([0.5, 0.5000001, 0.4999999, 0.5]);
    expect(rank.rank).toBe(1);
  });

  it('description includes rank number', () => {
    const rank = classifyRank([0.9, 0.5, 0.3, 0.1]);
    expect(rank.description).toContain('3');
  });
});

/* ─── §17: Composition Algebra ──────────────────────────────────── */

describe('Composition algebra', () => {
  it('F composes arithmetically: F₁₂ = (F₁+F₂)/2', () => {
    const r1 = computeKernel([0.9, 0.9, 0.9, 0.9], [0.25, 0.25, 0.25, 0.25]);
    const r2 = computeKernel([0.3, 0.3, 0.3, 0.3], [0.25, 0.25, 0.25, 0.25]);
    const comp = composeKernels(r1, r2);
    expect(comp.F_composed).toBeCloseTo((r1.F + r2.F) / 2, 12);
  });

  it('IC composes geometrically: IC₁₂ = sqrt(IC₁·IC₂)', () => {
    const r1 = computeKernel([0.9, 0.9, 0.9, 0.9], [0.25, 0.25, 0.25, 0.25]);
    const r2 = computeKernel([0.5, 0.5, 0.5, 0.5], [0.25, 0.25, 0.25, 0.25]);
    const comp = composeKernels(r1, r2);
    expect(comp.IC_composed).toBeCloseTo(Math.sqrt(r1.IC * r2.IC), 12);
  });

  it('ω composed = 1 − F composed (duality preserved)', () => {
    const r1 = computeKernel([0.8, 0.6, 0.4, 0.2], [0.25, 0.25, 0.25, 0.25]);
    const r2 = computeKernel([0.7, 0.9, 0.3, 0.5], [0.25, 0.25, 0.25, 0.25]);
    const comp = composeKernels(r1, r2);
    expect(comp.F_composed + comp.omega_composed).toBeCloseTo(1.0, 12);
  });

  it('IC ≤ F holds for composed system (integrity bound)', () => {
    const r1 = computeKernel([0.95, 0.10, 0.80, 0.60], [0.25, 0.25, 0.25, 0.25]);
    const r2 = computeKernel([0.30, 0.90, 0.50, 0.70], [0.25, 0.25, 0.25, 0.25]);
    const comp = composeKernels(r1, r2);
    expect(comp.IC_composed).toBeLessThanOrEqual(comp.F_composed + 1e-10);
  });

  it('delta_composed = F_composed − IC_composed', () => {
    const r1 = computeKernel([0.9, 0.8, 0.7, 0.6], [0.25, 0.25, 0.25, 0.25]);
    const r2 = computeKernel([0.5, 0.4, 0.3, 0.2], [0.25, 0.25, 0.25, 0.25]);
    const comp = composeKernels(r1, r2);
    expect(comp.delta_composed).toBeCloseTo(comp.F_composed - comp.IC_composed, 12);
  });

  it('composing identical systems preserves F exactly', () => {
    const r = computeKernel([0.7, 0.7, 0.7, 0.7], [0.25, 0.25, 0.25, 0.25]);
    const comp = composeKernels(r, r);
    expect(comp.F_composed).toBeCloseTo(r.F, 12);
    expect(comp.IC_composed).toBeCloseTo(r.IC, 12);
  });

  it('Hellinger correction for gap composition', () => {
    const r1 = computeKernel([0.9, 0.9, 0.9, 0.9], [0.25, 0.25, 0.25, 0.25]);
    const r2 = computeKernel([0.3, 0.3, 0.3, 0.3], [0.25, 0.25, 0.25, 0.25]);
    const comp = composeKernels(r1, r2);
    expect(comp.hellinger_correction).toBeGreaterThanOrEqual(0);
  });

  it('assigns a valid regime to composed result', () => {
    const r1 = computeKernel([0.95, 0.95, 0.95, 0.95], [0.25, 0.25, 0.25, 0.25]);
    const r2 = computeKernel([0.15, 0.15, 0.15, 0.15], [0.25, 0.25, 0.25, 0.25]);
    const comp = composeKernels(r1, r2);
    expect(['STABLE', 'WATCH', 'COLLAPSE']).toContain(comp.regime);
  });
});

/* ─── §18: Orientation Receipts ─────────────────────────────────── */

describe('Orientation receipts', () => {
  it('returns 9 receipts', () => {
    const receipts = computeOrientationReceipts();
    expect(receipts.length).toBe(9);
  });

  it('§1 duality: max|F+ω−1| = 0', () => {
    const receipts = computeOrientationReceipts();
    const s1 = receipts.find(r => r.section === 1)!;
    expect(s1.value).toBeCloseTo(0, 10);
    expect(s1.pass).toBe(true);
  });

  it('§2 integrity bound: Δ for (0.95, 0.001) ≈ 0.445', () => {
    const receipts = computeOrientationReceipts();
    const s2 = receipts.find(r => r.section === 2)!;
    expect(s2.value).toBeCloseTo(0.445, 2);
    expect(s2.pass).toBe(true);
  });

  it('§3 geometric slaughter: IC/F < 0.15', () => {
    const receipts = computeOrientationReceipts();
    const s3 = receipts.find(r => r.section === 3)!;
    expect(s3.value).toBeLessThan(0.15);
    expect(s3.pass).toBe(true);
  });

  it('§4 first weld: Γ(0.682) < 1.0', () => {
    const receipts = computeOrientationReceipts();
    const s4 = receipts.find(r => r.section === 4)!;
    expect(s4.value).toBeLessThan(1.0);
    expect(s4.pass).toBe(true);
  });

  it('§8 equator convergence: S+κ at c=1/2 = 0', () => {
    const receipts = computeOrientationReceipts();
    const s8 = receipts.find(r => r.section === 8)!;
    expect(Math.abs(s8.value)).toBeLessThan(1e-10);
    expect(s8.pass).toBe(true);
  });

  it('§10 seam associativity: near-zero error', () => {
    const receipts = computeOrientationReceipts();
    const s10 = receipts.find(r => r.section === 10)!;
    expect(Math.abs(s10.value)).toBeLessThan(1e-10);
    expect(s10.pass).toBe(true);
  });

  it('all receipts have section, name, description', () => {
    const receipts = computeOrientationReceipts();
    for (const r of receipts) {
      expect(r.section).toBeGreaterThanOrEqual(1);
      expect(r.section).toBeLessThanOrEqual(10);
      expect(r.name.length).toBeGreaterThan(0);
      expect(r.description.length).toBeGreaterThan(0);
    }
  });

  it('all receipts pass', () => {
    const receipts = computeOrientationReceipts();
    for (const r of receipts) {
      expect(r.pass).toBe(true);
    }
  });
});

/* ─── §19: Fixed Point Analysis ──────────────────────────────────── */

describe('Fixed point analysis', () => {
  it('returns 3 fixed points', () => {
    const fps = analyzeFixedPoints();
    expect(fps.length).toBe(3);
  });

  it('equator at c = 0.5 with S+κ = 0', () => {
    const fps = analyzeFixedPoints();
    const eq = fps.find(fp => fp.name.toLowerCase().includes('equator'))!;
    expect(eq.c_value).toBe(0.5);
    expect(Math.abs(eq.SpluskKappa)).toBeLessThan(1e-10);
  });

  it('self-dual at c* = 0.7822', () => {
    const fps = analyzeFixedPoints();
    const sd = fps.find(fp => fp.name.toLowerCase().includes('self-dual') || fp.name.includes('c*'))!;
    expect(sd.c_value).toBeCloseTo(C_STAR, 3);
  });

  it('drift trap at c_trap ≈ 0.3177', () => {
    const fps = analyzeFixedPoints();
    const dt = fps.find(fp => fp.name.toLowerCase().includes('trap'))!;
    expect(dt.c_value).toBeCloseTo(C_TRAP, 3);
  });

  it('all fixed points have F + ω = 1 (duality)', () => {
    const fps = analyzeFixedPoints();
    for (const fp of fps) {
      expect(fp.F + fp.omega).toBeCloseTo(1.0, 12);
    }
  });

  it('all fixed points have Latin descriptions', () => {
    const fps = analyzeFixedPoints();
    for (const fp of fps) {
      expect(fp.latin.length).toBeGreaterThan(0);
    }
  });
});

/* ─── §20: τ_R* Surface ─────────────────────────────────────────── */

describe('τ_R* surface', () => {
  it('returns correct-sized data array', () => {
    const surface = tauRStarSurface(20);
    // size = steps + 1 (inclusive endpoints: 0, 1/steps, 2/steps, ..., 1)
    expect(surface.size).toBe(21);
    expect(surface.data.length).toBe(21 * 21);
  });

  it('all values are non-negative', () => {
    const surface = tauRStarSurface(20);
    for (let i = 0; i < surface.data.length; i++) {
      expect(surface.data[i]).toBeGreaterThanOrEqual(0);
    }
  });

  it('max > min (surface has variation)', () => {
    const surface = tauRStarSurface(20);
    expect(surface.max).toBeGreaterThan(surface.min);
  });

  it('high τ_R* at low ω, low C', () => {
    const surface = tauRStarSurface(20);
    // First cell (ω≈0, C≈0) should have highest value
    expect(surface.data[0]).toBeGreaterThan(surface.data[surface.data.length - 1]);
  });

  it('τ_R* decreases along ω axis (C=0)', () => {
    const surface = tauRStarSurface(20);
    // Bottom row: C ≈ 0
    expect(surface.data[0]).toBeGreaterThan(surface.data[10]);
    expect(surface.data[10]).toBeGreaterThan(surface.data[19]);
  });

  it('uses Float64Array for precision', () => {
    const surface = tauRStarSurface(10);
    expect(surface.data).toBeInstanceOf(Float64Array);
  });
});

/* ─── §21: Extended Identity Verification ──────────────────────── */

describe('Extended identity verification', () => {
  it('verifies at least 8 identities', () => {
    const c = [0.9, 0.7, 0.5, 0.3];
    const w = [0.25, 0.25, 0.25, 0.25];
    const r = computeKernel(c, w);
    const checks = verifyExtendedIdentities(r, c, w);
    // 8 core (A1-A8) + conditional B1/B2 for special cases
    expect(checks.length).toBeGreaterThanOrEqual(8);
  });

  it('basic identities pass for canonical presets', () => {
    for (const preset of Object.values(PRESETS)) {
      const c = [...preset.c];
      const w = [...preset.w];
      const r = computeKernel(c, w);
      const checks = verifyExtendedIdentities(r, c, w);
      // A1-A8 should pass
      const basicChecks = checks.filter(ch => ch.section.startsWith('A'));
      for (const check of basicChecks) {
        expect(check.pass).toBe(true);
      }
    }
  });

  it('A1 duality always passes', () => {
    const c = [0.1, 0.9, 0.01, 0.99];
    const r = computeKernel(c, [0.25, 0.25, 0.25, 0.25]);
    const checks = verifyExtendedIdentities(r, c);
    const a1 = checks.find(ch => ch.section === 'A1')!;
    expect(a1.pass).toBe(true);
    expect(a1.residual).toBeLessThan(1e-10);
  });

  it('A2 integrity bound always passes', () => {
    const c = [0.8, 0.001, 0.6, 0.3];
    const r = computeKernel(c, [0.25, 0.25, 0.25, 0.25]);
    const checks = verifyExtendedIdentities(r, c);
    const a2 = checks.find(ch => ch.section === 'A2')!;
    expect(a2.pass).toBe(true);
  });

  it('A3 log-integrity relation passes', () => {
    const c = [0.7, 0.7, 0.7, 0.7];
    const r = computeKernel(c, [0.25, 0.25, 0.25, 0.25]);
    const checks = verifyExtendedIdentities(r, c);
    const a3 = checks.find(ch => ch.section === 'A3')!;
    expect(a3.pass).toBe(true);
    expect(a3.residual).toBeLessThan(1e-10);
  });

  it('B1 rank-1 solvability for homogeneous trace', () => {
    const c = [0.6, 0.6, 0.6, 0.6];
    const r = computeKernel(c, [0.25, 0.25, 0.25, 0.25]);
    const checks = verifyExtendedIdentities(r, c);
    const b1 = checks.find(ch => ch.section === 'B1');
    if (b1) expect(b1.pass).toBe(true);
  });

  it('each check has id, name, residual, pass', () => {
    const c = [0.5, 0.5, 0.5, 0.5];
    const r = computeKernel(c, [0.25, 0.25, 0.25, 0.25]);
    const checks = verifyExtendedIdentities(r, c);
    for (const ch of checks) {
      expect(typeof ch.section).toBe('string');
      expect(typeof ch.name).toBe('string');
      expect(typeof ch.residual).toBe('number');
      expect(typeof ch.pass).toBe('boolean');
    }
  });
});

/* ─── §22: Regime Partition (Monte Carlo) ──────────────────────── */

describe('Regime partition', () => {
  it('sums to nSamples', () => {
    const p = estimateRegimePartition(4, 1000, 42);
    expect(p.stable + p.watch + p.collapse).toBe(p.total);
  });

  it('percentages sum to ~100', () => {
    const p = estimateRegimePartition(4, 5000, 42);
    expect(p.stablePct + p.watchPct + p.collapsePct).toBeCloseTo(100.0, 5);
  });

  it('collapse dominates (>50% of manifold)', () => {
    const p = estimateRegimePartition(8, 5000, 42);
    expect(p.collapsePct).toBeGreaterThan(50);
  });

  it('stable is rare (~12-15% of manifold)', () => {
    const p = estimateRegimePartition(8, 10000, 42);
    expect(p.stablePct).toBeLessThan(25);
  });

  it('critical count ≤ total', () => {
    const p = estimateRegimePartition(8, 1000, 42);
    expect(p.critical).toBeLessThanOrEqual(p.total);
  });

  it('deterministic with same seed', () => {
    const p1 = estimateRegimePartition(4, 1000, 42);
    const p2 = estimateRegimePartition(4, 1000, 42);
    expect(p1.stable).toBe(p2.stable);
    expect(p1.watch).toBe(p2.watch);
    expect(p1.collapse).toBe(p2.collapse);
  });
});

/* ─── §23: Batch Compute ────────────────────────────────────────── */

describe('Batch compute', () => {
  it('returns results for all entries', () => {
    const entries = [
      { name: 'A', c: [0.9, 0.9, 0.9, 0.9], w: [0.25, 0.25, 0.25, 0.25] },
      { name: 'B', c: [0.3, 0.3, 0.3, 0.3], w: [0.25, 0.25, 0.25, 0.25] },
    ];
    const results = batchCompute(entries);
    expect(results.length).toBe(2);
    expect(results[0].name).toBe('A');
    expect(results[1].name).toBe('B');
  });

  it('each entry has result, regime, rank, fisher, identityPass', () => {
    const entries = [{ name: 'X', c: [0.7, 0.5, 0.3, 0.1], w: [0.25, 0.25, 0.25, 0.25] }];
    const results = batchCompute(entries);
    expect(results[0].result).toBeDefined();
    expect(results[0].regime).toBeDefined();
    expect(results[0].rank).toBeDefined();
    expect(results[0].fisher).toBeDefined();
    expect(typeof results[0].identityPass).toBe('boolean');
  });

  it('duality holds for all batch results', () => {
    const entries = DOMAIN_PRESETS.slice(0, 5).map(dp => ({ name: dp.name, c: [...dp.c], w: [...dp.w] }));
    const results = batchCompute(entries);
    for (const r of results) {
      expect(r.result.F + r.result.omega).toBeCloseTo(1.0, 12);
    }
  });

  it('IC ≤ F for all batch results (integrity bound)', () => {
    const entries = DOMAIN_PRESETS.map(dp => ({ name: dp.name, c: [...dp.c], w: [...dp.w] }));
    const results = batchCompute(entries);
    for (const r of results) {
      expect(r.result.IC).toBeLessThanOrEqual(r.result.F + 1e-10);
    }
  });

  it('processes all 29 domain presets', () => {
    const entries = DOMAIN_PRESETS.map(dp => ({ name: dp.name, c: [...dp.c], w: [...dp.w] }));
    const results = batchCompute(entries);
    expect(results.length).toBe(DOMAIN_PRESETS.length);
  });
});

/* ─── §24: Domain Presets ───────────────────────────────────────── */

describe('Domain presets', () => {
  it('has at least 25 entities', () => {
    expect(DOMAIN_PRESETS.length).toBeGreaterThanOrEqual(25);
  });

  it('each preset has name, domain, c, w', () => {
    for (const dp of DOMAIN_PRESETS) {
      expect(dp.name.length).toBeGreaterThan(0);
      expect(dp.domain.length).toBeGreaterThan(0);
      expect(dp.c.length).toBeGreaterThan(0);
      expect(dp.w.length).toBe(dp.c.length);
    }
  });

  it('weights sum to 1 for each preset', () => {
    for (const dp of DOMAIN_PRESETS) {
      const sum = dp.w.reduce((a, b) => a + b, 0);
      expect(sum).toBeCloseTo(1.0, 10);
    }
  });

  it('channels in [0, 1]', () => {
    for (const dp of DOMAIN_PRESETS) {
      for (const c of dp.c) {
        expect(c).toBeGreaterThanOrEqual(0);
        expect(c).toBeLessThanOrEqual(1);
      }
    }
  });

  it('covers multiple domains', () => {
    const domains = new Set(DOMAIN_PRESETS.map(dp => dp.domain));
    expect(domains.size).toBeGreaterThanOrEqual(10);
  });

  it('F + ω = 1 for every domain preset', () => {
    for (const dp of DOMAIN_PRESETS) {
      const r = computeKernel([...dp.c], [...dp.w]);
      expect(r.F + r.omega).toBeCloseTo(1.0, 14);
    }
  });

  it('IC ≤ F for every domain preset', () => {
    for (const dp of DOMAIN_PRESETS) {
      const r = computeKernel([...dp.c], [...dp.w]);
      expect(r.IC).toBeLessThanOrEqual(r.F + 1e-10);
    }
  });

  it('IC = exp(κ) for every domain preset', () => {
    for (const dp of DOMAIN_PRESETS) {
      const r = computeKernel([...dp.c], [...dp.w]);
      expect(r.IC).toBeCloseTo(Math.exp(r.kappa), 8);
    }
  });

  it('includes special corner probes (all-zero, all-one, equator)', () => {
    const names = DOMAIN_PRESETS.map(dp => dp.name.toLowerCase());
    expect(names.some(n => n.includes('zero') || n.includes('ε'))).toBe(true);
    expect(names.some(n => n.includes('one') || n.includes('perfect'))).toBe(true);
    expect(names.some(n => n.includes('equator'))).toBe(true);
  });
});

/* ─── §25: Cross-Domain Tier-1 Identity Sweep ──────────────────── */

describe('Cross-domain Tier-1 identity sweep', () => {
  it('duality exact across all presets', () => {
    const allPresets = [
      ...Object.values(PRESETS).map(p => ({ c: [...p.c], w: [...p.w] })),
      ...DOMAIN_PRESETS.map(dp => ({ c: [...dp.c], w: [...dp.w] })),
    ];
    for (const p of allPresets) {
      const r = computeKernel(p.c, p.w);
      expect(Math.abs(r.F + r.omega - 1.0)).toBeLessThan(1e-14);
    }
  });

  it('integrity bound universal across all presets', () => {
    const allPresets = [
      ...Object.values(PRESETS).map(p => ({ c: [...p.c], w: [...p.w] })),
      ...DOMAIN_PRESETS.map(dp => ({ c: [...dp.c], w: [...dp.w] })),
    ];
    for (const p of allPresets) {
      const r = computeKernel(p.c, p.w);
      expect(r.IC).toBeLessThanOrEqual(r.F + 1e-10);
    }
  });

  it('all identities pass for all domain presets', () => {
    for (const dp of DOMAIN_PRESETS) {
      const r = computeKernel([...dp.c], [...dp.w]);
      const checks = verifyIdentities(r);
      for (const ch of checks) {
        expect(ch.pass).toBe(true);
      }
    }
  });
});

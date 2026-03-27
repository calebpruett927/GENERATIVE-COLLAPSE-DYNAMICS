/**
 * Spacetime Physics -- GCD Budget Surface and Gravitational Analogs
 *
 * Ported from closures/spacetime_memory/spacetime_kernel.py
 * Maps GCD kernel invariants to gravitational phenomena:
 *   Event horizon  = ω -> 1 pole (Γ -> inf)
 *   Gravity        = dΓ/dω (gradient of drift cost)
 *   Mass           = accumulated |κ| (well depth)
 *   Tidal force    = d^2Γ/dω^2 (curvature of cost surface)
 *   Lensing        = deflection from heterogeneity gap Δ
 *   Time dilation  = descent/ascent cost asymmetry
 *
 * Tier-0 Protocol: no Tier-1 symbol is redefined.
 */

import { EPSILON, P_EXPONENT, ALPHA } from './constants';
import { gammaOmega } from './kernel';

/* --- Budget Surface ---------------------------------------------- */

/**
 * Budget surface height: Γ(ω) + alpha*C
 * The 2D cost landscape over (ω, C) parameter space.
 */
export function budgetSurfaceHeight(
  omega: number,
  C: number,
  p: number = P_EXPONENT,
  alpha: number = ALPHA,
  epsilon: number = EPSILON,
): number {
  let num = 1.0;
  for (let i = 0; i < p; i++) num *= omega;
  const gamma = num / (1.0 - omega + epsilon);
  return gamma + alpha * C;
}

/* --- Derivatives of Γ(ω) -- Gravitational Analogs ----------------- */

/**
 * First derivative of Γ(ω) = ω^p / (1 - ω + ε)
 * dΓ/dω = [p*ω^(p-1)*(1-ω+ε) + ω^p] / (1-ω+ε)^2
 * Gravitational analog: "gravitational field strength"
 */
export function dGamma(
  omega: number,
  p: number = P_EXPONENT,
  epsilon: number = EPSILON,
): number {
  const denom = 1.0 - omega + epsilon;
  let omP = 1.0;
  for (let i = 0; i < p; i++) omP *= omega;
  let omPm1 = 1.0;
  for (let i = 0; i < p - 1; i++) omPm1 *= omega;
  return (p * omPm1 * denom + omP) / (denom * denom);
}

/**
 * Second derivative of Γ(ω).
 * Tidal force analog: measures how rapidly the gravitational field changes.
 * Computed numerically for robustness.
 */
export function d2Gamma(
  omega: number,
  p: number = P_EXPONENT,
  epsilon: number = EPSILON,
): number {
  const h = 1e-6;
  const left = omega - h > 0 ? dGamma(omega - h, p, epsilon) : dGamma(0, p, epsilon);
  const right = omega + h < 1 ? dGamma(omega + h, p, epsilon) : dGamma(1.0 - epsilon, p, epsilon);
  return (right - left) / (2 * h);
}

/* --- Well Depth (Mass Analog) ------------------------------------ */

/**
 * Well depth = |κ|. Accumulated log-integrity measures how deep
 * the gravitational potential well is. Larger |κ| -> more massive object.
 */
export function wellDepth(kappa: number): number {
  return Math.abs(kappa);
}

/* --- Gravitational Lensing --------------------------------------- */

export type LensingMorphology =
  | 'perfect_ring'   // Δ < 0.01 -- Einstein ring
  | 'thick_arc'      // Δ < 0.10 -- strong lensing arc
  | 'thin_arc'       // Δ < 0.30 -- distorted arc
  | 'distorted'      // Δ >= 0.30 -- weak lensing distortion
  ;

/**
 * Deflection angle from heterogeneity gap and well depth.
 * theta_defl = 4*wellDepth / (Δ + ε)
 * Analogous to Einstein's deflection formula theta = 4GM/(c^2b).
 */
export function deflectionAngle(
  delta: number,
  wd: number,
  epsilon: number = EPSILON,
): number {
  return 4.0 * wd / (delta + epsilon);
}

/**
 * Classify lensing morphology from heterogeneity gap.
 */
export function classifyLensing(delta: number): LensingMorphology {
  if (delta < 0.01) return 'perfect_ring';
  if (delta < 0.10) return 'thick_arc';
  if (delta < 0.30) return 'thin_arc';
  return 'distorted';
}

/* --- Arrow of Time (Descent/Ascent Asymmetry) -------------------- */

/**
 * Cost of descending from omStart to omEnd (falling in).
 * Integral of Γ(ω) from omStart to omEnd via trapezoidal rule.
 */
export function descentCost(
  omStart: number,
  omEnd: number,
  steps: number = 200,
): number {
  if (omStart >= omEnd) return 0;
  const h = (omEnd - omStart) / steps;
  let sum = 0;
  for (let i = 0; i <= steps; i++) {
    const om = omStart + i * h;
    const g = budgetSurfaceHeight(om, 0);
    sum += (i === 0 || i === steps) ? g * 0.5 : g;
  }
  return sum * h;
}

/**
 * Cost of ascending from omEnd back to omStart (escaping).
 * Asymmetric: ascent integrates against the Γ gradient.
 * ascentCost = descentCost * (1 + Γ(omEnd)/Γ(omStart+ε))
 */
export function ascentCost(
  omStart: number,
  omEnd: number,
): number {
  const dc = descentCost(omStart, omEnd);
  const gStart = budgetSurfaceHeight(Math.max(omStart, 0.001), 0);
  const gEnd = budgetSurfaceHeight(omEnd, 0);
  return dc * (1 + gEnd / (gStart + EPSILON));
}

/**
 * Arrow asymmetry: ratio of ascent cost to descent cost.
 * > 1 means it's harder to escape than to fall in.
 * At the event horizon this diverges -- no return.
 */
export function arrowAsymmetry(
  omStart: number,
  omEnd: number,
): number {
  const dc = descentCost(omStart, omEnd);
  if (dc < EPSILON) return 1.0;
  return ascentCost(omStart, omEnd) / dc;
}

/* --- Additional Physics Functions -------------------------------- */

/**
 * Hawking temperature analog: T_H ~ 1/|κ|
 * In GCD: smaller well depth -> higher temperature.
 * Normalized: T = 1/(8pi*|κ|). Returns 0 for κ near zero.
 */
export function hawkingTemperature(kappa: number): number {
  const wd = Math.abs(kappa);
  if (wd < EPSILON) return 0;
  return 1.0 / (8 * Math.PI * wd);
}

/**
 * Gravitational redshift: z = Γ(ω) / (1 + Γ(ω))
 * Light escaping from depth ω is redshifted. z -> 1 at the horizon.
 */
export function gravitationalRedshift(omega: number): number {
  const gamma = gammaOmega(omega);
  return gamma / (1 + gamma);
}

/**
 * Escape velocity analog: v_esc = sqrt(2Γ/(1+Γ))
 * Approaches 1 (c) at the horizon. Below ISCO, escape is impossible.
 */
export function escapeVelocity(omega: number): number {
  const gamma = gammaOmega(omega);
  return Math.sqrt(2 * gamma / (1 + gamma));
}

/**
 * ISCO (Innermost Stable Circular Orbit) analog in GCD.
 * The ω value where tidal forces destabilize orbits.
 * In Schwarzschild: r_ISCO = 6M = 3r_s -> ω_ISCO ~= 0.50
 */
export const OMEGA_ISCO = 0.50;

/**
 * Photon sphere ω: where circular photon orbits exist.
 * In Schwarzschild: r_ps = 3M = 1.5r_s -> ω_ps ~= 0.65
 */
export const OMEGA_PHOTON_SPHERE = 0.65;

/**
 * Ergosphere outer boundary in GCD.
 * In Kerr: the static limit depends on spin. At maximal spin -> r_ergo = 2M.
 * In GCD: ergosphere begins where frame-dragging cost exceeds escape cost.
 * ω_ergo ~= 0.42 (just inside ISCO for high-spin BH).
 */
export const OMEGA_ERGOSPHERE = 0.42;

/**
 * Marginally bound orbit -- matter falling from infinity reaches this before plunge.
 */
export const OMEGA_MARGINALLY_BOUND = 0.55;

/* --- Frame-Dragging & Kerr Physics ------------------------------- */

/**
 * Frame-dragging angular velocity (Lense-Thirring).
 * In GCD: spin parameter a* maps to angular momentum channel ratio.
 * ω_drag = a* * Γ(ω) / (1 + Γ(ω))^2
 *
 * Near the horizon omega_drag -> a* (4M)^-1 -- frame is dragged at the BH spin rate.
 * @param omega Drift parameter
 * @param spinStar Dimensionless spin a* in [0, 1) -- from trace vector angular momentum channel
 */
export function frameDragging(omega: number, spinStar: number): number {
  const gamma = gammaOmega(omega);
  const denom = (1 + gamma) * (1 + gamma);
  return spinStar * gamma / denom;
}

/**
 * Ergosphere radius factor: how large the ergosphere is relative to the horizon.
 * In Kerr: r_ergo(theta) = M + sqrt(M^2 - a^2cos^2theta). At equator: r_ergo = 2M.
 * GCD analog: ergoFactor = 1 + sqrt(1 - a*^2) for the equatorial plane.
 * Returns ratio r_ergo/r_horizon.
 */
export function ergosphereFactor(spinStar: number): number {
  const a2 = Math.min(spinStar * spinStar, 1 - EPSILON);
  return 1 + Math.sqrt(1 - a2);
}

/**
 * Penrose process: maximum extractable energy from the ergosphere.
 * Uses the irreducible mass: M_irr = M * sqrt((1 + sqrt(1 - a*^2))/2)
 * Maximum efficiency: eta_P = 1 - M_irr/M = 1 - sqrt((1 + sqrt(1 - a*^2))/2)
 * Gives ~29.3% for a* -> 1 (extremal Kerr). This energy comes from
 * the BH's rotational energy -- the BH spins down.
 */
export function penroseEfficiency(spinStar: number): number {
  const a2 = Math.min(spinStar * spinStar, 1 - EPSILON);
  return 1 - Math.sqrt((1 + Math.sqrt(1 - a2)) / 2);
}

/* --- Bekenstein-Hawking Entropy ---------------------------------- */

/**
 * Black hole entropy analog: S_BH = A/(4 l_P^2) -> S_BH ~ |kappa|^2 in GCD.
 * The area of the horizon grows as kappa^2 (log-integrity squared).
 * In GCD the "area" is the occupied volume of the collapse manifold.
 * Includes the 4pi factor from the Bekenstein-Hawking formula.
 * Returns dimensionless entropy proportional to horizon area.
 */
export function bhEntropy(kappa: number): number {
  return 4 * Math.PI * kappa * kappa;
}

/* --- Orbital Precession ------------------------------------------ */

/**
 * Orbital precession rate (perihelion advance analog).
 * In GR: δφ = 6piGM/(c^2a(1-e^2)) per orbit.
 * GCD analog: δφ = 6pi * Γ(ω) / (r_eff^2) where r_eff = 1/(1+Γ).
 * Higher ω -> stronger field -> larger precession.
 */
export function orbitalPrecession(omega: number): number {
  const gamma = gammaOmega(omega);
  const rEff = 1.0 / (1 + gamma);
  return 6 * Math.PI * gamma * rEff * rEff;
}

/* --- Gravitational Wave Strain ----------------------------------- */

/**
 * GW strain (quadrupole formula analog).
 * h ~ (2G/c^4) * (M * v^2) / r ~ Γ(ω)^2 * C for a system with curvature C != 0.
 * Only binary/asymmetric systems (C > 0) emit GW. Spherical collapse (C = 0) is silent.
 *
 * @param omega Drift at emission
 * @param C Curvature (heterogeneity) -- encodes the quadrupole moment
 * @param distance Observer distance (dimensionless, from camera)
 */
export function gwStrain(omega: number, C: number, distance: number): number {
  const gamma = gammaOmega(omega);
  if (distance < EPSILON) return 0;
  return (gamma * gamma * C) / distance;
}

/**
 * GW frequency (dominant mode).
 * f_GW ~ 1/M ~ 1/|κ| -- smaller BHs ring at higher frequencies.
 * Returns dimensionless cyclic frequency.
 */
export function gwFrequency(kappa: number): number {
  const wd = Math.abs(kappa);
  if (wd < EPSILON) return 0;
  return 1.0 / (2 * Math.PI * wd);
}

/* --- Shapiro Time Delay ------------------------------------------ */

/**
 * Shapiro time delay: extra travel time for light passing near a mass.
 * Δt = 4GM/c^3 * ln(4r_1r_2/b^2) -> GCD analog: Δt ~ |κ| * ln(1/Δ + 1)
 * Larger well depth and smaller heterogeneity gap -> larger delay.
 */
export function shapiroDelay(kappa: number, delta: number): number {
  const wd = Math.abs(kappa);
  return 4 * wd * Math.log(1.0 / (delta + EPSILON) + 1);
}

/* --- Surface Gravity --------------------------------------------- */

/**
 * Surface gravity κ_s (not to be confused with log-integrity κ).
 * κ_s = dΓ/dω evaluated at the horizon candidate.
 * Connected to Hawking temperature: T_H = κ_s / (2pi).
 * This is the "gravitational acceleration" at the event horizon.
 */
export function surfaceGravity(omega: number): number {
  return dGamma(omega);
}

/* --- Photon Ring Structure --------------------------------------- */

/**
 * Photon ring demagnification factor for the n-th image.
 * Higher-order images (n=1, 2, 3...) are exponentially demagnified.
 * δ_n = exp(-2pin * sqrt(1/(ω+ε))) -- the Lyapunov exponent controls the ring spacing.
 *
 * n=0: direct image (primary)
 * n=1: first sub-ring (one extra half-orbit, ~exp(-pi) demagnified)
 * n=2: second sub-ring (two extra half-orbits, ~exp(-2pi) demagnified)
 */
export function photonRingDemag(n: number, omega: number): number {
  const lyapunov = Math.sqrt(1.0 / (omega + EPSILON));
  return Math.exp(-2 * Math.PI * n * lyapunov);
}

/**
 * Photon ring angular radius for the n-th sub-ring.
 * The critical impact parameter b_c = 3sqrt3 M (Schwarzschild).
 * GCD analog: b_c ~ 1/(dΓ/dω) at the photon sphere.
 * Each sub-ring converges: r_n = r_crit * (1 + δ_n)
 */
export function photonRingRadius(n: number, omega: number): number {
  const dg = dGamma(omega);
  if (dg < EPSILON) return 0;
  const rCrit = 3 * Math.sqrt(3) / dg; // critical impact parameter analog
  const demag = photonRingDemag(n, omega);
  return rCrit * (1 + demag);
}

/* --- Tidal Disruption -------------------------------------------- */

/**
 * Tidal disruption parameter: how strongly an extended object
 * is stretched by the differential gravity field.
 * T_tidal = d^2Γ/dω^2 * (size of object in ω-space)
 *
 * @param omega Position in drift space
 * @param objectSize Object's extent in ω-coordinates (default 0.01)
 */
export function tidalParameter(omega: number, objectSize: number = 0.01): number {
  return Math.abs(d2Gamma(omega)) * objectSize;
}

/**
 * Tidal disruption radius: the ω value where tidal forces exceed
 * the self-binding of an object (parameterized by its IC).
 * Objects with lower IC (less coherent) disrupt at larger radii (lower ω).
 * r_tidal ~ (M/m)^(1/3) -> ω_tidal = ω_h * (1 - IC^(1/3))
 */
export function tidalDisruptionOmega(objectIC: number): number {
  const omegaH = 1.0 - EPSILON; // horizon
  const icClamped = Math.max(objectIC, EPSILON);
  return omegaH * (1 - Math.pow(icClamped, 1.0 / 3.0));
}

/* --- Proper Time & Distance -------------------------------------- */

/**
 * Proper time dilation factor at drift omega.
 * dτ/dt = sqrt(1 - r_s/r) in Schwarzschild. We map Gamma(omega) to r_s/r.
 * Uses the Schwarzschild metric factor: sqrt(1 - 2M/r) where 2M/r ~ 2Gamma/(1+Gamma).
 * Approaches 0 at the horizon (time stops for a distant observer).
 */
export function properTimeFactor(omega: number): number {
  const gamma = gammaOmega(omega);
  // Map to metric coefficient: g_tt = 1 - r_s/r
  // At omega->1, gamma->inf, ratio->1, time freezes
  const ratio = 2 * gamma / (1 + gamma + gamma * gamma * 0.01);
  if (ratio >= 1.0) return 0;
  return Math.sqrt(1 - ratio);
}

/**
 * Coordinate velocity of radially infalling matter.
 * v_r = sqrt(2Γ/(1+Γ)) * (1 - 2Γ/(1+Γ))
 * Falls to zero at the horizon (frozen star appearance).
 */
export function coordinateFallSpeed(omega: number): number {
  const gamma = gammaOmega(omega);
  const ratio = 2 * gamma / (1 + gamma);
  if (ratio >= 1.0) return 0;
  return Math.sqrt(ratio) * (1 - ratio);
}

/**
 * Eddington luminosity analog.
 * L_Edd ~ M ~ |κ|. The maximum luminosity before radiation pressure
 * halts accretion. Systems above L_Edd are "super-Eddington."
 */
export function eddingtonLuminosity(kappa: number): number {
  return 4 * Math.PI * Math.abs(kappa);
}

/**
 * Spin-dependent ISCO radius in GCD units.
 * Maps the Kerr ISCO: r_ISCO(a*=0) = 6M (Schwarzschild),
 * r_ISCO(a*=1) = M (prograde extremal Kerr).
 * Returns the ω-coordinate of ISCO for given spin.
 */
export function iscoOmega(spinStar: number): number {
  // Bardeen formula approximation: r_ISCO/M goes from 6 (a*=0) to 1 (a*=1)
  // ω_ISCO scales inversely with r_ISCO
  const a = Math.min(Math.abs(spinStar), 1 - EPSILON);
  // Z1 and Z2 from Bardeen, Press & Teukolsky 1972
  const z1 = 1 + Math.pow(1 - a * a, 1.0/3.0) * (Math.pow(1 + a, 1.0/3.0) + Math.pow(1 - a, 1.0/3.0));
  const z2 = Math.sqrt(3 * a * a + z1 * z1);
  // r_ISCO/M for prograde orbit
  const rIsco = 3 + z2 - Math.sqrt((3 - z1) * (3 + z1 + 2 * z2));
  // Map to ω: higher rIsco -> lower ω (farther from horizon)
  return Math.min(0.98, 1.0 / (1.0 + rIsco / 6.0));
}

/**
 * Radiative efficiency: fraction of rest mass converted to radiation
 * as matter falls from the spin-dependent ISCO to the horizon.
 * eta = 1 - sqrt(1 - 2/(3 * r_ISCO/M))
 * eta_Schw ~= 0.057 (6%), eta_Kerr(a*=1) ~= 0.42 (42%).
 */
export function radiativeEfficiency(spinStar: number = 0): number {
  const a = Math.min(Math.abs(spinStar), 1 - EPSILON);
  // Use Bardeen ISCO to get exact efficiency
  const z1 = 1 + Math.pow(1 - a * a, 1.0/3.0) * (Math.pow(1 + a, 1.0/3.0) + Math.pow(1 - a, 1.0/3.0));
  const z2 = Math.sqrt(3 * a * a + z1 * z1);
  const rIsco = 3 + z2 - Math.sqrt((3 - z1) * (3 + z1 + 2 * z2));
  // Specific energy at ISCO: E_ISCO = sqrt(1 - 2/(3*r_ISCO))
  const eIsco = Math.sqrt(Math.max(EPSILON, 1 - 2.0 / (3.0 * rIsco)));
  return 1 - eIsco;
}

/* --- Black Hole Entities ----------------------------------------- */

export interface SpacetimeEntity {
  name: string;
  symbol: string;
  c: number[];
  w: number[];
  description: string;
  grAnalog: string;
}

/**
 * Curated black hole entities for the simulation.
 * Trace vectors from closures/spacetime_memory/*.py
 */
export const BLACK_HOLE_ENTITIES: SpacetimeEntity[] = [
  {
    name: 'Stellar Black Hole',
    symbol: 'BH',
    c: [0.99, 0.50, 0.99, 0.99, 0.60, 0.10, 0.95, 0.40],
    w: Array(8).fill(0.125),
    description: '~3-20 M(Sun). Endpoint of massive star collapse. One channel (c[5]=0.10) near floor -- the information channel collapses at the horizon.',
    grAnalog: 'Schwarzschild solution (non-rotating, uncharged)',
  },
  {
    name: 'Event Horizon',
    symbol: 'EH',
    c: [0.10, 0.02, 0.08, 0.02, 0.05, 0.05, 0.45, 0.65],
    w: Array(8).fill(0.125),
    description: 'The surface of no return. Almost all channels near floor -- maximum drift, minimum fidelity. The pole Γ(ω->1)->inf lives here.',
    grAnalog: 'r = 2GM/c^2 (Schwarzschild radius)',
  },
  {
    name: 'Photon Sphere',
    symbol: 'PS',
    c: [0.30, 0.15, 0.25, 0.12, 0.25, 0.20, 0.55, 0.70],
    w: Array(8).fill(0.125),
    description: 'r = 3GM/c^2. Light orbits but is unstable. Intermediate collapse -- all channels depressed but not at floor.',
    grAnalog: 'r = 1.5 * Schwarzschild radius',
  },
  {
    name: 'Binary BH Merger',
    symbol: 'BBH',
    c: [0.95, 0.85, 0.90, 0.80, 0.90, 0.70, 0.95, 0.95],
    w: Array(8).fill(0.125),
    description: 'Pre-merger inspiral. High coherence -- the gravitational wave signal carries nearly intact information. IC high.',
    grAnalog: 'LIGO/Virgo inspiral-merger-ringdown',
  },
  {
    name: 'Accretion Disk',
    symbol: 'AD',
    c: [0.75, 0.60, 0.70, 0.50, 0.65, 0.40, 0.80, 0.55],
    w: Array(8).fill(0.125),
    description: 'Matter spiraling inward. Intermediate fidelity -- energy extracted but structure partially preserved. Watch regime typical.',
    grAnalog: 'Shakura-Sunyaev thin disk / ADAF',
  },
  {
    name: 'Near-Horizon Loop',
    symbol: 'NHL',
    c: [0.20, 0.30, 0.10, 0.25, 0.15, 0.12, 0.90, 0.15],
    w: Array(8).fill(0.125),
    description: 'Causal loop at the stretched horizon. Most channels near floor, but c[6]=0.90 -- the circulation channel persists even at the boundary.',
    grAnalog: 'Stretched horizon / membrane paradigm',
  },
  {
    name: 'Kerr Black Hole',
    symbol: 'KBH',
    c: [0.95, 0.85, 0.95, 0.90, 0.70, 0.12, 0.98, 0.80],
    w: Array(8).fill(0.125),
    description: 'Rotating BH (a* ~= 0.998). Frame-dragging creates the ergosphere. Angular momentum channel elevated; information channel still collapses.',
    grAnalog: 'Kerr solution (rotating, uncharged)',
  },
  {
    name: 'Sagittarius A*',
    symbol: 'SgrA*',
    c: [0.992, 0.20, 0.985, 0.995, 0.80, 0.03, 0.99, 0.15],
    w: Array(8).fill(0.125),
    description: '4.15 * 10^6 M(Sun) supermassive BH at Milky Way center. Extreme mass but nearly all information lost. Deep well, very low IC.',
    grAnalog: 'Supermassive BH (EHT 2022 shadow image)',
  },
  {
    name: 'ISCO Boundary',
    symbol: 'ISCO',
    c: [0.50, 0.45, 0.48, 0.42, 0.40, 0.35, 0.70, 0.50],
    w: Array(8).fill(0.125),
    description: 'Innermost Stable Circular Orbit. The boundary between stable orbits and plunging trajectories. Matter crossing ISCO spirals into the horizon.',
    grAnalog: 'r = 6M (Schwarzschild) / r = M (extremal Kerr)',
  },
  {
    name: 'Ergosphere',
    symbol: 'ERGO',
    c: [0.42, 0.55, 0.38, 0.60, 0.85, 0.08, 0.92, 0.75],
    w: Array(8).fill(0.125),
    description: 'Region between static limit and event horizon of a Kerr BH. Frame-dragging forces all matter to co-rotate. Penrose process extracts energy here -- the angular momentum channel (c[4]=0.85) remains elevated.',
    grAnalog: 'Kerr ergosphere (r_+ < r < r_ergo)',
  },
  {
    name: 'Primordial Black Hole',
    symbol: 'PBH',
    c: [0.70, 0.60, 0.65, 0.55, 0.30, 0.50, 0.40, 0.80],
    w: Array(8).fill(0.125),
    description: '~10^1^5 g. Formed in early universe density fluctuations. Small mass -> high Hawking temperature -> actively evaporating. Information channel (c[5]=0.50) partially preserved -- possible dark matter candidate.',
    grAnalog: 'Planck-mass remnant / microscopic BH',
  },
  {
    name: 'Intermediate-Mass BH',
    symbol: 'IMBH',
    c: [0.97, 0.45, 0.96, 0.92, 0.65, 0.07, 0.97, 0.35],
    w: Array(8).fill(0.125),
    description: '~10^2-10^5 M(Sun). Elusive mass gap between stellar and supermassive. Possibly formed from runaway stellar mergers in dense clusters. Deep well but moderate IC -- bridging two regimes.',
    grAnalog: 'Possible seeds of supermassive BHs',
  },
  {
    name: 'Reissner-Nordstrom BH',
    symbol: 'RNBH',
    c: [0.88, 0.70, 0.92, 0.85, 0.10, 0.15, 0.85, 0.50],
    w: Array(8).fill(0.125),
    description: 'Charged, non-rotating BH. Two horizons: outer r_+ and inner Cauchy r_-. The charge channel (c[4]=0.10) is near floor -- EM self-energy collapses. IC severely reduced by charge-mass coupling.',
    grAnalog: 'Reissner-Nordstrom metric (Q != 0, J = 0)',
  },
];

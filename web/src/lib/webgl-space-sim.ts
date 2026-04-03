/**
 * WebGL Immersive Space Simulator — Ray-Traced Kerr Black Hole
 *
 * Full-screen ray-marching through curved Kerr spacetime:
 *   - Geodesic integration (Schwarzschild + frame-dragging correction)
 *   - Accretion disk with Novikov-Thorne temperature, Doppler beaming, gravitational redshift
 *   - Multiple disk images (light bending shows far side of disk over shadow)
 *   - Photon ring structure at shadow boundary
 *   - Procedural starfield with gravitational lensing
 *   - Particle overlay (relativistic jets, infalling matter, tidal streams)
 *   - Orbital camera with mouse look + scroll zoom
 *   - Real-time HUD with GCD kernel readouts
 *
 * All physics derived from GCD kernel — Tier-0 Protocol.
 * No Tier-1 symbol is redefined.
 */

import { gammaOmega, computeKernel, classifyRegime } from './kernel';
import {
  dGamma, hawkingTemperature, gravitationalRedshift,
  escapeVelocity, frameDragging, penroseEfficiency, bhEntropy,
  orbitalPrecession, gwStrain, surfaceGravity, properTimeFactor,
  radiativeEfficiency, OMEGA_ISCO, OMEGA_PHOTON_SPHERE,
  BLACK_HOLE_ENTITIES,
} from './spacetime';
import { EPSILON, P_EXPONENT } from './constants';

/* ===================================================================
   S1  LINEAR ALGEBRA (column-major 4×4)
   =================================================================== */

type Mat4 = Float32Array;
type Vec3 = [number, number, number];

function mat4(): Mat4 { return new Float32Array(16); }

function identity(): Mat4 {
  const m = mat4();
  m[0] = m[5] = m[10] = m[15] = 1;
  return m;
}

function perspective(fovy: number, aspect: number, near: number, far: number): Mat4 {
  const m = mat4();
  const f = 1.0 / Math.tan(fovy * 0.5);
  const nf = 1.0 / (near - far);
  m[0] = f / aspect;
  m[5] = f;
  m[10] = (far + near) * nf;
  m[11] = -1;
  m[14] = 2 * far * near * nf;
  return m;
}

function mul(a: Mat4, b: Mat4): Mat4 {
  const o = mat4();
  for (let i = 0; i < 4; i++)
    for (let j = 0; j < 4; j++)
      o[j * 4 + i] = a[i] * b[j * 4] + a[4 + i] * b[j * 4 + 1] +
                      a[8 + i] * b[j * 4 + 2] + a[12 + i] * b[j * 4 + 3];
  return o;
}

function rotX(a: number): Mat4 {
  const m = identity();
  const c = Math.cos(a), s = Math.sin(a);
  m[5] = c; m[6] = s; m[9] = -s; m[10] = c;
  return m;
}

function rotY(a: number): Mat4 {
  const m = identity();
  const c = Math.cos(a), s = Math.sin(a);
  m[0] = c; m[2] = -s; m[8] = s; m[10] = c;
  return m;
}

function translate(x: number, y: number, z: number): Mat4 {
  const m = identity();
  m[12] = x; m[13] = y; m[14] = z;
  return m;
}

function normalize(v: Vec3): Vec3 {
  const l = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) || 1;
  return [v[0] / l, v[1] / l, v[2] / l];
}

function cross(a: Vec3, b: Vec3): Vec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

function lookAt(eye: Vec3, target: Vec3, up: Vec3): Mat4 {
  const z = normalize([eye[0] - target[0], eye[1] - target[1], eye[2] - target[2]]);
  const x = normalize(cross(up, z));
  const y = cross(z, x);
  const m = identity();
  m[0] = x[0]; m[4] = x[1]; m[8]  = x[2]; m[12] = -(x[0]*eye[0]+x[1]*eye[1]+x[2]*eye[2]);
  m[1] = y[0]; m[5] = y[1]; m[9]  = y[2]; m[13] = -(y[0]*eye[0]+y[1]*eye[1]+y[2]*eye[2]);
  m[2] = z[0]; m[6] = z[1]; m[10] = z[2]; m[14] = -(z[0]*eye[0]+z[1]*eye[1]+z[2]*eye[2]);
  m[3] = 0;    m[7] = 0;    m[11] = 0;    m[15] = 1;
  return m;
}

/* ===================================================================
   S2  SHADERS — Ray-Traced Kerr Black Hole
   =================================================================== */

const RT_VERT = `
attribute vec2 aPos;
void main() {
  gl_Position = vec4(aPos, 0.0, 1.0);
}
`;

const RT_FRAG = `
precision highp float;

uniform vec2  uResolution;
uniform float uTime;
uniform float uSpinStar;
uniform vec3  uCamPos;
uniform vec3  uCamRight;
uniform vec3  uCamUp;
uniform vec3  uCamFwd;
uniform float uFovTan;
uniform float uGWStrain;

#define PI     3.14159265359
#define RS     1.0
#define M_BH   (RS * 0.5)
#define STEPS  200
#define FAR    80.0

// ── Hash & noise ──
float hash(vec2 p) {
  p = fract(p * vec2(123.34, 456.21));
  p += dot(p, p + 45.32);
  return fract(p.x * p.y);
}

float noise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  float a = hash(i);
  float b = hash(i + vec2(1.0, 0.0));
  float c = hash(i + vec2(0.0, 1.0));
  float d = hash(i + vec2(1.0, 1.0));
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

float fbm(vec2 p) {
  float v = 0.0, a = 0.5;
  for (int i = 0; i < 5; i++) {
    v += a * noise(p);
    p *= 2.1;
    a *= 0.48;
  }
  return v;
}

// ── Direction to equirectangular UV ──
vec2 dirToUV(vec3 d) {
  float lon = atan(d.z, d.x);
  float lat = asin(clamp(d.y, -1.0, 1.0));
  return vec2(lon / (2.0 * PI) + 0.5, lat / PI + 0.5);
}

// ── Procedural starfield from 3D direction ──
vec3 starfield(vec3 dir) {
  vec2 eq = dirToUV(dir);
  vec3 col = vec3(0.0);

  for (int layer = 0; layer < 5; layer++) {
    float scale = 80.0 + float(layer) * 120.0;
    float threshold = 0.968 + float(layer) * 0.004;
    vec2 id = floor(eq * scale);
    vec2 f  = fract(eq * scale) - 0.5;
    float h = hash(id + float(layer) * 100.0);
    if (h > threshold) {
      float brightness = (h - threshold) / (1.0 - threshold);
      float r = length(f);
      float star = smoothstep(0.08, 0.0, r) * brightness;
      if (layer < 2 && brightness > 0.5) {
        float spike = max(
          exp(-abs(f.x) * 40.0) * exp(-abs(f.y) * 250.0),
          exp(-abs(f.y) * 40.0) * exp(-abs(f.x) * 250.0)
        ) * brightness * 0.3;
        star += spike;
      }
      float temp = hash(id * 2.0 + 7.0);
      vec3 starCol = temp < 0.3
        ? mix(vec3(0.5, 0.65, 1.0), vec3(0.7, 0.85, 1.0), temp / 0.3)
        : temp < 0.6
          ? mix(vec3(0.9, 0.95, 1.0), vec3(1.0, 1.0, 0.9), (temp - 0.3) / 0.3)
          : temp < 0.8
            ? mix(vec3(1.0, 0.95, 0.7), vec3(1.0, 0.85, 0.5), (temp - 0.6) / 0.2)
            : vec3(1.0, 0.6, 0.3);
      float twinkle = 0.8 + 0.2 * sin(uTime * (1.0 + h * 2.0) + h * 100.0);
      col += starCol * star * twinkle;
    }
  }

  float neb  = fbm(eq * 8.0  + vec2(uTime * 0.002, 0.0));
  float neb2 = fbm(eq * 12.0 - vec2(0.0, uTime * 0.003));
  col += vec3(0.12, 0.03, 0.18) * neb  * 0.06;
  col += vec3(0.02, 0.08, 0.15) * neb2 * 0.04;

  float band = exp(-dir.y * dir.y * 8.0) * 0.025;
  col += vec3(0.15, 0.12, 0.10) * band;

  return col;
}

// ── Kerr ISCO (prograde, simulation coords) ──
float kerrISCO(float a) {
  float a2 = a * a;
  float z1 = 1.0 + pow(max(1.0 - a2, 1e-6), 1.0/3.0)
            * (pow(1.0 + a, 1.0/3.0) + pow(max(1.0 - a, 1e-4), 1.0/3.0));
  float z2 = sqrt(3.0 * a2 + z1 * z1);
  return (3.0 + z2 - sqrt(max((3.0 - z1) * (3.0 + z1 + 2.0 * z2), 0.0))) * M_BH;
}

// ── Blackbody color ──
vec3 blackbody(float T) {
  vec3 c;
  c.r = smoothstep(0.0,  0.33, T);
  c.g = smoothstep(0.15, 0.55, T) * (1.0 - smoothstep(1.2, 2.5, T) * 0.3);
  c.b = smoothstep(0.35, 0.85, T);
  c += vec3(0.15) * smoothstep(1.0, 2.0, T);
  return c;
}

// ── ACES filmic tone mapping ──
vec3 ACESFilm(vec3 x) {
  float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
  // ─ 1. Camera ray ─
  vec2 uv = (gl_FragCoord.xy - 0.5 * uResolution) / uResolution.y;
  vec3 rd = normalize(uCamRight * uv.x + uCamUp * uv.y + uCamFwd * uFovTan);
  vec3 pos = uCamPos;
  vec3 vel = rd;

  // ─ 2. Disk extent ─
  float diskInner = kerrISCO(uSpinStar);
  float diskOuter = diskInner * 5.5;

  // ─ 3. Accumulators ─
  vec3  color    = vec3(0.0);
  float alpha    = 0.0;
  bool  captured = false;
  float minDist  = FAR;

  // ─ 4. Ray march through curved spacetime ─
  for (int i = 0; i < STEPS; i++) {
    float r  = length(pos);
    minDist  = min(minDist, r);

    if (r < RS * 0.5) { captured = true; break; }
    if (r > FAR) break;
    if (alpha > 0.97) break;

    // Geodesic acceleration
    vec3  h_vec = cross(pos, vel);
    float h2    = dot(h_vec, h_vec);
    float r2    = r * r;
    float r5    = r2 * r2 * r;

    // Schwarzschild: a = -1.5 * h^2 * pos / r^5
    vec3 accel = -1.5 * h2 / r5 * pos;

    // Kerr frame-dragging
    vec3 rhat = pos / r;
    vec3 tang = cross(vec3(0.0, 1.0, 0.0), rhat);
    float tl  = length(tang);
    if (tl > 0.001) {
      tang /= tl;
      accel += tang * 2.0 * uSpinStar * M_BH * M_BH / (r2 * r + 0.01);
    }

    // GW perturbation
    float gwP = sin(r * 30.0 - uTime * 3.0) * sin(2.0 * atan(pos.z, pos.x));
    accel += rhat * gwP * uGWStrain * 0.004;

    // Adaptive step
    float h_near = clamp(0.1 * (r - RS * 0.4), 0.015, 0.5);
    float h_far  = 1.2;
    float step   = mix(h_near, h_far, smoothstep(3.5, 7.0, r));

    // Velocity-Verlet
    vec3 velHalf = vel + accel * step * 0.5;
    vec3 newPos  = pos + velHalf * step;

    float nr  = length(newPos);
    float nr5 = nr * nr * nr * nr * nr;
    vec3  nL  = cross(newPos, velHalf);
    float nL2 = dot(nL, nL);
    vec3  na  = -1.5 * nL2 / max(nr5, 1e-8) * newPos;
    vec3  nrh = newPos / max(nr, 1e-6);
    vec3  nt  = cross(vec3(0.0, 1.0, 0.0), nrh);
    float ntl = length(nt);
    if (ntl > 0.001) {
      nt /= ntl;
      na += nt * 2.0 * uSpinStar * M_BH * M_BH / (nr * nr * nr + 0.01);
    }
    vec3 newVel = vel + (accel + na) * step * 0.5;
    newVel = normalize(newVel);

    // Disk-plane crossing
    if (pos.y * newPos.y < 0.0 && alpha < 0.96) {
      float tCross = abs(pos.y) / (abs(pos.y) + abs(newPos.y) + 1e-6);
      vec3  crossP = mix(pos, newPos, tCross);
      float crossR = length(vec2(crossP.x, crossP.z));

      if (crossR > diskInner * 0.85 && crossR < diskOuter) {
        float phi = atan(crossP.z, crossP.x);

        // Novikov-Thorne temperature
        float rn = crossR / diskInner;
        float ntT = pow(rn, -0.75) * pow(max(1.0 - 1.0 / sqrt(rn), 0.001), 0.25);

        // Doppler beaming
        float vK   = sqrt(M_BH / crossR);
        vec3  vDir = normalize(cross(vec3(0.0, 1.0, 0.0),
                     normalize(vec3(crossP.x, 0.0, crossP.z))));
        float cosA  = dot(vDir, vel);
        float beta  = min(vK * 1.5, 0.92);
        float gamma = 1.0 / sqrt(max(1.0 - beta * beta, 0.01));
        float D     = 1.0 / (gamma * (1.0 - beta * cosA));
        float beam  = clamp(D * D * D, 0.08, 12.0);

        // Gravitational redshift
        float grs = sqrt(max(1.0 - RS / crossR, 0.001));

        // Frame-drag spiral + turbulence
        float dTw  = uSpinStar * 2.0 / (crossR * crossR + 0.5);
        float dPhi = phi + dTw * uTime * 0.3;
        float sp   = 0.5 + 0.30 * sin(dPhi * 3.0 - log(max(crossR, 0.01)) * 6.0
                                       + uTime * vK * 2.0)
                         + 0.15 * sin(dPhi * 5.0 - log(max(crossR, 0.01)) * 10.0
                                       + uTime * vK * 3.0 + 1.5);
        sp = clamp(sp, 0.3, 1.0);
        float turb = 0.7 + 0.2  * noise(vec2(dPhi * 8.0  + uTime * 0.3, crossR * 6.0))
                         + 0.10 * noise(vec2(dPhi * 25.0 + uTime * 0.8, crossR * 20.0));

        float T = ntT * beam * grs * grs;

        vec3  dCol = blackbody(T * 2.5);

        // ISCO stress glow
        float iscoG = exp(-pow((crossR - diskInner) / (diskInner * 0.12), 2.0)) * 2.5;
        dCol += vec3(1.0, 0.95, 0.88) * iscoG;

        // Doppler colour shift
        dCol *= vec3(1.0 + cosA * beta * 0.12,
                     1.0,
                     1.0 - cosA * beta * 0.12);

        // Edge softness
        float td = (crossR - diskInner) / (diskOuter - diskInner);
        float edge = smoothstep(0.0, 0.06, td) * smoothstep(1.0, 0.85, td);

        float intensity = T * sp * turb * edge * 1.8;
        vec3  emission  = dCol * intensity;
        float dA = clamp(intensity * 0.65, 0.0, 1.0);

        color += emission * (1.0 - alpha);
        alpha += dA * (1.0 - alpha);
      }
    }

    pos = newPos;
    vel = newVel;
  }

  // ─ 5. Final colour composition ─
  // Captured rays = pure black (no photon escapes the event horizon).
  // Only accumulated disk emission from crossings before capture is kept.
  vec3 finalColor = color;
  float photonR = 1.5 * RS;

  if (!captured) {
    // Background starfield (gravitationally lensed direction)
    vec3 bg = starfield(vel);
    finalColor += bg * (1.0 - alpha);

    // Photon-ring: bright ring at shadow edge from near-miss rays only
    float prGlow = exp(-pow((minDist - photonR) / (RS * 0.12), 2.0)) * 0.45;
    finalColor += vec3(1.0, 0.92, 0.55) * prGlow;

    // Einstein ring enhancement — rays deflected > pi accumulate extra brightness
    float erGlow = exp(-pow((minDist - photonR) / (RS * 0.04), 2.0)) * 0.25;
    finalColor += vec3(1.0, 0.97, 0.80) * erGlow;
  }

  // ACES tone mapping + gamma
  finalColor = ACESFilm(finalColor * 1.3);
  finalColor = pow(finalColor, vec3(1.0 / 2.2));

  gl_FragColor = vec4(finalColor, 1.0);
}
`;

// -- Particle shader (jets + infalling matter) --
const PART_VERT = `
attribute vec3 aPosition;
attribute vec3 aColor;
attribute float aSize;
uniform mat4 uMVP;
varying vec3 vColor;

void main() {
  gl_Position = uMVP * vec4(aPosition, 1.0);
  vColor = aColor;
  gl_PointSize = aSize;
}
`;

const PART_FRAG = `
precision mediump float;
varying vec3 vColor;

void main() {
  vec2 c = gl_PointCoord - 0.5;
  float r = length(c);
  if (r > 0.5) discard;
  float core  = exp(-r * r * 50.0);
  float halo  = exp(-r * r * 12.0);
  float outer = exp(-r * r *  3.0);
  float glow  = core * 0.5 + halo * 0.35 + outer * 0.15;
  vec3 coreCol  = mix(vColor, vec3(1.0, 0.97, 0.92), core * 0.6);
  vec3 fringeCol = coreCol + vec3(0.05, -0.02, 0.08) * halo * (1.0 - core);
  gl_FragColor = vec4(fringeCol * glow, glow * 0.9);
}
`;

/* ===================================================================
   S3  KERR ISCO (TypeScript)
   =================================================================== */

function computeKerrISCO(a: number): number {
  const a2 = a * a;
  const z1 = 1 + Math.cbrt(Math.max(1 - a2, 1e-12))
    * (Math.cbrt(1 + a) + Math.cbrt(Math.max(1 - a, 1e-6)));
  const z2 = Math.sqrt(3 * a2 + z1 * z1);
  return (3 + z2 - Math.sqrt(Math.max((3 - z1) * (3 + z1 + 2 * z2), 0))) * 0.5;
}

/* ===================================================================
   S4  PARTICLE SYSTEMS
   =================================================================== */

interface SpaceParticle {
  x: number; y: number; z: number;
  vx: number; vy: number; vz: number;
  life: number;
  maxLife: number;
  size: number;
  type: 'accretion' | 'jet' | 'infalling' | 'tidal' | 'penrose';
  cr: number; cg: number; cb: number;
}

function spawnParticle(type: SpaceParticle['type'], innerR: number, outerR: number): SpaceParticle {
  if (type === 'accretion') {
    const r = innerR + Math.random() * (outerR - innerR);
    const a = Math.random() * Math.PI * 2;
    const orbV = 1.0 / Math.sqrt(r);
    const ecc = 0.02 * Math.random();
    return {
      x: r * Math.cos(a) * (1 + ecc), y: (Math.random() - 0.5) * 0.04, z: r * Math.sin(a),
      vx: -orbV * Math.sin(a) * 0.3, vy: 0, vz: orbV * Math.cos(a) * 0.3,
      life: 0, maxLife: 4 + Math.random() * 6,
      size: 2 + Math.random() * 3,
      type, cr: 1, cg: 0.7 + Math.random() * 0.3, cb: 0.2 + Math.random() * 0.3,
    };
  } else if (type === 'jet') {
    const side = Math.random() > 0.5 ? 1 : -1;
    const spread = 0.08;
    const wobble = Math.sin(Date.now() * 0.001) * 0.03;
    return {
      x: (Math.random() - 0.5) * spread + wobble, y: side * innerR * 0.3,
      z: (Math.random() - 0.5) * spread,
      vx: (Math.random() - 0.5) * 0.04, vy: side * (1.8 + Math.random() * 0.8),
      vz: (Math.random() - 0.5) * 0.04,
      life: 0, maxLife: 1.5 + Math.random() * 2.5,
      size: 1.5 + Math.random() * 2,
      type, cr: 0.3 + Math.random() * 0.3, cg: 0.5 + Math.random() * 0.4, cb: 1.0,
    };
  } else if (type === 'tidal') {
    const a = Math.random() * Math.PI * 2;
    const r = innerR * (1.0 + Math.random() * 0.5);
    return {
      x: r * Math.cos(a), y: (Math.random() - 0.5) * 0.02, z: r * Math.sin(a),
      vx: -Math.cos(a) * 0.4, vy: (Math.random() - 0.5) * 0.1, vz: -Math.sin(a) * 0.4,
      life: 0, maxLife: 1.0 + Math.random() * 1.5,
      size: 1.0 + Math.random() * 1.5,
      type, cr: 1.0, cg: 0.3, cb: 0.1,
    };
  } else if (type === 'penrose') {
    const a = Math.random() * Math.PI * 2;
    const r = innerR * 1.1;
    const escaping = Math.random() > 0.5;
    return {
      x: r * Math.cos(a), y: (Math.random() - 0.5) * 0.05, z: r * Math.sin(a),
      vx: (escaping ? 1 : -1) * Math.cos(a) * 0.5,
      vy: escaping ? 0.3 : -0.1,
      vz: (escaping ? 1 : -1) * Math.sin(a) * 0.5,
      life: 0, maxLife: escaping ? 3.0 : 0.8,
      size: escaping ? 3.0 : 1.5,
      type,
      cr: escaping ? 0.2 : 0.8,
      cg: escaping ? 0.8 : 0.2,
      cb: escaping ? 1.0 : 0.3,
    };
  } else {
    const a = Math.random() * Math.PI * 2;
    const r = outerR * (1.2 + Math.random() * 0.5);
    const tangV = 0.08 * (Math.random() - 0.3);
    return {
      x: r * Math.cos(a), y: (Math.random() - 0.5) * 0.3, z: r * Math.sin(a),
      vx: -Math.cos(a) * 0.15 + tangV * (-Math.sin(a)),
      vy: 0,
      vz: -Math.sin(a) * 0.15 + tangV * Math.cos(a),
      life: 0, maxLife: 4 + Math.random() * 4,
      size: 1.5 + Math.random() * 2.5,
      type, cr: 0.9, cg: 0.5, cb: 0.2,
    };
  }
}

/* ===================================================================
   S5  HUD PANEL
   =================================================================== */

export interface HUDData {
  omega: number;
  gamma: number;
  regime: string;
  redshift: number;
  escapeV: number;
  hawkingT: number;
  distance: number;
  F: number;
  IC: number;
  kappa: number;
  S: number;
  C: number;
  delta: number;
  frameDrag: number;
  penroseEff: number;
  entropy: number;
  precession: number;
  gwStrainVal: number;
  surfGravity: number;
  timeFactor: number;
  radEfficiency: number;
  spinStar: number;
}

function computeHUD(cameraDistance: number, bhOmega: number, spin: number, curvature: number): HUDData {
  const maxDist = 20;
  const minOmega = 0.01;
  const observerOmega = Math.min(0.98, minOmega + (1 - Math.min(cameraDistance / maxDist, 1)) * (bhOmega - minOmega));

  const gamma = gammaOmega(observerOmega);
  const kr = computeKernel(BLACK_HOLE_ENTITIES[0].c, BLACK_HOLE_ENTITIES[0].w);
  const regime = classifyRegime(kr);

  return {
    omega: observerOmega,
    gamma,
    regime: regime.regime + (regime.isCritical ? ' (Critical)' : ''),
    redshift: gravitationalRedshift(observerOmega),
    escapeV: escapeVelocity(observerOmega),
    hawkingT: hawkingTemperature(kr.kappa),
    distance: cameraDistance,
    F: kr.F,
    IC: kr.IC,
    kappa: kr.kappa,
    S: kr.S,
    C: kr.C,
    delta: kr.delta,
    frameDrag: frameDragging(observerOmega, spin),
    penroseEff: penroseEfficiency(spin),
    entropy: bhEntropy(kr.kappa),
    precession: orbitalPrecession(observerOmega),
    gwStrainVal: gwStrain(bhOmega, curvature, cameraDistance),
    surfGravity: surfaceGravity(observerOmega),
    timeFactor: properTimeFactor(observerOmega),
    radEfficiency: radiativeEfficiency(spin),
    spinStar: spin,
  };
}

/* ===================================================================
   S6  MAIN INIT — Ray-Traced Renderer
   =================================================================== */

export interface SpaceSimControls {
  destroy: () => void;
  getHUD: () => HUDData;
}

export function initSpaceSim(
  canvas: HTMLCanvasElement,
  hudCallback?: (data: HUDData) => void,
): SpaceSimControls {
  const glCtx = canvas.getContext('webgl', {
    antialias: false, alpha: false, premultipliedAlpha: false,
    powerPreference: 'high-performance',
  });
  if (!glCtx) {
    console.error('WebGL not available');
    return { destroy: () => {}, getHUD: () => computeHUD(10, 0.95, 0.7, 0) };
  }
  const gl: WebGLRenderingContext = glCtx;

  gl.getExtension('OES_standard_derivatives');

  gl.enable(gl.DEPTH_TEST);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  gl.clearColor(0.0, 0.0, 0.0, 1.0);

  function compileShader(type: number, src: string): WebGLShader {
    const s = gl.createShader(type)!;
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS))
      console.error('Shader:', gl.getShaderInfoLog(s));
    return s;
  }
  function linkProgram(vs: string, fs: string): WebGLProgram {
    const p = gl.createProgram()!;
    gl.attachShader(p, compileShader(gl.VERTEX_SHADER, vs));
    gl.attachShader(p, compileShader(gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS))
      console.error('Link:', gl.getProgramInfoLog(p));
    return p;
  }

  // -- Programs (ray tracer + particles) --
  const rtProg   = linkProgram(RT_VERT, RT_FRAG);
  const partProg = linkProgram(PART_VERT, PART_FRAG);

  // -- Fullscreen quad --
  const quadVBO = gl.createBuffer()!;
  gl.bindBuffer(gl.ARRAY_BUFFER, quadVBO);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, 1,1]), gl.STATIC_DRAW);

  // -- BH params --
  const bhOmega = 0.95;
  const bhSpin  = 0.7;
  const bhC = BLACK_HOLE_ENTITIES[0].c.length > 3
    ? computeKernel(BLACK_HOLE_ENTITIES[0].c, BLACK_HOLE_ENTITIES[0].w).C : 0;

  // -- Disk extents for particles --
  const DISK_INNER = computeKerrISCO(bhSpin);
  const DISK_OUTER = DISK_INNER * 5.5;

  // -- Particles --
  const MAX_PARTICLES = 800;
  const particles: SpaceParticle[] = [];
  for (let i = 0; i < 450; i++) {
    const type: SpaceParticle['type'] = i < 240 ? 'accretion' : i < 340 ? 'jet' : 'infalling';
    const p = spawnParticle(type, DISK_INNER, DISK_OUTER);
    p.life = Math.random() * p.maxLife;
    particles.push(p);
  }
  const particleVBO = gl.createBuffer()!;

  // -- Camera --
  let azimuth = 0;
  let elevation = 0.35;
  let camDist = 8.0;
  let dragging = false;
  let lastX = 0, lastY = 0;
  let autoOrbit = true;
  let targetAzimuth = 0;
  let targetElevation = 0.35;
  let targetDist = 8.0;

  canvas.addEventListener('mousedown', (e) => {
    dragging = true; autoOrbit = false;
    lastX = e.clientX; lastY = e.clientY;
    canvas.style.cursor = 'grabbing';
  });
  canvas.addEventListener('mousemove', (e) => {
    if (!dragging) return;
    targetAzimuth += (e.clientX - lastX) * 0.005;
    targetElevation += (e.clientY - lastY) * 0.005;
    targetElevation = Math.max(-Math.PI * 0.45, Math.min(Math.PI * 0.45, targetElevation));
    lastX = e.clientX; lastY = e.clientY;
  });
  canvas.addEventListener('mouseup', () => { dragging = false; canvas.style.cursor = 'grab'; });
  canvas.addEventListener('mouseleave', () => { dragging = false; canvas.style.cursor = 'grab'; });

  canvas.addEventListener('touchstart', (e) => {
    dragging = true; autoOrbit = false;
    lastX = e.touches[0].clientX; lastY = e.touches[0].clientY;
    e.preventDefault();
  }, { passive: false });
  canvas.addEventListener('touchmove', (e) => {
    if (!dragging) return;
    targetAzimuth += (e.touches[0].clientX - lastX) * 0.005;
    targetElevation += (e.touches[0].clientY - lastY) * 0.005;
    targetElevation = Math.max(-Math.PI * 0.45, Math.min(Math.PI * 0.45, targetElevation));
    lastX = e.touches[0].clientX; lastY = e.touches[0].clientY;
    e.preventDefault();
  }, { passive: false });
  canvas.addEventListener('touchend', () => { dragging = false; });

  canvas.addEventListener('wheel', (e) => {
    targetDist += e.deltaY * 0.01;
    targetDist = Math.max(2.0, Math.min(25, targetDist));
    e.preventDefault();
  }, { passive: false });

  canvas.addEventListener('dblclick', () => {
    autoOrbit = true;
    targetElevation = 0.35;
    targetDist = 8.0;
  });

  canvas.style.cursor = 'grab';

  // -- Uniform locations --
  const rtLocs = {
    aPos:        gl.getAttribLocation(rtProg,  'aPos'),
    uResolution: gl.getUniformLocation(rtProg, 'uResolution'),
    uTime:       gl.getUniformLocation(rtProg, 'uTime'),
    uSpinStar:   gl.getUniformLocation(rtProg, 'uSpinStar'),
    uCamPos:     gl.getUniformLocation(rtProg, 'uCamPos'),
    uCamRight:   gl.getUniformLocation(rtProg, 'uCamRight'),
    uCamUp:      gl.getUniformLocation(rtProg, 'uCamUp'),
    uCamFwd:     gl.getUniformLocation(rtProg, 'uCamFwd'),
    uFovTan:     gl.getUniformLocation(rtProg, 'uFovTan'),
    uGWStrain:   gl.getUniformLocation(rtProg, 'uGWStrain'),
  };

  const partLocs = {
    aPosition: gl.getAttribLocation(partProg, 'aPosition'),
    aColor:    gl.getAttribLocation(partProg, 'aColor'),
    aSize:     gl.getAttribLocation(partProg, 'aSize'),
    uMVP:      gl.getUniformLocation(partProg, 'uMVP'),
  };

  // -- Resize (native DPR for max resolution) --
  const FOV = 0.9;
  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2.5);
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    canvas.width  = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    gl.viewport(0, 0, canvas.width, canvas.height);
  }
  resize();
  const resizeObs = new ResizeObserver(resize);
  resizeObs.observe(canvas);

  // -- HUD state --
  let hudData = computeHUD(camDist, bhOmega, bhSpin, bhC);

  // -- Animation --
  let running  = true;
  let lastTime = 0;

  function frame(now: number) {
    if (!running) return;
    const dt = Math.min((now - lastTime) / 1000, 0.05);
    lastTime = now;

    if (autoOrbit) targetAzimuth += dt * 0.08;
    azimuth   += (targetAzimuth   - azimuth)   * 0.05;
    elevation += (targetElevation - elevation) * 0.05;
    camDist   += (targetDist      - camDist)   * 0.05;

    const cx = camDist * Math.cos(elevation) * Math.sin(azimuth);
    const cy = camDist * Math.sin(elevation);
    const cz = camDist * Math.cos(elevation) * Math.cos(azimuth);

    // Camera direction vectors for ray construction
    const camPos: Vec3 = [cx, cy, cz];
    const fwd: Vec3 = normalize([-cx, -cy, -cz]);
    let worldUp: Vec3 = [0, 1, 0];
    if (Math.abs(fwd[1]) > 0.99) worldUp = [0, 0, 1];
    const right: Vec3 = normalize(cross(fwd, worldUp));
    const up: Vec3    = cross(right, fwd);

    const aspect = canvas.width / canvas.height;
    const proj   = perspective(FOV, aspect, 0.1, 100);
    const view   = lookAt(camPos, [0, 0, 0], [0, 1, 0]);
    const vp     = mul(proj, view);
    const t      = now / 1000;

    const currentGWStrain = Math.min(gwStrain(bhOmega, bhC, camDist), 1.0);

    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    // ═══ Pass 1: Ray-traced black hole ═══
    gl.depthMask(false);
    gl.disable(gl.DEPTH_TEST);
    gl.useProgram(rtProg);

    gl.bindBuffer(gl.ARRAY_BUFFER, quadVBO);
    gl.enableVertexAttribArray(rtLocs.aPos);
    gl.vertexAttribPointer(rtLocs.aPos, 2, gl.FLOAT, false, 0, 0);

    gl.uniform2f(rtLocs.uResolution, canvas.width, canvas.height);
    gl.uniform1f(rtLocs.uTime,       t);
    gl.uniform1f(rtLocs.uSpinStar,   bhSpin);
    gl.uniform3f(rtLocs.uCamPos,     camPos[0], camPos[1], camPos[2]);
    gl.uniform3f(rtLocs.uCamRight,   right[0],  right[1],  right[2]);
    gl.uniform3f(rtLocs.uCamUp,      up[0],     up[1],     up[2]);
    gl.uniform3f(rtLocs.uCamFwd,     fwd[0],    fwd[1],    fwd[2]);
    gl.uniform1f(rtLocs.uFovTan,     1.0 / Math.tan(FOV * 0.5));
    gl.uniform1f(rtLocs.uGWStrain,   currentGWStrain);

    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.disableVertexAttribArray(rtLocs.aPos);

    gl.depthMask(true);
    gl.enable(gl.DEPTH_TEST);

    // ═══ Pass 2: Particles ═══
    for (let i = particles.length - 1; i >= 0; i--) {
      const p = particles[i];
      p.life += dt;
      if (p.life > p.maxLife) {
        particles[i] = spawnParticle(p.type, DISK_INNER, DISK_OUTER);
        continue;
      }

      const r = Math.sqrt(p.x * p.x + p.z * p.z) || 0.1;

      if (p.type === 'accretion') {
        const orbSpeed = 1.0 / (r * Math.sqrt(r)) * 0.5;
        const ax = -p.z / r, az = p.x / r;
        const dragTorque = bhSpin * 0.005 / (r * r + 0.1);
        const dragInward = 0.008 / (r + 0.1);
        p.vx = ax * (orbSpeed + dragTorque) - p.x / r * dragInward;
        p.vz = az * (orbSpeed + dragTorque) - p.z / r * dragInward;
        if (r < DISK_INNER * 1.2) {
          const dimFactor = Math.max(0.1, (r - 0.2) / DISK_INNER);
          p.cr *= dimFactor; p.cg *= dimFactor; p.cb *= dimFactor;
        }
      } else if (p.type === 'jet') {
        p.vy *= 1.008;
        p.vx *= 0.99;
        p.vz *= 0.99;
      } else if (p.type === 'tidal') {
        const radialAccel = 0.15 / (r * r + 0.01);
        p.vx -= p.x / r * radialAccel * dt;
        p.vz -= p.z / r * radialAccel * dt;
        p.size *= 1.01;
        p.cr = Math.min(1.0, p.cr + dt * 0.3);
        p.cg = Math.min(1.0, p.cg + dt * 0.2);
        p.cb = Math.min(1.0, p.cb + dt * 0.15);
      } else if (p.type === 'penrose') {
        const radV = (p.vx * p.x + p.vz * p.z) / r;
        if (radV > 0) {
          p.vx *= 0.998;
          p.vz *= 0.998;
        } else {
          p.vx -= p.x / r * 0.1 * dt;
          p.vz -= p.z / r * 0.1 * dt;
        }
      } else {
        const gStrength = 0.05 / (r * r + 0.01);
        p.vx -= p.x / r * gStrength * dt;
        p.vz -= p.z / r * gStrength * dt;
        const dragV = bhSpin * 0.003 / (r * r + 0.1);
        p.vx += (-p.z / r) * dragV * dt;
        p.vz += (p.x / r) * dragV * dt;
      }

      p.x += p.vx * dt;
      p.y += p.vy * dt;
      p.z += p.vz * dt;

      const dist = Math.sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
      if (dist < 0.15 || dist > 20) {
        particles[i] = spawnParticle(p.type, DISK_INNER, DISK_OUTER);
      }
    }

    while (particles.length < MAX_PARTICLES) {
      const roll = Math.random();
      const type: SpaceParticle['type'] =
        roll < 0.40 ? 'accretion' :
        roll < 0.58 ? 'jet' :
        roll < 0.75 ? 'infalling' :
        roll < 0.88 ? 'tidal' : 'penrose';
      particles.push(spawnParticle(type, DISK_INNER, DISK_OUTER));
    }

    const pData = new Float32Array(particles.length * 7);
    for (let i = 0; i < particles.length; i++) {
      const p = particles[i];
      const fade = Math.min(1, Math.min(p.life / 0.3, (p.maxLife - p.life) / 0.5));
      const b = i * 7;
      pData[b]     = p.x;
      pData[b + 1] = p.y;
      pData[b + 2] = p.z;
      pData[b + 3] = p.cr * fade;
      pData[b + 4] = p.cg * fade;
      pData[b + 5] = p.cb * fade;
      pData[b + 6] = p.size * (0.5 + fade * 0.5);
    }

    gl.useProgram(partProg);
    gl.bindBuffer(gl.ARRAY_BUFFER, particleVBO);
    gl.bufferData(gl.ARRAY_BUFFER, pData, gl.DYNAMIC_DRAW);

    const pStride = 7 * 4;
    gl.enableVertexAttribArray(partLocs.aPosition);
    gl.vertexAttribPointer(partLocs.aPosition, 3, gl.FLOAT, false, pStride, 0);
    gl.enableVertexAttribArray(partLocs.aColor);
    gl.vertexAttribPointer(partLocs.aColor, 3, gl.FLOAT, false, pStride, 12);
    gl.enableVertexAttribArray(partLocs.aSize);
    gl.vertexAttribPointer(partLocs.aSize, 1, gl.FLOAT, false, pStride, 24);

    gl.uniformMatrix4fv(partLocs.uMVP, false, vp);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
    gl.drawArrays(gl.POINTS, 0, particles.length);

    gl.disableVertexAttribArray(partLocs.aPosition);
    gl.disableVertexAttribArray(partLocs.aColor);
    gl.disableVertexAttribArray(partLocs.aSize);

    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    // -- Update HUD --
    hudData = computeHUD(camDist, bhOmega, bhSpin, bhC);
    if (hudCallback) hudCallback(hudData);

    requestAnimationFrame(frame);
  }

  requestAnimationFrame((t) => { lastTime = t; frame(t); });

  return {
    destroy: () => {
      running = false;
      resizeObs.disconnect();
    },
    getHUD: () => hudData,
  };
}

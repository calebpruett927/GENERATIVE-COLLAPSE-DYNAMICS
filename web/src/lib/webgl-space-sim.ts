/**
 * WebGL Immersive Space Simulator -- ESA-style Black Hole Environment
 *
 * Full-screen 3D space simulation with:
 *   - Schwarzschild-like black hole (event horizon shadow)
 *   - Gravitational lensing distortion of background starfield
 *   - Accretion disk with Doppler beaming
 *   - Photon ring glow
 *   - Orbital camera with mouse look + scroll zoom
 *   - Real-time HUD with GCD kernel readouts (Γ, ω, regime, redshift, v_esc)
 *
 * All physics derived from GCD kernel -- Tier-0 Protocol.
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
   S1  LINEAR ALGEBRA (column-major 4*4)
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
   S2  SHADERS
   =================================================================== */

// -- Full-screen quad for space background + gravitational lensing --
const BG_VERT = `
attribute vec2 aPos;
varying vec2 vUV;
void main() {
  vUV = aPos * 0.5 + 0.5;
  gl_Position = vec4(aPos, 0.0, 1.0);
}
`;

const BG_FRAG = `
precision highp float;
varying vec2 vUV;
uniform vec2 uBHScreen;      // black hole center in screen [0,1]
uniform float uBHRadius;     // angular radius of event horizon
uniform float uLensStrength;  // gravitational lensing magnitude
uniform float uTime;
uniform float uSpinStar;     // dimensionless spin parameter a*
uniform float uGWStrain;     // gravitational wave strain

// Pseudo-random hash for star generation
float hash(vec2 p) {
  p = fract(p * vec2(123.34, 456.21));
  p += dot(p, p + 45.32);
  return fract(p.x * p.y);
}

// 2D simplex-like noise for nebula
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

// Fractal Brownian Motion for nebula clouds
float fbm(vec2 p) {
  float v = 0.0;
  float a = 0.5;
  for (int i = 0; i < 4; i++) {
    v += a * noise(p);
    p *= 2.1;
    a *= 0.5;
  }
  return v;
}

// Procedural starfield -- 5 layers for depth
vec3 starfield(vec2 uv) {
  vec3 col = vec3(0.0);
  // 5 layers at different scales and densities
  for (int layer = 0; layer < 5; layer++) {
    float scale = 60.0 + float(layer) * 90.0;
    float threshold = 0.965 + float(layer) * 0.005;
    vec2 id = floor(uv * scale);
    vec2 f = fract(uv * scale) - 0.5;
    float h = hash(id + float(layer) * 100.0);
    if (h > threshold) {
      float brightness = (h - threshold) / (1.0 - threshold);
      float r = length(f);
      // Point-spread function with diffraction spikes for bright stars
      float star = smoothstep(0.12, 0.0, r) * brightness;
      if (layer < 2 && brightness > 0.6) {
        // Diffraction spikes on brightest stars (4-point)
        float spike = max(
          exp(-abs(f.x) * 30.0) * exp(-abs(f.y) * 200.0),
          exp(-abs(f.y) * 30.0) * exp(-abs(f.x) * 200.0)
        ) * brightness * 0.4;
        star += spike;
      }
      // Star color temperature: blue-white to orange-red
      float temp = hash(id * 2.0 + 7.0);
      vec3 starCol = temp < 0.3
        ? mix(vec3(0.5, 0.6, 1.0), vec3(0.7, 0.8, 1.0), temp / 0.3)   // O/B blue
        : temp < 0.6
          ? mix(vec3(0.9, 0.95, 1.0), vec3(1.0, 1.0, 0.9), (temp - 0.3) / 0.3) // A/F white
          : temp < 0.8
            ? mix(vec3(1.0, 0.95, 0.7), vec3(1.0, 0.85, 0.5), (temp - 0.6) / 0.2)  // G/K yellow
            : vec3(1.0, 0.6, 0.3);  // M red
      // Twinkling
      float twinkle = 0.7 + 0.3 * sin(uTime * (1.0 + h * 3.0) + h * 100.0);
      col += starCol * star * twinkle;
    }
  }
  // Deep nebula layer (fBm-based)
  float neb = fbm(uv * 6.0 + vec2(uTime * 0.003, 0.0));
  float neb2 = fbm(uv * 10.0 - vec2(0.0, uTime * 0.005));
  col += vec3(0.12, 0.03, 0.18) * neb * 0.08;
  col += vec3(0.02, 0.08, 0.15) * neb2 * 0.05;
  return col;
}

void main() {
  vec2 uv = vUV;
  vec2 centered = uv - uBHScreen;
  float dist = length(centered);
  vec2 dir = centered / max(dist, 0.001);

  // -- Gravitational wave ripple distortion --
  float gwPhase = dist * 40.0 - uTime * 3.0;
  float gwRipple = sin(gwPhase) * uGWStrain * 0.02;
  vec2 gwOffset = dir * gwRipple;

  // -- Frame-dragging (Lense-Thirring): twist coordinates near BH --
  float dragAngle = uSpinStar * 0.8 / (dist * dist + 0.01);
  float cosD = cos(dragAngle);
  float sinD = sin(dragAngle);
  vec2 draggedCenter = vec2(
    centered.x * cosD - centered.y * sinD,
    centered.x * sinD + centered.y * cosD
  );
  // Blend: more twisting closer to BH
  float dragBlend = smoothstep(0.5, 0.05, dist);
  centered = mix(centered, draggedCenter, dragBlend) + gwOffset;
  dist = length(centered);
  dir = centered / max(dist, 0.001);

  // -- Gravitational lensing distortion --
  float rEinstein = sqrt(uLensStrength) * 0.15;
  // Higher-order deflection: point mass + spin correction
  float deflection = rEinstein * rEinstein / max(dist, 0.001);
  // Spin adds tangential deflection component
  float tangentialDefl = uSpinStar * deflection * 0.15;
  deflection = min(deflection, 0.5);
  tangentialDefl = min(tangentialDefl, 0.1);
  // Radial + tangential deflection
  vec2 tangent = vec2(-dir.y, dir.x);
  vec2 lensedUV = uv + dir * deflection + tangent * tangentialDefl;

  // -- Event horizon shadow --
  float horizonR = uBHRadius;
  float photonR = horizonR * 1.5;
  float shadowR = horizonR * 2.6;

  // -- Render starfield with lensing --
  vec3 col = starfield(lensedUV);

  // -- Photon ring sub-structure (n=0,1,2,3 sub-rings) --
  // Each successive sub-ring is exponentially demagnified and thinner
  // Physical: photons completing n half-orbits form nested ring images
  vec3 ringColor = vec3(1.0, 0.88, 0.45);
  for (int n = 0; n < 4; n++) {
    float nf = float(n);
    float demag = exp(-3.14159 * nf * 0.7);  // Lyapunov demagnification
    float subRingR = photonR * (1.0 + demag * 0.35);
    float subRingWidth = 0.0005 * photonR * photonR * demag;
    float subRingBright = 1.5 * demag;
    float rDist = abs(dist - subRingR);
    float ring = exp(-rDist * rDist / max(subRingWidth, 0.00001)) * subRingBright;
    // Sub-rings shift color: n=0 golden, n=1 warm white, n=2 blue-white, n=3 violet
    vec3 subCol = n == 0 ? ringColor :
                  n == 1 ? vec3(0.95, 0.90, 0.80) :
                  n == 2 ? vec3(0.7, 0.80, 1.0) :
                           vec3(0.55, 0.50, 0.95);
    col += subCol * ring;
  }

  // -- Einstein ring glow --
  float eRingDist = abs(dist - rEinstein);
  float eRing = exp(-eRingDist * eRingDist / (0.0008 * rEinstein * rEinstein)) * 0.4;
  col += vec3(0.5, 0.7, 1.0) * eRing;

  // -- Ergosphere shell (faint blue glow for spinning BH) --
  float ergoR = shadowR * (1.0 + sqrt(max(1.0 - uSpinStar * uSpinStar, 0.0)));
  float ergoDist = abs(dist - ergoR);
  float ergoGlow = exp(-ergoDist * ergoDist / (0.002 * ergoR * ergoR)) * uSpinStar * 0.25;
  col += vec3(0.2, 0.4, 0.9) * ergoGlow;

  // -- Event horizon shadow (soft penumbra) --
  // Physical: the shadow edge is NOT sharp -- photon orbits at r=3M
  // create a gradual dimming zone (penumbra) before total darkness
  float penumbraOuter = shadowR * 1.08;  // outer penumbra start
  float penumbraInner = shadowR * 0.82;  // inner shadow (fully dark)
  float penumbra = smoothstep(penumbraOuter, penumbraInner, dist);
  // Add gradient structure: scattered photons create faint ring zones
  float scatterRing = exp(-pow((dist - shadowR) / (shadowR * 0.06), 2.0)) * 0.15;
  col *= (1.0 - penumbra);
  col += vec3(0.6, 0.45, 0.25) * scatterRing * (1.0 - penumbra * 0.5);

  // -- Horizon edge glow (Hawking radiation + corona) --
  // Multi-layer corona: hot inner ring + warm outer halo
  float edgeGlow = exp(-(dist - shadowR) * (dist - shadowR) / (0.0008 * shadowR * shadowR));
  float coronaWide = exp(-(dist - shadowR) * (dist - shadowR) / (0.004 * shadowR * shadowR));
  col += vec3(0.9, 0.35, 0.08) * edgeGlow * 0.4;
  col += vec3(0.4, 0.15, 0.05) * coronaWide * 0.2;
  // Inner Cauchy horizon glow (for charged/spinning BHs) -- faint violet
  float innerHR = shadowR * 0.4;
  float innerGlow = exp(-(dist - innerHR) * (dist - innerHR) / (0.0005 * innerHR * innerHR));
  col += vec3(0.4, 0.1, 0.6) * innerGlow * uSpinStar * 0.15;

  // -- Gravitational wave + polarization pattern --
  float gwVis = sin(dist * 60.0 - uTime * 4.0) * uGWStrain * 0.15;
  float gwAngle = atan(centered.y, centered.x);
  float gwPattern = sin(2.0 * gwAngle) * gwVis; // quadrupole pattern
  col += vec3(0.3, 0.5, 0.8) * max(gwPattern, 0.0);

  gl_FragColor = vec4(col, 1.0);
}
`;

// -- Accretion disk shader --
const DISK_VERT = `
attribute vec3 aPosition;
attribute vec2 aTexCoord;
uniform mat4 uMVP;
varying vec2 vTexCoord;
varying vec3 vWorldPos;

void main() {
  gl_Position = uMVP * vec4(aPosition, 1.0);
  vTexCoord = aTexCoord;
  vWorldPos = aPosition;
}
`;

const DISK_FRAG = `
precision highp float;
varying vec2 vTexCoord;
varying vec3 vWorldPos;
uniform float uTime;
uniform float uInnerR;
uniform float uOuterR;
uniform float uSpinDisk;      // spin parameter for disk warping

// Hash-based noise
float hash2(vec2 p) {
  return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}
float noise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  return mix(
    mix(hash2(i), hash2(i + vec2(1.0, 0.0)), f.x),
    mix(hash2(i + vec2(0.0, 1.0)), hash2(i + vec2(1.0, 1.0)), f.x),
    f.y
  );
}
float fbmDisk(vec2 p) {
  float v = 0.0;
  float a = 0.5;
  for (int i = 0; i < 5; i++) {
    v += a * noise(p);
    p *= 2.3;
    a *= 0.45;
  }
  return v;
}

void main() {
  float r = length(vWorldPos.xz);
  float angle = atan(vWorldPos.z, vWorldPos.x);

  // Normalized radial position in disk
  float t = (r - uInnerR) / (uOuterR - uInnerR);
  t = clamp(t, 0.0, 1.0);

  // -- Novikov-Thorne temperature profile --
  // T(r) ~ r^{-3/4} * [1 - sqrt(r_ISCO/r)]^{1/4}
  float rNorm = max(r / uInnerR, 1.001);
  float ntFactor = pow(rNorm, -0.75) * pow(max(1.0 - 1.0 / sqrt(rNorm), 0.001), 0.25);
  float temp = ntFactor;

  // -- Keplerian velocity for Doppler beaming --
  float orbitalV = 1.0 / sqrt(max(r, 0.1));

  // -- Frame-dragging spiral: disk material co-rotates with BH --
  float dragTwist = uSpinDisk * 2.0 / (r * r + 0.5);
  float draggedAngle = angle + dragTwist * uTime * 0.3;

  // -- Multi-arm spiral structure (MHD instability analog) --
  float spiral1 = sin(draggedAngle * 3.0 - log(max(r, 0.01)) * 6.0 + uTime * orbitalV * 2.0);
  float spiral2 = sin(draggedAngle * 5.0 - log(max(r, 0.01)) * 10.0 + uTime * orbitalV * 3.0 + 1.5);
  float spiral = 0.5 + 0.3 * spiral1 + 0.15 * spiral2;
  spiral = clamp(spiral, 0.3, 1.0);

  // -- MHD turbulence (magneto-rotational instability) --
  float turb1 = fbmDisk(vec2(draggedAngle * 8.0 + uTime * 0.3, r * 6.0));
  float turb2 = noise(vec2(draggedAngle * 25.0 + uTime * 0.8, r * 20.0));
  float turb = 0.7 + 0.2 * turb1 + 0.1 * turb2;

  // -- ISCO stress edge: sharp brightness enhancement at inner edge --
  float iscoEdge = exp(-pow((r - uInnerR) / (uInnerR * 0.15), 2.0)) * 1.5;

  // -- Color: Novikov-Thorne temperature gradient --
  // Hottest inner: blue-white (T ~ 10^7 K for stellar BH)
  // Mid: yellow-orange
  // Outer: deep red -> infrared
  vec3 hotColor = vec3(0.85, 0.92, 1.0);     // blue-white
  vec3 warmColor = vec3(1.0, 0.75, 0.25);    // golden-yellow
  vec3 midColor = vec3(1.0, 0.45, 0.1);      // orange
  vec3 coolColor = vec3(0.7, 0.10, 0.03);    // deep red
  vec3 coldColor = vec3(0.3, 0.03, 0.01);    // near-IR

  vec3 diskColor;
  if (t < 0.1)       diskColor = mix(hotColor, warmColor, t / 0.1);
  else if (t < 0.3)  diskColor = mix(warmColor, midColor, (t - 0.1) / 0.2);
  else if (t < 0.6)  diskColor = mix(midColor, coolColor, (t - 0.3) / 0.3);
  else               diskColor = mix(coolColor, coldColor, (t - 0.6) / 0.4);

  // ISCO stress glow -- bright white line at inner edge
  diskColor += vec3(1.0, 0.9, 0.8) * iscoEdge;

  // -- Relativistic Doppler beaming --
  // Approaching side: blue-shifted + brighter (D^4 enhancement)
  // Receding side: red-shifted + dimmer
  float viewAngle = sin(draggedAngle + uTime * 0.2);
  float beta = orbitalV * 0.6;  // v/c fraction
  float dopplerFactor = 1.0 / (1.0 - beta * viewAngle);
  float beaming = dopplerFactor * dopplerFactor;  // D^2 approximation
  beaming = clamp(beaming, 0.3, 3.0);

  // Doppler color shift: approaching -> bluer, receding -> redder
  vec3 dopplerShift = vec3(
    1.0 - viewAngle * beta * 0.15,
    1.0,
    1.0 + viewAngle * beta * 0.15
  );
  diskColor *= dopplerShift;

  float brightness = temp * spiral * turb * beaming;

  // Edge softness (physical: inner = plunge region, outer = tidal truncation)
  float innerEdge = smoothstep(0.0, 0.04, t);
  float outerEdge = smoothstep(1.0, 0.88, t);
  brightness *= innerEdge * outerEdge;

  // Opacity: more opaque at inner edge (denser), thinner at outer edge
  float alpha = brightness * 0.9;
  alpha *= innerEdge * outerEdge;

  // Vertical thickness variation -- disk warping from spin
  float warp = abs(vWorldPos.y) / (0.05 + t * 0.1);
  alpha *= exp(-warp * warp);

  gl_FragColor = vec4(diskColor * brightness, alpha);
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
  // Multi-layer glow: bright core + soft halo + wide haze
  float core = exp(-r * r * 50.0);   // tight brilliant center
  float halo = exp(-r * r * 12.0);   // medium warm glow
  float outer = exp(-r * r * 3.0);   // wide soft haze
  float glow = core * 0.5 + halo * 0.35 + outer * 0.15;
  // Core whitening: hottest center approaches white
  vec3 coreColor = mix(vColor, vec3(1.0, 0.97, 0.92), core * 0.6);
  // Subtle chromatic fringe at halo edge
  vec3 fringeColor = coreColor + vec3(0.05, -0.02, 0.08) * halo * (1.0 - core);
  gl_FragColor = vec4(fringeColor * glow, glow * 0.9);
}
`;

/* ===================================================================
   S3  GEOMETRY GENERATORS
   =================================================================== */

// Accretion disk: flat annulus in the XZ plane
function generateDisk(innerR: number, outerR: number, segments: number, rings: number): {
  verts: Float32Array; indices: Uint16Array;
} {
  const count = (rings + 1) * segments;
  const verts = new Float32Array(count * 5); // xyz + uv
  let vi = 0;
  for (let ri = 0; ri <= rings; ri++) {
    const t = ri / rings;
    const r = innerR + t * (outerR - innerR);
    for (let si = 0; si < segments; si++) {
      const a = (si / segments) * Math.PI * 2;
      verts[vi++] = r * Math.cos(a);
      verts[vi++] = 0; // flat in XZ
      verts[vi++] = r * Math.sin(a);
      verts[vi++] = t;           // u = radial
      verts[vi++] = si / segments; // v = angular
    }
  }
  const idxCount = rings * segments * 6;
  const indices = new Uint16Array(idxCount);
  let ii = 0;
  for (let ri = 0; ri < rings; ri++) {
    for (let si = 0; si < segments; si++) {
      const c = ri * segments + si;
      const n = ri * segments + (si + 1) % segments;
      const a = (ri + 1) * segments + si;
      const an = (ri + 1) * segments + (si + 1) % segments;
      indices[ii++] = c; indices[ii++] = a; indices[ii++] = n;
      indices[ii++] = n; indices[ii++] = a; indices[ii++] = an;
    }
  }
  return { verts, indices };
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
    // Keplerian orbital velocity v ~ r^{-1/2}
    const orbV = 1.0 / Math.sqrt(r);
    // Add slight eccentricity for visual interest
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
    // Precessing jet -- slight wobble
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
    // Tidally disrupted matter -- stretched along radial direction
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
    // Penrose process: particle splitting near ergosphere
    // One piece escapes with MORE energy, the other falls in
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
    // Infalling -- with angular momentum
    const a = Math.random() * Math.PI * 2;
    const r = outerR * (1.2 + Math.random() * 0.5);
    const tangV = 0.08 * (Math.random() - 0.3); // slight angular momentum
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
  // Advanced physics
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
    // Advanced physics
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
   S6  MAIN INIT
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
    antialias: true, alpha: false, premultipliedAlpha: false,
  });
  if (!glCtx) {
    console.error('WebGL not available');
    return { destroy: () => {}, getHUD: () => computeHUD(10, 0.95, 0.7, 0) };
  }
  const gl: WebGLRenderingContext = glCtx;

  // -- Extensions --
  gl.getExtension('OES_standard_derivatives');

  gl.enable(gl.DEPTH_TEST);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  gl.clearColor(0.0, 0.0, 0.0, 1.0);

  // -- Compile helpers --
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

  // -- Programs --
  const bgProg = linkProgram(BG_VERT, BG_FRAG);
  const diskProg = linkProgram(DISK_VERT, DISK_FRAG);
  const partProg = linkProgram(PART_VERT, PART_FRAG);

  // -- Background quad --
  const quadVBO = gl.createBuffer()!;
  gl.bindBuffer(gl.ARRAY_BUFFER, quadVBO);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, 1,1]), gl.STATIC_DRAW);

  // -- Accretion disk geometry -- high resolution --
  const DISK_INNER = 0.6;
  const DISK_OUTER = 3.0;
  const disk = generateDisk(DISK_INNER, DISK_OUTER, 192, 48);

  const diskVBO = gl.createBuffer()!;
  gl.bindBuffer(gl.ARRAY_BUFFER, diskVBO);
  gl.bufferData(gl.ARRAY_BUFFER, disk.verts, gl.STATIC_DRAW);

  const diskIBO = gl.createBuffer()!;
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, diskIBO);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, disk.indices, gl.STATIC_DRAW);

  // -- Particles -- high density for realism --
  const MAX_PARTICLES = 600;
  const particles: SpaceParticle[] = [];
  // Seed initial particles
  for (let i = 0; i < 350; i++) {
    const type: SpaceParticle['type'] = i < 200 ? 'accretion' : i < 270 ? 'jet' : 'infalling';
    const p = spawnParticle(type, DISK_INNER, DISK_OUTER);
    p.life = Math.random() * p.maxLife; // stagger
    particles.push(p);
  }
  const particleVBO = gl.createBuffer()!;

  // -- Camera --
  let azimuth = 0;
  let elevation = 0.35; // slightly above disk plane
  let camDist = 8.0;
  let dragging = false;
  let lastX = 0, lastY = 0;
  let autoOrbit = true;
  let targetAzimuth = 0;
  let targetElevation = 0.35;
  let targetDist = 8.0;

  // Mouse/touch controls
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

  // -- BH params --
  const bhOmega = 0.95; // deep collapse -- near horizon
  const bhSpin = 0.7;   // moderate spin (a* = 0.7)
  const bhC = BLACK_HOLE_ENTITIES[0].c.length > 3
    ? computeKernel(BLACK_HOLE_ENTITIES[0].c, BLACK_HOLE_ENTITIES[0].w).C : 0;

  // -- Uniform locations --
  // Background
  const bgLocs = {
    aPos: gl.getAttribLocation(bgProg, 'aPos'),
    uBHScreen: gl.getUniformLocation(bgProg, 'uBHScreen'),
    uBHRadius: gl.getUniformLocation(bgProg, 'uBHRadius'),
    uLensStrength: gl.getUniformLocation(bgProg, 'uLensStrength'),
    uTime: gl.getUniformLocation(bgProg, 'uTime'),
    uSpinStar: gl.getUniformLocation(bgProg, 'uSpinStar'),
    uGWStrain: gl.getUniformLocation(bgProg, 'uGWStrain'),
  };
  // Disk
  const diskLocs = {
    aPosition: gl.getAttribLocation(diskProg, 'aPosition'),
    aTexCoord: gl.getAttribLocation(diskProg, 'aTexCoord'),
    uMVP: gl.getUniformLocation(diskProg, 'uMVP'),
    uTime: gl.getUniformLocation(diskProg, 'uTime'),
    uInnerR: gl.getUniformLocation(diskProg, 'uInnerR'),
    uOuterR: gl.getUniformLocation(diskProg, 'uOuterR'),
    uSpinDisk: gl.getUniformLocation(diskProg, 'uSpinDisk'),
  };
  // Particles
  const partLocs = {
    aPosition: gl.getAttribLocation(partProg, 'aPosition'),
    aColor: gl.getAttribLocation(partProg, 'aColor'),
    aSize: gl.getAttribLocation(partProg, 'aSize'),
    uMVP: gl.getUniformLocation(partProg, 'uMVP'),
  };

  // -- Resize handler --
  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    gl.viewport(0, 0, canvas.width, canvas.height);
  }
  resize();
  const resizeObs = new ResizeObserver(resize);
  resizeObs.observe(canvas);

  // -- HUD state --
  let hudData = computeHUD(camDist, bhOmega, bhSpin, bhC);

  // -- Animation --
  let running = true;
  let lastTime = 0;

  function frame(now: number) {
    if (!running) return;
    const dt = Math.min((now - lastTime) / 1000, 0.05);
    lastTime = now;

    // Smooth camera interpolation
    if (autoOrbit) targetAzimuth += dt * 0.08;
    azimuth += (targetAzimuth - azimuth) * 0.05;
    elevation += (targetElevation - elevation) * 0.05;
    camDist += (targetDist - camDist) * 0.05;

    const cx = camDist * Math.cos(elevation) * Math.sin(azimuth);
    const cy = camDist * Math.sin(elevation);
    const cz = camDist * Math.cos(elevation) * Math.cos(azimuth);

    const aspect = canvas.width / canvas.height;
    const proj = perspective(0.9, aspect, 0.1, 100);
    const view = lookAt([cx, cy, cz], [0, 0, 0], [0, 1, 0]);
    const vp = mul(proj, view);

    // -- Project BH center to screen --
    // BH is at origin; transform [0,0,0,1] through VP
    const clipX = vp[12] / vp[15];
    const clipY = vp[13] / vp[15];
    const screenBH: [number, number] = [clipX * 0.5 + 0.5, clipY * 0.5 + 0.5];

    // Apparent angular size of BH based on distance
    const bhAngularR = Math.atan(0.5 / camDist) / (0.9 / 2); // normalized to FOV
    // Lens strength varies with mass (|κ|) and distance
    const lensStrength = gammaOmega(bhOmega) / (camDist * 0.5);

    const t = now / 1000;

    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    // === Pass 1: Background + Lensing ===
    gl.depthMask(false);
    gl.disable(gl.DEPTH_TEST);
    gl.useProgram(bgProg);

    gl.bindBuffer(gl.ARRAY_BUFFER, quadVBO);
    gl.enableVertexAttribArray(bgLocs.aPos);
    gl.vertexAttribPointer(bgLocs.aPos, 2, gl.FLOAT, false, 0, 0);

    gl.uniform2fv(bgLocs.uBHScreen, screenBH);
    gl.uniform1f(bgLocs.uBHRadius, bhAngularR);
    gl.uniform1f(bgLocs.uLensStrength, lensStrength);
    gl.uniform1f(bgLocs.uTime, t);
    gl.uniform1f(bgLocs.uSpinStar, bhSpin);
    // GW strain varies with observer distance and BH curvature
    const currentGWStrain = gwStrain(bhOmega, bhC, camDist);
    gl.uniform1f(bgLocs.uGWStrain, Math.min(currentGWStrain, 1.0));

    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.disableVertexAttribArray(bgLocs.aPos);

    gl.depthMask(true);
    gl.enable(gl.DEPTH_TEST);

    // === Pass 2: Accretion Disk ===
    // Tilt disk slightly for visual interest
    const diskTilt = rotX(0.12);
    const diskMVP = mul(vp, diskTilt);

    gl.useProgram(diskProg);
    gl.bindBuffer(gl.ARRAY_BUFFER, diskVBO);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, diskIBO);

    const stride = 5 * 4;
    gl.enableVertexAttribArray(diskLocs.aPosition);
    gl.vertexAttribPointer(diskLocs.aPosition, 3, gl.FLOAT, false, stride, 0);
    if (diskLocs.aTexCoord >= 0) {
      gl.enableVertexAttribArray(diskLocs.aTexCoord);
      gl.vertexAttribPointer(diskLocs.aTexCoord, 2, gl.FLOAT, false, stride, 12);
    }

    gl.uniformMatrix4fv(diskLocs.uMVP, false, diskMVP);
    gl.uniform1f(diskLocs.uTime, t);
    gl.uniform1f(diskLocs.uInnerR, DISK_INNER);
    gl.uniform1f(diskLocs.uOuterR, DISK_OUTER);
    gl.uniform1f(diskLocs.uSpinDisk, bhSpin);

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.drawElements(gl.TRIANGLES, disk.indices.length, gl.UNSIGNED_SHORT, 0);

    gl.disableVertexAttribArray(diskLocs.aPosition);
    if (diskLocs.aTexCoord >= 0) gl.disableVertexAttribArray(diskLocs.aTexCoord);

    // === Pass 3: Particles ===
    // Update particles with GR-aware physics
    for (let i = particles.length - 1; i >= 0; i--) {
      const p = particles[i];
      p.life += dt;
      if (p.life > p.maxLife) {
        particles[i] = spawnParticle(p.type, DISK_INNER, DISK_OUTER);
        continue;
      }

      const r = Math.sqrt(p.x * p.x + p.z * p.z) || 0.1;

      if (p.type === 'accretion') {
        // Keplerian orbit + radiation drag inspiral + frame-dragging torque
        const orbSpeed = 1.0 / (r * Math.sqrt(r)) * 0.5;
        const ax = -p.z / r, az = p.x / r;
        // Frame drag: co-rotate in direction of BH spin
        const dragTorque = bhSpin * 0.005 / (r * r + 0.1);
        // Radiation drag: slow inspiral
        const dragInward = 0.008 / (r + 0.1);
        p.vx = ax * (orbSpeed + dragTorque) - p.x / r * dragInward;
        p.vz = az * (orbSpeed + dragTorque) - p.z / r * dragInward;
        // Gravitational redshift dims particle near horizon
        if (r < DISK_INNER * 1.2) {
          const dimFactor = Math.max(0.1, (r - 0.2) / DISK_INNER);
          p.cr *= dimFactor; p.cg *= dimFactor; p.cb *= dimFactor;
        }
      } else if (p.type === 'jet') {
        // Relativistic jet: accelerate + collimate
        p.vy *= 1.008;
        // Magnetic collimation -- push toward axis
        p.vx *= 0.99;
        p.vz *= 0.99;
      } else if (p.type === 'tidal') {
        // Tidal spaghettification: accelerate radially, stretch along infall
        const radialAccel = 0.15 / (r * r + 0.01);
        p.vx -= p.x / r * radialAccel * dt;
        p.vz -= p.z / r * radialAccel * dt;
        // Stretch the particle (size increases along motion direction)
        p.size *= 1.01;
        // Heating: turns white-hot as it falls
        p.cr = Math.min(1.0, p.cr + dt * 0.3);
        p.cg = Math.min(1.0, p.cg + dt * 0.2);
        p.cb = Math.min(1.0, p.cb + dt * 0.15);
      } else if (p.type === 'penrose') {
        // Penrose particle: if escaping, decelerate but escape; if falling, accelerate inward
        const radV = (p.vx * p.x + p.vz * p.z) / r;
        if (radV > 0) {
          // Escaping: slow deceleration (it has extracted BH energy)
          p.vx *= 0.998;
          p.vz *= 0.998;
        } else {
          // Falling: strong inward acceleration
          p.vx -= p.x / r * 0.1 * dt;
          p.vz -= p.z / r * 0.1 * dt;
        }
      } else {
        // Infalling: GR-corrected gravity (stronger near horizon)
        const gStrength = 0.05 / (r * r + 0.01);
        p.vx -= p.x / r * gStrength * dt;
        p.vz -= p.z / r * gStrength * dt;
        // Frame-drag: add tangential velocity component
        const dragV = bhSpin * 0.003 / (r * r + 0.1);
        p.vx += (-p.z / r) * dragV * dt;
        p.vz += (p.x / r) * dragV * dt;
      }

      p.x += p.vx * dt;
      p.y += p.vy * dt;
      p.z += p.vz * dt;

      // Respawn if too far or absorbed
      const dist = Math.sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
      if (dist < 0.15 || dist > 18) {
        particles[i] = spawnParticle(p.type, DISK_INNER, DISK_OUTER);
      }
    }

    // Spawn new particles with type distribution
    while (particles.length < MAX_PARTICLES) {
      const roll = Math.random();
      const type: SpaceParticle['type'] =
        roll < 0.40 ? 'accretion' :
        roll < 0.58 ? 'jet' :
        roll < 0.75 ? 'infalling' :
        roll < 0.88 ? 'tidal' : 'penrose';
      particles.push(spawnParticle(type, DISK_INNER, DISK_OUTER));
    }

    // Upload particle data
    const pData = new Float32Array(particles.length * 7); // xyz rgb size
    for (let i = 0; i < particles.length; i++) {
      const p = particles[i];
      const fade = Math.min(1, Math.min(p.life / 0.3, (p.maxLife - p.life) / 0.5));
      const b = i * 7;
      pData[b] = p.x; pData[b + 1] = p.y; pData[b + 2] = p.z;
      pData[b + 3] = p.cr * fade; pData[b + 4] = p.cg * fade; pData[b + 5] = p.cb * fade;
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

    // Reset blend mode
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

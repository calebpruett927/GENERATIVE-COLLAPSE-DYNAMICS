/**
 * WebGL 3D Black Hole Gravity Well Renderer
 *
 * Renders the GCD budget surface Γ(ω) as a 3D gravity funnel.
 * The event horizon (ω → 1) is at the center/bottom of the well.
 * Regime coloring: Stable (teal) → Watch (amber) → Collapse (red) → Horizon (dark)
 *
 * Tier-0 Protocol: no Tier-1 symbol is redefined.
 */

import { gammaOmega, computeKernel } from './kernel';
import { BLACK_HOLE_ENTITIES } from './spacetime';

/* ─── Matrix Utilities (column-major 4×4) ───────────────────────── */

type Mat4 = Float32Array;

function mat4Create(): Mat4 { return new Float32Array(16); }

function mat4Identity(): Mat4 {
  const m = mat4Create();
  m[0] = m[5] = m[10] = m[15] = 1;
  return m;
}

function mat4Perspective(fovy: number, aspect: number, near: number, far: number): Mat4 {
  const m = mat4Create();
  const f = 1.0 / Math.tan(fovy * 0.5);
  const nf = 1.0 / (near - far);
  m[0] = f / aspect;
  m[5] = f;
  m[10] = (far + near) * nf;
  m[11] = -1;
  m[14] = 2 * far * near * nf;
  return m;
}

function mat4Multiply(out: Mat4, a: Mat4, b: Mat4): Mat4 {
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      out[j * 4 + i] =
        a[i] * b[j * 4] + a[4 + i] * b[j * 4 + 1] +
        a[8 + i] * b[j * 4 + 2] + a[12 + i] * b[j * 4 + 3];
    }
  }
  return out;
}

function mat4RotateX(angle: number): Mat4 {
  const m = mat4Identity();
  const c = Math.cos(angle), s = Math.sin(angle);
  m[5] = c; m[6] = s; m[9] = -s; m[10] = c;
  return m;
}

function mat4RotateY(angle: number): Mat4 {
  const m = mat4Identity();
  const c = Math.cos(angle), s = Math.sin(angle);
  m[0] = c; m[2] = -s; m[8] = s; m[10] = c;
  return m;
}

function mat4Translate(x: number, y: number, z: number): Mat4 {
  const m = mat4Identity();
  m[12] = x; m[13] = y; m[14] = z;
  return m;
}

/* ─── Shader Sources ────────────────────────────────────────────── */

const VERT_SRC = `
attribute vec3 aPosition;
attribute vec3 aColor;
attribute vec3 aNormal;
uniform mat4 uMVP;
uniform mat4 uModelView;
varying vec3 vColor;
varying vec3 vNormal;
varying vec3 vWorldPos;
varying float vDepth;

void main() {
  gl_Position = uMVP * vec4(aPosition, 1.0);
  vColor = aColor;
  vNormal = mat3(uModelView) * aNormal;
  vWorldPos = aPosition;
  vDepth = -aPosition.y;
  gl_PointSize = 8.0;
}
`;

const FRAG_SRC = `
precision mediump float;
varying vec3 vColor;
varying vec3 vNormal;
varying vec3 vWorldPos;
varying float vDepth;

void main() {
  // Directional + ambient lighting
  vec3 lightDir = normalize(vec3(0.3, 1.0, 0.5));
  vec3 normal = normalize(vNormal);
  float diff = max(dot(normal, lightDir), 0.0) * 0.55;
  float ambient = 0.4;
  vec3 lit = vColor * (ambient + diff);

  // Blinn-Phong specular highlights
  vec3 viewDir = normalize(vec3(0.0, 2.0, 3.0) - vWorldPos);
  vec3 halfDir = normalize(lightDir + viewDir);
  float spec = pow(max(dot(normal, halfDir), 0.0), 32.0) * 0.35;
  lit += vec3(0.7, 0.85, 1.0) * spec;

  // Concentric rings (depth-based) -- thinner, crisper
  float ringPattern = fract(vDepth * 2.5);
  float ring = smoothstep(0.92, 0.96, ringPattern);
  lit = mix(lit, lit * 1.3 + vec3(0.03), ring * 0.25);

  // Radial grid lines (every 30 deg)
  float angle = atan(vWorldPos.z, vWorldPos.x);
  float angularPattern = fract(angle * 1.9099); // ~12 lines
  float angularLine = smoothstep(0.93, 0.97, angularPattern);
  lit = mix(lit, lit * 1.2 + vec3(0.02), angularLine * 0.2);

  // Depth darkening toward singularity
  float depthFade = 1.0 - clamp(vDepth * 0.06, 0.0, 0.65);
  lit *= depthFade;

  // Rim/Fresnel glow: edges of the well catch more light
  float rimDot = 1.0 - max(dot(normal, viewDir), 0.0);
  float rim = pow(rimDot, 3.0) * 0.4;
  // Rim color shifts from cool blue at top to hot orange at depth
  vec3 rimColor = mix(vec3(0.3, 0.6, 1.0), vec3(1.0, 0.4, 0.1), clamp(vDepth * 0.15, 0.0, 1.0));
  lit += rimColor * rim;

  // Event horizon inner glow: faint pulsing warmth at deepest point
  float horizonGlow = exp(-vDepth * 0.25) * 0.0 + smoothstep(4.0, 6.0, vDepth) * 0.3;
  lit += vec3(0.8, 0.2, 0.05) * horizonGlow;

  gl_FragColor = vec4(lit, 1.0);
}
`;

/* ─── Particle Shader ───────────────────────────────────────────── */

const PARTICLE_VERT = `
attribute vec3 aPosition;
attribute vec3 aColor;
uniform mat4 uMVP;
varying vec3 vColor;
varying float vAlpha;

void main() {
  gl_Position = uMVP * vec4(aPosition, 1.0);
  vColor = aColor;
  float dist = length(aPosition.xz);
  gl_PointSize = max(2.0, 6.0 - dist * 1.5);
  vAlpha = clamp(0.3 + dist * 0.3, 0.2, 0.9);
}
`;

const PARTICLE_FRAG = `
precision mediump float;
varying vec3 vColor;
varying float vAlpha;

void main() {
  vec2 coord = gl_PointCoord - vec2(0.5);
  float r = length(coord);
  if (r > 0.5) discard;
  // Multi-layer: bright core + warm halo + soft outer
  float core = exp(-r * r * 40.0);
  float halo = exp(-r * r * 10.0);
  float outer = exp(-r * r * 3.0);
  float glow = core * 0.5 + halo * 0.35 + outer * 0.15;
  // Core whitening for hot center
  vec3 col = mix(vColor, vec3(1.0, 0.95, 0.85), core * 0.5);
  gl_FragColor = vec4(col * glow, vAlpha * glow);
}
`;

/* ─── Color Utilities ───────────────────────────────────────────── */

function regimeColorForOmega(omega: number): [number, number, number] {
  if (omega < 0.038) {
    const t = omega / 0.038;
    return [0.1 + t * 0.1, 0.7 - t * 0.1, 0.6 + t * 0.1];
  } else if (omega < 0.30) {
    const t = (omega - 0.038) / (0.30 - 0.038);
    return [0.2 + t * 0.78, 0.6 - t * 0.05, 0.7 - t * 0.55];
  } else if (omega < 0.80) {
    const t = (omega - 0.30) / 0.50;
    return [1.0, 0.55 - t * 0.4, 0.15 - t * 0.1];
  } else {
    const t = (omega - 0.80) / 0.20;
    return [1.0 - t * 0.7, 0.15 - t * 0.12, 0.05 - t * 0.04];
  }
}

/* ─── Mesh Generation ───────────────────────────────────────────── */

interface MeshData {
  vertices: Float32Array;
  indices: Uint16Array;
  entityMarkers: Float32Array;
  entityCount: number;
}

function generateGravityWell(radialSteps: number, angularSteps: number): MeshData {
  const vertCount = (radialSteps + 1) * angularSteps;
  const vertices = new Float32Array(vertCount * 9);
  const R_MAX = 2.8;
  const DEPTH_SCALE = 0.45;
  const MAX_DEPTH = 6.0;

  let vi = 0;
  for (let ri = 0; ri <= radialSteps; ri++) {
    const rFrac = ri / radialSteps;
    const omega = 1.0 - Math.max(rFrac, 0.005) * 0.998;
    const screenR = rFrac * R_MAX;

    const gamma = gammaOmega(omega);
    const rawHeight = -Math.log(1 + gamma) * DEPTH_SCALE;
    const height = Math.max(-MAX_DEPTH, rawHeight);

    let [cr, cg, cb] = regimeColorForOmega(omega);

    // Accretion disk glow band (ω ∈ [0.45, 0.80])
    if (omega > 0.45 && omega < 0.80) {
      const diskT = Math.sin(((omega - 0.45) / 0.35) * Math.PI);
      const glow = diskT * diskT * 0.7;
      cr = Math.min(1, cr + glow * 0.9);
      cg = Math.min(1, cg + glow * 0.55);
      cb = Math.min(1, cb + glow * 0.15);
    }

    for (let ai = 0; ai < angularSteps; ai++) {
      const theta = (ai / angularSteps) * Math.PI * 2;

      vertices[vi++] = screenR * Math.cos(theta);
      vertices[vi++] = height;
      vertices[vi++] = screenR * Math.sin(theta);
      vertices[vi++] = cr;
      vertices[vi++] = cg;
      vertices[vi++] = cb;
      // Placeholder normals — computed below
      vertices[vi++] = 0;
      vertices[vi++] = 1;
      vertices[vi++] = 0;
    }
  }

  // Compute proper normals via cross-product of adjacent edges
  for (let ri = 0; ri <= radialSteps; ri++) {
    for (let ai = 0; ai < angularSteps; ai++) {
      const idx = (ri * angularSteps + ai) * 9;
      const px = vertices[idx], py = vertices[idx + 1], pz = vertices[idx + 2];

      const aiNext = (ai + 1) % angularSteps;
      const riNext = Math.min(ri + 1, radialSteps);

      const n1Idx = (ri * angularSteps + aiNext) * 9;
      const n2Idx = (riNext * angularSteps + ai) * 9;

      const dx1 = vertices[n1Idx] - px, dy1 = vertices[n1Idx + 1] - py, dz1 = vertices[n1Idx + 2] - pz;
      const dx2 = vertices[n2Idx] - px, dy2 = vertices[n2Idx + 1] - py, dz2 = vertices[n2Idx + 2] - pz;

      let nx = dy1 * dz2 - dz1 * dy2;
      let ny = dz1 * dx2 - dx1 * dz2;
      let nz = dx1 * dy2 - dy1 * dx2;
      const nl = Math.sqrt(nx * nx + ny * ny + nz * nz) || 1;
      nx /= nl; ny /= nl; nz /= nl;

      if (ny < 0) { nx = -nx; ny = -ny; nz = -nz; }

      vertices[idx + 6] = nx;
      vertices[idx + 7] = ny;
      vertices[idx + 8] = nz;
    }
  }

  // Indices
  const indexCount = radialSteps * angularSteps * 6;
  const indices = new Uint16Array(indexCount);
  let ii = 0;
  for (let ri = 0; ri < radialSteps; ri++) {
    for (let ai = 0; ai < angularSteps; ai++) {
      const cur = ri * angularSteps + ai;
      const nxt = ri * angularSteps + ((ai + 1) % angularSteps);
      const abv = (ri + 1) * angularSteps + ai;
      const abvN = (ri + 1) * angularSteps + ((ai + 1) % angularSteps);
      indices[ii++] = cur;
      indices[ii++] = abv;
      indices[ii++] = nxt;
      indices[ii++] = nxt;
      indices[ii++] = abv;
      indices[ii++] = abvN;
    }
  }

  // Entity markers on the well surface
  const entities = BLACK_HOLE_ENTITIES;
  const entityMarkers = new Float32Array(entities.length * 6);
  entities.forEach((e, i) => {
    const kr = computeKernel(e.c, e.w);
    const omega = kr.omega;
    const rFrac = Math.max(0, 1.0 - omega / 0.998);
    const screenR = rFrac * R_MAX;
    const gamma = gammaOmega(omega);
    const height = Math.max(-MAX_DEPTH, -Math.log(1 + gamma) * DEPTH_SCALE);
    const angle = (i / entities.length) * Math.PI * 2;

    const base = i * 6;
    entityMarkers[base] = screenR * Math.cos(angle);
    entityMarkers[base + 1] = height + 0.05;
    entityMarkers[base + 2] = screenR * Math.sin(angle);

    if (kr.omega < 0.038)      { entityMarkers[base + 3] = 0.2; entityMarkers[base + 4] = 0.85; entityMarkers[base + 5] = 0.6; }
    else if (kr.omega < 0.30)  { entityMarkers[base + 3] = 0.98; entityMarkers[base + 4] = 0.75; entityMarkers[base + 5] = 0.15; }
    else                       { entityMarkers[base + 3] = 0.95; entityMarkers[base + 4] = 0.3; entityMarkers[base + 5] = 0.3; }
  });

  return { vertices, indices, entityMarkers, entityCount: entities.length };
}

/* ─── Particle System ───────────────────────────────────────────── */

const NUM_PARTICLES = 250;

interface Particle {
  r: number;       // radial distance from center (screen coords)
  theta: number;   // angular position
  speed: number;   // angular speed multiplier
  infall: number;  // radial infall rate
}

function initParticles(): Particle[] {
  const particles: Particle[] = [];
  for (let i = 0; i < NUM_PARTICLES; i++) {
    particles.push({
      r: 0.4 + Math.random() * 1.8,
      theta: Math.random() * Math.PI * 2,
      speed: 0.5 + Math.random() * 1.5,
      infall: 0.0005 + Math.random() * 0.002,
    });
  }
  return particles;
}

function updateParticles(particles: Particle[], dt: number): Float32Array {
  const data = new Float32Array(NUM_PARTICLES * 6);
  const R_MAX = 2.8;
  const DEPTH_SCALE = 0.45;
  const MAX_DEPTH = 6.0;

  for (let i = 0; i < particles.length; i++) {
    const p = particles[i];

    // Kepler-like: angular speed ∝ 1/r^1.5
    p.theta += dt * p.speed / (p.r * p.r * 0.8 + 0.1);
    // Spiral inward
    p.r -= p.infall * dt * (1.0 / (p.r + 0.1));

    // Respawn at outer edge if too close to center
    if (p.r < 0.15) {
      p.r = 0.8 + Math.random() * 1.6;
      p.theta = Math.random() * Math.PI * 2;
      p.speed = 0.5 + Math.random() * 1.5;
    }

    // Convert to 3D position on the well surface
    const rFrac = p.r / R_MAX;
    const omega = 1.0 - Math.max(rFrac, 0.005) * 0.998;
    const gamma = gammaOmega(omega);
    const height = Math.max(-MAX_DEPTH, -Math.log(1 + gamma) * DEPTH_SCALE) + 0.02;

    const base = i * 6;
    data[base] = p.r * Math.cos(p.theta);
    data[base + 1] = height;
    data[base + 2] = p.r * Math.sin(p.theta);

    // Color: hot near center, cooler at edge
    const temp = 1.0 - Math.min(p.r / R_MAX, 1.0);
    data[base + 3] = 1.0;
    data[base + 4] = 0.6 + temp * 0.4;
    data[base + 5] = 0.3 * (1.0 - temp);
  }
  return data;
}

/* ─── WebGL Setup ───────────────────────────────────────────────── */

function compileShader(gl: WebGLRenderingContext, type: number, source: string): WebGLShader {
  const shader = gl.createShader(type)!;
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    console.error('Shader error:', gl.getShaderInfoLog(shader));
  }
  return shader;
}

function createProgram(gl: WebGLRenderingContext, vs: string, fs: string): WebGLProgram {
  const prog = gl.createProgram()!;
  gl.attachShader(prog, compileShader(gl, gl.VERTEX_SHADER, vs));
  gl.attachShader(prog, compileShader(gl, gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    console.error('Link error:', gl.getProgramInfoLog(prog));
  }
  return prog;
}

/* ─── Main Init ─────────────────────────────────────────────────── */

export function initBlackHole3D(canvas: HTMLCanvasElement): { destroy: () => void } {
  const glCtx = canvas.getContext('webgl', { antialias: true, alpha: false });
  if (!glCtx) {
    console.error('WebGL not available');
    return { destroy: () => {} };
  }
  // Non-null const so TypeScript narrows inside closures
  const gl: WebGLRenderingContext = glCtx;

  gl.enable(gl.DEPTH_TEST);
  gl.clearColor(0.023, 0.023, 0.047, 1.0);

  // ── Programs ──
  const meshProg = createProgram(gl, VERT_SRC, FRAG_SRC);
  const particleProg = createProgram(gl, PARTICLE_VERT, PARTICLE_FRAG);

  // ── Mesh ──
  const mesh = generateGravityWell(96, 128);

  const wellVBO = gl.createBuffer()!;
  gl.bindBuffer(gl.ARRAY_BUFFER, wellVBO);
  gl.bufferData(gl.ARRAY_BUFFER, mesh.vertices, gl.STATIC_DRAW);

  const wellIBO = gl.createBuffer()!;
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, wellIBO);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, mesh.indices, gl.STATIC_DRAW);

  // ── Entity markers ──
  const entityVBO = gl.createBuffer()!;
  gl.bindBuffer(gl.ARRAY_BUFFER, entityVBO);
  gl.bufferData(gl.ARRAY_BUFFER, mesh.entityMarkers, gl.STATIC_DRAW);

  // ── Particles ──
  const particles = initParticles();
  const particleVBO = gl.createBuffer()!;

  // ── Starfield background points ──
  const NUM_STARS = 500;
  const starData = new Float32Array(NUM_STARS * 6);
  for (let i = 0; i < NUM_STARS; i++) {
    const base = i * 6;
    // Distribute stars on a sphere
    const phi = Math.random() * Math.PI * 2;
    const cosTheta = 2 * Math.random() - 1;
    const sinTheta = Math.sqrt(1 - cosTheta * cosTheta);
    const radius = 8 + Math.random() * 4;
    starData[base] = radius * sinTheta * Math.cos(phi);
    starData[base + 1] = radius * cosTheta;
    starData[base + 2] = radius * sinTheta * Math.sin(phi);
    const brightness = 0.3 + Math.random() * 0.5;
    starData[base + 3] = brightness;
    starData[base + 4] = brightness;
    starData[base + 5] = brightness * (0.8 + Math.random() * 0.2);
  }
  const starVBO = gl.createBuffer()!;
  gl.bindBuffer(gl.ARRAY_BUFFER, starVBO);
  gl.bufferData(gl.ARRAY_BUFFER, starData, gl.STATIC_DRAW);

  // ── Camera state ──
  let rotX = -0.55;
  let rotY = 0;
  let distance = 6.0;
  let autoRotate = true;
  let isDragging = false;
  let lastMX = 0, lastMY = 0;

  // Mouse orbit
  canvas.addEventListener('mousedown', (e) => {
    isDragging = true; autoRotate = false;
    lastMX = e.clientX; lastMY = e.clientY;
  });
  canvas.addEventListener('mousemove', (e) => {
    if (!isDragging) return;
    rotY += (e.clientX - lastMX) * 0.007;
    rotX += (e.clientY - lastMY) * 0.007;
    rotX = Math.max(-Math.PI * 0.48, Math.min(Math.PI * 0.08, rotX));
    lastMX = e.clientX; lastMY = e.clientY;
  });
  canvas.addEventListener('mouseup', () => { isDragging = false; });
  canvas.addEventListener('mouseleave', () => { isDragging = false; });

  // Touch
  canvas.addEventListener('touchstart', (e) => {
    isDragging = true; autoRotate = false;
    lastMX = e.touches[0].clientX; lastMY = e.touches[0].clientY;
    e.preventDefault();
  }, { passive: false });
  canvas.addEventListener('touchmove', (e) => {
    if (!isDragging) return;
    const t = e.touches[0];
    rotY += (t.clientX - lastMX) * 0.007;
    rotX += (t.clientY - lastMY) * 0.007;
    rotX = Math.max(-Math.PI * 0.48, Math.min(Math.PI * 0.08, rotX));
    lastMX = t.clientX; lastMY = t.clientY;
    e.preventDefault();
  }, { passive: false });
  canvas.addEventListener('touchend', () => { isDragging = false; });

  // Zoom
  canvas.addEventListener('wheel', (e) => {
    distance += e.deltaY * 0.005;
    distance = Math.max(2.5, Math.min(14, distance));
    e.preventDefault();
  }, { passive: false });

  // Double-click reset
  canvas.addEventListener('dblclick', () => {
    rotX = -0.55; rotY = 0; distance = 6.0; autoRotate = true;
  });

  // ── Render Loop ──
  let animId = 0;
  let destroyed = false;
  let lastTime = performance.now();

  function render(now: number) {
    if (destroyed) return;

    const dt = Math.min((now - lastTime) / 16.67, 3); // normalize to ~60fps
    lastTime = now;

    // Resize
    const dpr = window.devicePixelRatio || 1;
    const dispW = canvas.clientWidth;
    const dispH = canvas.clientHeight;
    const bufW = Math.round(dispW * dpr);
    const bufH = Math.round(dispH * dpr);
    if (canvas.width !== bufW || canvas.height !== bufH) {
      canvas.width = bufW;
      canvas.height = bufH;
    }
    gl.viewport(0, 0, canvas.width, canvas.height);

    if (autoRotate) rotY += 0.004 * dt;

    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    // Matrices
    const aspect = dispW / dispH;
    const proj = mat4Perspective(Math.PI / 4.5, aspect, 0.1, 100);
    const t = mat4Translate(0, 0.8, -distance);
    const rx = mat4RotateX(rotX);
    const ry = mat4RotateY(rotY);
    const tmp = mat4Create();
    const view = mat4Create();
    mat4Multiply(tmp, rx, ry);
    mat4Multiply(view, t, tmp);
    const mvp = mat4Create();
    mat4Multiply(mvp, proj, view);

    const STRIDE_9 = 36; // 9 floats × 4 bytes
    const STRIDE_6 = 24; // 6 floats × 4 bytes

    // ── Draw well mesh ──
    gl.useProgram(meshProg);
    const uMVP_m = gl.getUniformLocation(meshProg, 'uMVP');
    const uMV_m = gl.getUniformLocation(meshProg, 'uModelView');
    gl.uniformMatrix4fv(uMVP_m, false, mvp);
    gl.uniformMatrix4fv(uMV_m, false, view);

    const aPos_m = gl.getAttribLocation(meshProg, 'aPosition');
    const aCol_m = gl.getAttribLocation(meshProg, 'aColor');
    const aNrm_m = gl.getAttribLocation(meshProg, 'aNormal');

    gl.bindBuffer(gl.ARRAY_BUFFER, wellVBO);
    gl.enableVertexAttribArray(aPos_m);
    gl.vertexAttribPointer(aPos_m, 3, gl.FLOAT, false, STRIDE_9, 0);
    gl.enableVertexAttribArray(aCol_m);
    gl.vertexAttribPointer(aCol_m, 3, gl.FLOAT, false, STRIDE_9, 12);
    gl.enableVertexAttribArray(aNrm_m);
    gl.vertexAttribPointer(aNrm_m, 3, gl.FLOAT, false, STRIDE_9, 24);

    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, wellIBO);
    gl.drawElements(gl.TRIANGLES, mesh.indices.length, gl.UNSIGNED_SHORT, 0);

    // ── Draw entity markers (reuse mesh program, draw as POINTS) ──
    gl.bindBuffer(gl.ARRAY_BUFFER, entityVBO);
    gl.vertexAttribPointer(aPos_m, 3, gl.FLOAT, false, STRIDE_6, 0);
    gl.vertexAttribPointer(aCol_m, 3, gl.FLOAT, false, STRIDE_6, 12);
    // Disable normal attrib — set constant
    gl.disableVertexAttribArray(aNrm_m);
    gl.vertexAttrib3f(aNrm_m, 0, 1, 0);
    gl.drawArrays(gl.POINTS, 0, mesh.entityCount);
    gl.enableVertexAttribArray(aNrm_m);

    // ── Draw particles ──
    gl.useProgram(particleProg);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE); // additive blending for glow
    gl.depthMask(false);

    const uMVP_p = gl.getUniformLocation(particleProg, 'uMVP');
    gl.uniformMatrix4fv(uMVP_p, false, mvp);

    const aPos_p = gl.getAttribLocation(particleProg, 'aPosition');
    const aCol_p = gl.getAttribLocation(particleProg, 'aColor');

    const particleData = updateParticles(particles, dt);
    gl.bindBuffer(gl.ARRAY_BUFFER, particleVBO);
    gl.bufferData(gl.ARRAY_BUFFER, particleData, gl.DYNAMIC_DRAW);

    gl.enableVertexAttribArray(aPos_p);
    gl.vertexAttribPointer(aPos_p, 3, gl.FLOAT, false, STRIDE_6, 0);
    gl.enableVertexAttribArray(aCol_p);
    gl.vertexAttribPointer(aCol_p, 3, gl.FLOAT, false, STRIDE_6, 12);
    // Disable unused attribs from mesh program
    if (aNrm_m >= 0) gl.disableVertexAttribArray(aNrm_m);
    gl.drawArrays(gl.POINTS, 0, NUM_PARTICLES);

    // ── Draw starfield (reuse particle program) ──
    gl.bindBuffer(gl.ARRAY_BUFFER, starVBO);
    gl.vertexAttribPointer(aPos_p, 3, gl.FLOAT, false, STRIDE_6, 0);
    gl.vertexAttribPointer(aCol_p, 3, gl.FLOAT, false, STRIDE_6, 12);
    gl.drawArrays(gl.POINTS, 0, NUM_STARS);

    gl.depthMask(true);
    gl.disable(gl.BLEND);

    animId = requestAnimationFrame(render);
  }

  animId = requestAnimationFrame(render);

  return {
    destroy() {
      destroyed = true;
      cancelAnimationFrame(animId);
      gl.deleteBuffer(wellVBO);
      gl.deleteBuffer(wellIBO);
      gl.deleteBuffer(entityVBO);
      gl.deleteBuffer(particleVBO);
      gl.deleteBuffer(starVBO);
      gl.deleteProgram(meshProg);
      gl.deleteProgram(particleProg);
    },
  };
}

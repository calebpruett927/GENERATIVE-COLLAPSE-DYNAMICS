import{c as be,g as xe,a as ke}from"./kernel.C-ROa8s-.js";import{B as k,r as Ve,p as He,s as Oe,i as Re,o as Ue,b as Ge,a as Ne,f as We,h as Ye,e as qe,g as Ke}from"./spacetime.CHHUQeVA.js";import"./constants.6x37F0HI.js";function te(){return new Float32Array(16)}function we(){const t=te();return t[0]=t[5]=t[10]=t[15]=1,t}function Xe(t,i,s,e){const r=te(),a=1/Math.tan(t*.5),n=1/(s-e);return r[0]=a/i,r[5]=a,r[10]=(e+s)*n,r[11]=-1,r[14]=2*e*s*n,r}function ue(t,i){const s=te();for(let e=0;e<4;e++)for(let r=0;r<4;r++)s[r*4+e]=t[e]*i[r*4]+t[4+e]*i[r*4+1]+t[8+e]*i[r*4+2]+t[12+e]*i[r*4+3];return s}function je(t){const i=we(),s=Math.cos(t),e=Math.sin(t);return i[5]=s,i[6]=e,i[9]=-e,i[10]=s,i}function me(t){const i=Math.sqrt(t[0]*t[0]+t[1]*t[1]+t[2]*t[2])||1;return[t[0]/i,t[1]/i,t[2]/i]}function ge(t,i){return[t[1]*i[2]-t[2]*i[1],t[2]*i[0]-t[0]*i[2],t[0]*i[1]-t[1]*i[0]]}function Je(t,i,s){const e=me([t[0]-i[0],t[1]-i[1],t[2]-i[2]]),r=me(ge(s,e)),a=ge(e,r),n=we();return n[0]=r[0],n[4]=r[1],n[8]=r[2],n[12]=-(r[0]*t[0]+r[1]*t[1]+r[2]*t[2]),n[1]=a[0],n[5]=a[1],n[9]=a[2],n[13]=-(a[0]*t[0]+a[1]*t[1]+a[2]*t[2]),n[2]=e[0],n[6]=e[1],n[10]=e[2],n[14]=-(e[0]*t[0]+e[1]*t[1]+e[2]*t[2]),n[3]=0,n[7]=0,n[11]=0,n[15]=1,n}const Qe=`
attribute vec2 aPos;
varying vec2 vUV;
void main() {
  vUV = aPos * 0.5 + 0.5;
  gl_Position = vec4(aPos, 0.0, 1.0);
}
`,Ze=`
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
`,$e=`
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
`,et=`
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
`,tt=`
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
`,ot=`
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
`;function rt(t,i,s,e){const r=(e+1)*s,a=new Float32Array(r*5);let n=0;for(let m=0;m<=e;m++){const g=m/e,C=t+g*(i-t);for(let R=0;R<s;R++){const E=R/s*Math.PI*2;a[n++]=C*Math.cos(E),a[n++]=0,a[n++]=C*Math.sin(E),a[n++]=g,a[n++]=R/s}}const v=e*s*6,f=new Uint16Array(v);let b=0;for(let m=0;m<e;m++)for(let g=0;g<s;g++){const C=m*s+g,R=m*s+(g+1)%s,E=(m+1)*s+g,K=(m+1)*s+(g+1)%s;f[b++]=C,f[b++]=E,f[b++]=R,f[b++]=R,f[b++]=E,f[b++]=K}return{verts:a,indices:f}}function W(t,i,s){if(t==="accretion"){const e=i+Math.random()*(s-i),r=Math.random()*Math.PI*2,a=1/Math.sqrt(e),n=.02*Math.random();return{x:e*Math.cos(r)*(1+n),y:(Math.random()-.5)*.04,z:e*Math.sin(r),vx:-a*Math.sin(r)*.3,vy:0,vz:a*Math.cos(r)*.3,life:0,maxLife:4+Math.random()*6,size:2+Math.random()*3,type:t,cr:1,cg:.7+Math.random()*.3,cb:.2+Math.random()*.3}}else if(t==="jet"){const e=Math.random()>.5?1:-1,r=.08,a=Math.sin(Date.now()*.001)*.03;return{x:(Math.random()-.5)*r+a,y:e*i*.3,z:(Math.random()-.5)*r,vx:(Math.random()-.5)*.04,vy:e*(1.8+Math.random()*.8),vz:(Math.random()-.5)*.04,life:0,maxLife:1.5+Math.random()*2.5,size:1.5+Math.random()*2,type:t,cr:.3+Math.random()*.3,cg:.5+Math.random()*.4,cb:1}}else if(t==="tidal"){const e=Math.random()*Math.PI*2,r=i*(1+Math.random()*.5);return{x:r*Math.cos(e),y:(Math.random()-.5)*.02,z:r*Math.sin(e),vx:-Math.cos(e)*.4,vy:(Math.random()-.5)*.1,vz:-Math.sin(e)*.4,life:0,maxLife:1+Math.random()*1.5,size:1+Math.random()*1.5,type:t,cr:1,cg:.3,cb:.1}}else if(t==="penrose"){const e=Math.random()*Math.PI*2,r=i*1.1,a=Math.random()>.5;return{x:r*Math.cos(e),y:(Math.random()-.5)*.05,z:r*Math.sin(e),vx:(a?1:-1)*Math.cos(e)*.5,vy:a?.3:-.1,vz:(a?1:-1)*Math.sin(e)*.5,life:0,maxLife:a?3:.8,size:a?3:1.5,type:t,cr:a?.2:.8,cg:a?.8:.2,cb:a?1:.3}}else{const e=Math.random()*Math.PI*2,r=s*(1.2+Math.random()*.5),a=.08*(Math.random()-.3);return{x:r*Math.cos(e),y:(Math.random()-.5)*.3,z:r*Math.sin(e),vx:-Math.cos(e)*.15+a*-Math.sin(e),vy:0,vz:-Math.sin(e)*.15+a*Math.cos(e),life:0,maxLife:4+Math.random()*4,size:1.5+Math.random()*2.5,type:t,cr:.9,cg:.5,cb:.2}}}function $(t,i,s,e){const n=Math.min(.98,.01+(1-Math.min(t/20,1))*(i-.01)),v=xe(n),f=be(k[0].c,k[0].w),b=ke(f);return{omega:n,gamma:v,regime:b.regime+(b.isCritical?" (Critical)":""),redshift:Ke(n),escapeV:qe(n),hawkingT:Ye(f.kappa),distance:t,F:f.F,IC:f.IC,kappa:f.kappa,S:f.S,C:f.C,delta:f.delta,frameDrag:We(n,s),penroseEff:Ne(s),entropy:Ge(f.kappa),precession:Ue(n),gwStrainVal:Re(i,e,t),surfGravity:Oe(n),timeFactor:He(n),radEfficiency:Ve(s),spinStar:s}}function nt(t,i){const s=t.getContext("webgl",{antialias:!0,alpha:!1,premultipliedAlpha:!1});if(!s)return console.error("WebGL not available"),{destroy:()=>{},getHUD:()=>$(10,.95,.7,0)};const e=s;e.getExtension("OES_standard_derivatives"),e.enable(e.DEPTH_TEST),e.enable(e.BLEND),e.blendFunc(e.SRC_ALPHA,e.ONE_MINUS_SRC_ALPHA),e.clearColor(0,0,0,1);function r(l,d){const u=e.createShader(l);return e.shaderSource(u,d),e.compileShader(u),e.getShaderParameter(u,e.COMPILE_STATUS)||console.error("Shader:",e.getShaderInfoLog(u)),u}function a(l,d){const u=e.createProgram();return e.attachShader(u,r(e.VERTEX_SHADER,l)),e.attachShader(u,r(e.FRAGMENT_SHADER,d)),e.linkProgram(u),e.getProgramParameter(u,e.LINK_STATUS)||console.error("Link:",e.getProgramInfoLog(u)),u}const n=a(Qe,Ze),v=a($e,et),f=a(tt,ot),b=e.createBuffer();e.bindBuffer(e.ARRAY_BUFFER,b),e.bufferData(e.ARRAY_BUFFER,new Float32Array([-1,-1,1,-1,-1,1,1,1]),e.STATIC_DRAW);const m=.6,g=3,C=rt(m,g,192,48),R=e.createBuffer();e.bindBuffer(e.ARRAY_BUFFER,R),e.bufferData(e.ARRAY_BUFFER,C.verts,e.STATIC_DRAW);const E=e.createBuffer();e.bindBuffer(e.ELEMENT_ARRAY_BUFFER,E),e.bufferData(e.ELEMENT_ARRAY_BUFFER,C.indices,e.STATIC_DRAW);const K=600,x=[];for(let l=0;l<350;l++){const d=l<200?"accretion":l<270?"jet":"infalling",u=W(d,m,g);u.life=Math.random()*u.maxLife,x.push(u)}const Ae=e.createBuffer();let H=0,z=.35,w=8,B=!1,F=0,L=0,O=!0,U=0,P=.35,_=8;t.addEventListener("mousedown",l=>{B=!0,O=!1,F=l.clientX,L=l.clientY,t.style.cursor="grabbing"}),t.addEventListener("mousemove",l=>{B&&(U+=(l.clientX-F)*.005,P+=(l.clientY-L)*.005,P=Math.max(-Math.PI*.45,Math.min(Math.PI*.45,P)),F=l.clientX,L=l.clientY)}),t.addEventListener("mouseup",()=>{B=!1,t.style.cursor="grab"}),t.addEventListener("mouseleave",()=>{B=!1,t.style.cursor="grab"}),t.addEventListener("touchstart",l=>{B=!0,O=!1,F=l.touches[0].clientX,L=l.touches[0].clientY,l.preventDefault()},{passive:!1}),t.addEventListener("touchmove",l=>{B&&(U+=(l.touches[0].clientX-F)*.005,P+=(l.touches[0].clientY-L)*.005,P=Math.max(-Math.PI*.45,Math.min(Math.PI*.45,P)),F=l.touches[0].clientX,L=l.touches[0].clientY,l.preventDefault())},{passive:!1}),t.addEventListener("touchend",()=>{B=!1}),t.addEventListener("wheel",l=>{_+=l.deltaY*.01,_=Math.max(2,Math.min(25,_)),l.preventDefault()},{passive:!1}),t.addEventListener("dblclick",()=>{O=!0,P=.35,_=8}),t.style.cursor="grab";const G=.95,I=.7,X=k[0].c.length>3?be(k[0].c,k[0].w).C:0,S={aPos:e.getAttribLocation(n,"aPos"),uBHScreen:e.getUniformLocation(n,"uBHScreen"),uBHRadius:e.getUniformLocation(n,"uBHRadius"),uLensStrength:e.getUniformLocation(n,"uLensStrength"),uTime:e.getUniformLocation(n,"uTime"),uSpinStar:e.getUniformLocation(n,"uSpinStar"),uGWStrain:e.getUniformLocation(n,"uGWStrain")},p={aPosition:e.getAttribLocation(v,"aPosition"),aTexCoord:e.getAttribLocation(v,"aTexCoord"),uMVP:e.getUniformLocation(v,"uMVP"),uTime:e.getUniformLocation(v,"uTime"),uInnerR:e.getUniformLocation(v,"uInnerR"),uOuterR:e.getUniformLocation(v,"uOuterR"),uSpinDisk:e.getUniformLocation(v,"uSpinDisk")},A={aPosition:e.getAttribLocation(f,"aPosition"),aColor:e.getAttribLocation(f,"aColor"),aSize:e.getAttribLocation(f,"aSize"),uMVP:e.getUniformLocation(f,"uMVP")};function ne(){const l=Math.min(window.devicePixelRatio||1,2),d=t.clientWidth,u=t.clientHeight;t.width=Math.round(d*l),t.height=Math.round(u*l),e.viewport(0,0,t.width,t.height)}ne();const ie=new ResizeObserver(ne);ie.observe(t);let j=$(w,G,I,X),ae=!0,J=0;function se(l){if(!ae)return;const d=Math.min((l-J)/1e3,.05);J=l,O&&(U+=d*.08),H+=(U-H)*.05,z+=(P-z)*.05,w+=(_-w)*.05;const u=w*Math.cos(z)*Math.sin(H),Me=w*Math.sin(z),Se=w*Math.cos(z)*Math.cos(H),ye=t.width/t.height,Ce=Xe(.9,ye,.1,100),Ee=Je([u,Me,Se],[0,0,0],[0,1,0]),D=ue(Ce,Ee),Pe=D[12]/D[15],Te=D[13]/D[15],Be=[Pe*.5+.5,Te*.5+.5],Fe=Math.atan(.5/w)/(.9/2),Le=xe(G)/(w*.5),le=l/1e3;e.clear(e.COLOR_BUFFER_BIT|e.DEPTH_BUFFER_BIT),e.depthMask(!1),e.disable(e.DEPTH_TEST),e.useProgram(n),e.bindBuffer(e.ARRAY_BUFFER,b),e.enableVertexAttribArray(S.aPos),e.vertexAttribPointer(S.aPos,2,e.FLOAT,!1,0,0),e.uniform2fv(S.uBHScreen,Be),e.uniform1f(S.uBHRadius,Fe),e.uniform1f(S.uLensStrength,Le),e.uniform1f(S.uTime,le),e.uniform1f(S.uSpinStar,I);const Ie=Re(G,X,w);e.uniform1f(S.uGWStrain,Math.min(Ie,1)),e.drawArrays(e.TRIANGLE_STRIP,0,4),e.disableVertexAttribArray(S.aPos),e.depthMask(!0),e.enable(e.DEPTH_TEST);const De=je(.12),ze=ue(D,De);e.useProgram(v),e.bindBuffer(e.ARRAY_BUFFER,R),e.bindBuffer(e.ELEMENT_ARRAY_BUFFER,E);const ce=20;e.enableVertexAttribArray(p.aPosition),e.vertexAttribPointer(p.aPosition,3,e.FLOAT,!1,ce,0),p.aTexCoord>=0&&(e.enableVertexAttribArray(p.aTexCoord),e.vertexAttribPointer(p.aTexCoord,2,e.FLOAT,!1,ce,12)),e.uniformMatrix4fv(p.uMVP,!1,ze),e.uniform1f(p.uTime,le),e.uniform1f(p.uInnerR,m),e.uniform1f(p.uOuterR,g),e.uniform1f(p.uSpinDisk,I),e.enable(e.BLEND),e.blendFunc(e.SRC_ALPHA,e.ONE_MINUS_SRC_ALPHA),e.drawElements(e.TRIANGLES,C.indices.length,e.UNSIGNED_SHORT,0),e.disableVertexAttribArray(p.aPosition),p.aTexCoord>=0&&e.disableVertexAttribArray(p.aTexCoord);for(let h=x.length-1;h>=0;h--){const o=x[h];if(o.life+=d,o.life>o.maxLife){x[h]=W(o.type,m,g);continue}const c=Math.sqrt(o.x*o.x+o.z*o.z)||.1;if(o.type==="accretion"){const y=1/(c*Math.sqrt(c))*.5,N=-o.z/c,_e=o.x/c,de=I*.005/(c*c+.1),fe=.008/(c+.1);if(o.vx=N*(y+de)-o.x/c*fe,o.vz=_e*(y+de)-o.z/c*fe,c<m*1.2){const Z=Math.max(.1,(c-.2)/m);o.cr*=Z,o.cg*=Z,o.cb*=Z}}else if(o.type==="jet")o.vy*=1.008,o.vx*=.99,o.vz*=.99;else if(o.type==="tidal"){const y=.15/(c*c+.01);o.vx-=o.x/c*y*d,o.vz-=o.z/c*y*d,o.size*=1.01,o.cr=Math.min(1,o.cr+d*.3),o.cg=Math.min(1,o.cg+d*.2),o.cb=Math.min(1,o.cb+d*.15)}else if(o.type==="penrose")(o.vx*o.x+o.vz*o.z)/c>0?(o.vx*=.998,o.vz*=.998):(o.vx-=o.x/c*.1*d,o.vz-=o.z/c*.1*d);else{const y=.05/(c*c+.01);o.vx-=o.x/c*y*d,o.vz-=o.z/c*y*d;const N=I*.003/(c*c+.1);o.vx+=-o.z/c*N*d,o.vz+=o.x/c*N*d}o.x+=o.vx*d,o.y+=o.vy*d,o.z+=o.vz*d;const M=Math.sqrt(o.x*o.x+o.y*o.y+o.z*o.z);(M<.15||M>18)&&(x[h]=W(o.type,m,g))}for(;x.length<K;){const h=Math.random(),o=h<.4?"accretion":h<.58?"jet":h<.75?"infalling":h<.88?"tidal":"penrose";x.push(W(o,m,g))}const T=new Float32Array(x.length*7);for(let h=0;h<x.length;h++){const o=x[h],c=Math.min(1,Math.min(o.life/.3,(o.maxLife-o.life)/.5)),M=h*7;T[M]=o.x,T[M+1]=o.y,T[M+2]=o.z,T[M+3]=o.cr*c,T[M+4]=o.cg*c,T[M+5]=o.cb*c,T[M+6]=o.size*(.5+c*.5)}e.useProgram(f),e.bindBuffer(e.ARRAY_BUFFER,Ae),e.bufferData(e.ARRAY_BUFFER,T,e.DYNAMIC_DRAW);const Q=28;e.enableVertexAttribArray(A.aPosition),e.vertexAttribPointer(A.aPosition,3,e.FLOAT,!1,Q,0),e.enableVertexAttribArray(A.aColor),e.vertexAttribPointer(A.aColor,3,e.FLOAT,!1,Q,12),e.enableVertexAttribArray(A.aSize),e.vertexAttribPointer(A.aSize,1,e.FLOAT,!1,Q,24),e.uniformMatrix4fv(A.uMVP,!1,D),e.blendFunc(e.SRC_ALPHA,e.ONE),e.drawArrays(e.POINTS,0,x.length),e.disableVertexAttribArray(A.aPosition),e.disableVertexAttribArray(A.aColor),e.disableVertexAttribArray(A.aSize),e.blendFunc(e.SRC_ALPHA,e.ONE_MINUS_SRC_ALPHA),j=$(w,G,I,X),i&&i(j),requestAnimationFrame(se)}return requestAnimationFrame(l=>{J=l,se(l)}),{destroy:()=>{ae=!1,ie.disconnect()},getHUD:()=>j}}const oe=document.getElementById("sim-canvas"),it=document.getElementById("hud-F"),at=document.getElementById("hud-omega"),st=document.getElementById("hud-IC"),lt=document.getElementById("hud-kappa"),ct=document.getElementById("hud-S"),dt=document.getElementById("hud-delta"),ft=document.getElementById("hud-gamma"),ut=document.getElementById("hud-redshift"),mt=document.getElementById("hud-vesc"),gt=document.getElementById("hud-hawking"),ht=document.getElementById("hud-dist"),pt=document.getElementById("hud-spin"),vt=document.getElementById("hud-drag"),bt=document.getElementById("hud-penrose"),xt=document.getElementById("hud-time"),Rt=document.getElementById("hud-entropy"),wt=document.getElementById("hud-prec"),At=document.getElementById("hud-gw"),Mt=document.getElementById("hud-surfg"),St=document.getElementById("hud-radeff"),he=document.getElementById("hud-regime"),pe=document.getElementById("regime-bar");let ve=0,ee="";const Y=document.querySelector("#hud-left .hud-panel"),q=document.querySelector("#hud-right .hud-panel");function yt(t){const i=performance.now();if(i-ve<66)return;ve=i,it.textContent=t.F.toFixed(4),at.textContent=t.omega.toFixed(4),st.textContent=t.IC.toFixed(4),lt.textContent=t.kappa.toFixed(4),ct.textContent=t.S.toFixed(4),dt.textContent=t.delta.toFixed(4),ft.textContent=t.gamma.toFixed(3),ut.textContent=t.redshift.toFixed(4),mt.textContent=t.escapeV.toFixed(4)+" c",gt.textContent=t.hawkingT.toFixed(6),ht.textContent=t.distance.toFixed(1)+" r_s",pt.textContent=t.spinStar.toFixed(3),vt.textContent=t.frameDrag.toFixed(4),bt.textContent=(t.penroseEff*100).toFixed(1)+"%",xt.textContent=t.timeFactor.toFixed(4),Rt.textContent=t.entropy.toFixed(3),wt.textContent=t.precession.toFixed(4)+" rad",At.textContent=t.gwStrainVal.toExponential(2),Mt.textContent=t.surfGravity.toFixed(4),St.textContent=(t.radEfficiency*100).toFixed(1)+"%";const s=t.regime.split(" ")[0];he.textContent=t.regime,he.className="hud-value regime-"+s.toLowerCase();const r={STABLE:"#059669",WATCH:"#d97706",COLLAPSE:"#dc2626"}[s]||"#333";if(pe.style.backgroundColor=r,pe.style.boxShadow="0 -2px 16px "+r+"80",s!==ee&&ee!==""){const a=r+"60";Y.style.borderColor=r,q.style.borderColor=r,Y.style.boxShadow="0 0 20px "+a,q.style.boxShadow="0 0 20px "+a,setTimeout(()=>{Y.style.borderColor="",q.style.borderColor="",Y.style.boxShadow="",q.style.boxShadow=""},1200)}ee=s}nt(oe,yt);const V=document.getElementById("controls-overlay");let re=setTimeout(()=>V.classList.add("hidden"),6e3);oe.addEventListener("mousedown",()=>{clearTimeout(re),V.classList.add("hidden")});oe.addEventListener("touchstart",()=>{clearTimeout(re),V.classList.add("hidden")});V.addEventListener("click",()=>{clearTimeout(re),V.classList.add("hidden")});

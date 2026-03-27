import{c as ge,g as he,a as Ie}from"./kernel.CLAplkPS.js";import{B as k,r as De,p as ze,s as _e,i as pe,o as ke,b as Ve,a as He,f as Oe,h as Ue,e as Ge,g as Ne}from"./spacetime.B0oqrIJp.js";import"./constants.6x37F0HI.js";function Z(){return new Float32Array(16)}function ve(){const t=Z();return t[0]=t[5]=t[10]=t[15]=1,t}function We(t,i,s,e){const r=Z(),l=1/Math.tan(t*.5),n=1/(s-e);return r[0]=l/i,r[5]=l,r[10]=(e+s)*n,r[11]=-1,r[14]=2*e*s*n,r}function ce(t,i){const s=Z();for(let e=0;e<4;e++)for(let r=0;r<4;r++)s[r*4+e]=t[e]*i[r*4]+t[4+e]*i[r*4+1]+t[8+e]*i[r*4+2]+t[12+e]*i[r*4+3];return s}function Ye(t){const i=ve(),s=Math.cos(t),e=Math.sin(t);return i[5]=s,i[6]=e,i[9]=-e,i[10]=s,i}function de(t){const i=Math.sqrt(t[0]*t[0]+t[1]*t[1]+t[2]*t[2])||1;return[t[0]/i,t[1]/i,t[2]/i]}function fe(t,i){return[t[1]*i[2]-t[2]*i[1],t[2]*i[0]-t[0]*i[2],t[0]*i[1]-t[1]*i[0]]}function qe(t,i,s){const e=de([t[0]-i[0],t[1]-i[1],t[2]-i[2]]),r=de(fe(s,e)),l=fe(e,r),n=ve();return n[0]=r[0],n[4]=r[1],n[8]=r[2],n[12]=-(r[0]*t[0]+r[1]*t[1]+r[2]*t[2]),n[1]=l[0],n[5]=l[1],n[9]=l[2],n[13]=-(l[0]*t[0]+l[1]*t[1]+l[2]*t[2]),n[2]=e[0],n[6]=e[1],n[10]=e[2],n[14]=-(e[0]*t[0]+e[1]*t[1]+e[2]*t[2]),n[3]=0,n[7]=0,n[11]=0,n[15]=1,n}const Ke=`
attribute vec2 aPos;
varying vec2 vUV;
void main() {
  vUV = aPos * 0.5 + 0.5;
  gl_Position = vec4(aPos, 0.0, 1.0);
}
`,Xe=`
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
  vec3 ringColor = vec3(1.0, 0.85, 0.4);
  for (int n = 0; n < 4; n++) {
    float nf = float(n);
    float demag = exp(-3.14159 * nf * 0.8);  // Lyapunov demagnification
    float subRingR = photonR * (1.0 + demag * 0.3);
    float subRingWidth = 0.0004 * photonR * photonR * demag;
    float subRingBright = 1.2 * demag;
    float rDist = abs(dist - subRingR);
    float ring = exp(-rDist * rDist / max(subRingWidth, 0.00001)) * subRingBright;
    // Sub-rings shift color: inner -> bluer, outer -> redder
    vec3 subCol = mix(ringColor, vec3(0.6, 0.75, 1.0), nf * 0.2);
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

  // -- Event horizon shadow (black disk) --
  float shadowEdge = smoothstep(shadowR, shadowR * 0.85, dist);
  col *= (1.0 - shadowEdge);

  // -- Horizon edge glow (Hawking radiation) --
  float edgeGlow = exp(-(dist - shadowR) * (dist - shadowR) / (0.001 * shadowR * shadowR));
  col += vec3(0.8, 0.3, 0.1) * edgeGlow * 0.3;
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
`,je=`
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
`,Je=`
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
`,Qe=`
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
`,Ze=`
precision mediump float;
varying vec3 vColor;

void main() {
  vec2 c = gl_PointCoord - 0.5;
  float r = length(c);
  if (r > 0.5) discard;
  float glow = exp(-r * r * 8.0);
  gl_FragColor = vec4(vColor * glow, glow * 0.8);
}
`;function $e(t,i,s,e){const r=(e+1)*s,l=new Float32Array(r*5);let n=0;for(let m=0;m<=e;m++){const g=m/e,w=t+g*(i-t);for(let A=0;A<s;A++){const y=A/s*Math.PI*2;l[n++]=w*Math.cos(y),l[n++]=0,l[n++]=w*Math.sin(y),l[n++]=g,l[n++]=A/s}}const v=e*s*6,f=new Uint16Array(v);let x=0;for(let m=0;m<e;m++)for(let g=0;g<s;g++){const w=m*s+g,A=m*s+(g+1)%s,y=(m+1)*s+g,Y=(m+1)*s+(g+1)%s;f[x++]=w,f[x++]=y,f[x++]=A,f[x++]=A,f[x++]=y,f[x++]=Y}return{verts:l,indices:f}}function W(t,i,s){if(t==="accretion"){const e=i+Math.random()*(s-i),r=Math.random()*Math.PI*2,l=1/Math.sqrt(e),n=.02*Math.random();return{x:e*Math.cos(r)*(1+n),y:(Math.random()-.5)*.04,z:e*Math.sin(r),vx:-l*Math.sin(r)*.3,vy:0,vz:l*Math.cos(r)*.3,life:0,maxLife:4+Math.random()*6,size:2+Math.random()*3,type:t,cr:1,cg:.7+Math.random()*.3,cb:.2+Math.random()*.3}}else if(t==="jet"){const e=Math.random()>.5?1:-1,r=.08,l=Math.sin(Date.now()*.001)*.03;return{x:(Math.random()-.5)*r+l,y:e*i*.3,z:(Math.random()-.5)*r,vx:(Math.random()-.5)*.04,vy:e*(1.8+Math.random()*.8),vz:(Math.random()-.5)*.04,life:0,maxLife:1.5+Math.random()*2.5,size:1.5+Math.random()*2,type:t,cr:.3+Math.random()*.3,cg:.5+Math.random()*.4,cb:1}}else if(t==="tidal"){const e=Math.random()*Math.PI*2,r=i*(1+Math.random()*.5);return{x:r*Math.cos(e),y:(Math.random()-.5)*.02,z:r*Math.sin(e),vx:-Math.cos(e)*.4,vy:(Math.random()-.5)*.1,vz:-Math.sin(e)*.4,life:0,maxLife:1+Math.random()*1.5,size:1+Math.random()*1.5,type:t,cr:1,cg:.3,cb:.1}}else if(t==="penrose"){const e=Math.random()*Math.PI*2,r=i*1.1,l=Math.random()>.5;return{x:r*Math.cos(e),y:(Math.random()-.5)*.05,z:r*Math.sin(e),vx:(l?1:-1)*Math.cos(e)*.5,vy:l?.3:-.1,vz:(l?1:-1)*Math.sin(e)*.5,life:0,maxLife:l?3:.8,size:l?3:1.5,type:t,cr:l?.2:.8,cg:l?.8:.2,cb:l?1:.3}}else{const e=Math.random()*Math.PI*2,r=s*(1.2+Math.random()*.5),l=.08*(Math.random()-.3);return{x:r*Math.cos(e),y:(Math.random()-.5)*.3,z:r*Math.sin(e),vx:-Math.cos(e)*.15+l*-Math.sin(e),vy:0,vz:-Math.sin(e)*.15+l*Math.cos(e),life:0,maxLife:4+Math.random()*4,size:1.5+Math.random()*2.5,type:t,cr:.9,cg:.5,cb:.2}}}function Q(t,i,s,e){const n=Math.min(.98,.01+(1-Math.min(t/20,1))*(i-.01)),v=he(n),f=ge(k[0].c,k[0].w),x=Ie(f);return{omega:n,gamma:v,regime:x.regime+(x.isCritical?" (Critical)":""),redshift:Ne(n),escapeV:Ge(n),hawkingT:Ue(f.kappa),distance:t,F:f.F,IC:f.IC,kappa:f.kappa,S:f.S,C:f.C,delta:f.delta,frameDrag:Oe(n,s),penroseEff:He(s),entropy:Ve(f.kappa),precession:ke(n),gwStrainVal:pe(i,e,t),surfGravity:_e(n),timeFactor:ze(n),radEfficiency:De(s),spinStar:s}}function et(t,i){const s=t.getContext("webgl",{antialias:!0,alpha:!1,premultipliedAlpha:!1});if(!s)return console.error("WebGL not available"),{destroy:()=>{},getHUD:()=>Q(10,.95,.7,0)};const e=s;e.getExtension("OES_standard_derivatives"),e.enable(e.DEPTH_TEST),e.enable(e.BLEND),e.blendFunc(e.SRC_ALPHA,e.ONE_MINUS_SRC_ALPHA),e.clearColor(0,0,0,1);function r(a,d){const u=e.createShader(a);return e.shaderSource(u,d),e.compileShader(u),e.getShaderParameter(u,e.COMPILE_STATUS)||console.error("Shader:",e.getShaderInfoLog(u)),u}function l(a,d){const u=e.createProgram();return e.attachShader(u,r(e.VERTEX_SHADER,a)),e.attachShader(u,r(e.FRAGMENT_SHADER,d)),e.linkProgram(u),e.getProgramParameter(u,e.LINK_STATUS)||console.error("Link:",e.getProgramInfoLog(u)),u}const n=l(Ke,Xe),v=l(je,Je),f=l(Qe,Ze),x=e.createBuffer();e.bindBuffer(e.ARRAY_BUFFER,x),e.bufferData(e.ARRAY_BUFFER,new Float32Array([-1,-1,1,-1,-1,1,1,1]),e.STATIC_DRAW);const m=.6,g=3,w=$e(m,g,192,48),A=e.createBuffer();e.bindBuffer(e.ARRAY_BUFFER,A),e.bufferData(e.ARRAY_BUFFER,w.verts,e.STATIC_DRAW);const y=e.createBuffer();e.bindBuffer(e.ELEMENT_ARRAY_BUFFER,y),e.bufferData(e.ELEMENT_ARRAY_BUFFER,w.indices,e.STATIC_DRAW);const Y=600,b=[];for(let a=0;a<350;a++){const d=a<200?"accretion":a<270?"jet":"infalling",u=W(d,m,g);u.life=Math.random()*u.maxLife,b.push(u)}const xe=e.createBuffer();let H=0,z=.35,R=8,B=!1,F=0,L=0,O=!0,U=0,P=.35,_=8;t.addEventListener("mousedown",a=>{B=!0,O=!1,F=a.clientX,L=a.clientY,t.style.cursor="grabbing"}),t.addEventListener("mousemove",a=>{B&&(U+=(a.clientX-F)*.005,P+=(a.clientY-L)*.005,P=Math.max(-Math.PI*.45,Math.min(Math.PI*.45,P)),F=a.clientX,L=a.clientY)}),t.addEventListener("mouseup",()=>{B=!1,t.style.cursor="grab"}),t.addEventListener("mouseleave",()=>{B=!1,t.style.cursor="grab"}),t.addEventListener("touchstart",a=>{B=!0,O=!1,F=a.touches[0].clientX,L=a.touches[0].clientY,a.preventDefault()},{passive:!1}),t.addEventListener("touchmove",a=>{B&&(U+=(a.touches[0].clientX-F)*.005,P+=(a.touches[0].clientY-L)*.005,P=Math.max(-Math.PI*.45,Math.min(Math.PI*.45,P)),F=a.touches[0].clientX,L=a.touches[0].clientY,a.preventDefault())},{passive:!1}),t.addEventListener("touchend",()=>{B=!1}),t.addEventListener("wheel",a=>{_+=a.deltaY*.01,_=Math.max(2,Math.min(25,_)),a.preventDefault()},{passive:!1}),t.addEventListener("dblclick",()=>{O=!0,P=.35,_=8}),t.style.cursor="grab";const G=.95,I=.7,q=k[0].c.length>3?ge(k[0].c,k[0].w).C:0,S={aPos:e.getAttribLocation(n,"aPos"),uBHScreen:e.getUniformLocation(n,"uBHScreen"),uBHRadius:e.getUniformLocation(n,"uBHRadius"),uLensStrength:e.getUniformLocation(n,"uLensStrength"),uTime:e.getUniformLocation(n,"uTime"),uSpinStar:e.getUniformLocation(n,"uSpinStar"),uGWStrain:e.getUniformLocation(n,"uGWStrain")},p={aPosition:e.getAttribLocation(v,"aPosition"),aTexCoord:e.getAttribLocation(v,"aTexCoord"),uMVP:e.getUniformLocation(v,"uMVP"),uTime:e.getUniformLocation(v,"uTime"),uInnerR:e.getUniformLocation(v,"uInnerR"),uOuterR:e.getUniformLocation(v,"uOuterR"),uSpinDisk:e.getUniformLocation(v,"uSpinDisk")},M={aPosition:e.getAttribLocation(f,"aPosition"),aColor:e.getAttribLocation(f,"aColor"),aSize:e.getAttribLocation(f,"aSize"),uMVP:e.getUniformLocation(f,"uMVP")};function te(){const a=Math.min(window.devicePixelRatio||1,2),d=t.clientWidth,u=t.clientHeight;t.width=Math.round(d*a),t.height=Math.round(u*a),e.viewport(0,0,t.width,t.height)}te();const oe=new ResizeObserver(te);oe.observe(t);let K=Q(R,G,I,q),ne=!0,X=0;function re(a){if(!ne)return;const d=Math.min((a-X)/1e3,.05);X=a,O&&(U+=d*.08),H+=(U-H)*.05,z+=(P-z)*.05,R+=(_-R)*.05;const u=R*Math.cos(z)*Math.sin(H),be=R*Math.sin(z),Ae=R*Math.cos(z)*Math.cos(H),Re=t.width/t.height,Me=We(.9,Re,.1,100),Ee=qe([u,be,Ae],[0,0,0],[0,1,0]),D=ce(Me,Ee),Se=D[12]/D[15],Ce=D[13]/D[15],we=[Se*.5+.5,Ce*.5+.5],ye=Math.atan(.5/R)/(.9/2),Pe=he(G)/(R*.5),ie=a/1e3;e.clear(e.COLOR_BUFFER_BIT|e.DEPTH_BUFFER_BIT),e.depthMask(!1),e.disable(e.DEPTH_TEST),e.useProgram(n),e.bindBuffer(e.ARRAY_BUFFER,x),e.enableVertexAttribArray(S.aPos),e.vertexAttribPointer(S.aPos,2,e.FLOAT,!1,0,0),e.uniform2fv(S.uBHScreen,we),e.uniform1f(S.uBHRadius,ye),e.uniform1f(S.uLensStrength,Pe),e.uniform1f(S.uTime,ie),e.uniform1f(S.uSpinStar,I);const Te=pe(G,q,R);e.uniform1f(S.uGWStrain,Math.min(Te,1)),e.drawArrays(e.TRIANGLE_STRIP,0,4),e.disableVertexAttribArray(S.aPos),e.depthMask(!0),e.enable(e.DEPTH_TEST);const Be=Ye(.12),Fe=ce(D,Be);e.useProgram(v),e.bindBuffer(e.ARRAY_BUFFER,A),e.bindBuffer(e.ELEMENT_ARRAY_BUFFER,y);const ae=20;e.enableVertexAttribArray(p.aPosition),e.vertexAttribPointer(p.aPosition,3,e.FLOAT,!1,ae,0),p.aTexCoord>=0&&(e.enableVertexAttribArray(p.aTexCoord),e.vertexAttribPointer(p.aTexCoord,2,e.FLOAT,!1,ae,12)),e.uniformMatrix4fv(p.uMVP,!1,Fe),e.uniform1f(p.uTime,ie),e.uniform1f(p.uInnerR,m),e.uniform1f(p.uOuterR,g),e.uniform1f(p.uSpinDisk,I),e.enable(e.BLEND),e.blendFunc(e.SRC_ALPHA,e.ONE_MINUS_SRC_ALPHA),e.drawElements(e.TRIANGLES,w.indices.length,e.UNSIGNED_SHORT,0),e.disableVertexAttribArray(p.aPosition),p.aTexCoord>=0&&e.disableVertexAttribArray(p.aTexCoord);for(let h=b.length-1;h>=0;h--){const o=b[h];if(o.life+=d,o.life>o.maxLife){b[h]=W(o.type,m,g);continue}const c=Math.sqrt(o.x*o.x+o.z*o.z)||.1;if(o.type==="accretion"){const C=1/(c*Math.sqrt(c))*.5,N=-o.z/c,Le=o.x/c,se=I*.005/(c*c+.1),le=.008/(c+.1);if(o.vx=N*(C+se)-o.x/c*le,o.vz=Le*(C+se)-o.z/c*le,c<m*1.2){const J=Math.max(.1,(c-.2)/m);o.cr*=J,o.cg*=J,o.cb*=J}}else if(o.type==="jet")o.vy*=1.008,o.vx*=.99,o.vz*=.99;else if(o.type==="tidal"){const C=.15/(c*c+.01);o.vx-=o.x/c*C*d,o.vz-=o.z/c*C*d,o.size*=1.01,o.cr=Math.min(1,o.cr+d*.3),o.cg=Math.min(1,o.cg+d*.2),o.cb=Math.min(1,o.cb+d*.15)}else if(o.type==="penrose")(o.vx*o.x+o.vz*o.z)/c>0?(o.vx*=.998,o.vz*=.998):(o.vx-=o.x/c*.1*d,o.vz-=o.z/c*.1*d);else{const C=.05/(c*c+.01);o.vx-=o.x/c*C*d,o.vz-=o.z/c*C*d;const N=I*.003/(c*c+.1);o.vx+=-o.z/c*N*d,o.vz+=o.x/c*N*d}o.x+=o.vx*d,o.y+=o.vy*d,o.z+=o.vz*d;const E=Math.sqrt(o.x*o.x+o.y*o.y+o.z*o.z);(E<.15||E>18)&&(b[h]=W(o.type,m,g))}for(;b.length<Y;){const h=Math.random(),o=h<.4?"accretion":h<.58?"jet":h<.75?"infalling":h<.88?"tidal":"penrose";b.push(W(o,m,g))}const T=new Float32Array(b.length*7);for(let h=0;h<b.length;h++){const o=b[h],c=Math.min(1,Math.min(o.life/.3,(o.maxLife-o.life)/.5)),E=h*7;T[E]=o.x,T[E+1]=o.y,T[E+2]=o.z,T[E+3]=o.cr*c,T[E+4]=o.cg*c,T[E+5]=o.cb*c,T[E+6]=o.size*(.5+c*.5)}e.useProgram(f),e.bindBuffer(e.ARRAY_BUFFER,xe),e.bufferData(e.ARRAY_BUFFER,T,e.DYNAMIC_DRAW);const j=28;e.enableVertexAttribArray(M.aPosition),e.vertexAttribPointer(M.aPosition,3,e.FLOAT,!1,j,0),e.enableVertexAttribArray(M.aColor),e.vertexAttribPointer(M.aColor,3,e.FLOAT,!1,j,12),e.enableVertexAttribArray(M.aSize),e.vertexAttribPointer(M.aSize,1,e.FLOAT,!1,j,24),e.uniformMatrix4fv(M.uMVP,!1,D),e.blendFunc(e.SRC_ALPHA,e.ONE),e.drawArrays(e.POINTS,0,b.length),e.disableVertexAttribArray(M.aPosition),e.disableVertexAttribArray(M.aColor),e.disableVertexAttribArray(M.aSize),e.blendFunc(e.SRC_ALPHA,e.ONE_MINUS_SRC_ALPHA),K=Q(R,G,I,q),i&&i(K),requestAnimationFrame(re)}return requestAnimationFrame(a=>{X=a,re(a)}),{destroy:()=>{ne=!1,oe.disconnect()},getHUD:()=>K}}const $=document.getElementById("sim-canvas"),tt=document.getElementById("hud-F"),ot=document.getElementById("hud-omega"),nt=document.getElementById("hud-IC"),rt=document.getElementById("hud-kappa"),it=document.getElementById("hud-S"),at=document.getElementById("hud-delta"),st=document.getElementById("hud-gamma"),lt=document.getElementById("hud-redshift"),ct=document.getElementById("hud-vesc"),dt=document.getElementById("hud-hawking"),ft=document.getElementById("hud-dist"),ut=document.getElementById("hud-spin"),mt=document.getElementById("hud-drag"),gt=document.getElementById("hud-penrose"),ht=document.getElementById("hud-time"),pt=document.getElementById("hud-entropy"),vt=document.getElementById("hud-prec"),xt=document.getElementById("hud-gw"),bt=document.getElementById("hud-surfg"),At=document.getElementById("hud-radeff"),ue=document.getElementById("hud-regime"),Rt=document.getElementById("regime-bar");let me=0;function Mt(t){const i=performance.now();if(i-me<66)return;me=i,tt.textContent=t.F.toFixed(4),ot.textContent=t.omega.toFixed(4),nt.textContent=t.IC.toFixed(4),rt.textContent=t.kappa.toFixed(4),it.textContent=t.S.toFixed(4),at.textContent=t.delta.toFixed(4),st.textContent=t.gamma.toFixed(3),lt.textContent=t.redshift.toFixed(4),ct.textContent=t.escapeV.toFixed(4)+" c",dt.textContent=t.hawkingT.toFixed(6),ft.textContent=t.distance.toFixed(1)+" r_s",ut.textContent=t.spinStar.toFixed(3),mt.textContent=t.frameDrag.toFixed(4),gt.textContent=(t.penroseEff*100).toFixed(1)+"%",ht.textContent=t.timeFactor.toFixed(4),pt.textContent=t.entropy.toFixed(3),vt.textContent=t.precession.toFixed(4)+" rad",xt.textContent=t.gwStrainVal.toExponential(2),bt.textContent=t.surfGravity.toFixed(4),At.textContent=(t.radEfficiency*100).toFixed(1)+"%";const s=t.regime.split(" ")[0];ue.textContent=t.regime,ue.className="hud-value regime-"+s.toLowerCase();const e={STABLE:"#059669",WATCH:"#d97706",COLLAPSE:"#dc2626"};Rt.style.backgroundColor=e[s]||"#333"}et($,Mt);const V=document.getElementById("controls-overlay");let ee=setTimeout(()=>V.classList.add("hidden"),6e3);$.addEventListener("mousedown",()=>{clearTimeout(ee),V.classList.add("hidden")});$.addEventListener("touchstart",()=>{clearTimeout(ee),V.classList.add("hidden")});V.addEventListener("click",()=>{clearTimeout(ee),V.classList.add("hidden")});

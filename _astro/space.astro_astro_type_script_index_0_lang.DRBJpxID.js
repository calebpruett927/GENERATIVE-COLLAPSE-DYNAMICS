import{c as ge,g as we,a as Te}from"./kernel.Cdj69J8o.js";import{B as I,r as Re,p as ze,s as Fe,i as ve,o as Ie,b as Le,a as _e,f as Be,h as ke,e as De,g as Ve}from"./spacetime.BgGuuTMt.js";import"./constants.6x37F0HI.js";function ee(){return new Float32Array(16)}function Oe(){const t=ee();return t[0]=t[5]=t[10]=t[15]=1,t}function Ue(t,a,l,e){const r=ee(),s=1/Math.tan(t*.5),n=1/(l-e);return r[0]=s/a,r[5]=s,r[10]=(e+l)*n,r[11]=-1,r[14]=2*e*l*n,r}function He(t,a){const l=ee();for(let e=0;e<4;e++)for(let r=0;r<4;r++)l[r*4+e]=t[e]*a[r*4]+t[4+e]*a[r*4+1]+t[8+e]*a[r*4+2]+t[12+e]*a[r*4+3];return l}function N(t){const a=Math.sqrt(t[0]*t[0]+t[1]*t[1]+t[2]*t[2])||1;return[t[0]/a,t[1]/a,t[2]/a]}function Y(t,a){return[t[1]*a[2]-t[2]*a[1],t[2]*a[0]-t[0]*a[2],t[0]*a[1]-t[1]*a[0]]}function Ge(t,a,l){const e=N([t[0]-a[0],t[1]-a[1],t[2]-a[2]]),r=N(Y(l,e)),s=Y(e,r),n=Oe();return n[0]=r[0],n[4]=r[1],n[8]=r[2],n[12]=-(r[0]*t[0]+r[1]*t[1]+r[2]*t[2]),n[1]=s[0],n[5]=s[1],n[9]=s[2],n[13]=-(s[0]*t[0]+s[1]*t[1]+s[2]*t[2]),n[2]=e[0],n[6]=e[1],n[10]=e[2],n[14]=-(e[0]*t[0]+e[1]*t[1]+e[2]*t[2]),n[3]=0,n[7]=0,n[11]=0,n[15]=1,n}const qe=`
attribute vec2 aPos;
void main() {
  gl_Position = vec4(aPos, 0.0, 1.0);
}
`,Ne=`
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
#define STEPS  350
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

    // Tighter capture: r < RS * 0.35 for sharper shadow boundary
    if (r < RS * 0.35) { captured = true; break; }
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

    // Adaptive step — much finer near horizon for crisp shadow edge
    float h_near = clamp(0.06 * (r - RS * 0.3), 0.008, 0.35);
    float h_far  = 1.0;
    float step   = mix(h_near, h_far, smoothstep(2.5, 6.0, r));

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
`,Ye=`
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
`,Ke=`
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
`;function We(t){const a=t*t,l=1+Math.cbrt(Math.max(1-a,1e-12))*(Math.cbrt(1+t)+Math.cbrt(Math.max(1-t,1e-6))),e=Math.sqrt(3*a+l*l);return(3+e-Math.sqrt(Math.max((3-l)*(3+l+2*e),0)))*.5}function H(t,a,l){if(t==="accretion"){const e=a+Math.random()*(l-a),r=Math.random()*Math.PI*2,s=1/Math.sqrt(e),n=.02*Math.random();return{x:e*Math.cos(r)*(1+n),y:(Math.random()-.5)*.04,z:e*Math.sin(r),vx:-s*Math.sin(r)*.3,vy:0,vz:s*Math.cos(r)*.3,life:0,maxLife:4+Math.random()*6,size:2+Math.random()*3,type:t,cr:1,cg:.7+Math.random()*.3,cb:.2+Math.random()*.3}}else if(t==="jet"){const e=Math.random()>.5?1:-1,r=.08,s=Math.sin(Date.now()*.001)*.03;return{x:(Math.random()-.5)*r+s,y:e*a*.3,z:(Math.random()-.5)*r,vx:(Math.random()-.5)*.04,vy:e*(1.8+Math.random()*.8),vz:(Math.random()-.5)*.04,life:0,maxLife:1.5+Math.random()*2.5,size:1.5+Math.random()*2,type:t,cr:.3+Math.random()*.3,cg:.5+Math.random()*.4,cb:1}}else if(t==="tidal"){const e=Math.random()*Math.PI*2,r=a*(1+Math.random()*.5);return{x:r*Math.cos(e),y:(Math.random()-.5)*.02,z:r*Math.sin(e),vx:-Math.cos(e)*.4,vy:(Math.random()-.5)*.1,vz:-Math.sin(e)*.4,life:0,maxLife:1+Math.random()*1.5,size:1+Math.random()*1.5,type:t,cr:1,cg:.3,cb:.1}}else if(t==="penrose"){const e=Math.random()*Math.PI*2,r=a*1.1,s=Math.random()>.5;return{x:r*Math.cos(e),y:(Math.random()-.5)*.05,z:r*Math.sin(e),vx:(s?1:-1)*Math.cos(e)*.5,vy:s?.3:-.1,vz:(s?1:-1)*Math.sin(e)*.5,life:0,maxLife:s?3:.8,size:s?3:1.5,type:t,cr:s?.2:.8,cg:s?.8:.2,cb:s?1:.3}}else{const e=Math.random()*Math.PI*2,r=l*(1.2+Math.random()*.5),s=.08*(Math.random()-.3);return{x:r*Math.cos(e),y:(Math.random()-.5)*.3,z:r*Math.sin(e),vx:-Math.cos(e)*.15+s*-Math.sin(e),vy:0,vz:-Math.sin(e)*.15+s*Math.cos(e),life:0,maxLife:4+Math.random()*4,size:1.5+Math.random()*2.5,type:t,cr:.9,cg:.5,cb:.2}}}function Z(t,a,l,e){const n=Math.min(.98,.01+(1-Math.min(t/20,1))*(a-.01)),y=we(n),h=ge(I[0].c,I[0].w),A=Te(h);return{omega:n,gamma:y,regime:A.regime+(A.isCritical?" (Critical)":""),redshift:Ve(n),escapeV:De(n),hawkingT:ke(h.kappa),distance:t,F:h.F,IC:h.IC,kappa:h.kappa,S:h.S,C:h.C,delta:h.delta,frameDrag:Be(n,l),penroseEff:_e(l),entropy:Le(h.kappa),precession:Ie(n),gwStrainVal:ve(a,e,t),surfGravity:Fe(n),timeFactor:ze(n),radEfficiency:Re(l),spinStar:l}}function Xe(t,a){const l=t.getContext("webgl",{antialias:!1,alpha:!1,premultipliedAlpha:!1,powerPreference:"high-performance"});if(!l)return console.error("WebGL not available"),{destroy:()=>{},getHUD:()=>Z(10,.95,.7,0)};const e=l;e.getExtension("OES_standard_derivatives"),e.enable(e.DEPTH_TEST),e.enable(e.BLEND),e.blendFunc(e.SRC_ALPHA,e.ONE_MINUS_SRC_ALPHA),e.clearColor(0,0,0,1);function r(i,m){const d=e.createShader(i);return e.shaderSource(d,m),e.compileShader(d),e.getShaderParameter(d,e.COMPILE_STATUS)||console.error("Shader:",e.getShaderInfoLog(d)),d}function s(i,m){const d=e.createProgram();return e.attachShader(d,r(e.VERTEX_SHADER,i)),e.attachShader(d,r(e.FRAGMENT_SHADER,m)),e.linkProgram(d),e.getProgramParameter(d,e.LINK_STATUS)||console.error("Link:",e.getProgramInfoLog(d)),d}const n=s(qe,Ne),y=s(Ye,Ke),h=e.createBuffer();e.bindBuffer(e.ARRAY_BUFFER,h),e.bufferData(e.ARRAY_BUFFER,new Float32Array([-1,-1,1,-1,-1,1,1,1]),e.STATIC_DRAW);const A=.95,E=.7,K=I[0].c.length>3?ge(I[0].c,I[0].w).C:0,S=We(E),_=S*5.5,xe=800,p=[];for(let i=0;i<450;i++){const m=i<240?"accretion":i<340?"jet":"infalling",d=H(m,S,_);d.life=Math.random()*d.maxLife,p.push(d)}const be=e.createBuffer();let B=0,z=.35,b=8,P=!1,w=0,T=0,k=!0,D=0,C=.35,F=8;t.addEventListener("mousedown",i=>{P=!0,k=!1,w=i.clientX,T=i.clientY,t.style.cursor="grabbing"}),t.addEventListener("mousemove",i=>{P&&(D+=(i.clientX-w)*.005,C+=(i.clientY-T)*.005,C=Math.max(-Math.PI*.45,Math.min(Math.PI*.45,C)),w=i.clientX,T=i.clientY)}),t.addEventListener("mouseup",()=>{P=!1,t.style.cursor="grab"}),t.addEventListener("mouseleave",()=>{P=!1,t.style.cursor="grab"}),t.addEventListener("touchstart",i=>{P=!0,k=!1,w=i.touches[0].clientX,T=i.touches[0].clientY,i.preventDefault()},{passive:!1}),t.addEventListener("touchmove",i=>{P&&(D+=(i.touches[0].clientX-w)*.005,C+=(i.touches[0].clientY-T)*.005,C=Math.max(-Math.PI*.45,Math.min(Math.PI*.45,C)),w=i.touches[0].clientX,T=i.touches[0].clientY,i.preventDefault())},{passive:!1}),t.addEventListener("touchend",()=>{P=!1}),t.addEventListener("wheel",i=>{F+=i.deltaY*.01,F=Math.max(2,Math.min(25,F)),i.preventDefault()},{passive:!1}),t.addEventListener("dblclick",()=>{k=!0,C=.35,F=8}),t.style.cursor="grab";const u={aPos:e.getAttribLocation(n,"aPos"),uResolution:e.getUniformLocation(n,"uResolution"),uTime:e.getUniformLocation(n,"uTime"),uSpinStar:e.getUniformLocation(n,"uSpinStar"),uCamPos:e.getUniformLocation(n,"uCamPos"),uCamRight:e.getUniformLocation(n,"uCamRight"),uCamUp:e.getUniformLocation(n,"uCamUp"),uCamFwd:e.getUniformLocation(n,"uCamFwd"),uFovTan:e.getUniformLocation(n,"uFovTan"),uGWStrain:e.getUniformLocation(n,"uGWStrain")},g={aPosition:e.getAttribLocation(y,"aPosition"),aColor:e.getAttribLocation(y,"aColor"),aSize:e.getAttribLocation(y,"aSize"),uMVP:e.getUniformLocation(y,"uMVP")},re=.9;function ne(){const i=Math.min(window.devicePixelRatio||1,3),m=t.clientWidth,d=t.clientHeight;t.width=Math.round(m*i),t.height=Math.round(d*i),e.viewport(0,0,t.width,t.height)}ne();const ae=new ResizeObserver(ne);ae.observe(t);let W=Z(b,A,E,K),ie=!0,X=0;function se(i){if(!ie)return;const m=Math.min((i-X)/1e3,.05);X=i,k&&(D+=m*.08),B+=(D-B)*.05,z+=(C-z)*.05,b+=(F-b)*.05;const d=b*Math.cos(z)*Math.sin(B),ce=b*Math.sin(z),le=b*Math.cos(z)*Math.cos(B),V=[d,ce,le],R=N([-d,-ce,-le]);let me=[0,1,0];Math.abs(R[1])>.99&&(me=[0,0,1]);const O=N(Y(R,me)),j=Y(O,R),Ce=t.width/t.height,Me=Ue(re,Ce,.1,100),ye=Ge(V,[0,0,0],[0,1,0]),Se=He(Me,ye),Pe=i/1e3,Ae=Math.min(ve(A,K,b),1);e.clear(e.COLOR_BUFFER_BIT|e.DEPTH_BUFFER_BIT),e.depthMask(!1),e.disable(e.DEPTH_TEST),e.useProgram(n),e.bindBuffer(e.ARRAY_BUFFER,h),e.enableVertexAttribArray(u.aPos),e.vertexAttribPointer(u.aPos,2,e.FLOAT,!1,0,0),e.uniform2f(u.uResolution,t.width,t.height),e.uniform1f(u.uTime,Pe),e.uniform1f(u.uSpinStar,E),e.uniform3f(u.uCamPos,V[0],V[1],V[2]),e.uniform3f(u.uCamRight,O[0],O[1],O[2]),e.uniform3f(u.uCamUp,j[0],j[1],j[2]),e.uniform3f(u.uCamFwd,R[0],R[1],R[2]),e.uniform1f(u.uFovTan,1/Math.tan(re*.5)),e.uniform1f(u.uGWStrain,Ae),e.drawArrays(e.TRIANGLE_STRIP,0,4),e.disableVertexAttribArray(u.aPos),e.depthMask(!0),e.enable(e.DEPTH_TEST);for(let f=p.length-1;f>=0;f--){const o=p[f];if(o.life+=m,o.life>o.maxLife){p[f]=H(o.type,S,_);continue}const c=Math.sqrt(o.x*o.x+o.z*o.z)||.1;if(o.type==="accretion"){const x=1/(c*Math.sqrt(c))*.5,U=-o.z/c,Ee=o.x/c,de=E*.005/(c*c+.1),fe=.008/(c+.1);if(o.vx=U*(x+de)-o.x/c*fe,o.vz=Ee*(x+de)-o.z/c*fe,c<S*1.2){const Q=Math.max(.1,(c-.2)/S);o.cr*=Q,o.cg*=Q,o.cb*=Q}}else if(o.type==="jet")o.vy*=1.008,o.vx*=.99,o.vz*=.99;else if(o.type==="tidal"){const x=.15/(c*c+.01);o.vx-=o.x/c*x*m,o.vz-=o.z/c*x*m,o.size*=1.01,o.cr=Math.min(1,o.cr+m*.3),o.cg=Math.min(1,o.cg+m*.2),o.cb=Math.min(1,o.cb+m*.15)}else if(o.type==="penrose")(o.vx*o.x+o.vz*o.z)/c>0?(o.vx*=.998,o.vz*=.998):(o.vx-=o.x/c*.1*m,o.vz-=o.z/c*.1*m);else{const x=.05/(c*c+.01);o.vx-=o.x/c*x*m,o.vz-=o.z/c*x*m;const U=E*.003/(c*c+.1);o.vx+=-o.z/c*U*m,o.vz+=o.x/c*U*m}o.x+=o.vx*m,o.y+=o.vy*m,o.z+=o.vz*m;const v=Math.sqrt(o.x*o.x+o.y*o.y+o.z*o.z);(v<.15||v>20)&&(p[f]=H(o.type,S,_))}for(;p.length<xe;){const f=Math.random(),o=f<.4?"accretion":f<.58?"jet":f<.75?"infalling":f<.88?"tidal":"penrose";p.push(H(o,S,_))}const M=new Float32Array(p.length*7);for(let f=0;f<p.length;f++){const o=p[f],c=Math.min(1,Math.min(o.life/.3,(o.maxLife-o.life)/.5)),v=f*7;M[v]=o.x,M[v+1]=o.y,M[v+2]=o.z,M[v+3]=o.cr*c,M[v+4]=o.cg*c,M[v+5]=o.cb*c,M[v+6]=o.size*(.5+c*.5)}e.useProgram(y),e.bindBuffer(e.ARRAY_BUFFER,be),e.bufferData(e.ARRAY_BUFFER,M,e.DYNAMIC_DRAW);const J=28;e.enableVertexAttribArray(g.aPosition),e.vertexAttribPointer(g.aPosition,3,e.FLOAT,!1,J,0),e.enableVertexAttribArray(g.aColor),e.vertexAttribPointer(g.aColor,3,e.FLOAT,!1,J,12),e.enableVertexAttribArray(g.aSize),e.vertexAttribPointer(g.aSize,1,e.FLOAT,!1,J,24),e.uniformMatrix4fv(g.uMVP,!1,Se),e.blendFunc(e.SRC_ALPHA,e.ONE),e.drawArrays(e.POINTS,0,p.length),e.disableVertexAttribArray(g.aPosition),e.disableVertexAttribArray(g.aColor),e.disableVertexAttribArray(g.aSize),e.blendFunc(e.SRC_ALPHA,e.ONE_MINUS_SRC_ALPHA),W=Z(b,A,E,K),a&&a(W),requestAnimationFrame(se)}return requestAnimationFrame(i=>{X=i,se(i)}),{destroy:()=>{ie=!1,ae.disconnect()},getHUD:()=>W}}const te=document.getElementById("sim-canvas"),je=document.getElementById("hud-F"),Je=document.getElementById("hud-omega"),Qe=document.getElementById("hud-IC"),Ze=document.getElementById("hud-kappa"),$e=document.getElementById("hud-S"),et=document.getElementById("hud-delta"),tt=document.getElementById("hud-gamma"),ot=document.getElementById("hud-redshift"),rt=document.getElementById("hud-vesc"),nt=document.getElementById("hud-hawking"),at=document.getElementById("hud-dist"),it=document.getElementById("hud-spin"),st=document.getElementById("hud-drag"),ct=document.getElementById("hud-penrose"),lt=document.getElementById("hud-time"),mt=document.getElementById("hud-entropy"),dt=document.getElementById("hud-prec"),ft=document.getElementById("hud-gw"),ht=document.getElementById("hud-surfg"),ut=document.getElementById("hud-radeff"),he=document.getElementById("hud-regime"),ue=document.getElementById("regime-bar");let pe=0,$="";const G=document.querySelector("#hud-left .hud-panel"),q=document.querySelector("#hud-right .hud-panel");function pt(t){const a=performance.now();if(a-pe<66)return;pe=a,je.textContent=t.F.toFixed(4),Je.textContent=t.omega.toFixed(4),Qe.textContent=t.IC.toFixed(4),Ze.textContent=t.kappa.toFixed(4),$e.textContent=t.S.toFixed(4),et.textContent=t.delta.toFixed(4),tt.textContent=t.gamma.toFixed(3),ot.textContent=t.redshift.toFixed(4),rt.textContent=t.escapeV.toFixed(4)+" c",nt.textContent=t.hawkingT.toFixed(6),at.textContent=t.distance.toFixed(1)+" r_s",it.textContent=t.spinStar.toFixed(3),st.textContent=t.frameDrag.toFixed(4),ct.textContent=(t.penroseEff*100).toFixed(1)+"%",lt.textContent=t.timeFactor.toFixed(4),mt.textContent=t.entropy.toFixed(3),dt.textContent=t.precession.toFixed(4)+" rad",ft.textContent=t.gwStrainVal.toExponential(2),ht.textContent=t.surfGravity.toFixed(4),ut.textContent=(t.radEfficiency*100).toFixed(1)+"%";const l=t.regime.split(" ")[0];he.textContent=t.regime,he.className="hud-value regime-"+l.toLowerCase();const r={STABLE:"#059669",WATCH:"#d97706",COLLAPSE:"#dc2626"}[l]||"#333";if(ue.style.backgroundColor=r,ue.style.boxShadow="0 -2px 16px "+r+"80",l!==$&&$!==""){const s=r+"60";G.style.borderColor=r,q.style.borderColor=r,G.style.boxShadow="0 0 20px "+s,q.style.boxShadow="0 0 20px "+s,setTimeout(()=>{G.style.borderColor="",q.style.borderColor="",G.style.boxShadow="",q.style.boxShadow=""},1200)}$=l}Xe(te,pt);const L=document.getElementById("controls-overlay");let oe=setTimeout(()=>L.classList.add("hidden"),6e3);te.addEventListener("mousedown",()=>{clearTimeout(oe),L.classList.add("hidden")});te.addEventListener("touchstart",()=>{clearTimeout(oe),L.classList.add("hidden")});L.addEventListener("click",()=>{clearTimeout(oe),L.classList.add("hidden")});

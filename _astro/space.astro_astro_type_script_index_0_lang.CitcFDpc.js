import{c as gt,g as wt,a as Tt}from"./kernel.C-ROa8s-.js";import{B as I,r as Rt,p as Ft,s as zt,i as vt,o as It,b as Lt,a as _t,f as Bt,h as kt,e as Dt,g as Vt}from"./spacetime.CHHUQeVA.js";import"./constants.6x37F0HI.js";function tt(){return new Float32Array(16)}function Ot(){const e=tt();return e[0]=e[5]=e[10]=e[15]=1,e}function Ut(e,a,l,t){const r=tt(),s=1/Math.tan(e*.5),n=1/(l-t);return r[0]=s/a,r[5]=s,r[10]=(t+l)*n,r[11]=-1,r[14]=2*t*l*n,r}function Ht(e,a){const l=tt();for(let t=0;t<4;t++)for(let r=0;r<4;r++)l[r*4+t]=e[t]*a[r*4]+e[4+t]*a[r*4+1]+e[8+t]*a[r*4+2]+e[12+t]*a[r*4+3];return l}function N(e){const a=Math.sqrt(e[0]*e[0]+e[1]*e[1]+e[2]*e[2])||1;return[e[0]/a,e[1]/a,e[2]/a]}function Y(e,a){return[e[1]*a[2]-e[2]*a[1],e[2]*a[0]-e[0]*a[2],e[0]*a[1]-e[1]*a[0]]}function Gt(e,a,l){const t=N([e[0]-a[0],e[1]-a[1],e[2]-a[2]]),r=N(Y(l,t)),s=Y(t,r),n=Ot();return n[0]=r[0],n[4]=r[1],n[8]=r[2],n[12]=-(r[0]*e[0]+r[1]*e[1]+r[2]*e[2]),n[1]=s[0],n[5]=s[1],n[9]=s[2],n[13]=-(s[0]*e[0]+s[1]*e[1]+s[2]*e[2]),n[2]=t[0],n[6]=t[1],n[10]=t[2],n[14]=-(t[0]*e[0]+t[1]*e[1]+t[2]*e[2]),n[3]=0,n[7]=0,n[11]=0,n[15]=1,n}const qt=`
attribute vec2 aPos;
void main() {
  gl_Position = vec4(aPos, 0.0, 1.0);
}
`,Nt=`
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
`,Yt=`
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
`,Kt=`
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
`;function Wt(e){const a=e*e,l=1+Math.cbrt(Math.max(1-a,1e-12))*(Math.cbrt(1+e)+Math.cbrt(Math.max(1-e,1e-6))),t=Math.sqrt(3*a+l*l);return(3+t-Math.sqrt(Math.max((3-l)*(3+l+2*t),0)))*.5}function H(e,a,l){if(e==="accretion"){const t=a+Math.random()*(l-a),r=Math.random()*Math.PI*2,s=1/Math.sqrt(t),n=.02*Math.random();return{x:t*Math.cos(r)*(1+n),y:(Math.random()-.5)*.04,z:t*Math.sin(r),vx:-s*Math.sin(r)*.3,vy:0,vz:s*Math.cos(r)*.3,life:0,maxLife:4+Math.random()*6,size:2+Math.random()*3,type:e,cr:1,cg:.7+Math.random()*.3,cb:.2+Math.random()*.3}}else if(e==="jet"){const t=Math.random()>.5?1:-1,r=.08,s=Math.sin(Date.now()*.001)*.03;return{x:(Math.random()-.5)*r+s,y:t*a*.3,z:(Math.random()-.5)*r,vx:(Math.random()-.5)*.04,vy:t*(1.8+Math.random()*.8),vz:(Math.random()-.5)*.04,life:0,maxLife:1.5+Math.random()*2.5,size:1.5+Math.random()*2,type:e,cr:.3+Math.random()*.3,cg:.5+Math.random()*.4,cb:1}}else if(e==="tidal"){const t=Math.random()*Math.PI*2,r=a*(1+Math.random()*.5);return{x:r*Math.cos(t),y:(Math.random()-.5)*.02,z:r*Math.sin(t),vx:-Math.cos(t)*.4,vy:(Math.random()-.5)*.1,vz:-Math.sin(t)*.4,life:0,maxLife:1+Math.random()*1.5,size:1+Math.random()*1.5,type:e,cr:1,cg:.3,cb:.1}}else if(e==="penrose"){const t=Math.random()*Math.PI*2,r=a*1.1,s=Math.random()>.5;return{x:r*Math.cos(t),y:(Math.random()-.5)*.05,z:r*Math.sin(t),vx:(s?1:-1)*Math.cos(t)*.5,vy:s?.3:-.1,vz:(s?1:-1)*Math.sin(t)*.5,life:0,maxLife:s?3:.8,size:s?3:1.5,type:e,cr:s?.2:.8,cg:s?.8:.2,cb:s?1:.3}}else{const t=Math.random()*Math.PI*2,r=l*(1.2+Math.random()*.5),s=.08*(Math.random()-.3);return{x:r*Math.cos(t),y:(Math.random()-.5)*.3,z:r*Math.sin(t),vx:-Math.cos(t)*.15+s*-Math.sin(t),vy:0,vz:-Math.sin(t)*.15+s*Math.cos(t),life:0,maxLife:4+Math.random()*4,size:1.5+Math.random()*2.5,type:e,cr:.9,cg:.5,cb:.2}}}function Z(e,a,l,t){const n=Math.min(.98,.01+(1-Math.min(e/20,1))*(a-.01)),y=wt(n),h=gt(I[0].c,I[0].w),A=Tt(h);return{omega:n,gamma:y,regime:A.regime+(A.isCritical?" (Critical)":""),redshift:Vt(n),escapeV:Dt(n),hawkingT:kt(h.kappa),distance:e,F:h.F,IC:h.IC,kappa:h.kappa,S:h.S,C:h.C,delta:h.delta,frameDrag:Bt(n,l),penroseEff:_t(l),entropy:Lt(h.kappa),precession:It(n),gwStrainVal:vt(a,t,e),surfGravity:zt(n),timeFactor:Ft(n),radEfficiency:Rt(l),spinStar:l}}function Xt(e,a){const l=e.getContext("webgl",{antialias:!1,alpha:!1,premultipliedAlpha:!1,powerPreference:"high-performance"});if(!l)return console.error("WebGL not available"),{destroy:()=>{},getHUD:()=>Z(10,.95,.7,0)};const t=l;t.getExtension("OES_standard_derivatives"),t.enable(t.DEPTH_TEST),t.enable(t.BLEND),t.blendFunc(t.SRC_ALPHA,t.ONE_MINUS_SRC_ALPHA),t.clearColor(0,0,0,1);function r(i,m){const d=t.createShader(i);return t.shaderSource(d,m),t.compileShader(d),t.getShaderParameter(d,t.COMPILE_STATUS)||console.error("Shader:",t.getShaderInfoLog(d)),d}function s(i,m){const d=t.createProgram();return t.attachShader(d,r(t.VERTEX_SHADER,i)),t.attachShader(d,r(t.FRAGMENT_SHADER,m)),t.linkProgram(d),t.getProgramParameter(d,t.LINK_STATUS)||console.error("Link:",t.getProgramInfoLog(d)),d}const n=s(qt,Nt),y=s(Yt,Kt),h=t.createBuffer();t.bindBuffer(t.ARRAY_BUFFER,h),t.bufferData(t.ARRAY_BUFFER,new Float32Array([-1,-1,1,-1,-1,1,1,1]),t.STATIC_DRAW);const A=.95,E=.7,K=I[0].c.length>3?gt(I[0].c,I[0].w).C:0,S=Wt(E),_=S*5.5,xt=800,p=[];for(let i=0;i<450;i++){const m=i<240?"accretion":i<340?"jet":"infalling",d=H(m,S,_);d.life=Math.random()*d.maxLife,p.push(d)}const bt=t.createBuffer();let B=0,F=.35,b=8,P=!1,w=0,T=0,k=!0,D=0,C=.35,z=8;e.addEventListener("mousedown",i=>{P=!0,k=!1,w=i.clientX,T=i.clientY,e.style.cursor="grabbing"}),e.addEventListener("mousemove",i=>{P&&(D+=(i.clientX-w)*.005,C+=(i.clientY-T)*.005,C=Math.max(-Math.PI*.45,Math.min(Math.PI*.45,C)),w=i.clientX,T=i.clientY)}),e.addEventListener("mouseup",()=>{P=!1,e.style.cursor="grab"}),e.addEventListener("mouseleave",()=>{P=!1,e.style.cursor="grab"}),e.addEventListener("touchstart",i=>{P=!0,k=!1,w=i.touches[0].clientX,T=i.touches[0].clientY,i.preventDefault()},{passive:!1}),e.addEventListener("touchmove",i=>{P&&(D+=(i.touches[0].clientX-w)*.005,C+=(i.touches[0].clientY-T)*.005,C=Math.max(-Math.PI*.45,Math.min(Math.PI*.45,C)),w=i.touches[0].clientX,T=i.touches[0].clientY,i.preventDefault())},{passive:!1}),e.addEventListener("touchend",()=>{P=!1}),e.addEventListener("wheel",i=>{z+=i.deltaY*.01,z=Math.max(2,Math.min(25,z)),i.preventDefault()},{passive:!1}),e.addEventListener("dblclick",()=>{k=!0,C=.35,z=8}),e.style.cursor="grab";const u={aPos:t.getAttribLocation(n,"aPos"),uResolution:t.getUniformLocation(n,"uResolution"),uTime:t.getUniformLocation(n,"uTime"),uSpinStar:t.getUniformLocation(n,"uSpinStar"),uCamPos:t.getUniformLocation(n,"uCamPos"),uCamRight:t.getUniformLocation(n,"uCamRight"),uCamUp:t.getUniformLocation(n,"uCamUp"),uCamFwd:t.getUniformLocation(n,"uCamFwd"),uFovTan:t.getUniformLocation(n,"uFovTan"),uGWStrain:t.getUniformLocation(n,"uGWStrain")},g={aPosition:t.getAttribLocation(y,"aPosition"),aColor:t.getAttribLocation(y,"aColor"),aSize:t.getAttribLocation(y,"aSize"),uMVP:t.getUniformLocation(y,"uMVP")},rt=.9;function nt(){const i=Math.min(window.devicePixelRatio||1,2.5),m=e.clientWidth,d=e.clientHeight;e.width=Math.round(m*i),e.height=Math.round(d*i),t.viewport(0,0,e.width,e.height)}nt();const at=new ResizeObserver(nt);at.observe(e);let W=Z(b,A,E,K),it=!0,X=0;function st(i){if(!it)return;const m=Math.min((i-X)/1e3,.05);X=i,k&&(D+=m*.08),B+=(D-B)*.05,F+=(C-F)*.05,b+=(z-b)*.05;const d=b*Math.cos(F)*Math.sin(B),ct=b*Math.sin(F),lt=b*Math.cos(F)*Math.cos(B),V=[d,ct,lt],R=N([-d,-ct,-lt]);let mt=[0,1,0];Math.abs(R[1])>.99&&(mt=[0,0,1]);const O=N(Y(R,mt)),j=Y(O,R),Ct=e.width/e.height,Mt=Ut(rt,Ct,.1,100),yt=Gt(V,[0,0,0],[0,1,0]),St=Ht(Mt,yt),Pt=i/1e3,At=Math.min(vt(A,K,b),1);t.clear(t.COLOR_BUFFER_BIT|t.DEPTH_BUFFER_BIT),t.depthMask(!1),t.disable(t.DEPTH_TEST),t.useProgram(n),t.bindBuffer(t.ARRAY_BUFFER,h),t.enableVertexAttribArray(u.aPos),t.vertexAttribPointer(u.aPos,2,t.FLOAT,!1,0,0),t.uniform2f(u.uResolution,e.width,e.height),t.uniform1f(u.uTime,Pt),t.uniform1f(u.uSpinStar,E),t.uniform3f(u.uCamPos,V[0],V[1],V[2]),t.uniform3f(u.uCamRight,O[0],O[1],O[2]),t.uniform3f(u.uCamUp,j[0],j[1],j[2]),t.uniform3f(u.uCamFwd,R[0],R[1],R[2]),t.uniform1f(u.uFovTan,1/Math.tan(rt*.5)),t.uniform1f(u.uGWStrain,At),t.drawArrays(t.TRIANGLE_STRIP,0,4),t.disableVertexAttribArray(u.aPos),t.depthMask(!0),t.enable(t.DEPTH_TEST);for(let f=p.length-1;f>=0;f--){const o=p[f];if(o.life+=m,o.life>o.maxLife){p[f]=H(o.type,S,_);continue}const c=Math.sqrt(o.x*o.x+o.z*o.z)||.1;if(o.type==="accretion"){const x=1/(c*Math.sqrt(c))*.5,U=-o.z/c,Et=o.x/c,dt=E*.005/(c*c+.1),ft=.008/(c+.1);if(o.vx=U*(x+dt)-o.x/c*ft,o.vz=Et*(x+dt)-o.z/c*ft,c<S*1.2){const Q=Math.max(.1,(c-.2)/S);o.cr*=Q,o.cg*=Q,o.cb*=Q}}else if(o.type==="jet")o.vy*=1.008,o.vx*=.99,o.vz*=.99;else if(o.type==="tidal"){const x=.15/(c*c+.01);o.vx-=o.x/c*x*m,o.vz-=o.z/c*x*m,o.size*=1.01,o.cr=Math.min(1,o.cr+m*.3),o.cg=Math.min(1,o.cg+m*.2),o.cb=Math.min(1,o.cb+m*.15)}else if(o.type==="penrose")(o.vx*o.x+o.vz*o.z)/c>0?(o.vx*=.998,o.vz*=.998):(o.vx-=o.x/c*.1*m,o.vz-=o.z/c*.1*m);else{const x=.05/(c*c+.01);o.vx-=o.x/c*x*m,o.vz-=o.z/c*x*m;const U=E*.003/(c*c+.1);o.vx+=-o.z/c*U*m,o.vz+=o.x/c*U*m}o.x+=o.vx*m,o.y+=o.vy*m,o.z+=o.vz*m;const v=Math.sqrt(o.x*o.x+o.y*o.y+o.z*o.z);(v<.15||v>20)&&(p[f]=H(o.type,S,_))}for(;p.length<xt;){const f=Math.random(),o=f<.4?"accretion":f<.58?"jet":f<.75?"infalling":f<.88?"tidal":"penrose";p.push(H(o,S,_))}const M=new Float32Array(p.length*7);for(let f=0;f<p.length;f++){const o=p[f],c=Math.min(1,Math.min(o.life/.3,(o.maxLife-o.life)/.5)),v=f*7;M[v]=o.x,M[v+1]=o.y,M[v+2]=o.z,M[v+3]=o.cr*c,M[v+4]=o.cg*c,M[v+5]=o.cb*c,M[v+6]=o.size*(.5+c*.5)}t.useProgram(y),t.bindBuffer(t.ARRAY_BUFFER,bt),t.bufferData(t.ARRAY_BUFFER,M,t.DYNAMIC_DRAW);const J=28;t.enableVertexAttribArray(g.aPosition),t.vertexAttribPointer(g.aPosition,3,t.FLOAT,!1,J,0),t.enableVertexAttribArray(g.aColor),t.vertexAttribPointer(g.aColor,3,t.FLOAT,!1,J,12),t.enableVertexAttribArray(g.aSize),t.vertexAttribPointer(g.aSize,1,t.FLOAT,!1,J,24),t.uniformMatrix4fv(g.uMVP,!1,St),t.blendFunc(t.SRC_ALPHA,t.ONE),t.drawArrays(t.POINTS,0,p.length),t.disableVertexAttribArray(g.aPosition),t.disableVertexAttribArray(g.aColor),t.disableVertexAttribArray(g.aSize),t.blendFunc(t.SRC_ALPHA,t.ONE_MINUS_SRC_ALPHA),W=Z(b,A,E,K),a&&a(W),requestAnimationFrame(st)}return requestAnimationFrame(i=>{X=i,st(i)}),{destroy:()=>{it=!1,at.disconnect()},getHUD:()=>W}}const et=document.getElementById("sim-canvas"),jt=document.getElementById("hud-F"),Jt=document.getElementById("hud-omega"),Qt=document.getElementById("hud-IC"),Zt=document.getElementById("hud-kappa"),$t=document.getElementById("hud-S"),te=document.getElementById("hud-delta"),ee=document.getElementById("hud-gamma"),oe=document.getElementById("hud-redshift"),re=document.getElementById("hud-vesc"),ne=document.getElementById("hud-hawking"),ae=document.getElementById("hud-dist"),ie=document.getElementById("hud-spin"),se=document.getElementById("hud-drag"),ce=document.getElementById("hud-penrose"),le=document.getElementById("hud-time"),me=document.getElementById("hud-entropy"),de=document.getElementById("hud-prec"),fe=document.getElementById("hud-gw"),he=document.getElementById("hud-surfg"),ue=document.getElementById("hud-radeff"),ht=document.getElementById("hud-regime"),ut=document.getElementById("regime-bar");let pt=0,$="";const G=document.querySelector("#hud-left .hud-panel"),q=document.querySelector("#hud-right .hud-panel");function pe(e){const a=performance.now();if(a-pt<66)return;pt=a,jt.textContent=e.F.toFixed(4),Jt.textContent=e.omega.toFixed(4),Qt.textContent=e.IC.toFixed(4),Zt.textContent=e.kappa.toFixed(4),$t.textContent=e.S.toFixed(4),te.textContent=e.delta.toFixed(4),ee.textContent=e.gamma.toFixed(3),oe.textContent=e.redshift.toFixed(4),re.textContent=e.escapeV.toFixed(4)+" c",ne.textContent=e.hawkingT.toFixed(6),ae.textContent=e.distance.toFixed(1)+" r_s",ie.textContent=e.spinStar.toFixed(3),se.textContent=e.frameDrag.toFixed(4),ce.textContent=(e.penroseEff*100).toFixed(1)+"%",le.textContent=e.timeFactor.toFixed(4),me.textContent=e.entropy.toFixed(3),de.textContent=e.precession.toFixed(4)+" rad",fe.textContent=e.gwStrainVal.toExponential(2),he.textContent=e.surfGravity.toFixed(4),ue.textContent=(e.radEfficiency*100).toFixed(1)+"%";const l=e.regime.split(" ")[0];ht.textContent=e.regime,ht.className="hud-value regime-"+l.toLowerCase();const r={STABLE:"#059669",WATCH:"#d97706",COLLAPSE:"#dc2626"}[l]||"#333";if(ut.style.backgroundColor=r,ut.style.boxShadow="0 -2px 16px "+r+"80",l!==$&&$!==""){const s=r+"60";G.style.borderColor=r,q.style.borderColor=r,G.style.boxShadow="0 0 20px "+s,q.style.boxShadow="0 0 20px "+s,setTimeout(()=>{G.style.borderColor="",q.style.borderColor="",G.style.boxShadow="",q.style.boxShadow=""},1200)}$=l}Xt(et,pe);const L=document.getElementById("controls-overlay");let ot=setTimeout(()=>L.classList.add("hidden"),6e3);et.addEventListener("mousedown",()=>{clearTimeout(ot),L.classList.add("hidden")});et.addEventListener("touchstart",()=>{clearTimeout(ot),L.classList.add("hidden")});L.addEventListener("click",()=>{clearTimeout(ot),L.classList.add("hidden")});

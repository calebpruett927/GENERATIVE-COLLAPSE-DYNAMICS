function B(n){return Math.min(1-1e-8,Math.max(1e-8,n))}function C(n,a){const l=n.length,s=a||n.map(()=>1/l),e=n.reduce((r,o,c)=>r+s[c]*o,0),t=1-e,d=n.map(r=>B(r)),i=d.reduce((r,o,c)=>r+s[c]*Math.log(o),0),$=Math.exp(i),f=-d.reduce((r,o,c)=>r+s[c]*(o*Math.log(o)+(1-o)*Math.log(1-o)),0),w=e,S=n.reduce((r,o,c)=>r+s[c]*(o-w)**2,0),h=Math.sqrt(S)/.5;let m="Watch",u="text-yellow-400";return t>=.3?(m="Collapse",u="text-red-400"):t<.038&&e>.9&&f<.15&&h<.14&&(m="Stable",u="text-green-400"),{F:e,omega:t,S:f,C:h,kappa:i,IC:$,regime:m,regimeColor:u}}const M=[{symbol:"F",name:"Fidelitas",latin:"quid supersit post collapsum",formula:"F(t) = Σ wᵢ cᵢ(t)",desc:"Fidelity — what survives collapse. The weighted arithmetic mean of the trace vector. Measures how much of the original signal is retained.",note:"Primitive. F is the first degree of freedom."},{symbol:"ω",name:"Derivatio",latin:"quantum collapsu deperdatur",formula:"ω(t) = 1 − F(t)",desc:"Drift — how much is lost to collapse. The complement of fidelity.",note:"Derived from F. The duality identity F + ω = 1 holds exactly (residual 0.0e+00)."},{symbol:"κ",name:"Log-Integritas",latin:"sensibilitas logarithmica",formula:"κ(t) = Σ wᵢ ln(c_{i,ε}(t))",desc:"Log-integrity — the logarithmic sensitivity of coherence. Always ≤ 0. Any channel near 0 drags κ toward -∞.",note:"Primitive. κ is the second degree of freedom."},{symbol:"IC",name:"Integritas Composita",latin:"cohaerentia multiplicativa",formula:"IC(t) = exp(κ(t))",desc:"Composite integrity — multiplicative coherence. The weighted geometric mean of the trace vector. IC ≤ F always (the integrity bound).",note:"Derived from κ. The log-integrity relation IC = exp(κ) links multiplicative and additive coherence."},{symbol:"S",name:"Entropia",latin:"incertitudo campi collapsus",formula:"S(t) = −Σ wᵢ [c ln(c) + (1−c) ln(1−c)]",desc:"Bernoulli field entropy — the uncertainty of the collapse field. This is NOT Shannon entropy. Shannon is the degenerate limit when the collapse field is removed.",note:"Primitive but not free. S is asymptotically determined by F and C (corr → −1 as n → ∞). Computed, not an independent DOF."},{symbol:"C",name:"Curvatura",latin:"coniunctio cum gradibus libertatis",formula:"C(t) = StdDev({cᵢ}) / 0.5",desc:"Curvature — coupling to uncontrolled degrees of freedom. Measures channel heterogeneity, normalized to [0,1].",note:"Primitive. C is the third degree of freedom. Together, F, κ, C span the full kernel."}],p=document.getElementById("invariant-sections");p&&M.forEach((n,a)=>{const l=["F","κ","S","C"].includes(n.symbol),s=l?"bg-amber-500":"bg-blue-400",e=l?"border-amber-500/20":"border-blue-400/20",t=document.createElement("div");t.className=`bg-kernel-900/30 border ${e} rounded-xl p-5 inv-section cursor-pointer transition hover:bg-kernel-900/50`,t.innerHTML=`
        <div class="flex items-start gap-3">
          <span class="w-2 h-2 rounded-full ${s} mt-2 shrink-0"></span>
          <div class="flex-1">
            <div class="flex items-center gap-3 mb-1">
              <span class="text-xl font-mono font-bold text-kernel-100">${n.symbol}</span>
              <span class="text-sm font-bold text-kernel-300">${n.name}</span>
              <span class="text-xs text-kernel-600 italic hidden md:inline">— ${n.latin}</span>
            </div>
            <div class="font-mono text-xs text-amber-400/80 bg-kernel-950/60 inline-block px-2 py-1 rounded mb-2">${n.formula}</div>
            <p class="text-sm text-kernel-400 leading-relaxed">${n.desc}</p>
            <p class="text-xs text-kernel-600 mt-2 italic">${n.note}</p>
          </div>
        </div>
      `,p.appendChild(t)});const x=document.createElement("div");x.className="bg-kernel-900/30 border border-kernel-800/20 rounded-xl p-5";x.innerHTML=`
    <div class="flex items-start gap-3">
      <span class="w-2 h-2 rounded-full bg-kernel-500 mt-2 shrink-0"></span>
      <div class="flex-1">
        <div class="flex items-center gap-3 mb-1">
          <span class="text-xl font-mono font-bold text-kernel-100">τ<sub>R</sub></span>
          <span class="text-sm font-bold text-kernel-300">Moratio Reditus</span>
          <span class="text-xs text-kernel-600 italic hidden md:inline">— tempus reentrandi</span>
        </div>
        <p class="text-sm text-kernel-400 leading-relaxed">
          Return time — the delay before re-entry into the return domain D<sub>θ</sub>.
          If no prior state within tolerance η exists, τ<sub>R</sub> = ∞<sub>rec</sub> (permanent no-return).
          This is a typed outcome, not an error. <em>Si τ<sub>R</sub> = ∞<sub>rec</sub>, nulla fides datur.</em>
        </p>
        <p class="text-xs text-kernel-600 mt-2 italic">Temporal invariant. Not computed from a single trace snapshot — requires history.</p>
      </div>
    </div>
  `;p?.appendChild(x);const F=document.getElementById("verify-btn"),L=document.getElementById("verify-results"),v=document.getElementById("invariant-outputs"),b=document.getElementById("identity-checks");F?.addEventListener("click",()=>{const n=document.getElementById("verify-channels").value,a=document.getElementById("verify-weights").value,l=n.split(",").map(t=>parseFloat(t.trim())).filter(t=>!isNaN(t));if(l.length<2)return;let s;a.trim()&&(s=a.split(",").map(t=>parseFloat(t.trim())).filter(t=>!isNaN(t)),s.length!==l.length&&(s=void 0));const e=C(l,s);if(v&&(v.innerHTML=[{label:"F",value:e.F.toFixed(4),color:"text-kernel-100"},{label:"ω",value:e.omega.toFixed(4),color:"text-kernel-100"},{label:"S",value:e.S.toFixed(4),color:"text-kernel-100"},{label:"C",value:e.C.toFixed(4),color:"text-kernel-100"},{label:"κ",value:e.kappa.toFixed(4),color:"text-kernel-100"},{label:"IC",value:e.IC.toFixed(4),color:"text-kernel-100"}].map(t=>`
        <div class="bg-kernel-950/80 rounded-lg p-3 text-center">
          <div class="text-xs text-kernel-500 font-mono">${t.label}</div>
          <div class="text-lg font-bold font-mono ${t.color}">${t.value}</div>
        </div>
      `).join("")),b){const t=Math.abs(e.F+e.omega-1),d=e.IC<=e.F+1e-12,i=Math.abs(e.IC-Math.exp(e.kappa));b.innerHTML=`
        <div class="flex items-center gap-3 p-3 bg-kernel-950/60 rounded-lg">
          <span class="${t<1e-10?"text-green-400":"text-red-400"} text-lg">
            ${t<1e-10?"✓":"✗"}
          </span>
          <div class="flex-1">
            <span class="text-sm text-kernel-200 font-medium">F + ω = 1</span>
            <span class="text-xs text-kernel-500 ml-2">Duality identity (complementum perfectum)</span>
          </div>
          <span class="text-xs font-mono text-kernel-400">residual: ${t.toExponential(1)}</span>
        </div>
        <div class="flex items-center gap-3 p-3 bg-kernel-950/60 rounded-lg">
          <span class="${d?"text-green-400":"text-red-400"} text-lg">
            ${d?"✓":"✗"}
          </span>
          <div class="flex-1">
            <span class="text-sm text-kernel-200 font-medium">IC ≤ F</span>
            <span class="text-xs text-kernel-500 ml-2">Integrity bound (limbus integritatis)</span>
          </div>
          <span class="text-xs font-mono text-kernel-400">Δ = ${(e.F-e.IC).toFixed(4)}</span>
        </div>
        <div class="flex items-center gap-3 p-3 bg-kernel-950/60 rounded-lg">
          <span class="${i<1e-10?"text-green-400":"text-red-400"} text-lg">
            ${i<1e-10?"✓":"✗"}
          </span>
          <div class="flex-1">
            <span class="text-sm text-kernel-200 font-medium">IC = exp(κ)</span>
            <span class="text-xs text-kernel-500 ml-2">Log-integrity relation</span>
          </div>
          <span class="text-xs font-mono text-kernel-400">residual: ${i.toExponential(1)}</span>
        </div>
        <div class="flex items-center gap-3 p-3 bg-kernel-950/60 rounded-lg">
          <span class="${e.regimeColor} text-lg">◉</span>
          <div class="flex-1">
            <span class="text-sm text-kernel-200 font-medium">Regime: <span class="${e.regimeColor} font-bold">${e.regime}</span></span>
            <span class="text-xs text-kernel-500 ml-2">Derived from four-gate criterion</span>
          </div>
        </div>
      `}L?.classList.remove("hidden")});F?.click();const I=document.getElementById("slaughter-n"),E=document.getElementById("slaughter-dead"),y=document.getElementById("slaughter-n-label"),k=document.getElementById("slaughter-dead-label");function g(){const n=parseInt(I?.value||"8"),a=parseInt(E?.value||"1")/100;y&&(y.textContent=`${n} channels`),k&&(k.textContent=a.toFixed(2));const l=Array(n-1).fill(1).concat([a]),s=C(l),e=document.getElementById("sl-F"),t=document.getElementById("sl-IC"),d=document.getElementById("sl-gap"),i=document.getElementById("sl-ratio");e&&(e.textContent=s.F.toFixed(4)),t&&(t.textContent=s.IC.toFixed(4)),d&&(d.textContent=(s.F-s.IC).toFixed(4)),i&&(i.textContent=(s.IC/s.F).toFixed(4))}I?.addEventListener("input",g);E?.addEventListener("input",g);g();document.querySelectorAll(".toc-link").forEach(n=>{n.addEventListener("click",a=>{const l=n.getAttribute("href");l?.startsWith("#")&&(a.preventDefault(),document.querySelector(l)?.scrollIntoView({behavior:"smooth",block:"start"}))})});

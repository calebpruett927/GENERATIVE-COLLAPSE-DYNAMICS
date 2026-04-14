function A(t){return Math.min(1-1e-8,Math.max(1e-8,t))}function M(t,o){const s=t.length,l=o||t.map(()=>1/s),e=t.reduce((i,d,g)=>i+l[g]*d,0),n=1-e,c=t.map(i=>A(i)),r=c.reduce((i,d,g)=>i+l[g]*Math.log(d),0),x=Math.exp(r),m=-c.reduce((i,d,g)=>i+l[g]*(d*Math.log(d)+(1-d)*Math.log(1-d)),0),v=e,p=t.reduce((i,d,g)=>i+l[g]*(d-v)**2,0),u=Math.sqrt(p)/.5;let f="Watch",a="text-yellow-400";return n>=.3?(f="Collapse",a="text-red-400"):n<.038&&e>.9&&m<.15&&u<.14&&(f="Stable",a="text-green-400"),{F:e,omega:n,S:m,C:u,kappa:r,IC:x,regime:f,regimeColor:a}}const R=[{symbol:"F",name:"Fidelity",latin:"quid supersit post collapsum",formula:"F(t) = Σ wᵢ cᵢ(t)",desc:"Fidelity — what survives collapse. The weighted arithmetic mean of the trace vector.",note:"Primitive. F is the first degree of freedom."},{symbol:"ω",name:"Drift",latin:"quantum collapsu deperdatur",formula:"ω(t) = 1 − F(t)",desc:"Drift — how much is lost to collapse. The complement of fidelity.",note:"Derived from F. The duality identity F + ω = 1 holds exactly (residual 0.0e+00)."},{symbol:"κ",name:"Log-Integrity",latin:"sensibilitas logarithmica",formula:"κ(t) = Σ wᵢ ln(c_{i,ε}(t))",desc:"Log-integrity — the logarithmic sensitivity of coherence. Always ≤ 0. Any channel near 0 drags κ toward −∞.",note:"Primitive. κ is the second degree of freedom."},{symbol:"IC",name:"Integrity Composite",latin:"cohaerentia multiplicativa",formula:"IC(t) = exp(κ(t))",desc:"Integrity composite — multiplicative coherence. The weighted geometric mean. IC ≤ F always (the integrity bound).",note:"Derived from κ. The log-integrity relation IC = exp(κ) links multiplicative and additive coherence."},{symbol:"S",name:"Entropy",latin:"incertitudo campi collapsus",formula:"S(t) = −Σ wᵢ [c ln(c) + (1−c) ln(1−c)]",desc:"Bernoulli field entropy — the uncertainty of the collapse field. Not Shannon entropy (that is the degenerate limit).",note:"Computed, not free. Asymptotically determined by F and C (corr → −1 as n → ∞). 3 effective DOF, not 4."},{symbol:"C",name:"Curvature",latin:"coniunctio cum gradibus libertatis",formula:"C(t) = StdDev({cᵢ}) / 0.5",desc:"Curvature — coupling to uncontrolled degrees of freedom. Measures channel heterogeneity, normalized to [0,1].",note:"Primitive. C is the third degree of freedom. Together, F, κ, C span the full kernel."}],I=document.getElementById("invariant-sections");I&&R.forEach(t=>{const o=["F","κ","S","C"].includes(t.symbol),s=o?"bg-amber-500":"bg-blue-400",l=o?"border-amber-500/20":"border-blue-400/20",e=document.createElement("div");e.className=`bg-kernel-900/30 border ${l} rounded-xl p-5 transition hover:bg-kernel-900/50`,e.innerHTML=`
        <div class="flex items-start gap-3">
          <span class="w-2 h-2 rounded-full ${s} mt-2 shrink-0"></span>
          <div class="flex-1">
            <div class="flex items-center gap-3 mb-1">
              <span class="text-xl font-mono font-bold text-kernel-100">${t.symbol}</span>
              <span class="text-sm font-bold text-kernel-300">${t.name}</span>
              <span class="text-xs text-kernel-600 italic hidden md:inline">— ${t.latin}</span>
            </div>
            <div class="font-mono text-xs text-amber-400/80 bg-kernel-950/60 inline-block px-2 py-1 rounded mb-2">${t.formula}</div>
            <p class="text-sm text-kernel-400 leading-relaxed">${t.desc}</p>
            <p class="text-xs text-kernel-600 mt-2 italic">${t.note}</p>
          </div>
        </div>
      `,I.appendChild(e)});const N=document.getElementById("verify-btn"),q=document.getElementById("verify-results"),k=document.getElementById("invariant-outputs"),E=document.getElementById("identity-checks");N?.addEventListener("click",()=>{const t=document.getElementById("verify-channels").value,o=document.getElementById("verify-weights").value,s=t.split(",").map(n=>parseFloat(n.trim())).filter(n=>!isNaN(n));if(s.length<2)return;let l;o.trim()&&(l=o.split(",").map(n=>parseFloat(n.trim())).filter(n=>!isNaN(n)),l.length!==s.length&&(l=void 0));const e=M(s,l);if(k&&(k.innerHTML=[{label:"F",value:e.F.toFixed(4),color:"text-kernel-100"},{label:"ω",value:e.omega.toFixed(4),color:"text-kernel-100"},{label:"S",value:e.S.toFixed(4),color:"text-kernel-100"},{label:"C",value:e.C.toFixed(4),color:"text-kernel-100"},{label:"κ",value:e.kappa.toFixed(4),color:"text-kernel-100"},{label:"IC",value:e.IC.toFixed(4),color:"text-kernel-100"}].map(n=>`
        <div class="bg-kernel-950/80 rounded-lg p-3 text-center">
          <div class="text-xs text-kernel-500 font-mono">${n.label}</div>
          <div class="text-lg font-bold font-mono ${n.color}">${n.value}</div>
        </div>
      `).join("")),E){const n=Math.abs(e.F+e.omega-1),c=e.IC<=e.F+1e-12,r=Math.abs(e.IC-Math.exp(e.kappa));E.innerHTML=`
        <div class="flex items-center gap-3 p-3 bg-kernel-950/60 rounded-lg">
          <span class="${n<1e-10?"text-green-400":"text-red-400"} text-lg">
            ${n<1e-10?"✓":"✗"}
          </span>
          <div class="flex-1">
            <span class="text-sm text-kernel-200 font-medium">F + ω = 1</span>
            <span class="text-xs text-kernel-500 ml-2">Duality identity</span>
          </div>
          <span class="text-xs font-mono text-kernel-400">residual: ${n.toExponential(1)}</span>
        </div>
        <div class="flex items-center gap-3 p-3 bg-kernel-950/60 rounded-lg">
          <span class="${c?"text-green-400":"text-red-400"} text-lg">
            ${c?"✓":"✗"}
          </span>
          <div class="flex-1">
            <span class="text-sm text-kernel-200 font-medium">IC ≤ F</span>
            <span class="text-xs text-kernel-500 ml-2">Integrity bound</span>
          </div>
          <span class="text-xs font-mono text-kernel-400">Δ = ${(e.F-e.IC).toFixed(4)}</span>
        </div>
        <div class="flex items-center gap-3 p-3 bg-kernel-950/60 rounded-lg">
          <span class="${r<1e-10?"text-green-400":"text-red-400"} text-lg">
            ${r<1e-10?"✓":"✗"}
          </span>
          <div class="flex-1">
            <span class="text-sm text-kernel-200 font-medium">IC = exp(κ)</span>
            <span class="text-xs text-kernel-500 ml-2">Log-integrity relation</span>
          </div>
          <span class="text-xs font-mono text-kernel-400">residual: ${r.toExponential(1)}</span>
        </div>
        <div class="flex items-center gap-3 p-3 bg-kernel-950/60 rounded-lg">
          <span class="${e.regimeColor} text-lg">◉</span>
          <div class="flex-1">
            <span class="text-sm text-kernel-200 font-medium">Regime: <span class="${e.regimeColor} font-bold">${e.regime}</span></span>
            <span class="text-xs text-kernel-500 ml-2">Derived from four-gate criterion</span>
          </div>
        </div>
      `}q?.classList.remove("hidden")});N?.click();const D=document.getElementById("slaughter-n"),T=document.getElementById("slaughter-dead"),$=document.getElementById("slaughter-n-label"),B=document.getElementById("slaughter-dead-label");function F(){const t=parseInt(D?.value||"8"),o=parseInt(T?.value||"1")/100;$&&($.textContent=`${t} channels`),B&&(B.textContent=o.toFixed(2));const s=Array(t-1).fill(1).concat([o]),l=M(s),e=document.getElementById("sl-F"),n=document.getElementById("sl-IC"),c=document.getElementById("sl-gap"),r=document.getElementById("sl-ratio");e&&(e.textContent=l.F.toFixed(4)),n&&(n.textContent=l.IC.toFixed(4)),c&&(c.textContent=(l.F-l.IC).toFixed(4)),r&&(r.textContent=(l.IC/l.F).toFixed(4))}D?.addEventListener("input",F);T?.addEventListener("input",F);F();const P=document.getElementById("gamma-omega"),S=document.getElementById("gamma-omega-label"),w=document.getElementById("gamma-value"),h=document.getElementById("gamma-trapped"),y=document.getElementById("gamma-regime");function H(){const t=parseInt(P?.value||"30")/100;S&&(S.textContent=t.toFixed(2));const o=Math.pow(t,3)/(1-t+1e-8),s=t>=.6823;if(w&&(w.textContent=o<100?o.toFixed(4):o.toExponential(2)),h&&(h.textContent=s?"YES":"No",h.className=`text-lg font-bold font-mono ${s?"text-red-400":"text-green-400"}`),y){let l="Watch",e="text-yellow-400";t>=.3?(l="Collapse",e="text-red-400"):t<.038&&(l="Stable",e="text-green-400"),y.textContent=l,y.className=`text-lg font-bold font-mono ${e}`}}P?.addEventListener("input",H);H();const O=document.getElementById("diag-btn"),L=document.getElementById("diag-channels"),b=document.getElementById("diag-results"),C=document.getElementById("diag-outputs");function V(){if(!L||!b||!C)return;const t=L.value.split(",").map(a=>parseFloat(a.trim())).filter(a=>!isNaN(a)&&a>=0&&a<=1);if(t.length<2){C.innerHTML='<p class="text-red-400 text-xs col-span-4">Need at least 2 valid channels (0-1)</p>',b.classList.remove("hidden");return}const o=1e-8,s=t.length,l=1/s,e=t.map(a=>Math.max(a,o)),n=e.reduce((a,i)=>a+l*i,0),c=1-n,r=e.reduce((a,i)=>a+l*Math.log(i),0),x=Math.exp(r),m=n-x,v=x/n;let p="Homogeneous",u="text-green-400";m>=.05?(p="Fragmented",u="text-red-400"):m>=.01?(p="Heterogeneous",u="text-yellow-400"):m>=1e-6&&(p="Coherent",u="text-blue-400");const f=[{name:"F",value:n.toFixed(6),color:"text-cyan-400"},{name:"ω",value:c.toFixed(6),color:"text-amber-400"},{name:"IC",value:x.toFixed(6),color:"text-purple-400"},{name:"Δ",value:m.toFixed(6),color:"text-red-400"},{name:"ρ",value:v.toFixed(4),color:"text-green-400"},{name:"κ",value:r.toFixed(6),color:"text-blue-400"},{name:"Label",value:p,color:u},{name:"n",value:String(s),color:"text-kernel-300"}];C.innerHTML=f.map(a=>`<div class="bg-kernel-950/60 rounded-lg p-2 text-center"><p class="text-[10px] text-kernel-500">${a.name}</p><p class="font-mono text-xs ${a.color}">${a.value}</p></div>`).join(""),b.classList.remove("hidden")}O?.addEventListener("click",V);document.querySelectorAll(".toc-link").forEach(t=>{t.addEventListener("click",o=>{const s=t.getAttribute("href");s?.startsWith("#")&&(o.preventDefault(),document.querySelector(s)?.scrollIntoView({behavior:"smooth",block:"start"}))})});

const a=document.querySelector('meta[name="base-url"]')?.getAttribute("content")||"/GENERATIVE-COLLAPSE-DYNAMICS/",y=[{sym:"F",name:"Fidelity",latin:"Fidelitas",formula:"F = Σ wᵢcᵢ",range:"[0, 1]",tier:1,desc:"What survives collapse — arithmetic mean of trace vector weighted by channel weights.",role:"Primitive"},{sym:"ω",name:"Drift",latin:"Derivatio",formula:"ω = 1 − F",range:"[0, 1]",tier:1,desc:"What is lost to collapse — measured departure from fidelity.",role:"Derived (from F)"},{sym:"S",name:"Bernoulli Field Entropy",latin:"Entropia",formula:"S = −Σ wᵢ[cᵢ ln cᵢ + (1−cᵢ) ln(1−cᵢ)]",range:"[0, ∞)",tier:1,desc:"Uncertainty of the collapse field. Shannon entropy is the degenerate limit.",role:"Primitive (computed, not free)"},{sym:"C",name:"Curvature",latin:"Curvatura",formula:"C = stddev(cᵢ) / 0.5",range:"[0, 1]",tier:1,desc:"Coupling to uncontrolled degrees of freedom — normalized standard deviation.",role:"Primitive (independent)"},{sym:"κ",name:"Log-Integrity",latin:"Log-Integritas",formula:"κ = Σ wᵢ ln(cᵢ,ε)",range:"(−∞, 0]",tier:1,desc:"Logarithmic sensitivity of coherence — one dead channel sends κ → −∞.",role:"Primitive"},{sym:"IC",name:"Composite Integrity",latin:"Integritas Composita",formula:"IC = exp(κ)",range:"(0, 1]",tier:1,desc:"Multiplicative coherence — the geometric mean. IC ≤ F always (integrity bound).",role:"Derived (from κ)"}],x=[{param:"ε",code:"EPSILON",value:"10⁻⁸",purpose:"Guard band — pole at ω=1 does not affect measurements to machine precision",source:"frozen_contract.EPSILON"},{param:"p",code:"P_EXPONENT",value:"3",purpose:"Unique integer where ω_trap is a Cardano root of x³ + x − 1 = 0",source:"frozen_contract.P_EXPONENT"},{param:"α",code:"ALPHA",value:"1.0",purpose:"Curvature cost coefficient (unit coupling) in D_C = α · C",source:"frozen_contract.ALPHA"},{param:"λ",code:"LAMBDA",value:"0.2",purpose:"Auxiliary coefficient",source:"frozen_contract.LAMBDA"},{param:"tol_seam",code:"TOL_SEAM",value:"0.005",purpose:"Seam residual tolerance: |s| ≤ tol for PASS — width where IC ≤ F holds at 100%",source:"frozen_contract.TOL_SEAM"},{param:"c*",code:"C_STAR",value:"0.7822",purpose:"Logistic self-dual fixed point: maximizes S + κ per channel",source:"constants"},{param:"c_trap",code:"OMEGA_TRAP",value:"0.3178",purpose:"Cardano root — the mirror of c* across equator (c_trap = 1 − c*)",source:"constants"}],g=[{regime:"Stable",condition:"ω < 0.038 AND F > 0.90 AND S < 0.15 AND C < 0.14",share:"12.5%",color:"#22c55e",note:"Conjunctive — ALL gates must be satisfied. Stability is rare."},{regime:"Watch",condition:"0.038 ≤ ω < 0.30 (or Stable gates not all met)",share:"24.4%",color:"#f59e0b",note:"Intermediate zone — some drift but not yet collapsed."},{regime:"Collapse",condition:"ω ≥ 0.30",share:"63.1%",color:"#ef4444",note:"Most of the manifold. Return from collapse is what the axiom measures."},{regime:"Critical (overlay)",condition:"IC < 0.30",share:"—",color:"#a855f7",note:"Severity overlay — accompanies any regime. Flags dangerously low integrity."}],v=[{lens:"Epistemology",drift:"Change in belief/evidence",fidelity:"Retained warrant",roughness:"Inference friction",return:"Justified re-entry",color:"#f59e0b"},{lens:"Ontology",drift:"State transition",fidelity:"Conserved properties",roughness:"Heterogeneity / seams",return:"Restored coherence",color:"#3b82f6"},{lens:"Phenomenology",drift:"Perceived shift",fidelity:"Stable features",roughness:"Distress / effort",return:"Coping that holds",color:"#a855f7"},{lens:"History",drift:"Periodization",fidelity:"What endures",roughness:"Rupture / confound",return:"Restitution",color:"#f43f5e"},{lens:"Policy",drift:"Regime shift",fidelity:"Mandate persistence",roughness:"Friction / cost",return:"Reinstatement",color:"#10b981"},{lens:"Semiotics",drift:"Sign departure",fidelity:"Convention survived",roughness:"Meaning loss",return:"Interpretant closure",color:"#06b6d4"}],o=[{stop:"Contract",role:"Define before evidence",desc:"Freeze sources, normalization, near-wall policy, thresholds. Declares the rules before the sentence is written.",icon:"📜"},{stop:"Canon",role:"Narrate with 5 words",desc:"Tell the story using Drift, Fidelity, Roughness, Return, Integrity. Prose-first, auditable by construction.",icon:"📖"},{stop:"Closures",role:"Publish thresholds",desc:"No mid-episode edits. Version the sheet. Stance must change when thresholds are crossed.",icon:"🔒"},{stop:"Integrity Ledger",role:"Debit/credit reconcile",desc:"Debit Drift + Roughness, credit Return. The account must reconcile: residual ≤ tol.",icon:"📊"},{stop:"Stance",role:"Derived verdict",desc:"Read from declared gates: Stable / Watch / Collapse (+ Critical overlay). Never asserted, always derived.",icon:"⚖️"}],h=[{wrong:"Shannon entropy",right:"Bernoulli field entropy",reason:"Shannon is the degenerate limit when the collapse field is removed"},{wrong:"AM-GM inequality",right:"Integrity bound (IC ≤ F)",reason:"Derived independently from Axiom-0; AM-GM is degenerate limit"},{wrong:"AM-GM gap",right:"Heterogeneity gap (Δ = F − IC)",reason:"Measures channel heterogeneity, not an inequality"},{wrong:"rederives / recovers",right:"Derives independently",reason:"Arrow runs from axiom to classical, not reverse"},{wrong:"unitarity",right:"Duality identity F + ω = 1",reason:"Structural identity of collapse, not quantum unitarity"},{wrong:"hyperparameter",right:"Frozen parameter",reason:"Seam-derived, not tuned — trans suturam congelatum"},{wrong:"constant (for frozen params)",right:"Frozen / consistent across the seam",reason:"Discovered by seam, not chosen by convention"}],p={gcd:["E1","E2","E3","E4","B1","B5","D1","D3"],rcft:["E1","E2","E3","B5","B6","D5","D8"],kinematics:["E1","E3","B1","B3","D1"],weyl:["E1","E3","B1","B7","D4"],standard_model:["E1","E2","E3","E4","B1","B3","B5","D3","D5"],nuclear_physics:["E1","E3","B3","B5","D3","D5"],quantum_mechanics:["E1","E2","E3","B1","B5","D1","D3"],atomic_physics:["E1","E2","E3","B1","B3","D3","D5"],materials_science:["E1","E3","B1","B3","D5"],everyday_physics:["E1","E3","B1","D1"],finance:["E1","E3","B1","B5","D1"],security:["E1","E3","B1"],astronomy:["E1","E3","B1","B7","D4"],evolution:["E1","E3","B1","B3","D1","D5"],consciousness_coherence:["E1","E3","B1","B5","D1"],awareness_cognition:["E1","E3","B1","D1","D5"],clinical_neuroscience:["E1","E3","B1","B5","D3"],dynamic_semiotics:["E1","E3","B1","B5","D1"],continuity_theory:["E1","E3","B1","D1","D8"],spacetime_memory:["E1","E3","B1","B7","D4","D5"]},u={gcd:"GCD",rcft:"RCFT",kinematics:"Kinematics",weyl:"WEYL",standard_model:"Std Model",nuclear_physics:"Nuclear",quantum_mechanics:"QM",atomic_physics:"Atomic",materials_science:"Materials",everyday_physics:"Everyday",finance:"Finance",security:"Security",astronomy:"Astronomy",evolution:"Evolution",consciousness_coherence:"Consciousness",awareness_cognition:"Awareness",clinical_neuroscience:"Clinical",dynamic_semiotics:"Semiotics",continuity_theory:"Continuity",spacetime_memory:"Spacetime"};let l="symbols",n="";function b(){return`<div class="grid grid-cols-1 md:grid-cols-2 gap-4">
      ${y.filter(e=>!n||`${e.sym} ${e.name} ${e.latin} ${e.desc}`.toLowerCase().includes(n)).map(e=>`
        <div class="bg-kernel-900 border border-kernel-700 rounded-lg p-5 hover:border-amber-600/40 transition">
          <div class="flex items-center gap-3 mb-2">
            <span class="text-2xl font-mono font-bold text-amber-400">${e.sym}</span>
            <div>
              <div class="text-sm font-bold text-kernel-200">${e.name}</div>
              <div class="text-[10px] text-kernel-500 italic">${e.latin}</div>
            </div>
            <span class="ml-auto text-[10px] px-2 py-0.5 rounded bg-blue-900/30 text-blue-400 border border-blue-700/40">Tier-${e.tier}</span>
          </div>
          <div class="font-mono text-xs text-kernel-300 bg-kernel-800 px-3 py-2 rounded mb-2">${e.formula}</div>
          <div class="text-xs text-kernel-400 mb-2">${e.desc}</div>
          <div class="flex justify-between text-[10px] text-kernel-500">
            <span>Range: <code class="text-kernel-300">${e.range}</code></span>
            <span>${e.role}</span>
          </div>
        </div>
      `).join("")}
    </div>`}function k(){return`<div class="space-y-3">
      ${x.filter(e=>!n||`${e.param} ${e.code} ${e.purpose}`.toLowerCase().includes(n)).map(e=>`
        <div class="bg-kernel-900 border border-kernel-700 rounded-lg p-4 flex items-start gap-4">
          <div class="text-xl font-mono font-bold text-cyan-400 w-16 shrink-0 text-center">${e.param}</div>
          <div class="flex-1">
            <div class="flex items-center gap-2 mb-1">
              <code class="text-xs bg-kernel-800 text-kernel-300 px-2 py-0.5 rounded">${e.code}</code>
              <span class="text-sm font-bold text-amber-400">${e.value}</span>
            </div>
            <div class="text-xs text-kernel-400">${e.purpose}</div>
            <div class="text-[10px] text-kernel-600 mt-1">Source: <code>${e.source}</code></div>
          </div>
        </div>
      `).join("")}
      <div class="text-xs text-kernel-500 mt-4 px-3 py-2 bg-kernel-900/50 rounded border border-kernel-800 italic">
        All parameters are frozen — consistent across the seam (<em>trans suturam congelatum</em>).
        They are discovered by the mathematics, not chosen by convention.
      </div>
    </div>`}function E(){return`<div class="space-y-4">
      ${g.map(t=>`
        <div class="bg-kernel-900 border border-kernel-700 rounded-lg p-5">
          <div class="flex items-center gap-3 mb-2">
            <span class="w-3 h-3 rounded-full" style="background:${t.color}"></span>
            <span class="text-sm font-bold text-kernel-200">${t.regime}</span>
            ${t.share!=="—"?`<span class="ml-auto text-xs text-kernel-500">${t.share} of Fisher space</span>`:""}
          </div>
          <div class="font-mono text-xs text-kernel-300 bg-kernel-800 px-3 py-2 rounded mb-2">${t.condition}</div>
          <div class="text-xs text-kernel-400">${t.note}</div>
        </div>
      `).join("")}
      <div class="bg-kernel-900 border border-amber-700/40 rounded-lg p-4 text-xs text-amber-400/80">
        <strong>Key insight</strong>: Stability is rare — 87.5% of the manifold lies outside the Stable regime.
        Return from collapse to stability is what the axiom measures.
      </div>
    </div>`}function D(){return`<div class="space-y-3">
      <p class="text-xs text-kernel-500 mb-2">
        The Rosetta enables cross-domain reading: same data + same contract → same verdict, different dialect.
        Integrity is never asserted — it is read from the reconciled ledger.
      </p>
      ${v.map(t=>`
        <div class="bg-kernel-900 border border-kernel-700 rounded-lg p-4">
          <div class="flex items-center gap-2 mb-3">
            <span class="w-2 h-2 rounded-full" style="background:${t.color}"></span>
            <span class="text-sm font-bold" style="color:${t.color}">${t.lens}</span>
            <a href="${a}rosetta/" class="ml-auto text-[10px] text-kernel-500 hover:text-kernel-300">Open Rosetta →</a>
          </div>
          <div class="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs">
            <div><span class="text-kernel-500 block text-[10px]">Drift</span><span class="text-kernel-300">${t.drift}</span></div>
            <div><span class="text-kernel-500 block text-[10px]">Fidelity</span><span class="text-kernel-300">${t.fidelity}</span></div>
            <div><span class="text-kernel-500 block text-[10px]">Roughness</span><span class="text-kernel-300">${t.roughness}</span></div>
            <div><span class="text-kernel-500 block text-[10px]">Return</span><span class="text-kernel-300">${t.return}</span></div>
          </div>
        </div>
      `).join("")}
    </div>`}function $(){return`<div class="space-y-1">
      <div class="flex items-center justify-center gap-1 text-xs font-mono text-kernel-500 mb-6 flex-wrap">
        ${o.map((t,e)=>`
          <span class="px-2 py-1 bg-kernel-800 rounded text-kernel-300">${t.stop}</span>
          ${e<o.length-1?'<span class="text-kernel-600">→</span>':""}
        `).join("")}
      </div>
      ${o.map((t,e)=>`
        <div class="bg-kernel-900 border border-kernel-700 rounded-lg p-5 flex items-start gap-4">
          <div class="text-2xl shrink-0">${t.icon}</div>
          <div>
            <div class="flex items-center gap-2 mb-1">
              <span class="text-xs bg-kernel-800 text-kernel-400 px-2 py-0.5 rounded font-mono">Step ${e+1}</span>
              <span class="text-sm font-bold text-kernel-200">${t.stop}</span>
              <span class="text-xs text-kernel-500 italic">— ${t.role}</span>
            </div>
            <div class="text-xs text-kernel-400">${t.desc}</div>
          </div>
        </div>
      `).join("")}
    </div>`}function B(){return`<div class="space-y-2">
      <p class="text-xs text-kernel-500 mb-3">
        Correct terminology is mandatory. GCD derives independently from Axiom-0. Classical results are degenerate limits.
      </p>
      <div class="overflow-x-auto">
        <table class="w-full text-xs">
          <thead>
            <tr class="text-kernel-400 border-b border-kernel-700">
              <th class="text-left py-2 pr-3">❌ WRONG</th>
              <th class="text-left py-2 pr-3">✓ RIGHT</th>
              <th class="text-left py-2">WHY</th>
            </tr>
          </thead>
          <tbody>
            ${h.filter(e=>!n||`${e.wrong} ${e.right} ${e.reason}`.toLowerCase().includes(n)).map(e=>`
              <tr class="border-b border-kernel-800">
                <td class="py-2 pr-3 text-red-400 line-through opacity-70">${e.wrong}</td>
                <td class="py-2 pr-3 text-green-400 font-bold">${e.right}</td>
                <td class="py-2 text-kernel-400">${e.reason}</td>
              </tr>
            `).join("")}
          </tbody>
        </table>
      </div>
    </div>`}function w(){const t=["E1","E2","E3","E4","B1","B3","B5","B6","B7","D1","D3","D4","D5","D8"],e=Object.keys(p),d=document.getElementById("cross-matrix");if(!d)return;let r='<thead><tr><th class="py-1 px-2 text-left text-kernel-500 sticky left-0 bg-kernel-950">Domain</th>';for(const s of t)r+=`<th class="py-1 px-2 text-center text-kernel-500 font-mono">${s}</th>`;r+="</tr></thead><tbody>";for(const s of e){const f=p[s]||[];r+='<tr class="border-b border-kernel-800/50 hover:bg-kernel-900/50">',r+=`<td class="py-1.5 px-2 sticky left-0 bg-kernel-950"><a href="${a}${s}/" class="text-kernel-300 hover:text-amber-400 no-underline">${u[s]}</a></td>`;for(const m of t)f.includes(m)?r+=`<td class="py-1 px-2 text-center"><a href="${a}identities/" class="inline-block w-3 h-3 rounded-full bg-amber-500/70 hover:bg-amber-400 transition no-underline" title="${u[s]} × ${m}"></a></td>`:r+='<td class="py-1 px-2 text-center"><span class="inline-block w-3 h-3 rounded-full bg-kernel-800/50"></span></td>';r+="</tr>"}r+="</tbody>",d.innerHTML=r}const C={symbols:b,frozen:k,regime:E,lenses:D,spine:$,terminology:B};function c(){const t=document.getElementById("ref-content");if(!t)return;const e=C[l];t.innerHTML=e?e():""}document.querySelectorAll(".ref-tab-btn").forEach(t=>t.addEventListener("click",()=>{l=t.dataset.tab,document.querySelectorAll(".ref-tab-btn").forEach(e=>{e.dataset.tab===l?(e.classList.remove("bg-kernel-800","text-kernel-400"),e.classList.add("bg-kernel-700","text-kernel-200")):(e.classList.remove("bg-kernel-700","text-kernel-200"),e.classList.add("bg-kernel-800","text-kernel-400"))}),c()}));const i=document.getElementById("ref-search");i&&i.addEventListener("input",()=>{n=i.value.toLowerCase().trim(),c()});c();w();

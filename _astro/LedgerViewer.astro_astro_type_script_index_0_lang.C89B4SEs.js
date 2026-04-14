import{c as v,a as $}from"./kernel.1iDFzIM1.js";import"./constants.DtcEBpaC.js";const f=1e-8,h={gcd:{name:"GCD",c:[.95,.9,.88,.92,.85,.91],desc:"Core framework"},rcft:{name:"RCFT",c:[.92,.88,.85,.9,.87,.86],desc:"Recursive collapse"},kinematics:{name:"Kinematics",c:[.8,.85,.75,.9,.82,.78],desc:"Motion analysis"},weyl:{name:"Weyl",c:[.7,.65,.6,.72,.68,.66],desc:"Cosmology"},security:{name:"Security",c:[.95,.92,.98,.9,.96,.93],desc:"Audit validation"},astronomy:{name:"Astronomy",c:[.75,.8,.7,.85,.72,.78],desc:"Stellar classification"},nuclear_physics:{name:"Nuclear",c:[.6,.55,.7,.5,.65,.58],desc:"Binding energy"},quantum_mechanics:{name:"Quantum",c:[.85,.8,.75,.9,.82,.78],desc:"Wavefunction"},finance:{name:"Finance",c:[.72,.68,.65,.7,.75,.66],desc:"Portfolio coherence"},atomic_physics:{name:"Atomic",c:[.88,.85,.82,.9,.86,.84],desc:"118 elements"},materials_science:{name:"Materials",c:[.82,.78,.8,.85,.76,.81],desc:"Element database"},everyday_physics:{name:"Everyday",c:[.9,.88,.85,.92,.87,.89],desc:"Thermo, optics"},evolution:{name:"Evolution",c:[.65,.7,.6,.75,.68,.62],desc:"40 organisms"},dynamic_semiotics:{name:"Semiotics",c:[.78,.75,.72,.8,.76,.74],desc:"30 sign systems"},consciousness_coherence:{name:"Consciousness",c:[.55,.6,.5,.65,.58,.52],desc:"20 systems"},continuity_theory:{name:"Continuity",c:[.85,.82,.88,.8,.84,.86],desc:"Persistence"},awareness_cognition:{name:"Awareness",c:[.7,.65,.72,.68,.75,.66],desc:"10 theorems"},standard_model:{name:"Standard Model",c:[.5,.45,.4,.55,.48,.42,.52,.46],desc:"31 particles"},clinical_neuroscience:{name:"Clinical Neuro",c:[.75,.7,.68,.78,.72,.66],desc:"Cortical kernel"},spacetime_memory:{name:"Spacetime",c:[.62,.58,.55,.65,.6,.56],desc:"40 entities"}},o=[];for(const[e,t]of Object.entries(h)){const n=Array.from({length:t.c.length},()=>1/t.c.length),i=v(t.c,n),a=$(i);o.push({id:e,name:t.name,desc:t.desc,c:t.c,kernel:i,regime:a.regime,isCritical:a.isCritical})}o.sort((e,t)=>t.kernel.F-e.kernel.F);let x=null;function y(){const e=document.getElementById("domain-presets");e.innerHTML=o.map(t=>`
        <button class="domain-btn px-2 py-1 text-xs rounded border ${{STABLE:"border-green-600 text-green-400",WATCH:"border-amber-600 text-amber-400",COLLAPSE:"border-red-600 text-red-400"}[t.regime]||""} bg-kernel-800 hover:bg-kernel-700 transition"
          data-domain="${t.id}" title="${t.desc}">
          ${t.name}
        </button>
      `).join(""),e.querySelectorAll(".domain-btn").forEach(t=>{t.addEventListener("click",()=>{const n=t.dataset.domain;p(n)})})}function p(e){const t=o.find(a=>a.id===e);if(!t)return;const n={STABLE:"text-green-400",WATCH:"text-amber-400",COLLAPSE:"text-red-400"},i=document.getElementById("domain-invariants");i.innerHTML=[{label:"F",value:t.kernel.F.toFixed(6),color:"text-kernel-200"},{label:"ω",value:t.kernel.omega.toFixed(6),color:"text-kernel-200"},{label:"IC",value:t.kernel.IC.toFixed(6),color:"text-kernel-200"},{label:"Δ",value:t.kernel.delta.toFixed(6),color:"text-kernel-200"},{label:"S",value:t.kernel.S.toFixed(6),color:"text-kernel-200"},{label:"Regime",value:t.regime+(t.isCritical?" ⚠":""),color:n[t.regime]||""}].map(a=>`
      <div class="bg-kernel-800 rounded p-2 text-center">
        <div class="text-xs text-kernel-500">${a.label}</div>
        <div class="font-mono text-sm ${a.color}">${a.value}</div>
      </div>
    `).join("")}function C(e,t){const n=Math.abs(e.kernel.F+e.kernel.omega-1),i=e.kernel.IC<=e.kernel.F+1e-10,a=Math.abs(e.kernel.IC-Math.exp(e.kernel.kappa)),s=e.kernel.IC/e.kernel.F,l=e.c.map((d,b)=>{const u=(d*100).toFixed(0),k=d>.8?"bg-green-600":d>.5?"bg-amber-600":"bg-red-600";return`
        <div class="flex items-center gap-2 text-[10px]">
          <span class="text-kernel-500 w-10 text-right">c<sub>${b+1}</sub></span>
          <div class="flex-1 bg-kernel-800 rounded h-3 overflow-hidden">
            <div class="${k} h-full" style="width:${u}%"></div>
          </div>
          <span class="text-kernel-400 w-10 font-mono">${d.toFixed(3)}</span>
        </div>
      `}).join(""),r=Math.pow(e.kernel.omega,3)/(1-e.kernel.omega+f),c=1*e.kernel.C,m=r+c;return`
      <tr class="detail-row" data-detail-for="${e.id}">
        <td colspan="${t}" class="p-0">
          <div class="bg-kernel-950 border-t border-b border-kernel-700 p-4 space-y-4">
            <div class="flex items-center justify-between">
              <h4 class="text-sm font-bold text-kernel-200">${e.name} — ${e.desc}</h4>
              <span class="text-xs text-kernel-500">${e.c.length} channels, equal weights</span>
            </div>
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
              <!-- Channel Values -->
              <div>
                <div class="text-xs text-kernel-500 font-semibold mb-2">Channel Values</div>
                <div class="space-y-1">${l}</div>
                <div class="mt-2 text-[10px] text-kernel-600">
                  min = ${Math.min(...e.c).toFixed(3)} · max = ${Math.max(...e.c).toFixed(3)} · spread = ${(Math.max(...e.c)-Math.min(...e.c)).toFixed(3)}
                </div>
              </div>
              <!-- Identity Checks -->
              <div>
                <div class="text-xs text-kernel-500 font-semibold mb-2">Identity Verification</div>
                <div class="space-y-2">
                  <div class="flex items-center gap-2 text-xs">
                    <span class="${n<1e-10?"text-emerald-400":"text-red-400"}">${n<1e-10?"✓":"✕"}</span>
                    <span class="text-kernel-300">F + ω = 1</span>
                    <span class="text-kernel-500 font-mono ml-auto">|res| = ${n.toExponential(1)}</span>
                  </div>
                  <div class="flex items-center gap-2 text-xs">
                    <span class="${i?"text-emerald-400":"text-red-400"}">${i?"✓":"✕"}</span>
                    <span class="text-kernel-300">IC ≤ F</span>
                    <span class="text-kernel-500 font-mono ml-auto">Δ = ${e.kernel.delta.toFixed(6)}</span>
                  </div>
                  <div class="flex items-center gap-2 text-xs">
                    <span class="${a<1e-10?"text-emerald-400":"text-red-400"}">${a<1e-10?"✓":"✕"}</span>
                    <span class="text-kernel-300">IC = exp(κ)</span>
                    <span class="text-kernel-500 font-mono ml-auto">|res| = ${a.toExponential(1)}</span>
                  </div>
                </div>
                <div class="mt-3 text-[10px] text-kernel-600 bg-kernel-800/50 rounded p-2">
                  IC/F = ${s.toFixed(4)} — ${s>.95?"nearly homogeneous":s>.7?"moderate heterogeneity":s>.4?"significant heterogeneity":"severe heterogeneity (geometric slaughter)"}
                </div>
              </div>
              <!-- Budget Snapshot -->
              <div>
                <div class="text-xs text-kernel-500 font-semibold mb-2">Budget Snapshot</div>
                <div class="space-y-1 text-xs">
                  <div class="flex justify-between text-kernel-400">
                    <span class="text-red-400">D<sub>ω</sub> = Γ(ω)</span>
                    <span class="font-mono">${r.toFixed(6)}</span>
                  </div>
                  <div class="flex justify-between text-kernel-400">
                    <span class="text-red-400">D<sub>C</sub> = α·C</span>
                    <span class="font-mono">${c.toFixed(6)}</span>
                  </div>
                  <div class="flex justify-between text-kernel-400 border-t border-kernel-700 pt-1">
                    <span class="text-red-400 font-semibold">Total Debit</span>
                    <span class="font-mono">${m.toFixed(6)}</span>
                  </div>
                </div>
                <div class="mt-2 text-[10px] text-kernel-600 bg-kernel-800/50 rounded p-2">
                  For the seam to close, credit R·τ<sub>R</sub> must bring |Δκ| within tol = 0.005
                </div>
              </div>
            </div>
          </div>
        </td>
      </tr>
    `}function g(){const e=document.getElementById("domain-tbody"),t=10;e.innerHTML=o.map(n=>{const a={STABLE:{bg:"bg-green-900/20",text:"text-green-400"},WATCH:{bg:"bg-amber-900/20",text:"text-amber-400"},COLLAPSE:{bg:"bg-red-900/20",text:"text-red-400"}}[n.regime]||{bg:"",text:""},s=n.kernel.IC/n.kernel.F,l=x===n.id;return`
        <tr class="domain-row border-b border-kernel-800 hover:bg-kernel-800/30 transition cursor-pointer ${a.bg}"
            data-domain="${n.id}">
          <td class="p-1.5 text-kernel-500 domain-chevron" style="width:20px">${l?"▾":"▸"}</td>
          <td class="p-1.5 text-kernel-300">${n.name}</td>
          <td class="text-right p-1.5">${n.kernel.F.toFixed(4)}</td>
          <td class="text-right p-1.5">${n.kernel.omega.toFixed(4)}</td>
          <td class="text-right p-1.5">${n.kernel.IC.toFixed(4)}</td>
          <td class="text-right p-1.5">${n.kernel.delta.toFixed(4)}</td>
          <td class="text-right p-1.5">${n.kernel.S.toFixed(4)}</td>
          <td class="text-right p-1.5">${n.kernel.C.toFixed(4)}</td>
          <td class="text-center p-1.5 ${a.text}">${n.regime}${n.isCritical?" ⚠":""}</td>
          <td class="text-center p-1.5 ${s>.9?"text-green-400":s>.5?"text-amber-400":"text-red-400"}">${s.toFixed(3)}</td>
        </tr>
        ${l?C(n,t):""}
      `}).join(""),e.querySelectorAll(".domain-row").forEach(n=>{n.addEventListener("click",()=>{const i=n.dataset.domain;x===i?x=null:x=i,g()})})}function F(){const e={STABLE:0,WATCH:0,COLLAPSE:0};o.forEach(i=>e[i.regime]++);const t=o.length,n=document.getElementById("domain-regime-bars");n.innerHTML=`
      <div class="flex gap-1 h-8 rounded overflow-hidden mb-2">
        ${e.STABLE>0?`<div class="bg-green-600 flex items-center justify-center text-xs text-white font-medium" style="width:${e.STABLE/t*100}%">${e.STABLE}</div>`:""}
        ${e.WATCH>0?`<div class="bg-amber-600 flex items-center justify-center text-xs text-white font-medium" style="width:${e.WATCH/t*100}%">${e.WATCH}</div>`:""}
        ${e.COLLAPSE>0?`<div class="bg-red-700 flex items-center justify-center text-xs text-white font-medium" style="width:${e.COLLAPSE/t*100}%">${e.COLLAPSE}</div>`:""}
      </div>
      <div class="flex justify-between text-xs text-kernel-500">
        <span class="text-green-400">Stable: ${e.STABLE} (${(e.STABLE/t*100).toFixed(0)}%)</span>
        <span class="text-amber-400">Watch: ${e.WATCH} (${(e.WATCH/t*100).toFixed(0)}%)</span>
        <span class="text-red-400">Collapse: ${e.COLLAPSE} (${(e.COLLAPSE/t*100).toFixed(0)}%)</span>
      </div>
    `}function L(){const e=document.getElementById("domain-rankings"),t=[...o].sort((s,l)=>l.kernel.F-s.kernel.F),n=[...o].sort((s,l)=>s.kernel.IC/s.kernel.F-l.kernel.IC/l.kernel.F),i=[...o].sort((s,l)=>l.kernel.delta-s.kernel.delta);function a(s,l,r,c){return`
        <div class="bg-kernel-800/50 rounded p-3">
          <div class="text-xs font-bold ${c} mb-2">${s}</div>
          <div class="space-y-1">
            ${l.slice(0,5).map((m,d)=>`
              <div class="flex items-center gap-2 text-[10px]">
                <span class="text-kernel-600 w-3">${d+1}.</span>
                <span class="text-kernel-300">${m.name}</span>
                <span class="text-kernel-500 font-mono ml-auto">${r(m)}</span>
              </div>
            `).join("")}
          </div>
        </div>
      `}e.innerHTML=`
      ${a("Highest Fidelity",t,s=>`F=${s.kernel.F.toFixed(4)}`,"text-green-400")}
      ${a("Largest Gap (Δ)",i,s=>`Δ=${s.kernel.delta.toFixed(4)}`,"text-amber-400")}
      ${a("Most Heterogeneous (IC/F)",n,s=>`IC/F=${(s.kernel.IC/s.kernel.F).toFixed(4)}`,"text-red-400")}
    `}y();g();F();L();o.length>0&&p(o[0].id);

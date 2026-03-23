import{c as o,a}from"./kernel.CbHwp2KC.js";import"./constants.C7mlF-md.js";const l={gcd:{name:"GCD",c:[.95,.9,.88,.92,.85,.91],desc:"Core framework"},rcft:{name:"RCFT",c:[.92,.88,.85,.9,.87,.86],desc:"Recursive collapse"},kinematics:{name:"Kinematics",c:[.8,.85,.75,.9,.82,.78],desc:"Motion analysis"},weyl:{name:"Weyl",c:[.7,.65,.6,.72,.68,.66],desc:"Cosmology"},security:{name:"Security",c:[.95,.92,.98,.9,.96,.93],desc:"Audit validation"},astronomy:{name:"Astronomy",c:[.75,.8,.7,.85,.72,.78],desc:"Stellar classification"},nuclear_physics:{name:"Nuclear",c:[.6,.55,.7,.5,.65,.58],desc:"Binding energy"},quantum_mechanics:{name:"Quantum",c:[.85,.8,.75,.9,.82,.78],desc:"Wavefunction"},finance:{name:"Finance",c:[.72,.68,.65,.7,.75,.66],desc:"Portfolio coherence"},atomic_physics:{name:"Atomic",c:[.88,.85,.82,.9,.86,.84],desc:"118 elements"},materials_science:{name:"Materials",c:[.82,.78,.8,.85,.76,.81],desc:"Element database"},everyday_physics:{name:"Everyday",c:[.9,.88,.85,.92,.87,.89],desc:"Thermo, optics"},evolution:{name:"Evolution",c:[.65,.7,.6,.75,.68,.62],desc:"40 organisms"},dynamic_semiotics:{name:"Semiotics",c:[.78,.75,.72,.8,.76,.74],desc:"30 sign systems"},consciousness_coherence:{name:"Consciousness",c:[.55,.6,.5,.65,.58,.52],desc:"20 systems"},continuity_theory:{name:"Continuity",c:[.85,.82,.88,.8,.84,.86],desc:"Persistence"},awareness_cognition:{name:"Awareness",c:[.7,.65,.72,.68,.75,.66],desc:"10 theorems"},standard_model:{name:"Standard Model",c:[.5,.45,.4,.55,.48,.42,.52,.46],desc:"31 particles"},clinical_neuroscience:{name:"Clinical Neuro",c:[.75,.7,.68,.78,.72,.66],desc:"Cortical kernel"},spacetime_memory:{name:"Spacetime",c:[.62,.58,.55,.65,.6,.56],desc:"40 entities"}},r=[];for(const[t,e]of Object.entries(l)){const s=Array.from({length:e.c.length},()=>1/e.c.length),i=o(e.c,s),n=a(i);r.push({id:t,name:e.name,desc:e.desc,kernel:i,regime:n.regime,isCritical:n.isCritical})}r.sort((t,e)=>e.kernel.F-t.kernel.F);function d(){const t=document.getElementById("domain-presets");t.innerHTML=r.map(e=>`
        <button class="domain-btn px-2 py-1 text-xs rounded border ${{STABLE:"border-green-600 text-green-400",WATCH:"border-amber-600 text-amber-400",COLLAPSE:"border-red-600 text-red-400"}[e.regime]||""} bg-kernel-800 hover:bg-kernel-700 transition"
          data-domain="${e.id}" title="${e.desc}">
          ${e.name}
        </button>
      `).join(""),t.querySelectorAll(".domain-btn").forEach(e=>{e.addEventListener("click",()=>{const s=e.dataset.domain;c(s)})})}function c(t){const e=r.find(n=>n.id===t);if(!e)return;const s={STABLE:"text-green-400",WATCH:"text-amber-400",COLLAPSE:"text-red-400"},i=document.getElementById("domain-invariants");i.innerHTML=[{label:"F",value:e.kernel.F.toFixed(6),color:"text-kernel-200"},{label:"ω",value:e.kernel.omega.toFixed(6),color:"text-kernel-200"},{label:"IC",value:e.kernel.IC.toFixed(6),color:"text-kernel-200"},{label:"Δ",value:e.kernel.delta.toFixed(6),color:"text-kernel-200"},{label:"S",value:e.kernel.S.toFixed(6),color:"text-kernel-200"},{label:"Regime",value:e.regime+(e.isCritical?" ⚠":""),color:s[e.regime]||""}].map(n=>`
      <div class="bg-kernel-800 rounded p-2 text-center">
        <div class="text-xs text-kernel-500">${n.label}</div>
        <div class="font-mono text-sm ${n.color}">${n.value}</div>
      </div>
    `).join("")}function m(){const t=document.getElementById("domain-tbody");t.innerHTML=r.map(e=>{const i={STABLE:{bg:"bg-green-900/20",text:"text-green-400"},WATCH:{bg:"bg-amber-900/20",text:"text-amber-400"},COLLAPSE:{bg:"bg-red-900/20",text:"text-red-400"}}[e.regime]||{bg:"",text:""},n=e.kernel.IC/e.kernel.F;return`
        <tr class="border-b border-kernel-800 hover:bg-kernel-800/30 transition ${i.bg}">
          <td class="p-1.5 text-kernel-300">${e.name}</td>
          <td class="text-right p-1.5">${e.kernel.F.toFixed(4)}</td>
          <td class="text-right p-1.5">${e.kernel.omega.toFixed(4)}</td>
          <td class="text-right p-1.5">${e.kernel.IC.toFixed(4)}</td>
          <td class="text-right p-1.5">${e.kernel.delta.toFixed(4)}</td>
          <td class="text-right p-1.5">${e.kernel.S.toFixed(4)}</td>
          <td class="text-right p-1.5">${e.kernel.C.toFixed(4)}</td>
          <td class="text-center p-1.5 ${i.text}">${e.regime}${e.isCritical?" ⚠":""}</td>
          <td class="text-center p-1.5 ${n>.9?"text-green-400":n>.5?"text-amber-400":"text-red-400"}">${n.toFixed(3)}</td>
        </tr>
      `}).join("")}function x(){const t={STABLE:0,WATCH:0,COLLAPSE:0};r.forEach(i=>t[i.regime]++);const e=r.length,s=document.getElementById("domain-regime-bars");s.innerHTML=`
      <div class="flex gap-1 h-8 rounded overflow-hidden mb-2">
        ${t.STABLE>0?`<div class="bg-green-600 flex items-center justify-center text-xs text-white font-medium" style="width:${t.STABLE/e*100}%">${t.STABLE}</div>`:""}
        ${t.WATCH>0?`<div class="bg-amber-600 flex items-center justify-center text-xs text-white font-medium" style="width:${t.WATCH/e*100}%">${t.WATCH}</div>`:""}
        ${t.COLLAPSE>0?`<div class="bg-red-700 flex items-center justify-center text-xs text-white font-medium" style="width:${t.COLLAPSE/e*100}%">${t.COLLAPSE}</div>`:""}
      </div>
      <div class="flex justify-between text-xs text-kernel-500">
        <span class="text-green-400">Stable: ${t.STABLE} (${(t.STABLE/e*100).toFixed(0)}%)</span>
        <span class="text-amber-400">Watch: ${t.WATCH} (${(t.WATCH/e*100).toFixed(0)}%)</span>
        <span class="text-red-400">Collapse: ${t.COLLAPSE} (${(t.COLLAPSE/e*100).toFixed(0)}%)</span>
      </div>
    `}d();m();x();r.length>0&&c(r[0].id);

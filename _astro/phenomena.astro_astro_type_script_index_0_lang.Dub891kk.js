import{c as g}from"./constants.DtcEBpaC.js";import{c as p,a as u}from"./kernel.1iDFzIM1.js";const i=g,x=[...new Set(i.map(t=>t.category))],h={"Channel Collapse":{color:"red",border:"border-red-700/50",badge:"bg-red-900/30 text-red-400",icon:"💀",description:"What happens when one or more channels approach ε (the guard band). Geometric slaughter, integrity cliffs, and the residue that becomes matter."},Seam:{color:"amber",border:"border-amber-700/50",badge:"bg-amber-900/30 text-amber-400",icon:"🔗",description:"The boundary zone between collapse and return. Where seams close (or fail to close), where the first weld occurs, and the desert where no return is possible."},"Structural States":{color:"blue",border:"border-blue-700/50",badge:"bg-blue-900/30 text-blue-400",icon:"🏗️",description:"Named phases of the collapse landscape — from the prison of drift (trapped but alive) to permanent detention (∞_rec) to liberating surplus. Every state is a measurement, not a judgment."},"Scale Ladder":{color:"green",border:"border-green-700/50",badge:"bg-green-900/30 text-green-400",icon:"🪜",description:"How coherence propagates across scales. Fidelity descends, integrity slaughters, then new degrees of freedom restore what was lost. The ladder oscillates between destruction and recovery."}},v=document.getElementById("phenomena-categories");v.innerHTML=x.map(t=>{const o=i.filter(e=>e.category===t),r=h[t];return`
      <div class="bg-kernel-900 border ${r.border} rounded-lg p-6">
        <div class="flex items-center gap-3 mb-4">
          <span class="text-2xl">${r.icon}</span>
          <div>
            <h2 class="text-lg font-bold text-${r.color}-300">${t}</h2>
            <p class="text-xs text-kernel-500 mt-1">${r.description}</p>
          </div>
        </div>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
          ${o.map(e=>`
            <div class="bg-kernel-800/80 rounded-lg p-4 border border-kernel-700 hover:border-${r.color}-700/50 transition">
              <div class="flex items-start justify-between mb-2">
                <div>
                  <span class="text-kernel-100 font-bold italic">${e.latin}</span>
                  <span class="text-kernel-500 text-xs ml-2">${e.english}</span>
                </div>
              </div>
              <p class="text-kernel-400 text-sm">${e.operational}</p>
            </div>
          `).join("")}
        </div>
      </div>`}).join("");const c=document.getElementById("dead-slider"),f=document.getElementById("dead-val"),y=document.getElementById("slaughter-results"),d=document.getElementById("icf-bar"),C=document.getElementById("icf-label");function m(){const t=parseFloat(c.value);f.textContent=t.toFixed(3);const o=[.95,.95,.95,.95,.95,.95,.95,t],r=o.map(()=>1/o.length),e=p(o,r),n=u(e),l=e.F>0?e.IC/e.F:0,b={STABLE:"text-green-400",WATCH:"text-amber-400",COLLAPSE:"text-red-400"};y.innerHTML=[{label:"F (Fidelity)",value:e.F.toFixed(6),color:"text-green-400"},{label:"IC (Integrity)",value:e.IC.toFixed(6),color:e.IC<.3?"text-red-400":"text-purple-400"},{label:"Δ (Gap)",value:e.delta.toFixed(6),color:e.delta>.3?"text-red-400":"text-kernel-300"},{label:"Regime",value:n.regime+(n.isCritical?" + CRITICAL":""),color:b[n.regime]}].map(s=>`
      <div class="bg-kernel-800 rounded p-2 text-center">
        <div class="text-xs text-kernel-500">${s.label}</div>
        <div class="font-mono text-sm ${s.color}">${s.value}</div>
      </div>
    `).join("");const a=Math.max(0,Math.min(100,l*100));d.style.width=`${a}%`,d.className=`h-full transition-all duration-200 rounded ${a>70?"bg-green-500":a>40?"bg-amber-500":"bg-red-500"}`,C.textContent=l.toFixed(4)}c.addEventListener("input",m);m();

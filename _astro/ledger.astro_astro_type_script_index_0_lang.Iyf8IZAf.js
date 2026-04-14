import{c as b,a as k}from"./kernel.1iDFzIM1.js";import"./constants.DtcEBpaC.js";const F=1e-8,g=.005,C={gcd:{name:"GCD",c:[.95,.9,.88,.92,.85,.91],desc:"Core framework"},rcft:{name:"RCFT",c:[.92,.88,.85,.9,.87,.86],desc:"Recursive collapse"},kinematics:{name:"Kinematics",c:[.8,.85,.75,.9,.82,.78],desc:"Motion analysis"},weyl:{name:"Weyl",c:[.7,.65,.6,.72,.68,.66],desc:"Cosmology"},security:{name:"Security",c:[.95,.92,.98,.9,.96,.93],desc:"Audit validation"},astronomy:{name:"Astronomy",c:[.75,.8,.7,.85,.72,.78],desc:"Stellar classification"},nuclear_physics:{name:"Nuclear",c:[.6,.55,.7,.5,.65,.58],desc:"Binding energy"},quantum_mechanics:{name:"Quantum",c:[.85,.8,.75,.9,.82,.78],desc:"Wavefunction"},finance:{name:"Finance",c:[.72,.68,.65,.7,.75,.66],desc:"Portfolio coherence"},atomic_physics:{name:"Atomic",c:[.88,.85,.82,.9,.86,.84],desc:"118 elements"},materials_science:{name:"Materials",c:[.82,.78,.8,.85,.76,.81],desc:"Element database"},everyday_physics:{name:"Everyday",c:[.9,.88,.85,.92,.87,.89],desc:"Thermo, optics"},evolution:{name:"Evolution",c:[.65,.7,.6,.75,.68,.62],desc:"40 organisms"},dynamic_semiotics:{name:"Semiotics",c:[.78,.75,.72,.8,.76,.74],desc:"30 sign systems"},consciousness_coherence:{name:"Consciousness",c:[.55,.6,.5,.65,.58,.52],desc:"20 systems"},continuity_theory:{name:"Continuity",c:[.85,.82,.88,.8,.84,.86],desc:"Persistence"},awareness_cognition:{name:"Awareness",c:[.7,.65,.72,.68,.75,.66],desc:"10 theorems"},standard_model:{name:"Standard Model",c:[.5,.45,.4,.55,.48,.42,.52,.46],desc:"31 particles"},clinical_neuroscience:{name:"Clinical Neuro",c:[.75,.7,.68,.78,.72,.66],desc:"Cortical kernel"},spacetime_memory:{name:"Spacetime",c:[.62,.58,.55,.65,.6,.56],desc:"40 entities"},immunology:{name:"Immunology",c:[.8,.75,.72,.82,.78,.7],desc:"Adaptive immunity"}},a=[];for(const[d,n]of Object.entries(C)){const m=Array.from({length:n.c.length},()=>1/n.c.length),e=b(n.c,m),s=k(e);a.push({id:d,name:n.name,c:n.c,kernel:e,regime:s.regime,isCritical:s.isCritical})}function h(d){return Math.pow(d,3)/(1-d+F)}function u(){const d=parseFloat(document.getElementById("sim-omega").value),n=parseFloat(document.getElementById("sim-curv").value),m=parseFloat(document.getElementById("sim-r").value),e=document.getElementById("sim-inf").checked,s=e?1/0:parseFloat(document.getElementById("sim-tau").value),l=h(d),i=1*n,c=e?0:m*s,x=l+i,t=c-x,o=Math.abs(t)<=g;document.getElementById("sim-omega-val").textContent=d.toFixed(2),document.getElementById("sim-curv-val").textContent=n.toFixed(2),document.getElementById("sim-r-val").textContent=m.toFixed(2),document.getElementById("sim-tau-val").textContent=e?"∞_rec":s.toFixed(1),document.getElementById("sim-dw-val").innerHTML=`D<sub>ω</sub> = ${l.toFixed(4)}`,document.getElementById("sim-dc-val").innerHTML=`D<sub>C</sub> = ${i.toFixed(4)}`;const v=d>=.3?"COLLAPSE":d>=.038?"WATCH":"STABLE",r={STABLE:"text-green-400",WATCH:"text-amber-400",COLLAPSE:"text-red-400"},p=document.getElementById("sim-result");p.innerHTML=`
      <div class="grid grid-cols-1 sm:grid-cols-3 gap-3 text-center">
        <div class="bg-kernel-800 rounded p-3">
          <div class="text-xs text-kernel-500">Debit Total</div>
          <div class="text-red-400 font-mono text-lg">${x.toFixed(4)}</div>
          <div class="text-xs text-kernel-600">D<sub>ω</sub>=${l.toFixed(4)} + D<sub>C</sub>=${i.toFixed(4)}</div>
        </div>
        <div class="bg-kernel-800 rounded p-3">
          <div class="text-xs text-kernel-500">Credit</div>
          <div class="${e?"text-red-400":"text-emerald-400"} font-mono text-lg">${e?"0 (∞_rec)":c.toFixed(4)}</div>
          <div class="text-xs text-kernel-600">R·τ<sub>R</sub> = ${m.toFixed(2)} × ${e?"∞":s.toFixed(1)}</div>
        </div>
        <div class="bg-kernel-800 rounded p-3">
          <div class="text-xs text-kernel-500">Net Δκ</div>
          <div class="${o?"text-emerald-400":"text-red-400"} font-mono text-lg">${t>=0?"+":""}${t.toFixed(4)}</div>
          <div class="text-xs text-kernel-600">tol = ${g}</div>
        </div>
      </div>
      <!-- Visual budget bar -->
      <div class="mt-3">
        <div class="flex h-6 rounded overflow-hidden bg-kernel-800">
          <div class="bg-red-600/80 flex items-center justify-center text-[10px] text-white" style="width:${Math.min(x/Math.max(x,c,.01)*50,50)}%">Debit</div>
          <div class="bg-emerald-600/80 flex items-center justify-center text-[10px] text-white" style="width:${Math.min(c/Math.max(x,c,.01)*50,50)}%">${e?"—":"Credit"}</div>
        </div>
      </div>
      <div class="mt-3 p-3 rounded ${o?"bg-emerald-900/20 border border-emerald-700/30":"bg-red-900/20 border border-red-700/30"}">
        <div class="flex items-center gap-3">
          <span class="${o?"text-emerald-400":"text-red-400"} text-lg font-bold">${o?"SUTURA":"GESTUS"}</span>
          <span class="text-xs text-kernel-400">${o?"The seam closes — epistemic credit granted":e?"τ_R = ∞_rec — permanent detention, no return credit":"|Δκ| > tol_seam — residual does not reconcile"}</span>
        </div>
        <div class="text-xs text-kernel-500 mt-1">
          Regime: <span class="${r[v]}">${v}</span> · |Δκ| = ${Math.abs(t).toFixed(4)} ${o?"≤":">"} ${g}
        </div>
      </div>
      <div class="text-xs text-kernel-600 italic mt-2">
        ${o?"Cum receptu, sutura est. — With a receipt, it is a weld.":"Sine receptu, gestus est. — Without a receipt, it is a gesture."}
      </div>
    `}["sim-omega","sim-curv","sim-r","sim-tau"].forEach(d=>{document.getElementById(d)?.addEventListener("input",u)});document.getElementById("sim-inf")?.addEventListener("change",u);u();function $(){const n=document.getElementById("id-trace").value.split(",").map(r=>parseFloat(r.trim())).filter(r=>!isNaN(r));if(n.length<2){document.getElementById("id-results").innerHTML='<p class="text-xs text-red-400">Enter at least 2 values.</p>';return}const m=Array.from({length:n.length},()=>1/n.length),e=b(n,m),s=k(e),l=Math.abs(e.F+e.omega-1),i=l<1e-10,c=e.IC<=e.F+1e-10,x=Math.abs(e.IC-Math.exp(e.kappa)),t=x<1e-10,o=i&&c&&t,v=document.getElementById("id-results");v.innerHTML=`
      <div class="grid grid-cols-1 md:grid-cols-3 gap-3 mb-4">
        <div class="bg-kernel-800 rounded p-3 border ${i?"border-emerald-700/40":"border-red-700/40"}">
          <div class="flex items-center gap-2 mb-1">
            <span class="${i?"text-emerald-400":"text-red-400"}">${i?"✓":"✕"}</span>
            <span class="text-sm font-bold text-kernel-200">F + ω = 1</span>
          </div>
          <div class="text-xs text-kernel-400 font-mono">
            F = ${e.F.toFixed(10)}<br>
            ω = ${e.omega.toFixed(10)}<br>
            |residual| = ${l.toExponential(2)}
          </div>
        </div>
        <div class="bg-kernel-800 rounded p-3 border ${c?"border-emerald-700/40":"border-red-700/40"}">
          <div class="flex items-center gap-2 mb-1">
            <span class="${c?"text-emerald-400":"text-red-400"}">${c?"✓":"✕"}</span>
            <span class="text-sm font-bold text-kernel-200">IC ≤ F</span>
          </div>
          <div class="text-xs text-kernel-400 font-mono">
            IC = ${e.IC.toFixed(10)}<br>
            F = ${e.F.toFixed(10)}<br>
            Δ = F − IC = ${e.delta.toFixed(10)}
          </div>
        </div>
        <div class="bg-kernel-800 rounded p-3 border ${t?"border-emerald-700/40":"border-red-700/40"}">
          <div class="flex items-center gap-2 mb-1">
            <span class="${t?"text-emerald-400":"text-red-400"}">${t?"✓":"✕"}</span>
            <span class="text-sm font-bold text-kernel-200">IC = exp(κ)</span>
          </div>
          <div class="text-xs text-kernel-400 font-mono">
            IC = ${e.IC.toFixed(10)}<br>
            exp(κ) = ${Math.exp(e.kappa).toFixed(10)}<br>
            |residual| = ${x.toExponential(2)}
          </div>
        </div>
      </div>
      <div class="grid grid-cols-3 sm:grid-cols-6 gap-2 mb-3">
        ${[{label:"F",val:e.F},{label:"ω",val:e.omega},{label:"IC",val:e.IC},{label:"Δ",val:e.delta},{label:"S",val:e.S},{label:"C",val:e.C}].map(r=>`
          <div class="bg-kernel-950 rounded p-2 text-center">
            <div class="text-[10px] text-kernel-500">${r.label}</div>
            <div class="font-mono text-xs text-kernel-200">${r.val.toFixed(4)}</div>
          </div>
        `).join("")}
      </div>
      <div class="flex items-center gap-3 p-3 rounded ${o?"bg-emerald-900/20 border border-emerald-700/30":"bg-red-900/20 border border-red-700/30"}">
        <span class="${o?"text-emerald-400":"text-red-400"} font-bold">${o?"ALL IDENTITIES HOLD":"IDENTITY VIOLATION"}</span>
        <span class="text-xs text-kernel-400">Regime: <span class="${s.regime==="STABLE"?"text-green-400":s.regime==="WATCH"?"text-amber-400":"text-red-400"}">${s.regime}</span>${s.isCritical?' <span class="text-red-400">⚠ CRITICAL</span>':""}</span>
        <span class="text-xs text-kernel-500">IC/F = ${(e.IC/e.F).toFixed(4)}</span>
      </div>
    `}document.getElementById("id-trace")?.addEventListener("input",$);$();function f(){const d=document.getElementById("cmp-a"),n=document.getElementById("cmp-b");a.sort((e,s)=>s.kernel.F-e.kernel.F),a.forEach((e,s)=>{const l=document.createElement("option");l.value=e.id,l.textContent=`${e.name} (F=${e.kernel.F.toFixed(3)})`,d.appendChild(l);const i=document.createElement("option");i.value=e.id,i.textContent=`${e.name} (F=${e.kernel.F.toFixed(3)})`,n.appendChild(i)}),a.length>=2&&(d.value=a[0].id,n.value=a[a.length-1].id);function m(){const e=a.find(t=>t.id===d.value),s=a.find(t=>t.id===n.value),l=["F","omega","IC","delta","S","C"],i={F:"F",omega:"ω",IC:"IC",delta:"Δ",S:"S",C:"C"},c={STABLE:"text-green-400",WATCH:"text-amber-400",COLLAPSE:"text-red-400"},x=document.getElementById("cmp-result");x.innerHTML=`
        <div class="overflow-x-auto">
          <table class="w-full text-xs font-mono">
            <thead>
              <tr class="text-kernel-500 border-b border-kernel-700">
                <th class="text-left p-2">Metric</th>
                <th class="text-right p-2 text-blue-400">${e.name}</th>
                <th class="text-right p-2 text-amber-400">${s.name}</th>
                <th class="text-right p-2">Δ (A − B)</th>
                <th class="text-center p-2">Comparison</th>
              </tr>
            </thead>
            <tbody class="divide-y divide-kernel-800">
              ${l.map(t=>{const o=e.kernel[t],v=s.kernel[t],r=o-v,p=t==="omega"||t==="C"||t==="S"?r<0?"A":r>0?"B":"=":r>0?"A":r<0?"B":"=";return`
                  <tr>
                    <td class="p-2 text-kernel-300">${i[t]}</td>
                    <td class="text-right p-2">${o.toFixed(6)}</td>
                    <td class="text-right p-2">${v.toFixed(6)}</td>
                    <td class="text-right p-2 ${r>0?"text-emerald-400":r<0?"text-red-400":""}">${r>=0?"+":""}${r.toFixed(6)}</td>
                    <td class="text-center p-2">${p==="A"?'<span class="text-blue-400">◀</span>':p==="B"?'<span class="text-amber-400">▶</span>':'<span class="text-kernel-500">=</span>'}</td>
                  </tr>
                `}).join("")}
              <tr class="border-t-2 border-kernel-600">
                <td class="p-2 text-kernel-300">Regime</td>
                <td class="text-right p-2 ${c[e.regime]}">${e.regime}${e.isCritical?" ⚠":""}</td>
                <td class="text-right p-2 ${c[s.regime]}">${s.regime}${s.isCritical?" ⚠":""}</td>
                <td colspan="2"></td>
              </tr>
              <tr>
                <td class="p-2 text-kernel-300">IC/F</td>
                <td class="text-right p-2">${(e.kernel.IC/e.kernel.F).toFixed(4)}</td>
                <td class="text-right p-2">${(s.kernel.IC/s.kernel.F).toFixed(4)}</td>
                <td class="text-right p-2">${e.kernel.IC/e.kernel.F-s.kernel.IC/s.kernel.F>=0?"+":""}${(e.kernel.IC/e.kernel.F-s.kernel.IC/s.kernel.F).toFixed(4)}</td>
                <td></td>
              </tr>
            </tbody>
          </table>
        </div>
        <!-- Composition -->
        <div class="mt-3 bg-kernel-800 rounded p-3">
          <div class="text-xs text-kernel-500 mb-2">Composition (if domains were composed):</div>
          <div class="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs text-center">
            <div>
              <div class="text-kernel-500">F<sub>12</sub> (arithmetic)</div>
              <div class="font-mono text-kernel-200">${((e.kernel.F+s.kernel.F)/2).toFixed(6)}</div>
            </div>
            <div>
              <div class="text-kernel-500">IC<sub>12</sub> (geometric)</div>
              <div class="font-mono text-kernel-200">${Math.sqrt(e.kernel.IC*s.kernel.IC).toFixed(6)}</div>
            </div>
            <div>
              <div class="text-kernel-500">Δ<sub>12</sub></div>
              <div class="font-mono text-kernel-200">${((e.kernel.F+s.kernel.F)/2-Math.sqrt(e.kernel.IC*s.kernel.IC)).toFixed(6)}</div>
            </div>
          </div>
        </div>
      `}d.addEventListener("change",m),n.addEventListener("change",m),m()}f();function I(){const d=a.map(t=>t.kernel.F),n=a.map(t=>t.kernel.IC),m=a.map(t=>t.kernel.delta),e={STABLE:0,WATCH:0,COLLAPSE:0};a.forEach(t=>e[t.regime]++);const s=a.filter(t=>t.isCritical).length,l=t=>t.reduce((o,v)=>o+v,0)/t.length,i=t=>Math.max(...t),c=t=>Math.min(...t),x=document.getElementById("agg-stats");x.innerHTML=`
      <div class="bg-kernel-900/50 border border-kernel-800/40 rounded-xl p-4 text-center">
        <div class="text-xs text-kernel-500 mb-1">Mean F</div>
        <div class="text-xl font-mono text-kernel-100">${l(d).toFixed(4)}</div>
        <div class="text-xs text-kernel-600">[${c(d).toFixed(3)} – ${i(d).toFixed(3)}]</div>
      </div>
      <div class="bg-kernel-900/50 border border-kernel-800/40 rounded-xl p-4 text-center">
        <div class="text-xs text-kernel-500 mb-1">Mean IC</div>
        <div class="text-xl font-mono text-kernel-100">${l(n).toFixed(4)}</div>
        <div class="text-xs text-kernel-600">[${c(n).toFixed(3)} – ${i(n).toFixed(3)}]</div>
      </div>
      <div class="bg-kernel-900/50 border border-kernel-800/40 rounded-xl p-4 text-center">
        <div class="text-xs text-kernel-500 mb-1">Mean Gap Δ</div>
        <div class="text-xl font-mono text-amber-400">${l(m).toFixed(4)}</div>
        <div class="text-xs text-kernel-600">Max: ${i(m).toFixed(4)}</div>
      </div>
      <div class="bg-kernel-900/50 border border-kernel-800/40 rounded-xl p-4 text-center">
        <div class="text-xs text-kernel-500 mb-1">Regime Distribution</div>
        <div class="text-xs mt-1 space-y-1">
          <div class="text-green-400">Stable: ${e.STABLE} / ${a.length}</div>
          <div class="text-amber-400">Watch: ${e.WATCH} / ${a.length}</div>
          <div class="text-red-400">Collapse: ${e.COLLAPSE} / ${a.length}</div>
          ${s>0?`<div class="text-red-400">⚠ Critical: ${s}</div>`:""}
        </div>
      </div>
    `}I();

import{c as d,a as f,g as C}from"./kernel.Cdj69J8o.js";import{E as x,b as I}from"./constants.6x37F0HI.js";function E(){const t=d([.97,.96,.98,.95,.97,.96,.98,.97]);m("stable-invariants",t,"emerald");const o=d([.85,.72,.91,.68,.77,.83,.9,.15]);m("watch-invariants",o,"amber");const r=d([.4,.35,.45,.3,.5,.25,.38,.42]);m("collapse-invariants",r,"red")}function m(a,t,e){const o=document.getElementById(a);if(!o)return;const s=[{label:"F",value:t.F.toFixed(4)},{label:"ω",value:t.omega.toFixed(4)},{label:"S",value:t.S.toFixed(4)},{label:"C",value:t.C.toFixed(4)},{label:"κ",value:t.kappa.toFixed(4)},{label:"IC",value:t.IC.toFixed(4)}];o.innerHTML=s.map(r=>`
      <div class="bg-kernel-950 rounded p-2 text-center">
        <div class="text-[10px] text-kernel-500">${r.label}</div>
        <div class="font-mono text-xs text-${e}-300">${r.value}</div>
      </div>
    `).join("")}function h(){const a=document.getElementById("slaughter-values");if(!a)return;const e=d([1,1,1,1,1,1,1,x]),o=[{label:"F",value:e.F.toFixed(4),color:"text-kernel-300"},{label:"IC",value:e.IC.toFixed(4),color:"text-red-400"},{label:"Δ",value:e.delta.toFixed(4),color:"text-amber-400"},{label:"IC/F",value:(e.IC/e.F).toFixed(4),color:"text-violet-400"}];a.innerHTML=o.map(s=>`
      <div class="bg-kernel-900 rounded-lg p-3">
        <div class="text-[10px] text-kernel-500">${s.label}</div>
        <div class="font-mono text-sm font-bold ${s.color}">${s.value}</div>
      </div>
    `).join("")}function u(){const a=document.getElementById("channel-input");if(!a)return;const t=a.value.split(",").map(l=>{const n=parseFloat(l.trim());return isNaN(n)?.5:Math.max(x,Math.min(1-x,n))}).filter(l=>!isNaN(l));if(t.length<2)return;const e=d(t),{regime:o,isCritical:s}=f(e),r=document.getElementById("explorer-banner");if(r){const l={STABLE:"bg-emerald-900/40 border border-emerald-700/50 text-emerald-400",WATCH:"bg-amber-900/40 border border-amber-700/50 text-amber-400",COLLAPSE:"bg-red-900/40 border border-red-700/50 text-red-400"},n=s?' <span class="text-violet-400 text-sm font-normal ml-2">+ CRITICAL</span>':"";r.className=`rounded-lg p-4 text-center ${l[o]||""}`,r.innerHTML=`<span class="text-2xl font-bold">${o}</span>${n}`}const v=document.getElementById("explorer-invariants");if(v){const l=[{label:"F (Fidelity)",value:e.F.toFixed(6)},{label:"ω (Drift)",value:e.omega.toFixed(6)},{label:"S (Entropy)",value:e.S.toFixed(6)},{label:"C (Curvature)",value:e.C.toFixed(6)},{label:"κ (Log-IC)",value:e.kappa.toFixed(6)},{label:"IC (Integrity)",value:e.IC.toFixed(6)}];v.innerHTML=l.map(n=>`
        <div class="bg-kernel-950 border border-kernel-800 rounded-lg p-3 text-center">
          <div class="text-[10px] text-kernel-500">${n.label}</div>
          <div class="font-mono text-sm text-kernel-200">${n.value}</div>
        </div>
      `).join("")}const b=document.getElementById("explorer-gates");if(b){const l=I,n=[{label:"ω < 0.038",pass:e.omega<l.omega_stable_max,value:e.omega.toFixed(4)},{label:"F > 0.90",pass:e.F>l.F_stable_min,value:e.F.toFixed(4)},{label:"S < 0.15",pass:e.S<l.S_stable_max,value:e.S.toFixed(4)},{label:"C < 0.14",pass:e.C<l.C_stable_max,value:e.C.toFixed(4)}];b.innerHTML=n.map(i=>{const c=i.pass?"bg-emerald-900/30 border-emerald-700/30":"bg-red-900/30 border-red-700/30",F=i.pass?'<span class="text-emerald-400 mr-1">✓</span>':'<span class="text-red-400 mr-1">✗</span>';return`
          <div class="rounded-lg p-2.5 border ${c}">
            <div class="text-[10px] text-kernel-500">${i.label}</div>
            <div class="flex items-center mt-1">${F}<span class="font-mono text-xs text-kernel-200">${i.value}</span></div>
          </div>
        `}).join("")}const p=document.getElementById("explorer-derived");if(p){const l=C(e.omega),n=e.F>0?e.IC/e.F:0,i=[{label:"Δ (Gap)",value:e.delta.toFixed(6)},{label:"IC/F Ratio",value:n.toFixed(6)},{label:"Γ(ω)",value:l<.001?l.toExponential(2):l.toFixed(4)},{label:"Channels",value:t.length.toString()}];p.innerHTML=i.map(c=>`
        <div class="bg-kernel-950 border border-kernel-800 rounded-lg p-3 text-center">
          <div class="text-[10px] text-kernel-500">${c.label}</div>
          <div class="font-mono text-sm text-kernel-200">${c.value}</div>
        </div>
      `).join("")}}const g=document.getElementById("channel-input");g&&g.addEventListener("input",u);document.querySelectorAll(".regime-preset").forEach(a=>{a.addEventListener("click",()=>{const t=document.getElementById("channel-input");t&&a instanceof HTMLElement&&(t.value=a.dataset.channels||"",u())})});E();h();u();

import{f as d,F as S,e as g,D as b}from"./constants.6x37F0HI.js";import{c as B,a as C,v as T}from"./kernel.Cdj69J8o.js";const l=new Set(d.map(n=>n.id)),F={epistemology:{active:"bg-purple-700/50 border-purple-500 text-purple-200",idle:"hover:bg-purple-900/30"},ontology:{active:"bg-blue-700/50 border-blue-500 text-blue-200",idle:"hover:bg-blue-900/30"},phenomenology:{active:"bg-teal-700/50 border-teal-500 text-teal-200",idle:"hover:bg-teal-900/30"},history:{active:"bg-amber-700/50 border-amber-500 text-amber-200",idle:"hover:bg-amber-900/30"},policy:{active:"bg-red-700/50 border-red-500 text-red-200",idle:"hover:bg-red-900/30"},physics:{active:"bg-cyan-700/50 border-cyan-500 text-cyan-200",idle:"hover:bg-cyan-900/30"},finance:{active:"bg-green-700/50 border-green-500 text-green-200",idle:"hover:bg-green-900/30"},security:{active:"bg-orange-700/50 border-orange-500 text-orange-200",idle:"hover:bg-orange-900/30"},semiotics:{active:"bg-indigo-700/50 border-indigo-500 text-indigo-200",idle:"hover:bg-indigo-900/30"}},A=["amber","green","red","blue","purple"];function H(){const n=document.getElementById("word-cards");n.innerHTML=S.map((e,t)=>`
        <div class="bg-kernel-900 border border-kernel-700 rounded-lg p-4">
          <div class="flex items-center justify-between mb-2">
            <span class="text-${A[t]}-400 font-bold">${e.word}</span>
            <span class="font-mono text-kernel-500 text-xs">${e.symbol}</span>
          </div>
          <div class="text-kernel-500 text-xs italic mb-2">${e.latin}</div>
          <div class="font-mono text-kernel-600 text-xs mb-2">${e.formula}</div>
          <div class="text-kernel-400 text-xs mb-2">${e.operational}</div>
          <div class="text-xs text-kernel-600 border-t border-kernel-800 pt-2 mt-2">
            <span class="text-kernel-500 font-medium">Ledger:</span> ${e.ledgerRole}
          </div>
          <div class="text-xs text-kernel-700 mt-1 italic">${e.morphology}</div>
        </div>`).join("")}function a(){const n=document.getElementById("lens-buttons");n.innerHTML=d.map(e=>{const t=l.has(e.id),s=F[e.id]||{active:"bg-kernel-700 border-kernel-500 text-kernel-200",idle:""},i=t?`${s.active} ring-1 ring-offset-1 ring-offset-kernel-900`:`bg-kernel-800/40 border-kernel-700 text-kernel-500 ${s.idle}`;return`<button data-lens="${e.id}"
        class="lens-btn px-3 py-1.5 text-xs rounded border transition cursor-pointer ${i}">
        ${e.name}</button>`}).join(""),n.querySelectorAll(".lens-btn").forEach(e=>{e.addEventListener("click",()=>{const t=e.dataset.lens;l.has(t)?l.delete(t):l.add(t),a(),c()}),e.addEventListener("dblclick",()=>{v(e.dataset.lens)})})}function v(n){const e=d.find(s=>s.id===n);if(!e)return;document.getElementById("lens-detail").classList.remove("hidden"),document.getElementById("lens-detail-name").textContent=e.name,document.getElementById("lens-detail-desc").textContent=e.description,document.getElementById("lens-detail-example").textContent=e.example}function c(){const n=document.getElementById("rosetta-body"),e=d.filter(t=>l.has(t.id));if(e.length===0){n.innerHTML='<tr><td colspan="5" class="text-center text-kernel-500 p-4">Select at least one lens.</td></tr>';return}n.innerHTML=e.map(t=>`<tr class="border-b border-kernel-800 hover:bg-kernel-800/30 transition cursor-pointer" data-lens-row="${t.id}">
        <td class="p-2 font-medium text-kernel-300 text-xs whitespace-nowrap">${t.name}</td>
        <td class="p-2 text-amber-300/80 text-xs">${t.drift}</td>
        <td class="p-2 text-green-300/80 text-xs">${t.fidelity}</td>
        <td class="p-2 text-red-300/80 text-xs">${t.roughness}</td>
        <td class="p-2 text-blue-300/80 text-xs">${t.return_}</td>
      </tr>`).join(""),n.querySelectorAll("tr[data-lens-row]").forEach(t=>{t.addEventListener("click",()=>v(t.dataset.lensRow))})}function R(){const n=document.getElementById("scenario-buttons");n.innerHTML=g.map((e,t)=>`<button data-scenario="${t}"
        class="scenario-btn px-3 py-1.5 text-xs rounded border border-kernel-700 bg-kernel-800 text-kernel-400 hover:bg-kernel-700 hover:text-kernel-200 transition">
        ${e.name}</button>`).join(""),n.querySelectorAll(".scenario-btn").forEach(e=>{e.addEventListener("click",()=>{const t=parseInt(e.dataset.scenario,10);j(t),n.querySelectorAll(".scenario-btn").forEach(s=>s.classList.remove("ring-1","ring-kernel-400","text-kernel-200")),e.classList.add("ring-1","ring-kernel-400","text-kernel-200")})})}function j(n){const e=g[n];document.getElementById("scenario-detail").classList.remove("hidden");const s=B(e.channels),i=C(s),u=T(s),p=document.getElementById("channel-bars");p.innerHTML=e.channels.map((r,o)=>{const L=(r*100).toFixed(0),m=e.channelNames[o].replace(/_/g," "),I=r<.3?"bg-red-500":r<.7?"bg-amber-500":"bg-green-500";return`<div class="flex items-center gap-2">
        <span class="text-kernel-500 text-xs w-36 truncate" title="${m}">${m}</span>
        <div class="flex-1 h-3 bg-kernel-800 rounded overflow-hidden">
          <div class="${I} h-full rounded" style="width: ${L}%"></div>
        </div>
        <span class="text-kernel-600 text-xs w-10 text-right">${r.toFixed(2)}</span>
      </div>`}).join("");const f=document.getElementById("scenario-invariants"),k=[{label:"F",value:s.F.toFixed(4),color:"green"},{label:"ω",value:s.omega.toFixed(4),color:"amber"},{label:"IC",value:s.IC.toFixed(4),color:"purple"},{label:"Δ",value:s.delta.toFixed(4),color:"kernel-300"},{label:"S",value:s.S.toFixed(4),color:"cyan"},{label:"C",value:s.C.toFixed(4),color:"red"}];f.innerHTML=k.map(r=>`<div class="bg-kernel-800 rounded p-2 text-center">
        <div class="text-${r.color}-400 text-xs font-medium">${r.label}</div>
        <div class="text-kernel-200 font-mono text-sm">${r.value}</div>
      </div>`).join("");const x=document.getElementById("scenario-regime"),y={STABLE:"bg-green-900/50 text-green-400 border border-green-700",WATCH:"bg-amber-900/50 text-amber-400 border border-amber-700",COLLAPSE:"bg-red-900/50 text-red-400 border border-red-700"};x.className=`mt-2 text-center text-sm font-bold rounded py-1 ${y[i.regime]||""}`,x.textContent=i.regime+(i.critical?" + CRITICAL":"");const h=document.getElementById("scenario-identities");h.innerHTML=u.map(r=>{const o=r.passed;return`<div class="text-xs ${o?"text-green-400":"text-red-400"}">
        ${o?"✓":"✗"} ${r.name}: ${r.value.toExponential(2)}</div>`}).join("");const $=document.getElementById("scenario-translations"),E=d.filter(r=>l.has(r.id));$.innerHTML=E.map(r=>`<div class="bg-kernel-800 rounded p-3 border border-kernel-700">
        <div class="font-medium text-xs text-kernel-300 mb-2">${r.name}</div>
        <div class="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs">
          <div><span class="text-amber-500 font-medium">Drift:</span> <span class="text-kernel-400">${r.drift.split("—")[0].trim()}</span></div>
          <div><span class="text-green-500 font-medium">Fidelity:</span> <span class="text-kernel-400">${r.fidelity.split("—")[0].trim()}</span></div>
          <div><span class="text-red-500 font-medium">Roughness:</span> <span class="text-kernel-400">${r.roughness.split("—")[0].trim()}</span></div>
          <div><span class="text-blue-500 font-medium">Return:</span> <span class="text-kernel-400">${r.return_.split("—")[0].trim()}</span></div>
        </div>
      </div>`).join("")}function M(){const n=document.getElementById("spine-stops");n.innerHTML=b.map((e,t)=>{const s=t===b.length-1;return`
        <div class="flex-1 flex flex-col items-center text-center">
          <div class="bg-kernel-800 border border-kernel-700 rounded-lg p-3 w-full">
            <div class="text-kernel-200 font-bold text-sm">${e.stop}</div>
            <div class="text-kernel-600 text-xs italic">${e.latin}</div>
            <div class="text-kernel-500 text-xs mt-1">${e.verb}</div>
            <div class="text-kernel-400 text-xs mt-2">${e.role}</div>
          </div>
          ${s?"":'<div class="text-kernel-600 text-lg my-1 md:hidden">↓</div>'}
        </div>
        ${s?"":'<div class="hidden md:flex items-center text-kernel-600 text-lg px-1">→</div>'}`}).join("")}document.getElementById("lens-all").addEventListener("click",()=>{d.forEach(n=>l.add(n.id)),a(),c()});document.getElementById("lens-none").addEventListener("click",()=>{l.clear(),a(),c()});H();a();c();R();M();

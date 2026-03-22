import{a as o,F as a}from"./constants.wjURl7WN.js";const n=new Set(o.map(r=>r.id)),i={epistemology:"bg-purple-700/40 border-purple-600 hover:bg-purple-700/60",ontology:"bg-blue-700/40 border-blue-600 hover:bg-blue-700/60",phenomenology:"bg-teal-700/40 border-teal-600 hover:bg-teal-700/60",history:"bg-amber-700/40 border-amber-600 hover:bg-amber-700/60",policy:"bg-red-700/40 border-red-600 hover:bg-red-700/60",physics:"bg-cyan-700/40 border-cyan-600 hover:bg-cyan-700/60",finance:"bg-green-700/40 border-green-600 hover:bg-green-700/60",security:"bg-orange-700/40 border-orange-600 hover:bg-orange-700/60",semiotics:"bg-indigo-700/40 border-indigo-600 hover:bg-indigo-700/60"};function d(){const r=document.getElementById("lens-buttons");r.innerHTML=o.map(t=>{const e=n.has(t.id),l=i[t.id]||"bg-kernel-700 border-kernel-600";return`
        <button data-lens="${t.id}"
          class="lens-btn px-3 py-1.5 text-xs rounded border transition
            ${e?l+" text-white":"bg-kernel-800/40 border-kernel-700 text-kernel-500"}">
          ${t.name}
        </button>
      `}).join(""),r.querySelectorAll(".lens-btn").forEach(t=>{t.addEventListener("click",()=>{const e=t.dataset.lens;n.has(e)?n.delete(e):n.add(e),d(),s()})})}function s(){const r=document.getElementById("rosetta-body"),t=o.filter(e=>n.has(e.id));if(t.length===0){r.innerHTML='<tr><td colspan="5" class="text-center text-kernel-500 p-4">Select at least one lens.</td></tr>';return}r.innerHTML=t.map(e=>`
      <tr class="border-b border-kernel-800 hover:bg-kernel-800/30 transition">
        <td class="p-2 font-medium text-kernel-300">${e.name}</td>
        <td class="p-2 text-amber-300/80 text-sm">${e.drift}</td>
        <td class="p-2 text-green-300/80 text-sm">${e.fidelity}</td>
        <td class="p-2 text-red-300/80 text-sm">${e.roughness}</td>
        <td class="p-2 text-blue-300/80 text-sm">${e.return_}</td>
      </tr>
    `).join("")}function b(){const r=document.getElementById("word-cards"),t=["amber","green","red","blue","purple"];r.innerHTML=a.map((e,l)=>`
      <div class="bg-kernel-800 rounded-lg p-3 border border-kernel-700">
        <div class="text-${t[l]}-400 font-bold text-sm mb-1">${e.word}</div>
        <div class="text-kernel-500 text-xs italic mb-2">${e.latin}</div>
        <div class="text-kernel-400 text-xs">${e.operational}</div>
        <div class="mt-2 pt-2 border-t border-kernel-700">
          <span class="text-kernel-600 text-xs">${e.ledgerRole}</span>
        </div>
      </div>
    `).join("")}document.getElementById("lens-all").addEventListener("click",()=>{o.forEach(r=>n.add(r.id)),d(),s()});document.getElementById("lens-none").addEventListener("click",()=>{n.clear(),d(),s()});d();s();b();

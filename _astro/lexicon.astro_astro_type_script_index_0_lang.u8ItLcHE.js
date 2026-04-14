import{L as a,c as o}from"./constants.DtcEBpaC.js";const l=document.getElementById("lexicon-body");l.innerHTML=a.map(e=>{const r=e.symbol==="—"?"text-kernel-600":"text-amber-400 font-mono";return`
      <tr class="border-b border-kernel-800 hover:bg-kernel-800/30 transition">
        <td class="p-2 text-kernel-200 font-medium italic">${e.latin}</td>
        <td class="p-2 ${r}">${e.symbol}</td>
        <td class="p-2 text-kernel-400">${e.literal}</td>
        <td class="p-2 text-kernel-500 text-xs">${e.operational}</td>
      </tr>`}).join("");const s=o,d=[...new Set(s.map(e=>e.category))],b={"Channel Collapse":{border:"border-red-700/50",badge:"bg-red-900/30 text-red-400"},Seam:{border:"border-amber-700/50",badge:"bg-amber-900/30 text-amber-400"},"Structural States":{border:"border-blue-700/50",badge:"bg-blue-900/30 text-blue-400"},"Scale Ladder":{border:"border-green-700/50",badge:"bg-green-900/30 text-green-400"}},i=document.getElementById("phenomena-sections");i.innerHTML=d.map(e=>{const r=s.filter(t=>t.category===e),n=b[e]||{border:"border-kernel-700",badge:"bg-kernel-800 text-kernel-400"};return`
      <div>
        <div class="flex items-center gap-2 mb-3">
          <span class="text-xs px-2 py-0.5 rounded ${n.badge}">${e}</span>
          <span class="text-kernel-600 text-xs">${r.length} terms</span>
        </div>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
          ${r.map(t=>`
            <div class="bg-kernel-800 rounded-lg p-3 border ${n.border}">
              <div class="flex items-center justify-between mb-1">
                <span class="text-kernel-200 font-medium italic text-sm">${t.latin}</span>
                <span class="text-kernel-500 text-xs">${t.english}</span>
              </div>
              <p class="text-kernel-500 text-xs">${t.operational}</p>
            </div>
          `).join("")}
        </div>
      </div>`}).join("");

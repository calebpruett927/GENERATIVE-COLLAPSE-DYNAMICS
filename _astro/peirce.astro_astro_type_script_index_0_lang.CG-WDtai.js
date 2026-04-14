import{d as s}from"./constants.DtcEBpaC.js";const n=document.getElementById("peirce-table-container");n.innerHTML=`
    <table class="w-full text-sm">
      <thead>
        <tr class="border-b border-kernel-600 text-kernel-400">
          <th class="text-left p-2 font-medium">Peirce</th>
          <th class="text-left p-2 font-medium">GCD Equivalent</th>
          <th class="text-left p-2 font-medium">Role</th>
          <th class="text-left p-2 font-medium text-kernel-500">Note</th>
        </tr>
      </thead>
      <tbody>
        ${s.map(e=>{const t=e.peirce.includes("MISSING");return`
            <tr class="${t?"bg-amber-950/30 border-b border-amber-800/50":"border-b border-kernel-800 hover:bg-kernel-800/30 transition"}">
              <td class="p-2 ${t?"text-amber-400 font-bold":"text-kernel-300"} text-sm">${e.peirce}</td>
              <td class="p-2 text-kernel-200 font-mono text-xs">${e.gcd}</td>
              <td class="p-2 text-kernel-400 text-xs">${e.role}</td>
              <td class="p-2 text-kernel-500 text-xs">${e.note}</td>
            </tr>`}).join("")}
      </tbody>
    </table>`;const o=[{title:"Highest Coherence",items:[{system:"Latin Lexicon",value:"IC/F = 0.933",note:"Morphological constraints → channel coherence"},{system:"Mathematical Notation",value:"IC/F = 0.927",note:"Formal systems minimize channel heterogeneity"},{system:"Music Notation",value:"IC/F = 0.891",note:"Pitch, duration, dynamics all cohere"}],color:"green"},{title:"Moderate Coherence",items:[{system:"Natural Language (English)",value:"IC/F = 0.712",note:"Ambiguity creates heterogeneity gap"},{system:"ASL",value:"IC/F = 0.680",note:"Visual modality trades some channels"},{system:"Emoji",value:"IC/F = 0.423",note:"High ground drift, low compositional depth"}],color:"amber"},{title:"Low Coherence",items:[{system:"Pirahã",value:"IC/F = 0.234",note:"Minimal recursion, no number words"},{system:"Morse Code",value:"IC/F = 0.312",note:"Binary encoding = extreme channel narrowing"},{system:"Smoke Signals",value:"IC/F = 0.198",note:"Low precision, high environmental noise"}],color:"red"}],i=document.getElementById("semiotic-findings");i.innerHTML=o.map(e=>`
    <div class="bg-kernel-800 rounded-lg p-3 border border-kernel-700">
      <div class="text-${e.color}-400 font-bold text-xs mb-2">${e.title}</div>
      <div class="space-y-2">
        ${e.items.map(t=>`
          <div>
            <div class="flex items-center justify-between">
              <span class="text-kernel-300 text-xs">${t.system}</span>
              <span class="font-mono text-${e.color}-400/80 text-xs">${t.value}</span>
            </div>
            <div class="text-kernel-600 text-[10px]">${t.note}</div>
          </div>
        `).join("")}
      </div>
    </div>
  `).join("");

import{D as s}from"./constants.DtcEBpaC.js";const i=[{bg:"bg-blue-900/30",border:"border-blue-700",accent:"text-blue-300",icon:"📋"},{bg:"bg-amber-900/30",border:"border-amber-700",accent:"text-amber-300",icon:"📖"},{bg:"bg-purple-900/30",border:"border-purple-700",accent:"text-purple-300",icon:"🔒"},{bg:"bg-green-900/30",border:"border-green-700",accent:"text-green-300",icon:"⚖️"},{bg:"bg-red-900/30",border:"border-red-700",accent:"text-red-300",icon:"🏛️"}],a=["Define rules before evidence. Freeze data sources, normalization procedures, near-wall policy, and thresholds. The contract declares the rules of the sentence before it is written. Nothing computed before the contract is frozen has evidential standing.","Tell the story using the five words: Drift, Fidelity, Roughness, Return, Integrity. The canon is the narrative body — prose-first, auditable by construction. Authors write in ordinary language; the five words provide the vocabulary.","Publish thresholds and their order. No mid-episode edits. Version the closure sheet. Stance must change when thresholds are crossed — this is what makes the system falsifiable. Published grammar rules that bind the narrative.","Debit Drift (D_ω) and Roughness (D_C). Credit Return (R·τ_R). The account must reconcile: |residual| ≤ tol_seam. The conservation budget Δκ = R·τ_R − (D_ω + D_C) is the proof that the sentence is well-formed.","Read from declared gates: Stable / Watch / Collapse, with optional Critical overlay (IC < 0.30). The verdict is derived from the gates — never asserted by the agent. If two agents feed the same data through the same contract, they must arrive at the same stance."],o=document.getElementById("spine-visual");o.innerHTML=`
    <div class="flex flex-col md:flex-row items-stretch gap-0">
      ${s.map((t,r)=>{const e=i[r],n=r===s.length-1;return`
          <div class="flex-1 flex flex-col items-center text-center">
            <div class="${e.bg} border ${e.border} rounded-lg p-4 w-full h-full flex flex-col justify-center">
              <div class="text-2xl mb-1">${e.icon}</div>
              <div class="${e.accent} font-bold text-sm">${t.stop}</div>
              <div class="text-kernel-600 text-xs italic">${t.latin}</div>
              <div class="text-kernel-500 text-xs mt-1 font-medium">${t.verb}</div>
            </div>
            ${n?"":'<div class="text-kernel-600 text-lg my-1 md:hidden">↓</div>'}
          </div>
          ${n?"":'<div class="hidden md:flex items-center text-kernel-500 text-xl px-2 font-bold">→</div>'}`}).join("")}
    </div>`;const d=document.getElementById("spine-cards");d.innerHTML=s.map((t,r)=>{const e=i[r];return`
      <div class="${e.bg} border ${e.border} rounded-lg p-4">
        <div class="flex items-center gap-3 mb-2">
          <span class="text-xl">${e.icon}</span>
          <div>
            <span class="${e.accent} font-bold">${t.stop}</span>
            <span class="text-kernel-600 text-xs italic ml-2">${t.latin} — ${t.verb}</span>
          </div>
        </div>
        <p class="text-sm text-kernel-400">${a[r]}</p>
      </div>`}).join("");

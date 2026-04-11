import{e as p,f as x}from"./constants.6x37F0HI.js";import{c as u,a as k,v as h,e as f}from"./kernel.Cdj69J8o.js";const c=document.getElementById("scenario-nav"),$=document.getElementById("scenario-detail");let l=0;function y(s){switch(s){case"STABLE":return{bg:"bg-green-900/30",text:"text-green-400",border:"border-green-700/50"};case"WATCH":return{bg:"bg-amber-900/30",text:"text-amber-400",border:"border-amber-700/50"};case"COLLAPSE":return{bg:"bg-red-900/30",text:"text-red-400",border:"border-red-700/50"};default:return{bg:"bg-kernel-800",text:"text-kernel-300",border:"border-kernel-700"}}}function F(){c.innerHTML=p.map((s,n)=>`
      <button data-idx="${n}" class="scenario-btn px-3 py-1.5 rounded text-xs font-medium transition
        ${n===l?"bg-kernel-700 text-kernel-200":"bg-kernel-800 text-kernel-400 hover:text-kernel-300"}">
        ${s.name}
      </button>
    `).join(""),c.querySelectorAll(".scenario-btn").forEach(s=>s.addEventListener("click",()=>{l=parseInt(s.dataset.idx),m()}))}function m(){F();const s=p[l],n=s.channels.map(()=>1/s.channels.length),t=u(s.channels,n),r=k(t),d=h(t),i=f(t),a=y(r.regime),o=d.every(e=>e.passed);$.innerHTML=`
      <!-- Scenario header -->
      <div class="bg-kernel-900 border border-kernel-700 rounded-lg p-6">
        <div class="flex items-start justify-between mb-4">
          <div>
            <h2 class="text-xl font-bold text-kernel-100">${s.name}</h2>
            <p class="text-sm text-kernel-400 mt-1">${s.description}</p>
          </div>
          <div class="${a.bg} ${a.border} border rounded px-3 py-1.5">
            <span class="${a.text} font-bold text-sm">${r.regime}${r.isCritical?" + CRITICAL":""}</span>
          </div>
        </div>

        <!-- Channel values -->
        <div class="mb-4">
          <h3 class="text-xs text-kernel-500 font-medium mb-2">TRACE VECTOR (${s.channels.length} channels)</h3>
          <div class="grid grid-cols-2 md:grid-cols-4 gap-2">
            ${s.channels.map((e,b)=>{const v=e*100,g=e>.7?"bg-green-500":e>.4?"bg-amber-500":"bg-red-500";return`
                <div class="bg-kernel-800 rounded p-2">
                  <div class="flex justify-between text-xs mb-1">
                    <span class="text-kernel-500 truncate">${s.channelNames[b]}</span>
                    <span class="text-kernel-300 font-mono">${e.toFixed(2)}</span>
                  </div>
                  <div class="h-1.5 bg-kernel-700 rounded overflow-hidden">
                    <div class="${g} h-full rounded" style="width: ${v}%"></div>
                  </div>
                </div>`}).join("")}
          </div>
        </div>

        <!-- Kernel output -->
        <div class="grid grid-cols-3 md:grid-cols-6 gap-2 mb-4">
          ${[{l:"F",v:t.F.toFixed(4),c:"text-green-400"},{l:"ω",v:t.omega.toFixed(4),c:t.omega>=.3?"text-red-400":"text-amber-400"},{l:"S",v:t.S.toFixed(4),c:"text-blue-400"},{l:"C",v:t.C.toFixed(4),c:"text-purple-400"},{l:"κ",v:t.kappa.toFixed(4),c:"text-cyan-400"},{l:"IC",v:t.IC.toFixed(4),c:t.IC<.3?"text-red-400":"text-kernel-200"}].map(e=>`
            <div class="bg-kernel-800 rounded p-2 text-center">
              <div class="text-xs text-kernel-500">${e.l}</div>
              <div class="font-mono text-sm ${e.c}">${e.v}</div>
            </div>
          `).join("")}
        </div>

        <!-- Identity checks -->
        <div class="flex items-center gap-2 mb-2">
          <span class="text-xs text-kernel-500">Identities:</span>
          ${d.map(e=>`<span class="text-xs ${e.passed?"text-green-400":"text-red-400"} font-mono">${e.name} ${e.passed?"✓":"✗"}</span>`).join('<span class="text-kernel-700">·</span>')}
          <span class="text-xs ml-2 ${o?"text-green-400":"text-red-400"}">${o?"ALL PASS":"IDENTITY FAIL"}</span>
        </div>

        <!-- Seam budget -->
        <div class="bg-kernel-800 rounded p-3 text-xs">
          <div class="flex gap-6">
            <span class="text-kernel-500">Γ(ω): <span class="text-amber-400 font-mono">${i.gamma.toFixed(6)}</span></span>
            <span class="text-kernel-500">D_C: <span class="text-amber-400 font-mono">${i.D_C.toFixed(6)}</span></span>
            <span class="text-kernel-500">Δ: <span class="text-amber-400 font-mono">${t.delta.toFixed(6)}</span></span>
            <span class="text-kernel-500">IC/F: <span class="text-amber-400 font-mono">${(t.F>0?t.IC/t.F:0).toFixed(4)}</span></span>
          </div>
        </div>
      </div>

      <!-- Rosetta lens translations -->
      <div class="bg-kernel-900 border border-kernel-700 rounded-lg p-6">
        <h2 class="text-lg font-bold text-kernel-200 mb-1">Through Every Lens</h2>
        <p class="text-xs text-kernel-500 mb-4">
          The same kernel result — same F, same ω, same IC — read through 9 Rosetta lenses.
          Each lens translates the five words into its own dialect. The verdict is identical.
        </p>
        <div class="space-y-3">
          ${x.map(e=>`
            <details class="group">
              <summary class="flex items-center gap-3 cursor-pointer bg-kernel-800 rounded-lg px-4 py-3 hover:bg-kernel-800/80 transition">
                <span class="text-sm font-bold text-kernel-200">${e.name}</span>
                <span class="text-xs text-kernel-500 ml-auto group-open:hidden">expand ▸</span>
                <span class="text-xs text-kernel-500 hidden group-open:inline">collapse ▾</span>
              </summary>
              <div class="bg-kernel-850 rounded-b-lg px-4 py-3 border-t border-kernel-700 ml-0">
                <div class="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
                  <div>
                    <div class="text-green-400 font-medium mb-1">Drift (ω = ${t.omega.toFixed(4)})</div>
                    <p class="text-kernel-400">${e.drift}</p>
                  </div>
                  <div>
                    <div class="text-blue-400 font-medium mb-1">Fidelity (F = ${t.F.toFixed(4)})</div>
                    <p class="text-kernel-400">${e.fidelity}</p>
                  </div>
                  <div>
                    <div class="text-amber-400 font-medium mb-1">Roughness (C = ${t.C.toFixed(4)})</div>
                    <p class="text-kernel-400">${e.roughness}</p>
                  </div>
                  <div>
                    <div class="text-purple-400 font-medium mb-1">Return (τ_R)</div>
                    <p class="text-kernel-400">${e.return_}</p>
                  </div>
                </div>
                <div class="mt-2 pt-2 border-t border-kernel-700">
                  <div class="text-cyan-400 font-medium text-xs mb-1">Integrity (IC = ${t.IC.toFixed(4)}, Δ = ${t.delta.toFixed(4)})</div>
                  <p class="text-kernel-400 text-xs">${e.integrity}</p>
                </div>
                <p class="text-kernel-500 text-xs mt-2 italic">${e.description}</p>
              </div>
            </details>
          `).join("")}
        </div>
      </div>

      <!-- Cross-domain comparison matrix -->
      <div class="bg-kernel-900 border border-kernel-700 rounded-lg p-6">
        <h2 class="text-lg font-bold text-kernel-200 mb-1">The Equalizer Proof</h2>
        <p class="text-xs text-kernel-500 mb-4">
          Nine lenses, one computation, one verdict. The table below shows that every lens arrives at
          the same regime classification and the same IC value. Only the vocabulary differs.
          <em>Non agens mensurat, sed structura.</em>
        </p>
        <div class="overflow-x-auto">
          <table class="w-full text-xs">
            <thead>
              <tr class="border-b border-kernel-700">
                <th class="text-left py-2 px-2 text-kernel-400">Lens</th>
                <th class="text-center py-2 px-2 text-kernel-400">F</th>
                <th class="text-center py-2 px-2 text-kernel-400">ω</th>
                <th class="text-center py-2 px-2 text-kernel-400">IC</th>
                <th class="text-center py-2 px-2 text-kernel-400">Δ</th>
                <th class="text-center py-2 px-2 text-kernel-400">Regime</th>
                <th class="text-left py-2 px-2 text-kernel-400">"What drifted?"</th>
              </tr>
            </thead>
            <tbody>
              ${x.map(e=>`
                <tr class="border-b border-kernel-800">
                  <td class="py-1.5 px-2 text-kernel-300 font-medium">${e.name}</td>
                  <td class="py-1.5 px-2 text-center font-mono text-kernel-400">${t.F.toFixed(3)}</td>
                  <td class="py-1.5 px-2 text-center font-mono text-kernel-400">${t.omega.toFixed(3)}</td>
                  <td class="py-1.5 px-2 text-center font-mono text-kernel-400">${t.IC.toFixed(3)}</td>
                  <td class="py-1.5 px-2 text-center font-mono text-kernel-400">${t.delta.toFixed(3)}</td>
                  <td class="py-1.5 px-2 text-center ${a.text}">${r.regime}</td>
                  <td class="py-1.5 px-2 text-kernel-500 truncate max-w-[200px]">${e.drift.split("—")[0].trim()}</td>
                </tr>
              `).join("")}
            </tbody>
          </table>
        </div>
        <div class="mt-3 p-2 bg-kernel-800/50 rounded text-xs text-kernel-500 text-center">
          All rows: identical F, ω, IC, Δ, Regime. The numbers don't change — only the reading.
        </div>
      </div>`}m();

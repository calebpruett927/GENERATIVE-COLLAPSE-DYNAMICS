import{F as b,E as u,P as v,A as g,g as f,T as h,a as y,O as k,C as $,R as C,K as I}from"./constants.DtcEBpaC.js";import{c as F,a as E,v as T}from"./kernel.1iDFzIM1.js";const d=document.querySelectorAll(".tab-btn"),_=document.querySelectorAll(".tab-content");d.forEach(e=>e.addEventListener("click",()=>{const s=e.dataset.tab;_.forEach(r=>r.classList.add("hidden")),document.getElementById(`tab-${s}`).classList.remove("hidden"),d.forEach(r=>{r.className="tab-btn px-3 py-1.5 rounded text-xs font-medium bg-kernel-800 text-kernel-400"}),e.className="tab-btn px-3 py-1.5 rounded text-xs font-medium bg-kernel-700 text-kernel-200"}));const L=Object.values(I);document.getElementById("tab-symbols").innerHTML=`
    <div class="bg-kernel-900 border border-kernel-700 rounded-lg p-6">
      <h2 class="text-lg font-bold text-kernel-200 mb-1">Tier-1 Kernel Symbols</h2>
      <p class="text-xs text-kernel-500 mb-4">
        The kernel K: [0,1]ⁿ × Δⁿ → (F, ω, S, C, κ, IC). Four primitive equations (F, κ, S, C)
        and two derived values (ω = 1−F, IC = exp(κ)), with 3 effective degrees of freedom.
        These symbols are reserved and immutable — any redefinition is symbol capture.
      </p>
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        ${L.map(e=>`
          <div class="bg-kernel-800 rounded-lg p-4 border border-kernel-700 hover:border-kernel-500 transition">
            <div class="flex items-baseline gap-2 mb-2">
              <span class="text-2xl font-bold text-kernel-100 font-mono">${e.symbol}</span>
              <span class="text-sm text-kernel-300">${e.name}</span>
            </div>
            <div class="text-xs text-kernel-500 italic mb-2">${e.latin}</div>
            <div class="bg-kernel-900 rounded px-3 py-2 text-sm font-mono text-amber-400 mb-2">${e.formula}</div>
            <div class="text-xs text-kernel-500">Range: <span class="text-kernel-300">${e.range}</span></div>
          </div>
        `).join("")}
      </div>
      <div class="mt-4 p-3 bg-kernel-800/50 rounded border border-kernel-700">
        <div class="text-xs text-kernel-400">
          <strong class="text-kernel-300">3 Effective DOF:</strong> Although 6 values are computed,
          ω = 1−F (derived), IC = exp(κ) (derived), and S ≈ f(F,C) (asymptotically determined).
          Only <span class="text-amber-400 font-mono">F</span>,
          <span class="text-amber-400 font-mono">κ</span>, and
          <span class="text-amber-400 font-mono">C</span> are independent.
        </div>
      </div>
    </div>`;const S=b;document.getElementById("tab-words").innerHTML=`
    <div class="bg-kernel-900 border border-kernel-700 rounded-lg p-6">
      <h2 class="text-lg font-bold text-kernel-200 mb-1">The Five-Word Vocabulary</h2>
      <p class="text-xs text-kernel-500 mb-4">
        These five plain-language words are the prose interface — the minimal lingua franca for narrating
        any collapse-return cycle through the Canon. Each word has an operational meaning tied to the
        frozen Contract and reconciled in the Integrity Ledger.
      </p>
      <div class="space-y-4">
        ${S.map((e,s)=>`
            <div class="bg-kernel-800 rounded-lg p-4 border border-kernel-700">
              <div class="flex items-center gap-3 mb-2">
                <span class="text-xl font-bold text-${["green","blue","amber","purple","cyan"][s%5]}-400">${e.word}</span>
                <span class="text-xs text-kernel-500 italic">${e.latin}</span>
                <span class="font-mono text-sm text-kernel-300 ml-auto">${e.symbol}</span>
              </div>
              <div class="bg-kernel-900 rounded px-3 py-1.5 text-xs font-mono text-amber-400 mb-2">${e.formula}</div>
              <p class="text-sm text-kernel-400 mb-2">${e.operational}</p>
              <div class="grid grid-cols-2 gap-2 text-xs">
                <div><span class="text-kernel-500">Ledger role:</span> <span class="text-kernel-300">${e.ledgerRole}</span></div>
                <div><span class="text-kernel-500">Morphology:</span> <span class="text-kernel-300">${e.morphology}</span></div>
              </div>
            </div>`).join("")}
      </div>
      <div class="mt-4 p-3 bg-kernel-800/50 rounded border border-kernel-700">
        <div class="text-xs text-kernel-400">
          <strong class="text-kernel-300">Conservation budget:</strong>
          <span class="font-mono text-amber-400">Δκ = R·τ_R − (D_ω + D_C)</span>.
          Debits (Drift + Roughness) must balance credits (Return). The budget is the <em>semantic warranty</em>
          behind the prose — it explains <em>why</em> the ledger must reconcile.
        </div>
      </div>
    </div>`;const A=[{name:"Duality Identity",latin:"Complementum Perfectum",formula:"F + ω = 1",proof:"F = Σ wᵢcᵢ, ω = 1 − F ⟹ F + ω = 1 exactly. Verified to residual 0.0e+00 across 10K random traces.",meaning:"Collapse and fidelity are complements. There is no third possibility — tertia via nulla. In Fisher coordinates, this is sin²θ + cos²θ = 1.",verified:"0.0e+00"},{name:"Integrity Bound",latin:"Limbus Integritatis",formula:"IC ≤ F",proof:"IC = exp(Σ wᵢ ln cᵢ) is the weighted geometric mean. F = Σ wᵢcᵢ is the weighted arithmetic mean. GM ≤ AM, with equality iff all cᵢ equal. More fundamentally: IC ≤ F is the solvability condition — for n=2, c₁,₂ = F ± √(F²−IC²) requires IC ≤ F for real solutions.",meaning:"Integrity cannot exceed fidelity. One dead channel kills IC via geometric slaughter while F stays healthy. The gap Δ = F − IC is the heterogeneity gap measuring channel divergence.",verified:"100% (all 23 domains)"},{name:"Log-Integrity Relation",latin:"Relatio Log-Integritatis",formula:"IC = exp(κ)",proof:"κ = Σ wᵢ ln(cᵢ,ε) ⟹ exp(κ) = exp(Σ wᵢ ln cᵢ) = Π cᵢ^wᵢ = IC. The logarithm converts the multiplicative coherence measure to an additive one.",meaning:"The link between multiplicative and additive coherence. κ is the log-space view of IC. This enables additive composition of seam budgets while preserving the multiplicative structure of integrity.",verified:"exact (by definition)"}],l={name:"Entropy Constraint",latin:"Limbus Entropiae",formula:"S ≈ f(F, C)",proof:"As n → ∞, the central limit theorem constrains the Bernoulli field entropy to be asymptotically determined by F and C. Empirically: corr(C, S) → −1.",meaning:"This is NOT an identity — it is a statistical constraint that tightens with increasing channel count n. It reduces the effective DOF from 4 to 3 (F, κ, C)."};document.getElementById("tab-identities").innerHTML=`
    <div class="bg-kernel-900 border border-kernel-700 rounded-lg p-6">
      <h2 class="text-lg font-bold text-kernel-200 mb-1">Algebraic Identities</h2>
      <p class="text-xs text-kernel-500 mb-4">
        Three identities that hold exactly by construction, plus one statistical constraint.
        These reduce 6 kernel outputs to 3 effective degrees of freedom.
      </p>
      <div class="space-y-4 mb-6">
        ${A.map((e,s)=>`
          <div class="bg-kernel-800 rounded-lg p-5 border border-kernel-700">
            <div class="flex items-center gap-3 mb-3">
              <span class="w-8 h-8 rounded-full bg-kernel-700 flex items-center justify-center text-sm font-bold text-kernel-200">${s+1}</span>
              <div>
                <span class="font-bold text-kernel-100">${e.name}</span>
                <span class="text-xs text-kernel-500 italic ml-2">${e.latin}</span>
              </div>
              <span class="ml-auto text-xs bg-green-900/50 text-green-400 px-2 py-0.5 rounded">verified: ${e.verified}</span>
            </div>
            <div class="bg-kernel-900 rounded px-4 py-3 text-center text-lg font-mono text-amber-400 mb-3">${e.formula}</div>
            <details class="text-sm">
              <summary class="text-kernel-300 cursor-pointer hover:text-kernel-100 transition">Proof sketch</summary>
              <p class="text-kernel-400 mt-2 text-xs">${e.proof}</p>
            </details>
            <p class="text-xs text-kernel-500 mt-2">${e.meaning}</p>
          </div>
        `).join("")}
      </div>
      <div class="bg-kernel-800 rounded-lg p-5 border border-amber-700/30">
        <div class="flex items-center gap-3 mb-3">
          <span class="w-8 h-8 rounded-full bg-amber-900/50 flex items-center justify-center text-sm font-bold text-amber-400">≈</span>
          <div>
            <span class="font-bold text-kernel-100">${l.name}</span>
            <span class="text-xs text-kernel-500 italic ml-2">${l.latin}</span>
          </div>
          <span class="ml-auto text-xs bg-amber-900/50 text-amber-400 px-2 py-0.5 rounded">statistical</span>
        </div>
        <div class="bg-kernel-900 rounded px-4 py-3 text-center text-lg font-mono text-amber-400 mb-3">${l.formula}</div>
        <details class="text-sm">
          <summary class="text-kernel-300 cursor-pointer hover:text-kernel-100 transition">Proof sketch</summary>
          <p class="text-kernel-400 mt-2 text-xs">${l.proof}</p>
        </details>
        <p class="text-xs text-kernel-500 mt-2">${l.meaning}</p>
      </div>
    </div>`;const w=[{name:"ε (EPSILON)",value:`${u}`,source:"Guard band",role:"Pole at ω=1 does not affect measurements to machine precision. Prevents log(0) in κ computation."},{name:"p (P_EXPONENT)",value:`${v}`,source:"Cardano root",role:"Unique integer where ω_trap is a root of x³ + x − 1 = 0. Governs Γ(ω) = ω^p/(1−ω+ε)."},{name:"α (ALPHA)",value:`${g}`,source:"Unit coupling",role:"Curvature cost coefficient: D_C = α·C. Unit coupling — curvature debit equals curvature directly."},{name:"λ (LAMBDA)",value:`${f}`,source:"Auxiliary",role:"Auxiliary coefficient for extended diagnostics."},{name:"tol_seam (TOL_SEAM)",value:`${h}`,source:"Seam closure",role:"Width where IC ≤ F holds at 100% across all 23 domains. |s| ≤ tol_seam for seam PASS."},{name:"c* (C_STAR)",value:`${y}`,source:"Logistic fixed point",role:"Self-dual fixed point of the logistic map. Maximizes S + κ per channel."},{name:"ω_trap (OMEGA_TRAP)",value:`${k}`,source:"Cardano root",role:"Drift trapping point: x³ + x − 1 = 0 where x = ω_trap. c_trap = 1 − ω_trap."},{name:"c_trap (C_TRAP)",value:`${$}`,source:"From ω_trap",role:"Channel-space trapping point. Location of the first weld (Γ drops below 1.0)."}];document.getElementById("tab-frozen").innerHTML=`
    <div class="bg-kernel-900 border border-kernel-700 rounded-lg p-6">
      <h2 class="text-lg font-bold text-kernel-200 mb-1">Frozen Parameters</h2>
      <p class="text-xs text-kernel-500 mb-4">
        <em>Trans suturam congelatum</em> — frozen across the seam. These values are the unique constants
        where seams close consistently across all 23 domains. They are discovered by the mathematics,
        not chosen by convention. Never say "we chose" or "hyperparameter."
      </p>
      <div class="overflow-x-auto">
        <table class="w-full text-sm">
          <thead>
            <tr class="border-b border-kernel-700">
              <th class="text-left py-2 px-3 text-kernel-400 font-medium text-xs">Parameter</th>
              <th class="text-right py-2 px-3 text-kernel-400 font-medium text-xs">Value</th>
              <th class="text-left py-2 px-3 text-kernel-400 font-medium text-xs">Source</th>
              <th class="text-left py-2 px-3 text-kernel-400 font-medium text-xs">Role</th>
            </tr>
          </thead>
          <tbody>
            ${w.map(e=>`
              <tr class="border-b border-kernel-800 hover:bg-kernel-800/50 transition">
                <td class="py-2 px-3 font-mono text-kernel-200 text-xs">${e.name}</td>
                <td class="py-2 px-3 text-right font-mono text-amber-400 text-xs">${e.value}</td>
                <td class="py-2 px-3 text-kernel-400 text-xs">${e.source}</td>
                <td class="py-2 px-3 text-kernel-500 text-xs">${e.role}</td>
              </tr>
            `).join("")}
          </tbody>
        </table>
      </div>
    </div>`;const a=C;document.getElementById("tab-regimes").innerHTML=`
    <div class="bg-kernel-900 border border-kernel-700 rounded-lg p-6">
      <h2 class="text-lg font-bold text-kernel-200 mb-1">Regime Gates</h2>
      <p class="text-xs text-kernel-500 mb-4">
        The four-gate criterion translates continuous Tier-1 invariants into discrete regime labels.
        Stable requires ALL four gates to pass simultaneously (conjunctive). Stability is rare — only 12.5% of Fisher space.
      </p>
      <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
        <div class="bg-green-900/20 border border-green-700/50 rounded-lg p-4">
          <div class="text-green-400 font-bold mb-2">Stable (12.5%)</div>
          <ul class="text-xs text-kernel-400 space-y-1">
            <li>ω &lt; ${a.omega_stable_max}</li>
            <li>F &gt; ${a.F_stable_min}</li>
            <li>S &lt; ${a.S_stable_max}</li>
            <li>C &lt; ${a.C_stable_max}</li>
          </ul>
          <div class="text-xs text-kernel-600 mt-2">All four must hold (conjunctive)</div>
        </div>
        <div class="bg-amber-900/20 border border-amber-700/50 rounded-lg p-4">
          <div class="text-amber-400 font-bold mb-2">Watch (24.4%)</div>
          <ul class="text-xs text-kernel-400 space-y-1">
            <li>${a.omega_watch_min} ≤ ω &lt; ${a.omega_watch_max}</li>
            <li>OR: any Stable gate fails</li>
          </ul>
          <div class="text-xs text-kernel-600 mt-2">Intermediate — at least one gate open</div>
        </div>
        <div class="bg-red-900/20 border border-red-700/50 rounded-lg p-4">
          <div class="text-red-400 font-bold mb-2">Collapse (63.1%)</div>
          <ul class="text-xs text-kernel-400 space-y-1">
            <li>ω ≥ ${a.omega_collapse_min}</li>
          </ul>
          <div class="text-xs text-kernel-600 mt-2">Ruptura est fons constantiae</div>
        </div>
      </div>
      <div class="bg-purple-900/20 border border-purple-700/50 rounded-lg p-4">
        <div class="text-purple-400 font-bold mb-2">Critical (overlay)</div>
        <div class="text-xs text-kernel-400">
          IC &lt; ${a.IC_critical_max} — severity overlay, not a regime. Accompanies any regime when integrity dangerously low.
        </div>
      </div>
      <div class="mt-4 p-3 bg-kernel-800/50 rounded border border-kernel-700 text-xs text-kernel-400">
        <strong class="text-kernel-300">Fisher space partition:</strong>
        Collapse occupies 63.1%, Watch 24.4%, Stable only 12.5%.
        Most of the manifold is NOT stable — return to stability is the exception, not the rule.
        This is what makes the Return Axiom meaningful.
      </div>
    </div>`;document.getElementById("tab-verify").innerHTML=`
    <div class="bg-kernel-900 border border-kernel-700 rounded-lg p-6">
      <h2 class="text-lg font-bold text-kernel-200 mb-1">Live Identity Verifier</h2>
      <p class="text-xs text-kernel-500 mb-4">
        Enter channel values and verify all three algebraic identities in real time.
        The kernel runs in your browser — same formulas, same frozen parameters.
      </p>
      <div class="mb-4">
        <label class="text-xs text-kernel-400 block mb-1">
          Channels (comma-separated, values in [0,1]):
        </label>
        <input id="channel-input" type="text" value="0.95, 0.80, 0.60, 0.40, 0.01"
          class="w-full bg-kernel-800 border border-kernel-600 rounded px-3 py-2 text-sm font-mono text-kernel-200 focus:border-amber-500 focus:outline-none" />
        <div class="text-xs text-kernel-600 mt-1">
          Presets:
          <button class="preset-btn text-amber-400 hover:text-amber-300 ml-1" data-vals="0.95,0.92,0.98,0.91,0.96,0.93,0.97,0.94">Perfect</button> ·
          <button class="preset-btn text-amber-400 hover:text-amber-300" data-vals="0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.001">Slaughter</button> ·
          <button class="preset-btn text-amber-400 hover:text-amber-300" data-vals="0.50,0.50,0.50,0.50">Equator</button> ·
          <button class="preset-btn text-amber-400 hover:text-amber-300" data-vals="0.10,0.05,0.02,0.15,0.08">Deep collapse</button>
        </div>
      </div>
      <div id="verify-results"></div>
    </div>`;function o(){const s=document.getElementById("channel-input").value.split(",").map(t=>parseFloat(t.trim())).filter(t=>!isNaN(t)&&t>=0&&t<=1);if(s.length<2){document.getElementById("verify-results").innerHTML='<div class="text-red-400 text-sm">Need at least 2 valid channels in [0,1]</div>';return}const r=s.map(()=>1/s.length),n=F(s,r),i=E(n),m=T(n),x=t=>t?'<span class="text-green-400 font-bold">✓ PASS</span>':'<span class="text-red-400 font-bold">✗ FAIL</span>',p=i.regime==="STABLE"?"text-green-400":i.regime==="WATCH"?"text-amber-400":"text-red-400";document.getElementById("verify-results").innerHTML=`
      <div class="grid grid-cols-2 md:grid-cols-3 gap-3 mb-4">
        ${[{label:"F (Fidelity)",val:n.F.toFixed(8)},{label:"ω (Drift)",val:n.omega.toFixed(8)},{label:"S (Entropy)",val:n.S.toFixed(8)},{label:"C (Curvature)",val:n.C.toFixed(8)},{label:"κ (Log-integrity)",val:n.kappa.toFixed(8)},{label:"IC (Integrity)",val:n.IC.toFixed(8)}].map(t=>`
          <div class="bg-kernel-800 rounded p-2 text-center">
            <div class="text-xs text-kernel-500">${t.label}</div>
            <div class="font-mono text-sm text-kernel-200">${t.val}</div>
          </div>
        `).join("")}
      </div>
      <div class="space-y-2 mb-4">
        ${m.map(t=>`
          <div class="flex items-center justify-between bg-kernel-800 rounded px-4 py-2 text-sm">
            <span class="text-kernel-300">${t.name}</span>
            <div class="flex items-center gap-3">
              <span class="font-mono text-xs text-kernel-500">residual: ${t.residual.toExponential(2)}</span>
              ${x(t.passed)}
            </div>
          </div>
        `).join("")}
      </div>
      <div class="grid grid-cols-2 gap-3">
        <div class="bg-kernel-800 rounded p-3 text-center">
          <div class="text-xs text-kernel-500">Regime</div>
          <div class="font-bold ${p}">${i.regime}${i.isCritical?" + CRITICAL":""}</div>
        </div>
        <div class="bg-kernel-800 rounded p-3 text-center">
          <div class="text-xs text-kernel-500">Δ (Het. Gap)</div>
          <div class="font-mono text-kernel-200">${n.delta.toFixed(8)}</div>
        </div>
      </div>`}const c=document.getElementById("channel-input");c.addEventListener("input",o);document.querySelectorAll(".preset-btn").forEach(e=>e.addEventListener("click",()=>{c.value=e.dataset.vals,o()}));o();

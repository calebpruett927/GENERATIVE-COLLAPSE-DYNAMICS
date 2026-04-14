import{c as g,a as u,D as I,P as T,e as j,b as N,k as P,u as M,v as W}from"./kernel.1iDFzIM1.js";import{R as b,T as q}from"./constants.DtcEBpaC.js";function E(e){return e==="STABLE"?"text-green-400":e==="WATCH"?"text-amber-400":"text-red-400"}function G(e){return e?'<span class="text-green-400 font-bold">PASS</span>':'<span class="text-red-400 font-bold">FAIL</span>'}function K(e){const t=b,a=[{name:"ω < "+t.omega_stable_max,pass:e.omega<t.omega_stable_max,value:e.omega,target:t.omega_stable_max,unit:"drift",fix:e.omega>=t.omega_stable_max?`Reduce ω by ${(e.omega-t.omega_stable_max+.001).toFixed(4)} — raise the mean of all channels.`:""},{name:"F > "+t.F_stable_min,pass:e.F>t.F_stable_min,value:e.F,target:t.F_stable_min,unit:"fidelity",fix:e.F<=t.F_stable_min?`Increase F by ${(t.F_stable_min-e.F+.001).toFixed(4)} — raise the weakest channels.`:""},{name:"S < "+t.S_stable_max,pass:e.S<t.S_stable_max,value:e.S,target:t.S_stable_max,unit:"entropy",fix:e.S>=t.S_stable_max?`Reduce S by ${(e.S-t.S_stable_max+.001).toFixed(4)} — move channels away from 0.5.`:""},{name:"C < "+t.C_stable_max,pass:e.C<t.C_stable_max,value:e.C,target:t.C_stable_max,unit:"curvature",fix:e.C>=t.C_stable_max?`Reduce C by ${(e.C-t.C_stable_max+.001).toFixed(4)} — make channels more uniform.`:""}],l=e.IC<t.IC_critical_max,n=a.every(s=>s.pass);return`
      <div class="grid grid-cols-1 md:grid-cols-2 gap-2">
        ${a.map(s=>`
          <div class="bg-kernel-800 rounded p-2 flex justify-between items-center">
            <div>
              <span class="text-xs font-mono text-kernel-300">${s.name}</span>
              <span class="text-xs text-kernel-500 ml-1">current: <span class="font-mono">${s.value.toFixed(4)}</span></span>
            </div>
            <div>${G(s.pass)}</div>
          </div>
          ${s.pass?"":`<div class="text-xs text-amber-300 px-2 -mt-1 mb-1">↳ ${s.fix}</div>`}
        `).join("")}
      </div>
      ${l?`
        <div class="bg-red-900/30 border border-red-700 rounded p-2 mt-2 text-xs text-red-300">
          <strong>CRITICAL</strong>: IC = ${e.IC.toFixed(6)} &lt; 0.30 — multiplicative coherence dangerously low.
          At least one channel is near ε (guard band). Raise the weakest channel.
        </div>
      `:""}
      ${n&&!l?`
        <div class="bg-green-900/30 border border-green-700 rounded p-2 mt-2 text-xs text-green-300">
          All four Stable gates satisfied. System is CONFORMANT at Stable regime.
        </div>
      `:""}
    `}function V(e,t){if(e.length===0)return"";const a=g(e,t);let l=0;for(let d=1;d<e.length;d++)e[d]<e[l]&&(l=d);const n=[...e];n[l]=Math.min(1,n[l]+.1);const s=g(n,t),i=a.IC>0?s.IC/a.IC:1/0;return`
      <div class="bg-kernel-800 rounded p-3 text-xs">
        <div class="font-bold text-kernel-300 mb-2">Channel Sensitivity</div>
        <div class="grid grid-cols-2 md:grid-cols-4 gap-2 mb-2">
          ${e.map((d,x)=>`
            <div class="rounded px-2 py-1 ${x===l?"bg-red-900/40 border border-red-700":"bg-kernel-700"}">
              <span class="text-kernel-500">c[${x}]</span>
              <span class="font-mono text-kernel-200 ml-1">${d.toFixed(4)}</span>
              ${x===l?'<span class="text-red-400 ml-1">← weakest</span>':""}
            </div>
          `).join("")}
        </div>
        <div class="text-kernel-400">
          Raising c[${l}] from <span class="font-mono">${e[l].toFixed(4)}</span>
          → <span class="font-mono">${n[l].toFixed(4)}</span>:
          IC changes <span class="font-mono">${a.IC.toFixed(6)}</span>
          → <span class="font-mono ${s.IC>a.IC?"text-green-400":"text-kernel-200"}">${s.IC.toFixed(6)}</span>
          (${i!==1/0?i.toFixed(1)+"×":"∞"} improvement),
          Δ: <span class="font-mono">${a.delta.toFixed(4)}</span>
          → <span class="font-mono">${s.delta.toFixed(4)}</span>
        </div>
      </div>
    `}function y(e,t){return M(e.isHomogeneous?[e.F,e.F]:[e.F+.1,e.F-.1,e.F]),`
      <div class="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs">
        <div class="bg-kernel-800 rounded p-2 text-center">
          <div class="text-kernel-500">F (fidelity)</div>
          <div class="font-mono text-lg text-kernel-100">${e.F.toFixed(6)}</div>
        </div>
        <div class="bg-kernel-800 rounded p-2 text-center">
          <div class="text-kernel-500">ω (drift)</div>
          <div class="font-mono text-lg text-kernel-100">${e.omega.toFixed(6)}</div>
        </div>
        <div class="bg-kernel-800 rounded p-2 text-center">
          <div class="text-kernel-500">IC (integrity)</div>
          <div class="font-mono text-lg text-kernel-100">${e.IC.toFixed(6)}</div>
        </div>
        <div class="bg-kernel-800 rounded p-2 text-center">
          <div class="text-kernel-500">Δ (gap)</div>
          <div class="font-mono text-lg text-kernel-100">${e.delta.toFixed(6)}</div>
        </div>
        <div class="bg-kernel-800 rounded p-2 text-center">
          <div class="text-kernel-500">S (entropy)</div>
          <div class="font-mono text-kernel-200">${e.S.toFixed(6)}</div>
        </div>
        <div class="bg-kernel-800 rounded p-2 text-center">
          <div class="text-kernel-500">C (curvature)</div>
          <div class="font-mono text-kernel-200">${e.C.toFixed(6)}</div>
        </div>
        <div class="bg-kernel-800 rounded p-2 text-center">
          <div class="text-kernel-500">κ (log-integrity)</div>
          <div class="font-mono text-kernel-200">${e.kappa.toFixed(6)}</div>
        </div>
        <div class="bg-kernel-800 rounded p-2 text-center">
          <div class="text-kernel-500">Regime</div>
          <div class="font-mono text-lg ${E(t.regime)}">${t.regime}${t.isCritical?" ⚠":""}</div>
        </div>
      </div>
    `}const w=document.querySelectorAll(".pv-tab"),J=document.querySelectorAll(".pv-panel");function _(e){w.forEach(t=>{const a=t.dataset.tab===e;t.setAttribute("aria-selected",String(a)),t.classList.toggle("bg-amber-600",a),t.classList.toggle("text-white",a),t.classList.toggle("text-kernel-400",!a),t.classList.toggle("hover:text-kernel-200",!a)}),J.forEach(t=>{t.classList.toggle("hidden",t.id!==`pv-${e}`)})}w.forEach(e=>{e.addEventListener("click",()=>_(e.dataset.tab))});_("random");document.getElementById("prec-run").addEventListener("click",()=>{const e=parseInt(document.getElementById("prec-n").value),t=parseInt(document.getElementById("prec-trials").value),a=parseFloat(document.getElementById("prec-tol").value),l=document.getElementById("prec-results");l.innerHTML='<div class="bg-kernel-900 border border-kernel-600 rounded-lg p-4 text-center text-yellow-400 text-sm">Running...</div>',setTimeout(()=>{const n=performance.now();let s=0,i=0,d=0;const x={STABLE:0,WATCH:0,COLLAPSE:0};let c=0;const p=[];for(let r=0;r<t;r++){const v=Array.from({length:e},()=>Math.random()),F=Array.from({length:e},()=>1/e),m=g(v,F),k=u(m),L=Math.abs(m.F+m.omega-1),B=Math.abs(m.IC-Math.exp(m.kappa)),R=m.IC<=m.F+1e-15;s=Math.max(s,L),i=Math.max(i,B),R||d++,x[k.regime]++,k.isCritical&&c++,r<5&&p.push({c:v.slice(0,4),F:m.F,omega:m.omega,S:m.S,C:m.C,kappa:m.kappa,IC:m.IC,dualErr:L,expErr:B,icLeF:R,regime:k.regime+(k.isCritical?" (CRITICAL)":"")})}const A=performance.now()-n,$=d===0&&s<a,D=$?"text-green-400":"text-red-400",H=$?"CONFORMANT":"NONCONFORMANT";l.innerHTML=`
        <div class="bg-kernel-900 border ${$?"border-green-600":"border-red-600"} rounded-lg p-4">
          <div class="flex justify-between items-center">
            <div>
              <span class="text-2xl font-bold ${D}">${H}</span>
              <span class="text-kernel-500 text-sm ml-2">${t.toLocaleString()} trials × ${e} channels in ${A.toFixed(1)}ms</span>
            </div>
            <div class="text-right text-xs text-kernel-500">
              Rate: ${(t/A*1e3).toFixed(0)} evals/s · Tolerance: ${a.toExponential(0)}
            </div>
          </div>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-3 gap-3">
          <div class="bg-kernel-900 border border-kernel-700 rounded-lg p-4 text-center">
            <div class="text-xs text-kernel-500 mb-1">Duality: F + ω = 1</div>
            <div class="font-mono text-sm ${s<1e-14?"text-green-400":"text-yellow-400"}">
              max |residual| = ${s.toExponential(2)}
            </div>
            <div class="text-xs text-kernel-600 mt-1">${s<1e-14?"Exact to machine ε":"Within tolerance"}</div>
          </div>
          <div class="bg-kernel-900 border border-kernel-700 rounded-lg p-4 text-center">
            <div class="text-xs text-kernel-500 mb-1">Integrity Bound: IC ≤ F</div>
            <div class="font-mono text-sm ${d===0?"text-green-400":"text-red-400"}">
              violations: ${d} / ${t}
            </div>
            <div class="text-xs text-kernel-600 mt-1">${d===0?"Zero violations":"BOUND VIOLATED"}</div>
          </div>
          <div class="bg-kernel-900 border border-kernel-700 rounded-lg p-4 text-center">
            <div class="text-xs text-kernel-500 mb-1">Log-Integrity: IC = exp(κ)</div>
            <div class="font-mono text-sm ${i<1e-12?"text-green-400":"text-yellow-400"}">
              max |residual| = ${i.toExponential(2)}
            </div>
            <div class="text-xs text-kernel-600 mt-1">${i<1e-12?"Exact to machine ε":"Within tolerance"}</div>
          </div>
        </div>

        <div class="bg-kernel-900 border border-kernel-700 rounded-lg p-4">
          <h3 class="text-sm font-bold text-kernel-300 mb-3">Regime Distribution (${t.toLocaleString()} random traces)</h3>
          <div class="flex gap-1 h-6 rounded overflow-hidden mb-2">
            ${["STABLE","WATCH","COLLAPSE"].map(r=>{const v=x[r]/t*100,F={STABLE:"bg-green-600",WATCH:"bg-amber-600",COLLAPSE:"bg-red-700"};return v>0?`<div class="${F[r]}" style="width:${v}%" title="${r}: ${v.toFixed(1)}%"></div>`:""}).join("")}
          </div>
          <div class="flex justify-between text-xs text-kernel-500">
            <span class="text-green-400">Stable: ${(x.STABLE/t*100).toFixed(1)}%</span>
            <span class="text-amber-400">Watch: ${(x.WATCH/t*100).toFixed(1)}%</span>
            <span class="text-red-400">Collapse: ${(x.COLLAPSE/t*100).toFixed(1)}%</span>
            ${c>0?`<span class="text-purple-400">Critical: ${(c/t*100).toFixed(1)}%</span>`:""}
          </div>
        </div>

        <div class="bg-kernel-900 border border-kernel-700 rounded-lg p-4">
          <h3 class="text-sm font-bold text-kernel-300 mb-3">Sample Traces (first 5)</h3>
          <div class="overflow-x-auto">
            <table class="w-full text-xs font-mono">
              <thead><tr class="text-kernel-500 border-b border-kernel-700">
                <th class="text-left p-1">c[0..3]</th><th class="text-right p-1">F</th>
                <th class="text-right p-1">ω</th><th class="text-right p-1">S</th>
                <th class="text-right p-1">C</th><th class="text-right p-1">κ</th>
                <th class="text-right p-1">IC</th><th class="text-right p-1">|F+ω−1|</th>
                <th class="text-right p-1">IC≤F</th><th class="text-left p-1">Regime</th>
              </tr></thead>
              <tbody class="text-kernel-300">
                ${p.map(r=>`
                  <tr class="border-b border-kernel-800">
                    <td class="p-1 text-kernel-500">[${r.c.map(v=>v.toFixed(3)).join(", ")}…]</td>
                    <td class="text-right p-1">${r.F.toFixed(6)}</td>
                    <td class="text-right p-1">${r.omega.toFixed(6)}</td>
                    <td class="text-right p-1">${r.S.toFixed(6)}</td>
                    <td class="text-right p-1">${r.C.toFixed(6)}</td>
                    <td class="text-right p-1">${r.kappa.toFixed(6)}</td>
                    <td class="text-right p-1">${r.IC.toFixed(6)}</td>
                    <td class="text-right p-1 ${r.dualErr<1e-14?"text-green-500":"text-yellow-500"}">${r.dualErr.toExponential(1)}</td>
                    <td class="text-right p-1 ${r.icLeF?"text-green-500":"text-red-500"}">${r.icLeF?"✓":"✗"}</td>
                    <td class="p-1">${r.regime}</td>
                  </tr>
                `).join("")}
              </tbody>
            </table>
          </div>
        </div>
      `},50)});let o=[.85,.9,.8,.75,.95,.7,.88,.82];function h(){const e=parseInt(document.getElementById("manual-n").value);for(;o.length<e;)o.push(.5);o=o.slice(0,e);const t=document.getElementById("manual-sliders");t.innerHTML=o.map((a,l)=>`
      <div class="flex items-center gap-2">
        <label class="text-xs text-kernel-500 w-10 shrink-0">c[${l}]</label>
        <input type="range" min="0" max="1" step="0.001" value="${a}"
          class="manual-slider flex-1 accent-amber-500 h-2" data-idx="${l}" />
        <input type="number" min="0" max="1" step="0.01" value="${a.toFixed(3)}"
          class="manual-num w-20 bg-kernel-800 border border-kernel-600 rounded px-1 py-0.5 text-xs text-kernel-200 font-mono" data-idx="${l}" />
      </div>
    `).join(""),t.querySelectorAll(".manual-slider").forEach(a=>{a.addEventListener("input",()=>{const l=parseInt(a.dataset.idx);o[l]=parseFloat(a.value);const n=t.querySelector(`.manual-num[data-idx="${l}"]`);n&&(n.value=parseFloat(a.value).toFixed(3)),C()})}),t.querySelectorAll(".manual-num").forEach(a=>{a.addEventListener("change",()=>{const l=parseInt(a.dataset.idx),n=Math.max(0,Math.min(1,parseFloat(a.value)||0));o[l]=n,a.value=n.toFixed(3);const s=t.querySelector(`.manual-slider[data-idx="${l}"]`);s&&(s.value=String(n)),C()})}),C()}function C(){const e=o.length,t=Array(e).fill(1/e),a=g(o,t),l=u(a),n=W(a),s=M(o),i=document.getElementById("manual-results");i.innerHTML=`
      <div class="bg-kernel-900 border border-kernel-700 rounded-lg p-4">
        <div class="flex justify-between items-center mb-3">
          <h3 class="text-sm font-bold text-kernel-300">Kernel Output</h3>
          <span class="text-xs text-kernel-500">${s.description}</span>
        </div>
        ${y(a,l)}
      </div>
      <div class="bg-kernel-900 border border-kernel-700 rounded-lg p-4">
        <h3 class="text-sm font-bold text-kernel-300 mb-3">Regime Gates — Conformance Status</h3>
        ${K(a)}
      </div>
      <div class="bg-kernel-900 border border-kernel-700 rounded-lg p-4">
        <h3 class="text-sm font-bold text-kernel-300 mb-3">Identity Verification</h3>
        <div class="grid grid-cols-1 md:grid-cols-3 gap-2">
          ${n.map(d=>`
            <div class="bg-kernel-800 rounded p-2 text-center text-xs">
              <div class="text-kernel-500">${d.name}</div>
              <div class="font-mono ${d.pass?"text-green-400":"text-red-400"}">${d.formula}</div>
              <div class="font-mono text-kernel-400 mt-1">residual: ${d.residual.toExponential(2)}</div>
            </div>
          `).join("")}
        </div>
      </div>
      <div class="bg-kernel-900 border border-kernel-700 rounded-lg p-4">
        ${V(o,t)}
      </div>
    `}document.getElementById("manual-n").addEventListener("change",h);document.getElementById("manual-fill").addEventListener("change",e=>{const t=e.target.value;if(!t)return;const a=o.length;if(t==="random")o=Array.from({length:a},()=>Math.random());else{const l=parseFloat(t);o=Array(a).fill(l)}h(),e.target.value=""});document.getElementById("manual-reset").addEventListener("click",()=>{o=[.85,.9,.8,.75,.95,.7,.88,.82],document.getElementById("manual-n").value="8",h()});h();const U=new Set(I.map(e=>e.domain)),Z=document.getElementById("preset-domain");U.forEach(e=>{const t=document.createElement("option");t.value=e,t.textContent=e.replace(/_/g," "),Z.appendChild(t)});document.getElementById("preset-run").addEventListener("click",()=>{const e=document.getElementById("preset-domain").value,t=document.getElementById("preset-sort").value,a=document.getElementById("preset-results"),l=[];if(e==="all")for(const[,n]of Object.entries(T)){const s=g([...n.c],[...n.w]);l.push({name:n.name,domain:"structural",desc:"",r:s,regime:u(s),c:[...n.c]})}for(const n of I){if(e!=="all"&&n.domain!==e)continue;const s=g([...n.c],[...n.w]);l.push({name:n.name,domain:n.domain,desc:n.description,r:s,regime:u(s),c:[...n.c]})}t==="F"?l.sort((n,s)=>s.r.F-n.r.F):t==="IC"?l.sort((n,s)=>s.r.IC-n.r.IC):t==="delta"?l.sort((n,s)=>s.r.delta-n.r.delta):t==="omega"?l.sort((n,s)=>n.r.omega-s.r.omega):l.sort((n,s)=>n.name.localeCompare(s.name)),a.innerHTML=`
      <div class="bg-kernel-900 border border-kernel-700 rounded-lg p-4">
        <h3 class="text-sm font-bold text-kernel-300 mb-3">${l.length} Entities</h3>
        <div class="overflow-x-auto">
          <table class="w-full text-xs font-mono">
            <thead><tr class="text-kernel-500 border-b border-kernel-700">
              <th class="text-left p-1.5">Entity</th>
              <th class="text-left p-1.5">Domain</th>
              <th class="text-right p-1.5">F</th>
              <th class="text-right p-1.5">ω</th>
              <th class="text-right p-1.5">IC</th>
              <th class="text-right p-1.5">Δ</th>
              <th class="text-right p-1.5">S</th>
              <th class="text-right p-1.5">C</th>
              <th class="text-center p-1.5">Regime</th>
              <th class="text-center p-1.5">Stable?</th>
            </tr></thead>
            <tbody class="text-kernel-300">
              ${l.map(n=>{const s=n.r.omega<b.omega_stable_max&&n.r.F>b.F_stable_min&&n.r.S<b.S_stable_max&&n.r.C<b.C_stable_max;return`
                  <tr class="border-b border-kernel-800 hover:bg-kernel-800/50 cursor-pointer preset-row"
                      data-channels="${JSON.stringify(n.c)}" title="${n.desc}">
                    <td class="p-1.5 text-kernel-200">${n.name}</td>
                    <td class="p-1.5 text-kernel-500">${n.domain}</td>
                    <td class="text-right p-1.5">${n.r.F.toFixed(4)}</td>
                    <td class="text-right p-1.5">${n.r.omega.toFixed(4)}</td>
                    <td class="text-right p-1.5">${n.r.IC.toFixed(4)}</td>
                    <td class="text-right p-1.5">${n.r.delta.toFixed(4)}</td>
                    <td class="text-right p-1.5">${n.r.S.toFixed(4)}</td>
                    <td class="text-right p-1.5">${n.r.C.toFixed(4)}</td>
                    <td class="text-center p-1.5"><span class="${E(n.regime.regime)}">${n.regime.regime}${n.regime.isCritical?" ⚠":""}</span></td>
                    <td class="text-center p-1.5">${s?'<span class="text-green-400">✓</span>':'<span class="text-red-400">✗</span>'}</td>
                  </tr>
                `}).join("")}
            </tbody>
          </table>
        </div>
        <p class="text-xs text-kernel-600 mt-2">Click any row to load it into the Manual Trace editor.</p>
      </div>
      <div id="preset-detail" class="space-y-3"></div>
    `,a.querySelectorAll(".preset-row").forEach(n=>{n.addEventListener("click",()=>{const s=JSON.parse(n.dataset.channels);o=s,document.getElementById("manual-n").value=String(s.length),h(),_("manual")})})});document.getElementById("seam-run").addEventListener("click",()=>{const e=parseFloat(document.getElementById("seam-omega").value),t=parseFloat(document.getElementById("seam-C").value),a=parseFloat(document.getElementById("seam-R").value),l=parseFloat(document.getElementById("seam-tauR").value),n=document.getElementById("seam-results"),s=j(e,t,a,l),i=N(e,t),d=s.pass?"border-green-600":"border-red-600",x=s.pass?'<span class="text-green-400 font-bold text-lg">SEAM PASS</span>':'<span class="text-red-400 font-bold text-lg">SEAM FAIL</span>';n.innerHTML=`
      <div class="bg-kernel-900 border ${d} rounded-lg p-4">
        <div class="flex justify-between items-center mb-3">
          ${x}
          <span class="text-xs text-kernel-500">|residual| ≤ tol_seam (${q})</span>
        </div>
        <div class="grid grid-cols-2 md:grid-cols-3 gap-3">
          <div class="bg-kernel-800 rounded p-3 text-center">
            <div class="text-xs text-kernel-500">Γ(ω) — drift cost</div>
            <div class="font-mono text-lg text-red-300">${s.gamma.toFixed(6)}</div>
            <div class="text-xs text-kernel-600">ω³/(1−ω+ε)</div>
          </div>
          <div class="bg-kernel-800 rounded p-3 text-center">
            <div class="text-xs text-kernel-500">D_C — curvature cost</div>
            <div class="font-mono text-lg text-amber-300">${s.D_C.toFixed(6)}</div>
            <div class="text-xs text-kernel-600">α · C</div>
          </div>
          <div class="bg-kernel-800 rounded p-3 text-center">
            <div class="text-xs text-kernel-500">R · τ_R — return credit</div>
            <div class="font-mono text-lg text-green-300">${s.credit.toFixed(6)}</div>
          </div>
          <div class="bg-kernel-800 rounded p-3 text-center">
            <div class="text-xs text-kernel-500">Δκ — budget</div>
            <div class="font-mono text-lg text-kernel-100">${s.deltaKappa.toFixed(6)}</div>
            <div class="text-xs text-kernel-600">credit − (D_ω + D_C)</div>
          </div>
          <div class="bg-kernel-800 rounded p-3 text-center">
            <div class="text-xs text-kernel-500">Seam residual</div>
            <div class="font-mono text-lg ${s.pass?"text-green-400":"text-red-400"}">${s.residual.toFixed(6)}</div>
          </div>
          <div class="bg-kernel-800 rounded p-3 text-center">
            <div class="text-xs text-kernel-500">τ_R* diagnostic</div>
            <div class="font-mono text-lg text-kernel-100">${i.toFixed(4)}</div>
            <div class="text-xs text-kernel-600">higher = better return</div>
          </div>
        </div>
      </div>
      ${s.pass?"":`
        <div class="bg-kernel-900 border border-amber-700 rounded-lg p-4">
          <h3 class="text-sm font-bold text-amber-300 mb-2">How to Close the Seam</h3>
          <ul class="text-xs text-kernel-400 space-y-1 list-disc list-inside">
            <li>Total debit: D_ω + D_C = ${(s.D_omega+s.D_C).toFixed(6)}</li>
            <li>Current credit: R · τ_R = ${s.credit.toFixed(6)}</li>
            <li>Shortfall: ${Math.abs(s.residual).toFixed(6)}</li>
            <li><strong>Option 1:</strong> Increase R (return rate) to at least ${((s.D_omega+s.D_C)/(l||1)).toFixed(4)}</li>
            <li><strong>Option 2:</strong> Increase τ_R (return time) to at least ${((s.D_omega+s.D_C)/(a||1)).toFixed(4)}</li>
            <li><strong>Option 3:</strong> Reduce ω (drift) — currently Γ(${e.toFixed(3)}) = ${s.gamma.toFixed(6)}</li>
            <li><strong>Option 4:</strong> Reduce C (curvature) — currently D_C = ${s.D_C.toFixed(6)}</li>
          </ul>
        </div>
      `}
    `});const f=[...Object.entries(T).map(([e,t])=>({label:`[structural] ${t.name}`,c:[...t.c],w:[...t.w]})),...I.map(e=>({label:`[${e.domain}] ${e.name}`,c:[...e.c],w:[...e.w]}))],O=document.getElementById("comp-a"),S=document.getElementById("comp-b");f.forEach((e,t)=>{const a=document.createElement("option");a.value=String(t),a.textContent=e.label,O.appendChild(a);const l=document.createElement("option");l.value=String(t),l.textContent=e.label,S.appendChild(l)});f.length>1&&(S.value="1");document.getElementById("comp-run").addEventListener("click",()=>{const e=parseInt(O.value),t=parseInt(S.value),a=f[e],l=f[t],n=document.getElementById("comp-results"),s=g(a.c,a.w),i=g(l.c,l.w),d=u(s),x=u(i),c=P(s,i),p=Math.abs(c.delta_composed-c.gap_predicted);n.innerHTML=`
      <div class="bg-kernel-900 border border-kernel-700 rounded-lg p-4">
        <h3 class="text-sm font-bold text-kernel-300 mb-3">Input Traces</h3>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <div class="text-xs text-kernel-400 mb-2 font-bold">A: ${a.label}</div>
            ${y(s,d)}
          </div>
          <div>
            <div class="text-xs text-kernel-400 mb-2 font-bold">B: ${l.label}</div>
            ${y(i,x)}
          </div>
        </div>
      </div>

      <div class="bg-kernel-900 border border-amber-700 rounded-lg p-4">
        <h3 class="text-sm font-bold text-kernel-300 mb-3">Composed Result (A ∘ B)</h3>
        <div class="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs">
          <div class="bg-kernel-800 rounded p-2 text-center">
            <div class="text-kernel-500">F₁₂ = (F₁+F₂)/2</div>
            <div class="font-mono text-lg text-kernel-100">${c.F_composed.toFixed(6)}</div>
            <div class="text-kernel-600">arithmetic</div>
          </div>
          <div class="bg-kernel-800 rounded p-2 text-center">
            <div class="text-kernel-500">IC₁₂ = √(IC₁·IC₂)</div>
            <div class="font-mono text-lg text-kernel-100">${c.IC_composed.toFixed(6)}</div>
            <div class="text-kernel-600">geometric</div>
          </div>
          <div class="bg-kernel-800 rounded p-2 text-center">
            <div class="text-kernel-500">Δ₁₂ (actual)</div>
            <div class="font-mono text-lg text-kernel-100">${c.delta_composed.toFixed(6)}</div>
          </div>
          <div class="bg-kernel-800 rounded p-2 text-center">
            <div class="text-kernel-500">Δ₁₂ (predicted)</div>
            <div class="font-mono text-lg ${p<1e-10?"text-green-400":"text-amber-400"}">${c.gap_predicted.toFixed(6)}</div>
            <div class="text-kernel-600">Hellinger correction: ${c.hellinger_correction.toFixed(6)}</div>
          </div>
          <div class="bg-kernel-800 rounded p-2 text-center">
            <div class="text-kernel-500">ω₁₂</div>
            <div class="font-mono text-kernel-200">${c.omega_composed.toFixed(6)}</div>
          </div>
          <div class="bg-kernel-800 rounded p-2 text-center">
            <div class="text-kernel-500">Regime</div>
            <div class="font-mono ${E(c.regime)}">${c.regime}${c.isCritical?" ⚠":""}</div>
          </div>
          <div class="bg-kernel-800 rounded p-2 text-center col-span-2">
            <div class="text-kernel-500">Gap composition law error</div>
            <div class="font-mono ${p<1e-10?"text-green-400":"text-amber-400"}">${p.toExponential(2)}</div>
            <div class="text-kernel-600">${p<1e-10?"Verified to machine precision":"Approximate (expected for non-identical)"}</div>
          </div>
        </div>
      </div>
    `});

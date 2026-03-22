import{P as y,c as f,a as k,v as h,b as $}from"./kernel.sDW4GxI3.js";import{R as E,K as I}from"./constants.wjURl7WN.js";let r=8,i=[.95,.92,.97,.93,.96,.94,.91,.98],o=Array(8).fill(.125);function C(){L(),m(),w()}function L(){const t=document.getElementById("preset-buttons");t.innerHTML="";for(const[n,e]of Object.entries(y)){const s=document.createElement("button");s.className="text-xs px-3 py-1.5 bg-kernel-800 text-kernel-300 rounded hover:bg-kernel-700 hover:text-kernel-100 transition",s.textContent=e.name,s.addEventListener("click",()=>{i=[...e.c],o=[...e.w],r=i.length,m(),g()}),t.appendChild(s)}}function m(){const t=document.getElementById("channel-inputs"),n=document.getElementById("weight-inputs");t.innerHTML="",n.innerHTML="";for(let e=0;e<r;e++){const s=document.createElement("div");s.className="flex items-center gap-2",s.innerHTML=`
        <span class="text-xs text-kernel-500 w-8">c${e+1}</span>
        <input type="range" min="0" max="1" step="0.001" value="${i[e]??.5}"
          class="flex-1 h-1.5 accent-amber-500" data-ch="${e}" />
        <input type="number" min="0" max="1" step="0.01" value="${(i[e]??.5).toFixed(3)}"
          class="w-20 px-2 py-1 bg-kernel-800 border border-kernel-600 rounded text-kernel-200 text-xs text-right" data-ch-num="${e}" />
      `,t.appendChild(s);const a=document.createElement("div");a.className="flex items-center gap-2",a.innerHTML=`
        <span class="text-xs text-kernel-500 w-8">w${e+1}</span>
        <input type="range" min="0" max="1" step="0.001" value="${o[e]??1/r}"
          class="flex-1 h-1.5 accent-blue-400" data-w="${e}" />
        <input type="number" min="0" max="1" step="0.01" value="${(o[e]??1/r).toFixed(4)}"
          class="w-20 px-2 py-1 bg-kernel-800 border border-kernel-600 rounded text-kernel-200 text-xs text-right" data-w-num="${e}" />
      `,n.appendChild(a)}x()}function w(){document.getElementById("channel-inputs").addEventListener("input",t=>{const n=t.target,e=n.dataset.ch??n.dataset.chNum;if(e!==void 0){i[parseInt(e)]=parseFloat(n.value);const s=n.closest("div"),a=s.querySelector(`[data-ch="${e}"]`),l=s.querySelector(`[data-ch-num="${e}"]`);a&&n!==a&&(a.value=n.value),l&&n!==l&&(l.value=parseFloat(n.value).toFixed(3))}}),document.getElementById("weight-inputs").addEventListener("input",t=>{const n=t.target,e=n.dataset.w??n.dataset.wNum;if(e!==void 0){o[parseInt(e)]=parseFloat(n.value);const s=n.closest("div"),a=s.querySelector(`[data-w="${e}"]`),l=s.querySelector(`[data-w-num="${e}"]`);a&&n!==a&&(a.value=n.value),l&&n!==l&&(l.value=parseFloat(n.value).toFixed(4)),x()}}),document.getElementById("btn-add-ch").addEventListener("click",()=>{r++,i.push(.5),o.push(0),m()}),document.getElementById("btn-rm-ch").addEventListener("click",()=>{r>2&&(r--,i.pop(),o.pop(),m())}),document.getElementById("btn-uniform").addEventListener("click",()=>{o=Array(r).fill(1/r),m()}),document.getElementById("btn-compute").addEventListener("click",g)}function x(){const t=o.reduce((e,s)=>e+s,0),n=document.getElementById("weight-sum");n.textContent=`Σw = ${t.toFixed(4)}`,n.className=Math.abs(t-1)<.01?"text-xs text-green-400 mt-2":"text-xs text-red-400 mt-2"}function g(){const t=f(i,o),{regime:n,isCritical:e}=k(t),s=h(t);document.getElementById("results-panel").classList.remove("hidden");const a=document.getElementById("regime-badge"),l=E[e?"CRITICAL":n];a.style.backgroundColor=l.bg,a.style.color=l.text,a.style.borderColor=l.border,a.className="text-center py-3 rounded-lg font-bold text-lg border-2",a.textContent=e?`${n} + CRITICAL`:n;const p=document.getElementById("invariants-grid"),c=I,v=[{key:"F",value:t.F,sym:c.F},{key:"omega",value:t.omega,sym:c.omega},{key:"S",value:t.S,sym:c.S},{key:"C",value:t.C,sym:c.C},{key:"kappa",value:t.kappa,sym:c.kappa},{key:"IC",value:t.IC,sym:c.IC}];p.innerHTML=v.map(d=>`
      <div class="bg-kernel-800 rounded-lg p-3 border border-kernel-700">
        <div class="text-xs text-kernel-500">${d.sym.name} (${d.sym.symbol})</div>
        <div class="text-lg font-mono font-bold text-kernel-100">${d.value.toFixed(6)}</div>
        <div class="text-xs text-kernel-600 mt-1">${d.sym.latin}</div>
      </div>
    `).join(""),p.innerHTML+=`
      <div class="bg-kernel-800 rounded-lg p-3 border border-kernel-700 col-span-2 md:col-span-3">
        <div class="flex justify-between items-center">
          <span class="text-xs text-kernel-500">Heterogeneity Gap (Δ = F − IC)</span>
          <span class="text-lg font-mono font-bold ${t.delta>.1?"text-amber-400":"text-kernel-100"}">${t.delta.toFixed(6)}</span>
        </div>
        <div class="w-full bg-kernel-900 rounded-full h-2 mt-2">
          <div class="h-2 rounded-full transition-all ${t.delta>.3?"bg-red-500":t.delta>.1?"bg-amber-500":"bg-green-500"}"
            style="width: ${Math.min(t.delta*100,100)}%"></div>
        </div>
      </div>
    `;const b=document.getElementById("identity-checks");b.innerHTML=s.map(d=>`
      <div class="flex items-center justify-between text-sm">
        <span class="text-kernel-300">
          <span class="${d.pass?"text-green-400":"text-red-400"}">${d.pass?"✓":"✗"}</span>
          ${d.name}: ${d.formula}
        </span>
        <span class="font-mono text-xs ${d.pass?"text-green-400":"text-red-400"}">
          |residual| = ${d.residual.toExponential(2)}
        </span>
      </div>
    `).join(""),u(t),document.getElementById("seam-R").addEventListener("input",()=>u(t)),document.getElementById("seam-tauR").addEventListener("input",()=>u(t))}function u(t){const n=parseFloat(document.getElementById("seam-R").value),e=parseFloat(document.getElementById("seam-tauR").value),s=$(t.omega,t.C,n,e),a=document.getElementById("seam-results");a.innerHTML=`
      <div class="grid grid-cols-2 gap-2 text-xs">
        <div class="bg-kernel-800 rounded p-2">
          <span class="text-kernel-500">Γ(ω)</span>
          <span class="float-right font-mono text-kernel-200">${s.gamma.toFixed(6)}</span>
        </div>
        <div class="bg-kernel-800 rounded p-2">
          <span class="text-kernel-500">D_ω (drift debit)</span>
          <span class="float-right font-mono text-kernel-200">${s.D_omega.toFixed(6)}</span>
        </div>
        <div class="bg-kernel-800 rounded p-2">
          <span class="text-kernel-500">D_C (curvature debit)</span>
          <span class="float-right font-mono text-kernel-200">${s.D_C.toFixed(6)}</span>
        </div>
        <div class="bg-kernel-800 rounded p-2">
          <span class="text-kernel-500">Δκ (budget)</span>
          <span class="float-right font-mono text-kernel-200">${s.deltaKappa.toFixed(6)}</span>
        </div>
      </div>
      <div class="mt-2 p-2 rounded text-center font-bold text-sm ${s.pass?"bg-green-900/50 text-green-400 border border-green-700":"bg-red-900/50 text-red-400 border border-red-700"}">
        Seam ${s.pass?"PASS":"FAIL"} — |s| = ${Math.abs(s.residual).toFixed(6)} ${s.pass?"≤":">"} tol=${.005}
      </div>
    `}C();

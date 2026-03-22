import{c as f,v as F,a as E}from"./kernel.sDW4GxI3.js";import"./constants.wjURl7WN.js";const y=document.getElementById("prec-run");y.addEventListener("click",()=>{const d=parseInt(document.getElementById("prec-n").value),r=parseInt(document.getElementById("prec-trials").value),p=document.getElementById("prec-results");p.innerHTML='<div class="bg-kernel-900 border border-kernel-600 rounded-lg p-4 text-center text-yellow-400 text-sm">Running...</div>',setTimeout(()=>{const $=performance.now();let l=0,a=0,i=0,m=0;const n={STABLE:0,WATCH:0,COLLAPSE:0};let c=0;const g=[];for(let t=0;t<r;t++){const s=Array.from({length:d},()=>Math.random()),x=Array.from({length:d},()=>1/d),e=f(s,x);F(e);const o=E(e),v=Math.abs(e.F+e.omega-1),b=Math.abs(e.IC-Math.exp(e.kappa)),u=e.IC<=e.F+1e-15;l=Math.max(l,v),a=Math.max(a,b),u||i++,m++,n[o.regime]++,o.isCritical&&c++,t<5&&g.push({c:s.slice(0,4),F:e.F,omega:e.omega,S:e.S,C:e.C,kappa:e.kappa,IC:e.IC,dualErr:v,expErr:b,icLeF:u,regime:o.regime+(o.isCritical?" (CRITICAL)":"")})}const h=performance.now()-$,k=i===0&&l<1e-10?"text-green-400":"text-red-400",C=i===0&&l<1e-10?"CONFORMANT":"NONCONFORMANT";p.innerHTML=`
        <!-- Verdict banner -->
        <div class="bg-kernel-900 border ${i===0?"border-green-600":"border-red-600"} rounded-lg p-4">
          <div class="flex justify-between items-center">
            <div>
              <span class="text-2xl font-bold ${k}">${C}</span>
              <span class="text-kernel-500 text-sm ml-2">${r.toLocaleString()} trials × ${d} channels in ${h.toFixed(1)}ms</span>
            </div>
            <div class="text-right text-xs text-kernel-500">
              Rate: ${(r/h*1e3).toFixed(0)} evals/s
            </div>
          </div>
        </div>

        <!-- Identity verification summary -->
        <div class="grid grid-cols-1 md:grid-cols-3 gap-3">
          <div class="bg-kernel-900 border border-kernel-700 rounded-lg p-4 text-center">
            <div class="text-xs text-kernel-500 mb-1">Duality: F + ω = 1</div>
            <div class="font-mono text-sm ${l<1e-14?"text-green-400":"text-yellow-400"}">
              max |residual| = ${l.toExponential(2)}
            </div>
            <div class="text-xs text-kernel-600 mt-1">${l<1e-14?"Exact to machine ε":"Within tolerance"}</div>
          </div>

          <div class="bg-kernel-900 border border-kernel-700 rounded-lg p-4 text-center">
            <div class="text-xs text-kernel-500 mb-1">Integrity Bound: IC ≤ F</div>
            <div class="font-mono text-sm ${i===0?"text-green-400":"text-red-400"}">
              violations: ${i} / ${m}
            </div>
            <div class="text-xs text-kernel-600 mt-1">${i===0?"Zero violations":"BOUND VIOLATED"}</div>
          </div>

          <div class="bg-kernel-900 border border-kernel-700 rounded-lg p-4 text-center">
            <div class="text-xs text-kernel-500 mb-1">Log-Integrity: IC = exp(κ)</div>
            <div class="font-mono text-sm ${a<1e-12?"text-green-400":"text-yellow-400"}">
              max |residual| = ${a.toExponential(2)}
            </div>
            <div class="text-xs text-kernel-600 mt-1">${a<1e-12?"Exact to machine ε":"Within tolerance"}</div>
          </div>
        </div>

        <!-- Regime distribution -->
        <div class="bg-kernel-900 border border-kernel-700 rounded-lg p-4">
          <h3 class="text-sm font-bold text-kernel-300 mb-3">Regime Distribution (${r.toLocaleString()} random traces)</h3>
          <div class="flex gap-1 h-6 rounded overflow-hidden mb-2">
            ${["STABLE","WATCH","COLLAPSE"].map(t=>{const s=n[t]/r*100,x={STABLE:"bg-green-600",WATCH:"bg-amber-600",COLLAPSE:"bg-red-700"};return s>0?`<div class="${x[t]}" style="width:${s}%" title="${t}: ${s.toFixed(1)}%"></div>`:""}).join("")}
          </div>
          <div class="flex justify-between text-xs text-kernel-500">
            <span class="text-green-400">Stable: ${(n.STABLE/r*100).toFixed(1)}%</span>
            <span class="text-amber-400">Watch: ${(n.WATCH/r*100).toFixed(1)}%</span>
            <span class="text-red-400">Collapse: ${(n.COLLAPSE/r*100).toFixed(1)}%</span>
            ${c>0?`<span class="text-purple-400">Critical: ${(c/r*100).toFixed(1)}%</span>`:""}
          </div>
        </div>

        <!-- Sample traces -->
        <div class="bg-kernel-900 border border-kernel-700 rounded-lg p-4">
          <h3 class="text-sm font-bold text-kernel-300 mb-3">Sample Traces (first 5)</h3>
          <div class="overflow-x-auto">
            <table class="w-full text-xs font-mono">
              <thead>
                <tr class="text-kernel-500 border-b border-kernel-700">
                  <th class="text-left p-1">c[0..3]</th>
                  <th class="text-right p-1">F</th>
                  <th class="text-right p-1">ω</th>
                  <th class="text-right p-1">S</th>
                  <th class="text-right p-1">C</th>
                  <th class="text-right p-1">κ</th>
                  <th class="text-right p-1">IC</th>
                  <th class="text-right p-1">|F+ω−1|</th>
                  <th class="text-right p-1">IC≤F</th>
                  <th class="text-left p-1">Regime</th>
                </tr>
              </thead>
              <tbody class="text-kernel-300">
                ${g.map(t=>`
                  <tr class="border-b border-kernel-800">
                    <td class="p-1 text-kernel-500">[${t.c.map(s=>s.toFixed(3)).join(", ")}…]</td>
                    <td class="text-right p-1">${t.F.toFixed(6)}</td>
                    <td class="text-right p-1">${t.omega.toFixed(6)}</td>
                    <td class="text-right p-1">${t.S.toFixed(6)}</td>
                    <td class="text-right p-1">${t.C.toFixed(6)}</td>
                    <td class="text-right p-1">${t.kappa.toFixed(6)}</td>
                    <td class="text-right p-1">${t.IC.toFixed(6)}</td>
                    <td class="text-right p-1 ${t.dualErr<1e-14?"text-green-500":"text-yellow-500"}">${t.dualErr.toExponential(1)}</td>
                    <td class="text-right p-1 ${t.icLeF?"text-green-500":"text-red-500"}">${t.icLeF?"✓":"✗"}</td>
                    <td class="p-1">${t.regime}</td>
                  </tr>
                `).join("")}
              </tbody>
            </table>
          </div>
        </div>
      `},50)});

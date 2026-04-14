import{c as m,a as x}from"./kernel.1iDFzIM1.js";import{E as h}from"./constants.DtcEBpaC.js";const I=document.getElementById("id-verify-btn"),v=document.getElementById("id-channels"),f=document.getElementById("id-results");I.addEventListener("click",()=>{const n=v.value.split(",").map(c=>parseFloat(c.trim())).filter(c=>!isNaN(c));if(n.length<2)return;const e=m(n),s=Math.abs(e.F+e.omega-1),t=document.getElementById("id-r1");t.innerHTML=`<span class="${s<1e-14?"text-green-400":"text-red-400"}">${s<1e-14?"✓":"✗"}</span>
      <span class="text-kernel-300">F + ω = ${(e.F+e.omega).toFixed(15)}</span>
      <span class="text-kernel-500">residual: ${s.toExponential(2)}</span>`;const a=e.IC<=e.F+1e-15,l=document.getElementById("id-r2");l.innerHTML=`<span class="${a?"text-green-400":"text-red-400"}">${a?"✓":"✗"}</span>
      <span class="text-kernel-300">IC = ${e.IC.toFixed(8)} ≤ F = ${e.F.toFixed(8)}</span>
      <span class="text-kernel-500">Δ = ${e.delta.toFixed(8)}</span>`;const i=Math.exp(e.kappa),d=Math.abs(e.IC-i),r=document.getElementById("id-r3");r.innerHTML=`<span class="${d<1e-14?"text-green-400":"text-red-400"}">${d<1e-14?"✓":"✗"}</span>
      <span class="text-kernel-300">IC = ${e.IC.toFixed(10)} vs exp(κ) = ${i.toFixed(10)}</span>
      <span class="text-kernel-500">residual: ${d.toExponential(2)}</span>`;const o=x(e);document.getElementById("id-kernel-out").textContent=`F=${e.F.toFixed(4)} ω=${e.omega.toFixed(4)} S=${e.S.toFixed(4)} C=${e.C.toFixed(4)} κ=${e.kappa.toFixed(4)} IC=${e.IC.toFixed(4)} | ${o.regime}${o.isCritical?"+CRITICAL":""}`,f.classList.remove("hidden")});v.addEventListener("keydown",n=>{n.key==="Enter"&&I.click()});const C=document.getElementById("sla-perfect"),E=document.getElementById("sla-dead"),k=document.getElementById("sla-out");function u(){const n=parseInt(C.value),e=parseInt(E.value)/1e3;document.getElementById("sla-perfect-val").textContent=n.toString(),document.getElementById("sla-dead-val").textContent=e.toFixed(3);const s=Array(n).fill(.999);s.push(Math.max(e,h));const t=m(s),a=x(t),l=t.F>0?t.IC/t.F:0;k.innerHTML=`
      <div class="text-kernel-400">Channels: ${n} × 0.999 + 1 × ${e.toFixed(3)}</div>
      <div><span class="text-green-400">F = ${t.F.toFixed(4)}</span> (arithmetic: high — ${n} good channels keep F up)</div>
      <div><span class="text-red-400">IC = ${t.IC.toFixed(6)}</span> (geometric: destroyed by the dead channel)</div>
      <div>Δ = ${t.delta.toFixed(4)} | IC/F = ${l.toFixed(4)} | C = ${t.C.toFixed(4)}</div>
      <div>Regime: <span class="${a.regime==="STABLE"?"text-green-400":a.regime==="WATCH"?"text-yellow-400":"text-red-400"}">${a.regime}</span>${a.isCritical?' <span class="text-purple-400">+CRITICAL</span>':""}</div>
    `}C.addEventListener("input",u);E.addEventListener("input",u);u();const B=document.getElementById("rp-run-btn");B.addEventListener("click",()=>{const n=parseInt(document.getElementById("rp-n").value),e=parseInt(document.getElementById("rp-samples").value),s=document.getElementById("rp-results");s.classList.remove("hidden"),s.innerHTML='<div class="text-kernel-400">Scanning...</div>',requestAnimationFrame(()=>{let t=0,a=0,l=0,i=0;for(let g=0;g<e;g++){const F=[];for(let $=0;$<n;$++)F.push(Math.random());const y=m(F),p=x(y);p.regime==="STABLE"?t++:p.regime==="WATCH"?a++:l++,p.isCritical&&i++}const d=(100*t/e).toFixed(1),r=(100*a/e).toFixed(1),o=(100*l/e).toFixed(1),c=(100*i/e).toFixed(1);s.innerHTML=`
        <div class="text-kernel-400">Scanned ${e.toLocaleString()} random ${n}-channel traces:</div>
        <div class="flex gap-6 mt-1">
          <span class="text-green-400">Stable: ${d}%</span>
          <span class="text-yellow-400">Watch: ${r}%</span>
          <span class="text-red-400">Collapse: ${o}%</span>
          <span class="text-purple-400">Critical overlay: ${c}%</span>
        </div>
        <div class="text-kernel-500 text-xs mt-1">Expected: Stable ~12.5% | Watch ~24.4% | Collapse ~63.1%</div>
      `})});

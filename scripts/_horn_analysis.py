"""Horn analysis: map all 34 organisms onto the unified geometry budget surface."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

_WS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_WS / "src"))
sys.path.insert(0, str(_WS / "closures"))

from awareness_cognition.awareness_kernel import ORGANISM_CATALOG

from umcp.frozen_contract import ALPHA, EPSILON, P_EXPONENT


def kernel(trace):
    c = np.array(trace, dtype=np.float64)
    w = np.ones(len(c)) / len(c)
    c_eps = np.clip(c, EPSILON, 1 - EPSILON)
    F = float(np.dot(w, c_eps))
    omega = 1.0 - F
    kappa = float(np.dot(w, np.log(c_eps)))
    IC = float(np.exp(kappa))
    S = float(-np.dot(w, c_eps * np.log(c_eps) + (1 - c_eps) * np.log(1 - c_eps)))
    C = float(np.std(c_eps) / 0.5)
    return F, omega, kappa, IC, S, C


def gamma_cost(omega):
    return omega**P_EXPONENT / (1 - omega + EPSILON)


def horn_slope(omega):
    h = 1e-8
    return (gamma_cost(omega + h) - gamma_cost(omega - h)) / (2 * h)


def christoffel_11(omega):
    h = 1e-8
    dG = (gamma_cost(omega + h) - gamma_cost(omega - h)) / (2 * h)
    d2G = (gamma_cost(omega + h) - 2 * gamma_cost(omega) + gamma_cost(omega - h)) / h**2
    g11 = 1 + dG**2
    return dG * d2G / g11


# ── Compute all organisms ──
results = []
for org in ORGANISM_CATALOG:
    trace = org.channels
    F, omega, kappa, IC, S, C = kernel(trace)
    Aw = org.awareness_mean
    Ap = org.aptitude_mean
    G = gamma_cost(omega)
    chris = christoffel_11(omega)
    z = G + ALPHA * C
    delta = F - IC

    gates = []
    if omega >= 0.038:
        gates.append(("omega", omega - 0.038))
    if F <= 0.90:
        gates.append(("F", 0.90 - F))
    if S >= 0.15:
        gates.append(("S", S - 0.15))
    if C >= 0.14:
        gates.append(("C", C - 0.14))
    bind = max(gates, key=lambda x: x[1])[0] if gates else "STABLE"

    results.append(
        {
            "name": org.name,
            "omega": omega,
            "F": F,
            "IC": IC,
            "delta": delta,
            "C": C,
            "S": S,
            "G": G,
            "chris": chris,
            "z": z,
            "kappa": kappa,
            "bind": bind,
            "Aw": Aw,
            "Ap": Ap,
            "trace": trace,
        }
    )

results.sort(key=lambda r: r["Aw"])

print("=" * 130)
print("PART 1: ALL 34 ORGANISMS ON THE BUDGET HORN  (sorted by Aw)")
print("=" * 130)
hdr = f"{'Organism':<25} {'Aw':>5} {'Ap':>5} {'omega':>6} {'F':>6} {'IC':>6} {'Delta':>6} {'C':>6} {'Gamma':>9} {'chr_11':>9} {'z':>9} {'Bind':>6}"
print(hdr)
print("-" * 130)
for r in results:
    print(
        f"{r['name']:<25} {r['Aw']:5.3f} {r['Ap']:5.3f} {r['omega']:6.3f} {r['F']:6.3f} {r['IC']:6.3f} {r['delta']:6.3f} {r['C']:6.3f} {r['G']:9.6f} {r['chris']:9.4f} {r['z']:9.4f} {r['bind']:>6}"
    )

# ── Part 2: Binding gate transition ──
print()
print("=" * 100)
print("PART 2: THE BINDING GATE TRANSITION")
print("=" * 100)
omega_bind = [r for r in results if r["bind"] == "omega"]
C_bind = [r for r in results if r["bind"] == "C"]
S_bind = [r for r in results if r["bind"] == "S"]
F_bind = [r for r in results if r["bind"] == "F"]

print(
    f"  omega-bound: {len(omega_bind)}  (Aw range: [{min(r['Aw'] for r in omega_bind):.3f}, {max(r['Aw'] for r in omega_bind):.3f}])"
)
if C_bind:
    print(
        f"  C-bound:     {len(C_bind)}  (Aw range: [{min(r['Aw'] for r in C_bind):.3f}, {max(r['Aw'] for r in C_bind):.3f}])"
    )
if S_bind:
    print(f"  S-bound:     {len(S_bind)}")
if F_bind:
    print(f"  F-bound:     {len(F_bind)}")

print()
print("  C-binding organisms (the awareness elite):")
for r in sorted(C_bind, key=lambda x: x["Aw"]):
    excess_C = r["C"] - 0.14
    excess_omega = r["omega"] - 0.038
    print(
        f"    {r['name']:<25} Aw={r['Aw']:.3f}  C={r['C']:.3f} (excess={excess_C:.3f})  omega={r['omega']:.3f} (excess={excess_omega:.3f})"
    )

# ── Part 3: Correlations ──
print()
print("=" * 100)
print("PART 3: THE AWARENESS-CURVATURE COUPLING")
print("=" * 100)
aws = np.array([r["Aw"] for r in results])
cs = np.array([r["C"] for r in results])
omegas = np.array([r["omega"] for r in results])
deltas = np.array([r["delta"] for r in results])
ics = np.array([r["IC"] for r in results])
gs = np.array([r["G"] for r in results])

rho_aw_c, p_aw_c = spearmanr(aws, cs)
rho_aw_omega, p_aw_omega = spearmanr(aws, omegas)
rho_aw_delta, p_aw_delta = spearmanr(aws, deltas)
rho_aw_ic, p_aw_ic = spearmanr(aws, ics)
rho_c_omega, p_c_omega = spearmanr(cs, omegas)
rho_aw_G, p_aw_G = spearmanr(aws, gs)

# Cast p-values to float for type checker (spearmanr returns namedtuple)
p_aw_c = float(p_aw_c)
p_aw_omega = float(p_aw_omega)
p_aw_delta = float(p_aw_delta)
p_aw_ic = float(p_aw_ic)
p_c_omega = float(p_c_omega)
p_aw_G = float(p_aw_G)

print(f"  Spearman(Aw, C):      rho = {rho_aw_c:+.4f}  (p = {p_aw_c:.2e})  {'**SIG**' if p_aw_c < 0.01 else ''}")
print(
    f"  Spearman(Aw, omega):  rho = {rho_aw_omega:+.4f}  (p = {p_aw_omega:.2e})  {'**SIG**' if p_aw_omega < 0.01 else ''}"
)
print(
    f"  Spearman(Aw, Delta):  rho = {rho_aw_delta:+.4f}  (p = {p_aw_delta:.2e})  {'**SIG**' if p_aw_delta < 0.01 else ''}"
)
print(f"  Spearman(Aw, IC):     rho = {rho_aw_ic:+.4f}  (p = {p_aw_ic:.2e})  {'**SIG**' if p_aw_ic < 0.01 else ''}")
print(
    f"  Spearman(C, omega):   rho = {rho_c_omega:+.4f}  (p = {p_c_omega:.2e})  {'**SIG**' if p_c_omega < 0.01 else ''}"
)
print(f"  Spearman(Aw, Gamma):  rho = {rho_aw_G:+.4f}  (p = {p_aw_G:.2e})  {'**SIG**' if p_aw_G < 0.01 else ''}")

# ── Part 4: Gravitational depth ──
print()
print("=" * 100)
print("PART 4: COST OF RETURN (GRAVITATIONAL DEPTH)")
print("=" * 100)
G_stable = gamma_cost(0.038)
print(f"  Gamma at Stable edge (omega=0.038): {G_stable:.8f}")
print()
print(f"  {'Organism':<25} {'Gamma':>12} {'slope':>12} {'cost-to-Stable':>15} {'gravity_ratio':>14}")
print("  " + "-" * 85)
for r in sorted(results, key=lambda x: x["G"], reverse=True)[:15]:
    slope = horn_slope(r["omega"])
    cost = r["G"] - G_stable
    ratio = r["G"] / G_stable
    print(f"  {r['name']:<25} {r['G']:12.6f} {slope:12.6f} {cost:15.6f} {ratio:14.1f}")

# ── Part 5: Loop simulation ──
print()
print("=" * 100)
print("PART 5: SIMULATED COLLAPSE-RETURN LOOPS")
print("=" * 100)
rng = np.random.default_rng(42)

print(f"  {'Organism':<25} {'loop_area':>10} {'kappa_dep':>10} {'Gamma_asym':>10} {'C_drift':>8} {'Aw':>5}")
print("  " + "-" * 75)
for r in results[::3]:  # Sample every 3rd
    trace_arr = np.array(r["trace"], dtype=np.float64)
    w = np.ones(len(trace_arr)) / len(trace_arr)
    traj_F, traj_K = [], []
    c_curr = np.clip(trace_arr.copy(), EPSILON, 1 - EPSILON)
    for _ in range(10):
        c_curr = np.clip(c_curr * rng.uniform(0.85, 0.98, len(c_curr)), EPSILON, 1 - EPSILON)
        traj_F.append(float(np.dot(w, c_curr)))
        traj_K.append(float(np.dot(w, np.log(c_curr))))
    k_bottom = traj_K[-1]
    for _ in range(10):
        c_curr = np.clip(c_curr + 0.15 * (trace_arr - c_curr) + rng.normal(0, 0.005, len(c_curr)), EPSILON, 1 - EPSILON)
        traj_F.append(float(np.dot(w, c_curr)))
        traj_K.append(float(np.dot(w, np.log(c_curr))))
    k_top = traj_K[-1]
    n = len(traj_F)
    area = 0.5 * abs(sum(traj_F[i] * traj_K[(i + 1) % n] - traj_F[(i + 1) % n] * traj_K[i] for i in range(n)))
    k_dep = abs(k_top - k_bottom)
    G_top = gamma_cost(1 - traj_F[0])
    G_bot = gamma_cost(1 - traj_F[9])
    asym = G_bot / max(G_top, 1e-20)
    C0 = float(np.std(trace_arr) / 0.5)
    C1 = float(np.std(c_curr) / 0.5)
    print(f"  {r['name']:<25} {area:10.6f} {k_dep:10.6f} {asym:10.2f} {abs(C1 - C0):8.4f} {r['Aw']:5.3f}")

# ── Part 6: Structural impossibility ──
print()
print("=" * 100)
print("PART 6: WHY STABLE IS STRUCTURALLY IMPOSSIBLE FOR AWARE ORGANISMS")
print("=" * 100)
print("  Uniform 10-channel trace analysis:")
for c_u in [0.95, 0.96, 0.962, 0.965, 0.97, 0.98, 0.99]:
    F_u, omega_u, kappa_u, IC_u, S_u, C_u = kernel([c_u] * 10)
    stable = omega_u < 0.038 and F_u > 0.90 and S_u < 0.15 and C_u < 0.14
    fails = []
    if omega_u >= 0.038:
        fails.append(f"omega({omega_u:.4f})")
    if S_u >= 0.15:
        fails.append(f"S({S_u:.4f})")
    print(
        f"    c={c_u:.3f}  omega={omega_u:.4f}  S={S_u:.4f}  C={C_u:.6f}  Stable={'YES' if stable else 'NO  FAILS: ' + ', '.join(fails)}"
    )

print()
print("  Human-like split: Aw=0.94, varying Ap:")
for ap in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.94, 0.962]:
    t = [0.94] * 5 + [ap] * 5
    F_t, omega_t, _, IC_t, S_t, C_t = kernel(t)
    stable_t = omega_t < 0.038 and F_t > 0.90 and S_t < 0.15 and C_t < 0.14
    fails_t = []
    if omega_t >= 0.038:
        fails_t.append(f"omega({omega_t:.4f})")
    if S_t >= 0.15:
        fails_t.append(f"S({S_t:.4f})")
    if C_t >= 0.14:
        fails_t.append(f"C({C_t:.4f})")
    print(
        f"    Ap={ap:.3f}  F={F_t:.4f}  omega={omega_t:.4f}  C={C_t:.4f}  S={S_t:.4f}  {'STABLE' if stable_t else 'FAILS: ' + ', '.join(fails_t)}"
    )

# ── Part 7: The three zones of the horn ──
print()
print("=" * 100)
print("PART 7: THREE ZONES OF THE HORN — WHERE LIFE CIRCULATES")
print("=" * 100)
# What fraction of organisms in each regime region?
stable_count = sum(1 for r in results if r["omega"] < 0.038)
watch_count = sum(1 for r in results if 0.038 <= r["omega"] < 0.30)
collapse_count = sum(1 for r in results if r["omega"] >= 0.30)
print(f"  Stable:   {stable_count}/34 ({100 * stable_count / 34:.1f}%)")
print(f"  Watch:    {watch_count}/34 ({100 * watch_count / 34:.1f}%)")
print(f"  Collapse: {collapse_count}/34 ({100 * collapse_count / 34:.1f}%)")
print()
print("  Theoretical (Fisher space): Stable 12.5%, Watch 24.4%, Collapse 63.1%")
print(
    f"  Biological:                 Stable {100 * stable_count / 34:.1f}%, Watch {100 * watch_count / 34:.1f}%, Collapse {100 * collapse_count / 34:.1f}%"
)
print()

# Where on omega axis do organisms cluster?
print("  omega distribution (34 organisms):")
for lo, hi, _label in [(0.3, 0.5, "Deep Collapse"), (0.5, 0.7, "Mid Collapse"), (0.7, 0.95, "Far Collapse")]:
    cnt = sum(1 for r in results if lo <= r["omega"] < hi)
    names = [r["name"] for r in results if lo <= r["omega"] < hi]
    print(f"    omega in [{lo:.1f}, {hi:.1f}): {cnt} organisms")
    for n in names[:5]:
        print(f"      - {n}")
    if len(names) > 5:
        print(f"      ... and {len(names) - 5} more")

# Mean omega by clade
print()
print("  Mean omega by phylogenetic group:")
clades = {}
for r in results:
    clade = r["name"].split()[0] if "Human" in r["name"] else r["name"]
    group = "H. sapiens" if "Human" in r["name"] else r.get("name", "?")
    # Simplified grouping
for group_name, group_filter in [
    ("Invertebrates", lambda r: r["Aw"] < 0.1 and "Human" not in r["name"]),
    (
        "Fish/Reptiles",
        lambda r: (
            0.0 < r["Aw"] < 0.25
            and "Human" not in r["name"]
            and r["name"] in {"Cleaner wrasse", "Archerfish", "Monitor lizard"}
        ),
    ),
    ("Birds", lambda r: r["name"] in {"Pigeon", "Magpie", "NC crow", "African grey"}),
    ("Non-primate mammals", lambda r: r["name"] in {"Mouse", "Dog", "Pig", "Elephant", "Bottlenose dolphin"}),
    (
        "Primates (non-human)",
        lambda r: r["name"] in {"Capuchin monkey", "Gorilla", "Orangutan", "Bonobo", "Chimpanzee"},
    ),
    ("Homo sapiens", lambda r: "Human" in r["name"]),
]:
    grp = [r for r in results if group_filter(r)]
    if grp:
        mean_aw = np.mean([r["Aw"] for r in grp])
        mean_omega = np.mean([r["omega"] for r in grp])
        mean_C = np.mean([r["C"] for r in grp])
        mean_G = np.mean([r["G"] for r in grp])
        binds = [r["bind"] for r in grp]
        print(
            f"    {group_name:<25} n={len(grp):2d}  <Aw>={mean_aw:.3f}  <omega>={mean_omega:.3f}  <C>={mean_C:.3f}  <Gamma>={mean_G:.4f}  binds: {set(binds)}"
        )

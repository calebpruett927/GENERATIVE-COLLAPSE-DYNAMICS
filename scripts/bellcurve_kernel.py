"""Compute kernel invariants for the Bell Curve Revisited casepack."""

from __future__ import annotations

import json
import math
import statistics

EPSILON = 1e-8


def compute_kernel(channels: list[float], weights: list[float]) -> dict:
    n = len(channels)
    c = [max(EPSILON, min(1 - EPSILON, x)) for x in channels]
    F = sum(w * ci for w, ci in zip(weights, c, strict=True))
    omega = 1.0 - F
    kappa = sum(w * math.log(max(ci, EPSILON)) for w, ci in zip(weights, c, strict=True))
    IC = math.exp(kappa)
    S = -sum(w * (ci * math.log(ci) + (1 - ci) * math.log(1 - ci)) for w, ci in zip(weights, c, strict=True))
    C = statistics.stdev(c) / 0.5 if n > 1 else 0.0
    delta = F - IC
    return {"F": F, "omega": omega, "S": S, "C": C, "kappa": kappa, "IC": IC, "delta": delta}


# Equal weights, 6 channels
w = [1 / 6] * 6

# Channels (from Tables 2-5):
# c1: HS completion PGS effect (normalized)
# c2: College attendance PGS effect (normalized)
# c3: College completion PGS effect (normalized)
# c4: Graduate school PGS effect (normalized)
# c5: Assortative mating PGS correlation (normalized)
# c6: Fertility PGS effect (normalized)

# Entity 1: Early cohort (1919) - PGS most predictive
early = [0.430, 0.645, 0.525, 0.370, 0.340, 0.466667]

# Entity 2: Middle cohort (1937) - median
middle = [0.400, 0.575, 0.475, 0.310, 0.300, 0.400]

# Entity 3: Late cohort (1955) - PGS declining
late = [0.360, 0.500, 0.425, 0.350, 0.180, 0.300]

entities = [
    ("Early cohort (1919)", early),
    ("Middle cohort (1937)", middle),
    ("Late cohort (1955)", late),
]

print("=== KERNEL COMPUTATIONS ===")
for name, channels in entities:
    r = compute_kernel(channels, w)
    print(f"\n{name}: c = {channels}")
    for k, v in r.items():
        print(f"  {k:8s} = {v:.6f}")

print("\n=== REGIME CLASSIFICATION ===")
rows = []
for t, (name, channels) in enumerate(entities):
    r = compute_kernel(channels, w)
    omega = r["omega"]
    F = r["F"]
    S = r["S"]
    Cv = r["C"]
    if omega >= 0.30:
        regime = "Collapse"
    elif omega < 0.038 and F > 0.90 and S < 0.15 and Cv < 0.14:
        regime = "Stable"
    else:
        regime = "Watch"
    critical = r["IC"] < 0.30
    print(f"  {name}: regime={regime}, critical={critical}")
    rows.append(
        {
            "t": t,
            "omega": round(r["omega"], 10),
            "F": round(r["F"], 10),
            "S": round(r["S"], 10),
            "C": round(r["C"], 10),
            "tau_R": "INF_REC",
            "kappa": round(r["kappa"], 10),
            "IC": round(r["IC"], 10),
            "regime": {"label": regime, "critical_overlay": critical},
        }
    )

print("\n=== INVARIANTS JSON (rows) ===")
print(json.dumps(rows, indent=2))

# Tier-1 identity checks
print("\n=== TIER-1 IDENTITY CHECKS ===")
for name, channels in entities:
    r = compute_kernel(channels, w)
    check_fow = abs(r["F"] + r["omega"] - 1.0)
    check_ic_exp = abs(r["IC"] - math.exp(r["kappa"]))
    check_ic_le_f = r["IC"] <= r["F"]
    print(f"  {name}:")
    print(f"    |F + omega - 1| = {check_fow:.2e}")
    print(f"    |IC - exp(kappa)| = {check_ic_exp:.2e}")
    print(f"    IC <= F: {check_ic_le_f} (IC={r['IC']:.6f}, F={r['F']:.6f})")

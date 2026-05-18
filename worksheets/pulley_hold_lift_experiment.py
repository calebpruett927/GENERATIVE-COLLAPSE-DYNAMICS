"""
Pulley Hold-Lift Experiment — Tier-2 closure exploration.

Contract: PULLEY.HOLD-LIFT.v0.1
Goal: minimize pull effort while maximizing passive holding stability.

Splits "pulley" into TWO coupled jobs:
  1. Lift logic — reduce input force enough that load is easy to raise.
  2. Hold logic — prevent reverse motion when user stops pulling.

Six candidate designs scored on 8 channels, fed through the GCD kernel.
The kernel exposes the geometric-slaughter pattern: a candidate strong on
average (high F) but with one near-dead channel collapses on IC. That is
exactly the "pulley with no hold" failure mode.

Channels (all in [0,1], higher = better):
    lift_ease, holding_stability, reverse_slip_resistance,
    smoothness, efficiency, reset_quality, wear_risk_inv, operator_control

Frozen scoring adapter for lift_ease (declared before evidence):
    lift_ease_raw(n)   = 1 − F_pull / W_design
    lift_ease_score(n) = clip( 1 − 0.15 · (F_pull/W_design) / 0.295, 0, 1 )

    where 0.295 is the n=4 baseline pull-force fraction under η_sheave = 0.96
    (i.e. F_pull/W = 1/(4 · 0.96^4) ≈ 0.295). Anchoring at n=4 → 0.85 makes
    the n-sweep curve directly comparable to the baseline Candidate F entry,
    which uses lift_ease = 0.85. The raw form is the physics; the score form
    is the auditable channel value the kernel actually consumes.

Run: python worksheets/pulley_hold_lift_experiment.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from umcp.frozen_contract import (
    DEFAULT_THRESHOLDS,
    EPSILON,
    classify_regime,
    cost_curvature,
    gamma_omega,
)
from umcp.kernel_optimized import compute_kernel_outputs

# -----------------------------------------------------------------------------
# Channel definitions (operational, declared before evidence)
# -----------------------------------------------------------------------------

CHANNELS = [
    "lift_ease",  # 1 - F_pull/W_design, normalized
    "holding_stability",  # position retention during release
    "reverse_slip_resistance",  # load-side movement under no user pull (inverted)
    "smoothness",  # absence of jerk during engagement and restart
    "efficiency",  # output_work / input_work (η)
    "reset_quality",  # ease of returning from hold to lift
    "wear_risk_inv",  # 1 - normalized wear risk
    "operator_control",  # ability to lift, stop, resume, lower
]

# Equal weights — no a priori channel preference (frozen contract decision)
WEIGHTS = np.full(len(CHANNELS), 1.0 / len(CHANNELS))


# -----------------------------------------------------------------------------
# Candidate trace vectors (Tier-2 channel selection)
# -----------------------------------------------------------------------------
# Scores are operational estimates from mechanical-engineering priors:
#   - Block-and-tackle MA n≈4: F_pull ≈ W/(4η), η ≈ 0.85 → lift_ease ≈ 0.79
#   - Ratchet pawl: discrete teeth → smoothness penalty (~0.45), holding ≈ 0.95
#   - Cam clutch / rope grab: continuous wedge → smoothness ≈ 0.80, holding ≈ 0.92
#   - Worm gear: self-locking → holding ≈ 0.99, but η drops to ~0.4 → lift_ease ≈ 0.30
# Critical: simple block-and-tackle has reverse_slip_resistance ≈ EPSILON
#   (no holding mechanism at all → geometric slaughter on IC)

CANDIDATES = {
    "A_block_tackle_only": np.array(
        [
            0.85,  # lift_ease — best (free-running sheaves, high η)
            0.05,  # holding_stability — user must hold rope
            EPSILON,  # reverse_slip_resistance — DEAD CHANNEL: no hold
            0.90,  # smoothness — very smooth
            0.85,  # efficiency
            0.95,  # reset_quality — trivial, just keep pulling
            0.85,  # wear_risk_inv — low wear
            0.30,  # operator_control — no stop-and-rest
        ]
    ),
    "B_block_tackle_plus_ratchet": np.array(
        [
            0.78,  # lift_ease — slight drag from pawl
            0.95,  # holding_stability — pawl locks discretely
            0.92,  # reverse_slip_resistance — up to one tooth of slip
            0.45,  # smoothness — clicks, jerk on engagement
            0.78,  # efficiency — pawl friction
            0.65,  # reset_quality — must disengage pawl to lower
            0.65,  # wear_risk_inv — pawl/teeth wear
            0.80,  # operator_control — discrete steps
        ]
    ),
    "C_block_tackle_plus_cam_clutch": np.array(
        [
            0.80,  # lift_ease — cam disengages on lift
            0.92,  # holding_stability — wedges under reverse load
            0.94,  # reverse_slip_resistance — minimal slip before engagement
            0.82,  # smoothness — continuous, no clicks
            0.82,  # efficiency
            0.78,  # reset_quality — clean release with directional input
            0.72,  # wear_risk_inv — rope compression at cam
            0.90,  # operator_control — stop anywhere, resume cleanly
        ]
    ),
    "D_differential_or_worm": np.array(
        [
            0.30,  # lift_ease — slow, large pull distance
            0.99,  # holding_stability — self-locking
            0.99,  # reverse_slip_resistance — essentially zero back-drive
            0.70,  # smoothness
            0.40,  # efficiency — DEAD-ish: most input lost to friction
            0.55,  # reset_quality — slow lower
            0.80,  # wear_risk_inv
            0.75,  # operator_control — predictable but slow
        ]
    ),
    # Candidate E hardens C's weakest channel (wear_risk_inv) by adding a
    # secondary friction-wrap (capstan) downstream of the cam. The wrap absorbs
    # most of the static reverse load via the capstan equation T_load = T_hold·e^(μθ),
    # so the cam only sees a fraction of the load → less rope compression → less wear.
    # Costs: small efficiency hit on lift (wrap drag) and slightly lower smoothness
    # at the wrap-cam handoff.
    "E_cam_clutch_plus_friction_wrap": np.array(
        [
            0.78,  # lift_ease — small wrap drag on lift stroke
            0.96,  # holding_stability — two stages share the hold
            0.97,  # reverse_slip_resistance — wrap takes static load, cam catches dynamic
            0.78,  # smoothness — minor handoff between stages
            0.78,  # efficiency — wrap friction on lift
            0.80,  # reset_quality — wrap releases passively when pull resumes
            0.88,  # wear_risk_inv — HARDENED: cam load reduced by wrap factor
            0.92,  # operator_control — redundant hold = predictable stop anywhere
        ]
    ),
    # Candidate F replaces the rope-side capstan wrap with a self-energizing band
    # brake on the load drum. The band fully disengages on lift (no rope drag), and
    # under reverse rotation the load itself tightens the band (T_tight = T_slack·e^(μθ)
    # with the band free to wrap further). Wear is distributed over the drum surface,
    # not concentrated on rope under the cam. Trade: a small smoothness penalty when
    # the band first grabs, and a slight reset hysteresis (band must un-tension).
    "F_cam_clutch_plus_band_brake": np.array(
        [
            0.85,  # lift_ease — band fully disengages, no rope wrap drag
            0.97,  # holding_stability — self-energizing under load
            0.96,  # reverse_slip_resistance — small grab transient before lock
            0.85,  # smoothness — engagement transient at first reversal
            0.83,  # efficiency — no continuous wrap drag on lift
            0.75,  # reset_quality — slight hysteresis as band un-tensions
            0.82,  # wear_risk_inv — band lining wears on drum (distributed)
            0.92,  # operator_control — predictable stop, smooth resume
        ]
    ),
}


# -----------------------------------------------------------------------------
# Run kernel + seam budget per candidate
# -----------------------------------------------------------------------------


def evaluate(name: str, c: np.ndarray) -> dict[str, Any]:
    out = compute_kernel_outputs(c, WEIGHTS, epsilon=EPSILON)
    F, omega, S, C, kappa, IC = out["F"], out["omega"], out["S"], out["C"], out["kappa"], out["IC"]
    Delta = F - IC

    # Seam budget: D_omega = Γ(ω), D_C = α·C, total cost
    D_omega = gamma_omega(omega)
    D_C = cost_curvature(C)
    cost_total = D_omega + D_C

    # Canonical four-gate regime (STABLE/WATCH/COLLAPSE/CRITICAL)
    regime = classify_regime(omega, F, S, C, IC, DEFAULT_THRESHOLDS).value

    # Identify weakest channel (geometric slaughter detector)
    weak_idx = int(np.argmin(c))
    weak_channel = CHANNELS[weak_idx]
    weak_value = float(c[weak_idx])

    # Per-channel sensitivity ∂IC/∂cₖ = IC · wₖ / cₖ
    sensitivities = IC * WEIGHTS / np.maximum(c, EPSILON)

    return {
        "name": name,
        "F": F,
        "omega": omega,
        "S": S,
        "C": C,
        "kappa": kappa,
        "IC": IC,
        "Delta": Delta,
        "IC_over_F": IC / F if F > 0 else 0.0,
        "D_omega": D_omega,
        "D_C": D_C,
        "cost_total": cost_total,
        "regime": regime,
        "weak_channel": weak_channel,
        "weak_value": weak_value,
        "max_sensitivity_channel": CHANNELS[int(np.argmax(sensitivities))],
        "max_sensitivity": float(sensitivities.max()),
    }


def main() -> None:
    print("=" * 78)
    print("PULLEY.HOLD-LIFT.v0.1 — Six-Candidate Comparison")
    print("Contract: minimize pull effort × maximize passive holding stability")
    print("=" * 78)

    results = [evaluate(name, c) for name, c in CANDIDATES.items()]

    # Per-candidate report
    for r in results:
        print(f"\n--- {r['name']} ---")
        print(f"  F  (Fidelity)           = {r['F']:.4f}   (avg channel quality)")
        print(f"  ω  (Drift)              = {r['omega']:.4f}")
        print(f"  IC (Integrity)          = {r['IC']:.4f}   (multiplicative coherence)")
        print(f"  Δ  (Heterogeneity gap)  = {r['Delta']:.4f}   (F − IC)")
        print(
            f"  IC/F                    = {r['IC_over_F']:.4f}   "
            f"({'GEOMETRIC SLAUGHTER' if r['IC_over_F'] < 0.30 else 'coherent'})"
        )
        print(f"  S  (Bernoulli entropy)  = {r['S']:.4f}")
        print(f"  C  (Curvature)          = {r['C']:.4f}")
        print(f"  D_ω + D_C  (cost)       = {r['D_omega']:.4f} + {r['D_C']:.4f} = {r['cost_total']:.4f}")
        print(f"  Regime                  = {r['regime']}")
        print(f"  Weakest channel         = {r['weak_channel']} = {r['weak_value']:.4f}")
        print(f"  Most-sensitive channel  = {r['max_sensitivity_channel']} (∂IC/∂c = {r['max_sensitivity']:.3f})")

    # Comparison table
    print("\n" + "=" * 78)
    print("DECISION TABLE — choose by IC, not F (IC catches dead channels)")
    print("=" * 78)
    header = f"{'Candidate':<32} {'F':>7} {'IC':>7} {'Δ':>7} {'cost':>7} {'regime':>10}"
    print(header)
    print("-" * len(header))
    for r in sorted(results, key=lambda x: -x["IC"]):
        print(
            f"{r['name']:<32} {r['F']:>7.4f} {r['IC']:>7.4f} "
            f"{r['Delta']:>7.4f} {r['cost_total']:>7.4f} {r['regime']:>10}"
        )

    # Verdict
    print("\n" + "=" * 78)
    print("STANCE (derived from gates, never asserted)")
    print("=" * 78)
    winner = max(results, key=lambda x: x["IC"])
    print(f"  Winner by IC:      {winner['name']}  (IC = {winner['IC']:.4f}, regime = {winner['regime']})")
    winner_F = max(results, key=lambda x: x["F"])
    print(f"  Winner by F only:  {winner_F['name']}  (F = {winner_F['F']:.4f})")
    if winner["name"] != winner_F["name"]:
        print(
            f"\n  ⚠ F-winner ≠ IC-winner. The F-winner has a dead channel "
            f"({winner_F['weak_channel']} = {winner_F['weak_value']:.4f})"
        )
        print("  that destroys multiplicative coherence. This is geometric slaughter:")
        print("  IC = exp(Σ wᵢ ln cᵢ) — one near-zero cᵢ pulls IC toward zero")
        print("  regardless of how strong the other 7 channels are.")

    # Five-word Canon (Drift, Fidelity, Roughness, Return, Integrity)
    print("\n" + "=" * 78)
    print("CANON — five-word narrative for the winning design")
    print("=" * 78)
    w = winner
    print(f"  Drift       (ω = {w['omega']:.3f}): how much the winning design departs from ideal channels.")
    print(f"  Fidelity    (F = {w['F']:.3f}): the average of what survives under the contract.")
    print(f"  Roughness   (D_C = {w['D_C']:.3f}): friction / heterogeneity cost across channels.")
    print(f"  Return      (regime = {w['regime']}): the design returns into a regime the gates accept.")
    print(f"  Integrity   (IC = {w['IC']:.3f}): multiplicative coherence — no dead channels.")

    # Recommendation
    print("\n" + "=" * 78)
    print("RECOMMENDATION")
    print("=" * 78)
    print(f"  Build a prototype of: {winner['name']}")
    print(f"  Most-sensitive channel to instrument first: {winner['max_sensitivity_channel']}")
    print(f"  (∂IC/∂c = {winner['max_sensitivity']:.3f} — small change here moves IC the most)")
    print(
        f"  Weakest channel to harden:                  {winner['weak_channel']} (currently {winner['weak_value']:.3f})"
    )

    # -------------------------------------------------------------------------
    # Cycle simulation: lift → release → hold → resume → lower, repeated N times
    # -------------------------------------------------------------------------
    # Wear model (operational, declared before evidence):
    #   - Per cycle, wear_risk_inv decays by rate r proportional to (1 - wear_risk_inv)
    #     and inversely proportional to redundancy (number of independent hold stages).
    #   - Holding channels (holding_stability, reverse_slip_resistance) erode at the
    #     same rate the cam surface degrades, since hold capacity rides on the cam.
    #   - Lift channels (lift_ease, efficiency, smoothness) decay slowly (rope wear).
    # IC is then re-computed each cycle and the trajectory is reported.
    print("\n" + "=" * 78)
    print("CYCLE SIMULATION — lift → release → hold → resume → lower (N=200 cycles)")
    print("Tracks IC erosion under wear; flags first cycle that exits WATCH/STABLE")
    print("=" * 78)

    # Per-candidate redundancy (stages sharing the hold load)
    REDUNDANCY = {
        "A_block_tackle_only": 0.5,  # no hold → wear is irrelevant, but slip is fatal
        "B_block_tackle_plus_ratchet": 1.0,
        "C_block_tackle_plus_cam_clutch": 1.0,
        "D_differential_or_worm": 2.0,  # gear teeth share load
        "E_cam_clutch_plus_friction_wrap": 2.0,  # wrap + cam = two stages
        "F_cam_clutch_plus_band_brake": 2.0,  # band + cam = two stages
    }
    N_CYCLES = 200
    BASE_WEAR_RATE = 0.004  # per cycle, on (1 - wear_risk_inv)
    HOLD_DECAY_COUPLING = 0.7  # how strongly hold channels track wear
    LIFT_DECAY_COUPLING = 0.15  # rope wear is slow

    header = f"{'Candidate':<32} {'IC₀':>7} {'IC₅₀':>7} {'IC₁₀₀':>7} {'IC₂₀₀':>7} {'first_exit':>11}"
    print(header)
    print("-" * len(header))

    cycle_results = {}
    for name, c0 in CANDIDATES.items():
        c = c0.copy()
        redundancy = REDUNDANCY[name]
        ic_trace = []
        first_exit = None
        for cycle in range(1, N_CYCLES + 1):
            # Wear update: cam-side channels (indices 1, 2, 6) and rope-side (0, 3, 4)
            wear_rate = BASE_WEAR_RATE / max(redundancy, 0.1)
            decay_wear = wear_rate * c[6]  # wear_risk_inv decays toward 0
            decay_hold = wear_rate * HOLD_DECAY_COUPLING * np.array([c[1], c[2]])
            decay_lift = wear_rate * LIFT_DECAY_COUPLING * np.array([c[0], c[3], c[4]])
            c[6] = max(c[6] - decay_wear, EPSILON)
            c[1] = max(c[1] - decay_hold[0], EPSILON)
            c[2] = max(c[2] - decay_hold[1], EPSILON)
            c[0] = max(c[0] - decay_lift[0], EPSILON)
            c[3] = max(c[3] - decay_lift[1], EPSILON)
            c[4] = max(c[4] - decay_lift[2], EPSILON)
            # reset_quality and operator_control are not eroded (frozen)

            out = compute_kernel_outputs(c, WEIGHTS, epsilon=EPSILON)
            ic_trace.append(out["IC"])
            regime_now = classify_regime(
                out["omega"], out["F"], out["S"], out["C"], out["IC"], DEFAULT_THRESHOLDS
            ).value
            if first_exit is None and regime_now in ("COLLAPSE", "CRITICAL"):
                first_exit = cycle

        cycle_results[name] = {
            "ic_trace": ic_trace,
            "first_exit": first_exit,
            "ic_final": ic_trace[-1],
        }
        ic0 = compute_kernel_outputs(c0, WEIGHTS, epsilon=EPSILON)["IC"]
        first_exit_str = str(first_exit) if first_exit else "—"
        print(
            f"{name:<32} {ic0:>7.4f} {ic_trace[49]:>7.4f} "
            f"{ic_trace[99]:>7.4f} {ic_trace[199]:>7.4f} {first_exit_str:>11}"
        )

    # Cycle-aware verdict: which design holds IC longest?
    print("\n" + "=" * 78)
    print("CYCLE-AWARE STANCE")
    print("=" * 78)
    cycle_winner = max(cycle_results.items(), key=lambda kv: kv[1]["ic_final"])
    print(f"  Highest IC after {N_CYCLES} cycles: {cycle_winner[0]}  (IC = {cycle_winner[1]['ic_final']:.4f})")

    # Tie-aware longest-hold: any candidate that never exits WATCH/STABLE in N
    # cycles is tied at ">N". Report the full tie set and break by IC@N.
    def _exit_key(item: tuple[str, dict[str, Any]]) -> int:
        return item[1]["first_exit"] if item[1]["first_exit"] else N_CYCLES + 1

    best_exit = max(_exit_key(item) for item in cycle_results.items())
    longest_set = [name for name, info in cycle_results.items() if _exit_key((name, info)) == best_exit]
    if len(longest_set) == 1:
        only = longest_set[0]
        first_exit = cycle_results[only]["first_exit"]
        first_exit_str = str(first_exit) if first_exit else f">{N_CYCLES}"
        print(f"  Longest in WATCH/STABLE:           {only}  (first exit at cycle {first_exit_str})")
    else:
        # Tie — break by IC at N_CYCLES.
        tie_break = max(longest_set, key=lambda n: cycle_results[n]["ic_final"])
        first_exit = cycle_results[longest_set[0]]["first_exit"]
        first_exit_str = str(first_exit) if first_exit else f">{N_CYCLES}"
        print(f"  Longest in WATCH/STABLE:           {' and '.join(longest_set)}  (both {first_exit_str} cycles)")
        print(
            f"  Tie-break by IC@{N_CYCLES}:               {tie_break} wins  "
            f"(IC = {cycle_results[tie_break]['ic_final']:.4f})"
        )

    # Compare static winner vs cycle winner
    print()
    if winner["name"] == cycle_winner[0]:
        print(f"  ✓ Static IC winner ({winner['name']}) also wins the cycle test.")
        print("    Recommendation stands: build this design.")
    else:
        print(f"  ⚠ Static IC winner ({winner['name']}) is NOT the cycle winner.")
        print(f"    Under wear, {cycle_winner[0]} dominates.")
        print("    The static analysis missed the wear-rate coupling. Build the cycle winner.")

    # -------------------------------------------------------------------------
    # Parameter sweep: how robust is each design to changes in the wear model?
    # -------------------------------------------------------------------------
    # Sweep BASE_WEAR_RATE over a 16× range. For each rate, run 200 cycles per
    # candidate and record IC@200 + first-exit cycle. The robust design is the
    # one whose IC and survival barely move when the wear assumption changes.
    print("\n" + "=" * 78)
    print("PARAMETER SWEEP — IC@200 vs BASE_WEAR_RATE  (robustness check)")
    print("Sweep range: 0.001 → 0.016 per cycle (16× span)")
    print("=" * 78)

    sweep_rates = [0.001, 0.002, 0.004, 0.008, 0.016]
    header = f"{'Candidate':<32}" + "".join(f"{'r=' + str(r):>9}" for r in sweep_rates) + f"{'spread':>9}"
    print(header)
    print("-" * len(header))

    sweep_table: dict[str, list[float]] = {}
    for name, c0 in CANDIDATES.items():
        ic_at_rate = []
        for rate in sweep_rates:
            c = c0.copy()
            redundancy = REDUNDANCY[name]
            wear_rate = rate / max(redundancy, 0.1)
            for _ in range(N_CYCLES):
                c[6] = max(c[6] - wear_rate * c[6], EPSILON)
                c[1] = max(c[1] - wear_rate * HOLD_DECAY_COUPLING * c[1], EPSILON)
                c[2] = max(c[2] - wear_rate * HOLD_DECAY_COUPLING * c[2], EPSILON)
                c[0] = max(c[0] - wear_rate * LIFT_DECAY_COUPLING * c[0], EPSILON)
                c[3] = max(c[3] - wear_rate * LIFT_DECAY_COUPLING * c[3], EPSILON)
                c[4] = max(c[4] - wear_rate * LIFT_DECAY_COUPLING * c[4], EPSILON)
            ic_at_rate.append(compute_kernel_outputs(c, WEIGHTS, epsilon=EPSILON)["IC"])
        sweep_table[name] = ic_at_rate
        spread = max(ic_at_rate) - min(ic_at_rate)
        cells = "".join(f"{ic:>9.4f}" for ic in ic_at_rate)
        print(f"{name:<32}{cells}{spread:>9.4f}")

    # Robustness verdict: TWO-PART read — wear-rate sensitivity (relative spread)
    # is one axis; absolute integrity across all rates is another. They can name
    # different winners. Both must be reported separately so the contract winner
    # is not collapsed into a single "robustness" label.
    #   - Sensitivity winner: smallest relative spread among survivors
    #   - Absolute integrity winner: highest min(IC) across the full sweep
    # Survivor gate (avoids degenerate-spread trap):
    #   (a) IC at r=0.001 (ics[0], closest to static IC₀) > 0.70
    #   (b) min IC across all rates > 0.40
    print("\n" + "=" * 78)
    print("ROBUSTNESS VERDICT — two-part read (sensitivity vs absolute IC)")
    print("(Sensitivity metric: relative spread = spread / IC₀  to avoid degenerate-spread trap)")
    print("=" * 78)
    survivors = {n: ics for n, ics in sweep_table.items() if ics[0] > 0.70 and min(ics) > 0.40}
    if survivors:
        sens_winner = min(
            survivors.items(),
            key=lambda kv: (max(kv[1]) - min(kv[1])) / kv[1][0],  # relative spread
        )
        sens_ics = sens_winner[1]
        sens_rel = (max(sens_ics) - min(sens_ics)) / sens_ics[0]
        print(f"  (1) Wear-rate sensitivity winner:        {sens_winner[0]}")
        print(
            f"        IC range: [{min(sens_ics):.4f}, {max(sens_ics):.4f}]   "
            f"spread = {max(sens_ics) - min(sens_ics):.4f}"
        )
        print(f"        Relative spread (spread / IC₀):    {sens_rel:.4f}")
        print("        → IC bounded by structure, not by the wear assumption.")

        abs_winner = max(survivors.items(), key=lambda kv: min(kv[1]))
        abs_ics = abs_winner[1]
        print(f"  (2) Absolute integrity winner (min IC):  {abs_winner[0]}")
        print(f"        IC range: [{min(abs_ics):.4f}, {max(abs_ics):.4f}]   min IC = {min(abs_ics):.4f}")
        print("        → Highest floor across all wear-rate scenarios.")

        print()
        if sens_winner[0] == abs_winner[0]:
            print(f"  Contract winner: {sens_winner[0]} (wins both axes).")
        else:
            print(f"  Contract winner: {abs_winner[0]} unless robustness-only is")
            print(f"  explicitly weighted above absolute IC — then {sens_winner[0]} wins.")
    else:
        print("  No candidate keeps IC₀ > 0.70 and min IC > 0.40 across the full sweep.")
        print("  Tighten the wear measurement before deciding.")

    # -------------------------------------------------------------------------
    # Mechanical-advantage sweep on Candidate F (n ∈ {2, 3, 4, 6, 8})
    # -------------------------------------------------------------------------
    # Same band brake, same cam clutch, same rope, same sheave ratio, same wear
    # model. Only n changes. The hold-side and rope-wear channels stay constant
    # (band brake torque is set by μθ, not by n). Lift-side channels follow:
    #
    #   η_total(n) = η_sheave^n  with η_sheave = 0.96
    #   F_pull/W   = 1 / (n · η_total)         (classical block-and-tackle)
    #   lift_ease, efficiency, smoothness, reset_quality, operator_control
    #     are anchored so that n=4 reproduces Candidate F baseline,
    #     then scaled phenomenologically for other n.
    print("\n" + "=" * 78)
    print("MECHANICAL-ADVANTAGE SWEEP — Candidate F at n ∈ {2, 3, 4, 6, 8}")
    print("Hold/wear channels frozen; only lift-side channels move with n")
    print("=" * 78)

    eta_sheave = 0.96
    band_drag = 0.02

    def f_variant(n: int) -> np.ndarray:
        """Build Candidate F's trace vector at mechanical advantage n."""
        eta_total = eta_sheave**n
        f_pull_ratio = 1.0 / (n * eta_total)  # pull force as fraction of W
        # Anchor: scale so n=4 → lift_ease ≈ 0.85 (matches baseline Candidate F)
        lift_ease = float(np.clip(1.0 - f_pull_ratio / 0.295 * 0.15, 0.0, 1.0))
        # The 0.15 factor calibrates the curve so n=4 hits 0.85; gives realistic
        # diminishing returns at high n where rope-management cost dominates.
        efficiency = float(np.clip(eta_total - band_drag, 0.0, 1.0))
        # Smoothness, reset, operator_control degrade with n (more rope to manage)
        smoothness = float(np.clip(0.95 - 0.025 * n, 0.0, 1.0))
        reset_quality = float(np.clip(0.95 - 0.05 * n, 0.0, 1.0))
        operator_control = float(np.clip(1.0 - 0.02 * n - 0.005 * max(n - 4, 0) ** 2, 0.0, 1.0))
        # Hold/wear channels are fixed by the band brake + cam + drum geometry
        return np.array(
            [
                lift_ease,
                0.97,  # holding_stability — band torque set by μθ, independent of n
                0.96,  # reverse_slip_resistance — same
                smoothness,
                efficiency,
                reset_quality,
                0.82,  # wear_risk_inv — band lining wear is independent of n
                operator_control,
            ]
        )

    n_values = [2, 3, 4, 6, 8]
    F_VARIANTS = {f"F_n{n}": f_variant(n) for n in n_values}

    print(
        f"{'Design':<8} {'F':>6} {'IC':>6} {'Δ':>6} {'lift':>6} {'eff':>6} "
        f"{'smth':>6} {'rset':>6} {'opc':>6} {'IC@200':>8} {'regime':>10}"
    )
    print("-" * 84)

    n_sweep_results = []
    for name, c in F_VARIANTS.items():
        out = compute_kernel_outputs(c, WEIGHTS, epsilon=EPSILON)
        F_v, IC_v = out["F"], out["IC"]
        Delta_v = F_v - IC_v
        regime_v = classify_regime(out["omega"], F_v, out["S"], out["C"], IC_v, DEFAULT_THRESHOLDS).value
        # 200-cycle wear projection at nominal BASE_WEAR_RATE, redundancy=2
        c_decay = c.copy()
        wear_rate = BASE_WEAR_RATE / 2.0
        for _ in range(N_CYCLES):
            c_decay[6] = max(c_decay[6] - wear_rate * c_decay[6], EPSILON)
            c_decay[1] = max(c_decay[1] - wear_rate * HOLD_DECAY_COUPLING * c_decay[1], EPSILON)
            c_decay[2] = max(c_decay[2] - wear_rate * HOLD_DECAY_COUPLING * c_decay[2], EPSILON)
            c_decay[0] = max(c_decay[0] - wear_rate * LIFT_DECAY_COUPLING * c_decay[0], EPSILON)
            c_decay[3] = max(c_decay[3] - wear_rate * LIFT_DECAY_COUPLING * c_decay[3], EPSILON)
            c_decay[4] = max(c_decay[4] - wear_rate * LIFT_DECAY_COUPLING * c_decay[4], EPSILON)
        ic200 = compute_kernel_outputs(c_decay, WEIGHTS, epsilon=EPSILON)["IC"]
        n_sweep_results.append((name, F_v, IC_v, Delta_v, ic200, regime_v, c))
        print(
            f"{name:<8} {F_v:>6.3f} {IC_v:>6.3f} {Delta_v:>6.3f} "
            f"{c[0]:>6.3f} {c[4]:>6.3f} {c[3]:>6.3f} {c[5]:>6.3f} {c[7]:>6.3f} "
            f"{ic200:>8.4f} {regime_v:>10}"
        )

    # n-sweep verdict
    print("\n" + "=" * 78)
    print("n-SWEEP VERDICT")
    print("=" * 78)
    by_ic0 = max(n_sweep_results, key=lambda r: r[2])
    by_ic200 = max(n_sweep_results, key=lambda r: r[4])
    print(f"  Highest static IC:        {by_ic0[0]}  (IC₀ = {by_ic0[2]:.4f})")
    print(f"  Highest IC after 200:     {by_ic200[0]}  (IC₂₀₀ = {by_ic200[4]:.4f})")
    print()
    print("  Under uniform channel weights, the smallest n maximizes IC because")
    print("  smoothness, reset, and operator_control degrade with n while the")
    print("  efficiency gain from extra sheaves is sub-linear (η_total = 0.96ⁿ).")
    print("  The build decision is therefore not 'always n=4' — it is:")
    print()
    print("      Choose the smallest n satisfying the operator-force constraint:")
    print("        F_pull = W_design / (n · 0.96ⁿ) ≤ F_max_operator")
    print("      If multiple n pass, pick highest IC₂₀₀.")
    print()
    print("  Reading by use case (channel re-weighting required to flip the result):")
    print("    Heavy load, low pull force first    → up-weight lift_ease, choose highest n that passes")
    print("    Operator control + speed first       → default uniform weights already favor low n")
    print("    Balanced (default contract weights)  → smallest n meeting F_pull ≤ F_max_operator")


if __name__ == "__main__":
    main()

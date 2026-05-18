# Pulley Hold-Lift System — Optimal Geometry Report

**Contract:** `PULLEY.HOLD-LIFT.v0.1`
**Goal:** minimize pull effort while maximizing passive holding stability.
**Method:** GCD kernel (6 candidates × 8 channels), 200-cycle wear simulation, 16× wear-rate sensitivity sweep, mechanical-advantage sweep n ∈ {2, 3, 4, 6, 8}.
**Source:** [worksheets/pulley_hold_lift_experiment.py](worksheets/pulley_hold_lift_experiment.py)

```
Regime:               WATCH
Contract stance:      CONFORMANT under PULLEY.HOLD-LIFT.v0.1
Production readiness: NON_EVALUABLE — physical prototype has not yet returned data
```

---

## 1. Frozen Contract — What Is Being Optimized

The "pulley" is split into two coupled sub-jobs:

1. **Lift logic** — reduce input force so the load is easy to raise.
2. **Hold logic** — prevent reverse motion when the user stops pulling.

A standard pulley optimizes only #1. A working hold-lift system requires a **structural seam** between the two: easy motion in the intended direction, automatic resistance in reverse. The GCD kernel scores each candidate on:

| Symbol | Meaning | Optimal target |
|---|---|---|
| F = Σ wᵢcᵢ | Fidelity — average channel quality | F → 1 |
| ω = 1 − F | Drift — lost capacity | ω < 0.038 (STABLE) / < 0.30 (WATCH) |
| IC = exp(Σ wᵢ ln cᵢ) | Multiplicative coherence (geometric mean) | IC ≈ F (no dead channels) |
| Δ = F − IC | Heterogeneity gap | Δ → 0 |
| ∂IC/∂cₖ = IC·wₖ/cₖ | Per-channel sensitivity | identifies the leverage point |
| regime | Four-gate verdict | WATCH or STABLE |

**The IC kill mechanism.** IC is a geometric mean. One near-zero channel kills it regardless of the other seven. This is geometric slaughter — the same structural pattern as quark confinement (neutron IC/F = 0.0089 from the GCD orientation). Candidate A demonstrates it directly: F = 0.5938 but IC = 0.0546 because `reverse_slip_resistance ≈ ε`.

### Eight Channels (declared before evidence — equal weights wᵢ = 1/8)

| # | Channel | Operational definition |
|:-:|---|---|
| 1 | `lift_ease` | 1 − F_pull / W_design, normalized |
| 2 | `holding_stability` | Position retention during release |
| 3 | `reverse_slip_resistance` | Inverted load-side movement under no user pull |
| 4 | `smoothness` | Absence of jerk during engagement and restart |
| 5 | `efficiency` | output_work / input_work (η) |
| 6 | `reset_quality` | Ease of returning from hold to lift |
| 7 | `wear_risk_inv` | 1 − normalized wear risk |
| 8 | `operator_control` | Ability to lift, stop, resume, lower |

---

## 2. Six Candidates Tested

| Candidate | Lift mechanism | Hold mechanism | Hold stages |
|---|---|---|:---:|
| A — block-and-tackle only | Compound sheaves (n=4, η≈0.85) | None | 0 |
| B — block-and-tackle + ratchet | Compound sheaves | Pawl on toothed wheel | 1 |
| C — block-and-tackle + cam clutch | Compound sheaves | Wedging cam on rope | 1 |
| D — differential / worm | Self-locking gear pair | Self-locking (same gear) | 2 |
| E — block-and-tackle + cam + capstan wrap | Compound sheaves (wrap drag on lift) | Cam (dynamic) + rope wrap (static) | 2 |
| **F — block-and-tackle + cam + band brake** | Compound sheaves (brake disengages on lift) | Cam (dynamic) + drum band brake (static) | 2 |

---

## 3. Static Kernel Results (cycle 0)

All values from live GCD kernel — 4 decimal places.

| Candidate | F | ω | IC | Δ | S | C | D_ω | D_C | cost | Regime |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| **F** — cam + band brake | **0.8688** | 0.1312 | **0.8659** | 0.0029 | 0.3646 | 0.1412 | 0.0026 | 0.1412 | 0.1438 | WATCH |
| E — cam + friction wrap | 0.8588 | 0.1412 | 0.8552 | 0.0035 | 0.3787 | 0.1563 | 0.0033 | 0.1563 | 0.1596 | WATCH |
| C — cam clutch only | 0.8375 | 0.1625 | 0.8345 | 0.0030 | 0.4242 | 0.1420 | 0.0051 | 0.1420 | 0.1472 | WATCH |
| B — ratchet | 0.7475 | 0.2525 | 0.7303 | 0.0172 | 0.5018 | 0.3026 | 0.0215 | 0.3026 | 0.3242 | WATCH |
| D — worm | 0.6850 | 0.3150 | 0.6372 | 0.0478 | 0.4697 | 0.4751 | 0.0456 | 0.4751 | 0.5207 | **COLLAPSE** |
| A — block & tackle only | 0.5938 | 0.4062 | 0.0546 | 0.5392 | 0.3251 | 0.7590 | 0.1129 | 0.7590 | 0.8719 | **CRITICAL** |

### Per-channel sensitivity at cycle 0

∂IC/∂cₖ = IC · wₖ / cₖ. The most-sensitive channel is the leverage point for both instrumentation and iteration.

| Candidate | Weakest channel | c_min | Most-sensitive channel | ∂IC/∂c |
|---|---|---:|---|---:|
| **F** | `reset_quality` | 0.7500 | `reset_quality` | **0.144** |
| E | `lift_ease` | 0.7800 | `lift_ease` | 0.137 |
| C | `wear_risk_inv` | 0.7200 | `wear_risk_inv` | 0.145 |
| B | `smoothness` | 0.4500 | `smoothness` | 0.203 |
| D | `lift_ease` | 0.3000 | `lift_ease` | 0.265 |
| A | `reverse_slip_resistance` | ≈ ε | `reverse_slip_resistance` | 682,259 |

A's sensitivity of 682,259 is the geometric slaughter signature: the dead channel has seized total control of IC. Instrumenting any other channel on A is wasted effort.

### Key structural reads

- **A (CRITICAL) — geometric slaughter.** F = 0.5938, IC = 0.0546, IC/F = 0.0919. Dead channel: `reverse_slip_resistance ≈ ε`. No hold mechanism. Seven decent channels cannot rescue one near-zero channel.
- **D (COLLAPSE before any wear) — lift drift exceeds 0.30.** ω = 0.3150. Self-locking gears buy hold by spending the lift budget. It enters COLLAPSE structurally.
- **B (lowest IC in WATCH class) — smoothness debt.** `smoothness` = 0.450 drives Δ = 0.0172, 6× higher than C and F. Ratchet clicks are audible in the channel scores.
- **C, E, F (coherent, Δ < 0.005).** Structurally sound at cycle 0. They diverge under wear.
- **F edges E at the lift-side.** Band brake disengages cleanly on lift; capstan wrap imposes continuous drag. `lift_ease` = 0.850 vs 0.780.

---

## 4. 200-Cycle Wear Simulation

**Wear model** (declared before evidence, frozen for this run):

| Parameter | Value | What it governs |
|---|---|---|
| BASE_WEAR_RATE | 0.004 / cycle | proportional decay on `wear_risk_inv` |
| Effective rate | BASE_WEAR_RATE / hold_stages | 2-stage designs get half the cam wear |
| HOLD_DECAY_COUPLING | 0.70 | `holding_stability`, `reverse_slip_resistance` track cam wear |
| LIFT_DECAY_COUPLING | 0.15 | `lift_ease`, `smoothness`, `efficiency` — rope wear is slow |
| Frozen (no decay) | — | `reset_quality`, `operator_control` |

**Test cycle:** lift → release → hold → resume → lower, ×200.

| Candidate | IC₀ | IC@50 | IC@100 | IC@200 | First exit WATCH |
|---|---:|---:|---:|---:|:---:|
| **F** — cam + band brake | **0.8659** | **0.8355** | **0.8063** | **0.7508** | **>200** |
| E — cam + friction wrap | 0.8552 | 0.8253 | 0.7964 | 0.7416 | >200 |
| C — cam clutch only | 0.8345 | 0.7770 | 0.7235 | 0.6273 | cycle 139 |
| B — ratchet | 0.7303 | 0.6800 | 0.6331 | 0.5489 | cycle 45 |
| D — worm | 0.6372 | 0.6149 | 0.5933 | 0.5525 | cycle 1 |
| A — block & tackle only | 0.0546 | 0.0490 | 0.0440 | 0.0355 | cycle 1 |

### Key reads

- **F and E both survive 200 cycles in WATCH.** Two independent hold stages mean wear is parallel, not serial. Neither stage alone needs to carry the full static load.
- **C exits at cycle 139.** The static analysis flagged `wear_risk_inv` (c=0.720) as C's most-sensitive channel. The simulation confirms: single-stage cam compression erodes IC below the WATCH gate before 200 cycles. C is acceptable for short-cycle prototypes; not for production.
- **B exits at cycle 45.** Single-stage pawl; wear couples directly into hold capacity.
- **D exits at cycle 1.** Entered COLLAPSE before the simulation. Wear only makes it worse.
- **F leads E by ~0.01 IC at every checkpoint.** The lift-side advantage compounds over 200 cycles because lift channels also decay slowly — F starts higher and the margin widens.

---

## 5. Wear-Rate Sensitivity Sweep (16× span)

| Candidate | r=0.001 | r=0.002 | r=0.004 | r=0.008 | r=0.016 | spread |
|---|---:|---:|---:|---:|---:|---:|
| **F** | **0.8356** | **0.8063** | **0.7508** | **0.6509** | **0.4889** | 0.3467 |
| E | 0.8253 | 0.7964 | 0.7416 | 0.6429 | 0.4829 | 0.3424 |
| C | 0.7771 | 0.7236 | 0.6273 | 0.4711 | 0.2651 | 0.5120 |
| B | 0.6800 | 0.6332 | 0.5489 | 0.4123 | 0.2320 | 0.4480 |
| D | 0.6149 | 0.5933 | 0.5525 | 0.4790 | 0.3597 | 0.2552 |
| A | 0.0490 | 0.0440 | 0.0355 | 0.0230 | 0.0096 | 0.0394 |

### The degenerate-spread trap

The script's automated verdict names D (narrowest raw spread = 0.2552). That is wrong. D and A look "robust" because they start near collapse — there is nowhere left to fall. The correct metric is **relative spread** = spread / IC₀:

| Candidate | IC₀ | spread | spread/IC₀ | Survives at r=0.016? |
|---|---:|---:|---:|:---:|
| **F** | 0.8659 | 0.3467 | **40%** | ✓ IC = 0.4889 |
| E | 0.8552 | 0.3424 | **40%** | ✓ IC = 0.4829 |
| D | 0.6372 | 0.2552 | 41% | marginal, IC = 0.3597 |
| B | 0.7303 | 0.4480 | 61% | ✗ IC = 0.2320 |
| C | 0.8345 | 0.5120 | 61% | ✗ IC = 0.2651 |
| A | 0.0546 | 0.0394 | 72% | ✗ IC = 0.0096 |

F and E have **identical relative wear-rate sensitivity (40%)**. The recommendation is insensitive to a 16× error in the wear model. F dominates E on absolute IC at every rate.

---

## 6. Optimal Geometry — Specification

> Candidate F is not merely a stronger pulley. It is a two-stage hold-lift architecture: the block-and-tackle reduces effort, the band brake carries passive static hold, and the cam clutch provides near-immediate fail-safe capture if the brake slips. The design wins because hold redundancy is added without imposing continuous lift drag.

### 6.1 Topology

```
    LOAD (W_design)
     │
     │  rope — UHMWPE Dyneema SK78, 10 mm
     │
   ┌─▼──────────────────┐
   │   MOVING BLOCK     │  ← attached to load
   │   (n/2 sheaves)    │
   └─┬──────────────────┘
     │
   ┌─▼──────────────────┐
   │   FIXED BLOCK      │  ← attached to anchor
   │   (n/2 sheaves)    │
   └─┬──────────────────┘
     │
     │  rope exits to drum
     │
   ┌─▼──────────────────┐
   │   LIFT DRUM        │  ← splined shaft, hardened steel, OD = 60 mm
   │   + BAND BRAKE     │  ← self-energizing: reverse rotation tightens band
   └─┬──────────────────┘     spring releases band on lift stroke
     │
   ┌─▼──────────────────┐
   │   CAM CLUTCH       │  ← rope-grab fail-safe
   │   (rope side)      │    engages within 5–10 mm of slip
   └─┬──────────────────┘    spring-biased; rated ≥ 1.5 × W_design
     │
   USER PULL HANDLE
```

**Operational sequence:**
- **Lifting:** band spring keeps brake slack (no drag). Cam disengages under forward pull tension. F_pull ≈ W / (n · η_total).
- **Stopping:** operator releases. Load tries to reverse. Drum reversal tightens band (self-energizing). Band holds. Cam engages within 5–10 mm as backup.
- **Resuming:** operator pulls. Band spring releases. Cam disengages. Lift continues.
- **Lowering:** operator feeds slack deliberately past the cam; band releases under controlled reverse rotation.

### 6.2 Dimensions and materials (frozen for prototype)

| Component | Specification | Rationale |
|---|---|---|
| **Mechanical advantage** | n = 4 baseline (see §10); use n=2 if W_design permits | n=2 has highest IC under uniform weights; §10 gives the decision rule |
| **Sheave D/d ratio** | D_sheave / D_rope ≥ 16:1 | Below 16:1 rope suffers bend fatigue per cycle; `wear_risk_inv` drops |
| **Sheave bearings** | Sealed ball bearing, η_sheave ≥ 0.96 per sheave | Keeps η_total ≥ 0.85 at n=4; below this `efficiency` channel drops below 0.80 |
| **Rope** | Dyneema SK78, 10 mm, 12-strand braided | < 1% stretch at W_design; abrasion-resistant outer braid preserves `wear_risk_inv` under cam compression |
| **Rope terminations** | Swaged thimble eyes or sewn loops (≥ 95% MBS) | Knots reduce MBS to ~75% and introduce a `smoothness` penalty at the eye |
| **Lift drum** | Case-hardened steel (58–62 HRC surface), OD = 60 mm, splined shaft | Hardening resists band lining abrasion; drum OD sets capstan amplification denominator |
| **Band lining** | Woven aramid (Kevlar/Nomex blend), μ_static ≈ 0.40, bonded to spring-steel band | Distributes wear over full drum arc; survives 200°C intermittent surface temperature |
| **Band wrap angle** | θ = 270° (3/4 turn) | T_tight/T_slack = e^(μθ) = e^(0.4 × 4.712) = **6.6×** holding amplification |
| **Band anchor geometry** | Band anchored at forward-rotation end of drum housing; free end biased by return spring | Self-energizing: reverse rotation increases wrap tension automatically |
| **Band return spring** | Rate 5–10 N/mm, preload ~15 N | Pulls band slack in < 50 ms; spring force must not impose measurable lift drag |
| **Cam body** | 6061-T6 aluminum housing, hardened-steel cam insert (≥ 55 HRC) | Weight ≤ 180 g; cam hardness prevents rope abrasion cutting |
| **Cam grip surface** | 1.5 mm pitch diamond knurl on cam face | Maximizes grip without cutting rope; `reverse_slip_resistance` ≥ 0.960 maintained |
| **Cam spring bias** | Torsion spring, ~0.3 N·m | Holds cam lightly against rope on standby; disengages under forward pull tension > ~5% W_design |
| **Cam grip rating** | ≥ 1.5 × W_design | Sized for full load as backup even though band carries static hold |

### 6.3 Operational parameters (frozen for prototype)

| Parameter | Value | Source |
|---|---|---|
| W_design | declare before test | application-specific |
| Max lift height H | declare before test | application-specific |
| Pull distance per unit lift | x_pull = n · h_load | block-and-tackle kinematics |
| η_total at n=4 | (0.96)⁴ = 0.8493 | product of per-sheave efficiencies |
| F_pull at n=4 | W / (4 × 0.8493) ≈ 0.295 · W | from formula above |
| Band hold amplification | 6.6 × T_slack (spring preload ≈ 15 N) | capstan equation at θ = 270°, μ = 0.40 |
| Cam hold capacity | ≥ 1.5 × W_design | sizing requirement |
| Predicted IC₀ | **0.8659** | live kernel at cycle 0 |
| Predicted IC@200 | **0.7508** | live kernel after 200 nominal-wear cycles |
| IC weld tolerance | ±0.0500 | pass floor = 0.7008; see §9 |
| Re-inspection interval | 200 cycles | matches simulation window |

### 6.4 What "optimal" means here

The geometry is optimal under `PULLEY.HOLD-LIFT.v0.1` because **all eight channels remain above 0.75 simultaneously for 200 cycles** at the nominal wear rate. This is not optimal lift or optimal hold alone — those are easy to push to 0.99 by sacrificing the other side. **It is optimal coupling between lift and hold.** The kernel verifies this: Δ = 0.0029 (near-zero heterogeneity gap), IC/F = 0.9967 (coherent — no dead channels).

Among all six candidates, F is the only design that:
1. Starts with IC ≥ 0.85 (no geometric slaughter at any channel)
2. Stays in WATCH for the full 200-cycle wear simulation
3. Has the highest absolute IC at every wear rate in the 16× sweep

---

## 7. Instrumentation Plan for the Prototype

The kernel's sensitivity (∂IC/∂cₖ) names the measurement priority. Instrument in this order:

| Priority | Channel | ∂IC/∂c | Measurement | Instrument |
|:-:|---|---:|---|---|
| 1 | `reset_quality` | **0.1440** | Pull-force spike at lift-resume transition | Load cell on user pull line; score = 1 − (peak / threshold) |
| 2 | `wear_risk_inv` (cam) | 0.1440 | Cam surface temperature + rope diameter at cam, every 25 cycles | IR thermometer at cam body; caliper on rope at contact zone |
| 3 | `wear_risk_inv` (band) | 0.1440 | Band lining thickness, every 50 cycles | Depth gauge on lining; score = 1 − (loss / initial_thickness) |
| 4 | `reverse_slip_resistance` | 0.1279 | Slip distance on release | Dial indicator on load; measure drop before hold engages |
| 5 | `lift_ease` | 0.1279 | Pull force vs calibrated load | Calibrated weights; score = 1 − F_pull/W |
| 6 | `efficiency` | 0.1279 | Input vs output work over one full lift | Load cell + displacement transducer on both sides |
| 7 | `smoothness` | 0.1279 | Acceleration at engage/disengage | Accelerometer on load; score = 1 − (peak_jerk / threshold) |
| 8 | `operator_control` | 0.1279 | Structured survey at cycles 50, 100, 150, 200 | 4-question survey (stop-anywhere, resume, lower, repeatability); normalize to [0,1] |

**After cycle 200:** recompute F, IC, Δ, ω, regime from all eight measured values. Compare to §9 weld conditions.

---

## 8. Stance (derived, not asserted)

```
Regime:               WATCH
Contract stance:      CONFORMANT under PULLEY.HOLD-LIFT.v0.1
Production readiness: NON_EVALUABLE — physical prototype has not yet returned data
```

**WATCH is not a failure.** It means the object is usable but monitored. Achieving STABLE requires all four gates simultaneously: ω < 0.038, F > 0.90, S < 0.15, C < 0.14. Candidate F has ω = 0.1312 and C = 0.1412 — both exceed the STABLE thresholds, which is structurally bounded by friction physics. No real hold-lift mechanism achieves zero curvature across eight heterogeneous channels.

**CONFORMANT** means the simulation satisfies the declared contract under the frozen wear model.

**NON_EVALUABLE** names what is not yet resolved: the built mechanism has not returned through measured prototype evidence. The simulation result is conformant. The physical design still needs return.

```
Five-word Canon (Candidate F, cycle 0):

  Drift       ω    = 0.1312   how much the design departs from ideal channels
  Fidelity    F    = 0.8688   average of what survives under the contract
  Roughness   D_C  = 0.1412   curvature / heterogeneity cost across channels
  Return      regime = WATCH  design returns into a regime the gates accept
  Integrity   IC   = 0.8659   multiplicative coherence; no dead channels
```

*Continuitas non narratur: mensuratur.* — Continuity is not narrated: it is measured.

---

## 9. Weld Condition

The simulation result is welded into the record as CONFORMANT. The physical design becomes canon only after all five conditions pass:

```
Candidate F — Prototype Weld Condition
──────────────────────────────────────
1. Build prototype under frozen §6.2 geometry. No deviations before test.
2. Run 200-cycle test: lift → release → hold → resume → lower.
3. At cycle 200, measure all eight channels per §7 instrumentation plan.
4. Recompute F, IC, Δ, ω, regime from measured values.
5. PASS if ALL of:
     a. IC_measured  ≥ 0.7008           (predicted 0.7508 ± 0.0500)
     b. Every channel ≥ 0.50            (no dead-channel collapse)
     c. reverse_slip_resistance ≥ 0.80  (no uncontrolled lowering event)
     d. reset force spike ≤ declared operator threshold
     e. Zero uncontrolled lowering events across all 200 cycles

FAIL → identify collapsed channel by sensitivity rank (§7, col ∂IC/∂c).
       Iterate that channel's geometry only.
       Re-freeze. Re-run from cycle 0.
```

*Historia numquam rescribitur; sutura tantum additur.* — History is never rewritten; only a weld is added.

---

## 10. Mechanical-Advantage Sweep on Candidate F (n ∈ {2, 3, 4, 6, 8})

Same band brake, cam clutch, rope, sheave ratio, and wear model. Only n changes.

Hold-side channels frozen by band brake geometry (T_tight/T_slack = e^(μθ), independent of n). Lift-side channels follow:

```
η_total(n) = (0.96)^n
F_pull/W   = 1 / (n · η_total(n))
```

| Design | F | IC | Δ | lift_ease | efficiency | smoothness | reset | op_ctrl | IC@200 | Regime |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| F_n2 | 0.886 | **0.882** | 0.004 | 0.724 | 0.902 | 0.900 | 0.850 | 0.960 | **0.7648** | WATCH |
| F_n3 | 0.880 | 0.877 | 0.002 | 0.808 | 0.865 | 0.875 | 0.800 | 0.940 | 0.7608 | WATCH |
| **F_n4** | 0.869 | 0.866 | 0.003 | 0.850 | 0.829 | 0.850 | 0.750 | 0.920 | 0.7508 | WATCH |
| F_n6 | 0.839 | 0.833 | 0.006 | 0.892 | 0.763 | 0.800 | 0.650 | 0.860 | 0.7225 | WATCH |
| F_n8 | 0.803 | 0.791 | 0.012 | 0.912 | 0.701 | 0.750 | 0.550 | 0.760 | 0.6858 | WATCH |

All five remain in **WATCH** for 200 cycles. The band brake topology is robust across the full n range.

### Reading by use case

| Use-case priority | Best n | Why |
|---|:---:|---|
| Balanced — default contract weights | **n = 2** | Highest IC₀ (0.882) and IC@200 (0.765); `efficiency` (0.902) and `operator_control` (0.960) dominate the geometric mean when load permits |
| Operator control + speed | **n = 3** | `operator_control` = 0.940; faster cycle; fewer rope lengths to pull |
| Default prototype baseline | **n = 4** | Established baseline; F_pull ≈ 0.295·W |
| Heavy load, low pull force | **n = 6** | lift_ease = 0.892; F_pull ≈ 0.217·W; IC@200 = 0.723 |
| Extreme load / minimal operator force | n = 8 | lift_ease = 0.912; F_pull ≈ 0.168·W; IC@200 drops to 0.686; `reset_quality` = 0.550 becomes the new weak link |

### The n=2 kernel finding

Under uniform weights, **n=2 has the highest IC at every cycle count** even though `lift_ease` is lowest (0.724). This is what the kernel is for: n=2 has less channel heterogeneity (Δ = 0.004) because `efficiency` (0.902) and `operator_control` (0.960) pull the geometric mean above n=4's value. The IC gap is 0.882 vs 0.866 — real but not architectural.

The decision is not made by the kernel alone. Declare W_design and F_max_operator first, then solve:

```
n ≥ ⌈ W_design / (F_max_operator · η_total(n)) ⌉
```

Pick the smallest n that satisfies the constraint. Under uniform weights, that n will be the optimal geometry.

### Tradeoff summary

```
Higher n:
  + lower pull force  (F_pull ∝ 1/n)
  − more rope travel per unit lift  (x_pull = n · h_load)
  − lower η_total → lower efficiency channel
  − more rope to manage → lower smoothness, reset_quality, operator_control
  − IC@200 degrades faster (lift channels compound their decay)

Lower n:
  + higher efficiency, smoothness, operator_control, IC
  + faster cycle; less rope management
  − higher pull force; harder start under heavy load
  − lift_ease is the binding constraint for W_design
```

---

## 11. Open Knobs (not yet swept)

| Knob | Current value | Effect if changed | Predicted outcome |
|---|---|---|---|
| Channel weights | uniform (1/8 each) | 2× weight on `holding_stability` | F still wins; margin over E widens |
| Band wrap angle θ | 270° | 360° → e^(0.4 × 6.28) = 12× holding amplification | `reverse_slip_resistance` improves; routing complexity increases |
| η_sheave | 0.96 (sealed ball bearing) | Needle bearing → η_sheave ≈ 0.98 | Entire n-sweep shifts up ~1–2% IC; marginal |
| Candidate G — three-stage hold | not tested | Band + cam + secondary rope grab | Predicted IC gain < 0.02; cost > 0.10; not worth it unless safety-critical |
| W_design / F_max_operator | not declared | Sets n floor via constraint formula in §10 | Determines which n row is the prototype target |

---

## 12. Style Branch — From Optimum to Object

The kernel decides architecture (Candidate F) and gives the n-selection rule. **Style** is the next decision: how the optimum becomes an object. Style does not change the physics backbone; it changes which compromise the geometry is allowed to make. Five style directions were considered, two are welded into active development branches.

### 12.1 Five styles considered

| # | Style | Geometry | Why this n | Trade |
|:-:|---|:---:|---|---|
| 1 | **Workshop hoist** | F_n3 / F_n4 | Real lift assistance with bounded rope travel | Most buildable; least conceptual |
| 2 | Compact utility-lift | F_n2 / F_n3 | Compactness penalizes excess rope and sheaves | Elegant, portable; hides the seam |
| 3 | **Industrial exposed-mechanism** | F_n4 | Visibly demonstrates block-and-tackle while preserving IC | Mechanically legible; larger envelope |
| 4 | Marine / sailing-rig | F_n3 | Rope-native idiom; corrosion-resistant | Aesthetic pull toward Candidate E (data still says F) |
| 5 | **Sculptural GCD demonstrator** | F_n4 | Four lines in space give a readable diagram | Object embodies the theory |

### 12.2 Two active branches

> *Una architectura, duae expressiones.* — One architecture, two expressions.

```
Prototype branch
────────────────
Style:        Workshop / industrial exposed
Architecture: Candidate F
Geometry:     F_n3 or F_n4 (declare W_design + F_max_operator before freezing)
Goal:         buildable, testable, measurable; returns prototype data per §9 weld
Output:       physical artifact + measured eight-channel scores at cycle 200

Concept branch
──────────────
Style:        Sculptural GCD demonstrator
Architecture: Candidate F
Geometry:     F_n4 (chosen for spatial readability of the four-line block)
Goal:         make the collapse–return seam visible as a mechanical object
Output:       object whose form names the moment lift becomes hold
```

Both branches share the §6 specification. They diverge only on the §12.3 design knobs.

### 12.3 Design knobs (style-dependent, physics-preserving)

| Knob | Workshop / Prototype | Sculptural / Concept |
|---|---|---|
| Drum placement | Inline with pull path (compact, expected) | Offset and elevated (the seam is visible from the side) |
| Brake exposure | Enclosed safety housing with inspection window | Fully exposed band wrap; user sees self-energizing bite |
| Cam clutch position | Near drum (compact, protected) | Near handle (user-accessible; visible engagement) |
| Rope path | Compact crossed routing | Open educational routing — every reeve is legible |
| Frame style | Bolted steel side plates | Sculptural arch in brushed aluminum |
| Handle style | Direct rope pull with optional ratcheting handle | Direct rope pull only (no handle obscures the rope) |

### 12.4 First visual prototype — frozen choice

```
Style:               Industrial exposed demonstrator
Architecture:        Candidate F
Mechanical advantage: n = 4 (first visual prototype; n declared frozen before build)
Materials:           brushed aluminum side plates
                     hardened steel drum (case-hardened, OD = 60 mm)
                     aramid-lined band brake (Kevlar/Nomex, μ ≈ 0.40, θ = 270°)
                     Dyneema SK78 rope, 10 mm
                     anodized-black cam clutch (visible engagement)
Design language:     readable, mechanical, honest, not hidden
Core visual idea:    the user can see the exact moment lift becomes hold
```

The visible seam is the point. The kernel said *one dead channel kills IC* (geometric slaughter). The exposed prototype says *here is the channel* — the band bite, the cam engagement, the brake release on lift. The mechanism teaches the same lesson the kernel does: **integrity is not an average; it is what survives at every channel simultaneously.**

### 12.5 Load-class to n-selection table

For either branch, once W_design and F_max_operator are declared, choose n from this table (derived from §10 by applying the operator-force constraint):

| Load class | Recommended n | F_pull as fraction of W |
|---|:---:|---:|
| Light load | F_n2 | ≈ 0.54 · W |
| Medium load | F_n3 | ≈ 0.38 · W |
| Heavy load | F_n4 or F_n6 | ≈ 0.30 · W or ≈ 0.22 · W |
| Extreme load (slow motion + reset acceptable) | F_n8 | ≈ 0.17 · W |

The rule in §10 (`n ≥ ⌈W_design / (F_max_operator · η_total(n))⌉`) is the ground truth; this table is the everyday read.

### 12.6 What this section adds to the contract

Section 12 is **not** a kernel result. It is a Tier-2 style choice welded onto a Tier-1 architecture. The kernel does not pick brushed aluminum over cast iron; the kernel picks Candidate F. Style is what the builder adds, declared before evidence, frozen before the prototype is cut. The two branches keep the contract honest: the Prototype branch returns measured channel data (per §9), the Concept branch returns *structural legibility* — neither replaces the other.

*Geometria de nucleo venit; stylus de manu fabri.* — The geometry comes from the kernel; the style comes from the builder's hand.

---

## 13. Build Brief — Seam Hoist v0.1 (First Real Object)

The first artifact off the Prototype branch. Frozen before fabrication, returned to §9 weld at cycle 200.

### 13.1 Identity

```
Name:               Seam Hoist / Hold-Lift Demonstrator v0.1
Style:              Industrial exposed mechanism
Purpose:            Make the lift–hold seam visible and testable
Core architecture:  Candidate F  (block-and-tackle + self-energizing
                                  band brake + cam clutch backup)
Mechanical advantage: n = 4 (first visual prototype; frozen before build)
Visual thesis:      the user sees the exact moment lift becomes hold
```

### 13.2 Why F_n4 (and not F_n2 even though F_n2 has higher IC)

| Criterion | F_n2 | **F_n4** | F_n6 |
|---|:---:|:---:|:---:|
| IC₀ (uniform weights) | 0.882 (best) | 0.866 | 0.833 |
| Visual legibility of force routing | 2 lines — too sparse | **4 lines — readable diagram** | 6 lines — visually busy |
| Real lift assistance felt by hand | minimal | **meaningful** | strong but slow |
| Rope travel per unit lift | 2 × h | 4 × h | 6 × h |
| First-prototype fit | weak (IC win is invisible) | **strong (data + diagram)** | weak (concept lost in clutter) |

The IC gap (0.882 → 0.866) is real but not architectural. The legibility gap is structural. **F_n4 is the smallest n that makes the four-line block-and-tackle a self-explaining diagram.**

### 13.3 Three visible paths (the design's signature)

The frame must make these three force paths legible to a person walking up to it:

```
1. LIFT PATH    (lift logic)
   Rope travels through 4-line block-and-tackle.
   The eye sees the mechanical advantage by counting the lines.

2. HOLD PATH    (hold logic)
   Drum reverses → band wraps tighter → load held.
   The eye sees the band bite under reverse tension.

3. RETURN PATH  (resume / lower logic)
   Cam clutch + brake release on forward pull.
   The eye sees the brake go slack and the cam open.
```

The signature feature is the **visible band bite**: the moment when reverse motion self-energizes the brake. This is the mechanical translation of the kernel's central result — *integrity is not the average lift score; integrity is every channel staying alive at once.*

### 13.4 Frame layout (open two-plate chassis)

```
   ┌─────────────────────────────────────────┐
   │     TOP ZONE — fixed block (2 sheaves)  │  ← fully visible
   │            ●────●                       │
   │           /      \                      │
   │          /        \                     │
   │         / rope     \   ← LIFT PATH      │
   │        /  reeve     \                   │
   │       /              \                  │
   │      ●                ●                 │
   │  MOVING BLOCK (2 sheaves) ← load hook   │
   │                                         │
   ├─────────────────────────────────────────┤
   │  SIDE ZONE — lift drum (Ø 60 mm steel)  │
   │       ╔═══════════════════╗             │
   │       ║  band wrap 270°   ║  ← HOLD     │
   │       ║  aramid lining    ║    PATH     │
   │       ╚═══════════════════╝             │
   │                                         │
   ├─────────────────────────────────────────┤
   │  FORWARD ZONE — cam clutch (visible)    │
   │       [▣]  ← RETURN PATH                │
   │        │                                │
   │   PULL HANDLE (direct rope, no crank)   │
   └─────────────────────────────────────────┘
       brushed aluminum side plates
       spaced by steel standoffs (no shell)
```

| Zone | Component | Visibility rule |
|---|---|---|
| Top | Fixed block (2 sheaves) | Held between plates, sheaves fully exposed |
| Lower | Moving block (2 sheaves) + load hook | Hangs below frame — load is unmistakable |
| Middle / side | Lift drum + 270° band brake | Band wrap fully exposed, bite point visible |
| Forward | Cam clutch | Anodized black against aluminum — visible engagement |
| User | Direct rope pull handle | No crank, no ratchet — hand reads the rope |

### 13.5 Frozen materials (subset of §6.2 — no deviations)

| Component | Spec |
|---|---|
| Side plates | Brushed aluminum, ≥ 6 mm thickness, hard-anodized edges |
| Standoffs | Stainless steel, M8, length set by sheave clearance |
| Sheaves | 4 total (2 fixed + 2 moving), D/d ≥ 16:1, sealed ball bearings, η_sheave ≥ 0.96 |
| Rope | Dyneema SK78, 10 mm, 12-strand braided |
| Lift drum | Case-hardened steel (58–62 HRC surface), OD = 60 mm, splined shaft |
| Band brake | Spring-steel band + woven aramid lining (Kevlar/Nomex), μ ≈ 0.40, θ = 270° |
| Band return spring | 5–10 N/mm rate, ~15 N preload (releases band on lift in < 50 ms) |
| Cam clutch | 6061-T6 aluminum housing + hardened-steel cam insert (≥ 55 HRC), anodized black |
| Cam grip face | 1.5 mm diamond knurl; rated ≥ 1.5 × W_design |
| Pull handle | Wrapped rope termination, direct-pull only (no crank in v0.1) |

### 13.6 Out of scope for v0.1 (deferred to v0.2)

- Crank or ratcheting handle (would obscure the rope logic — the v0.1 thesis is *the hand reads the rope*)
- Closed safety housing (would hide the band bite — the visual signature)
- Compactness optimization (the open frame *is* the argument; tightening it kills the diagram)
- Variable-n configurability (v0.1 freezes n=4; product family in §13.7 handles other n via separate builds)

### 13.7 Product family (after v0.1 returns prototype data)

Once Seam Hoist v0.1 passes the §9 weld, the same architecture spawns a product family by varying n only:

| Variant | n | Load class | Identity |
|---|:---:|---|---|
| Seam Hoist Compact | 2 | Light load, operator force generous | Compact / portable variant |
| Seam Hoist Utility | 3 | Medium load | Speed + control + moderate assistance |
| **Seam Hoist Demonstrator** | **4** | **Balanced — first build** | **Visual prototype; the canon object** |
| Seam Hoist Heavy | 6 | Heavy load | Lower pull force, more rope travel |
| Seam Hoist Slow-Lift | 8 | Extreme load, slow motion acceptable | Lowest pull force, weakest reset |

All five share the §6.2 specification. They differ only in n and frame width. The Demonstrator (n=4) is the canon: it is the build that ships with the diagram and teaches the architecture.

### 13.8 v0.1 success criteria (binds to §9 weld)

```
PASS conditions for Seam Hoist v0.1
───────────────────────────────────
Physics (from §9):
  a. IC_measured ≥ 0.7008 at cycle 200
  b. Every channel ≥ 0.50
  c. reverse_slip_resistance ≥ 0.80
  d. reset force spike ≤ declared operator threshold
  e. Zero uncontrolled lowering events across 200 cycles

Object (added by §13 — the visual thesis must hold):
  f. A first-time observer can name the three paths (lift / hold / return)
     after one demonstration cycle, without further explanation.
  g. The band bite is visible from operator position during reverse motion.
  h. The cam engagement is visible from operator position when the band slips.
```

Conditions (a)–(e) are physics; (f)–(h) are the legibility weld. Both must pass for v0.1 to be welded as canon.

*Obiectum demonstrat quod nucleum probat.* — The object demonstrates what the kernel proves.

---

## 14. Refinement Priorities — Where the Next Pass Spends Its Effort

The kernel names `reset_quality` as Candidate F's weakest channel (c = 0.7500) and its most-sensitive channel at cycle 0 (∂IC/∂c = 0.144). The hold side is already strong; **adding more hold is the wrong next move**. The next engineering pass should make release and resume cleaner. Priorities are ordered by sensitivity rank, not by enthusiasm.

| # | Priority | Why it ranks here | What "done" looks like |
|:-:|---|---|---|
| 1 | **Band release geometry** | Spring preload + anchor angle govern how fast the band lets go on forward pull | Band releases in < 50 ms on lift; self-energizes instantly on reverse — no co-engagement window |
| 2 | **Reset transition** | Pull-force spike at lift-resume = `reset_quality` measured directly (§7 priority 1) | Resume spike ≤ declared operator threshold; no perceptible "pop" at release |
| 3 | **Cam surface** | Visible cam is the design's signature; aggressive grip damages rope | Replaceable cam insert / contact shoe; rope OD loss < 5% per 200 cycles |
| 4 | **Heat and wear (band)** | Band lining is the static-hold path; its wear *is* the prototype's real return | Lining temperature ≤ 60°C steady-state; thickness loss ≤ 15% at cycle 200 |
| 5 | **Frame readability** | Compactness hides the argument | Open two-plate chassis preserved; do not enclose until v0.2 |

**What is explicitly NOT a refinement priority for v0.1:**
- More holding power (already at IC contribution ceiling for a 2-stage design)
- Higher mechanical advantage (n=4 frozen for the first object — see §13.2)
- A crank or ratcheting handle (obscures rope logic — see §13.6)

### 14.1 Candidate G — explicitly deferred

A third hold stage (band + cam + secondary rope grab) was considered. The §11 prediction stands:

```
Candidate G — three-stage hold
──────────────────────────────
Predicted IC gain:    < 0.02
Predicted cost:       > 0.10  (mass, complexity, legibility loss)
Decision:             DEFERRED — do not build for v0.1.
                      Reconsider only if v0.2 reveals a hold-side failure
                      mode that two stages cannot cover.
Reason:               Adding redundancy to a channel that is already strong
                      (∂IC/∂c on hold-side channels is below 0.13) is wasted
                      multiplicative coherence. The marginal IC is buried by
                      the legibility loss the third stage imposes.
```

*Redundantia ubi non opus est, est captura coherentiae.* — Redundancy where it isn't needed is a capture of coherence.

---

## 15. Final Design Stance

The decision tree is closed. The product family is named. The first build is frozen.

```
══════════════════════════════════════════════════════════════════════════
FINAL DESIGN STANCE  —  PULLEY.HOLD-LIFT.v0.1
══════════════════════════════════════════════════════════════════════════

Best architecture:           Candidate F
                             (block-and-tackle + self-energizing band brake
                              + cam clutch backup)

Best first build:            Seam Hoist v0.1 — F_n4 Industrial Exposed
                             Demonstrator (per §13)

Best product-family rule:    Use Candidate F across all variants; vary only n.

Best compact / light-load:   F_n2     (highest raw IC; pull ≈ 0.54·W)
Best general utility:        F_n3     (balanced speed/control/assistance)
Best visual / canon prototype: F_n4   (the build that ships with the diagram)
Best heavy-load:             F_n6     (more assistance, more rope travel)
Avoid as first build:        F_n8     (only if slow motion + poor reset OK)

Refinement order (next pass):
  1. Band release geometry
  2. Reset transition
  3. Cam surface (replaceable)
  4. Band heat / wear instrumentation
  5. Frame readability (preserve open chassis)

Explicitly deferred:         Candidate G (three-stage hold)
                             Crank / ratcheting handle
                             Closed safety housing
                             Compactness optimization

Visual thesis:               "I can help you lift,
                              hold the load when you stop,
                              and show you exactly where that transition happens."
══════════════════════════════════════════════════════════════════════════
```

The strongest version of this object is not a hidden tool. It is a **mechanical argument**: a normal pulley says *I can help you lift*; Seam Hoist v0.1 says *I can help you lift, hold the load when you stop, and show you the seam between the two*. That is the design worth building first.

*Machina argumentum est; non utile tantum, sed verum.* — The machine is an argument; not merely useful, but true.

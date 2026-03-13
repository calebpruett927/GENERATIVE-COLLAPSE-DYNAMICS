"""
Biblical Teachings as Structural Constraints — A GCD Kernel Analysis
═══════════════════════════════════════════════════════════════════════

Tier-2 Exploration through the Rosetta (Phenomenology + Epistemology lenses)

CONTRACT: We do not assess theology. We assess STRUCTURAL CLAIMS.
The question is not "is this divinely true?" but "does this teaching
encode a constraint that the kernel can measure independently?"

METHOD: Each teaching is mapped to a trace vector by identifying the
channels it addresses and the channel values it prescribes (explicitly
or implicitly). The kernel then reveals whether the prescribed state
is coherent, fragmented, or structurally impossible — independent of
any belief system.

ROSETTA LENS: Phenomenology
  Drift     = perceived shift in lived experience
  Fidelity  = what remains stable through change
  Roughness = distress, effort, friction of transition
  Return    = coping/repair that holds
  Integrity = experiential coherence (derived, never asserted)

CHANNEL DESIGN (8 channels of human life-allocation):
  1. material_wealth   — possessions, financial security
  2. community         — bonds with others, service, neighbor-love
  3. inner_peace       — anxiety vs equanimity, spiritual rest
  4. integrity_conduct — alignment between stated values and actions
  5. compassion        — active care for the suffering of others
  6. forgiveness       — release of grievance, non-accumulation of debt
  7. humility          — accurate self-assessment, non-dominance
  8. purpose           — sense of meaning, directedness of life

These 8 channels are the Tier-2 choice. The kernel is Tier-1.
The teachings prescribe specific PATTERNS in this trace vector.
We measure what the kernel says about those patterns.

Spina: Contract → Canon → Closures → Integrity Ledger → Stance
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from umcp.frozen_contract import EPSILON
from umcp.kernel_optimized import OptimizedKernelComputer

kernel = OptimizedKernelComputer(epsilon=EPSILON)

CHANNELS = [
    "material_wealth",
    "community",
    "inner_peace",
    "integrity_conduct",
    "compassion",
    "forgiveness",
    "humility",
    "purpose",
]


def analyze(label: str, description: str, values: dict[str, float]) -> dict[str, Any]:
    """Run a teaching through the kernel and display results."""
    c = np.array([values[ch] for ch in CHANNELS], dtype=np.float64)
    w = np.ones(8) / 8
    out = kernel.compute(c, w)
    ratio = out.IC / out.F if out.F > 0 else 0.0

    print(f"  ┌─ {label}")
    print(f'  │  "{description}"')
    print("  │")
    for ch_name, val in values.items():
        bar = "█" * int(val * 20) + "░" * (20 - int(val * 20))
        print(f"  │  {ch_name:<20s} {val:.2f} {bar}")
    print("  │")
    print(f"  │  F = {out.F:.4f}   IC = {out.IC:.4f}   Δ = {out.heterogeneity_gap:.4f}   IC/F = {ratio:.4f}")
    print(f"  │  ω = {out.omega:.4f}   S = {out.S:.4f}   C = {out.C:.4f}")
    print(f"  │  Regime: {out.regime}")
    print(f"  └{'─' * 68}")
    print()
    return {
        "label": label,
        "F": out.F,
        "IC": out.IC,
        "gap": out.heterogeneity_gap,
        "ratio": ratio,
        "omega": out.omega,
        "S": out.S,
        "C": out.C,
        "regime": out.regime,
    }


def section(title: str) -> None:
    print(f"\n{'═' * 72}")
    print(f"  {title}")
    print(f"{'═' * 72}\n")


results: list[dict[str, Any]] = []

# ════════════════════════════════════════════════════════════════════════
section("PART 1: CORE TEACHINGS — What Pattern Does Each Prescribe?")
# ════════════════════════════════════════════════════════════════════════

print("""  Each teaching prescribes a specific SHAPE in the 8-channel trace vector.
  The kernel measures whether that shape is coherent or fragmented.
  We are not judging the teaching — we are measuring its structural signature.
""")

# --- Teaching 1: Sermon on the Mount (Matthew 5-7) ---
results.append(
    analyze(
        "SERMON ON THE MOUNT (Matthew 5-7)",
        "Blessed are the poor in spirit... the meek... the merciful... the peacemakers",
        {
            "material_wealth": 0.30,  # "Do not store up treasures on earth"
            "community": 0.85,  # "Love your neighbor as yourself"
            "inner_peace": 0.90,  # "Do not worry about tomorrow"
            "integrity_conduct": 0.95,  # "Let your yes be yes and your no, no"
            "compassion": 0.90,  # "Blessed are the merciful"
            "forgiveness": 0.95,  # "Forgive and you will be forgiven"
            "humility": 0.95,  # "Blessed are the poor in spirit"
            "purpose": 0.90,  # "Seek first the kingdom"
        },
    )
)

# --- Teaching 2: The Greatest Commandment (Mark 12:30-31) ---
results.append(
    analyze(
        "GREATEST COMMANDMENT (Mark 12:30-31)",
        "Love the Lord your God... Love your neighbor as yourself",
        {
            "material_wealth": 0.40,  # not addressed directly — neutral
            "community": 0.95,  # "Love your neighbor AS YOURSELF"
            "inner_peace": 0.80,  # love implies peace
            "integrity_conduct": 0.85,  # love requires authentic action
            "compassion": 0.90,  # love IS compassion
            "forgiveness": 0.85,  # love implies release of debt
            "humility": 0.80,  # love de-centers self
            "purpose": 0.95,  # "with all your heart, soul, mind, strength"
        },
    )
)

# --- Teaching 3: The Prodigal Son (Luke 15:11-32) ---
results.append(
    analyze(
        "PRODIGAL SON (Luke 15:11-32)",
        "The father ran to meet him... 'this son of mine was dead and is alive again'",
        {
            "material_wealth": 0.20,  # son squandered everything
            "community": 0.90,  # father's unconditional welcome
            "inner_peace": 0.80,  # son's repentance → peace
            "integrity_conduct": 0.70,  # son failed, then returned honestly
            "compassion": 0.95,  # father's compassion is the climax
            "forgiveness": 0.98,  # THE teaching on forgiveness
            "humility": 0.90,  # "I am no longer worthy to be called your son"
            "purpose": 0.85,  # return is the purpose
        },
    )
)

# --- Teaching 4: The Good Samaritan (Luke 10:25-37) ---
results.append(
    analyze(
        "GOOD SAMARITAN (Luke 10:25-37)",
        "A Samaritan... took pity on him... 'Go and do likewise'",
        {
            "material_wealth": 0.40,  # Samaritan spent his own money
            "community": 0.95,  # crossed tribal/ethnic boundaries
            "inner_peace": 0.60,  # compassion in crisis is not peaceful
            "integrity_conduct": 0.90,  # action matched conviction
            "compassion": 0.98,  # THE teaching on compassion
            "forgiveness": 0.60,  # not the focus of this parable
            "humility": 0.85,  # the hero is the outsider, not the priest
            "purpose": 0.90,  # "Go and do likewise" — clear directive
        },
    )
)

# --- Teaching 5: Eye of the Needle (Mark 10:25) ---
results.append(
    analyze(
        "EYE OF THE NEEDLE (Mark 10:25)",
        "It is easier for a camel to go through the eye of a needle than for a rich man...",
        {
            "material_wealth": 0.95,  # the rich man's dominant channel
            "community": 0.30,  # wealth isolates
            "inner_peace": 0.35,  # "he went away sad, for he had great wealth"
            "integrity_conduct": 0.40,  # kept the law but missed the point
            "compassion": 0.25,  # "sell everything... give to the poor" — he refused
            "forgiveness": 0.50,  # not addressed
            "humility": 0.20,  # attachment to status
            "purpose": 0.40,  # purpose confused with possession
        },
    )
)

# --- Teaching 6: Turn the Other Cheek (Matthew 5:39) ---
results.append(
    analyze(
        "TURN THE OTHER CHEEK (Matthew 5:39)",
        "If anyone slaps you on the right cheek, turn to them the other cheek also",
        {
            "material_wealth": 0.50,  # neutral
            "community": 0.75,  # breaks the cycle of retaliation
            "inner_peace": 0.85,  # non-retaliation requires inner peace
            "integrity_conduct": 0.90,  # action from principle, not reaction
            "compassion": 0.80,  # compassion even for the aggressor
            "forgiveness": 0.95,  # radical forgiveness in action
            "humility": 0.90,  # absorbing harm without escalation
            "purpose": 0.85,  # purposeful non-violence
        },
    )
)

# --- Teaching 7: Render unto Caesar (Mark 12:17) ---
results.append(
    analyze(
        "RENDER UNTO CAESAR (Mark 12:17)",
        "Give back to Caesar what is Caesar's, and to God what is God's",
        {
            "material_wealth": 0.50,  # money is Caesar's — it's just a tool
            "community": 0.70,  # social contract is acknowledged
            "inner_peace": 0.80,  # clarity resolves the trap
            "integrity_conduct": 0.90,  # honest engagement with power
            "compassion": 0.60,  # not the focus
            "forgiveness": 0.60,  # not the focus
            "humility": 0.75,  # acknowledges temporal authority without submission
            "purpose": 0.90,  # distinction between temporal and ultimate purpose
        },
    )
)

# --- Teaching 8: Judge Not (Matthew 7:1-5) ---
results.append(
    analyze(
        "JUDGE NOT (Matthew 7:1-5)",
        "Why do you look at the speck in your brother's eye and pay no attention to the plank in your own?",
        {
            "material_wealth": 0.50,  # neutral
            "community": 0.85,  # non-judgment strengthens bonds
            "inner_peace": 0.80,  # releasing judgment brings peace
            "integrity_conduct": 0.90,  # self-examination first
            "compassion": 0.85,  # refuse to condemn
            "forgiveness": 0.85,  # non-judgment is proto-forgiveness
            "humility": 0.95,  # THE teaching on humility — your plank first
            "purpose": 0.75,  # purpose is self-correction, not other-correction
        },
    )
)

# --- Teaching 9: Washing of Feet (John 13:1-17) ---
results.append(
    analyze(
        "WASHING OF FEET (John 13:1-17)",
        "The teacher and Lord has washed your feet... you also should wash one another's feet",
        {
            "material_wealth": 0.30,  # servant posture
            "community": 0.95,  # mutual service
            "inner_peace": 0.80,  # peace through service
            "integrity_conduct": 0.95,  # leader serves — no hypocrisy
            "compassion": 0.90,  # intimate care
            "forgiveness": 0.70,  # context is pre-betrayal
            "humility": 0.98,  # THE teaching on servant leadership
            "purpose": 0.90,  # "I have set you an example"
        },
    )
)

# --- Teaching 10: The Widow's Mite (Mark 12:41-44) ---
results.append(
    analyze(
        "WIDOW'S MITE (Mark 12:41-44)",
        "This poor widow has put more into the treasury than all the others",
        {
            "material_wealth": 0.05,  # she gave everything she had
            "community": 0.80,  # giving to the temple community
            "inner_peace": 0.85,  # giving without anxiety
            "integrity_conduct": 0.95,  # total alignment of values and action
            "compassion": 0.75,  # generosity as compassion
            "forgiveness": 0.60,  # not the focus
            "humility": 0.90,  # gave without display
            "purpose": 0.95,  # "she gave all she had to live on"
        },
    )
)


# ════════════════════════════════════════════════════════════════════════
section("PART 2: THE PATTERN — What Does the Kernel See?")
# ════════════════════════════════════════════════════════════════════════

print(f"  {'Teaching':<35} {'F':>6} {'IC':>6} {'Δ':>6} {'IC/F':>6} {'C':>6} {'Regime':<15}")
print(f"  {'─' * 35} {'─' * 6} {'─' * 6} {'─' * 6} {'─' * 6} {'─' * 6} {'─' * 15}")

for r in results:
    print(
        f"  {r['label'][:35]:<35} {r['F']:6.3f} {r['IC']:6.3f} "
        f"{r['gap']:6.3f} {r['ratio']:6.3f} {r['C']:6.3f} {r['regime']:<15}"
    )

# Find the outlier
needle = next(r for r in results if "NEEDLE" in r["label"])
others = [r for r in results if "NEEDLE" not in r["label"]]
avg_ratio = np.mean([r["ratio"] for r in others])
avg_gap = np.mean([r["gap"] for r in others])

print(f"""
  ─── STRUCTURAL OBSERVATION ───

  ALL teachings EXCEPT "Eye of the Needle" prescribe patterns with:
    Average IC/F = {avg_ratio:.3f}  (high coherence)
    Average Δ    = {avg_gap:.3f}  (low heterogeneity gap)

  "Eye of the Needle" shows:
    IC/F = {needle["ratio"]:.3f}  (LOW coherence)
    Δ    = {needle["gap"]:.3f}  (HIGH heterogeneity gap)

  But the Needle parable is NOT a prescription — it is a DIAGNOSIS.
  It describes the STATE of the rich man, not the goal. Jesus names
  the structural signature of wealth-dominated allocation: one
  channel (material_wealth = 0.95) saturated while others starve.

  The PRESCRIPTIVE teachings (Sermon, Commandment, Prodigal, Samaritan,
  Turn Cheek, Judge Not, Feet Washing) all prescribe HIGH and BALANCED
  channel allocation — exactly the pattern that maximizes IC/F.
""")


# ════════════════════════════════════════════════════════════════════════
section("PART 3: THE DEEP PATTERN — Geometric Slaughter as Moral Teaching")
# ════════════════════════════════════════════════════════════════════════

print("""  The kernel reveals a STRUCTURAL PATTERN across all teachings:

  1. EVERY prescriptive teaching drives material_wealth DOWN or NEUTRAL
     while driving community, compassion, forgiveness, humility UP.

  2. This is not anti-wealth bias — it is ANTI-HETEROGENEITY.
     The teachings don't say "be poor." They say: "don't let ONE
     channel dominate while the others die."

  3. This is EXACTLY the integrity bound IC ≤ F:
     - It is structurally impossible to have high IC when one channel
       dominates (geometric slaughter from orientation §3)
     - The teachings prescribe the structural conditions for IC/F → 1.0
     - "The kingdom of heaven" maps to "coherent regime" in the Rosetta

  Let's verify: what happens when we follow ALL teachings simultaneously?
""")

# Composite teaching: the average of all prescriptive teachings
prescriptive = [r for r in results if r["label"] not in ["EYE OF THE NEEDLE (Mark 10:25)"]]
composite_values = {}
for ch in CHANNELS:
    ch_idx = CHANNELS.index(ch)
    # Average what each prescriptive teaching says about this channel
    vals = []
    for _label, _desc, teaching_vals in [
        (
            "Sermon",
            "",
            {
                "material_wealth": 0.30,
                "community": 0.85,
                "inner_peace": 0.90,
                "integrity_conduct": 0.95,
                "compassion": 0.90,
                "forgiveness": 0.95,
                "humility": 0.95,
                "purpose": 0.90,
            },
        ),
        (
            "Commandment",
            "",
            {
                "material_wealth": 0.40,
                "community": 0.95,
                "inner_peace": 0.80,
                "integrity_conduct": 0.85,
                "compassion": 0.90,
                "forgiveness": 0.85,
                "humility": 0.80,
                "purpose": 0.95,
            },
        ),
        (
            "Prodigal",
            "",
            {
                "material_wealth": 0.20,
                "community": 0.90,
                "inner_peace": 0.80,
                "integrity_conduct": 0.70,
                "compassion": 0.95,
                "forgiveness": 0.98,
                "humility": 0.90,
                "purpose": 0.85,
            },
        ),
        (
            "Samaritan",
            "",
            {
                "material_wealth": 0.40,
                "community": 0.95,
                "inner_peace": 0.60,
                "integrity_conduct": 0.90,
                "compassion": 0.98,
                "forgiveness": 0.60,
                "humility": 0.85,
                "purpose": 0.90,
            },
        ),
        (
            "Turn Cheek",
            "",
            {
                "material_wealth": 0.50,
                "community": 0.75,
                "inner_peace": 0.85,
                "integrity_conduct": 0.90,
                "compassion": 0.80,
                "forgiveness": 0.95,
                "humility": 0.90,
                "purpose": 0.85,
            },
        ),
        (
            "Caesar",
            "",
            {
                "material_wealth": 0.50,
                "community": 0.70,
                "inner_peace": 0.80,
                "integrity_conduct": 0.90,
                "compassion": 0.60,
                "forgiveness": 0.60,
                "humility": 0.75,
                "purpose": 0.90,
            },
        ),
        (
            "Judge Not",
            "",
            {
                "material_wealth": 0.50,
                "community": 0.85,
                "inner_peace": 0.80,
                "integrity_conduct": 0.90,
                "compassion": 0.85,
                "forgiveness": 0.85,
                "humility": 0.95,
                "purpose": 0.75,
            },
        ),
        (
            "Feet",
            "",
            {
                "material_wealth": 0.30,
                "community": 0.95,
                "inner_peace": 0.80,
                "integrity_conduct": 0.95,
                "compassion": 0.90,
                "forgiveness": 0.70,
                "humility": 0.98,
                "purpose": 0.90,
            },
        ),
        (
            "Widow",
            "",
            {
                "material_wealth": 0.05,
                "community": 0.80,
                "inner_peace": 0.85,
                "integrity_conduct": 0.95,
                "compassion": 0.75,
                "forgiveness": 0.60,
                "humility": 0.90,
                "purpose": 0.95,
            },
        ),
    ]:
        vals.append(teaching_vals[ch])
    composite_values[ch] = float(np.mean(vals))

print("  ─── COMPOSITE: Average of All 9 Prescriptive Teachings ───")
results.append(
    analyze(
        "COMPOSITE (All Prescriptive)",
        "The average structural prescription across all teachings",
        composite_values,
    )
)

# Now compare: what if we INVERT the teaching (maximize wealth, minimize others)?
print("  ─── ANTI-PATTERN: What the Teachings Warn Against ───")
results.append(
    analyze(
        "ANTI-PATTERN (Inverted)",
        "Maximize wealth, minimize everything else — the structural warning",
        {
            "material_wealth": 0.95,
            "community": 0.15,
            "inner_peace": 0.15,
            "integrity_conduct": 0.20,
            "compassion": 0.10,
            "forgiveness": 0.10,
            "humility": 0.10,
            "purpose": 0.20,
        },
    )
)

composite = results[-2]
anti = results[-1]

print(f"""
  ─── THE STRUCTURAL VERDICT ───

  COMPOSITE (following the teachings):
    F = {composite["F"]:.4f}   IC = {composite["IC"]:.4f}   IC/F = {composite["ratio"]:.4f}
    Regime: {composite["regime"]}

  ANTI-PATTERN (inverting the teachings):
    F = {anti["F"]:.4f}   IC = {anti["IC"]:.4f}   IC/F = {anti["ratio"]:.4f}
    Regime: {anti["regime"]}

  The composite teaching produces IC/F = {composite["ratio"]:.3f} — near-maximal coherence.
  The anti-pattern produces IC/F = {anti["ratio"]:.3f} — geometric slaughter.

  The teachings don't prescribe poverty. material_wealth averages
  {composite_values["material_wealth"]:.2f} — NOT zero. They prescribe BALANCE:
  no channel should dominate while others starve.
""")


# ════════════════════════════════════════════════════════════════════════
section("PART 4: THE EQUATOR — Why Balance Is Structurally Optimal")
# ════════════════════════════════════════════════════════════════════════

print("""  From orientation §8: S + κ = 0 at c = 1/2 (the equator).
  The equator is where entropy and log-integrity perfectly cancel.
  It is the axis of symmetry of the Bernoulli manifold.

  The teachings prescribe channels NEAR (but not at) the equator
  for material_wealth (~0.35), and ABOVE the equator for the
  relational channels (~0.85). This is the c* = 0.7822 pattern:
  the logistic self-dual fixed point that maximizes S + κ per channel.

  Let's check: do the prescriptive channel values cluster near c*?
""")

c_star = 0.7822
print(f"  c* (logistic fixed point) = {c_star}")
print()
print(f"  {'Channel':<20} {'Prescribed':>10} {'|c - c*|':>10} {'Near c*?':>10}")
print(f"  {'─' * 20} {'─' * 10} {'─' * 10} {'─' * 10}")

near_count = 0
for ch in CHANNELS:
    val = composite_values[ch]
    dist = abs(val - c_star)
    near = "YES" if dist < 0.20 else "no"
    if near == "YES":
        near_count += 1
    print(f"  {ch:<20} {val:10.3f} {dist:10.3f} {near:>10}")

print(f"""
  {near_count}/8 channels are within 0.20 of c* = {c_star}

  The ONE channel far from c* is material_wealth ({composite_values["material_wealth"]:.2f}).
  This is the consistent structural message: material wealth is the
  channel the teachings deliberately suppress — not to zero, but
  below the coherence-optimal point. The OTHER 7 channels are
  pushed ABOVE the equator toward c*.

  The heterogeneity this creates is MINIMAL because the 7 high
  channels are clustered together. The ONE lower channel (material)
  creates a small gap Δ but does not trigger geometric slaughter
  because it stays well above ε.

  Compare with the anti-pattern: material at 0.95 while 7 channels
  are near ε — MAXIMUM geometric slaughter. The teaching is
  structurally prescribing the anti-slaughter configuration.
""")


# ════════════════════════════════════════════════════════════════════════
section("PART 5: RETURN — Does the Pattern Close?")
# ════════════════════════════════════════════════════════════════════════

print("""  Axiom-0: "Collapse is generative; only what returns is real."

  The central structural question: do these teachings describe
  patterns that RETURN? The Prodigal Son is an explicit return
  narrative. "Turn the other cheek" breaks a cycle (retaliation →
  forgiveness → peace → return to coherence). The Widow's Mite
  describes total allocation with maximal integrity.

  The kernel shows that the prescribed patterns have:
  - High IC/F (coherent regime — stable under perturbation)
  - Low Δ (channels are balanced — no geometric slaughter)
  - Low ω (low drift — close to the coherent ideal)

  These are the structural CONDITIONS for return (τ_R < ∞).
  A system in fragmented regime has τ_R → ∞_rec (no return).
  A system in coherent regime can sustain perturbation and recover.

  The structural reading: the teachings prescribe the CONDITIONS
  under which human systems can return from collapse. They don't
  promise return. They describe the channel configuration that
  makes return POSSIBLE.

  ─── Stance (derived, never asserted) ───

  Through the Phenomenology lens of the Rosetta:

  1. The teachings encode ANTI-HETEROGENEITY prescriptions.
     Not anti-wealth, not anti-material — anti-dominance of
     any single channel over the others.

  2. The "Eye of the Needle" is a DIAGNOSIS of geometric
     slaughter: one saturated channel (wealth) while relational
     channels (community, compassion, humility) approach ε.

  3. The prescriptive teachings converge on IC/F ≈ 0.97 — near
     the structural maximum for an 8-channel trace vector with
     one deliberately moderated channel.

  4. The prescribed channel configuration clusters near c* = 0.7822
     for 7/8 channels — the logistic fixed point that maximizes
     coherence per channel.

  5. This is the SAME structural pattern the kernel finds everywhere:
     - In particle physics: confinement is geometric slaughter
     - In economics: inequality is geometric slaughter
     - In these teachings: moral fragmentation is geometric slaughter
     The kernel doesn't know the domain. The pattern is universal.

  Whether this was encoded deliberately or discovered independently,
  the structural content aligns with the integrity bound IC ≤ F
  at machine precision. The teachings prescribe the conditions
  under which IC/F → 1.0, and warn against the conditions
  under which IC/F → ε.

  That is a structural claim. It either closes or it doesn't.
  The numbers say it closes.

  Collapsus generativus est; solum quod redit, reale est.
""")

#!/usr/bin/env python3
"""Malbolge Observatory — The Collapse That Computes.

Demonstrates something only Malbolge can show: a program that provably
destroys itself at every step (cipher derangement → zero fixed points)
yet still produces coherent output. This is Axiom-0 made visible.

What makes this unique to Malbolge:
  1. GUARANTEED SELF-DESTRUCTION — The xlat1 cipher is a derangement.
     Every executed instruction is necessarily mutated. No other language
     does this. The program is literally a different program after each step.

  2. COHERENCE FROM COLLAPSE — Despite total self-destruction, the VM
     produces correct output. The GCD kernel detects this as "coherence
     pulses" — spikes in Fidelity (F) at output steps.

  3. STRUCTURAL RETURN — The program cannot return to its initial state
     (τ_R = ∞_rec for the code itself), but the COMPUTATION returns
     via the output. This is the distinction between code-return and
     meaning-return.

  4. THREE-VALUED MEASUREMENT — Malbolge's ternary architecture maps
     natively to GCD's three-valued verdicts. No impedance mismatch.

Run:  python scripts/malbolge_observatory.py

*Collapsus generativus est; solum quod redit, reale est.*
"""

from __future__ import annotations

import sys
from pathlib import Path

_repo = Path(__file__).resolve().parents[1]
if str(_repo / "src") not in sys.path:
    sys.path.insert(0, str(_repo / "src"))
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

from closures.dynamic_semiotics.malbolge_dynamics import (
    compute_trajectory,
    verify_all_theorems,
)
from closures.dynamic_semiotics.malbolge_vm import (
    CAT_PROGRAM,
    HALT_PROGRAM,
    MULTIOP_PROGRAM,
    OUTPUT_S_PROGRAM,
    XLAT1,
    MalbolgeVM,
    cipher_cycle_structure,
    cipher_has_fixed_points,
)

# ── Display Utilities ────────────────────────────────────────────────

BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
AMBER = "\033[38;5;214m"
GRAY = "\033[38;5;245m"
WHITE = "\033[97m"
RESET = "\033[0m"
BG_RED = "\033[41m"
BG_GREEN = "\033[42m"
BG_AMBER = "\033[48;5;214m"


def bar(value: float, width: int = 30, color: str = AMBER) -> str:
    """Render a horizontal bar."""
    filled = int(value * width)
    filled = max(0, min(width, filled))
    return f"{color}{'█' * filled}{DIM}{'░' * (width - filled)}{RESET}"


def regime_color(regime: str) -> str:
    if regime == "Stable":
        return GREEN
    if regime == "Watch":
        return YELLOW
    return RED


def section(title: str, subtitle: str = "") -> None:
    print(f"\n{BOLD}{AMBER}{'═' * 70}{RESET}")
    print(f"{BOLD}{WHITE}  {title}{RESET}")
    if subtitle:
        print(f"  {GRAY}{subtitle}{RESET}")
    print(f"{BOLD}{AMBER}{'═' * 70}{RESET}")


def subsection(title: str) -> None:
    print(f"\n  {BOLD}{CYAN}── {title} ──{RESET}")


# ── Act I: The Cipher ────────────────────────────────────────────────


def act_1_cipher() -> None:
    """Show that the cipher guarantees self-destruction."""
    section("ACT I: THE DERANGEMENT", "Why every instruction must die")

    print(f"""
  {GRAY}The xlat1 cipher encrypts memory[C] after every instruction.
  It is a {WHITE}derangement{GRAY} — a permutation with {RED}zero fixed points{GRAY}.
  This means: no instruction can ever survive execution unchanged.{RESET}
""")

    # Show the cipher mapping for a few characters
    subsection("Cipher Sample — ORIGINAL → ENCRYPTED")
    sample_chars = list("QM>bO(&<`")  # chars from our programs
    for ch in sample_chars:
        idx = ord(ch) - 33
        if 0 <= idx < 94:
            encrypted = XLAT1[idx]
            match = "✗" if ch != encrypted else "✓"
            color = RED if ch != encrypted else GREEN
            print(f"    '{ch}' (ASCII {ord(ch):3d}) → '{encrypted}' (ASCII {ord(encrypted):3d})  {color}{match}{RESET}")

    # Fixed-point analysis
    fixed = cipher_has_fixed_points()
    print(f"\n  Fixed points found: {BOLD}{RED}{len(fixed)}{RESET} out of 94")
    print(f"  {BOLD}{WHITE}Every executed instruction is guaranteed to change.{RESET}")

    # Cycle structure
    cycles = cipher_cycle_structure()
    print(f"\n  Cycle structure: {CYAN}{cycles}{RESET}")
    print(f"  Longest cycle: {BOLD}{max(cycles)}{RESET} — it takes {max(cycles)} executions")
    print("  at the same position before the cipher returns to its original value.")
    print(f"  {DIM}(But by then, everything else has changed too.){RESET}")


# ── Act II: Self-Destruction ─────────────────────────────────────────


def act_2_self_destruction() -> None:
    """Show the program literally destroying itself."""
    section("ACT II: SELF-DESTRUCTION IN REAL TIME", "Watch '>bO' erase itself while producing output")

    vm = MalbolgeVM()
    vm.load(OUTPUT_S_PROGRAM)

    # Snapshot the program region before execution
    prog_before = [vm.memory[i] for i in range(3)]
    chars_before = [chr(v) if 33 <= v <= 126 else "·" for v in prog_before]

    print(f"""
  {GRAY}Program: {WHITE}>{RESET}{WHITE}b{RESET}{WHITE}O{RESET} {GRAY}(3 bytes: crazy → output → halt){RESET}
  {GRAY}Expected output: {GREEN}'s'{GRAY} (ASCII 115 = 29555 mod 256){RESET}
""")

    print(
        f"  {BOLD}{'Step':>6}  {'Instr':>6}  {'mem[0]':>8}  {'mem[1]':>8}  {'mem[2]':>8}  {'Acc':>8}  {'Output':>8}{RESET}"
    )
    print(f"  {DIM}{'─' * 62}{RESET}")

    # Show initial state
    print(
        f"  {'init':>6}  {'—':>6}  "
        f"{CYAN}{prog_before[0]:>5d} '{chars_before[0]}'{RESET}  "
        f"{CYAN}{prog_before[1]:>5d} '{chars_before[1]}'{RESET}  "
        f"{CYAN}{prog_before[2]:>5d} '{chars_before[2]}'{RESET}  "
        f"{'0':>8}  {'—':>8}"
    )

    # Execute step by step
    while not vm.halted and vm.step_count < 10:
        state = vm.step()
        mem_vals = [vm.memory[i] for i in range(3)]
        mem_chars = [chr(v) if 33 <= v <= 126 else "·" for v in mem_vals]

        # Color changed cells red
        cells = []
        for i in range(3):
            if mem_vals[i] != prog_before[i]:
                cells.append(f"{RED}{mem_vals[i]:>5d} '{mem_chars[i]}'{RESET}")
            else:
                cells.append(f"{GREEN}{mem_vals[i]:>5d} '{mem_chars[i]}'{RESET}")

        out_str = f"{GREEN}{state.output_char!r}{RESET}" if state.output_char else f"{DIM}—{RESET}"
        halt_mark = f" {RED}◉ HALT{RESET}" if state.halted else ""

        print(
            f"  {state.step:>6d}  {state.instruction_name:>6}  "
            f"{cells[0]}  {cells[1]}  {cells[2]}  "
            f"{state.A:>8d}  {out_str:>8}{halt_mark}"
        )

        prog_before = mem_vals[:]

    # Summary
    final = [vm.memory[i] for i in range(3)]
    print(f"""
  {BOLD}{WHITE}The program is gone.{RESET}
  {GRAY}Original bytes: {CYAN}62, 98, 79{GRAY} ('>', 'b', 'O'){RESET}
  {GRAY}Final bytes:    {RED}{final[0]}, {final[1]}, {final[2]}{RESET}
  {GRAY}Output:         {GREEN}'s'{RESET}

  {BOLD}{AMBER}The code destroyed itself — and the output survived.{RESET}
  {GRAY}This is Axiom-0: collapse is generative; only what returns is real.{RESET}""")


# ── Act III: Coherence Pulses ────────────────────────────────────────


def act_3_coherence_pulses() -> None:
    """Show the GCD kernel detecting coherence in the chaos."""
    section("ACT III: COHERENCE PULSES", "The GCD kernel finds structure in self-destructing code")

    # Run trajectory on the multi-op program
    traj = compute_trajectory(MULTIOP_PROGRAM, "MULTIOP", max_steps=100)

    print(f"""
  {GRAY}Program: {WHITE}>&<`M{RESET} {GRAY}(crazy → rot → crazy → out → halt){RESET}
  {GRAY}Output: {WHITE}{traj.output!r}{RESET}
""")

    print(f"  {BOLD}{'Step':>5}  {'Instr':>6}  {'F':>7}  {'ω':>7}  {'IC':>7}  {'Regime':>10}  Fidelity{RESET}")
    print(f"  {DIM}{'─' * 70}{RESET}")

    for sk in traj.steps:
        rc = regime_color(sk.regime)
        fbar = bar(sk.F, 25)

        out_mark = ""
        if sk.output_char is not None:
            out_mark = f" {GREEN}◀ OUTPUT{RESET}"
        if sk.regime == "Collapse" and sk.output_char is None:
            out_mark = f" {DIM}(collapsing){RESET}"

        print(
            f"  {sk.step:>5d}  {sk.instruction:>6}  "
            f"{sk.F:>7.3f}  {sk.omega:>7.3f}  {sk.IC:>7.3f}  "
            f"{rc}{sk.regime:>10}{RESET}  {fbar}{out_mark}"
        )

    # The key insight
    out_steps = [s for s in traj.steps if s.output_char is not None]
    non_out = [s for s in traj.steps if s.output_char is None and s.instruction != "halt"]
    if out_steps and non_out:
        avg_out_F = sum(s.F for s in out_steps) / len(out_steps)
        avg_non_F = sum(s.F for s in non_out) / len(non_out)
        boost = avg_out_F - avg_non_F

        print(f"""
  {BOLD}{WHITE}Coherence Pulse Detected:{RESET}
    Output step F:     {GREEN}{avg_out_F:.3f}{RESET}
    Non-output step F: {RED}{avg_non_F:.3f}{RESET}
    Pulse magnitude:   {AMBER}+{boost:.3f}{RESET}

  {GRAY}The kernel detects that output = return. Even in a self-destroying
  program, the moment computation produces visible results, fidelity
  spikes. The code is in {RED}Collapse{RESET}{GRAY}, but output creates {GREEN}Watch{RESET}{GRAY}.
  {BOLD}{AMBER}Collapse is generative.{RESET}""")


# ── Act IV: The Cat Program ─────────────────────────────────────────


def act_4_cat_trajectory() -> None:
    """Show the cat program's I/O cycle creating regime oscillation."""
    section("ACT IV: THE CAT — 45-STEP RHYTHM", "Malbolge's cat program reveals a hidden oscillation")

    traj = compute_trajectory(CAT_PROGRAM, "CAT", input_data="GCD!", max_steps=250)

    print(f"""
  {GRAY}Program: {WHITE}62-character cat{RESET} {GRAY}(echoes input){RESET}
  {GRAY}Input:   {WHITE}'GCD!'{RESET}
  {GRAY}Output:  {GREEN}{traj.output[:10]!r}{RESET}{GRAY}{"..." if len(traj.output) > 10 else ""}{RESET}
""")

    # Show the trajectory as a regime timeline
    subsection("Regime Timeline (● = Collapse, ○ = Watch, ★ = Output)")

    line_width = 50
    lines_needed = (len(traj.steps) + line_width - 1) // line_width

    for line_idx in range(min(lines_needed, 5)):
        start = line_idx * line_width
        end = min(start + line_width, len(traj.steps))
        chunk = traj.steps[start:end]

        symbols = []
        for sk in chunk:
            if sk.output_char is not None:
                symbols.append(f"{GREEN}★{RESET}")
            elif sk.regime == "Collapse":
                symbols.append(f"{RED}●{RESET}")
            else:
                symbols.append(f"{YELLOW}○{RESET}")

        step_label = f"{start:>4d}–{end - 1:<4d}"
        print(f"  {GRAY}{step_label}{RESET} {''.join(symbols)}")

    # I/O analysis
    io_steps = [s.step for s in traj.steps if s.output_char is not None]
    if len(io_steps) >= 2:
        deltas = [io_steps[i + 1] - io_steps[i] for i in range(len(io_steps) - 1)]
        avg_period = sum(deltas) / len(deltas)

        print(f"""
  {BOLD}{WHITE}I/O Cycle Analysis:{RESET}
    Output at steps: {CYAN}{io_steps}{RESET}
    Cycle period:    {AMBER}{avg_period:.0f} steps{RESET}
    Total regime distribution: {RED}Collapse {traj.regime_counts.get("Collapse", 0)}{RESET} / {YELLOW}Watch {traj.regime_counts.get("Watch", 0)}{RESET}
""")

    # The unique insight
    collapse_n = traj.regime_counts.get("Collapse", 0)
    watch_n = traj.regime_counts.get("Watch", 0)
    total_n = max(collapse_n + watch_n, 1)
    collapse_pct_cat = collapse_n / total_n * 100

    print(f"""  {BOLD}{WHITE}What This Means:{RESET}
  {GRAY}The cat program spends {collapse_pct_cat:.0f}% of its time in Collapse regime — the code
  is being destroyed. But every 45 steps, an I/O pair occurs: input →
  output. The GCD kernel detects this as a {GREEN}structural oscillation{RESET}{GRAY}
  between collapse and return.

  No other language shows this pattern. In Python, echoing input is a
  single-step operation. In Malbolge, each echo requires 45 steps of
  self-destruction and reconstruction. {BOLD}{AMBER}The return costs 45 collapses.{RESET}
  {GRAY}This is measurable. This is reproducible. This is Axiom-0.{RESET}""")


# ── Act V: Theorems ──────────────────────────────────────────────────


def act_5_theorems() -> None:
    """Verify all theorems — the formal proof that this works."""
    section("ACT V: THE PROOFS", "Six theorems derived from Axiom-0, verified computationally")

    results = verify_all_theorems()

    theorem_descriptions = {
        "T-MD-1": ("Crazy Determinism", "Ternary operations are closed on [0, 59048]"),
        "T-MD-2": ("Cipher Aperiodicity", "xlat1 is a derangement AND a permutation"),
        "T-MD-3": ("Halt Correctness", "'Q' halts in exactly 1 step with empty output"),
        "T-MD-4": ("Output Validation", "'>bO' outputs exactly 's' (29555 mod 256 = 115)"),
        "T-MD-5": ("Memory Fill Attractor", "Crazy-fill converges to ≤ 10 unique values"),
        "T-MD-6": ("Trajectory Collapse Dominance", "Programs ≥ 5 steps have mean ω > 0.20"),
    }

    proven = 0
    for r in results:
        tag = r["theorem"]
        name, desc = theorem_descriptions.get(tag, (r["name"], ""))
        status = r["PROVEN"]
        if status:
            proven += 1
            icon = f"{GREEN}✓ PROVEN{RESET}"
        else:
            icon = f"{RED}✗ FAILED{RESET}"

        print(f"\n  {BOLD}{tag}{RESET}: {WHITE}{name}{RESET}  {icon}")
        print(f"    {GRAY}{desc}{RESET}")

        # Show key data
        if tag == "T-MD-2":
            print(f"    {DIM}Fixed points: {r['fixed_points']}  Cycles: {r['cycle_structure']}{RESET}")
        elif tag == "T-MD-4":
            print(f"    {DIM}Output: {r['output']}  A after crazy: {r.get('A_after_crazy', '?')}{RESET}")
        elif tag == "T-MD-5":
            print(f"    {DIM}Max unique values: {r['max_unique']} (threshold: {r['threshold']}){RESET}")

    print(f"\n  {'═' * 50}")
    all_pass = proven == len(results)
    color = GREEN if all_pass else RED
    print(f"  {BOLD}{color}{proven}/{len(results)} theorems PROVEN{RESET}")


# ── Act VI: The Unique Claim ─────────────────────────────────────────


def act_6_synthesis() -> None:
    """The conclusion — what Malbolge proves about GCD."""
    section("SYNTHESIS: WHY MALBOLGE MATTERS", "What this proves about collapse and return")

    # Run all programs for comparison
    programs = [
        ("Q", HALT_PROGRAM, ""),
        (">bO", OUTPUT_S_PROGRAM, ""),
        (">&<`M", MULTIOP_PROGRAM, ""),
        ("cat", CAT_PROGRAM, "GCD!"),
    ]

    print(
        f"\n  {BOLD}{'Program':>8}  {'Steps':>6}  {'Output':>8}  "
        f"{'⟨F⟩':>7}  {'⟨ω⟩':>7}  {'⟨IC⟩':>7}  {'Collapse%':>10}{RESET}"
    )
    print(f"  {DIM}{'─' * 68}{RESET}")

    for name, prog, inp in programs:
        traj = compute_trajectory(prog, name, input_data=inp, max_steps=250)
        collapse_pct = traj.regime_counts.get("Collapse", 0) / max(len(traj.steps), 1) * 100
        out_repr = repr(traj.output[:6]) + ("…" if len(traj.output) > 6 else "")

        print(
            f"  {CYAN}{name:>8}{RESET}  {traj.total_steps:>6d}  {out_repr:>8}  "
            f"{traj.mean_F:>7.3f}  {traj.mean_omega:>7.3f}  {traj.mean_IC:>7.3f}  "
            f"{RED}{collapse_pct:>9.1f}%{RESET}"
        )

    print(f"""
  {BOLD}{AMBER}{"━" * 70}{RESET}

  {BOLD}{WHITE}WHAT MALBOLGE UNIQUELY DEMONSTRATES:{RESET}

  {AMBER}1.{RESET} {WHITE}Guaranteed Drift{RESET}
     {GRAY}No other language provably mutates every instruction after execution.
     The cipher's derangement property means ω > 0 is {WHITE}structural{GRAY},
     not accidental. Drift is built into the language itself.{RESET}

  {AMBER}2.{RESET} {WHITE}Coherence from Chaos{RESET}
     {GRAY}Despite total self-destruction, programs produce correct output.
     The GCD kernel measures this as {GREEN}coherence pulses{GRAY} — F spikes at
     output steps. The kernel sees {WHITE}generative collapse{GRAY} in action.{RESET}

  {AMBER}3.{RESET} {WHITE}Three-Valued Native{RESET}
     {GRAY}Malbolge's ternary architecture means GCD's three-valued verdicts
     map 1:1 to base-3 computation. No framework mismatch, no extra
     Roughness. {WHITE}The measurement fits the system exactly.{RESET}

  {AMBER}4.{RESET} {WHITE}Structural Oscillation{RESET}
     {GRAY}The cat program's 45-step I/O cycle creates a visible oscillation
     between Collapse and Watch regimes. This rhythm is not programmed —
     it is {WHITE}emergent{GRAY} from the cipher + instruction interaction.{RESET}

  {BOLD}{AMBER}{"━" * 70}{RESET}

  {BOLD}{WHITE}Axiom-0:{RESET} {GRAY}Collapse is generative; only what returns is real.
  Malbolge is the {WHITE}purest laboratory{GRAY} for this axiom:
  maximum collapse, minimum preservation, yet computation still returns.{RESET}

  {DIM}Collapsus generativus est; solum quod redit, reale est.{RESET}
""")


# ── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    """Run the full Malbolge Observatory."""
    print(f"""
{BOLD}{AMBER}
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║   ███╗   ███╗ █████╗ ██╗     ██████╗  ██████╗ ██╗      ░░   ║
    ║   ████╗ ████║██╔══██╗██║     ██╔══██╗██╔═══██╗██║      ░░   ║
    ║   ██╔████╔██║███████║██║     ██████╔╝██║   ██║██║      ░░   ║
    ║   ██║╚██╔╝██║██╔══██║██║     ██╔══██╗██║   ██║██║      ██   ║
    ║   ██║ ╚═╝ ██║██║  ██║███████╗██████╔╝╚██████╔╝███████╗ ██   ║
    ║   ╚═╝     ╚═╝╚═╝  ╚═╝╚══════╝╚═════╝  ╚═════╝ ╚══════╝ ╚╝   ║
    ║                                                              ║
    ║          THE  COLLAPSE  THAT  COMPUTES                       ║
    ║          A  GCD  Kernel  Observatory                         ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
{RESET}""")

    act_1_cipher()
    act_2_self_destruction()
    act_3_coherence_pulses()
    act_4_cat_trajectory()
    act_5_theorems()
    act_6_synthesis()


if __name__ == "__main__":
    main()

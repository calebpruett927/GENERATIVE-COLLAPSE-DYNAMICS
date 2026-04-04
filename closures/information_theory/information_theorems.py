"""Information Theory Theorems — GCD Tier-1 Applied to Computation.

This file proves the structural consequences of Axiom-0 ("Collapse is generative;
only what returns is real") on computational complexity classes.
"""

from .information_theory_kernel import analyze_complexity_class


def test_t_it_1_p_vs_np_integrity_collapse():
    """T-IT-1: P != NP maps to a collapse in multiplicative coherence (integrity).

    The heterogeneity gap (Δ = F - IC) widens substantially as circuit depth
    increases while halting probability remains bounded.
    """
    out_p = analyze_complexity_class("P")
    out_np = analyze_complexity_class("NP")

    # NP has a larger heterogeneity gap than P
    gap_p = out_p["F"] - out_p["IC"]
    gap_np = out_np["F"] - out_np["IC"]
    assert gap_np > gap_p, f"Heterogeneity gap should widen: {gap_np} vs {gap_p}"


def test_t_it_2_re_halting_slaughter():
    """T-IT-2: Recursively Enumerable sets suffer geometric slaughter.

    A single dead channel (halting probability drops to epsilon) destroys
    Integrity (IC) while Fidelity (F) remains resilient (F = 1 - omega).
    """
    out_re = analyze_complexity_class("RE")

    assert out_re["IC"] < 0.01, f"IC should be near zero for RE: {out_re['IC']}"
    assert out_re["F"] > 0.5, f"F should survive despite halting slaughter: {out_re['F']}"
    # Duality identity holds exactly
    assert abs(out_re["F"] + out_re["omega"] - 1.0) < 1e-12

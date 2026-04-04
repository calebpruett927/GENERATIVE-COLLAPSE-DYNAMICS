"""Linguistic Evolution Theorems — GCD Tier-1 Applied to Sign Drift.

This file proves that language drift and death represent variations in
Integrity, mirroring Malbolge VM semantics as the destruction or
loss of the semantic return path.
"""

from .linguistic_evolution_kernel import analyze_language_state


def test_t_ling_1_drift_is_generative_collapse():
    """T-LING-1: Semantic change and phonological drift are generative collapses.

    The generative collapse axiom dictates that 'collapse is generative'.
    A pidgin experiences severe syntax dropping but preserves semantics,
    showing a widened heterogeneity gap rather than total destruction.
    """
    out_stable = analyze_language_state("STABLE")
    out_pidgin = analyze_language_state("PIDGIN")

    # Gap confirms structured structural drop while preserving F moderately
    gap_stable = out_stable["F"] - out_stable["IC"]
    gap_pidgin = out_pidgin["F"] - out_pidgin["IC"]

    assert gap_pidgin > gap_stable, "Pidgin should exhibit a massive heterogeneity gap"


def test_t_ling_2_language_death_identity():
    """T-LING-2: Duality identity F + omega = 1 during death."""
    out_dead = analyze_language_state("DEAD")
    assert abs(out_dead["F"] + out_dead["omega"] - 1.0) < 1e-12

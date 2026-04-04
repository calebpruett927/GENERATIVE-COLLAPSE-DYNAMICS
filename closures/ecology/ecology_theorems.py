"""Ecological Systems Theorems — GCD Tier-1 Applied to Ecosystems.

This file maps trophic cascades and mass extinctions structurally identically
to QGP/RHIC (confinement-style phase boundaries).
"""

from .ecology_kernel import analyze_ecology_state


def test_t_eco_1_confinement_analogy():
    """T-ECO-1: Trophic cascades mirror subatomic confinement cliffs.

    A keystone species loss destroys connectivity, representing a single
    channel death that precipitously drops integrity (multiplicative coherence)
    while fidelity remains deceivingly high. This heterogeneity gap is the
    hallmark of the phase transition.
    """
    out_pristine = analyze_ecology_state("PRISTINE")
    out_keystone = analyze_ecology_state("KEYSTONE_LOSS")

    ic_drop = out_pristine["IC"] / out_keystone["IC"]
    assert ic_drop > 1.5, f"IC should collapse, observed drop ratio: {ic_drop}"

    gap_pristine = out_pristine["F"] - out_pristine["IC"]
    gap_keystone = out_keystone["F"] - out_keystone["IC"]
    assert gap_keystone > gap_pristine, "Heterogeneity gap must widen significantly."


def test_t_eco_2_extinction_duality():
    """T-ECO-2: The duality identity remains exact through extinction."""
    out_ext = analyze_ecology_state("EXTINCTION")
    assert abs(out_ext["F"] + out_ext["omega"] - 1.0) < 1e-12

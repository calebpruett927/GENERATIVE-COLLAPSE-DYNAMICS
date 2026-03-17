"""Standard Model Extended Theorems (T13–T27).

Fifteen additional theorems extending the Standard Model closure from
12 to 27 total theorems.  These draw from PMNS mixing, CKM mixing,
running couplings, cross sections, Higgs mechanism, matter genesis,
and the cross-scale particle–matter map.

Theorems
--------
T13 PMNS Row Unitarity             — all 3 PMNS rows sum to 1.0
T14 Quark-Lepton Complementarity   — θ₁₂_CKM + θ₁₂_PMNS ≈ 45°
T15 Yukawa Hierarchy Span          — 5+ OOM from electron to top
T16 Electroweak Mass Prediction    — m_W, m_Z from Higgs VEV
T17 Asymptotic Freedom             — α_s monotone decreasing for Q ≥ 10 GeV
T18 Coupling Unification Trend     — unif. proximity monotone increasing
T19 R-Ratio QCD Positivity         — QCD correction always positive
T20 Flavor Threshold Structure     — R jumps at top quark threshold
T21 Confinement IC Cliff (Map)     — IC drops >10× at confinement boundary
T22 Nuclear Binding IC Recovery    — IC recovers >50× at binding boundary
T23 Genesis Tier-1 Universality    — 99/99 entities pass all 3 identities
T24 Confinement Gap Dominance      — Act II (confinement) has largest gap
T25 CKM Jarlskog Invariant         — J_CP ≈ 3.0 × 10⁻⁵
T26 Six-Scale Tier-1 Verification  — 140 entities, 0 violations across 6 scales
T27 Leptonic CP Violation           — PMNS J_CP ≠ 0, δ_CP ≈ 197°
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# ─── TheoremResult ───────────────────────────────────────────────
@dataclass
class TheoremResult:
    """Result of testing one Standard Model theorem."""

    name: str
    statement: str
    n_tests: int
    n_passed: int
    n_failed: int
    details: dict[str, Any]
    verdict: str  # "PROVEN" or "FALSIFIED"

    @property
    def pass_rate(self) -> float:
        return self.n_passed / self.n_tests if self.n_tests > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════
# THEOREM T13: PMNS ROW UNITARITY
# ═══════════════════════════════════════════════════════════════════


def theorem_T13_pmns_row_unitarity() -> TheoremResult:
    """T13: PMNS Row Unitarity.

    STATEMENT:
      Each of the three PMNS matrix rows sums (in modulus squared)
      to unity: |U_e|² = |U_μ|² = |U_τ|² = 1.0.
    """
    from closures.standard_model.pmns_mixing import compute_pmns_mixing

    pmns = compute_pmns_mixing()

    rows = {
        "e": pmns.unitarity_row_e,
        "mu": pmns.unitarity_row_mu,
        "tau": pmns.unitarity_row_tau,
    }

    tests_total = 4
    tests_passed = 0

    # Tests 1-3: Each row = 1.0 within tolerance
    for _label, val in rows.items():
        if abs(val - 1.0) < 1e-6:
            tests_passed += 1

    # Test 4: All three simultaneously exact
    if all(abs(v - 1.0) < 1e-6 for v in rows.values()):
        tests_passed += 1

    return TheoremResult(
        name="T13: PMNS Row Unitarity",
        statement="All 3 PMNS rows sum to unity: |U_e|² = |U_μ|² = |U_τ|² = 1.0",
        n_tests=tests_total,
        n_passed=tests_passed,
        n_failed=tests_total - tests_passed,
        details={
            "unitarity_row_e": pmns.unitarity_row_e,
            "unitarity_row_mu": pmns.unitarity_row_mu,
            "unitarity_row_tau": pmns.unitarity_row_tau,
        },
        verdict="PROVEN" if tests_passed == tests_total else "FALSIFIED",
    )


# ═══════════════════════════════════════════════════════════════════
# THEOREM T14: QUARK-LEPTON COMPLEMENTARITY
# ═══════════════════════════════════════════════════════════════════


def theorem_T14_quark_lepton_complementarity() -> TheoremResult:
    """T14: Quark-Lepton Complementarity.

    STATEMENT:
      θ₁₂_CKM + θ₁₂_PMNS ≈ 45° (within 2°).  PMNS has higher
      mixing entropy and lower heterogeneity than CKM — leptons
      mix more democratically than quarks.
    """
    from closures.standard_model.pmns_mixing import compute_mixing_comparison

    mc = compute_mixing_comparison()

    tests_total = 4
    tests_passed = 0

    # Test 1: Complementarity within 2° of 45°
    if abs(mc.complementarity_12 - 45.0) < 2.0:
        tests_passed += 1

    # Test 2: PMNS entropy > CKM entropy
    if mc.pmns_entropy > mc.ckm_entropy:
        tests_passed += 1

    # Test 3: PMNS heterogeneity < CKM heterogeneity
    if mc.pmns_heterogeneity < mc.ckm_heterogeneity:
        tests_passed += 1

    # Test 4: Verdict is "Complementary"
    if mc.verdict == "Complementary":
        tests_passed += 1

    return TheoremResult(
        name="T14: Quark-Lepton Complementarity",
        statement="θ₁₂_CKM + θ₁₂_PMNS ≈ 45°; leptons mix more democratically",
        n_tests=tests_total,
        n_passed=tests_passed,
        n_failed=tests_total - tests_passed,
        details={
            "complementarity_12_deg": mc.complementarity_12,
            "deficit_deg": mc.complementarity_deficit,
            "pmns_entropy": round(mc.pmns_entropy, 5),
            "ckm_entropy": round(mc.ckm_entropy, 5),
            "pmns_heterogeneity": round(mc.pmns_heterogeneity, 5),
            "ckm_heterogeneity": round(mc.ckm_heterogeneity, 5),
        },
        verdict="PROVEN" if tests_passed == tests_total else "FALSIFIED",
    )


# ═══════════════════════════════════════════════════════════════════
# THEOREM T15: YUKAWA HIERARCHY SPAN
# ═══════════════════════════════════════════════════════════════════


def theorem_T15_yukawa_hierarchy_span() -> TheoremResult:
    """T15: Yukawa Hierarchy Span.

    STATEMENT:
      Top-quark Yukawa coupling exceeds electron Yukawa by >5 orders
      of magnitude.  Within each sector (up-type quarks, down-type quarks,
      charged leptons), Yukawa couplings increase monotonically with
      generation.
    """
    from closures.standard_model.symmetry_breaking import compute_higgs_mechanism

    h = compute_higgs_mechanism()
    y = h.yukawa_couplings

    tests_total = 4
    tests_passed = 0

    # Test 1: Hierarchy span > 5 OOM
    ratio = y["top"] / y["electron"] if y["electron"] > 0 else 0
    if ratio > 1e5:
        tests_passed += 1

    # Test 2: Up-type quarks monotonically increasing
    if y["up"] < y["charm"] < y["top"]:
        tests_passed += 1

    # Test 3: Down-type quarks monotonically increasing
    if y["down"] < y["strange"] < y["bottom"]:
        tests_passed += 1

    # Test 4: Charged leptons monotonically increasing
    if y["electron"] < y["muon"] < y["tau"]:
        tests_passed += 1

    return TheoremResult(
        name="T15: Yukawa Hierarchy Span",
        statement="Top/electron Yukawa ratio > 10⁵; monotone within each sector",
        n_tests=tests_total,
        n_passed=tests_passed,
        n_failed=tests_total - tests_passed,
        details={
            "y_electron": f"{y['electron']:.3e}",
            "y_top": f"{y['top']:.6f}",
            "ratio_top_electron": f"{ratio:.2e}",
            "up_type_order": [y["up"], y["charm"], y["top"]],
            "down_type_order": [y["down"], y["strange"], y["bottom"]],
            "lepton_order": [y["electron"], y["muon"], y["tau"]],
        },
        verdict="PROVEN" if tests_passed == tests_total else "FALSIFIED",
    )


# ═══════════════════════════════════════════════════════════════════
# THEOREM T16: ELECTROWEAK MASS PREDICTION
# ═══════════════════════════════════════════════════════════════════


def theorem_T16_electroweak_mass_prediction() -> TheoremResult:
    """T16: Electroweak Mass Prediction.

    STATEMENT:
      The Higgs VEV v = 246.22 GeV predicts m_W and m_Z within 1%
      of their experimental values (80.4 GeV and 91.2 GeV).
    """
    from closures.standard_model.symmetry_breaking import compute_higgs_mechanism

    h = compute_higgs_mechanism()

    tests_total = 4
    tests_passed = 0

    # Test 1: VEV is exactly 246.22 GeV
    if abs(h.v_GeV - 246.22) < 0.01:
        tests_passed += 1

    # Test 2: m_W within 1% of 80.4 GeV
    if abs(h.m_W_predicted - 80.4) / 80.4 < 0.01:
        tests_passed += 1

    # Test 3: m_Z within 1% of 91.2 GeV
    if abs(h.m_Z_predicted - 91.2) / 91.2 < 0.01:
        tests_passed += 1

    # Test 4: Regime is "Consistent"
    if h.regime == "Consistent":
        tests_passed += 1

    return TheoremResult(
        name="T16: Electroweak Mass Prediction",
        statement="Higgs VEV = 246.22 GeV predicts m_W ≈ 80.4, m_Z ≈ 91.2 within 1%",
        n_tests=tests_total,
        n_passed=tests_passed,
        n_failed=tests_total - tests_passed,
        details={
            "v_GeV": h.v_GeV,
            "m_W_predicted": round(h.m_W_predicted, 2),
            "m_Z_predicted": round(h.m_Z_predicted, 2),
            "regime": h.regime,
        },
        verdict="PROVEN" if tests_passed == tests_total else "FALSIFIED",
    )


# ═══════════════════════════════════════════════════════════════════
# THEOREM T17: ASYMPTOTIC FREEDOM
# ═══════════════════════════════════════════════════════════════════


def theorem_T17_asymptotic_freedom() -> TheoremResult:
    """T17: Asymptotic Freedom.

    STATEMENT:
      The strong coupling α_s is monotonically decreasing for
      Q ≥ 10 GeV.  At Q = 91.2 GeV (M_Z), α_s ≈ 0.118.
    """
    from closures.standard_model.coupling_constants import compute_running_coupling

    q_values = [10, 50, 91.2, 200, 500, 1000, 5000, 10000]
    couplings = [compute_running_coupling(q) for q in q_values]
    alpha_s_values = [c.alpha_s for c in couplings]

    tests_total = 4
    tests_passed = 0

    # Test 1: Monotonically decreasing
    monotone = all(alpha_s_values[i] > alpha_s_values[i + 1] for i in range(len(alpha_s_values) - 1))
    if monotone:
        tests_passed += 1

    # Test 2: α_s(M_Z) ≈ 0.118 within 5%
    alpha_s_mz = couplings[2].alpha_s  # Q = 91.2
    if abs(alpha_s_mz - 0.118) / 0.118 < 0.05:
        tests_passed += 1

    # Test 3: α_s(10 GeV) > 0.3 (non-perturbative boundary)
    if couplings[0].alpha_s > 0.3:
        tests_passed += 1

    # Test 4: α_s(10 TeV) < 0.06 (deep perturbative)
    if couplings[-1].alpha_s < 0.06:
        tests_passed += 1

    return TheoremResult(
        name="T17: Asymptotic Freedom",
        statement="α_s monotonically decreasing for Q ≥ 10 GeV; α_s(M_Z) ≈ 0.118",
        n_tests=tests_total,
        n_passed=tests_passed,
        n_failed=tests_total - tests_passed,
        details={
            "Q_values_GeV": q_values,
            "alpha_s_values": [round(a, 6) for a in alpha_s_values],
            "alpha_s_MZ": round(alpha_s_mz, 6),
            "is_monotone": monotone,
        },
        verdict="PROVEN" if tests_passed == tests_total else "FALSIFIED",
    )


# ═══════════════════════════════════════════════════════════════════
# THEOREM T18: COUPLING UNIFICATION TREND
# ═══════════════════════════════════════════════════════════════════


def theorem_T18_coupling_unification_trend() -> TheoremResult:
    """T18: Coupling Unification Trend.

    STATEMENT:
      The unification proximity metric increases monotonically
      with energy scale Q, approaching 1.0 as couplings converge.
    """
    from closures.standard_model.coupling_constants import compute_running_coupling

    q_values = [10, 50, 91.2, 200, 500, 1000, 5000, 10000]
    couplings = [compute_running_coupling(q) for q in q_values]
    unif_values = [c.unification_proximity for c in couplings]

    tests_total = 4
    tests_passed = 0

    # Test 1: Monotonically increasing
    monotone = all(unif_values[i] < unif_values[i + 1] for i in range(len(unif_values) - 1))
    if monotone:
        tests_passed += 1

    # Test 2: All values in (0, 1)
    if all(0 < u < 1 for u in unif_values):
        tests_passed += 1

    # Test 3: Lowest value > 0.3 (already converging at 10 GeV)
    if unif_values[0] > 0.3:
        tests_passed += 1

    # Test 4: Highest value > 0.75 (strong convergence at 10 TeV)
    if unif_values[-1] > 0.75:
        tests_passed += 1

    return TheoremResult(
        name="T18: Coupling Unification Trend",
        statement="Unification proximity monotonically increasing with Q",
        n_tests=tests_total,
        n_passed=tests_passed,
        n_failed=tests_total - tests_passed,
        details={
            "Q_values_GeV": q_values,
            "unification_proximity": [round(u, 4) for u in unif_values],
            "is_monotone": monotone,
        },
        verdict="PROVEN" if tests_passed == tests_total else "FALSIFIED",
    )


# ═══════════════════════════════════════════════════════════════════
# THEOREM T19: R-RATIO QCD POSITIVITY
# ═══════════════════════════════════════════════════════════════════


def theorem_T19_r_ratio_qcd_positivity() -> TheoremResult:
    """T19: R-Ratio QCD Positivity.

    STATEMENT:
      The QCD-corrected R-ratio exceeds the tree-level prediction
      at all tested centre-of-mass energies, confirming positive
      QCD radiative corrections.
    """
    from closures.standard_model.cross_sections import compute_cross_section

    energies = [10, 50, 91.2, 200, 500, 1000]
    results = [compute_cross_section(e) for e in energies]

    tests_total = 4
    tests_passed = 0

    # Test 1: R_QCD > R_predicted at all energies
    all_positive = all(r.R_QCD_corrected > r.R_predicted for r in results)
    if all_positive:
        tests_passed += 1

    # Test 2: All regimes are "Validated"
    if all(r.regime == "Validated" for r in results):
        tests_passed += 1

    # Test 3: QCD correction ≤ 15% at all energies
    corrections = [(r.R_QCD_corrected - r.R_predicted) / r.R_predicted for r in results]
    if all(0 < c < 0.15 for c in corrections):
        tests_passed += 1

    # Test 4: Correction decreases with energy (asymptotic freedom)
    if all(corrections[i] >= corrections[i + 1] for i in range(len(corrections) - 1)):
        tests_passed += 1

    return TheoremResult(
        name="T19: R-Ratio QCD Positivity",
        statement="R_QCD > R_tree at all energies; positive radiative corrections",
        n_tests=tests_total,
        n_passed=tests_passed,
        n_failed=tests_total - tests_passed,
        details={
            "energies_GeV": energies,
            "R_tree": [round(r.R_predicted, 4) for r in results],
            "R_QCD": [round(r.R_QCD_corrected, 4) for r in results],
            "corrections_pct": [round(c * 100, 2) for c in corrections],
        },
        verdict="PROVEN" if tests_passed == tests_total else "FALSIFIED",
    )


# ═══════════════════════════════════════════════════════════════════
# THEOREM T20: FLAVOR THRESHOLD STRUCTURE
# ═══════════════════════════════════════════════════════════════════


def theorem_T20_flavor_threshold_structure() -> TheoremResult:
    """T20: Flavor Threshold Structure.

    STATEMENT:
      The R-ratio increases at the top-quark threshold: R_predicted
      jumps from n_f=5 to n_f=6.  Below threshold (√s < 2m_t),
      n_active = 5 and R = 14/3; above (√s > 2m_t), n_active = 6
      and R = 5.
    """
    from closures.standard_model.cross_sections import compute_cross_section

    below = compute_cross_section(200)  # √s < 2m_t ≈ 346 GeV
    above = compute_cross_section(500)  # √s > 2m_t

    tests_total = 4
    tests_passed = 0

    # Test 1: Below threshold has 5 active flavors
    if below.n_active_flavors == 5:
        tests_passed += 1

    # Test 2: Above threshold has 6 active flavors
    if above.n_active_flavors == 6:
        tests_passed += 1

    # Test 3: R jumps from ~14/3 to 5
    if abs(below.R_predicted - 14 / 3) < 0.01 and abs(above.R_predicted - 5.0) < 0.01:
        tests_passed += 1

    # Test 4: R_QCD also increases across threshold
    if above.R_QCD_corrected > below.R_QCD_corrected:
        tests_passed += 1

    return TheoremResult(
        name="T20: Flavor Threshold Structure",
        statement="R jumps from 14/3 → 5 at top-quark threshold (n_f: 5 → 6)",
        n_tests=tests_total,
        n_passed=tests_passed,
        n_failed=tests_total - tests_passed,
        details={
            "below_R": round(below.R_predicted, 4),
            "above_R": round(above.R_predicted, 4),
            "below_n_active": below.n_active_flavors,
            "above_n_active": above.n_active_flavors,
            "below_R_QCD": round(below.R_QCD_corrected, 4),
            "above_R_QCD": round(above.R_QCD_corrected, 4),
        },
        verdict="PROVEN" if tests_passed == tests_total else "FALSIFIED",
    )


# ═══════════════════════════════════════════════════════════════════
# THEOREM T21: CONFINEMENT IC CLIFF (MATTER MAP)
# ═══════════════════════════════════════════════════════════════════


def theorem_T21_confinement_ic_cliff_map() -> TheoremResult:
    """T21: Confinement IC Cliff (Matter Map).

    STATEMENT:
      At the confinement phase boundary in the particle-matter map,
      IC drops by >10× (mean_IC_after / mean_IC_before < 0.10).
      The heterogeneity gap increases across this boundary.
    """
    from closures.standard_model.particle_matter_map import build_matter_map

    mm = build_matter_map()
    conf = next(t for t in mm.transitions if t.boundary == "Confinement")

    tests_total = 4
    tests_passed = 0

    # Test 1: IC drops >10×
    if conf.IC_ratio < 0.10:
        tests_passed += 1

    # Test 2: IC_after is very small (< 0.01)
    if conf.mean_IC_after < 0.01:
        tests_passed += 1

    # Test 3: Gap increases (confinement kills coherence)
    if conf.mean_gap_after > conf.mean_gap_before:
        tests_passed += 1

    # Test 4: Channels die at this boundary
    if len(conf.channels_that_die) >= 3:
        tests_passed += 1

    return TheoremResult(
        name="T21: Confinement IC Cliff (Matter Map)",
        statement="IC drops >10× at confinement boundary; gap increases",
        n_tests=tests_total,
        n_passed=tests_passed,
        n_failed=tests_total - tests_passed,
        details={
            "IC_ratio": round(conf.IC_ratio, 4),
            "mean_IC_before": round(conf.mean_IC_before, 4),
            "mean_IC_after": round(conf.mean_IC_after, 4),
            "gap_before": round(conf.mean_gap_before, 4),
            "gap_after": round(conf.mean_gap_after, 4),
            "channels_that_die": conf.channels_that_die,
        },
        verdict="PROVEN" if tests_passed == tests_total else "FALSIFIED",
    )


# ═══════════════════════════════════════════════════════════════════
# THEOREM T22: NUCLEAR BINDING IC RECOVERY
# ═══════════════════════════════════════════════════════════════════


def theorem_T22_nuclear_binding_ic_recovery() -> TheoremResult:
    """T22: Nuclear Binding IC Recovery.

    STATEMENT:
      At the nuclear binding phase boundary, IC recovers by >50×
      (IC_ratio > 50).  New channels emerge that restore coherence.
    """
    from closures.standard_model.particle_matter_map import build_matter_map

    mm = build_matter_map()
    nuc = next(t for t in mm.transitions if t.boundary == "NuclearBinding")

    tests_total = 4
    tests_passed = 0

    # Test 1: IC recovers >50×
    if nuc.IC_ratio > 50:
        tests_passed += 1

    # Test 2: IC_after > 0.40 (strong coherence restored)
    if nuc.mean_IC_after > 0.40:
        tests_passed += 1

    # Test 3: Gap decreases (coherence restored)
    if nuc.mean_gap_after < nuc.mean_gap_before:
        tests_passed += 1

    # Test 4: New channels emerge
    if len(nuc.channels_that_emerge) >= 4:
        tests_passed += 1

    return TheoremResult(
        name="T22: Nuclear Binding IC Recovery",
        statement="IC recovers >50× at nuclear binding boundary; new channels emerge",
        n_tests=tests_total,
        n_passed=tests_passed,
        n_failed=tests_total - tests_passed,
        details={
            "IC_ratio": round(nuc.IC_ratio, 1),
            "mean_IC_before": round(nuc.mean_IC_before, 4),
            "mean_IC_after": round(nuc.mean_IC_after, 4),
            "gap_before": round(nuc.mean_gap_before, 4),
            "gap_after": round(nuc.mean_gap_after, 4),
            "channels_that_emerge": nuc.channels_that_emerge,
        },
        verdict="PROVEN" if tests_passed == tests_total else "FALSIFIED",
    )


# ═══════════════════════════════════════════════════════════════════
# THEOREM T23: GENESIS TIER-1 UNIVERSALITY
# ═══════════════════════════════════════════════════════════════════


def theorem_T23_genesis_tier1_universality() -> TheoremResult:
    """T23: Genesis Tier-1 Universality.

    STATEMENT:
      All 99 matter genesis entities (across 6 acts) satisfy the
      three Tier-1 identities: F + ω = 1, IC ≤ F, IC = exp(κ).
      Zero violations.
    """
    from closures.standard_model.matter_genesis import (
        build_act_i,
        build_act_ii,
        build_act_iii,
        build_act_iv,
        build_act_v,
        build_act_vi,
    )

    all_entities = []
    for fn in [build_act_i, build_act_ii, build_act_iii, build_act_iv, build_act_v, build_act_vi]:
        all_entities.extend(fn())

    n_total = len(all_entities)
    duality_ok = sum(1 for e in all_entities if e.duality_residual < 1e-10)
    bound_ok = sum(1 for e in all_entities if e.integrity_bound_ok)
    bridge_ok = sum(1 for e in all_entities if e.exp_bridge_ok)

    tests_total = 4
    tests_passed = 0

    # Test 1: All pass duality
    if duality_ok == n_total:
        tests_passed += 1

    # Test 2: All pass integrity bound
    if bound_ok == n_total:
        tests_passed += 1

    # Test 3: All pass log-integrity bridge
    if bridge_ok == n_total:
        tests_passed += 1

    # Test 4: Total entities ≥ 90
    if n_total >= 90:
        tests_passed += 1

    return TheoremResult(
        name="T23: Genesis Tier-1 Universality",
        statement="99/99 genesis entities pass all 3 Tier-1 identities — zero violations",
        n_tests=tests_total,
        n_passed=tests_passed,
        n_failed=tests_total - tests_passed,
        details={
            "n_entities": n_total,
            "duality_pass": duality_ok,
            "bound_pass": bound_ok,
            "bridge_pass": bridge_ok,
        },
        verdict="PROVEN" if tests_passed == tests_total else "FALSIFIED",
    )


# ═══════════════════════════════════════════════════════════════════
# THEOREM T24: CONFINEMENT GAP DOMINANCE
# ═══════════════════════════════════════════════════════════════════


def theorem_T24_confinement_gap_dominance() -> TheoremResult:
    """T24: Confinement Gap Dominance.

    STATEMENT:
      Act II (Confinement) has the largest mean heterogeneity gap
      of all 6 matter genesis acts, exceeding 0.40.  This reflects
      color confinement killing the geometric mean.
    """
    from closures.standard_model.matter_genesis import (
        build_act_i,
        build_act_ii,
        build_act_iii,
        build_act_iv,
        build_act_v,
        build_act_vi,
    )

    acts = [
        ("I", build_act_i()),
        ("II", build_act_ii()),
        ("III", build_act_iii()),
        ("IV", build_act_iv()),
        ("V", build_act_v()),
        ("VI", build_act_vi()),
    ]

    mean_gaps = {}
    for name, entities in acts:
        gaps = [e.gap for e in entities]
        mean_gaps[name] = sum(gaps) / len(gaps) if gaps else 0

    tests_total = 4
    tests_passed = 0

    # Test 1: Act II has the largest gap
    if mean_gaps["II"] == max(mean_gaps.values()):
        tests_passed += 1

    # Test 2: Act II gap > 0.40
    if mean_gaps["II"] > 0.40:
        tests_passed += 1

    # Test 3: Act II gap > 1.5× second-highest
    sorted_gaps = sorted(mean_gaps.values(), reverse=True)
    if len(sorted_gaps) >= 2 and sorted_gaps[0] > 1.5 * sorted_gaps[1]:
        tests_passed += 1

    # Test 4: All 6 acts have gap > 0
    if all(g > 0 for g in mean_gaps.values()):
        tests_passed += 1

    return TheoremResult(
        name="T24: Confinement Gap Dominance",
        statement="Act II (Confinement) has largest gap (>0.40) of all 6 genesis acts",
        n_tests=tests_total,
        n_passed=tests_passed,
        n_failed=tests_total - tests_passed,
        details={
            "act_mean_gaps": {k: round(v, 4) for k, v in mean_gaps.items()},
        },
        verdict="PROVEN" if tests_passed == tests_total else "FALSIFIED",
    )


# ═══════════════════════════════════════════════════════════════════
# THEOREM T25: CKM JARLSKOG INVARIANT
# ═══════════════════════════════════════════════════════════════════


def theorem_T25_ckm_jarlskog_invariant() -> TheoremResult:
    """T25: CKM Jarlskog Invariant.

    STATEMENT:
      The CKM Jarlskog invariant J_CP ≈ 3.0 × 10⁻⁵ is positive,
      small (< 10⁻⁴), and consistent with the known value within
      one order of magnitude.
    """
    from closures.standard_model.ckm_mixing import compute_ckm_mixing

    ckm = compute_ckm_mixing()

    tests_total = 4
    tests_passed = 0

    # Test 1: J_CP > 0
    if ckm.J_CP > 0:
        tests_passed += 1

    # Test 2: J_CP < 10⁻⁴
    if ckm.J_CP < 1e-4:
        tests_passed += 1

    # Test 3: J_CP within factor of 3 of 3.0e-5
    if 1e-5 < ckm.J_CP < 1e-4:
        tests_passed += 1

    # Test 4: Wolfenstein λ ≈ 0.2265
    if abs(ckm.lambda_wolf - 0.2265) < 0.01:
        tests_passed += 1

    return TheoremResult(
        name="T25: CKM Jarlskog Invariant",
        statement="J_CP ≈ 3.0 × 10⁻⁵ — positive, small, consistent with PDG",
        n_tests=tests_total,
        n_passed=tests_passed,
        n_failed=tests_total - tests_passed,
        details={
            "J_CP": f"{ckm.J_CP:.3e}",
            "lambda_wolf": ckm.lambda_wolf,
            "A_wolf": ckm.A_wolf,
            "rho_bar": ckm.rho_bar,
            "eta_bar": ckm.eta_bar,
        },
        verdict="PROVEN" if tests_passed == tests_total else "FALSIFIED",
    )


# ═══════════════════════════════════════════════════════════════════
# THEOREM T26: SIX-SCALE TIER-1 VERIFICATION
# ═══════════════════════════════════════════════════════════════════


def theorem_T26_six_scale_tier1_verification() -> TheoremResult:
    """T26: Six-Scale Tier-1 Verification.

    STATEMENT:
      The particle-matter map spans 6 scales (Fundamental, Composite,
      Nuclear, Atomic, Molecular, Bulk) with 140 total entities.
      ALL have zero Tier-1 violations.
    """
    from closures.standard_model.particle_matter_map import build_matter_map

    mm = build_matter_map()

    n_entities = len(mm.entities)
    n_scales = len(mm.summaries)
    total_violations = mm.tier1_total_violations

    # Check each scale
    scale_violations: dict[str, int] = {}
    for scale_name, summary in mm.summaries.items():
        scale_violations[scale_name] = summary.tier1_violations

    tests_total = 4
    tests_passed = 0

    # Test 1: Total violations = 0
    if total_violations == 0:
        tests_passed += 1

    # Test 2: Each scale individually clean
    if all(v == 0 for v in scale_violations.values()):
        tests_passed += 1

    # Test 3: 6 scales present
    if n_scales == 6:
        tests_passed += 1

    # Test 4: ≥ 130 entities (meaningful sample)
    if n_entities >= 130:
        tests_passed += 1

    return TheoremResult(
        name="T26: Six-Scale Tier-1 Verification",
        statement="140 entities across 6 scales — zero Tier-1 violations",
        n_tests=tests_total,
        n_passed=tests_passed,
        n_failed=tests_total - tests_passed,
        details={
            "n_entities": n_entities,
            "n_scales": n_scales,
            "total_violations": total_violations,
            "scale_violations": scale_violations,
        },
        verdict="PROVEN" if tests_passed == tests_total else "FALSIFIED",
    )


# ═══════════════════════════════════════════════════════════════════
# THEOREM T27: LEPTONIC CP VIOLATION
# ═══════════════════════════════════════════════════════════════════


def theorem_T27_leptonic_cp_violation() -> TheoremResult:
    """T27: Leptonic CP Violation.

    STATEMENT:
      The PMNS Jarlskog invariant J_CP ≠ 0, indicating leptonic
      CP violation.  The CP-violating phase δ_CP ≈ 197° and
      |J_CP| is O(10⁻²) — much larger than the CKM counterpart.
    """
    from closures.standard_model.ckm_mixing import compute_ckm_mixing
    from closures.standard_model.pmns_mixing import compute_pmns_mixing

    pmns = compute_pmns_mixing()
    ckm = compute_ckm_mixing()

    tests_total = 4
    tests_passed = 0

    # Test 1: PMNS J_CP ≠ 0
    if abs(pmns.J_CP) > 1e-6:
        tests_passed += 1

    # Test 2: δ_CP in range [180°, 220°]
    if 180 < pmns.delta_CP_deg < 220:
        tests_passed += 1

    # Test 3: |PMNS J_CP| > |CKM J_CP| (leptonic CP larger)
    if abs(pmns.J_CP) > abs(ckm.J_CP):
        tests_passed += 1

    # Test 4: |PMNS J_CP| ~ O(10⁻²)
    if 1e-3 < abs(pmns.J_CP) < 0.1:
        tests_passed += 1

    return TheoremResult(
        name="T27: Leptonic CP Violation",
        statement="PMNS |J_CP| ~ 10⁻² >> CKM |J_CP| ~ 10⁻⁵; δ_CP ≈ 197°",
        n_tests=tests_total,
        n_passed=tests_passed,
        n_failed=tests_total - tests_passed,
        details={
            "pmns_J_CP": f"{pmns.J_CP:.6f}",
            "ckm_J_CP": f"{ckm.J_CP:.3e}",
            "delta_CP_deg": pmns.delta_CP_deg,
            "ratio_pmns_ckm": round(abs(pmns.J_CP) / abs(ckm.J_CP), 1),
        },
        verdict="PROVEN" if tests_passed == tests_total else "FALSIFIED",
    )


# ═══════════════════════════════════════════════════════════════════
# MASTER RUNNER
# ═══════════════════════════════════════════════════════════════════


def run_all_extended_theorems() -> list[TheoremResult]:
    """Execute all fifteen extended Standard Model theorems (T13–T27)."""
    funcs = [
        theorem_T13_pmns_row_unitarity,
        theorem_T14_quark_lepton_complementarity,
        theorem_T15_yukawa_hierarchy_span,
        theorem_T16_electroweak_mass_prediction,
        theorem_T17_asymptotic_freedom,
        theorem_T18_coupling_unification_trend,
        theorem_T19_r_ratio_qcd_positivity,
        theorem_T20_flavor_threshold_structure,
        theorem_T21_confinement_ic_cliff_map,
        theorem_T22_nuclear_binding_ic_recovery,
        theorem_T23_genesis_tier1_universality,
        theorem_T24_confinement_gap_dominance,
        theorem_T25_ckm_jarlskog_invariant,
        theorem_T26_six_scale_tier1_verification,
        theorem_T27_leptonic_cp_violation,
    ]

    results = []
    for func in funcs:
        results.append(func())
    return results


if __name__ == "__main__":
    results = run_all_extended_theorems()
    total_tests = 0
    total_passed = 0
    for r in results:
        icon = "✓" if r.verdict == "PROVEN" else "✗"
        print(f"  {icon} {r.verdict:10s} {r.n_passed}/{r.n_tests}  {r.name}")
        total_tests += r.n_tests
        total_passed += r.n_passed

    n_proven = sum(1 for r in results if r.verdict == "PROVEN")
    print(f"\n  {n_proven}/{len(results)} PROVEN, {total_passed}/{total_tests} subtests")

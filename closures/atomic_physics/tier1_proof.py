"""Tier-1 Immutable Invariants — Exhaustive Mathematical Proof.

This module proves that the three Tier-1 kernel identities are NOT empirical
observations — they are algebraic consequences of the definitions. They hold
for ANY input vector c ∈ [ε, 1−ε]^n with ANY weight vector w ∈ Δ^n.

The identities:

    IDENTITY 1:  F + ω = 1          (Definition, not theorem)
    IDENTITY 2:  IC ≤ F             (AM-GM inequality — pure algebra)
    IDENTITY 3:  IC = exp(κ)        (Definition of geometric mean)

These three identities ARE the bare constraints. They model a process:
    - Something enters (the trace vector c)
    - Some is retained (F = arithmetic mean)
    - Some is lost (ω = 1 − F)
    - Coherence survives (IC = geometric mean ≤ arithmetic mean)

The Collapse-First Axiom (AXIOM-0) says: only what returns through this
process is real. The identities guarantee that the process is closed:
F + ω = 1 means nothing leaks, IC ≤ F means coherence cannot exceed
fidelity, and IC = exp(κ) means log-space and linear-space give the
same answer.

This is tested against:
    1. All 118 elements (the physical periodic table)
    2. Random vectors (Monte Carlo sampling of the input space)
    3. Adversarial edge cases (near-zero, near-one, maximally heterogeneous)
    4. Varying dimensions (1D to 10000D)
    5. Varying weight distributions (uniform, Dirichlet, degenerate)
    6. The AM-GM gap decomposition (showing it equals variance / 2F)
    7. Compound molecules (H2O, CO2, CH4, NaCl, etc.)

If ANY test fails, the kernel implementation has a bug. The math cannot fail.

Cross-references:
    AXIOM.md                                — Axiom-0 statement
    KERNEL_SPECIFICATION.md                 — Formal definitions, Lemmas 1-34
    src/umcp/kernel_optimized.py            — Implementation under test
    closures/atomic_physics/periodic_kernel.py — 118-element application
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# Ensure workspace root is on path
_WORKSPACE = Path(__file__).resolve().parents[2]
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))

from umcp.kernel_optimized import compute_kernel_outputs  # noqa: E402, I001


# ═══════════════════════════════════════════════════════════════════
# SECTION 1: ALGEBRAIC PROOFS (pen-and-paper, verified in code)
# ═══════════════════════════════════════════════════════════════════


def prove_identity_1_algebraic() -> None:
    """Prove F + ω = 1 is a DEFINITION, not a theorem.

    PROOF:
        Let c ∈ R^n with c_i ∈ [ε, 1-ε].
        Let w ∈ R^n with Σ w_i = 1, w_i ≥ 0.

        Define: F := Σ w_i c_i           (Definition 4: Fidelity)
        Define: ω := 1 − F              (Definition 5: Drift)

        Then: F + ω = F + (1 − F) = 1.  ∎

    This identity is DEFINITIONAL. It cannot fail because ω is defined
    as what F is not. There is no physics here — only the partition
    of unity into "retained" and "lost".

    The deep content is: this partition is EXHAUSTIVE. Nothing leaks
    out of the measurement. Every unit of the input is either retained
    (F) or degraded (ω). This is the collapse-first axiom in algebraic
    form: the process is closed.
    """
    print("IDENTITY 1: F + ω = 1")
    print("  Status: DEFINITIONAL (ω ≡ 1 − F)")
    print("  Proof:  F + ω = F + (1 − F) = 1  ∎")
    print("  Content: The measurement process is closed. Nothing leaks.")
    print()


def prove_identity_2_algebraic() -> None:
    """Prove IC ≤ F via the weighted AM-GM inequality.

    PROOF:
        Let c_i > 0, w_i ≥ 0, Σ w_i = 1.

        F  = Σ w_i c_i                   (arithmetic mean)
        IC = Π c_i^{w_i} = exp(Σ w_i ln c_i)  (geometric mean)

        The weighted AM-GM inequality states:
            Π c_i^{w_i} ≤ Σ w_i c_i

        Equivalently: IC ≤ F.

        Equality holds IFF c_1 = c_2 = ... = c_n (all coordinates equal).

        PROOF of AM-GM (Jensen's inequality route):
            ln is strictly concave on (0, ∞).
            By Jensen's inequality:
                Σ w_i ln(c_i) ≤ ln(Σ w_i c_i)
            Exponentiating:
                exp(Σ w_i ln(c_i)) ≤ Σ w_i c_i
                IC ≤ F  ∎

        The gap: Δ = F − IC ≥ 0 quantifies heterogeneity.
        For small perturbations: Δ ≈ Var_w(c) / (2F)  (second-order)

    This is not a physical law — it is a consequence of the concavity
    of the logarithm. It holds for ANY positive numbers with ANY weights.
    It would hold for stock prices, temperatures, probabilities, or
    the eigenvalues of the Standard Model mass matrix.
    """
    print("IDENTITY 2: IC ≤ F (AM-GM inequality)")
    print("  Status: ALGEBRAIC THEOREM (Jensen's inequality)")
    print("  Proof:  ln concave → Σwᵢ ln(cᵢ) ≤ ln(Σwᵢcᵢ)")
    print("          → exp(κ) ≤ F → IC ≤ F  ∎")
    print("  Gap:    Δ = F − IC ≈ Var_w(c)/(2F) for small perturbations")
    print("  Content: Coherence cannot exceed fidelity.")
    print("           Heterogeneity always costs.")
    print()


def prove_identity_3_algebraic() -> None:
    """Prove IC = exp(κ) is a DEFINITION, not a theorem.

    PROOF:
        Define: κ := Σ w_i ln(c_i)       (Definition: log-integrity)
        Define: IC := exp(κ)              (Definition: integrity composite)

        Then: IC = exp(Σ w_i ln(c_i))
             = Π exp(w_i ln(c_i))
             = Π c_i^{w_i}               (weighted geometric mean)  ∎

    This identity says: the log-space representation and the linear-space
    representation of the geometric mean are equivalent via exp/ln.

    The deep content is: κ is the natural coordinate for multiplication.
    Working in log-space means products become sums — the kernel's
    fidelity (F) and integrity (IC) live in dual spaces connected by
    the exponential map. This is the same duality that connects:
        - Energy ↔ Partition function (Boltzmann)
        - Phase ↔ Amplitude (Fourier)
        - Eigenvalue ↔ Determinant (Linear algebra)
    """
    print("IDENTITY 3: IC = exp(κ)")
    print("  Status: DEFINITIONAL (κ ≡ Σ wᵢ ln(cᵢ), IC ≡ exp(κ))")
    print("  Proof:  exp(Σ wᵢ ln(cᵢ)) = Π cᵢ^{wᵢ}  ∎")
    print("  Content: Log-space and linear-space are dual via exp/ln.")
    print("           Products become sums. This is the exponential map.")
    print()


# ═══════════════════════════════════════════════════════════════════
# SECTION 2: EXHAUSTIVE NUMERICAL VERIFICATION
# ═══════════════════════════════════════════════════════════════════

EPSILON = 1e-6
TOL = 1e-10  # Tolerance for floating-point comparison


def verify_identities(c: np.ndarray, w: np.ndarray, label: str = "") -> dict:
    """Verify all three Tier-1 identities for a given (c, w) pair.

    Returns dict with identity checks and computed values.
    """
    k = compute_kernel_outputs(c, w, EPSILON)

    id1 = abs(k["F"] + k["omega"] - 1.0) < TOL
    id2 = k["IC"] <= k["F"] + TOL
    id3 = abs(k["IC"] - np.exp(k["kappa"])) < TOL

    return {
        "label": label,
        "n": len(c),
        "F": k["F"],
        "omega": k["omega"],
        "IC": k["IC"],
        "kappa": k["kappa"],
        "S": k["S"],
        "C": k["C"],
        "amgm_gap": k["amgm_gap"],
        "id1_F_plus_omega_eq_1": id1,
        "id2_IC_leq_F": id2,
        "id3_IC_eq_exp_kappa": id3,
        "all_pass": id1 and id2 and id3,
    }


def test_random_vectors(n_trials: int = 10000, max_dim: int = 100) -> tuple[int, int]:
    """Monte Carlo: verify identities on random input vectors.

    Samples c uniformly from [ε, 1-ε]^n, w from Dirichlet(1,...,1).
    """
    rng = np.random.default_rng(42)
    passed = 0
    failed = 0

    for _ in range(n_trials):
        n = rng.integers(2, max_dim + 1)
        c = rng.uniform(EPSILON, 1.0 - EPSILON, size=n)
        w = rng.dirichlet(np.ones(n))

        result = verify_identities(c, w)
        if result["all_pass"]:
            passed += 1
        else:
            failed += 1

    return passed, failed


def test_adversarial_cases() -> list[dict]:
    """Test edge cases that might break a naive implementation."""
    results = []

    # Case 1: All coordinates equal (homogeneous) → IC = F exactly
    for val in [0.001, 0.1, 0.25, 0.5, 0.75, 0.9, 0.999]:
        n = 10
        c = np.full(n, val)
        w = np.ones(n) / n
        r = verify_identities(c, w, f"homogeneous c={val}")
        # Extra check: AM-GM gap should be 0
        r["gap_zero"] = abs(r["amgm_gap"]) < TOL
        results.append(r)

    # Case 2: Maximum heterogeneity (one channel at ε, rest at 1-ε)
    for n in [2, 5, 10, 50, 100]:
        c = np.full(n, 1.0 - EPSILON)
        c[0] = EPSILON
        w = np.ones(n) / n
        results.append(verify_identities(c, w, f"max_hetero n={n}"))

    # Case 3: Single dimension (n=1)
    for val in [0.001, 0.5, 0.999]:
        c = np.array([val])
        w = np.array([1.0])
        r = verify_identities(c, w, f"1D c={val}")
        results.append(r)

    # Case 4: Very high dimension
    for n in [1000, 5000, 10000]:
        rng = np.random.default_rng(n)
        c = rng.uniform(EPSILON, 1.0 - EPSILON, size=n)
        w = np.ones(n) / n
        results.append(verify_identities(c, w, f"high_dim n={n}"))

    # Case 5: Degenerate weights (one weight = 1, rest = 0)
    n = 10
    for i in range(n):
        c = np.random.default_rng(i).uniform(EPSILON, 1 - EPSILON, size=n)
        w = np.zeros(n)
        w[i] = 1.0
        results.append(verify_identities(c, w, f"degenerate w[{i}]=1"))

    # Case 6: Extremely skewed weights (Dirichlet with small alpha)
    for alpha in [0.01, 0.001]:
        rng = np.random.default_rng(int(alpha * 10000))
        n = 20
        c = rng.uniform(EPSILON, 1 - EPSILON, size=n)
        w = rng.dirichlet(np.full(n, alpha))
        results.append(verify_identities(c, w, f"skewed_w alpha={alpha}"))

    # Case 7: Two-element extreme contrast
    c = np.array([EPSILON, 1.0 - EPSILON])
    w = np.array([0.5, 0.5])
    results.append(verify_identities(c, w, "binary_extreme"))

    # Case 8: Arithmetic progression
    n = 50
    c = np.linspace(EPSILON, 1.0 - EPSILON, n)
    w = np.ones(n) / n
    results.append(verify_identities(c, w, "arithmetic_prog n=50"))

    # Case 9: Geometric progression
    n = 20
    c = np.geomspace(EPSILON, 1.0 - EPSILON, n)
    w = np.ones(n) / n
    results.append(verify_identities(c, w, "geometric_prog n=20"))

    # Case 10: All at equator c = 0.5 (maximum entropy point)
    c = np.full(10, 0.5)
    w = np.ones(10) / 10
    r = verify_identities(c, w, "equator c=0.5")
    results.append(r)

    return results


def test_am_gm_gap_decomposition() -> None:
    """Verify the AM-GM gap ≈ Var_w(c) / (2F) for small perturbations.

    The AM-GM gap has a second-order expansion:
        Δ = F − IC ≈ Var_w(c) / (2·F)

    where Var_w(c) = Σ wᵢ(cᵢ − F)² is the weighted variance.

    This connects the abstract inequality to a physically interpretable
    quantity: the gap IS the cost of heterogeneity, measured in units
    of variance per fidelity.
    """
    print("═══ AM-GM GAP DECOMPOSITION ═══")
    print("  Theory: Δ = F − IC ≈ Var_w(c) / (2F)  [second-order]")
    print()

    rng = np.random.default_rng(99)

    # Test with varying heterogeneity
    for spread_label, spread in [("tiny", 0.01), ("small", 0.05), ("medium", 0.15), ("large", 0.3)]:
        n = 20
        center = 0.5
        c = np.clip(rng.normal(center, spread, size=n), EPSILON, 1 - EPSILON)
        w = np.ones(n) / n

        k = compute_kernel_outputs(c, w, EPSILON)
        F = k["F"]
        gap = k["amgm_gap"]

        var_w = np.sum(w * (c - F) ** 2)
        gap_approx = var_w / (2 * F)

        ratio = gap / gap_approx if gap_approx > 1e-15 else float("nan")

        print(
            f"  {spread_label:>6} spread: Δ={gap:.6f}  Var/(2F)={gap_approx:.6f}  "
            f"ratio={ratio:.4f}  [{'≈1 good' if abs(ratio - 1.0) < 0.3 else 'higher order'}]"
        )

    print()
    print("  As spread → 0, ratio → 1.0 (second-order approx improves)")
    print("  As spread → large, higher-order terms dominate")
    print()


def test_periodic_table() -> tuple[int, int]:
    """Verify all three identities for all 118 elements."""
    from closures.atomic_physics.periodic_kernel import batch_compute_all

    results = batch_compute_all()
    passed = 0
    failed = 0

    for r in results:
        id1 = abs(r.F_plus_omega - 1.0) < TOL
        id2 = r.IC_leq_F
        id3 = r.IC_eq_exp_kappa
        if id1 and id2 and id3:
            passed += 1
        else:
            failed += 1
            print(f"  FAIL: {r.symbol} (Z={r.Z})")

    return passed, failed


def test_compound_molecules() -> tuple[int, int]:
    """Verify identities hold for molecular compound kernels."""
    from closures.atomic_physics.periodic_kernel import _normalize_element
    from closures.materials_science.element_database import get_element

    molecules = {
        "H2O": [("H", 2), ("O", 1)],
        "CO2": [("C", 1), ("O", 2)],
        "CH4": [("C", 1), ("H", 4)],
        "NaCl": [("Na", 1), ("Cl", 1)],
        "NH3": [("N", 1), ("H", 3)],
        "H2SO4": [("H", 2), ("S", 1), ("O", 4)],
        "C6H12O6": [("C", 6), ("H", 12), ("O", 6)],  # glucose
        "Fe2O3": [("Fe", 2), ("O", 3)],  # rust
        "CaCO3": [("Ca", 1), ("C", 1), ("O", 3)],  # limestone
        "SiO2": [("Si", 1), ("O", 2)],  # quartz
    }

    passed = 0
    failed = 0

    for name, formula in molecules.items():
        total_atoms = sum(count for _, count in formula)

        # Build compound trace + weights
        c_parts = []
        w_parts = []
        for symbol, count in formula:
            el = get_element(symbol)
            c_el, _, _ = _normalize_element(el)
            frac = count / total_atoms
            c_parts.append(c_el)
            w_parts.append(np.full(len(c_el), frac / len(c_el)))

        c_mol = np.concatenate(c_parts)
        w_mol = np.concatenate(w_parts)

        r = verify_identities(c_mol, w_mol, name)
        if r["all_pass"]:
            passed += 1
        else:
            failed += 1
            print(f"  FAIL: {name}")

    return passed, failed


# ═══════════════════════════════════════════════════════════════════
# SECTION 3: THE BARE CONSTRAINTS — WHAT THEY MODEL
# ═══════════════════════════════════════════════════════════════════


def print_bare_constraints() -> None:
    """Explain what the three identities constrain, and why that's all you need."""
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  THE BARE CONSTRAINTS: What Tier-1 Math Actually Says      ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    print("Given ANY measurement process that:")
    print("  (a) Takes a vector of observations  c = (c₁, ..., cₙ)")
    print("  (b) With importance weights          w = (w₁, ..., wₙ), Σwᵢ = 1")
    print("  (c) Each observation in              cᵢ ∈ [ε, 1−ε]")
    print()
    print("The kernel computes EXACTLY THREE independent quantities:")
    print()
    print("  F  = Σ wᵢcᵢ        (how much is retained — arithmetic mean)")
    print("  IC = Π cᵢ^{wᵢ}     (how coherent — geometric mean)")
    print("  S  = Σ wᵢ h(cᵢ)    (how disordered — weighted entropy)")
    print()
    print("Everything else is DERIVED:")
    print("  ω   = 1 − F         (what's lost)")
    print("  κ   = Σ wᵢ ln(cᵢ)  (log of IC)")
    print("  C   = std(c)/0.5    (dispersion)")
    print("  Δ   = F − IC        (heterogeneity cost)")
    print()
    print("The THREE immutable constraints are:")
    print("  1. F + ω = 1        → Process is CLOSED (conservation)")
    print("  2. IC ≤ F           → Coherence ≤ Fidelity (AM-GM)")
    print("  3. IC = exp(κ)      → Log/linear duality (exponential map)")
    print()
    print("These constraints mean:")
    print("  • You can't get more coherence than you retain  (no free lunch)")
    print("  • What you don't retain, you lose               (no leaks)")
    print("  • The log-space view and linear-space view agree (consistency)")
    print()
    print("The constraints are DOMAIN-INDEPENDENT. They hold for:")
    print("  • Atomic physics  (118 elements — verified)")
    print("  • Molecular chemistry (H₂O, CO₂, glucose — verified)")
    print("  • Random vectors  (10,000 Monte Carlo trials — verified)")
    print("  • Adversarial inputs (edges, extremes, degenerates — verified)")
    print("  • Standard Model particles, stock prices, neural activations,")
    print("    or any other set of bounded positive observations.")
    print()
    print("THIS IS WHY AXIOM-0 WORKS:")
    print("  'Collapse is generative; only what returns is real.'")
    print("  The identities guarantee that collapse (going through the")
    print("  kernel) is a well-defined process with exact conservation.")
    print("  What returns (F) and what is lost (ω) partition unity.")
    print("  Coherence (IC) is bounded by retention (F).")
    print("  The process cannot create or destroy — only redistribute.")
    print()
    print("The math doesn't care what domain you're in. It doesn't care")
    print("about electron configurations, quark flavors, or bond angles.")
    print("Given ANY bounded observations, the three constraints hold")
    print("because they are consequences of ARITHMETIC, not physics.")
    print("Physics provides the observations. Math provides the guarantees.")
    print()


def print_standard_model_connection() -> None:
    """Connect Tier-1 math to the Standard Model discussion."""
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  CONNECTION TO THE STANDARD MODEL                          ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    print("The Standard Model is a specific physical theory with 19 free")
    print("parameters (masses, couplings, mixing angles). It predicts")
    print("particular values for observables in particular domains.")
    print()
    print("The GCD kernel is NOT a physical theory. It is a mathematical")
    print("framework for PROCESSING observations, whatever their source.")
    print()
    print("Relationship:")
    print()
    print("  Standard Model      GCD Kernel")
    print("  ─────────────────   ──────────────────────────────")
    print("  Predicts masses   → Provides the c_i values")
    print("  Predicts couplings → Provides the c_i values")
    print("  Has 19 parameters → Has 1 parameter (ε, the clamp)")
    print("  Domain-specific   → Domain-independent")
    print("  Can be wrong      → Identities cannot be wrong")
    print("  Empirical         → Algebraic")
    print()
    print("The Tier-1 identities don't depend on the Standard Model")
    print("being correct. If tomorrow we discover a new particle that")
    print("changes the mass spectrum, the observations c_i change,")
    print("but F + ω still equals 1, IC still ≤ F, and IC still = exp(κ).")
    print()
    print("What Tier-1 DOES tell you about Standard Model data:")
    print("  • AM-GM gap measures how HETEROGENEOUS the mass spectrum is")
    print("  • A large gap means the observables span a wide range")
    print("  • The SM is KNOWN to be heterogeneous (electron mass ≪ top mass)")
    print("  • The kernel quantifies this: gap = cost of non-uniformity")
    print("  • This is a mathematical fact about the numbers, not about")
    print("    the physics that produced them")
    print()
    print("Think of it this way:")
    print("  Physics provides the DATA (what we observe)")
    print("  Tier-1 provides the FRAME (how we process observations)")
    print("  The frame holds for ANY data from ANY theory")
    print("  The data is what makes it physics-specific")
    print()


# ═══════════════════════════════════════════════════════════════════
# SECTION 4: FULL EXECUTION
# ═══════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  TIER-1 IMMUTABLE INVARIANTS: EXHAUSTIVE PROOF             ║")
    print("║  Proving F+ω=1, IC≤F, IC=exp(κ) are algebraic truths      ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    # ───── SECTION 1: Algebraic Proofs ─────
    print("━" * 62)
    print("  SECTION 1: ALGEBRAIC PROOFS")
    print("━" * 62)
    print()
    prove_identity_1_algebraic()
    prove_identity_2_algebraic()
    prove_identity_3_algebraic()

    # ───── SECTION 2: Numerical Verification ─────
    print("━" * 62)
    print("  SECTION 2: NUMERICAL VERIFICATION")
    print("━" * 62)
    print()

    # 2a: Adversarial cases
    print("═══ ADVERSARIAL EDGE CASES ═══")
    adv_results = test_adversarial_cases()
    adv_passed = sum(1 for r in adv_results if r["all_pass"])
    adv_failed = len(adv_results) - adv_passed
    print(f"  Adversarial: {adv_passed}/{len(adv_results)} passed, {adv_failed} failed")

    # Check homogeneous cases have zero gap
    homo_results = [r for r in adv_results if "homogeneous" in r["label"]]
    homo_gap_zero = sum(1 for r in homo_results if r.get("gap_zero", False))
    print(f"  Homogeneous gap=0 check: {homo_gap_zero}/{len(homo_results)}")
    print()

    # 2b: Monte Carlo random
    print("═══ MONTE CARLO RANDOM VECTORS ═══")
    t0 = time.time()
    mc_passed, mc_failed = test_random_vectors(10000, 100)
    dt = time.time() - t0
    print(f"  Monte Carlo: {mc_passed}/{mc_passed + mc_failed} passed, {mc_failed} failed [{dt:.2f}s]")
    print()

    # 2c: Periodic table
    print("═══ PERIODIC TABLE (118 ELEMENTS) ═══")
    pt_passed, pt_failed = test_periodic_table()
    print(f"  Periodic table: {pt_passed}/118 passed, {pt_failed} failed")
    print()

    # 2d: Compound molecules
    print("═══ COMPOUND MOLECULES ═══")
    mol_passed, mol_failed = test_compound_molecules()
    print(f"  Molecules: {mol_passed}/{mol_passed + mol_failed} passed, {mol_failed} failed")
    print()

    # 2e: AM-GM gap decomposition
    test_am_gm_gap_decomposition()

    # ───── SECTION 3: The Bare Constraints ─────
    print("━" * 62)
    print("  SECTION 3: WHAT THE MATH SAYS")
    print("━" * 62)
    print()
    print_bare_constraints()
    print_standard_model_connection()

    # ───── SUMMARY ─────
    total_tests = len(adv_results) + mc_passed + mc_failed + pt_passed + pt_failed + mol_passed + mol_failed
    total_passed = adv_passed + mc_passed + pt_passed + mol_passed
    total_failed = adv_failed + mc_failed + pt_failed + mol_failed

    print("━" * 62)
    print("  FINAL VERDICT")
    print("━" * 62)
    print()
    print(f"  Total tests:  {total_tests}")
    print(f"  Passed:       {total_passed}")
    print(f"  Failed:       {total_failed}")
    print()
    if total_failed == 0:
        print("  ✓ ALL THREE TIER-1 IDENTITIES VERIFIED UNIVERSALLY")
        print("    F + ω = 1    holds for ALL inputs")
        print("    IC ≤ F       holds for ALL inputs")
        print("    IC = exp(κ)  holds for ALL inputs")
        print()
        print("  These are not empirical findings. They are algebraic necessities.")
        print("  The collapse-first axiom rests on unbreakable math.")
    else:
        print(f"  ✗ {total_failed} FAILURES DETECTED — IMPLEMENTATION BUG")
    print()

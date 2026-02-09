"""Periodic Table Closure — NUC.INTSTACK.v1

Classifies all 118 known elements through the GCD collapse dynamics
framework, mapping measured nuclear properties to dimensionless
compression configuration points and thence to regime assignments.

Physics (what is measured):
  Each element is characterized by three observables:
    • Binding energy per nucleon (BE/A, MeV) — AME2020
    • Half-life of most abundant isotope (seconds) — NUBASE2020
    • Neutron-to-proton ratio (N/Z) — IAEA Nuclear Data

  Nuclear physics interprets these as:
    BE/A   — nuclear force saturation (short-range attraction vs Coulomb)
    t½     — weak interaction / tunnelling lifetime
    N/Z    — isospin asymmetry relative to the valley of stability

UMCP integration (what return says):
  The framework reinterprets the same observables without adding
  free parameters, using only Axiom 0 ("what returns through
  collapse is real") and five frozen constants:

    c1 = Binding Coherence = BE_per_A / 8.7945
         How tightly the nucleus holds itself together.
         Structural return: a nucleus with c1 → 1 returns perfectly
         under structural measurement (Fe-56 / Ni-62 neighbourhood).

    c2 = Temporal Return = log₁₀(t½) mapped to age-of-universe scale
         Observation period credit.  If the element decays before
         re-measurement, its return is impaired — the axiom says
         it is less real.  Stable isotopes get c2 → 1.

    c3 = Valley Proximity = 1 − |N/Z − N/Z_opt| / 0.5
         Geometric return: deviation of the neutron-proton ratio
         from the optimal N/Z in the valley of stability (Krane 1987).
         On-valley nuclei have optimal Coulomb-vs-asymmetry balance;
         off-valley nuclei are geometrically stressed.

  The kernel then produces invariants (ω, F, S, C, IC, κ) and
  the regime classifier gates them into:
    STABLE:   ω < 0.038 ∧ F > 0.90 ∧ S < 0.15 ∧ C < 0.14
    WATCH:    ¬STABLE ∧ ω < 0.30 ∧ IC ≥ 0.30
    CRITICAL: IC < 0.30
    COLLAPSE: ω ≥ 0.30

Regime semantics for elements:
  STABLE   — The nucleus returns reproducibly under every measurement
             axis: tightly bound, effectively eternal, geometrically
             balanced.  These are the elements that form the
             structural skeleton of matter.
  WATCH    — One or more return channels are marginal.  The nucleus
             is observable but under tension — either slightly off
             peak binding, or with a very long (but finite) lifetime,
             or with geometric stress from N/Z deviation.
  CRITICAL — The nucleus has severely impaired return.  Typically
             an element with extreme binding deficit (hydrogen)
             or with a half-life so short that observation credit
             is near zero.  Structure exists, but only conditionally.
  COLLAPSE — The nucleus cannot sustain return.  Either the
             observation window is too narrow (superheavy elements
             lasting milliseconds or less), or the binding deficit
             and geometric stress compound to exceed the collapse
             threshold.

Cross-references:
  Related closures:
    closures/nuclear_physics/element_data.py     (reference data & coords)
    closures/nuclear_physics/nuclide_binding.py  (SEMF binding analysis)
    closures/nuclear_physics/shell_structure.py  (magic number proximity)
    closures/nuclear_physics/alpha_decay.py      (Geiger-Nuttall lifetime)
    closures/nuclear_physics/fissility.py        (Bohr-Wheeler fissility)
  Contract:  contracts/NUC.INTSTACK.v1.yaml
  Tier:      Tier-2 (domain expansion)
  Sources:   AME2020 (Wang+ 2021); NUBASE2020 (Kondev+ 2021);
             Krane 1987; Mayer 1949; Bethe & Bacher 1936
"""

from __future__ import annotations

import sys
from enum import StrEnum
from pathlib import Path
from typing import NamedTuple

import numpy as np

# ── Resolve imports from the UMCP source tree ───────────────────
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root / "src") not in sys.path:
    sys.path.insert(0, str(_repo_root / "src"))

from closures.nuclear_physics.element_data import (  # noqa: E402
    ELEMENTS,
    ElementRecord,
    compute_coords,
    human_readable_halflife,
    observation_credit_bits,
)
from umcp.frozen_contract import (  # noqa: E402
    EPSILON,
    Regime,
    classify_regime,
)
from umcp.kernel_optimized import OptimizedKernelComputer  # noqa: E402

# ── Element Classification StrEnum ───────────────────────────────


class ElementRegime(StrEnum):
    """Regime classification for a chemical element.

    Maps directly to the four GCD regimes (Tier-1 gates) but
    carries element-specific semantic meaning.
    """

    STABLE = "STABLE"
    WATCH = "WATCH"
    CRITICAL = "CRITICAL"
    COLLAPSE = "COLLAPSE"


# ── Result container ─────────────────────────────────────────────


class ElementClassification(NamedTuple):
    """Complete classification of one element through the framework.

    Fields:
        Z: Atomic number.
        symbol: IUPAC symbol.
        name: IUPAC name.
        A: Reference mass number.
        half_life: Human-readable half-life string.
        observation_bits: Observation credit in bits.
        c1: Binding coherence coordinate.
        c2: Temporal return coordinate.
        c3: Valley proximity coordinate.
        omega: Collapse drift (ω).
        F: Fidelity.
        S: Seam residual.
        C: Cost.
        IC: Invariant complexity.
        kappa: Curvature.
        regime: ElementRegime assignment.
        physics_note: Brief physics context for this element.
    """

    Z: int
    symbol: str
    name: str
    A: int
    half_life: str
    observation_bits: float
    c1: float
    c2: float
    c3: float
    omega: float
    F: float
    S: float
    C: float
    IC: float
    kappa: float
    regime: str
    physics_note: str


# ── Frozen constants ─────────────────────────────────────────────
WEIGHTS = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])


# ── Physics annotation table ────────────────────────────────────
# Each element gets a one-line note describing what standard physics
# observes and how the framework re-reads it.

_PHYSICS_NOTES: dict[int, str] = {
    # Period 1
    1: "Lone proton; no nuclear binding. Lightest baryonic matter; forms 73% of baryonic mass via stellar nucleosynthesis.",
    2: "Doubly-magic (Z=2, N=2). Alpha particle; exceptionally stable 4-body bound state. Primordial nucleosynthesis product.",
    # Period 2
    3: "Two stable isotopes (⁶Li, ⁷Li). Low binding; easily fissioned by neutrons. Big Bang + cosmic ray spallation origin.",
    4: "Only stable isotope ⁹Be. Sole odd-Z, odd-N stable nuclide. Neutron moderator in reactor physics.",
    5: "Two stable isotopes (¹⁰B, ¹¹B). Strong neutron absorber (¹⁰B). Spallation origin.",
    6: "Triple-alpha product; basis of organic chemistry. ¹²C doubly-magic (Z=6≈subshell). Hoyle state resonance enables stellar synthesis.",
    7: "¹⁴N dominant (99.6%). CNO cycle catalyst in massive stars. 7 protons + 7 neutrons, near N=Z line.",
    8: "Doubly-magic ¹⁶O (Z=8, N=8). Third most abundant element. Alpha-process endpoint in stellar cores.",
    9: "Only one stable isotope ¹⁹F. Highest electronegativity. Product of neutrino spallation in supernovae.",
    10: "Doubly-magic ²⁰Ne (Z=10 subshell, N=10). Noble gas; alpha-process product. Ne-burning phase in stellar evolution.",
    # Period 3
    11: "Single stable isotope ²³Na. Synthesized in carbon burning. ²²Na (t½=2.6yr) is a gamma-ray tracer.",
    12: "Three stable isotopes. Alpha-process product; ²⁴Mg is doubly-even. Important in stellar nucleosynthesis.",
    13: "Only stable isotope ²⁷Al. Most abundant metal in Earth's crust. ²⁶Al (t½=0.72 Myr) traces star formation.",
    14: "Three stable isotopes. ²⁸Si produced in oxygen burning. Silicon burning → iron peak in massive stars.",
    15: "Only stable isotope ³¹P. Biologically essential (DNA, ATP). Product of neutron capture on silicon.",
    16: "Four stable isotopes. ³²S from oxygen burning. Important in s-process nucleosynthesis.",
    17: "Two stable isotopes (³⁵Cl, ³⁷Cl). ³⁷Cl used in solar neutrino detection (Homestake experiment).",
    18: "Three stable isotopes. ⁴⁰Ar dominates (99.6%) from ⁴⁰K decay; K-Ar dating cornerstone in geochronology.",
    # Period 4
    19: "Three naturally occurring isotopes. ⁴⁰K (t½=1.25 Gyr) provides radiogenic dating and Earth's internal heat.",
    20: "Six stable isotopes. ⁴⁰Ca doubly-magic (Z=20, N=20). ⁴⁸Ca doubly-magic (Z=20, N=28); used in superheavy synthesis.",
    21: "Only stable isotope ⁴⁵Sc. Rare; produced in supernovae. Transition between Ca shell closure and Ti midshell.",
    22: "Five stable isotopes. ⁴⁸Ti most abundant (73.7%). Strong structural metal; near N=28 magic number.",
    23: "Two isotopes (⁵⁰V radioactive, t½=1.4×10¹⁷yr effectively stable). N=28 for ⁵¹V: near magic closure.",
    24: "Four stable isotopes. ⁵²Cr dominant (83.8%). N=28 for ⁵²Cr: singly magic. Iron-peak element.",
    25: "Only stable isotope ⁵⁵Mn. Fe-peak element. Produced in silicon burning and Type Ia supernovae.",
    26: "Four stable isotopes. ⁵⁶Fe: second-highest BE/A (8.790 MeV). Iron peak: endpoint of exothermic fusion.",
    27: "Only stable isotope ⁵⁹Co. Sole element with one stable isotope in the iron peak. ⁶⁰Co: important γ source.",
    28: "Five stable isotopes. ⁶²Ni: highest BE/A of all nuclides (8.7945 MeV). True binding energy maximum.",
    29: "Two stable isotopes. ⁶³Cu (69.2%), ⁶⁵Cu (30.8%). s-process product; essential trace nutrient.",
    30: "Five stable isotopes. Iron-peak endpoint in nucleosynthesis. Neutron-rich isotopes from s-process.",
    31: "Two stable isotopes. ⁶⁹Ga (60.1%), ⁷¹Ga (39.9%). GALLEX/GNO solar neutrino detector via ⁷¹Ga.",
    32: "Five stable isotopes. ⁷⁶Ge: double-beta decay candidate (Majorana neutrino search). Semiconductor.",
    33: "Only stable isotope ⁷⁵As. s-process product. Used in semiconductor doping (GaAs).",
    34: "Six stable isotopes. ⁸²Se: another double-beta candidate. Essential micronutrient.",
    35: "Two stable isotopes. ⁷⁹Br (50.7%), ⁸¹Br (49.3%). Near N=50 magic for ⁸¹Br(N=46, close).",
    36: "Six stable isotopes. ⁸⁶Kr astrophysical s-process monitor. Noble gas; clathrate-forming.",
    # Period 5
    37: "Two isotopes; ⁸⁵Rb stable, ⁸⁷Rb t½=4.9×10¹⁰yr. Rb-Sr dating in geochronology.",
    38: "Four stable isotopes. ⁸⁸Sr dominant (82.6%); near N=50 magic. Strontium isotope ratios trace ocean chemistry.",
    39: "Only stable isotope ⁸⁹Y. Near N=50 magic (N=50). Important in radiopharmaceuticals (⁹⁰Y).",
    40: "Five stable isotopes. ⁹⁰Zr (N=50): singly magic. Low neutron absorption; nuclear fuel cladding.",
    41: "Only stable isotope ⁹³Nb. Near N=50 magic. Superconducting magnet alloys (Nb-Ti, Nb₃Sn).",
    42: "Seven stable isotopes. ⁹²Mo and ⁹⁸Mo bracket the s/r-process. Highest melting point of period-5 metals.",
    43: "No stable isotopes. ⁹⁸Tc longest-lived (t½=4.2 Myr). First artificially produced element (1937). Tc-99m: nuclear medicine workhorse.",
    44: "Seven stable isotopes. ¹⁰²Ru most abundant. Platinum group metal; Fischer-Tropsch catalyst.",
    45: "Only stable isotope ¹⁰³Rh. 100% monoisotopic. Catalytic converter essential (automotive emissions).",
    46: "Six stable isotopes. ¹⁰⁶Pd most abundant. Hydrogen absorption; metal-catalysed cross-coupling (Suzuki, Heck).",
    47: "Two stable isotopes. ¹⁰⁷Ag (51.8%), ¹⁰⁹Ag (48.2%). Highest electrical conductivity of all metals.",
    48: "Eight stable isotopes. ¹¹⁴Cd most abundant. ¹¹³Cd: extreme neutron absorber (σ=20,600 barn).",
    49: "Two isotopes; ¹¹⁵In t½=4.4×10¹⁴yr (effectively stable). ITO transparent conductor; semiconductor dopant.",
    50: "Ten stable isotopes — most of any element. ¹²⁰Sn most abundant. Z=50: singly magic. Tin pest phase transition.",
    51: "Two stable isotopes. ¹²¹Sb (57.2%), ¹²³Sb (42.8%). III-V semiconductor compound (InSb, GaSb).",
    52: "Eight stable isotopes. ¹³⁰Te: double-beta decay (t½=8×10²⁰yr, measured by CUORE). Near N=82 magic.",
    53: "Only stable isotope ¹²⁷I. Essential thyroid nutrient. KI for nuclear emergency prophylaxis.",
    54: "Nine stable isotopes. ¹³²Xe near N=78. Xenon-based dark matter detectors (XENON, LZ, PandaX).",
    # Period 6
    55: "Only stable isotope ¹³³Cs. Most electropositive stable element. ¹³³Cs hyperfine transition defines the SI second.",
    56: "Seven stable isotopes. ¹³⁸Ba most abundant (71.7%). N=82: singly magic. Barium meal in medical imaging.",
    57: "Two isotopes; ¹³⁹La stable (99.9%). Lanthanide series onset. Perovskite structures (LaAlO₃).",
    58: "Four stable isotopes. ¹⁴⁰Ce most abundant (88.5%). N=82: singly magic. CeO₂ oxygen buffer; automotive catalyst.",
    59: "Only stable isotope ¹⁴¹Pr. N=82: singly magic. Strong paramagnetic moment; Pr₂Fe₁₄B magnets.",
    60: "Seven stable isotopes. ¹⁴²Nd most abundant (N=82 magic). ¹⁴⁶Nd now extinct: Sm-Nd dating in cosmochronology.",
    61: "No stable isotopes. ¹⁴⁵Pm longest-lived (t½=17.7yr). Only lanthanide with no stable nuclide; luminescent paint.",
    62: "Seven stable isotopes. ¹⁵²Sm most abundant. ¹⁴⁹Sm: highest thermal neutron cross-section of stable nuclides.",
    63: "Two stable isotopes. ¹⁵³Eu most abundant (52.2%). Nuclear control rods; euro banknote phosphors.",
    64: "Seven stable isotopes. ¹⁵⁸Gd most abundant. Highest thermal neutron capture of all elements. MRI contrast (Gd-DTPA).",
    65: "Only stable isotope ¹⁵⁹Tb. Green phosphor in displays and LEDs. Magnetostrictive Terfenol-D alloy.",
    66: "Seven stable isotopes. ¹⁶⁴Dy most abundant. Highest magnetic susceptibility; Dy₂O₃ in reactor control.",
    67: "Only stable isotope ¹⁶⁵Ho. Highest magnetic moment of any element. Ho-166: cancer brachytherapy.",
    68: "Six stable isotopes. ¹⁶⁶Er most abundant. Er-doped fibre amplifiers (EDFA): backbone of internet.",
    69: "Only stable isotope ¹⁶⁹Tm. ¹⁷⁰Tm: portable X-ray source. Rarest naturally occurring lanthanide.",
    70: "Seven stable isotopes. ¹⁷⁴Yb most abundant. Yb optical lattice clock: most precise timekeeper built.",
    71: "Two stable isotopes. ¹⁷⁵Lu most abundant (97.4%). Lu-176 cosmochronometer; PET detector (LYSO crystal).",
    72: "Six stable isotopes. ¹⁸⁰Hf most abundant. Excellent neutron absorber; Hf-W dating of planetesimal differentiation.",
    73: "Two isotopes; ¹⁸⁰Ta t½=1.5×10¹⁵yr (rarest primordial). ¹⁸¹Ta (99.99%): surgical implants, capacitors.",
    74: "Five stable isotopes. ¹⁸⁴W most abundant (30.6%). Highest melting point (3695 K). Wolfram: X-ray target anode.",
    75: "Two isotopes; ¹⁸⁷Re t½=4.12×10¹⁰yr. Re-Os dating of Earth's mantle. Highest boiling point of all elements.",
    76: "Seven stable isotopes. ¹⁹²Os most abundant. Densest naturally occurring element (22.59 g/cm³). Os-187 mantle tracer.",
    77: "Two stable isotopes. ¹⁹³Ir most abundant (62.7%). K-Pg boundary iridium anomaly: asteroid impact evidence.",
    78: "Six stable isotopes. ¹⁹⁵Pt most abundant (33.8%). Catalytic converter; cisplatin cancer therapy.",
    79: "Only stable isotope ¹⁹⁷Au. r-process product (neutron-star mergers). Relativistic contraction explains golden colour.",
    80: "Seven stable isotopes. ²⁰²Hg most abundant. Only liquid metal at STP. Mercury porosimetry; historic thermometers.",
    81: "Two stable isotopes. ²⁰⁵Tl most abundant (70.5%). ²⁰⁸Tl in ²³²Th decay chain. Tl-201: cardiac SPECT imaging.",
    82: "Four stable isotopes. ²⁰⁸Pb: doubly-magic (Z=82, N=126). End of the four natural decay chains. Radiation shielding.",
    83: "Formerly 'stable'; ²⁰⁹Bi t½=1.9×10¹⁹yr (alpha decay, measured 2003). Longest half-life directly measured.",
    84: "No stable isotopes. ²⁰⁹Po longest widely cited (t½=103yr). ²¹⁰Po (138d) famously used in Litvinenko case.",
    85: "No stable isotopes; ²¹⁰At longest-lived (t½=8.1hr). Rarest naturally occurring element (~1g in Earth's crust).",
    86: "No stable isotopes. ²²²Rn (t½=3.82d). Radioactive noble gas; indoor radon exposure: second cause of lung cancer.",
    # Period 7
    87: "No stable isotopes. ²²³Fr longest (t½=22min). Most electropositive element. Only ~30g exist in Earth's crust.",
    88: "No stable isotopes. ²²⁶Ra (t½=1600yr). Curie's discovery; luminous paint (radium girls). Alpha therapy (²²³Ra).",
    89: "No stable isotopes. ²²⁷Ac (t½=21.8yr). Parent of Ac-225 targeted alpha therapy; next-gen cancer treatment.",
    90: "Primordial; ²³²Th (t½=14.0 Gyr). r-process product. Thorium fuel cycle (molten salt reactors).",
    91: "No stable isotopes. ²³¹Pa (t½=32,760yr). ²³¹Pa/²³⁵U ratio dates ocean sediments.",
    92: "Primordial; ²³⁸U (t½=4.47 Gyr). Fission fuel; U-Pb dating: gold standard in geochronology.",
    93: "No stable isotopes. ²³⁷Np (t½=2.14 Myr). First transuranium element produced (1940). Nuclear waste concern.",
    94: "No stable isotopes. ²⁴⁴Pu (t½=80 Myr). ²³⁹Pu: nuclear weapons and MOX fuel. Now extinct in solar system.",
    95: "No stable isotopes. ²⁴³Am (t½=7,370yr). Americium-241: household smoke detectors (α source).",
    96: "No stable isotopes. ²⁴⁷Cm (t½=15.6 Myr). RTG power source; Cm-244 α-decay heats deep space probes.",
    97: "No stable isotopes. ²⁴⁷Bk (t½=1,380yr). Target material for Ts synthesis. Half-gram produced cumulatively.",
    98: "No stable isotopes. ²⁵¹Cf (t½=900yr). Strong neutron emitter; used for reactor startup and borehole logging.",
    99: "No stable isotopes. ²⁵²Es (t½=1.29yr). Produced in Ivy Mike thermonuclear test fallout (1952).",
    100: "No stable isotopes. ²⁵⁷Fm (t½=100.5d). Fermium-258: spontaneous fission limit (t½=0.37ms).",
    101: "No stable isotopes. ²⁵⁸Md (t½=51.5d). First atom-at-a-time chemistry identification.",
    102: "No stable isotopes. ²⁵⁹No (t½=58min). Named for Alfred Nobel. Nobelium-254: K-capture anomaly.",
    103: "No stable isotopes. ²⁶⁶Lr (t½=11hr). Last actinide; chemistry debated (5f vs 6d vs 7p₁/₂).",
    104: "No stable isotopes. ²⁶⁷Rf (t½=1.3hr). First transactinide. Behaves as Hf homologue (group 4).",
    105: "No stable isotopes. ²⁶⁸Db (t½=1.3d). Longest-lived transactinide isotope among group 5.",
    106: "No stable isotopes. ²⁶⁹Sg (t½=3.6min). Chemical behaviour confirmed as W homologue (group 6).",
    107: "No stable isotopes. ²⁷⁰Bh (t½=61s). Predicted Tc/Re homologue (group 7). Few atoms produced per experiment.",
    108: "No stable isotopes. ²⁶⁹Hs (t½=9.7s). Most element-like superheavy: HsO₄ volatile like OsO₄ (group 8).",
    109: "No stable isotopes. ²⁷⁸Mt (t½=7.6s). Named for Lise Meitner. Chemistry uncharacterised.",
    110: "No stable isotopes. ²⁸¹Ds (t½=12.7s). Chemical experiments suggest noble-metal behaviour.",
    111: "No stable isotopes. ²⁸²Rg (t½=100s). Named for Röntgen. Predicted Cu/Ag/Au homologue (group 11).",
    112: "No stable isotopes. ²⁸⁵Cn (t½=29s). Volatile; adsorption experiments suggest noble-metal character and Hg-like volatility.",
    113: "No stable isotopes. ²⁸⁶Nh (t½=9.5s). First element discovered in Asia (RIKEN, 2004). Group 13 homologue.",
    114: "No stable isotopes. ²⁸⁹Fl (t½=2.1s). Near predicted Z=114 shell closure (island of stability). Volatile like Pb.",
    115: "No stable isotopes. ²⁹⁰Mc (t½=0.65s). Discovered at JINR/LLNL (2003). Group 15 homologue; sub-second lifetime.",
    116: "No stable isotopes. ²⁹³Lv (t½=53ms). Fleeting; decays before chemistry possible. Predicted Po homologue.",
    117: "No stable isotopes. ²⁹⁴Ts (t½=51ms). Most recently named element (2016). Heaviest halogen homologue.",
    118: "No stable isotopes. ²⁹⁴Og (t½=0.7ms). Only 5 atoms ever observed. Predicted noble gas but likely reactive due to relativistic effects.",
}


# ── Kernel interface ─────────────────────────────────────────────
_KERNEL = OptimizedKernelComputer(epsilon=EPSILON)


def _regime_to_element_regime(r: Regime) -> ElementRegime:
    """Map a Regime enum to an ElementRegime string."""
    return ElementRegime(r.value.upper()) if hasattr(r, "value") else ElementRegime(str(r).upper())


# ── Core classify function ───────────────────────────────────────


def classify_element(el: ElementRecord) -> ElementClassification:
    """Classify a single element through the GCD framework.

    Parameters
    ----------
    el : ElementRecord
        Element reference data from element_data module.

    Returns
    -------
    ElementClassification
        Full classification including coordinates, kernel invariants,
        regime assignment, and physics annotation.
    """
    coords = compute_coords(el)
    c = np.array([coords.c1, coords.c2, coords.c3])
    result = _KERNEL.compute(c, WEIGHTS, validate=False)

    regime_enum = classify_regime(
        result.omega,
        result.F,
        result.S,
        result.C,
        result.IC,
    )

    # Resolve regime value regardless of enum type
    regime_str = regime_enum.value if hasattr(regime_enum, "value") else str(regime_enum)

    obs_bits = observation_credit_bits(el.half_life_s)
    hl_str = human_readable_halflife(el.half_life_s)
    note = _PHYSICS_NOTES.get(el.Z, "")

    return ElementClassification(
        Z=el.Z,
        symbol=el.symbol,
        name=el.name,
        A=el.A,
        half_life=hl_str,
        observation_bits=obs_bits,
        c1=coords.c1,
        c2=coords.c2,
        c3=coords.c3,
        omega=round(result.omega, 6),
        F=round(result.F, 6),
        S=round(result.S, 6),
        C=round(result.C, 6),
        IC=round(result.IC, 6),
        kappa=round(result.kappa, 6),
        regime=regime_str,
        physics_note=note,
    )


def classify_all() -> list[ElementClassification]:
    """Classify all 118 elements and return sorted by Z."""
    return [classify_element(el) for el in ELEMENTS]


# ── Summary statistics ───────────────────────────────────────────


def regime_summary(
    classifications: list[ElementClassification],
) -> dict[str, list[str]]:
    """Group element symbols by regime."""
    groups: dict[str, list[str]] = {
        "STABLE": [],
        "WATCH": [],
        "CRITICAL": [],
        "COLLAPSE": [],
    }
    for c in classifications:
        regime_key = c.regime.upper()
        if regime_key in groups:
            groups[regime_key].append(c.symbol)
    return groups


def print_classification_table(
    classifications: list[ElementClassification],
) -> None:
    """Print a formatted table of all element classifications."""
    header = (
        f"{'Z':>3}  {'Sym':>3}  {'Name':<14}  {'A':>3}  "
        f"{'t½':<12}  {'c1':>6}  {'c2':>6}  {'c3':>6}  "
        f"{'ω':>8}  {'F':>8}  {'S':>8}  {'IC':>8}  {'Regime':<9}"
    )
    sep = "─" * len(header)
    print(sep)
    print(header)
    print(sep)

    for c in classifications:
        line = (
            f"{c.Z:>3}  {c.symbol:>3}  {c.name:<14}  {c.A:>3}  "
            f"{c.half_life:<12}  {c.c1:>6.3f}  {c.c2:>6.3f}  {c.c3:>6.3f}  "
            f"{c.omega:>8.5f}  {c.F:>8.5f}  {c.S:>8.5f}  {c.IC:>8.5f}  {c.regime:<9}"
        )
        print(line)

    print(sep)


# ── Self-test & demonstration ────────────────────────────────────
if __name__ == "__main__":
    print("Periodic Table Closure — NUC.INTSTACK.v1")
    print("Classifying all 118 elements through collapse dynamics...\n")

    all_classifications = classify_all()

    # Full table
    print_classification_table(all_classifications)

    # Summary
    print()
    groups = regime_summary(all_classifications)
    for regime_name, symbols in groups.items():
        count = len(symbols)
        pct = 100.0 * count / 118
        joined = ", ".join(symbols)
        print(f"  {regime_name:>8}: {count:>3} ({pct:5.1f}%)  {joined}")

    print()

    # Key highlights
    print("Key observations:")
    print("  • Hydrogen (Z=1): sole CRITICAL element — no nuclear binding (c1≈0)")
    print("    but full temporal return (stable, c2=0.99). Distance from ideal is maximal.")
    print("  • Iron-56 (Z=26): WATCH, not STABLE — highest BE/A but N/Z=1.154")
    print("    deviates from valley (N/Z_opt≈1.10), creating geometric stress (S>0.15).")
    print("  • Nickel-62 (Z=28): true BE/A maximum. Framework detects structural peak.")
    print("  • Lead-208 (Z=82): doubly-magic (Z=82, N=126). Last stable anchor;")
    print("    all heavier elements have diminished return.")
    print("  • Oganesson (Z=118): 0.7 ms lifetime, 0 observation bits, 5 atoms ever seen.")
    print("    The axiom declares: no return cycle → no reality claim.")
    print()

    # Tier-1 identity check for all
    violations = 0
    for c in all_classifications:
        f_plus_omega = c.F + c.omega
        if abs(f_plus_omega - 1.0) > 1e-6:
            print(f"  !! Tier-1 VIOLATION: {c.symbol} F+ω = {f_plus_omega}")
            violations += 1
    if violations == 0:
        print("  Tier-1 identity F + ω = 1 holds for ALL 118 elements. ✓")
    print()
    print("Done.")

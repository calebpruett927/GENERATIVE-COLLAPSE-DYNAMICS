"""Element Data Module — NUC.INTSTACK.v1

Reference nuclear data for the 118 known elements, providing
binding energy per nucleon (AME2020), half-lives (NUBASE2020),
and neutron/proton counts for the most abundant stable isotope
(or longest-lived isotope for elements without stable isotopes).

Each element is mapped to three dimensionless coordinates:
  c1 = Binding Coherence   — BE/A normalized to Fe-56 peak
  c2 = Temporal Return     — half-life on a log-scale fraction of
                             the age of the universe
  c3 = Valley Proximity    — deviation of N/Z from the optimal
                             valley of stability

Data sources:
  BE/A:  Atomic Mass Evaluation 2020 (AME2020), Wang et al. 2021
  t½:    NUBASE2020, Kondev et al. 2021
  N/Z:   IAEA Nuclear Data Services

Cross-references:
  Related closures:
    closures/nuclear_physics/nuclide_binding.py  (Bethe-Weizsäcker SEMF)
    closures/nuclear_physics/shell_structure.py  (magic number proximity)
    closures/nuclear_physics/alpha_decay.py      (Geiger-Nuttall lifetime)
    closures/nuclear_physics/fissility.py        (Bohr-Wheeler fissility)
  Contract:  contracts/NUC.INTSTACK.v1.yaml
  Tier:      Tier-2 (domain expansion)
"""

from __future__ import annotations

import math
from typing import NamedTuple

# ── Frozen normalization constants ───────────────────────────────
BE_PEAK_REF = 8.7945  # Fe-56 BE/A peak (MeV/nucleon)
AGE_UNIVERSE_S = 4.35e17  # 13.8 Gyr in seconds
LOG_UNIVERSE = math.log10(AGE_UNIVERSE_S)  # ~17.638
NZ_VALLEY_SCALE = 0.5  # Deviation normalization for c3
COORD_FLOOR = 0.01  # Minimum coordinate value
COORD_CEIL = 0.99  # Maximum coordinate value


class ElementRecord(NamedTuple):
    """Immutable record for one element's reference data.

    Fields:
        Z: Atomic number (proton count).
        symbol: IUPAC standard symbol.
        name: IUPAC standard name.
        A: Mass number of most abundant stable (or longest-lived) isotope.
        BE_per_A: Binding energy per nucleon (MeV), AME2020.
        half_life_s: Half-life in seconds; 0 means stable.
        N: Neutron count for the reference isotope.
    """

    Z: int
    symbol: str
    name: str
    A: int
    BE_per_A: float
    half_life_s: float
    N: int


class ElementCoords(NamedTuple):
    """Dimensionless compression configuration point for one element.

    Fields:
        c1: Binding coherence — structural return, ∈ [0.01, 0.99].
        c2: Temporal return   — observation period credit, ∈ [0.01, 0.99].
        c3: Valley proximity  — geometric return, ∈ [0.01, 0.99].
    """

    c1: float
    c2: float
    c3: float


def _nz_optimal(A: int) -> float:
    """Optimal N/Z ratio from the valley of stability.

    Semi-empirical fit: N/Z_opt ≈ 1 + 0.015 · A^(2/3).
    Source: Krane, Introductory Nuclear Physics (1987), eq. 3.28.
    """
    return 1.0 + 0.015 * A ** (2.0 / 3.0)


def _clamp(value: float) -> float:
    """Clamp a coordinate to [COORD_FLOOR, COORD_CEIL]."""
    return max(COORD_FLOOR, min(COORD_CEIL, value))


def compute_coords(el: ElementRecord) -> ElementCoords:
    """Map an element's nuclear properties to dimensionless coordinates.

    Parameters
    ----------
    el : ElementRecord
        Element reference data.

    Returns
    -------
    ElementCoords
        Three-dimensional compression configuration point.

    Coordinate semantics (axiom-traceable):
        c1 = structural return: how tightly bound is the nucleus?
             Directly encodes whether the nucleus holds itself together.
             A fully bound nucleus (Fe-56, c1→1) returns perfectly
             under structural measurement.

        c2 = temporal return: does the observation period credit suffice?
             Encodes log₁₀(t½) on a scale from 10⁻²⁵ s to the age of
             the universe.  Stable isotopes (t½ = ∞) get maximum credit.
             An element that decays before re-measurement has impaired
             return — the axiom says it is less real.

        c3 = geometric return: is the internal geometry balanced?
             Deviation of N/Z from the valley of stability.
             On-valley nuclei have optimal Coulomb-vs-asymmetry balance;
             off-valley nuclei are geometrically stressed.
    """
    # c1: binding coherence
    c1 = _clamp(el.BE_per_A / BE_PEAK_REF)

    # c2: temporal return
    if el.half_life_s == 0:
        c2 = COORD_CEIL
    else:
        log_t = math.log10(max(el.half_life_s, 1e-25))
        c2 = _clamp((log_t + 25) / (LOG_UNIVERSE + 25))

    # c3: valley proximity
    nz_actual = el.N / el.Z if el.Z > 0 else 1.0
    nz_opt = _nz_optimal(el.A)
    deviation = abs(nz_actual - nz_opt)
    c3 = _clamp(1.0 - deviation / NZ_VALLEY_SCALE)

    return ElementCoords(
        c1=round(c1, 6),
        c2=round(c2, 6),
        c3=round(c3, 6),
    )


def observation_credit_bits(half_life_s: float) -> float:
    """Return the observation credit in bits.

    Credit = log₂(t½ / t_measurement), where t_measurement = 1 s.
    This counts how many independent measurement cycles are available
    before the element has a significant probability of decaying.

    Returns 0.0 if t½ < 1 second (no complete return cycle possible).
    Returns float('inf') for stable isotopes.
    """
    if half_life_s == 0:
        return float("inf")
    if half_life_s < 1.0:
        return 0.0
    return math.log2(half_life_s)


def human_readable_halflife(half_life_s: float) -> str:
    """Format half-life for human readability."""
    if half_life_s == 0:
        return "stable"
    if half_life_s < 1e-3:
        return f"{half_life_s * 1e6:.1f} us"
    if half_life_s < 1:
        return f"{half_life_s * 1e3:.1f} ms"
    if half_life_s < 60:
        return f"{half_life_s:.1f} s"
    if half_life_s < 3600:
        return f"{half_life_s / 60:.1f} min"
    if half_life_s < 86400:
        return f"{half_life_s / 3600:.1f} hr"
    if half_life_s < 3.156e7:
        return f"{half_life_s / 86400:.1f} d"
    if half_life_s < 3.156e9:
        return f"{half_life_s / 3.156e7:.1f} yr"
    if half_life_s < 3.156e15:
        return f"{half_life_s / 3.156e7:.0f} yr"
    if half_life_s < 3.156e16:
        return f"{half_life_s / 3.156e7 / 1e6:.2f} Myr"
    return f"{half_life_s / 3.156e7 / 1e9:.2f} Gyr"


# ─────────────────────────────────────────────────────────────────
# COMPLETE ELEMENT TABLE
# 118 elements, IUPAC standard names and symbols.
# Data: AME2020 (BE/A), NUBASE2020 (t½), IAEA (N/Z).
#
# Format: ElementRecord(Z, symbol, name, A, BE/A, t½, N)
#   A   = mass number of most abundant stable isotope, or
#         longest-lived isotope if no stable isotope exists
#   BE/A = binding energy per nucleon in MeV (AME2020)
#   t½  = half-life in seconds (0 = stable)
#   N   = neutron count = A − Z
# ─────────────────────────────────────────────────────────────────

ELEMENTS: tuple[ElementRecord, ...] = (
    # ── Period 1 ─────────────────────────────────────────────────
    ElementRecord(1, "H", "Hydrogen", 1, 0.000, 0, 0),
    ElementRecord(2, "He", "Helium", 4, 7.074, 0, 2),
    # ── Period 2 ─────────────────────────────────────────────────
    ElementRecord(3, "Li", "Lithium", 7, 5.606, 0, 4),
    ElementRecord(4, "Be", "Beryllium", 9, 6.463, 0, 5),
    ElementRecord(5, "B", "Boron", 11, 6.928, 0, 6),
    ElementRecord(6, "C", "Carbon", 12, 7.680, 0, 6),
    ElementRecord(7, "N", "Nitrogen", 14, 7.476, 0, 7),
    ElementRecord(8, "O", "Oxygen", 16, 7.976, 0, 8),
    ElementRecord(9, "F", "Fluorine", 19, 7.779, 0, 10),
    ElementRecord(10, "Ne", "Neon", 20, 8.032, 0, 10),
    # ── Period 3 ─────────────────────────────────────────────────
    ElementRecord(11, "Na", "Sodium", 23, 8.112, 0, 12),
    ElementRecord(12, "Mg", "Magnesium", 24, 8.261, 0, 12),
    ElementRecord(13, "Al", "Aluminium", 27, 8.332, 0, 14),
    ElementRecord(14, "Si", "Silicon", 28, 8.448, 0, 14),
    ElementRecord(15, "P", "Phosphorus", 31, 8.481, 0, 16),
    ElementRecord(16, "S", "Sulfur", 32, 8.493, 0, 16),
    ElementRecord(17, "Cl", "Chlorine", 35, 8.520, 0, 18),
    ElementRecord(18, "Ar", "Argon", 40, 8.595, 0, 22),
    # ── Period 4 ─────────────────────────────────────────────────
    ElementRecord(19, "K", "Potassium", 39, 8.557, 0, 20),
    ElementRecord(20, "Ca", "Calcium", 40, 8.551, 0, 20),
    ElementRecord(21, "Sc", "Scandium", 45, 8.619, 0, 24),
    ElementRecord(22, "Ti", "Titanium", 48, 8.723, 0, 26),
    ElementRecord(23, "V", "Vanadium", 51, 8.742, 0, 28),
    ElementRecord(24, "Cr", "Chromium", 52, 8.776, 0, 28),
    ElementRecord(25, "Mn", "Manganese", 55, 8.765, 0, 30),
    ElementRecord(26, "Fe", "Iron", 56, 8.790, 0, 30),
    ElementRecord(27, "Co", "Cobalt", 59, 8.768, 0, 32),
    ElementRecord(28, "Ni", "Nickel", 58, 8.732, 0, 30),
    ElementRecord(29, "Cu", "Copper", 63, 8.752, 0, 34),
    ElementRecord(30, "Zn", "Zinc", 64, 8.736, 0, 34),
    ElementRecord(31, "Ga", "Gallium", 69, 8.724, 0, 38),
    ElementRecord(32, "Ge", "Germanium", 74, 8.713, 0, 42),
    ElementRecord(33, "As", "Arsenic", 75, 8.701, 0, 42),
    ElementRecord(34, "Se", "Selenium", 80, 8.711, 0, 46),
    ElementRecord(35, "Br", "Bromine", 79, 8.696, 0, 44),
    ElementRecord(36, "Kr", "Krypton", 84, 8.717, 0, 48),
    # ── Period 5 ─────────────────────────────────────────────────
    ElementRecord(37, "Rb", "Rubidium", 85, 8.697, 0, 48),
    ElementRecord(38, "Sr", "Strontium", 88, 8.733, 0, 50),
    ElementRecord(39, "Y", "Yttrium", 89, 8.714, 0, 50),
    ElementRecord(40, "Zr", "Zirconium", 90, 8.710, 0, 50),
    ElementRecord(41, "Nb", "Niobium", 93, 8.664, 0, 52),
    ElementRecord(42, "Mo", "Molybdenum", 98, 8.635, 0, 56),
    ElementRecord(43, "Tc", "Technetium", 97, 8.635, 1.33e14, 54),
    ElementRecord(44, "Ru", "Ruthenium", 102, 8.607, 0, 58),
    ElementRecord(45, "Rh", "Rhodium", 103, 8.584, 0, 58),
    ElementRecord(46, "Pd", "Palladium", 106, 8.567, 0, 60),
    ElementRecord(47, "Ag", "Silver", 107, 8.554, 0, 60),
    ElementRecord(48, "Cd", "Cadmium", 114, 8.540, 0, 66),
    ElementRecord(49, "In", "Indium", 115, 8.516, 0, 66),
    ElementRecord(50, "Sn", "Tin", 120, 8.505, 0, 70),
    ElementRecord(51, "Sb", "Antimony", 121, 8.482, 0, 70),
    ElementRecord(52, "Te", "Tellurium", 130, 8.440, 0, 78),
    ElementRecord(53, "I", "Iodine", 127, 8.446, 0, 74),
    ElementRecord(54, "Xe", "Xenon", 132, 8.428, 0, 78),
    # ── Period 6 ─────────────────────────────────────────────────
    ElementRecord(55, "Cs", "Caesium", 133, 8.410, 0, 78),
    ElementRecord(56, "Ba", "Barium", 138, 8.394, 0, 82),
    ElementRecord(57, "La", "Lanthanum", 139, 8.378, 0, 82),
    ElementRecord(58, "Ce", "Cerium", 140, 8.376, 0, 82),
    ElementRecord(59, "Pr", "Praseodymium", 141, 8.354, 0, 82),
    ElementRecord(60, "Nd", "Neodymium", 142, 8.346, 0, 82),
    ElementRecord(61, "Pm", "Promethium", 145, 8.320, 5.58e8, 84),
    ElementRecord(62, "Sm", "Samarium", 152, 8.280, 0, 90),
    ElementRecord(63, "Eu", "Europium", 153, 8.272, 0, 90),
    ElementRecord(64, "Gd", "Gadolinium", 158, 8.248, 0, 94),
    ElementRecord(65, "Tb", "Terbium", 159, 8.237, 0, 94),
    ElementRecord(66, "Dy", "Dysprosium", 164, 8.213, 0, 98),
    ElementRecord(67, "Ho", "Holmium", 165, 8.199, 0, 98),
    ElementRecord(68, "Er", "Erbium", 166, 8.189, 0, 98),
    ElementRecord(69, "Tm", "Thulium", 169, 8.172, 0, 100),
    ElementRecord(70, "Yb", "Ytterbium", 174, 8.147, 0, 104),
    ElementRecord(71, "Lu", "Lutetium", 175, 8.138, 0, 104),
    ElementRecord(72, "Hf", "Hafnium", 180, 8.106, 0, 108),
    ElementRecord(73, "Ta", "Tantalum", 181, 8.097, 0, 108),
    ElementRecord(74, "W", "Tungsten", 184, 8.078, 0, 110),
    ElementRecord(75, "Re", "Rhenium", 187, 8.054, 0, 112),
    ElementRecord(76, "Os", "Osmium", 192, 8.032, 0, 116),
    ElementRecord(77, "Ir", "Iridium", 193, 8.021, 0, 116),
    ElementRecord(78, "Pt", "Platinum", 195, 8.010, 0, 117),
    ElementRecord(79, "Au", "Gold", 197, 7.916, 0, 118),
    ElementRecord(80, "Hg", "Mercury", 202, 7.895, 0, 122),
    ElementRecord(81, "Tl", "Thallium", 205, 7.875, 0, 124),
    ElementRecord(82, "Pb", "Lead", 208, 7.868, 0, 126),
    ElementRecord(83, "Bi", "Bismuth", 209, 7.848, 6.02e26, 126),
    ElementRecord(84, "Po", "Polonium", 209, 7.835, 3.27e9, 125),
    ElementRecord(85, "At", "Astatine", 210, 7.825, 2.92e4, 125),
    ElementRecord(86, "Rn", "Radon", 222, 7.695, 3.30e5, 136),
    # ── Period 7 ─────────────────────────────────────────────────
    ElementRecord(87, "Fr", "Francium", 223, 7.681, 1.32e3, 136),
    ElementRecord(88, "Ra", "Radium", 226, 7.662, 5.05e10, 138),
    ElementRecord(89, "Ac", "Actinium", 227, 7.651, 6.87e8, 138),
    ElementRecord(90, "Th", "Thorium", 232, 7.615, 4.42e17, 142),
    ElementRecord(91, "Pa", "Protactinium", 231, 7.618, 1.03e12, 140),
    ElementRecord(92, "U", "Uranium", 238, 7.570, 1.41e17, 146),
    ElementRecord(93, "Np", "Neptunium", 237, 7.575, 6.77e13, 144),
    ElementRecord(94, "Pu", "Plutonium", 244, 7.523, 2.52e15, 150),
    ElementRecord(95, "Am", "Americium", 243, 7.526, 2.33e11, 148),
    ElementRecord(96, "Cm", "Curium", 247, 7.500, 4.92e14, 151),
    ElementRecord(97, "Bk", "Berkelium", 247, 7.490, 4.35e10, 150),
    ElementRecord(98, "Cf", "Californium", 251, 7.462, 2.84e10, 153),
    ElementRecord(99, "Es", "Einsteinium", 252, 7.450, 3.56e7, 153),
    ElementRecord(100, "Fm", "Fermium", 257, 7.420, 8.67e6, 157),
    ElementRecord(101, "Md", "Mendelevium", 258, 7.400, 4.45e6, 157),
    ElementRecord(102, "No", "Nobelium", 259, 7.380, 3.48e3, 157),
    ElementRecord(103, "Lr", "Lawrencium", 266, 7.350, 3.96e4, 163),
    ElementRecord(104, "Rf", "Rutherfordium", 267, 7.320, 4.68e3, 163),
    ElementRecord(105, "Db", "Dubnium", 268, 7.300, 1.15e5, 163),
    ElementRecord(106, "Sg", "Seaborgium", 269, 7.280, 2.16e2, 163),
    ElementRecord(107, "Bh", "Bohrium", 270, 7.260, 6.10e1, 163),
    ElementRecord(108, "Hs", "Hassium", 269, 7.240, 9.70e0, 161),
    ElementRecord(109, "Mt", "Meitnerium", 278, 7.220, 7.60e0, 169),
    ElementRecord(110, "Ds", "Darmstadtium", 281, 7.200, 1.28e1, 171),
    ElementRecord(111, "Rg", "Roentgenium", 282, 7.180, 1.00e2, 171),
    ElementRecord(112, "Cn", "Copernicium", 285, 7.160, 2.90e1, 173),
    ElementRecord(113, "Nh", "Nihonium", 286, 7.140, 9.50e0, 173),
    ElementRecord(114, "Fl", "Flerovium", 289, 7.120, 2.10e0, 175),
    ElementRecord(115, "Mc", "Moscovium", 290, 7.100, 6.50e-1, 175),
    ElementRecord(116, "Lv", "Livermorium", 293, 7.080, 5.30e-2, 177),
    ElementRecord(117, "Ts", "Tennessine", 294, 7.060, 5.10e-2, 177),
    ElementRecord(118, "Og", "Oganesson", 294, 7.040, 7.00e-4, 176),
)

# Lookup by Z (atomic number) and by symbol
ELEMENTS_BY_Z: dict[int, ElementRecord] = {el.Z: el for el in ELEMENTS}
ELEMENTS_BY_SYMBOL: dict[str, ElementRecord] = {el.symbol: el for el in ELEMENTS}


# ── Self-test ────────────────────────────────────────────────────
if __name__ == "__main__":
    assert len(ELEMENTS) == 118, f"Expected 118 elements, got {len(ELEMENTS)}"
    for el in ELEMENTS:
        assert el.Z == el.A - el.N, f"{el.symbol}: Z + N != A"
        coords = compute_coords(el)
        assert 0.01 <= coords.c1 <= 0.99, f"{el.symbol}: c1 out of range"
        assert 0.01 <= coords.c2 <= 0.99, f"{el.symbol}: c2 out of range"
        assert 0.01 <= coords.c3 <= 0.99, f"{el.symbol}: c3 out of range"
    print(f"All {len(ELEMENTS)} elements validated.")
    print()
    # Show a few examples
    for sym in ("H", "He", "C", "Fe", "Au", "Pb", "U", "Og"):
        el = ELEMENTS_BY_SYMBOL[sym]
        c = compute_coords(el)
        bits = observation_credit_bits(el.half_life_s)
        hl = human_readable_halflife(el.half_life_s)
        credit = "unlimited" if bits == float("inf") else f"{bits:.1f} bits"
        print(
            f"  Z={el.Z:>3} {el.symbol:>3} {el.name:<14}  "
            f"c=({c.c1:.3f}, {c.c2:.3f}, {c.c3:.3f})  "
            f"t={hl:<14}  credit={credit}"
        )

"""Electron Configuration Closure — ATOM.INTSTACK.v1

Determines electron configuration using the Aufbau principle,
Hund's rules, and computes shell completeness as a GCD embedding.

Physics:
  Aufbau order: 1s, 2s, 2p, 3s, 3p, 4s, 3d, 4p, 5s, 4d, ...
  Shell capacity: 2n²
  Subshell capacity: 2(2l+1)
  Noble gas stability → F_eff = 1 (shell closure = collapse attractor)

UMCP integration:
  ω_eff = 1 − (filled_fraction of outermost shell)
  F_eff = 1 − ω_eff
  Regime: ClosedShell / NearClosed / OpenShell / HalfFilled

Cross-references:
  Contract:  contracts/ATOM.INTSTACK.v1.yaml
"""

from __future__ import annotations

from enum import StrEnum
from typing import NamedTuple


class ConfigRegime(StrEnum):
    """Regime based on shell completeness."""

    CLOSED_SHELL = "ClosedShell"
    NEAR_CLOSED = "NearClosed"
    HALF_FILLED = "HalfFilled"
    OPEN_SHELL = "OpenShell"


class ElectronConfigResult(NamedTuple):
    """Result of electron configuration computation."""

    configuration: str  # e.g. "1s² 2s² 2p⁶ 3s¹"
    noble_gas_core: str  # e.g. "[Ne]"
    valence_electrons: int
    shell_completeness: float  # fraction of outermost shell filled
    omega_eff: float
    F_eff: float
    period: int
    group_block: str  # s, p, d, f
    regime: str


# ── Aufbau filling order ─────────────────────────────────────────
# (n, l, capacity, label)
AUFBAU_ORDER: list[tuple[int, int, int, str]] = [
    (1, 0, 2, "1s"),
    (2, 0, 2, "2s"),
    (2, 1, 6, "2p"),
    (3, 0, 2, "3s"),
    (3, 1, 6, "3p"),
    (4, 0, 2, "4s"),
    (3, 2, 10, "3d"),
    (4, 1, 6, "4p"),
    (5, 0, 2, "5s"),
    (4, 2, 10, "4d"),
    (5, 1, 6, "5p"),
    (6, 0, 2, "6s"),
    (4, 3, 14, "4f"),
    (5, 2, 10, "5d"),
    (6, 1, 6, "6p"),
    (7, 0, 2, "7s"),
    (5, 3, 14, "5f"),
    (6, 2, 10, "6d"),
    (7, 1, 6, "7p"),
]

# Noble gas configurations
NOBLE_GASES: dict[int, str] = {
    2: "[He]",
    10: "[Ne]",
    18: "[Ar]",
    36: "[Kr]",
    54: "[Xe]",
    86: "[Rn]",
    118: "[Og]",
}

# ── Aufbau exceptions ────────────────────────────────────────────
# Elements whose ground-state config differs from strict Aufbau filling.
# Cause: half-filled or fully-filled d/f subshell stability.
# Format: Z → list[ (label, population) ] giving the FULL configuration.
# Source: NIST Atomic Spectra Database (Kramida et al. 2023)
AUFBAU_EXCEPTIONS: dict[int, list[tuple[str, int]]] = {
    24: [  # Cr: [Ar] 3d⁵ 4s¹ (half-filled d)
        ("1s", 2),
        ("2s", 2),
        ("2p", 6),
        ("3s", 2),
        ("3p", 6),
        ("4s", 1),
        ("3d", 5),
    ],
    29: [  # Cu: [Ar] 3d¹⁰ 4s¹ (fully-filled d)
        ("1s", 2),
        ("2s", 2),
        ("2p", 6),
        ("3s", 2),
        ("3p", 6),
        ("4s", 1),
        ("3d", 10),
    ],
    41: [  # Nb: [Kr] 4d⁴ 5s¹
        ("1s", 2),
        ("2s", 2),
        ("2p", 6),
        ("3s", 2),
        ("3p", 6),
        ("4s", 2),
        ("3d", 10),
        ("4p", 6),
        ("5s", 1),
        ("4d", 4),
    ],
    42: [  # Mo: [Kr] 4d⁵ 5s¹ (half-filled d)
        ("1s", 2),
        ("2s", 2),
        ("2p", 6),
        ("3s", 2),
        ("3p", 6),
        ("4s", 2),
        ("3d", 10),
        ("4p", 6),
        ("5s", 1),
        ("4d", 5),
    ],
    44: [  # Ru: [Kr] 4d⁷ 5s¹
        ("1s", 2),
        ("2s", 2),
        ("2p", 6),
        ("3s", 2),
        ("3p", 6),
        ("4s", 2),
        ("3d", 10),
        ("4p", 6),
        ("5s", 1),
        ("4d", 7),
    ],
    45: [  # Rh: [Kr] 4d⁸ 5s¹
        ("1s", 2),
        ("2s", 2),
        ("2p", 6),
        ("3s", 2),
        ("3p", 6),
        ("4s", 2),
        ("3d", 10),
        ("4p", 6),
        ("5s", 1),
        ("4d", 8),
    ],
    46: [  # Pd: [Kr] 4d¹⁰ (no 5s!)
        ("1s", 2),
        ("2s", 2),
        ("2p", 6),
        ("3s", 2),
        ("3p", 6),
        ("4s", 2),
        ("3d", 10),
        ("4p", 6),
        ("4d", 10),
    ],
    47: [  # Ag: [Kr] 4d¹⁰ 5s¹ (fully-filled d)
        ("1s", 2),
        ("2s", 2),
        ("2p", 6),
        ("3s", 2),
        ("3p", 6),
        ("4s", 2),
        ("3d", 10),
        ("4p", 6),
        ("5s", 1),
        ("4d", 10),
    ],
    78: [  # Pt: [Xe] 4f¹⁴ 5d⁹ 6s¹
        ("1s", 2),
        ("2s", 2),
        ("2p", 6),
        ("3s", 2),
        ("3p", 6),
        ("4s", 2),
        ("3d", 10),
        ("4p", 6),
        ("5s", 2),
        ("4d", 10),
        ("5p", 6),
        ("6s", 1),
        ("4f", 14),
        ("5d", 9),
    ],
    79: [  # Au: [Xe] 4f¹⁴ 5d¹⁰ 6s¹ (fully-filled d)
        ("1s", 2),
        ("2s", 2),
        ("2p", 6),
        ("3s", 2),
        ("3p", 6),
        ("4s", 2),
        ("3d", 10),
        ("4p", 6),
        ("5s", 2),
        ("4d", 10),
        ("5p", 6),
        ("6s", 1),
        ("4f", 14),
        ("5d", 10),
    ],
}

# Superscript digits
_SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def _classify_regime(shell_completeness: float) -> ConfigRegime:
    if shell_completeness >= 0.99:
        return ConfigRegime.CLOSED_SHELL
    if shell_completeness >= 0.85:
        return ConfigRegime.NEAR_CLOSED
    if 0.45 <= shell_completeness <= 0.55:
        return ConfigRegime.HALF_FILLED
    return ConfigRegime.OPEN_SHELL


def _subshell_cap(label: str) -> int:
    """Return the maximum capacity of a subshell from its label (e.g. '3d' → 10)."""
    l_char = label[-1]
    return {"s": 2, "p": 6, "d": 10, "f": 14}.get(l_char, 2)


def compute_electron_config(Z: int) -> ElectronConfigResult:
    """Compute electron configuration and shell completeness.

    Parameters
    ----------
    Z : int
        Atomic number (1–118).

    Returns
    -------
    ElectronConfigResult
    """
    if Z < 1 or Z > 118:
        msg = f"Z must be in [1, 118], got {Z}"
        raise ValueError(msg)

    # Use Aufbau exception table if this element has a known exception
    if Z in AUFBAU_EXCEPTIONS:
        filled = [(label, count, _subshell_cap(label)) for label, count in AUFBAU_EXCEPTIONS[Z]]
    else:
        electrons_left = Z
        filled = []
        for _, _l, cap, label in AUFBAU_ORDER:
            if electrons_left <= 0:
                break
            count = min(electrons_left, cap)
            filled.append((label, count, cap))
            electrons_left -= count

    # Build configuration string
    config_parts = [f"{label}{str(count).translate(_SUP)}" for label, count, _ in filled]
    config_str = " ".join(config_parts)

    # Noble gas core abbreviation
    noble_core = ""
    for ng_z, ng_label in sorted(NOBLE_GASES.items(), reverse=True):
        if ng_z < Z:
            noble_core = ng_label
            break

    # Valence shell analysis
    if filled:
        last_label, last_count, last_cap = filled[-1]
        shell_completeness = last_count / last_cap
        valence = last_count
        # Determine block
        l_char = last_label[-1]
        group_block = {"s": "s-block", "p": "p-block", "d": "d-block", "f": "f-block"}.get(l_char, "unknown")
    else:
        shell_completeness = 0.0
        valence = 0
        group_block = "unknown"

    # Period
    period = 1
    if Z > 2:
        period = 2
    if Z > 10:
        period = 3
    if Z > 18:
        period = 4
    if Z > 36:
        period = 5
    if Z > 54:
        period = 6
    if Z > 86:
        period = 7

    omega_eff = max(0.0, 1.0 - shell_completeness)
    f_eff = 1.0 - omega_eff
    regime = _classify_regime(shell_completeness)

    return ElectronConfigResult(
        configuration=config_str,
        noble_gas_core=noble_core,
        valence_electrons=valence,
        shell_completeness=round(shell_completeness, 6),
        omega_eff=round(omega_eff, 6),
        F_eff=round(f_eff, 6),
        period=period,
        group_block=group_block,
        regime=regime.value,
    )


# ── Self-test ────────────────────────────────────────────────────
if __name__ == "__main__":
    for z in [1, 2, 6, 10, 11, 18, 26, 29, 36]:
        r = compute_electron_config(z)
        print(f"Z={z:3d}  {r.configuration:40s}  completeness={r.shell_completeness:.3f}  {r.regime}")

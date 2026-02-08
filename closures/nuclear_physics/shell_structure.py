"""Shell Structure Closure — NUC.INTSTACK.v1

Detects magic number proximity, doubly-magic status, and estimates
shell correction energy for nuclear stability analysis.

Physics:
  Magic numbers: Z = [2, 8, 20, 28, 50, 82, 114]
                 N = [2, 8, 20, 28, 50, 82, 126, 184]

  Doubly-magic nuclei (both Z and N are magic) have exceptional
  stability: He-4, O-16, Ca-40, Ca-48, Ni-56, Ni-78, Sn-100,
  Sn-132, Pb-208, Fl-298*.  (*theoretical)

  Shell correction energy is the deviation of actual binding energy
  from the smooth SEMF prediction.  Large negative δ_shell indicates
  extra stability from closed shells.

  The empirical shell correction is:
    δ_shell = BE_measured − BE_SEMF
  Positive δ_shell → more bound than SEMF predicts (shell closure).
  Negative δ_shell → less bound (mid-shell).

UMCP integration:
  Shell effects modulate the effective drift ω_eff by improving
  (or degrading) binding beyond the smooth SEMF curve.
  Doubly-magic nuclei are GCD-Stable attractors within the
  binding landscape.

Regime classification (shell_structure):
  DoublyMagic:  both Z and N are magic
  SinglyMagic:  either Z or N (not both) is magic
  NearMagic:    within ±2 of a magic number
  MidShell:     far from any magic number

Cross-references:
  Contract:  contracts/NUC.INTSTACK.v1.yaml
  Sources:   Mayer 1949; Jensen 1949; Haxel, Jensen & Suess 1949
"""

from __future__ import annotations

from enum import StrEnum
from typing import NamedTuple


class ShellRegime(StrEnum):
    """Regime based on magic number proximity."""

    DOUBLY_MAGIC = "DoublyMagic"
    SINGLY_MAGIC = "SinglyMagic"
    NEAR_MAGIC = "NearMagic"
    MID_SHELL = "MidShell"


class ShellResult(NamedTuple):
    """Result of shell structure analysis."""

    Z: int
    N: int
    A: int
    magic_proton: bool  # Z is a magic number
    magic_neutron: bool  # N is a magic number
    doubly_magic: bool  # Both Z and N magic
    nearest_magic_Z: int  # Closest magic Z
    distance_to_magic_Z: int  # |Z − nearest_magic_Z|
    nearest_magic_N: int  # Closest magic N
    distance_to_magic_N: int  # |N − nearest_magic_N|
    shell_correction: float  # δ_shell if measured BE provided, else 0
    regime: str


# ── Frozen constants ─────────────────────────────────────────────
MAGIC_Z: list[int] = [2, 8, 20, 28, 50, 82, 114]
MAGIC_N: list[int] = [2, 8, 20, 28, 50, 82, 126, 184]
NEAR_MAGIC_WINDOW = 2  # within ±2 of a magic number

# SEMF coefficients for shell correction estimation
A_V = 15.67
A_S = 17.23
A_C = 0.714
A_A = 23.29
A_P = 11.2


def _nearest_magic(x: int, magic_list: list[int]) -> tuple[int, int]:
    """Find nearest magic number and distance."""
    best_m = magic_list[0]
    best_d = abs(x - best_m)
    for m in magic_list[1:]:
        d = abs(x - m)
        if d < best_d:
            best_m = m
            best_d = d
    return best_m, best_d


def _semf_binding(Z: int, A: int) -> float:
    """Compute smooth SEMF binding energy for shell correction."""
    if A < 1:
        return 0.0
    N = A - Z
    vol = A_V * A
    surf = A_S * A ** (2.0 / 3.0)
    coul = A_C * Z * (Z - 1) / A ** (1.0 / 3.0)
    asym = A_A * (A - 2 * Z) ** 2 / A
    if Z % 2 == 0 and N % 2 == 0:
        pair = A_P / A**0.5
    elif Z % 2 == 1 and N % 2 == 1:
        pair = -A_P / A**0.5
    else:
        pair = 0.0
    return vol - surf - coul - asym + pair


def _classify_regime(
    magic_z: bool,
    magic_n: bool,
    dist_z: int,
    dist_n: int,
) -> ShellRegime:
    """Classify shell structure regime."""
    if magic_z and magic_n:
        return ShellRegime.DOUBLY_MAGIC
    if magic_z or magic_n:
        return ShellRegime.SINGLY_MAGIC
    if dist_z <= NEAR_MAGIC_WINDOW or dist_n <= NEAR_MAGIC_WINDOW:
        return ShellRegime.NEAR_MAGIC
    return ShellRegime.MID_SHELL


def compute_shell(
    Z: int,
    A: int,
    *,
    BE_total_measured: float | None = None,
) -> ShellResult:
    """Analyze shell structure for a nuclide.

    Parameters
    ----------
    Z : int
        Atomic number.
    A : int
        Mass number.
    BE_total_measured : float | None
        If provided, the total measured binding energy (MeV) for
        computing the shell correction δ_shell = BE_meas − BE_SEMF.

    Returns
    -------
    ShellResult
    """
    if A < 1:
        msg = f"Mass number A must be ≥ 1, got {A}"
        raise ValueError(msg)

    N = A - Z

    magic_z = Z in MAGIC_Z
    magic_n = N in MAGIC_N
    doubly = magic_z and magic_n

    near_z, dist_z = _nearest_magic(Z, MAGIC_Z)
    near_n, dist_n = _nearest_magic(N, MAGIC_N)

    # Shell correction
    if BE_total_measured is not None:
        be_semf = _semf_binding(Z, A)
        delta_shell = BE_total_measured - be_semf
    else:
        delta_shell = 0.0

    regime = _classify_regime(magic_z, magic_n, dist_z, dist_n)

    return ShellResult(
        Z=Z,
        N=N,
        A=A,
        magic_proton=magic_z,
        magic_neutron=magic_n,
        doubly_magic=doubly,
        nearest_magic_Z=near_z,
        distance_to_magic_Z=dist_z,
        nearest_magic_N=near_n,
        distance_to_magic_N=dist_n,
        shell_correction=round(delta_shell, 4),
        regime=regime.value,
    )


# ── Self-test ────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        (2, 4, 28.296, "He-4 (doubly magic)"),
        (8, 16, 127.619, "O-16 (doubly magic)"),
        (20, 40, 342.052, "Ca-40 (doubly magic)"),
        (20, 48, 415.991, "Ca-48 (doubly magic)"),
        (28, 62, 545.259, "Ni-62 (magic Z)"),
        (37, 85, None, "Rb-85 (mid-shell)"),
        (50, 120, None, "Sn-120 (magic Z)"),
        (82, 208, 1636.430, "Pb-208 (doubly magic)"),
        (92, 238, 1801.695, "U-238 (mid-shell)"),
    ]

    print(f"{'Nuclide':25s} {'Z':>3s} {'N':>3s} {'Magic':>8s} {'δ_shell':>8s} {'Regime':>12s}")
    print("-" * 70)
    for z, a, be, name in tests:
        r = compute_shell(z, a, BE_total_measured=be)
        magic_str = ("Z" if r.magic_proton else ".") + ("N" if r.magic_neutron else ".")
        print(f"{name:25s} {r.Z:3d} {r.N:3d} {magic_str:>8s} {r.shell_correction:8.2f} {r.regime:>12s}")

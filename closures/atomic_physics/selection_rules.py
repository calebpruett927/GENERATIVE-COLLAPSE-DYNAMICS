"""Selection Rules Closure — ATOM.INTSTACK.v1

Validates electric dipole (E1) selection rules for atomic transitions
and maps rule compliance to GCD invariants.

Physics:
  Electric dipole selection rules:
    Δl = ±1              (orbital angular momentum)
    Δm_l = 0, ±1         (magnetic quantum number)
    Δj = 0, ±1           (total angular momentum, j=0 → j=0 forbidden)
    Δs = 0               (spin conservation)
    Parity change: yes   (Laporte rule)

UMCP integration:
  ω_eff = fraction of rules violated
  F_eff = fraction of rules satisfied
  Regime: Allowed / ForbiddenWeak / ForbiddenStrong

Cross-references:
  Contract:  contracts/ATOM.INTSTACK.v1.yaml
"""

from __future__ import annotations

from enum import StrEnum
from typing import NamedTuple


class SelectionRegime(StrEnum):
    ALLOWED = "Allowed"
    FORBIDDEN_WEAK = "ForbiddenWeak"
    FORBIDDEN_STRONG = "ForbiddenStrong"


class SelectionResult(NamedTuple):
    """Result of selection rule check."""

    delta_l: int
    delta_ml: int
    delta_j: float
    delta_s: float
    parity_change: bool
    l_rule_ok: bool
    ml_rule_ok: bool
    j_rule_ok: bool
    s_rule_ok: bool
    parity_rule_ok: bool
    rules_satisfied: int
    rules_total: int
    omega_eff: float
    F_eff: float
    transition_type: str  # "E1", "M1", "E2", etc.
    regime: str


def compute_selection_rules(
    n_i: int,
    l_i: int,
    ml_i: int,
    s_i: float,
    j_i: float,
    n_f: int,
    l_f: int,
    ml_f: int,
    s_f: float,
    j_f: float,
) -> SelectionResult:
    """Check E1 selection rules for an atomic transition.

    Parameters
    ----------
    n_i, l_i, ml_i, s_i, j_i : initial state quantum numbers
    n_f, l_f, ml_f, s_f, j_f : final state quantum numbers

    Returns
    -------
    SelectionResult
    """
    dl = l_f - l_i
    dml = ml_f - ml_i
    dj = j_f - j_i
    ds = s_f - s_i

    # Parity: (-1)^l changes by (-1)^Δl → requires Δl odd
    parity_change = (dl % 2) != 0

    # Check each rule
    l_ok = abs(dl) == 1
    ml_ok = abs(dml) <= 1
    j_ok = abs(dj) <= 1 and not (j_i == 0 and j_f == 0)
    s_ok = abs(ds) < 1e-9
    parity_ok = parity_change

    satisfied = sum([l_ok, ml_ok, j_ok, s_ok, parity_ok])
    total = 5

    # Classify transition type
    if satisfied == total:
        transition_type = "E1"
    elif l_ok and not parity_change:
        transition_type = "M1"
    elif abs(dl) == 2:
        transition_type = "E2"
    elif abs(dl) == 0 and not parity_change:
        transition_type = "M1"
    else:
        transition_type = "Forbidden"

    omega_eff = (total - satisfied) / total
    f_eff = 1.0 - omega_eff

    if satisfied == total:
        regime = SelectionRegime.ALLOWED
    elif satisfied >= 3:
        regime = SelectionRegime.FORBIDDEN_WEAK
    else:
        regime = SelectionRegime.FORBIDDEN_STRONG

    return SelectionResult(
        delta_l=dl,
        delta_ml=dml,
        delta_j=dj,
        delta_s=ds,
        parity_change=parity_change,
        l_rule_ok=l_ok,
        ml_rule_ok=ml_ok,
        j_rule_ok=j_ok,
        s_rule_ok=s_ok,
        parity_rule_ok=parity_ok,
        rules_satisfied=satisfied,
        rules_total=total,
        omega_eff=round(omega_eff, 6),
        F_eff=round(f_eff, 6),
        transition_type=transition_type,
        regime=regime.value,
    )


# ── Self-test ────────────────────────────────────────────────────
if __name__ == "__main__":
    # Allowed: 2p → 1s  (Lyman-α)
    r = compute_selection_rules(2, 1, 0, 0.5, 1.5, 1, 0, 0, 0.5, 0.5)
    print(f"2p→1s: {r.transition_type}  {r.regime}  rules={r.rules_satisfied}/{r.rules_total}")

    # Forbidden: 2s → 1s  (Δl=0)
    r = compute_selection_rules(2, 0, 0, 0.5, 0.5, 1, 0, 0, 0.5, 0.5)
    print(f"2s→1s: {r.transition_type}  {r.regime}  rules={r.rules_satisfied}/{r.rules_total}")

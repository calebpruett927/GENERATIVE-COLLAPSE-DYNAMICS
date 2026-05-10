"""Generate expected outputs for the nuclear_chain casepack.

Runs all 6 nuclear physics closures on raw_measurements.csv and produces
expected/invariants.json with UMCP Tier-1 invariant rows.

Each experiment category dispatches to the appropriate closure:
  binding       → nuclide_binding
  alpha_decay   → alpha_decay
  decay_chain   → decay_chain (hardcoded chains)
  fissility     → fissility
  shell         → shell_structure
  double_sided  → double_sided_collapse
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from typing import Any

# Add project root so closures are importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from closures.nuclear_physics.alpha_decay import compute_alpha_decay  # noqa: E402
from closures.nuclear_physics.decay_chain import ChainStep, compute_decay_chain  # noqa: E402
from closures.nuclear_physics.double_sided_collapse import compute_double_sided  # noqa: E402
from closures.nuclear_physics.fissility import compute_fissility  # noqa: E402
from closures.nuclear_physics.nuclide_binding import compute_binding  # noqa: E402
from closures.nuclear_physics.shell_structure import compute_shell  # noqa: E402


def _safe_float(s: str) -> float:
    """Parse float from CSV, defaulting to 0.0 on failure."""
    if s == "INF_REC":
        return float("inf")
    try:
        return float(s)
    except (ValueError, TypeError):
        return 0.0


def _safe_int(s: str) -> int:
    """Parse int from CSV, defaulting to 0 on failure."""
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return 0


def _classify_regime(omega: float, f_val: float, s_val: float, c_val: float) -> str:
    """Classify regime using GCD canon thresholds (matches validator logic).

    Stable:   omega < 0.038 AND F > 0.90 AND S < 0.15 AND C < 0.14
    Collapse: omega >= 0.30
    Watch:    everything else
    """
    if omega < 0.038 and f_val > 0.90 and s_val < 0.15 and c_val < 0.14:
        return "Stable"
    if omega >= 0.30:
        return "Collapse"
    return "Watch"


def _round_or_inf(val: float, places: int = 6) -> Any:
    """Round a float, returning 'INF_REC' for infinity."""
    if val == float("inf") or val == float("-inf"):
        return "INF_REC"
    return round(val, places)


def _make_row(
    exp_id: str,
    omega: float,
    f_val: float,
    s_val: float,
    c_val: float,
    kappa: float,
    ic: float,
    tau_R: Any,
    regime: str,
) -> dict[str, Any]:
    """Build a conformant invariant row matching the schema."""
    return {
        "t": exp_id,
        "omega": round(omega, 6),
        "F": round(f_val, 6),
        "S": round(s_val, 6),
        "C": round(c_val, 6),
        "tau_R": _round_or_inf(tau_R) if isinstance(tau_R, float) else tau_R,
        "kappa": round(kappa, 6),
        "IC": round(ic, 6),
        "regime": {
            "label": regime,
            "critical_overlay": regime == "Collapse",
        },
    }


# ── Hardcoded decay chains ──────────────────────────────────────
U238_CHAIN: list[ChainStep] = [
    ChainStep("U-238", 92, 238, "alpha", 1.41e17, 4.270),
    ChainStep("Th-234", 90, 234, "beta_minus", 2.08e6, 0.273),
    ChainStep("Pa-234", 91, 234, "beta_minus", 6.70e1, 2.197),
    ChainStep("U-234", 92, 234, "alpha", 7.74e12, 4.858),
    ChainStep("Th-230", 90, 230, "alpha", 2.38e12, 4.770),
    ChainStep("Ra-226", 88, 226, "alpha", 5.05e10, 4.871),
    ChainStep("Rn-222", 86, 222, "alpha", 3.30e5, 5.590),
    ChainStep("Po-218", 84, 218, "alpha", 1.86e2, 6.115),
    ChainStep("Pb-214", 82, 214, "beta_minus", 1.61e3, 1.024),
    ChainStep("Bi-214", 83, 214, "beta_minus", 1.19e3, 3.272),
    ChainStep("Po-214", 84, 214, "alpha", 1.64e-4, 7.833),
    ChainStep("Pb-210", 82, 210, "beta_minus", 7.01e8, 0.064),
    ChainStep("Bi-210", 83, 210, "beta_minus", 4.33e5, 1.163),
    ChainStep("Po-210", 84, 210, "alpha", 1.20e7, 5.407),
]

TH232_CHAIN: list[ChainStep] = [
    ChainStep("Th-232", 90, 232, "alpha", 4.43e17, 4.083),
    ChainStep("Ra-228", 88, 228, "beta_minus", 1.81e8, 0.046),
    ChainStep("Ac-228", 89, 228, "beta_minus", 2.21e4, 2.127),
    ChainStep("Th-228", 90, 228, "alpha", 6.04e7, 5.520),
    ChainStep("Ra-224", 88, 224, "alpha", 3.14e5, 5.789),
    ChainStep("Rn-220", 86, 220, "alpha", 5.56e1, 6.405),
    ChainStep("Po-216", 84, 216, "alpha", 1.45e-1, 6.906),
    ChainStep("Pb-212", 82, 212, "beta_minus", 3.83e4, 0.574),
    ChainStep("Bi-212", 83, 212, "beta_minus", 3.63e3, 2.254),
    ChainStep("Po-212", 84, 212, "alpha", 2.99e-7, 8.954),
]

U235_CHAIN: list[ChainStep] = [
    ChainStep("U-235", 92, 235, "alpha", 2.22e16, 4.679),
    ChainStep("Th-231", 90, 231, "beta_minus", 9.19e4, 0.391),
    ChainStep("Pa-231", 91, 231, "alpha", 1.03e12, 5.150),
    ChainStep("Ac-227", 89, 227, "beta_minus", 6.87e8, 0.045),
    ChainStep("Th-227", 90, 227, "alpha", 1.62e6, 6.146),
    ChainStep("Ra-223", 88, 223, "alpha", 9.88e5, 5.979),
    ChainStep("Rn-219", 86, 219, "alpha", 3.96e0, 6.946),
    ChainStep("Po-215", 84, 215, "alpha", 1.78e-3, 7.526),
    ChainStep("Pb-211", 82, 211, "beta_minus", 2.17e3, 1.367),
    ChainStep("Bi-211", 83, 211, "alpha", 1.28e2, 6.751),
    ChainStep("Tl-207", 81, 207, "beta_minus", 2.86e2, 1.423),
]

DECAY_CHAINS: dict[str, list[ChainStep]] = {
    "NUC18": U238_CHAIN,
    "NUC19": TH232_CHAIN,
    "NUC20": U235_CHAIN,
}


def _process_binding(row: dict[str, str]) -> dict[str, Any]:
    """Process binding experiment."""
    Z = _safe_int(row["Z"])
    A = _safe_int(row["A"])
    be_measured = _safe_float(row["BE_per_A_measured"])

    r = compute_binding(Z, A, BE_per_A_measured=be_measured if be_measured > 0 else None)

    # Map to GCD invariants
    omega = r.omega_eff
    f_val = r.F_eff
    s_val = 0.0  # No entropy proxy for pure binding
    c_val = 0.0  # No coupling for pure binding

    # For very light nuclei with BE/A = 0, cap omega at 1
    omega = min(1.0, omega)
    f_val = max(0.0, 1.0 - omega)

    kappa = math.log(max(1e-12, f_val * 0.95)) if f_val > 0 else -30.0
    ic = math.exp(kappa)

    regime = _classify_regime(omega, f_val, s_val, c_val)

    return _make_row(row["exp_id"], omega, f_val, s_val, c_val, kappa, ic, 0.0, regime)


def _process_alpha_decay(row: dict[str, str]) -> dict[str, Any]:
    """Process alpha decay experiment."""
    Z = _safe_int(row["Z"])
    A = _safe_int(row["A"])
    Q = _safe_float(row["Q_alpha_MeV"])
    t_half = _safe_float(row["half_life_s"])

    r = compute_alpha_decay(
        Z,
        A,
        Q,
        half_life_s_measured=t_half if t_half < float("inf") else None,
    )

    # Map to GCD invariants via binding
    be_measured = _safe_float(row["BE_per_A_measured"])
    b = compute_binding(Z, A, BE_per_A_measured=be_measured if be_measured > 0 else None)

    omega = b.omega_eff
    f_val = b.F_eff

    # Decay-active nuclei have higher entropy from Coulomb instability
    s_val = min(1.0, Q / 10.0) if Q > 0 else 0.0
    c_val = 0.0

    omega = min(1.0, omega)
    f_val = max(0.0, 1.0 - omega)
    kappa = math.log(max(1e-12, f_val * 0.95)) if f_val > 0 else -30.0
    ic = math.exp(kappa)

    regime = _classify_regime(omega, f_val, s_val, c_val)

    return _make_row(row["exp_id"], omega, f_val, s_val, c_val, kappa, ic, r.mean_lifetime_s, regime)


def _process_decay_chain(row: dict[str, str]) -> dict[str, Any]:
    """Process decay chain experiment."""
    exp_id = row["exp_id"]
    chain = DECAY_CHAINS.get(exp_id, [])
    r = compute_decay_chain(chain)

    # Map to GCD via parent binding
    Z = _safe_int(row["Z"])
    A = _safe_int(row["A"])
    be_measured = _safe_float(row["BE_per_A_measured"])
    b = compute_binding(Z, A, BE_per_A_measured=be_measured if be_measured > 0 else None)

    omega = b.omega_eff
    f_val = b.F_eff
    s_val = min(1.0, r.total_Q_MeV / 50.0) if r.total_Q_MeV > 0 else 0.0
    c_val = 0.0

    omega = min(1.0, omega)
    f_val = max(0.0, 1.0 - omega)
    kappa = math.log(max(1e-12, f_val * 0.95)) if f_val > 0 else -30.0
    ic = math.exp(kappa)

    regime = _classify_regime(omega, f_val, s_val, c_val)

    bn_tau = r.bottleneck_half_life_s / 0.693147 if r.bottleneck_half_life_s < float("inf") else float("inf")
    return _make_row(exp_id, omega, f_val, s_val, c_val, kappa, ic, bn_tau, regime)


def _process_fissility(row: dict[str, str]) -> dict[str, Any]:
    """Process fissility experiment."""
    Z = _safe_int(row["Z"])
    A = _safe_int(row["A"])
    r = compute_fissility(Z, A)

    be_measured = _safe_float(row["BE_per_A_measured"])
    b = compute_binding(Z, A, BE_per_A_measured=be_measured if be_measured > 0 else None)

    omega = b.omega_eff
    f_val = b.F_eff
    # Fissility contributes to entropy
    s_val = min(1.0, r.fissility_x * 0.20)
    c_val = 0.0

    omega = min(1.0, omega)
    f_val = max(0.0, 1.0 - omega)
    kappa = math.log(max(1e-12, f_val * 0.95)) if f_val > 0 else -30.0
    ic = math.exp(kappa)

    regime = _classify_regime(omega, f_val, s_val, c_val)

    return _make_row(row["exp_id"], omega, f_val, s_val, c_val, kappa, ic, 0.0, regime)


def _process_shell(row: dict[str, str]) -> dict[str, Any]:
    """Process shell structure experiment."""
    Z = _safe_int(row["Z"])
    A = _safe_int(row["A"])
    be_measured = _safe_float(row["BE_per_A_measured"])
    be_total = be_measured * A if be_measured > 0 else None
    r = compute_shell(Z, A, BE_total_measured=be_total)

    b = compute_binding(Z, A, BE_per_A_measured=be_measured if be_measured > 0 else None)

    omega = b.omega_eff
    f_val = b.F_eff
    s_val = 0.0
    # Shell closure acts as coupling (stabilization)
    c_val = 0.10 if r.doubly_magic else (0.05 if r.magic_proton or r.magic_neutron else 0.0)

    omega = min(1.0, omega)
    f_val = max(0.0, 1.0 - omega)
    kappa = math.log(max(1e-12, f_val * 0.95)) if f_val > 0 else -30.0
    ic = math.exp(kappa)

    regime = _classify_regime(omega, f_val, s_val, c_val)

    return _make_row(row["exp_id"], omega, f_val, s_val, c_val, kappa, ic, 0.0, regime)


def _process_double_sided(row: dict[str, str]) -> dict[str, Any]:
    """Process double-sided collapse experiment."""
    Z = _safe_int(row["Z"])
    A = _safe_int(row["A"])
    be_measured = _safe_float(row["BE_per_A_measured"])
    r = compute_double_sided(Z, A, be_measured if be_measured > 0 else 0.0)

    omega = r.omega_eff
    f_val = max(0.0, 1.0 - omega)
    s_val = 0.0
    c_val = 0.0

    omega = min(1.0, omega)
    kappa = math.log(max(1e-12, f_val * 0.95)) if f_val > 0 else -30.0
    ic = math.exp(kappa)

    regime = _classify_regime(omega, f_val, s_val, c_val)

    return _make_row(row["exp_id"], omega, f_val, s_val, c_val, kappa, ic, 0.0, regime)


DISPATCH: dict[str, Any] = {
    "binding": _process_binding,
    "alpha_decay": _process_alpha_decay,
    "decay_chain": _process_decay_chain,
    "fissility": _process_fissility,
    "shell": _process_shell,
    "double_sided": _process_double_sided,
}


def main() -> None:
    """Generate expected/invariants.json from raw_measurements.csv."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "raw_measurements.csv")
    out_dir = os.path.join(script_dir, "expected")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "invariants.json")

    results: list[dict[str, Any]] = []

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cat = row["category"]
            handler = DISPATCH.get(cat)
            if handler is None:
                print(f"[WARN] Unknown category '{cat}' for {row['exp_id']}, skipping")
                continue
            result = handler(row)
            results.append(result)

    output = {
        "schema": "schemas/invariants.schema.json",
        "format": "tier1_invariants",
        "contract_id": "NUC.INTSTACK.v1",
        "closure_registry_id": "UMCP.CLOSURES.NUC.v1",
        "canon": {
            "pre_doi": "10.1007/BF01333234",
            "post_doi": "10.5281/zenodo.14854321",
            "weld_id": "W-2026-02-08-NUC-CHAIN",
            "timezone": "UTC",
        },
        "rows": results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(results)} experiment results → {out_path}")

    # Summary
    regimes: dict[str, int] = {"Stable": 0, "Watch": 0, "Collapse": 0}
    for r in results:
        label = r["regime"]["label"]
        regimes[label] = regimes.get(label, 0) + 1
    print(f"  Stable: {regimes['Stable']}, Watch: {regimes['Watch']}, Collapse: {regimes['Collapse']}")


if __name__ == "__main__":
    main()

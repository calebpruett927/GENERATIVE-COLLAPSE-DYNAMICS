"""Generate expected outputs for the quantum_mechanics_complete casepack.

Runs all 6 quantum mechanics closures on raw_measurements.csv and produces
expected/invariants.json with UMCP Tier-1 invariant rows.

Each experiment category dispatches to the appropriate closure:
  wavefunction → wavefunction_collapse
  entanglement → entanglement
  tunneling    → tunneling
  oscillator   → harmonic_oscillator
  spin         → spin_measurement
  uncertainty  → uncertainty_principle
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

from closures.quantum_mechanics.entanglement import compute_entanglement  # noqa: E402
from closures.quantum_mechanics.harmonic_oscillator import compute_harmonic_oscillator  # noqa: E402
from closures.quantum_mechanics.spin_measurement import compute_spin_measurement  # noqa: E402
from closures.quantum_mechanics.tunneling import compute_tunneling  # noqa: E402
from closures.quantum_mechanics.uncertainty_principle import compute_uncertainty  # noqa: E402
from closures.quantum_mechanics.wavefunction_collapse import compute_wavefunction_collapse  # noqa: E402


def _safe_float(s: str) -> float:
    """Parse float from CSV, defaulting to 0.0 on failure."""
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
    elif omega >= 0.30:
        return "Collapse"
    else:
        return "Watch"


def _parse_list(row: dict[str, str], prefix: str, count: int) -> list[float]:
    """Parse numbered columns like psi_amp_0, psi_amp_1, ... into a list."""
    result: list[float] = []
    for i in range(count):
        key = f"{prefix}_{i}"
        val = _safe_float(row.get(key, "0"))
        result.append(val)
    return result


def compute_invariants_for_experiment(row: dict[str, str]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute UMCP Tier-1 invariants for a single QM experiment.

    Returns (invariant_row, domain_extensions).
    """
    cat = row["category"]
    exp_id = row["exp_id"]

    # Initialize defaults
    omega = 0.0
    f_val = 1.0
    s_val = 0.0
    c_val = 0.0
    kappa = 0.0
    ic = 1.0
    domain_regime = "Stable"
    domain_extensions: dict[str, Any] = {}

    if cat == "wavefunction":
        # Parse amplitudes and measurement probabilities
        psi_amps = _parse_list(row, "psi_amp", 4)
        meas_probs = _parse_list(row, "meas_prob", 4)
        obs_idx = _safe_int(row.get("observed_idx", "0"))

        # Filter out trailing zeros for proper normalization
        psi_active = [a for a in psi_amps if a > 0]
        meas_active = [m for m in meas_probs if m > 0]

        result = compute_wavefunction_collapse(
            psi_amplitudes=psi_active if psi_active else psi_amps[:2],
            measurement_probs=meas_active if meas_active else meas_probs[:2],
            observed_outcome_idx=obs_idx,
        )

        omega = min(1.0, result["delta_P"])
        f_val = max(0.0, 1.0 - omega)
        s_val = min(1.0, 1.0 - result["purity"])
        c_val = 0.0  # No coupling in single-particle measurement
        domain_regime = result["regime"]
        domain_extensions["wavefunction_collapse"] = result

    elif cat == "entanglement":
        rho_eigs = _parse_list(row, "rho_eig", 4)
        bell_corrs_raw = [
            _safe_float(row.get("bell_E_ab", "0")),
            _safe_float(row.get("bell_E_ab2", "0")),
            _safe_float(row.get("bell_E_a2b", "0")),
            _safe_float(row.get("bell_E_a2b2", "0")),
        ]

        # Filter zero eigenvalues for proper computation
        rho_active = [e for e in rho_eigs if e > 0]
        if not rho_active:
            rho_active = rho_eigs

        bell_corrs = bell_corrs_raw if any(abs(b) > 0 for b in bell_corrs_raw) else None

        result = compute_entanglement(
            rho_eigenvalues=rho_active,
            bell_correlations=bell_corrs,
        )

        conc = result["concurrence"]
        omega = max(0.0, 1.0 - conc)
        f_val = max(0.0, conc)
        s_val = min(1.0, result["S_vN"])
        c_val = min(1.0, result["bell_parameter"] / 2.8284)  # Normalize by Tsirelson bound
        domain_regime = result["regime"]
        domain_extensions["entanglement"] = result

    elif cat == "tunneling":
        e_particle = _safe_float(row["E_particle_eV"])
        v_barrier = _safe_float(row["V_barrier_eV"])
        width = _safe_float(row["barrier_width_nm"])
        mass_str = row.get("particle_mass_kg", "")
        mass = _safe_float(mass_str) if mass_str and mass_str != "0.0" else None

        result = compute_tunneling(e_particle, v_barrier, width, mass)

        t_coeff = result["T_coeff"]
        # For tunneling: opacity is the "normal" state; high T is unusual
        omega = min(1.0, 1.0 - t_coeff)
        f_val = max(0.0, t_coeff)
        s_val = min(1.0, 1.0 - t_coeff)  # Opacity → entropy
        c_val = 0.0
        domain_regime = result["regime"]
        domain_extensions["tunneling"] = result

    elif cat == "oscillator":
        n = _safe_int(row["n_quanta"])
        hw = _safe_float(row["omega_freq_eV"])
        e_obs = _safe_float(row["E_observed_eV"])
        alpha = _safe_float(row.get("coherent_alpha", "0"))
        squeeze = _safe_float(row.get("squeeze_r", "0"))

        result = compute_harmonic_oscillator(n, hw, e_obs, alpha, squeeze)

        delta_e = result["delta_E"]
        omega = min(1.0, delta_e)
        f_val = max(0.0, 1.0 - omega)
        s_val = min(1.0, 1.0 / (n + 1))  # Higher n → more entropy
        c_val = 0.0
        domain_regime = result["regime"]
        domain_extensions["harmonic_oscillator"] = result

    elif cat == "spin":
        s_total = _safe_float(row["S_total"])
        s_z_obs = _safe_float(row["S_z_observed"])
        b_field = _safe_float(row["B_field_T"])
        g = _safe_float(row.get("g_factor", "0"))
        g_factor = g if g > 0 else None

        result = compute_spin_measurement(s_total, s_z_obs, b_field, g_factor)

        spin_fid = result["spin_fidelity"]
        omega = max(0.0, 1.0 - spin_fid)
        f_val = max(0.0, spin_fid)
        s_val = min(1.0, 1.0 - spin_fid)
        c_val = 0.0
        domain_regime = result["regime"]
        domain_extensions["spin_measurement"] = result

    elif cat == "uncertainty":
        dx = _safe_float(row["delta_x_m"])
        dp = _safe_float(row["delta_p_kgms"])

        result = compute_uncertainty(dx, dp, units="SI")

        ratio = result["heisenberg_ratio"]
        if ratio < 1.0:
            # Heisenberg violation → Collapse
            omega = 1.0
            f_val = 0.0
            s_val = 1.0
            domain_regime = "Violation"
        else:
            # Map ratio to drift: closer to minimum → lower omega
            omega = min(1.0, max(0.0, 1.0 - 1.0 / ratio))
            f_val = max(0.0, 1.0 - omega)
            s_val = min(1.0, math.log(ratio) / 10.0) if ratio > 1 else 0.0
        c_val = 0.0
        domain_regime = result["regime"]
        domain_extensions["uncertainty_principle"] = result

    # IC = exp(kappa), kappa = ln(IC)
    # Round IC first, then derive kappa from rounded IC so exp(kappa) == IC exactly
    ic = max(0.001, f_val)
    ic_rounded = round(ic, 6)
    kappa = math.log(ic_rounded)

    # Round omega/F/S/C before regime classification (validator sees rounded values)
    omega_r = round(omega, 6)
    f_r = round(f_val, 6)
    s_r = round(s_val, 6)
    c_r = round(c_val, 6)

    # Critical overlay: IC < 0.30
    critical_overlay = ic_rounded < 0.30

    # Classify regime using GCD canon thresholds (must match validator)
    umcp_regime = _classify_regime(omega_r, f_r, s_r, c_r)

    invariant_row: dict[str, Any] = {
        "t": exp_id,
        "omega": omega_r,
        "F": f_r,
        "S": s_r,
        "C": c_r,
        "tau_R": "INF_REC",
        "kappa": kappa,
        "IC": ic_rounded,
        "regime": {
            "label": umcp_regime,
            "critical_overlay": critical_overlay,
        },
    }
    domain_extensions["domain_regime"] = domain_regime

    return invariant_row, domain_extensions


def main() -> None:
    """Generate expected/invariants.json from raw_measurements.csv."""
    casepack_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(casepack_dir, "raw_measurements.csv")
    expected_dir = os.path.join(casepack_dir, "expected")
    os.makedirs(expected_dir, exist_ok=True)

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows_raw = list(reader)

    invariant_rows: list[dict[str, Any]] = []
    all_domain_extensions: dict[str, Any] = {}

    for row in rows_raw:
        inv_row, dom_ext = compute_invariants_for_experiment(row)
        invariant_rows.append(inv_row)
        all_domain_extensions[row["exp_id"]] = dom_ext

    # Count by category
    categories: dict[str, int] = {}
    for r in rows_raw:
        cat = r["category"]
        categories[cat] = categories.get(cat, 0) + 1

    # Build invariants.json
    invariants: dict[str, Any] = {
        "schema": "schemas/invariants.schema.json",
        "format": "tier1_invariants",
        "contract_id": "QM.INTSTACK.v1",
        "closure_registry_id": "UMCP.CLOSURES.QM.v1",
        "canon": {
            "pre_doi": "10.1103/PhysRev.47.777",
            "post_doi": "10.5281/zenodo.14854321",
            "weld_id": "W-2026-02-07-QM-COMPLETE",
            "timezone": "UTC",
        },
        "rows": invariant_rows,
        "notes": (
            "Complete quantum mechanics casepack covering 30 experiments across "
            "6 subdomains: wavefunction collapse (5), entanglement (5), "
            "tunneling (5), harmonic oscillator (5), spin measurement (5), "
            "and uncertainty principle (5). Each closure maps quantum observables "
            "to UMCP Tier-1 invariants. Wavefunction collapse is the literal "
            "archetype of Generative Collapse Dynamics. Domain-specific regime "
            "labels mapped to UMCP standard: Stable/Watch/Collapse."
        ),
        "extensions": {
            "quantum_mechanics_summary": {
                "total_experiments": len(invariant_rows),
                "categories": categories,
                "subdomains_covered": sorted(categories.keys()),
            },
            "domain_closure_outputs": all_domain_extensions,
        },
    }

    out_path = os.path.join(expected_dir, "invariants.json")
    with open(out_path, "w") as f:
        json.dump(invariants, f, indent=2)

    print(f"✓ Generated {out_path}")
    print(f"  {len(invariant_rows)} rows written")

    # Print regime summary
    regimes: dict[str, int] = {}
    for inv_row in invariant_rows:
        regime_obj: dict[str, object] = inv_row["regime"]
        label = str(regime_obj.get("label", "Unknown"))
        regimes[label] = regimes.get(label, 0) + 1
    for regime, count in sorted(regimes.items()):
        print(f"  {regime}: {count}")


if __name__ == "__main__":
    main()

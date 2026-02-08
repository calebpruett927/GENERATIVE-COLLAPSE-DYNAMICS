"""Generate expected outputs for the astronomy_complete casepack.

Runs all 6 astronomy closures on raw_measurements.csv and produces
expected/invariants.json with UMCP Tier-1 invariant rows.
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

from closures.astronomy.distance_ladder import compute_distance_ladder  # noqa: E402
from closures.astronomy.gravitational_dynamics import compute_gravitational_dynamics  # noqa: E402
from closures.astronomy.orbital_mechanics import compute_orbital_mechanics  # noqa: E402
from closures.astronomy.spectral_analysis import compute_spectral_analysis  # noqa: E402
from closures.astronomy.stellar_evolution import compute_stellar_evolution  # noqa: E402
from closures.astronomy.stellar_luminosity import compute_stellar_luminosity  # noqa: E402


def _safe_float(s: str) -> float:
    try:
        return float(s)
    except (ValueError, TypeError):
        return 0.0


# Domain-specific regime labels → UMCP standard three-valued regime
_REGIME_MAP: dict[str, str] = {
    # Stellar luminosity
    "Consistent": "Stable",
    "Mild": "Watch",
    "Significant": "Watch",
    "Anomalous": "Collapse",
    # Spectral
    "Excellent": "Stable",
    "Good": "Stable",
    "Marginal": "Watch",
    "Poor": "Collapse",
    # Orbital
    "Stable": "Stable",
    "Eccentric": "Watch",
    "Unstable": "Collapse",
    "Escape": "Collapse",
    # Distance
    "High": "Stable",
    "Moderate": "Watch",
    "Low": "Watch",
    "Unreliable": "Collapse",
    # Dynamics
    "Equilibrium": "Stable",
    "Relaxing": "Watch",
    "Disturbed": "Watch",
    "Unbound": "Collapse",
    # Evolution
    "Pre-MS": "Stable",
    "Main-Seq": "Stable",
    "Subgiant": "Watch",
    "Giant": "Watch",
    "Post-AGB": "Collapse",
}


def _map_regime(domain_label: str) -> str:
    """Map domain-specific regime to UMCP standard Stable/Watch/Collapse."""
    return _REGIME_MAP.get(domain_label, "Watch")


def compute_invariants_for_star(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute UMCP Tier-1 invariants for a single star/object row.

    Returns (invariant_row, domain_extensions) where domain_extensions
    are stored at the top level (not per-row, since schema forbids it).
    """
    cat = row.get("category", "")
    star_id = row["star_id"]

    # Initialize result
    omega = 0.0
    F_val = 1.0
    S_val = 0.0
    C_val = 0.0
    kappa = 0.0
    IC = 1.0
    domain_regime = "Stable"
    domain_extensions: dict[str, Any] = {}

    if cat in ("main_sequence", "subgiant", "giant", "supergiant", "white_dwarf"):
        m_star = _safe_float(row["M_star"])
        t_eff = _safe_float(row["T_eff"])
        r_star = _safe_float(row["R_star"])
        l_obs = _safe_float(row["L_obs"])
        b_v = _safe_float(row["B_V"])
        sp_class = row["spectral_class"]
        m_abs = _safe_float(row["M_abs"])
        m_app = _safe_float(row["m_app"])
        pi_arcsec = _safe_float(row["pi_arcsec"])
        z_cosmo = _safe_float(row["z_cosmo"])
        age = _safe_float(row["age_Gyr"])

        # Run closures
        lum = compute_stellar_luminosity(m_star, t_eff, r_star) if m_star > 0 else None
        spec = compute_spectral_analysis(t_eff, b_v, sp_class) if t_eff > 0 else None
        dist = compute_distance_ladder(m_app, m_abs, pi_arcsec, z_cosmo)
        evol = compute_stellar_evolution(m_star, l_obs, t_eff, age) if m_star > 0 else None

        # Map to UMCP invariants
        if lum:
            delta_l = lum["delta_L"]
            omega = min(1.0, delta_l)
            F_val = max(0.0, 1.0 - omega)
            domain_extensions["stellar_luminosity"] = lum

        if spec:
            # S = spectral chi2 normalized
            S_val = min(1.0, spec["chi2_spectral"] / 5.0)
            domain_extensions["spectral_analysis"] = spec

        if dist:
            C_val = min(1.0, dist["distance_consistency"])
            domain_extensions["distance_ladder"] = dist

        if evol:
            domain_extensions["stellar_evolution"] = evol

        # IC = exp(kappa), kappa = ln(IC)
        # Use measurement precision as IC proxy: smaller delta_L → higher IC
        if omega < 1.0:
            IC = max(0.001, 1.0 - omega)
            kappa = math.log(IC) if IC > 0 else -10.0
        else:
            IC = 0.001
            kappa = math.log(IC)

        # Regime from luminosity closure
        if lum:
            domain_regime = lum["regime"]
        elif evol:
            domain_regime = evol["regime"]

    elif cat == "orbital":
        p_orb = _safe_float(row["P_orb_s"])
        a_semi = _safe_float(row["a_semi_AU"])
        e_orb = _safe_float(row["e_orb"])
        m_total = _safe_float(row["M_total"])

        orb = compute_orbital_mechanics(p_orb, a_semi, m_total, e_orb)
        omega = min(1.0, e_orb)  # eccentricity as drift
        F_val = max(0.0, 1.0 - omega)
        S_val = min(1.0, orb["kepler_residual"] * 100)  # Kepler residual × 100
        C_val = min(1.0, e_orb)
        IC = max(0.001, F_val)
        kappa = math.log(IC) if IC > 0 else -10.0
        domain_regime = orb["regime"]
        domain_extensions["orbital_mechanics"] = orb

    elif cat == "galactic":
        v_rot = _safe_float(row["v_rot_kms"])
        r_obs = _safe_float(row["r_obs_kpc"])
        sigma_v = _safe_float(row["sigma_v_kms"])
        m_lum = _safe_float(row["M_luminous"])

        dyn = compute_gravitational_dynamics(v_rot, r_obs, sigma_v, m_lum)
        omega = min(1.0, dyn["dark_matter_fraction"])  # DM fraction as hidden state
        F_val = max(0.0, 1.0 - omega)
        S_val = min(1.0, dyn["virial_ratio"])
        C_val = min(1.0, dyn["dark_matter_fraction"])
        IC = max(0.001, F_val)
        kappa = math.log(IC) if IC > 0 else -10.0
        domain_regime = dyn["regime"]
        domain_extensions["gravitational_dynamics"] = dyn

    # Critical overlay: IC < 0.30
    critical_overlay = IC < 0.30

    # Map domain regime to UMCP standard
    umcp_regime = _map_regime(domain_regime)

    invariant_row = {
        "t": star_id,
        "omega": round(omega, 6),
        "F": round(F_val, 6),
        "S": round(S_val, 6),
        "C": round(C_val, 6),
        "tau_R": "INF_REC",
        "kappa": round(kappa, 12),
        "IC": round(IC, 6),
        "regime": {
            "label": umcp_regime,
            "critical_overlay": critical_overlay,
        },
    }
    # Store domain regime for top-level extensions
    domain_extensions["domain_regime"] = domain_regime

    return invariant_row, domain_extensions


def main() -> None:
    casepack_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(casepack_dir, "raw_measurements.csv")
    expected_dir = os.path.join(casepack_dir, "expected")
    os.makedirs(expected_dir, exist_ok=True)

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows_raw = list(reader)

    invariant_rows = []
    all_domain_extensions = {}  # star_id → domain closure outputs
    for row in rows_raw:
        inv_row, dom_ext = compute_invariants_for_star(row)
        invariant_rows.append(inv_row)
        all_domain_extensions[row["star_id"]] = dom_ext

    # Build invariants.json
    invariants = {
        "schema": "schemas/invariants.schema.json",
        "format": "tier1_invariants",
        "contract_id": "ASTRO.INTSTACK.v1",
        "closure_registry_id": "UMCP.CLOSURES.ASTRO.v1",
        "canon": {
            "pre_doi": "10.1086/133630",
            "post_doi": "10.5281/zenodo.14854321",
            "weld_id": "W-2026-02-07-ASTRO-COMPLETE",
            "timezone": "UTC",
        },
        "rows": invariant_rows,
        "notes": (
            "Complete astronomy casepack covering 20 stellar objects (O through M types, "
            "giants, supergiants, white dwarfs), 5 orbital systems (planets + comet), "
            "and 3 galactic dynamics targets. Closures: stellar_luminosity, "
            "spectral_analysis, distance_ladder, orbital_mechanics, "
            "gravitational_dynamics, stellar_evolution. "
            "Domain-specific regime labels mapped to UMCP standard: "
            "Stable/Watch/Collapse. Original domain regimes in extensions."
        ),
        "extensions": {
            "astronomy_summary": {
                "total_objects": len(invariant_rows),
                "stellar": sum(
                    1
                    for r in rows_raw
                    if r["category"] in ("main_sequence", "subgiant", "giant", "supergiant", "white_dwarf")
                ),
                "orbital": sum(1 for r in rows_raw if r["category"] == "orbital"),
                "galactic": sum(1 for r in rows_raw if r["category"] == "galactic"),
                "spectral_types_covered": sorted(
                    {r["spectral_class"][0] for r in rows_raw if r["spectral_class"] and r["spectral_class"] != "n/a"}
                ),
            },
            "domain_closure_outputs": all_domain_extensions,
        },
    }

    out_path = os.path.join(expected_dir, "invariants.json")
    with open(out_path, "w") as f:
        json.dump(invariants, f, indent=2)

    print(f"✓ Generated {out_path}")
    print(f"  {len(invariant_rows)} rows written")

    # Print summary
    regimes: dict[str, int] = {}
    for row in invariant_rows:
        r = row["regime"]["label"]
        regimes[r] = regimes.get(r, 0) + 1
    for regime, count in sorted(regimes.items()):
        print(f"  {regime}: {count}")


if __name__ == "__main__":
    main()

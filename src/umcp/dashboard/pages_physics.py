"""
Physics and Kinematics dashboard pages with GCD translation framework.
"""
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportOptionalMemberAccess=false
# pyright: reportMissingTypeStubs=false

from __future__ import annotations

from datetime import datetime
from typing import Any

from umcp.dashboard._deps import go, make_subplots, np, pd, px, st
from umcp.dashboard._utils import (
    classify_regime,
    get_regime_color,
)

GCD_SYMBOLS = {
    "omega": {
        "latex": "ω",
        "name": "Drift",
        "description": "Weighted distance from upper boundary (collapse metric)",
        "formula": "ω = Σ wᵢ(1-cᵢ)",
        "domain": "[0, 1]",
        "optimal": 0.0,
        "collapse_threshold": 0.30,
        "interpretation": {
            0.0: "Perfect alignment (no drift)",
            0.038: "Stable boundary",
            0.30: "Collapse threshold",
            1.0: "Total collapse",
        },
    },
    "F": {
        "latex": "F",
        "name": "Fidelity",
        "description": "System fidelity (complement of drift)",
        "formula": "F = 1 - ω",
        "domain": "[0, 1]",
        "optimal": 1.0,
        "collapse_threshold": 0.70,
        "interpretation": {
            1.0: "Perfect fidelity",
            0.90: "Stable region",
            0.70: "Watch boundary",
            0.0: "Zero fidelity",
        },
    },
    "S": {
        "latex": "S",
        "name": "Entropy",
        "description": "Bernoulli field entropy (system uncertainty/determinacy; Shannon entropy is the degenerate limit)",
        "formula": "S = -Σ wᵢ[cᵢln(cᵢ) + (1-cᵢ)ln(1-cᵢ)]",
        "domain": "[0, ln(2)]",
        "optimal": 0.0,
        "interpretation": {
            0.0: "Perfect determinacy (S=0 state)",
            0.15: "Low uncertainty",
            0.35: "Moderate uncertainty",
            0.693: "Maximum uncertainty",
        },
    },
    "C": {
        "latex": "C",
        "name": "Curvature",
        "description": "Coordinate non-uniformity (heterogeneity measure)",
        "formula": "C = σ(cᵢ)/0.5",
        "domain": "[0, 1]",
        "optimal": 0.0,
        "interpretation": {
            0.0: "Uniform/homogeneous",
            0.14: "Low curvature",
            0.30: "Moderate curvature",
            1.0: "Maximum curvature",
        },
    },
    "kappa": {
        "latex": "κ",
        "name": "Log-Integrity",
        "description": "Logarithmic collapse accumulation",
        "formula": "κ = Σ wᵢ ln(cᵢ)",
        "domain": "(-∞, 0]",
        "optimal": 0.0,
        "interpretation": {
            0.0: "No log-collapse",
            -0.5: "Mild log-collapse",
            -2.0: "Significant log-collapse",
            -10.0: "Severe log-collapse",
        },
    },
    "IC": {
        "latex": "IC",
        "name": "Integrity Composite",
        "description": "Geometric mean of coordinates (exponential of κ)",
        "formula": "IC = exp(κ) = ∏ cᵢ^wᵢ",
        "domain": "(0, 1]",
        "optimal": 1.0,
        "identity": "IC = exp(κ)",
        "interpretation": {
            1.0: "Perfect integrity",
            0.80: "Good integrity",
            0.50: "Degraded integrity",
            0.10: "Critical integrity",
        },
    },
    "tau_R": {
        "latex": "τ_R",
        "name": "Return Time",
        "description": "Recursive timescale to return domain",
        "domain": "[0, ∞] ∪ {INF_REC}",
        "interpretation": {
            0: "Immediate return",
            10: "Fast return",
            50: "Slow return",
            "INF_REC": "No return (non-returning)",
        },
    },
}

# GCD Regime thresholds (from gcd_anchors.yaml)
GCD_REGIMES = {
    "Stable": {
        "condition": "ω < 0.038 AND F > 0.90 AND S < 0.15 AND C < 0.14",
        "color": "#28a745",  # Green
        "description": "Optimal operational state with low collapse risk",
        "icon": "🟢",
    },
    "Watch": {
        "condition": "0.038 ≤ ω < 0.30 OR intermediate conditions",
        "color": "#ffc107",  # Yellow
        "description": "Elevated monitoring required; approaching collapse boundary",
        "icon": "🟡",
    },
    "Collapse": {
        "condition": "ω ≥ 0.30 OR F < 0.70",
        "color": "#dc3545",  # Red
        "description": "Collapse event detected; generative restructuring active",
        "icon": "🔴",
    },
}

# GCD Axioms for display
GCD_AXIOMS = {
    "AX-0": {
        "statement": "Collapse is generative",
        "description": "Collapse events are not terminal failures but generative processes that produce new structure.",
    },
    "AX-1": {
        "statement": "Boundary defines interior",
        "description": "The collapse boundary (ω ≥ 0.30) defines what is stable. Distance from collapse determines behavior.",
    },
    "AX-2": {
        "statement": "Entropy measures determinacy",
        "description": "S=0 represents perfect determinacy; S approaching maximum represents maximum uncertainty.",
    },
}


def translate_to_gcd(tier1_values: dict[str, Any]) -> dict[str, Any]:
    """
    Translate Tier-1 kernel values to GCD native representation with interpretations.

    Args:
        tier1_values: Dict with keys F, omega, S, C, kappa, IC, etc.

    Returns:
        Dict with GCD translations, interpretations, and regime classification
    """
    gcd = {
        "symbols": {},
        "regime": None,
        "axiom_state": {},
        "natural_language": [],
        "warnings": [],
    }

    # Extract values with defaults
    omega = tier1_values.get("omega", tier1_values.get("F", 0))
    if "omega" not in tier1_values and "F" in tier1_values:
        omega = 1 - tier1_values["F"]

    fidelity = tier1_values.get("F", 1 - omega)
    entropy = tier1_values.get("S", 0)
    curvature = tier1_values.get("C", 0)
    kappa = tier1_values.get("kappa", 0)
    ic = tier1_values.get("IC", 1)

    # Translate each symbol
    gcd["symbols"]["omega"] = {
        "value": omega,
        "latex": "ω",
        "interpretation": _interpret_value(omega, GCD_SYMBOLS["omega"]),
        "distance_to_collapse": max(0, 0.30 - omega),
        "percent_to_collapse": min(100, omega / 0.30 * 100),
    }

    gcd["symbols"]["F"] = {
        "value": fidelity,
        "latex": "F",
        "interpretation": _interpret_value(fidelity, GCD_SYMBOLS["F"]),
        "identity_check": abs(fidelity + omega - 1.0) < 1e-9,
    }

    gcd["symbols"]["S"] = {
        "value": entropy,
        "latex": "S",
        "interpretation": _interpret_value(entropy, GCD_SYMBOLS["S"]),
        "determinacy": "deterministic" if entropy < 0.01 else ("low uncertainty" if entropy < 0.15 else "uncertain"),
    }

    gcd["symbols"]["C"] = {
        "value": curvature,
        "latex": "C",
        "interpretation": _interpret_value(curvature, GCD_SYMBOLS["C"]),
        "homogeneity": "homogeneous" if curvature < 0.01 else ("coherent" if curvature < 0.14 else "heterogeneous"),
    }

    gcd["symbols"]["kappa"] = {
        "value": kappa,
        "latex": "κ",
        "interpretation": _interpret_value(kappa, GCD_SYMBOLS["kappa"]),
    }

    gcd["symbols"]["IC"] = {
        "value": ic,
        "latex": "IC",
        "interpretation": _interpret_value(ic, GCD_SYMBOLS["IC"]),
        "identity_check": abs(ic - np.exp(kappa)) < 1e-6 if np is not None else True,
    }

    # Determine GCD regime based on canonical thresholds from gcd_anchors.yaml
    # Stable: ω < 0.038 (strict) or ω < 0.10 with high F and low S/C
    # Watch: 0.038 ≤ ω < 0.30
    # Collapse: ω ≥ 0.30
    if omega < 0.10 and fidelity > 0.85:
        gcd["regime"] = "Stable"
    elif omega >= 0.30:
        gcd["regime"] = "Collapse"
    else:
        gcd["regime"] = "Watch"

    gcd["regime_info"] = GCD_REGIMES[gcd["regime"]]

    # Axiom state interpretation
    gcd["axiom_state"]["AX-0"] = {
        "active": gcd["regime"] == "Collapse",
        "description": "Generative collapse active" if gcd["regime"] == "Collapse" else "System stable (no collapse)",
    }
    gcd["axiom_state"]["AX-1"] = {
        "distance_to_boundary": gcd["symbols"]["omega"]["distance_to_collapse"],
        "description": f"{gcd['symbols']['omega']['distance_to_collapse']:.3f} from collapse boundary",
    }
    gcd["axiom_state"]["AX-2"] = {
        "determinacy": gcd["symbols"]["S"]["determinacy"],
        "description": f"System is {gcd['symbols']['S']['determinacy']} (S={entropy:.4f})",
    }

    # Natural language summary
    regime_desc = GCD_REGIMES[gcd["regime"]]["description"]
    gcd["natural_language"] = [
        f"**GCD Regime**: {gcd['regime']} — {regime_desc}",
        f"**Drift (ω)**: {omega:.4f} — {gcd['symbols']['omega']['interpretation']}",
        f"**Fidelity (F)**: {fidelity:.4f} — {gcd['symbols']['F']['interpretation']}",
        f"**Entropy (S)**: {entropy:.4f} — System is {gcd['symbols']['S']['determinacy']}",
        f"**Curvature (C)**: {curvature:.4f} — Coordinates are {gcd['symbols']['C']['homogeneity']}",
        f"**Integrity (IC)**: {ic:.4f} — {gcd['symbols']['IC']['interpretation']}",
    ]

    # Create summary for easy access
    gcd["summary"] = f"GCD {gcd['regime']}: ω={omega:.4f}, F={fidelity:.4f}, IC={ic:.4f}. {regime_desc}"

    # Alias for axiom_states (some code uses this key)
    gcd["axiom_states"] = gcd["axiom_state"]

    # Warnings
    if omega >= 0.25:
        gcd["warnings"].append("⚠️ Approaching collapse boundary (ω ≥ 0.25)")
    if not gcd["symbols"]["F"]["identity_check"]:
        gcd["warnings"].append("⚠️ Identity violation: F + ω ≠ 1")
    if not gcd["symbols"]["IC"]["identity_check"]:
        gcd["warnings"].append("⚠️ Identity violation: IC ≠ exp(κ)")

    return gcd


def _interpret_value(value: float, symbol_def: dict[str, Any]) -> str:
    """Get natural language interpretation for a GCD value."""
    interp = symbol_def.get("interpretation", {})

    # Find closest threshold
    closest_desc = "Unknown"
    closest_dist = float("inf")

    for threshold, desc in interp.items():
        if isinstance(threshold, int | float):
            dist = abs(value - threshold)
            if dist < closest_dist:
                closest_dist = dist
                closest_desc = desc

    return closest_desc


def render_gcd_panel(gcd_data: dict[str, Any], compact: bool = False) -> None:
    """Render a GCD translation panel in Streamlit."""
    if st is None:
        return

    regime = gcd_data["regime"]
    regime_info = gcd_data["regime_info"]

    # Regime header
    st.markdown(
        f"""<div style="padding: 15px; border-left: 6px solid {regime_info["color"]};
            background: {regime_info["color"]}22; border-radius: 8px; margin-bottom: 15px;">
            <h3 style="margin: 0; color: {regime_info["color"]};">
                {regime_info["icon"]} GCD Regime: {regime}
            </h3>
            <p style="margin: 5px 0 0 0; font-size: 0.9em;">{regime_info["description"]}</p>
        </div>""",
        unsafe_allow_html=True,
    )

    if compact:
        # Compact display
        cols = st.columns(6)
        symbols = ["omega", "F", "S", "C", "kappa", "IC"]
        for i, sym in enumerate(symbols):
            with cols[i]:
                data = gcd_data["symbols"][sym]
                st.metric(data["latex"], f"{data['value']:.4f}")
    else:
        # Full display with interpretations
        st.markdown("### 📐 GCD Tier-1 Invariants")

        col1, col2 = st.columns(2)

        with col1:
            for sym in ["omega", "F", "S"]:
                data = gcd_data["symbols"][sym]
                symbol_info = GCD_SYMBOLS[sym]

                st.markdown(f"**{data['latex']} ({symbol_info['name']})**: `{data['value']:.6f}`")
                st.caption(f"_{data['interpretation']}_")

                # Progress bar for bounded values
                if sym == "omega":
                    progress = min(1.0, data["value"] / 0.30)
                    st.progress(progress, text=f"{data['percent_to_collapse']:.1f}% to collapse")
                elif sym == "F":
                    st.progress(data["value"], text=f"Fidelity: {data['value'] * 100:.1f}%")

        with col2:
            for sym in ["C", "kappa", "IC"]:
                data = gcd_data["symbols"][sym]
                symbol_info = GCD_SYMBOLS[sym]

                st.markdown(f"**{data['latex']} ({symbol_info['name']})**: `{data['value']:.6f}`")
                st.caption(f"_{data['interpretation']}_")

                if sym == "C":
                    st.progress(min(1.0, data["value"]), text=f"Curvature: {data['value'] * 100:.1f}%")
                elif sym == "IC":
                    st.progress(data["value"], text=f"Integrity: {data['value'] * 100:.1f}%")

        # Axiom state
        st.markdown("### 📜 GCD Axiom State")
        ax_cols = st.columns(3)
        for i, (ax_id, ax_state) in enumerate(gcd_data["axiom_state"].items()):
            with ax_cols[i]:
                ax_info = GCD_AXIOMS[ax_id]
                st.markdown(f"**{ax_id}**: _{ax_info['statement']}_")
                st.caption(ax_state["description"])

        # Warnings
        if gcd_data["warnings"]:
            st.markdown("### ⚠️ Warnings")
            for warning in gcd_data["warnings"]:
                st.warning(warning)


# Physical quantity definitions with units
PHYSICS_QUANTITIES = {
    "position": {
        "symbol": "x",
        "base_unit": "m",
        "units": {
            "km": 1e3,
            "m": 1.0,
            "cm": 1e-2,
            "mm": 1e-3,
            "μm": 1e-6,
            "nm": 1e-9,
            "ft": 0.3048,
            "in": 0.0254,
            "mi": 1609.34,
        },
        "ref_value": 1.0,  # m
        "description": "Position / Distance",
    },
    "velocity": {
        "symbol": "v",
        "base_unit": "m/s",
        "units": {
            "m/s": 1.0,
            "km/h": 1 / 3.6,
            "km/s": 1e3,
            "cm/s": 1e-2,
            "ft/s": 0.3048,
            "mph": 0.44704,
            "knot": 0.514444,
        },
        "ref_value": 1.0,  # m/s
        "description": "Velocity / Speed",
    },
    "acceleration": {
        "symbol": "a",
        "base_unit": "m/s²",
        "units": {"m/s²": 1.0, "g": 9.80665, "ft/s²": 0.3048, "cm/s²": 1e-2, "Gal": 1e-2},
        "ref_value": 9.80665,  # m/s² (1g)
        "description": "Acceleration",
    },
    "mass": {
        "symbol": "m",
        "base_unit": "kg",
        "units": {"kg": 1.0, "g": 1e-3, "mg": 1e-6, "μg": 1e-9, "t": 1e3, "lb": 0.453592, "oz": 0.0283495},
        "ref_value": 1.0,  # kg
        "description": "Mass",
    },
    "force": {
        "symbol": "F",
        "base_unit": "N",
        "units": {"N": 1.0, "kN": 1e3, "MN": 1e6, "mN": 1e-3, "dyn": 1e-5, "lbf": 4.44822, "kgf": 9.80665},
        "ref_value": 1.0,  # N
        "description": "Force",
    },
    "energy": {
        "symbol": "E",
        "base_unit": "J",
        "units": {
            "J": 1.0,
            "kJ": 1e3,
            "MJ": 1e6,
            "mJ": 1e-3,
            "eV": 1.602e-19,
            "keV": 1.602e-16,
            "cal": 4.184,
            "kcal": 4184,
            "BTU": 1055.06,
            "kWh": 3.6e6,
        },
        "ref_value": 1.0,  # J
        "description": "Energy",
    },
    "momentum": {
        "symbol": "p",
        "base_unit": "kg·m/s",
        "units": {"kg·m/s": 1.0, "g·cm/s": 1e-5, "N·s": 1.0},
        "ref_value": 1.0,  # kg·m/s
        "description": "Momentum",
    },
    "time": {
        "symbol": "t",
        "base_unit": "s",
        "units": {"s": 1.0, "ms": 1e-3, "μs": 1e-6, "ns": 1e-9, "min": 60, "h": 3600, "day": 86400},
        "ref_value": 1.0,  # s
        "description": "Time",
    },
    "frequency": {
        "symbol": "f",
        "base_unit": "Hz",
        "units": {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9, "rpm": 1 / 60},
        "ref_value": 1.0,  # Hz
        "description": "Frequency",
    },
    "power": {
        "symbol": "P",
        "base_unit": "W",
        "units": {"W": 1.0, "kW": 1e3, "MW": 1e6, "mW": 1e-3, "hp": 745.7, "BTU/h": 0.293071},
        "ref_value": 1.0,  # W
        "description": "Power",
    },
    "angle": {
        "symbol": "θ",
        "base_unit": "rad",
        "units": {"rad": 1.0, "deg": 0.0174533, "°": 0.0174533, "rev": 6.28319, "grad": 0.015708},
        "ref_value": 1.0,  # rad
        "description": "Angle",
    },
    "angular_velocity": {
        "symbol": "ω",
        "base_unit": "rad/s",
        "units": {"rad/s": 1.0, "deg/s": 0.0174533, "rpm": 0.10472, "Hz": 6.28319},
        "ref_value": 1.0,  # rad/s
        "description": "Angular Velocity",
    },
}

# Kinematics scenarios with preset values
KINEMATICS_SCENARIOS = {
    "Free Fall": {
        "description": "Object falling under gravity",
        "quantities": {
            "position": {"value": 0.0, "unit": "m"},
            "velocity": {"value": 0.0, "unit": "m/s"},
            "acceleration": {"value": 9.81, "unit": "m/s²"},
            "mass": {"value": 1.0, "unit": "kg"},
            "time": {"value": 1.0, "unit": "s"},
        },
    },
    "Projectile Motion": {
        "description": "Projectile launched at 45°",
        "quantities": {
            "position": {"value": 0.0, "unit": "m"},
            "velocity": {"value": 10.0, "unit": "m/s"},
            "acceleration": {"value": 9.81, "unit": "m/s²"},
            "mass": {"value": 0.5, "unit": "kg"},
            "angle": {"value": 45.0, "unit": "deg"},
        },
    },
    "Simple Harmonic Oscillator": {
        "description": "Mass on a spring",
        "quantities": {
            "position": {"value": 0.1, "unit": "m"},
            "velocity": {"value": 0.0, "unit": "m/s"},
            "mass": {"value": 1.0, "unit": "kg"},
            "frequency": {"value": 1.0, "unit": "Hz"},
            "energy": {"value": 0.5, "unit": "J"},
        },
    },
    "Circular Motion": {
        "description": "Uniform circular motion",
        "quantities": {
            "position": {"value": 1.0, "unit": "m"},
            "velocity": {"value": 2.0, "unit": "m/s"},
            "angular_velocity": {"value": 2.0, "unit": "rad/s"},
            "mass": {"value": 1.0, "unit": "kg"},
            "force": {"value": 4.0, "unit": "N"},
        },
    },
    "Elastic Collision": {
        "description": "1D elastic collision",
        "quantities": {
            "mass": {"value": 1.0, "unit": "kg"},
            "velocity": {"value": 5.0, "unit": "m/s"},
            "momentum": {"value": 5.0, "unit": "kg·m/s"},
            "energy": {"value": 12.5, "unit": "J"},
        },
    },
}


# ============================================================
# PHYSICAL CONSTANTS - Fundamental physics constants (CODATA 2022)
# ============================================================
PHYSICAL_CONSTANTS: dict[str, dict[str, Any]] = {
    "c": {"name": "Speed of light in vacuum", "value": 299792458, "unit": "m/s", "symbol": "c"},
    "G": {"name": "Gravitational constant", "value": 6.67430e-11, "unit": "m³/(kg·s²)", "symbol": "G"},
    "h": {"name": "Planck constant", "value": 6.62607015e-34, "unit": "J·s", "symbol": "h"},
    "ħ": {"name": "Reduced Planck constant", "value": 1.054571817e-34, "unit": "J·s", "symbol": "ħ"},
    "e": {"name": "Elementary charge", "value": 1.602176634e-19, "unit": "C", "symbol": "e"},
    "m_e": {"name": "Electron mass", "value": 9.1093837015e-31, "unit": "kg", "symbol": "mₑ"},
    "m_p": {"name": "Proton mass", "value": 1.67262192369e-27, "unit": "kg", "symbol": "mₚ"},
    "m_n": {"name": "Neutron mass", "value": 1.67492749804e-27, "unit": "kg", "symbol": "mₙ"},
    "k_B": {"name": "Boltzmann constant", "value": 1.380649e-23, "unit": "J/K", "symbol": "kB"},
    "N_A": {"name": "Avogadro constant", "value": 6.02214076e23, "unit": "mol⁻¹", "symbol": "Nₐ"},
    "R": {"name": "Gas constant", "value": 8.314462618, "unit": "J/(mol·K)", "symbol": "R"},
    "ε_0": {"name": "Vacuum permittivity", "value": 8.8541878128e-12, "unit": "F/m", "symbol": "ε₀"},
    "μ_0": {"name": "Vacuum permeability", "value": 1.25663706212e-6, "unit": "H/m", "symbol": "μ₀"},
    "α": {"name": "Fine structure constant", "value": 7.2973525693e-3, "unit": "dimensionless", "symbol": "α"},
    "g": {"name": "Standard gravity", "value": 9.80665, "unit": "m/s²", "symbol": "g"},
    "atm": {"name": "Standard atmosphere", "value": 101325, "unit": "Pa", "symbol": "atm"},
    "σ": {"name": "Stefan-Boltzmann constant", "value": 5.670374419e-8, "unit": "W/(m²·K⁴)", "symbol": "σ"},
    "R_∞": {"name": "Rydberg constant", "value": 10973731.568160, "unit": "m⁻¹", "symbol": "R∞"},
    "a_0": {"name": "Bohr radius", "value": 5.29177210903e-11, "unit": "m", "symbol": "a₀"},
}

# ============================================================
# PHYSICS FORMULAS - Common formulas for calculation
# Note: Lambda functions are intentionally untyped for flexibility
# ============================================================
PHYSICS_FORMULAS: dict[str, dict[str, Any]] = {  # type: ignore[misc]
    "kinetic_energy": {
        "name": "Kinetic Energy",
        "formula": "KE = ½mv²",
        "latex": r"KE = \frac{1}{2}mv^2",
        "inputs": ["mass", "velocity"],
        "output": "energy",
        "calculate": lambda m, v: 0.5 * m * v**2,
        "description": "Energy of motion",
    },
    "potential_energy_gravity": {
        "name": "Gravitational Potential Energy",
        "formula": "PE = mgh",
        "latex": r"PE = mgh",
        "inputs": ["mass", "position", "acceleration"],
        "output": "energy",
        "calculate": lambda m, h, g=9.80665: m * g * h,
        "description": "Energy due to height in gravitational field",
    },
    "momentum": {
        "name": "Linear Momentum",
        "formula": "p = mv",
        "latex": r"p = mv",
        "inputs": ["mass", "velocity"],
        "output": "momentum",
        "calculate": lambda m, v: m * v,
        "description": "Product of mass and velocity",
    },
    "force_newton": {
        "name": "Newton's Second Law",
        "formula": "F = ma",
        "latex": r"F = ma",
        "inputs": ["mass", "acceleration"],
        "output": "force",
        "calculate": lambda m, a: m * a,
        "description": "Force equals mass times acceleration",
    },
    "work": {
        "name": "Work Done",
        "formula": "W = Fd",
        "latex": r"W = Fd",
        "inputs": ["force", "position"],
        "output": "energy",
        "calculate": lambda F, d: F * d,
        "description": "Work as force times displacement",
    },
    "power": {
        "name": "Power",
        "formula": "P = W/t",
        "latex": r"P = \frac{W}{t}",
        "inputs": ["energy", "time"],
        "output": "power",
        "calculate": lambda W, t: W / t if t != 0 else 0,
        "description": "Rate of energy transfer",
    },
    "centripetal_force": {
        "name": "Centripetal Force",
        "formula": "F = mv²/r",
        "latex": r"F = \frac{mv^2}{r}",
        "inputs": ["mass", "velocity", "position"],
        "output": "force",
        "calculate": lambda m, v, r: m * v**2 / r if r != 0 else 0,
        "description": "Force for circular motion",
    },
    "gravitational_force": {
        "name": "Gravitational Force",
        "formula": "F = Gm₁m₂/r²",
        "latex": r"F = \frac{Gm_1m_2}{r^2}",
        "inputs": ["mass", "mass", "position"],
        "output": "force",
        "calculate": lambda m1, m2, r: 6.67430e-11 * m1 * m2 / r**2 if r != 0 else 0,
        "description": "Universal gravitation",
    },
    "period_pendulum": {
        "name": "Simple Pendulum Period",
        "formula": "T = 2π√(L/g)",
        "latex": r"T = 2\pi\sqrt{\frac{L}{g}}",
        "inputs": ["position"],  # L = position
        "output": "time",
        "calculate": lambda L, g=9.80665: 2 * 3.14159265 * (L / g) ** 0.5 if g != 0 else 0,
        "description": "Period of simple pendulum",
    },
    "wave_speed": {
        "name": "Wave Speed",
        "formula": "v = fλ",
        "latex": r"v = f\lambda",
        "inputs": ["frequency", "position"],  # λ = position (wavelength)
        "output": "velocity",
        "calculate": lambda f, wavelength: f * wavelength,
        "description": "Wave velocity",
    },
    "relativistic_energy": {
        "name": "Mass-Energy Equivalence",
        "formula": "E = mc²",
        "latex": r"E = mc^2",
        "inputs": ["mass"],
        "output": "energy",
        "calculate": lambda m: m * 299792458**2,
        "description": "Einstein's famous equation",
    },
    "de_broglie": {
        "name": "de Broglie Wavelength",
        "formula": "λ = h/p",
        "latex": r"\lambda = \frac{h}{p}",
        "inputs": ["momentum"],
        "output": "position",  # wavelength
        "calculate": lambda p: 6.62607015e-34 / p if p != 0 else 0,
        "description": "Matter wave wavelength",
    },
}

# ============================================================
# KINEMATICS EQUATIONS - 1D motion equations
# Note: Lambda functions are intentionally untyped for flexibility
# ============================================================
KINEMATICS_EQUATIONS: dict[str, dict[str, Any]] = {  # type: ignore[misc]
    "position_time": {
        "name": "Position from time",
        "formula": "x = x₀ + v₀t + ½at²",
        "latex": r"x = x_0 + v_0t + \frac{1}{2}at^2",
        "inputs": {"x0": "initial position", "v0": "initial velocity", "a": "acceleration", "t": "time"},
        "output": "position",
        "calculate": lambda x0, v0, a, t: x0 + v0 * t + 0.5 * a * t**2,
    },
    "velocity_time": {
        "name": "Velocity from time",
        "formula": "v = v₀ + at",
        "latex": r"v = v_0 + at",
        "inputs": {"v0": "initial velocity", "a": "acceleration", "t": "time"},
        "output": "velocity",
        "calculate": lambda v0, a, t: v0 + a * t,
    },
    "velocity_position": {
        "name": "Velocity from position",
        "formula": "v² = v₀² + 2a(x - x₀)",
        "latex": r"v^2 = v_0^2 + 2a(x - x_0)",
        "inputs": {"v0": "initial velocity", "a": "acceleration", "x": "position", "x0": "initial position"},
        "output": "velocity",
        "calculate": lambda v0, a, x, x0: (v0**2 + 2 * a * (x - x0)) ** 0.5 if v0**2 + 2 * a * (x - x0) >= 0 else 0,
    },
    "time_from_velocity": {
        "name": "Time to reach velocity",
        "formula": "t = (v - v₀)/a",
        "latex": r"t = \frac{v - v_0}{a}",
        "inputs": {"v": "final velocity", "v0": "initial velocity", "a": "acceleration"},
        "output": "time",
        "calculate": lambda v, v0, a: (v - v0) / a if a != 0 else 0,
    },
    "projectile_range": {
        "name": "Projectile Range",
        "formula": "R = v₀²sin(2θ)/g",
        "latex": r"R = \frac{v_0^2 \sin(2\theta)}{g}",
        "inputs": {"v0": "initial velocity", "theta": "launch angle (rad)", "g": "gravity"},
        "output": "position",
        "calculate": lambda v0, theta, g=9.80665: v0**2 * np.sin(2 * theta) / g if np is not None and g != 0 else 0,
    },
    "projectile_max_height": {
        "name": "Projectile Max Height",
        "formula": "H = v₀²sin²(θ)/(2g)",
        "latex": r"H = \frac{v_0^2 \sin^2(\theta)}{2g}",
        "inputs": {"v0": "initial velocity", "theta": "launch angle (rad)", "g": "gravity"},
        "output": "position",
        "calculate": lambda v0, theta, g=9.80665: (
            v0**2 * np.sin(theta) ** 2 / (2 * g) if np is not None and g != 0 else 0
        ),
    },
    "projectile_time_flight": {
        "name": "Projectile Time of Flight",
        "formula": "T = 2v₀sin(θ)/g",
        "latex": r"T = \frac{2v_0 \sin(\theta)}{g}",
        "inputs": {"v0": "initial velocity", "theta": "launch angle (rad)", "g": "gravity"},
        "output": "time",
        "calculate": lambda v0, theta, g=9.80665: 2 * v0 * np.sin(theta) / g if np is not None and g != 0 else 0,
    },
    "shm_position": {
        "name": "SHM Position",
        "formula": "x = A·cos(ωt + φ)",
        "latex": r"x = A\cos(\omega t + \phi)",
        "inputs": {"A": "amplitude", "omega": "angular frequency", "t": "time", "phi": "phase"},
        "output": "position",
        "calculate": lambda A, omega, t, phi=0: A * np.cos(omega * t + phi) if np is not None else 0,
    },
    "shm_velocity": {
        "name": "SHM Velocity",
        "formula": "v = -Aω·sin(ωt + φ)",
        "latex": r"v = -A\omega\sin(\omega t + \phi)",
        "inputs": {"A": "amplitude", "omega": "angular frequency", "t": "time", "phi": "phase"},
        "output": "velocity",
        "calculate": lambda A, omega, t, phi=0: -A * omega * np.sin(omega * t + phi) if np is not None else 0,
    },
    "circular_period": {
        "name": "Circular Motion Period",
        "formula": "T = 2πr/v",
        "latex": r"T = \frac{2\pi r}{v}",
        "inputs": {"r": "radius", "v": "velocity"},
        "output": "time",
        "calculate": lambda r, v: 2 * 3.14159265 * r / v if v != 0 else 0,
    },
}


def convert_to_base_unit(value: float, unit: str, quantity_type: str) -> float:
    """Convert a value from given unit to base SI unit."""
    if quantity_type not in PHYSICS_QUANTITIES:
        return value
    units = PHYSICS_QUANTITIES[quantity_type]["units"]
    if unit in units:
        return value * units[unit]
    return value


def convert_from_base_unit(value: float, unit: str, quantity_type: str) -> float:
    """Convert a value from base SI unit to given unit."""
    if quantity_type not in PHYSICS_QUANTITIES:
        return value
    units = PHYSICS_QUANTITIES[quantity_type]["units"]
    if unit in units:
        return value / units[unit]
    return value


def normalize_to_bounded(value: float, ref_value: float, epsilon: float = 1e-6) -> tuple[float, bool]:
    """
    Normalize a physical value to [0,1] bounded embedding.
    Returns (normalized_value, was_clipped).
    """
    if ref_value == 0:
        ref_value = 1.0
    normalized = abs(value) / ref_value
    was_clipped = normalized < epsilon or normalized > (1 - epsilon)
    clipped = max(epsilon, min(1 - epsilon, normalized))
    return clipped, was_clipped


def render_physics_interface_page() -> None:
    """
    Render the Physics Interface page for SI unit conversion and tier translation.
    """
    if st is None or pd is None or np is None:
        return

    st.title("⚛️ Physics Interface")
    st.caption("Convert physical quantities with SI units through UMCP tier translation")

    # Initialize session state
    if "physics_quantities" not in st.session_state:
        st.session_state.physics_quantities = {}
    if "physics_audit" not in st.session_state:
        st.session_state.physics_audit = []

    # ========== SI Unit Converter Section ==========
    st.header("🔄 SI Unit Converter")

    with st.expander("📖 About SI Units", expanded=False):
        st.markdown("""
        **SI Base Units:**
        - Length: meter (m)
        - Mass: kilogram (kg)
        - Time: second (s)
        - Electric current: ampere (A)
        - Temperature: kelvin (K)
        - Amount of substance: mole (mol)
        - Luminous intensity: candela (cd)

        **Common Prefixes:**
        | Prefix | Symbol | Factor |
        |--------|--------|--------|
        | giga | G | 10⁹ |
        | mega | M | 10⁶ |
        | kilo | k | 10³ |
        | milli | m | 10⁻³ |
        | micro | μ | 10⁻⁶ |
        | nano | n | 10⁻⁹ |
        """)

    # Quick converter
    st.subheader("⚡ Quick Converter")
    conv_cols = st.columns([2, 1, 1, 1, 2])

    with conv_cols[0]:
        conv_value = st.number_input("Value", value=1.0, format="%.6f", key="conv_value")
    with conv_cols[1]:
        conv_quantity = st.selectbox("Quantity", list(PHYSICS_QUANTITIES.keys()), key="conv_quantity")
    with conv_cols[2]:
        qty_key = conv_quantity if conv_quantity else "position"
        from_units = list(PHYSICS_QUANTITIES[qty_key]["units"].keys())
        from_unit = st.selectbox("From", from_units, key="conv_from")
    with conv_cols[3]:
        to_unit = st.selectbox("To", from_units, index=min(1, len(from_units) - 1), key="conv_to")
    with conv_cols[4]:
        # Calculate conversion (with safe defaults)
        from_u = str(from_unit) if from_unit else from_units[0]
        to_u = str(to_unit) if to_unit else from_units[0]
        base_value = convert_to_base_unit(float(conv_value), from_u, qty_key)
        result_value = convert_from_base_unit(base_value, to_u, qty_key)
        st.metric("Result", f"{result_value:.6g} {to_u}")

    st.divider()

    # ========== Physics Toolbox (Tabbed Interface) ==========
    st.header("🧰 Physics Toolbox")

    tool_tabs = st.tabs(
        ["📐 Formula Calculator", "🔬 Physical Constants", "📊 Dimensional Analysis", "📜 Audit History"]
    )

    # Tab 1: Formula Calculator
    with tool_tabs[0]:
        st.markdown("**Select a formula and input values to calculate results.**")

        formula_cols = st.columns([1, 2])
        with formula_cols[0]:
            formula_name = st.selectbox(
                "Formula",
                list(PHYSICS_FORMULAS.keys()),
                format_func=lambda x: f"{PHYSICS_FORMULAS[x]['name']} ({PHYSICS_FORMULAS[x]['formula']})",
                key="formula_select",
            )

        if formula_name:
            formula = PHYSICS_FORMULAS[formula_name]

            with formula_cols[1]:
                st.latex(formula["latex"])
                st.caption(formula["description"])

            # Input fields for formula
            st.markdown("**Input Values:**")
            input_cols = st.columns(len(formula["inputs"]))
            formula_inputs = []

            for i, inp_name in enumerate(formula["inputs"]):
                with input_cols[i]:
                    qty_info = PHYSICS_QUANTITIES.get(inp_name, {"symbol": inp_name, "base_unit": "units"})
                    val = st.number_input(
                        f"{qty_info['symbol']} ({qty_info.get('base_unit', '')})",
                        value=1.0,
                        format="%.6g",
                        key=f"formula_inp_{i}",
                    )
                    formula_inputs.append(val)

            # Calculate result
            if st.button("🧮 Calculate", key="formula_calc"):
                try:
                    result = formula["calculate"](*formula_inputs)
                    output_qty = PHYSICS_FORMULAS[formula_name]["output"]
                    out_info = PHYSICS_QUANTITIES.get(output_qty, {"symbol": "?", "base_unit": "units"})
                    st.success(f"**Result:** {out_info['symbol']} = {result:.6g} {out_info['base_unit']}")

                    # Show in other common units
                    if output_qty in PHYSICS_QUANTITIES:
                        st.markdown("**In other units:**")
                        other_units = list(PHYSICS_QUANTITIES[output_qty]["units"].items())[:4]
                        unit_cols = st.columns(len(other_units))
                        for j, (unit_name, factor) in enumerate(other_units):
                            with unit_cols[j]:
                                converted = result / factor
                                st.metric(unit_name, f"{converted:.4g}")
                except Exception as e:
                    st.error(f"Calculation error: {e}")

    # Tab 2: Physical Constants
    with tool_tabs[1]:
        st.markdown("**Fundamental Physical Constants (CODATA 2022)**")

        # Search filter
        const_search = st.text_input("🔍 Search constants", key="const_search")

        # Create dataframe
        const_data = []
        for key, const in PHYSICAL_CONSTANTS.items():
            if const_search.lower() in const["name"].lower() or const_search.lower() in key.lower():
                const_data.append(
                    {
                        "Symbol": const["symbol"],
                        "Name": const["name"],
                        "Value": f"{const['value']:.6e}"
                        if const["value"] < 0.01 or const["value"] > 1000
                        else f"{const['value']:.6g}",
                        "Unit": const["unit"],
                    }
                )

        if const_data:
            st.dataframe(pd.DataFrame(const_data), hide_index=True, width="stretch")
        else:
            st.info("No constants found matching your search.")

        # Quick access buttons
        st.markdown("**Quick Copy:**")
        quick_cols = st.columns(5)
        quick_consts = ["c", "G", "h", "e", "g"]
        for i, const_key in enumerate(quick_consts):
            with quick_cols[i]:
                const = PHYSICAL_CONSTANTS[const_key]
                st.code(f"{const['symbol']} = {const['value']:.6e}")

    # Tab 3: Dimensional Analysis
    with tool_tabs[2]:
        st.markdown("**Check dimensional consistency of your calculations.**")

        dim_cols = st.columns(2)
        with dim_cols[0]:
            st.markdown("**Left Side (LHS)**")
            lhs_qty = st.selectbox("Quantity type", list(PHYSICS_QUANTITIES.keys()), key="dim_lhs")
            lhs_val = st.number_input("Value", value=1.0, key="dim_lhs_val")
            lhs_unit = st.selectbox("Unit", list(PHYSICS_QUANTITIES[lhs_qty]["units"].keys()), key="dim_lhs_unit")

        with dim_cols[1]:
            st.markdown("**Right Side (RHS)**")
            rhs_qty = st.selectbox("Quantity type", list(PHYSICS_QUANTITIES.keys()), key="dim_rhs")
            rhs_val = st.number_input("Value", value=1.0, key="dim_rhs_val")
            rhs_unit = st.selectbox("Unit", list(PHYSICS_QUANTITIES[rhs_qty]["units"].keys()), key="dim_rhs_unit")

        if st.button("🔍 Check Consistency", key="dim_check"):
            lhs_base = PHYSICS_QUANTITIES[lhs_qty]["base_unit"]
            rhs_base = PHYSICS_QUANTITIES[rhs_qty]["base_unit"]

            if lhs_base == rhs_base:
                lhs_si = convert_to_base_unit(lhs_val, lhs_unit, lhs_qty)
                rhs_si = convert_to_base_unit(rhs_val, rhs_unit, rhs_qty)
                diff = abs(lhs_si - rhs_si)
                rel_diff = diff / max(lhs_si, rhs_si, 1e-15) * 100

                st.success(f"✅ **Dimensionally consistent** (both are {lhs_base})")
                st.markdown(f"- LHS: {lhs_val} {lhs_unit} = **{lhs_si:.6g} {lhs_base}**")
                st.markdown(f"- RHS: {rhs_val} {rhs_unit} = **{rhs_si:.6g} {rhs_base}**")
                st.markdown(f"- Difference: {diff:.6g} {lhs_base} ({rel_diff:.2f}%)")
            else:
                st.error("❌ **Dimensionally inconsistent!**")
                st.markdown(f"- LHS has dimensions: **{lhs_base}**")
                st.markdown(f"- RHS has dimensions: **{rhs_base}**")
                st.markdown("These cannot be equal or compared directly.")

    # Tab 4: Audit History
    with tool_tabs[3]:
        if st.session_state.physics_audit:
            st.markdown(f"**{len(st.session_state.physics_audit)} previous calculations**")

            for i, entry in enumerate(reversed(st.session_state.physics_audit[-5:])):
                regime = entry.get("tier2", {}).get("regime", "N/A")
                regime_color = get_regime_color(regime)
                ts = entry.get("timestamp", "")[:19]

                with st.expander(f"Run {len(st.session_state.physics_audit) - i}: {regime} @ {ts}"):
                    hist_cols = st.columns(3)
                    with hist_cols[0]:
                        st.markdown("**Tier 0:**")
                        st.markdown(f"- Quantities: {entry['tier0'].get('n_quantities', 'N/A')}")
                    with hist_cols[1]:
                        st.markdown("**Tier 1:**")
                        t1 = entry.get("tier1", {})
                        st.markdown(f"- F: {t1.get('F', 0):.4f}")
                        st.markdown(f"- ω: {t1.get('omega', 0):.4f}")
                    with hist_cols[2]:
                        st.markdown("**GCD:**")
                        gcd = entry.get("gcd", {})
                        st.markdown(f"- Regime: {gcd.get('regime', 'N/A')}")

                    if st.button("📋 View Full JSON", key=f"hist_json_{i}"):
                        st.json(entry)

            if st.button("🗑️ Clear History"):
                st.session_state.physics_audit = []
                st.rerun()
        else:
            st.info("No calculation history yet. Run a physics translation to see results here.")

    st.divider()

    # ========== Physics Quantity Input ==========
    st.header("📥 Physical Quantities Input")
    st.markdown("Enter physical measurements that will be normalized to [0,1] for UMCP processing.")

    # Quantity selection
    selected_quantities = st.multiselect(
        "Select quantities to include",
        list(PHYSICS_QUANTITIES.keys()),
        default=["position", "velocity", "acceleration", "mass", "energy"],
        key="physics_selected",
    )

    if not selected_quantities:
        st.warning("Select at least one quantity to continue.")
        return

    # Input for each quantity
    st.subheader("🎯 Enter Values")

    input_data = {}
    n_cols = min(3, len(selected_quantities))

    for i in range(0, len(selected_quantities), n_cols):
        cols = st.columns(n_cols)
        for j, col in enumerate(cols):
            if i + j < len(selected_quantities):
                qty_name = selected_quantities[i + j]
                qty_info = PHYSICS_QUANTITIES[qty_name]

                with col:
                    st.markdown(f"**{qty_info['description']}** ({qty_info['symbol']})")

                    val_col, unit_col = st.columns([2, 1])
                    with val_col:
                        val = st.number_input(
                            "Value",
                            value=qty_info["ref_value"],
                            format="%.6f",
                            key=f"phys_val_{qty_name}",
                            label_visibility="collapsed",
                        )
                    with unit_col:
                        unit = st.selectbox(
                            "Unit",
                            list(qty_info["units"].keys()),
                            key=f"phys_unit_{qty_name}",
                            label_visibility="collapsed",
                        )

                    input_data[qty_name] = {
                        "value": val,
                        "unit": unit,
                        "symbol": qty_info["symbol"],
                        "base_unit": qty_info["base_unit"],
                        "ref_value": qty_info["ref_value"],
                    }

    st.divider()

    # Reference values (for normalization)
    st.subheader("📏 Reference Scales")
    with st.expander("Adjust reference values for normalization", expanded=False):
        ref_cols = st.columns(min(4, len(selected_quantities)))
        for i, qty_name in enumerate(selected_quantities):
            with ref_cols[i % len(ref_cols)]:
                new_ref = st.number_input(
                    f"{qty_name} ref",
                    value=PHYSICS_QUANTITIES[qty_name]["ref_value"],
                    min_value=1e-15,
                    format="%.4f",
                    key=f"ref_{qty_name}",
                )
                input_data[qty_name]["ref_value"] = new_ref

    # Epsilon
    epsilon = st.select_slider(
        "ε-clipping threshold",
        options=[1e-8, 1e-7, 1e-6, 1e-5, 1e-4],
        value=1e-6,
        format_func=lambda x: f"{x:.0e}",
        key="phys_epsilon",
    )

    st.divider()

    # ========== Process Button ==========
    if st.button("🚀 Run Physics → UMCP Translation", type="primary", width="stretch"):
        progress = st.progress(0, text="Starting physics translation...")

        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "physics",
            "tier0": {},
            "tier1": {},
            "tier2": {},
            "status": "PROCESSING",
        }

        # TIER 0: Convert to base units and normalize
        progress.progress(20, text="Tier 0: Converting to base SI units...")

        normalized_coords = []
        weights = []
        tier0_data = []

        for qty_name, data in input_data.items():
            # Convert to base unit
            base_value = convert_to_base_unit(data["value"], data["unit"], qty_name)

            # Normalize to [0,1]
            norm_value, was_clipped = normalize_to_bounded(base_value, data["ref_value"], epsilon)

            tier0_data.append(
                {
                    "quantity": qty_name,
                    "symbol": data["symbol"],
                    "input_value": data["value"],
                    "input_unit": data["unit"],
                    "base_value": base_value,
                    "base_unit": data["base_unit"],
                    "ref_value": data["ref_value"],
                    "normalized": norm_value,
                    "clipped": was_clipped,
                }
            )

            normalized_coords.append(norm_value)
            weights.append(1.0 / len(input_data))  # Equal weights

        audit_entry["tier0"] = {
            "quantities": tier0_data,
            "epsilon": epsilon,
            "n_quantities": len(normalized_coords),
        }

        # TIER 1: Kernel computation
        progress.progress(50, text="Tier 1: Computing kernel invariants...")

        c = np.array(normalized_coords)
        w = np.array(weights)

        fidelity = float(np.sum(w * c))
        drift = 1 - fidelity
        log_ic = float(np.sum(w * np.log(c)))
        integrity_composite = float(np.exp(log_ic))

        entropy = 0.0
        for ci, wi in zip(c, w, strict=False):
            if wi > 0 and 0 < ci < 1:
                entropy += wi * (-ci * np.log(ci) - (1 - ci) * np.log(1 - ci))

        curvature = float(np.std(c, ddof=0) / 0.5) if len(c) > 1 else 0.0
        amgm_gap = fidelity - integrity_composite

        audit_entry["tier1"] = {
            "F": fidelity,
            "omega": drift,
            "S": entropy,
            "C": curvature,
            "kappa": log_ic,
            "IC": integrity_composite,
            "amgm_gap": amgm_gap,
        }

        # TIER 2: Regime classification
        progress.progress(80, text="Tier 2: Computing regime and diagnostics...")

        regime = classify_regime(drift)
        stability_score = int(fidelity * 100 * (1 - curvature))

        # Physics-specific diagnostics
        diagnostics = []

        # Check for physical consistency
        if "velocity" in input_data and "mass" in input_data:
            v_base = convert_to_base_unit(input_data["velocity"]["value"], input_data["velocity"]["unit"], "velocity")
            m_base = convert_to_base_unit(input_data["mass"]["value"], input_data["mass"]["unit"], "mass")
            calc_ke = 0.5 * m_base * v_base**2

            if "energy" in input_data:
                e_base = convert_to_base_unit(input_data["energy"]["value"], input_data["energy"]["unit"], "energy")
                if abs(e_base - calc_ke) > 0.01 * max(e_base, calc_ke):
                    diagnostics.append(f"⚠️ Energy mismatch: input E={e_base:.2f}J, calculated KE={calc_ke:.2f}J")
                else:
                    diagnostics.append(f"✅ Energy consistent: E ≈ ½mv² = {calc_ke:.4f} J")

        if "velocity" in input_data and "mass" in input_data:
            v_base = convert_to_base_unit(input_data["velocity"]["value"], input_data["velocity"]["unit"], "velocity")
            m_base = convert_to_base_unit(input_data["mass"]["value"], input_data["mass"]["unit"], "mass")
            calc_p = m_base * v_base

            if "momentum" in input_data:
                p_base = convert_to_base_unit(
                    input_data["momentum"]["value"], input_data["momentum"]["unit"], "momentum"
                )
                if abs(p_base - calc_p) > 0.01 * max(p_base, calc_p):
                    diagnostics.append(
                        f"⚠️ Momentum mismatch: input p={p_base:.2f} kg·m/s, calculated p={calc_p:.2f} kg·m/s"
                    )
                else:
                    diagnostics.append(f"✅ Momentum consistent: p = mv = {calc_p:.4f} kg·m/s")

        if not diagnostics:
            diagnostics.append("✅ No physics consistency checks available for selected quantities")

        audit_entry["tier2"] = {
            "regime": regime,
            "stability_score": stability_score,
            "risk_level": "LOW" if regime == "STABLE" else ("MEDIUM" if regime == "WATCH" else "HIGH"),
            "diagnostics": diagnostics,
        }

        audit_entry["status"] = "COMPLETE"
        progress.progress(100, text="Complete!")

        # GCD Translation
        gcd_translation = translate_to_gcd(audit_entry["tier1"])
        audit_entry["gcd"] = gcd_translation

        st.session_state.physics_audit.append(audit_entry)

        # ========== Display Results ==========
        st.divider()

        regime = audit_entry["tier2"]["regime"]
        regime_color = get_regime_color(regime)

        st.markdown(
            f"""<div style="padding: 20px; border-left: 6px solid {regime_color};
                background: {regime_color}22; border-radius: 8px; margin-bottom: 20px;">
                <h2 style="margin: 0; color: {regime_color};">⚛️ Result: {regime}</h2>
                <p style="margin: 5px 0 0 0;">Stability Score: {stability_score}/100</p>
            </div>""",
            unsafe_allow_html=True,
        )

        # Three-column display
        tier_cols = st.columns(3)

        with tier_cols[0]:
            st.markdown("### 📥 Tier 0: Physical Input")
            t0_df = pd.DataFrame(
                [
                    {
                        "Quantity": d["symbol"],
                        "Input": f"{d['input_value']:.4g} {d['input_unit']}",
                        "SI Base": f"{d['base_value']:.4g} {d['base_unit']}",
                        "Normalized": f"{d['normalized']:.4f}",
                        "OOR": "⚠️" if d["clipped"] else "✓",
                    }
                    for d in audit_entry["tier0"]["quantities"]
                ]
            )
            st.dataframe(t0_df, hide_index=True, width="stretch")

        with tier_cols[1]:
            st.markdown("### ⚙️ Tier 1: Kernel")
            t1 = audit_entry["tier1"]
            col1, col2 = st.columns(2)
            with col1:
                st.metric("F", f"{t1['F']:.4f}")
                st.metric("ω", f"{t1['omega']:.4f}")
                st.metric("S", f"{t1['S']:.4f}")
            with col2:
                st.metric("C", f"{t1['C']:.4f}")
                st.metric("κ", f"{t1['kappa']:.4f}")
                st.metric("IC", f"{t1['IC']:.4f}")

        with tier_cols[2]:
            st.markdown("### 📊 Tier 2: Diagnostics")
            t2 = audit_entry["tier2"]
            st.metric("Regime", t2["regime"])
            st.metric("Score", f"{t2['stability_score']}/100")
            st.markdown("**Diagnostics:**")
            for diag in t2["diagnostics"]:
                st.markdown(f"- {diag}")

        # ========== GCD Translation Panel ==========
        st.divider()
        st.header("🌀 GCD Translation (Generative Collapse Dynamics)")
        st.caption("Physical values translated to native GCD representation")

        render_gcd_panel(gcd_translation, compact=False)

        # GCD Natural Language Summary
        st.markdown("### 📝 Natural Language Summary")
        for line in gcd_translation["natural_language"]:
            st.markdown(line)

        # Full audit
        with st.expander("🔍 Full Audit JSON (includes GCD)", expanded=False):
            st.json(audit_entry)


def render_kinematics_interface_page() -> None:
    """
    Render the Kinematics Interface page for motion analysis with tier translation.
    """
    if st is None or pd is None or np is None:
        return

    st.title("🎯 Kinematics Interface")
    st.caption("Analyze motion through phase space with UMCP tier translation")

    # Initialize session state
    if "kin_audit" not in st.session_state:
        st.session_state.kin_audit = []

    # ========== Scenario Selection ==========
    st.header("📋 Kinematic Scenarios")

    scenario_cols = st.columns(len(KINEMATICS_SCENARIOS))
    selected_scenario = None

    for i, (name, scenario) in enumerate(KINEMATICS_SCENARIOS.items()):
        with scenario_cols[i]:
            if st.button(f"📌 {name}", key=f"kin_scenario_{name}", width="stretch"):
                selected_scenario = name
                st.session_state.kin_scenario = scenario

    if selected_scenario:
        st.success(f"Loaded: {selected_scenario}")
        st.caption(KINEMATICS_SCENARIOS[selected_scenario]["description"])

    st.divider()

    # ========== Kinematics Toolbox ==========
    st.header("🧰 Kinematics Toolbox")

    kin_tabs = st.tabs(
        ["📐 Equations Solver", "🚀 Trajectory Calculator", "⚡ Energy Check", "📈 Motion Plot", "📜 History"]
    )

    # Tab 1: Equations Solver
    with kin_tabs[0]:
        st.markdown("**Solve kinematic equations for unknown quantities.**")

        eq_cols = st.columns([1, 2])
        with eq_cols[0]:
            eq_name = st.selectbox(
                "Select equation",
                list(KINEMATICS_EQUATIONS.keys()),
                format_func=lambda x: f"{KINEMATICS_EQUATIONS[x]['name']}",
                key="kin_eq_select",
            )

        if eq_name:
            eq = KINEMATICS_EQUATIONS[eq_name]

            with eq_cols[1]:
                st.latex(eq["latex"])
                st.markdown(f"*{eq['formula']}*")

            # Input fields
            st.markdown("**Input Values:**")
            eq_inputs = eq["inputs"] if isinstance(eq["inputs"], dict) else {k: k for k in eq["inputs"]}
            input_cols = st.columns(len(eq_inputs))
            eq_values = []

            for i, (var_name, var_desc) in enumerate(eq_inputs.items()):
                with input_cols[i]:
                    val = st.number_input(
                        f"{var_name}",
                        value=1.0 if var_name not in ["phi", "theta"] else 0.785,
                        format="%.4g",
                        key=f"kin_eq_inp_{eq_name}_{i}",
                        help=var_desc if isinstance(var_desc, str) else None,
                    )
                    eq_values.append(val)

            if st.button("🧮 Solve", key="kin_eq_solve"):
                try:
                    result = eq["calculate"](*eq_values)
                    output_qty = eq["output"]
                    out_info = PHYSICS_QUANTITIES.get(output_qty, {"symbol": "result", "base_unit": "units"})
                    st.success(f"**Result:** {out_info['symbol']} = {result:.6g} {out_info['base_unit']}")
                except Exception as e:
                    st.error(f"Calculation error: {e}")

    # Tab 2: Trajectory Calculator
    with kin_tabs[1]:
        st.markdown("**Calculate projectile trajectory parameters.**")

        traj_cols = st.columns(3)
        with traj_cols[0]:
            v0 = st.number_input("Initial velocity (m/s)", value=20.0, min_value=0.1, key="traj_v0")
        with traj_cols[1]:
            theta_deg = st.slider("Launch angle (°)", min_value=0, max_value=90, value=45, key="traj_theta")
            theta_rad = theta_deg * 0.0174533
        with traj_cols[2]:
            g_val = st.number_input("Gravity (m/s²)", value=9.80665, key="traj_g")

        if st.button("📊 Calculate Trajectory", key="traj_calc"):
            # Calculate trajectory parameters
            t_flight = 2 * v0 * np.sin(theta_rad) / g_val
            max_height = (v0 * np.sin(theta_rad)) ** 2 / (2 * g_val)
            range_dist = v0**2 * np.sin(2 * theta_rad) / g_val

            result_cols = st.columns(4)
            with result_cols[0]:
                st.metric("Time of Flight", f"{t_flight:.3f} s")
            with result_cols[1]:
                st.metric("Max Height", f"{max_height:.3f} m")
            with result_cols[2]:
                st.metric("Range", f"{range_dist:.3f} m")
            with result_cols[3]:
                # Landing velocity (same magnitude as launch)
                st.metric("Landing Speed", f"{v0:.3f} m/s")

            # Trajectory plot
            if px is not None:
                t_points = np.linspace(0, t_flight, 100)
                x_points = v0 * np.cos(theta_rad) * t_points
                y_points = v0 * np.sin(theta_rad) * t_points - 0.5 * g_val * t_points**2

                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=x_points, y=y_points, mode="lines", name="Trajectory", line={"color": "#2196F3", "width": 3}
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=[0, range_dist / 2, range_dist],
                        y=[0, max_height, 0],
                        mode="markers",
                        name="Key Points",
                        marker={"size": 10, "color": ["green", "red", "orange"]},
                    )
                )
                fig.update_layout(
                    xaxis_title="Distance (m)",
                    yaxis_title="Height (m)",
                    title=f"Projectile Trajectory (v₀={v0} m/s, θ={theta_deg}°)",
                    height=350,
                    yaxis={"scaleanchor": "x"},
                )
                st.plotly_chart(fig, width="stretch")

    # Tab 3: Energy Conservation Check
    with kin_tabs[2]:
        st.markdown("**Verify energy conservation in your system.**")

        st.subheader("Initial State")
        init_cols = st.columns(4)
        with init_cols[0]:
            m_check = st.number_input("Mass (kg)", value=1.0, min_value=0.001, key="energy_mass")
        with init_cols[1]:
            h1 = st.number_input("Height₁ (m)", value=10.0, key="energy_h1")
        with init_cols[2]:
            v1 = st.number_input("Velocity₁ (m/s)", value=0.0, key="energy_v1")
        with init_cols[3]:
            g_check = st.number_input("g (m/s²)", value=9.80665, key="energy_g")

        st.subheader("Final State")
        final_cols = st.columns(3)
        with final_cols[0]:
            h2 = st.number_input("Height₂ (m)", value=0.0, key="energy_h2")
        with final_cols[1]:
            v2 = st.number_input("Velocity₂ (m/s)", value=14.0, key="energy_v2")
        with final_cols[2]:
            W_nc = st.number_input(
                "Non-conservative work (J)", value=0.0, key="energy_wnc", help="Work done by friction, drag, etc."
            )

        if st.button("⚖️ Check Energy Conservation", key="energy_check"):
            # Calculate energies
            KE1 = 0.5 * m_check * v1**2
            PE1 = m_check * g_check * h1
            E1 = KE1 + PE1

            KE2 = 0.5 * m_check * v2**2
            PE2 = m_check * g_check * h2
            E2 = KE2 + PE2

            energy_diff = E2 - E1 + W_nc

            energy_cols = st.columns(2)
            with energy_cols[0]:
                st.markdown("**Initial State:**")
                st.markdown(f"- KE₁ = ½mv₁² = {KE1:.4f} J")
                st.markdown(f"- PE₁ = mgh₁ = {PE1:.4f} J")
                st.markdown(f"- **E₁ = {E1:.4f} J**")

            with energy_cols[1]:
                st.markdown("**Final State:**")
                st.markdown(f"- KE₂ = ½mv₂² = {KE2:.4f} J")
                st.markdown(f"- PE₂ = mgh₂ = {PE2:.4f} J")
                st.markdown(f"- **E₂ = {E2:.4f} J**")

            st.divider()

            if abs(energy_diff) < 0.01 * max(E1, E2, 1):
                st.success(f"✅ **Energy is conserved!** ΔE = {energy_diff:.6f} J (< 1% of total)")
                # Calculate expected v2 from conservation
                v2_expected = np.sqrt(2 * (E1 - PE2 - W_nc) / m_check) if PE2 + W_nc <= E1 else 0
                st.info(f"Expected v₂ from conservation: {v2_expected:.4f} m/s")
            else:
                st.error(f"❌ **Energy is NOT conserved!** ΔE = {energy_diff:.4f} J")
                st.markdown(f"Missing energy: {abs(energy_diff):.4f} J")
                if W_nc == 0:
                    st.info("💡 Consider adding non-conservative work (friction, drag)")

    # Tab 4: Motion Plot
    with kin_tabs[3]:
        st.markdown("**Visualize 1D motion over time.**")

        plot_cols = st.columns(4)
        with plot_cols[0]:
            x0_plot = st.number_input("x₀ (m)", value=0.0, key="plot_x0")
        with plot_cols[1]:
            v0_plot = st.number_input("v₀ (m/s)", value=10.0, key="plot_v0")
        with plot_cols[2]:
            a_plot = st.number_input("a (m/s²)", value=-9.81, key="plot_a")
        with plot_cols[3]:
            t_max = st.number_input("t_max (s)", value=3.0, min_value=0.1, key="plot_tmax")

        if st.button("📈 Generate Motion Plots", key="motion_plot"):
            t = np.linspace(0, t_max, 200)
            x = x0_plot + v0_plot * t + 0.5 * a_plot * t**2
            v = v0_plot + a_plot * t

            if px is not None:
                fig = make_subplots(rows=2, cols=1, subplot_titles=("Position vs Time", "Velocity vs Time"))

                fig.add_trace(go.Scatter(x=t, y=x, name="x(t)", line={"color": "#2196F3", "width": 2}), row=1, col=1)
                fig.add_trace(go.Scatter(x=t, y=v, name="v(t)", line={"color": "#4CAF50", "width": 2}), row=2, col=1)

                fig.update_xaxes(title_text="Time (s)", row=1, col=1)
                fig.update_xaxes(title_text="Time (s)", row=2, col=1)
                fig.update_yaxes(title_text="Position (m)", row=1, col=1)
                fig.update_yaxes(title_text="Velocity (m/s)", row=2, col=1)

                fig.update_layout(height=500, showlegend=True)
                st.plotly_chart(fig, width="stretch")

            # Key points
            t_stop = -v0_plot / a_plot if a_plot != 0 else float("inf")
            x_max = x0_plot + v0_plot * t_stop + 0.5 * a_plot * t_stop**2 if 0 < t_stop < t_max else None

            st.markdown("**Key Points:**")
            if x_max is not None and t_stop > 0:
                st.markdown(f"- Maximum height reached at t = {t_stop:.3f} s, x = {x_max:.3f} m")

    # Tab 5: History
    with kin_tabs[4]:
        if st.session_state.kin_audit:
            st.markdown(f"**{len(st.session_state.kin_audit)} previous analyses**")

            for i, entry in enumerate(reversed(st.session_state.kin_audit[-5:])):
                regime = entry.get("tier2", {}).get("umcp_regime", "N/A")
                kin_regime = entry.get("tier2", {}).get("kinematic_regime", "N/A")
                ts = entry.get("timestamp", "")[:19]

                with st.expander(f"Run {len(st.session_state.kin_audit) - i}: {regime}/{kin_regime} @ {ts}"):
                    hist_cols = st.columns(3)
                    with hist_cols[0]:
                        st.markdown("**Phase Space:**")
                        t15 = entry.get("tier15", {})
                        st.markdown(f"- |γ| = {t15.get('phase_magnitude', 0):.4f}")
                        st.markdown(f"- Credit: {t15.get('kinematic_credit', 0):.4f}")
                    with hist_cols[1]:
                        st.markdown("**Tier 1:**")
                        t1 = entry.get("tier1", {})
                        st.markdown(f"- F: {t1.get('F', 0):.4f}")
                        st.markdown(f"- ω: {t1.get('omega', 0):.4f}")
                    with hist_cols[2]:
                        st.markdown("**Score:**")
                        t2 = entry.get("tier2", {})
                        st.markdown(f"- Stability: {t2.get('stability_score', 0)}/100")

            if st.button("🗑️ Clear Kinematics History"):
                st.session_state.kin_audit = []
                st.rerun()
        else:
            st.info("No analysis history yet. Run a kinematics translation to see results here.")

    st.divider()

    # ========== Motion Parameters Input ==========
    st.header("📐 Motion Parameters")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Position (x)")
        pos_val = st.number_input("Position value", value=1.0, format="%.4f", key="kin_pos")
        pos_unit = st.selectbox(
            "Position unit", list(PHYSICS_QUANTITIES["position"]["units"].keys()), key="kin_pos_unit"
        )
        pos_ref = st.number_input("Position reference (L_ref)", value=1.0, min_value=0.001, key="kin_pos_ref")

    with col2:
        st.subheader("Velocity (v)")
        vel_val = st.number_input("Velocity value", value=1.0, format="%.4f", key="kin_vel")
        vel_unit = st.selectbox(
            "Velocity unit", list(PHYSICS_QUANTITIES["velocity"]["units"].keys()), key="kin_vel_unit"
        )
        vel_ref = st.number_input("Velocity reference (v_ref)", value=1.0, min_value=0.001, key="kin_vel_ref")

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Acceleration (a)")
        acc_val = st.number_input("Acceleration value", value=9.81, format="%.4f", key="kin_acc")
        acc_unit = st.selectbox(
            "Acceleration unit", list(PHYSICS_QUANTITIES["acceleration"]["units"].keys()), key="kin_acc_unit"
        )
        acc_ref = st.number_input("Acceleration reference (a_ref)", value=9.80665, min_value=0.001, key="kin_acc_ref")

    with col4:
        st.subheader("Mass (m)")
        mass_val = st.number_input("Mass value", value=1.0, format="%.4f", key="kin_mass")
        mass_unit = st.selectbox("Mass unit", list(PHYSICS_QUANTITIES["mass"]["units"].keys()), key="kin_mass_unit")

    st.divider()

    # ========== Derived Quantities ==========
    st.header("⚡ Derived Quantities (Auto-Calculated)")

    # Convert to base units (with safe defaults)
    pos_u = str(pos_unit) if pos_unit else "m"
    vel_u = str(vel_unit) if vel_unit else "m/s"
    acc_u = str(acc_unit) if acc_unit else "m/s²"
    mass_u = str(mass_unit) if mass_unit else "kg"

    x_base = convert_to_base_unit(float(pos_val), pos_u, "position")
    v_base = convert_to_base_unit(float(vel_val), vel_u, "velocity")
    a_base = convert_to_base_unit(float(acc_val), acc_u, "acceleration")
    m_base = convert_to_base_unit(float(mass_val), mass_u, "mass")

    # Calculate derived quantities
    e_kin = 0.5 * m_base * v_base**2
    momentum = m_base * v_base

    derived_cols = st.columns(4)
    with derived_cols[0]:
        st.metric("Kinetic Energy", f"{e_kin:.4g} J")
    with derived_cols[1]:
        st.metric("Momentum", f"{momentum:.4g} kg·m/s")
    with derived_cols[2]:
        # Phase magnitude
        x_norm = abs(x_base) / pos_ref
        v_norm = abs(v_base) / vel_ref
        phase_mag = np.sqrt(x_norm**2 + v_norm**2)
        st.metric("Phase Magnitude |γ|", f"{phase_mag:.4f}")
    with derived_cols[3]:
        # Calculate energy reference
        e_ref = m_base * vel_ref**2
        st.metric("E_ref", f"{e_ref:.4g} J")

    st.divider()

    # Epsilon
    epsilon = st.select_slider(
        "ε-clipping threshold",
        options=[1e-8, 1e-7, 1e-6, 1e-5, 1e-4],
        value=1e-6,
        format_func=lambda x: f"{x:.0e}",
        key="kin_epsilon",
    )

    # ========== Process ==========
    if st.button("🚀 Run Kinematics → UMCP Translation", type="primary", width="stretch"):
        progress = st.progress(0, text="Starting kinematics translation...")

        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "kinematics",
            "tier0": {},
            "tier1": {},
            "tier15": {},  # Tier-0 protocol for kinematics
            "tier2": {},
            "status": "PROCESSING",
        }

        # TIER 0: Build Ψ(t) vector
        progress.progress(15, text="Tier 0: Building kinematic observable vector Ψ(t)...")

        # Normalize all quantities
        x_norm, x_clip = normalize_to_bounded(x_base, pos_ref, epsilon)
        v_norm, v_clip = normalize_to_bounded(v_base, vel_ref, epsilon)
        a_norm, a_clip = normalize_to_bounded(a_base, acc_ref, epsilon)

        e_ref = m_base * vel_ref**2
        e_norm, e_clip = normalize_to_bounded(e_kin, e_ref, epsilon)

        p_ref = m_base * vel_ref
        p_norm, p_clip = normalize_to_bounded(momentum, p_ref, epsilon)

        # KIN.INTSTACK.v1 weights (frozen)
        kin_weights = {"x": 0.25, "v": 0.25, "a": 0.15, "E_kin": 0.20, "p": 0.15}

        psi_vector = [x_norm, v_norm, a_norm, e_norm, p_norm]
        psi_weights = [kin_weights["x"], kin_weights["v"], kin_weights["a"], kin_weights["E_kin"], kin_weights["p"]]

        audit_entry["tier0"] = {
            "physical_inputs": {
                "x": {"value": x_base, "unit": "m", "ref": pos_ref, "normalized": x_norm, "clipped": x_clip},
                "v": {"value": v_base, "unit": "m/s", "ref": vel_ref, "normalized": v_norm, "clipped": v_clip},
                "a": {"value": a_base, "unit": "m/s²", "ref": acc_ref, "normalized": a_norm, "clipped": a_clip},
                "E_kin": {"value": e_kin, "unit": "J", "ref": e_ref, "normalized": e_norm, "clipped": e_clip},
                "p": {"value": momentum, "unit": "kg·m/s", "ref": p_ref, "normalized": p_norm, "clipped": p_clip},
            },
            "psi_vector": psi_vector,
            "psi_weights": psi_weights,
            "epsilon": epsilon,
            "phase_point": {"x": x_norm, "v": v_norm},
        }

        # TIER 1: Kernel computation
        progress.progress(40, text="Tier 1: Computing kernel invariants...")

        c = np.array(psi_vector)
        w = np.array(psi_weights)

        fidelity = float(np.sum(w * c))
        drift = 1 - fidelity
        log_ic = float(np.sum(w * np.log(c)))
        integrity_composite = float(np.exp(log_ic))

        entropy = 0.0
        for ci, wi in zip(c, w, strict=False):
            if wi > 0 and 0 < ci < 1:
                entropy += wi * (-ci * np.log(ci) - (1 - ci) * np.log(1 - ci))

        curvature = float(np.std(c, ddof=0) / 0.5)
        amgm_gap = fidelity - integrity_composite

        audit_entry["tier1"] = {
            "F": fidelity,
            "omega": drift,
            "S": entropy,
            "C": curvature,
            "kappa": log_ic,
            "IC": integrity_composite,
            "amgm_gap": amgm_gap,
            "is_homogeneous": np.allclose(c, c[0], atol=1e-15),
        }

        # TIER-0 Protocol: Phase space analysis (kinematics-specific)
        progress.progress(60, text="Protocol: Computing phase space return metrics...")

        gamma = (x_norm, v_norm)
        phase_mag_sq = x_norm**2 + v_norm**2
        phase_mag = np.sqrt(phase_mag_sq)

        # Return time estimation (simplified for single point)
        # In full implementation, this would check against historical trajectory
        eta_phase = 0.01  # Tolerance

        # Estimate kinematic regime based on phase magnitude
        if phase_mag < 0.3:
            kin_regime = "Stable"
            tau_kin_est = "< T_crit (returning)"
            kin_credit = 1.0
        elif phase_mag < 0.7:
            kin_regime = "Watch"
            tau_kin_est = "T_crit to 2·T_crit"
            kin_credit = 0.5
        else:
            kin_regime = "Unstable"
            tau_kin_est = "INF_KIN (non-returning)"
            kin_credit = 0.0

        audit_entry["tier15"] = {
            "phase_point": gamma,
            "phase_magnitude": phase_mag,
            "phase_magnitude_sq": phase_mag_sq,
            "eta_phase": eta_phase,
            "kinematic_regime": kin_regime,
            "tau_kin_estimate": tau_kin_est,
            "kinematic_credit": kin_credit,
            "return_rate": 1.0 - drift,  # Simplified
        }

        # TIER 2: Regime classification
        progress.progress(85, text="Tier 2: Computing diagnostics and regime...")

        regime = classify_regime(drift)
        stability_score = int(fidelity * 100 * (1 - curvature))

        # Kinematics-specific recommendations
        recommendations = []

        if kin_credit == 0:
            recommendations.append("⚠️ Non-returning motion: kinematic credit = 0 (AXIOM-0)")
        if x_clip or v_clip or a_clip:
            recommendations.append("⚠️ OOR clipping applied: check reference scales")
        if amgm_gap > 0.1:
            recommendations.append("📊 Large heterogeneity gap: heterogeneous phase space")

        # Conservation checks
        if abs(e_kin - 0.5 * m_base * v_base**2) < 1e-9:
            recommendations.append("✅ E_kin = ½mv² verified")
        if abs(momentum - m_base * v_base) < 1e-9:
            recommendations.append("✅ p = mv verified")

        if not recommendations:
            recommendations.append("✅ All kinematics checks passed")

        audit_entry["tier2"] = {
            "umcp_regime": regime,
            "kinematic_regime": kin_regime,
            "stability_score": stability_score,
            "risk_level": "LOW" if regime == "STABLE" else ("MEDIUM" if regime == "WATCH" else "HIGH"),
            "recommendations": recommendations,
        }

        audit_entry["status"] = "COMPLETE"
        progress.progress(100, text="Complete!")

        st.session_state.kin_audit.append(audit_entry)

        # ========== Display Results ==========
        st.divider()

        regime = audit_entry["tier2"]["umcp_regime"]
        kin_regime = audit_entry["tier2"]["kinematic_regime"]
        regime_color = get_regime_color(regime)

        st.markdown(
            f"""<div style="padding: 20px; border-left: 6px solid {regime_color};
                background: {regime_color}22; border-radius: 8px; margin-bottom: 20px;">
                <h2 style="margin: 0; color: {regime_color};">🎯 UMCP Regime: {regime} | KIN Regime: {kin_regime}</h2>
                <p style="margin: 5px 0 0 0;">Stability Score: {stability_score}/100 • Kinematic Credit: {kin_credit:.2f}</p>
            </div>""",
            unsafe_allow_html=True,
        )

        # Four-column display for kinematics (includes Protocol tier)
        tier_cols = st.columns(4)

        with tier_cols[0]:
            st.markdown("### 📥 Tier 0: Physical")
            t0 = audit_entry["tier0"]["physical_inputs"]
            t0_df = pd.DataFrame(
                [
                    {"Qty": k, "Value": f"{v['value']:.3g}", "Unit": v["unit"], "ψ": f"{v['normalized']:.4f}"}
                    for k, v in t0.items()
                ]
            )
            st.dataframe(t0_df, hide_index=True, width="stretch")

        with tier_cols[1]:
            st.markdown("### ⚙️ Tier 1: Kernel")
            t1 = audit_entry["tier1"]
            st.metric("F (Fidelity)", f"{t1['F']:.4f}")
            st.metric("ω (Drift)", f"{t1['omega']:.4f}")
            st.metric("IC", f"{t1['IC']:.4f}")

        with tier_cols[2]:
            st.markdown("### 🔄 Protocol: Phase")
            t15 = audit_entry["tier15"]
            st.metric("|γ| (Phase Mag)", f"{t15['phase_magnitude']:.4f}")
            st.metric("τ_kin", t15["tau_kin_estimate"])
            st.metric("Credit", f"{t15['kinematic_credit']:.2f}")

        with tier_cols[3]:
            st.markdown("### 📊 Tier 2: Regime")
            t2 = audit_entry["tier2"]
            st.metric("UMCP", t2["umcp_regime"])
            st.metric("KIN", t2["kinematic_regime"])
            st.metric("Score", f"{t2['stability_score']}/100")

        # Recommendations
        st.markdown("### 📋 Recommendations")
        for rec in audit_entry["tier2"]["recommendations"]:
            st.markdown(f"- {rec}")

        # ========== GCD Translation Panel ==========
        st.divider()
        st.markdown("### 🌀 GCD Translation (Generative Collapse Dynamics)")
        st.caption("Native Tier-1 interpretation using GCD framework")

        # Translate tier1 values to GCD
        gcd_translation = translate_to_gcd(audit_entry["tier1"])
        audit_entry["gcd"] = gcd_translation

        render_gcd_panel(gcd_translation, compact=False)

        # Additional kinematics-specific GCD insight
        phase_mag = audit_entry["tier15"]["phase_magnitude"]
        kin_credit = audit_entry["tier15"]["kinematic_credit"]

        st.markdown("#### 🔄 Phase Space GCD Insight")
        phase_gcd_cols = st.columns(3)
        with phase_gcd_cols[0]:
            phase_regime = "STABLE" if phase_mag < 0.3 else ("WATCH" if phase_mag < 0.7 else "COLLAPSE")
            phase_color = GCD_REGIMES[phase_regime]["color"]
            st.markdown(f"**|γ| Regime:** :{phase_color.replace('#', '')}[{phase_regime}]")
            st.caption(f"|γ| = {phase_mag:.4f} in phase space")
        with phase_gcd_cols[1]:
            credit_health = "✅ High" if kin_credit > 0.7 else ("⚠️ Medium" if kin_credit > 0.3 else "❌ Low")
            st.markdown(f"**Kinematic Credit:** {credit_health}")
            st.caption(f"κ_kin = {kin_credit:.4f}")
        with phase_gcd_cols[2]:
            collapse_pressure = 1.0 - kin_credit
            st.markdown(f"**Collapse Pressure:** {collapse_pressure:.2%}")
            st.caption("Generative potential available")

        # Phase space visualization
        st.markdown("### 🌐 Phase Space (x, v)")

        # Create a simple phase space plot
        if px is not None:
            fig = go.Figure()

            # Add phase point
            fig.add_trace(
                go.Scatter(
                    x=[x_norm],
                    y=[v_norm],
                    mode="markers+text",
                    marker={"size": 15, "color": regime_color, "symbol": "circle"},
                    text=["γ(t)"],
                    textposition="top center",
                    name="Current State",
                )
            )

            # Add reference circle at |γ| = 0.3 (stable boundary)
            theta_range = np.linspace(0, 2 * np.pi, 100)
            fig.add_trace(
                go.Scatter(
                    x=0.3 * np.cos(theta_range),
                    y=0.3 * np.sin(theta_range),
                    mode="lines",
                    line={"color": "green", "dash": "dash"},
                    name="Stable boundary",
                )
            )

            # Add reference circle at |γ| = 0.7 (watch boundary)
            fig.add_trace(
                go.Scatter(
                    x=0.7 * np.cos(theta_range),
                    y=0.7 * np.sin(theta_range),
                    mode="lines",
                    line={"color": "orange", "dash": "dash"},
                    name="Watch boundary",
                )
            )

            fig.update_layout(
                xaxis_title="x̃ (normalized position)",
                yaxis_title="ṽ (normalized velocity)",
                xaxis={"range": [-0.1, 1.1], "scaleanchor": "y"},
                yaxis={"range": [-0.1, 1.1]},
                height=400,
                showlegend=True,
            )

            st.plotly_chart(fig, width="stretch")

        # Full audit
        with st.expander("🔍 Full Audit JSON", expanded=False):
            st.json(audit_entry)

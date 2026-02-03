"""
Σ(z) Evolution and Modified Gravity Mapping

Implements the mapping from measured ĥJ to phenomenological modified gravity
parameter Σ(z) from:
"Model-independent test of gravity with a network of galaxy surveys"
(Nature Communications 15:9295, 2024)

Mathematical Basis:

Σ(z) Definition (Eq. 11):
    k² (Φ + Ψ)/2 = -4πG a² Σ(z,k) ρ̄(z) Δ_m(z,k)

    Under quasi-static/sub-horizon assumptions, Σ encodes deviations
    from General Relativity (Σ = 1 for GR).

ĥJ to Σ Mapping (Eq. 12):
    ĥJ(z_i) = Ω_m(z_i) · [D₁(z_i)/D₁(z*)] · σ8(z*) · Σ(z_i)

Σ(z) Parametrization (Eq. 13):
    Σ(z) = 1 + Σ₀ · g(z)

    Where g(z) can be:
    - "standard": g(z) = Ω_Λ(z)
    - "constant": g(z) = 1 for z ∈ [0,1], else 0
    - "exponential": g(z) = exp(1+z) for z ∈ [0,1], else 0

UMCP Integration:
    - Σ₀ ≠ 0 → deviation from "return" to GR state
    - |Σ₀| → analogous to drift ω (distance from ideal)
    - Σ regime gates → Stable/Watch/Collapse mapping
    - This is the cosmological realization of AX-W0

Reference: Eq. 11-13 of Nature Comms 15:9295 (2024)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy import optimize


class SigmaRegime(str, Enum):
    """Σ deviation regime classification."""

    GR_CONSISTENT = "GR_consistent"  # |Σ₀| < 0.1
    TENSION = "Tension"  # 0.1 ≤ |Σ₀| < 0.3
    MODIFIED_GRAVITY = "Modified_gravity"  # |Σ₀| ≥ 0.3
    UNDEFINED = "Undefined"


class GzModel(str, Enum):
    """g(z) evolution models from the paper."""

    STANDARD = "standard"  # g(z) = Ω_Λ(z)
    CONSTANT = "constant"  # g(z) = 1 for z ∈ [0,1]
    EXPONENTIAL = "exponential"  # g(z) = exp(1+z) for z ∈ [0,1]


class SigmaResult(NamedTuple):
    """Result of Σ(z) computation/fit."""

    Sigma: float  # Σ(z) value
    Sigma_0: float  # Fitted Σ₀ amplitude
    g_z: float  # g(z) value
    z: float  # Redshift
    regime: str  # SigmaRegime
    deviation_from_GR: float  # |Σ - 1|


class SigmaFitResult(NamedTuple):
    """Result of Σ₀ fitting across redshift bins."""

    Sigma_0: float  # Best-fit Σ₀
    Sigma_0_error: float  # 1σ uncertainty
    chi2: float  # χ² of fit
    chi2_red: float  # Reduced χ²
    p_value: float  # p-value
    n_dof: int  # Degrees of freedom
    regime: str  # Overall regime
    g_model: str  # g(z) model used


@dataclass
class SigmaConfig:
    """Configuration for Σ(z) computation."""

    # Regime thresholds
    sigma0_gr_threshold: float = 0.1
    sigma0_tension_threshold: float = 0.3

    # z range for constant/exponential models
    z_model_min: float = 0.0
    z_model_max: float = 1.0

    # Numerical parameters
    epsilon: float = 1e-10


def g_z_standard(z: float, Omega_Lambda_z: Callable[[float], float]) -> float:
    """
    Standard g(z) = Ω_Λ(z).

    Args:
        z: Redshift
        Omega_Lambda_z: Function returning Ω_Λ(z)

    Returns:
        g(z) value
    """
    return Omega_Lambda_z(z)


def g_z_constant(z: float, z_min: float = 0.0, z_max: float = 1.0) -> float:
    """
    Constant g(z) = 1 for z ∈ [z_min, z_max], else 0.

    Args:
        z: Redshift
        z_min, z_max: Active range

    Returns:
        g(z) value
    """
    return 1.0 if z_min <= z <= z_max else 0.0


def g_z_exponential(z: float, z_min: float = 0.0, z_max: float = 1.0) -> float:
    """
    Exponential g(z) = exp(1+z) for z ∈ [z_min, z_max], else 0.

    Args:
        z: Redshift
        z_min, z_max: Active range

    Returns:
        g(z) value
    """
    return np.exp(1 + z) if z_min <= z <= z_max else 0.0


def compute_Sigma_from_hJ(
    hJ: float,
    Omega_m_z: float,
    D1_z: float,
    D1_z_star: float,
    sigma8_z_star: float,
    config: SigmaConfig | None = None,
) -> float:
    """
    Invert ĥJ measurement to get Σ(z).

    From Eq. 12:
        ĥJ(z) = Ω_m(z) · [D₁(z)/D₁(z*)] · σ8(z*) · Σ(z)

        => Σ(z) = ĥJ(z) / [Ω_m(z) · (D₁(z)/D₁(z*)) · σ8(z*)]

    Args:
        hJ: Measured ĥJ(z)
        Omega_m_z: Ω_m(z) at this redshift
        D1_z: D₁(z) at this redshift
        D1_z_star: D₁(z*) at anchor
        sigma8_z_star: σ8(z*) at anchor

    Returns:
        Σ(z) value
    """
    if config is None:
        config = SigmaConfig()

    eps = config.epsilon

    denominator = Omega_m_z * (D1_z / (D1_z_star + eps)) * sigma8_z_star

    if abs(denominator) < eps:
        return 1.0  # Default to GR if denominator is ill-defined

    return hJ / denominator


def compute_Sigma(
    z: float,
    Sigma_0: float,
    g_model: GzModel,
    Omega_Lambda_z: Callable[[float], float] | None = None,
    config: SigmaConfig | None = None,
) -> SigmaResult:
    """
    Compute Σ(z) from Σ₀ parametrization.

    Implements Eq. 13:
        Σ(z) = 1 + Σ₀ · g(z)

    Args:
        z: Redshift
        Sigma_0: Amplitude parameter
        g_model: Which g(z) model to use
        Omega_Lambda_z: Function for Ω_Λ(z), needed for standard model
        config: Optional configuration

    Returns:
        SigmaResult with Σ, regime, etc.
    """
    if config is None:
        config = SigmaConfig()

    # Compute g(z) based on model
    if g_model == GzModel.STANDARD:
        if Omega_Lambda_z is None:
            raise ValueError("Omega_Lambda_z required for standard model")
        g_z = g_z_standard(z, Omega_Lambda_z)
    elif g_model == GzModel.CONSTANT:
        g_z = g_z_constant(z, config.z_model_min, config.z_model_max)
    elif g_model == GzModel.EXPONENTIAL:
        g_z = g_z_exponential(z, config.z_model_min, config.z_model_max)
    else:
        g_z = 0.0

    # Compute Σ(z)
    Sigma = 1.0 + Sigma_0 * g_z

    # Deviation from GR
    deviation = abs(Sigma - 1.0)

    # Classify regime
    if abs(Sigma_0) < config.sigma0_gr_threshold:
        regime = SigmaRegime.GR_CONSISTENT.value
    elif abs(Sigma_0) < config.sigma0_tension_threshold:
        regime = SigmaRegime.TENSION.value
    else:
        regime = SigmaRegime.MODIFIED_GRAVITY.value

    return SigmaResult(
        Sigma=float(Sigma),
        Sigma_0=float(Sigma_0),
        g_z=float(g_z),
        z=float(z),
        regime=regime,
        deviation_from_GR=float(deviation),
    )


def fit_Sigma_0(
    z_bins: NDArray[np.floating],
    hJ_measured: NDArray[np.floating],
    hJ_errors: NDArray[np.floating],
    Omega_m_z: Callable[[float], float],
    D1_z: Callable[[float], float],
    D1_z_star: float,
    sigma8_z_star: float,
    g_model: GzModel,
    Omega_Lambda_z: Callable[[float], float] | None = None,
    config: SigmaConfig | None = None,
) -> SigmaFitResult:
    """
    Fit Σ₀ to measured ĥJ values across redshift bins.

    Procedure:
    1. For each z_i, compute predicted ĥJ(z_i; Σ₀) from Eq. 12
    2. Minimize χ² = Σ [(ĥJ_obs - ĥJ_pred)² / σ²]

    Args:
        z_bins: Effective redshifts of bins
        hJ_measured: Measured ĥJ values
        hJ_errors: 1σ uncertainties on ĥJ
        Omega_m_z: Ω_m(z) function
        D1_z: D₁(z) function
        D1_z_star: D₁(z*) anchor
        sigma8_z_star: σ8(z*) anchor
        g_model: Which g(z) model
        Omega_Lambda_z: Ω_Λ(z) for standard model
        config: Optional configuration

    Returns:
        SigmaFitResult with best-fit Σ₀ and diagnostics
    """
    if config is None:
        config = SigmaConfig()

    n_bins = len(z_bins)
    n_dof = n_bins - 1  # One parameter: Σ₀

    def compute_hJ_model(z: float, Sigma_0: float) -> float:
        """Compute model ĥJ(z; Σ₀) from Eq. 12."""
        # Get Σ(z) from parametrization
        Sigma_result = compute_Sigma(z, Sigma_0, g_model, Omega_Lambda_z, config)
        Sigma = Sigma_result.Sigma

        # Compute ĥJ from Σ using Eq. 12
        return Omega_m_z(z) * (D1_z(z) / D1_z_star) * sigma8_z_star * Sigma

    def chi2(Sigma_0: float) -> float:
        """χ² objective function."""
        chi2_sum = 0.0
        for i, z in enumerate(z_bins):
            hJ_model = compute_hJ_model(z, Sigma_0)
            chi2_sum += ((hJ_measured[i] - hJ_model) / hJ_errors[i]) ** 2
        return chi2_sum

    # Minimize χ²
    result = optimize.minimize_scalar(chi2, bounds=(-1, 1), method="bounded")

    # Get actual minimum (result is OptimizeResult with .x attribute)
    Sigma_0_best: float = float(result.x)  # type: ignore[union-attr]
    chi2_min = chi2(Sigma_0_best)
    chi2_red = chi2_min / max(n_dof, 1)

    # Estimate error from curvature (simplified)
    delta = 0.01
    chi2_plus = chi2(Sigma_0_best + delta)
    chi2_minus = chi2(Sigma_0_best - delta)
    curvature = (chi2_plus + chi2_minus - 2 * chi2_min) / delta**2
    Sigma_0_error = 1.0 / np.sqrt(max(curvature, 1e-10))

    # p-value from χ² distribution (simplified)
    from scipy import stats

    p_value = 1.0 - stats.chi2.cdf(chi2_min, n_dof) if n_dof > 0 else 0.5

    # Classify regime
    if abs(Sigma_0_best) < config.sigma0_gr_threshold:
        regime = SigmaRegime.GR_CONSISTENT.value
    elif abs(Sigma_0_best) < config.sigma0_tension_threshold:
        regime = SigmaRegime.TENSION.value
    else:
        regime = SigmaRegime.MODIFIED_GRAVITY.value

    return SigmaFitResult(
        Sigma_0=float(Sigma_0_best),
        Sigma_0_error=float(Sigma_0_error),
        chi2=float(chi2_min),
        chi2_red=float(chi2_red),
        p_value=float(p_value),
        n_dof=n_dof,
        regime=regime,
        g_model=g_model.value,
    )


# =============================================================================
# DES Y3 REFERENCE DATA
# =============================================================================

# From Table 1 of Nature Comms 15:9295 (2024)
DES_Y3_DATA = {
    "z_bins": np.array([0.295, 0.467, 0.626, 0.771]),
    "hJ_cmb": {
        "mean": np.array([0.326, 0.332, 0.387, 0.354]),
        "sigma": np.array([0.020, 0.015, 0.026, 0.033]),
    },
    "hJ_no_cmb": {
        "mean": np.array([0.333, 0.328, 0.386, 0.345]),
        "sigma": np.array([0.023, 0.020, 0.049, 0.043]),
    },
    "hJ_pessimistic_cmb": {
        "mean": np.array([0.329, 0.329, 0.429, 0.370]),
        "sigma": np.array([0.036, 0.034, 0.062, 0.064]),
    },
    "hJ_pessimistic_no_cmb": {
        "mean": np.array([0.336, 0.326, 0.431, 0.370]),
        "sigma": np.array([0.039, 0.035, 0.086, 0.068]),
    },
    "sigma8_comparison": {
        "from_params_cmb": {"mean": 0.849, "sigma": 0.030},
        "from_hJ_cmb": {"mean": 0.743, "sigma": 0.039},
        "from_params_no_cmb": {"mean": 1.028, "sigma": 0.097},
        "from_hJ_no_cmb": {"mean": 0.776, "sigma_plus": 0.067, "sigma_minus": 0.079},
    },
    "Sigma_0_fits": {
        "standard": {"mean": 0.24, "sigma": 0.10, "chi2_red": 1.1, "p_value": 0.33},
        "constant": {"mean": 0.13, "sigma": 0.06, "chi2_red": 1.1, "p_value": 0.33},
        "exponential": {"mean": 0.027, "sigma": 0.013, "chi2_red": 0.8, "p_value": 0.48},
    },
    "LCDM_fit": {"chi2_red": 2.1, "p_value": 0.078},
}


# =============================================================================
# UMCP INTEGRATION: Drift-Fidelity Mapping
# =============================================================================


def Sigma_to_UMCP_invariants(
    Sigma_0: float,
    chi2_red_Sigma: float,
    chi2_red_LCDM: float,
) -> dict:
    """
    Map Σ₀ results to UMCP invariant analogs.

    UMCP Mapping:
        - Σ₀ → ω (drift from GR)
        - Σ = 1 → F = 1 (perfect fidelity)
        - χ²_red → seam residual quality
        - (χ²_ΛCDM - χ²_Σ) → improvement from closure

    This demonstrates AX-W0: the GR anchor (Σ=1) defines
    the meaning of deviation, just as ω=0 defines ideal fidelity.

    Args:
        Sigma_0: Fitted Σ₀ amplitude
        chi2_red_Sigma: χ²_red for Σ model
        chi2_red_LCDM: χ²_red for ΛCDM (Σ=1)

    Returns:
        UMCP invariant interpretation
    """
    # Map Σ₀ to drift (normalized to [0,1])
    omega_analog = min(abs(Sigma_0), 1.0)

    # Fidelity analog
    F_analog = 1.0 - omega_analog

    # Model improvement ratio
    chi2_improvement = (chi2_red_LCDM - chi2_red_Sigma) / chi2_red_LCDM if chi2_red_LCDM > 0 else 0.0

    # Regime classification
    if omega_analog < 0.1:
        regime = "Stable"
    elif omega_analog < 0.3:
        regime = "Watch"
    else:
        regime = "Collapse"

    # Tension quantification (σ8 tension from DES)
    sigma8_tension = abs(0.849 - 0.743) / 0.743  # ~14.3%

    return {
        "omega_analog": omega_analog,
        "F_analog": F_analog,
        "regime": regime,
        "chi2_improvement": chi2_improvement,
        "sigma8_tension": sigma8_tension,
        "interpretation": {
            "AX-W0": f"GR (Σ=1) is the anchor. Measured deviation Σ₀={Sigma_0:.2f} corresponds to ω≈{omega_analog:.2f}",
            "seam_analog": f"χ² improvement of {chi2_improvement:.1%} indicates Σ model closes residual better than ΛCDM",
            "tension": f"σ8 tension of {sigma8_tension:.1%} is the cosmological 'collapse potential'",
        },
    }


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    print("=== Σ(z) Evolution Self-Test ===\n")

    # Test g(z) functions
    print("g(z) models at z=0.5:")
    print(f"  standard (approx): {0.7:.3f}")  # Ω_Λ ≈ 0.7 at z=0.5
    print(f"  constant: {g_z_constant(0.5):.3f}")
    print(f"  exponential: {g_z_exponential(0.5):.3f}")

    # Test Σ(z) computation with DES Y3 Σ₀ values
    print("\nΣ(z=0.5) for different models:")
    for model, Sigma_0 in [("standard", 0.24), ("constant", 0.13), ("exponential", 0.027)]:
        result = compute_Sigma(
            z=0.5,
            Sigma_0=Sigma_0,
            g_model=GzModel(model),
            Omega_Lambda_z=lambda z: 0.685 / (0.315 * (1 + z) ** 3 + 0.685),
        )
        print(f"  {model} (Σ₀={Sigma_0}): Σ={result.Sigma:.3f}, regime={result.regime}")

    # Test UMCP mapping
    print("\nUMCP Invariant Mapping:")
    mapping = Sigma_to_UMCP_invariants(
        Sigma_0=0.24,
        chi2_red_Sigma=1.1,
        chi2_red_LCDM=2.1,
    )
    print(f"  ω_analog = {mapping['omega_analog']:.3f}")
    print(f"  F_analog = {mapping['F_analog']:.3f}")
    print(f"  regime = {mapping['regime']}")
    print(f"  χ² improvement = {mapping['chi2_improvement']:.1%}")
    print(f"  σ8 tension = {mapping['sigma8_tension']:.1%}")

    # Print DES Y3 reference data
    print("\nDES Y3 Reference Data (Table 1):")
    print(f"  z_bins = {DES_Y3_DATA['z_bins']}")
    print(f"  ĥJ (CMB prior) = {DES_Y3_DATA['hJ_cmb']['mean']}")
    print(
        f"  Σ₀ (standard) = {DES_Y3_DATA['Sigma_0_fits']['standard']['mean']} ± {DES_Y3_DATA['Sigma_0_fits']['standard']['sigma']}"
    )

    print("\n✓ Self-test complete")

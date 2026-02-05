"""
Universal UMCP Calculator

A unified calculator that integrates all UMCP concepts into a single interface:
- Tier-1 Kernel Invariants (ω, F, S, C, τ_R, κ, IC)
- Cost Closures (Γ(ω), D_C, budget identity)
- Regime Classification (STABLE/WATCH/COLLAPSE/CRITICAL)
- Seam Accounting (residual, PASS/FAIL)
- GCD Metrics (energy, collapse, flux, resonance)
- Kinematics Metrics (phase space, energy mechanics)
- RCFT Metrics (fractal dimension, recursive field)
- Uncertainty Propagation (delta-method)
- Human-Verifiable Checksums (SS1M triads)

Usage:
    from umcp.universal_calculator import UniversalCalculator

    calc = UniversalCalculator()
    result = calc.compute_all(coordinates=[0.9, 0.85, 0.92], weights=[0.5, 0.3, 0.2])
    print(result.summary())

CLI Usage:
    python -m umcp.universal_calculator --coordinates 0.9,0.85,0.92 --weights 0.5,0.3,0.2

Cross-references:
    - KERNEL_SPECIFICATION.md (34 formal lemmas)
    - INFRASTRUCTURE_GEOMETRY.md (three-layer architecture)
    - TIER_SYSTEM.md (tier boundaries)
    - closures/gcd/*.py (GCD closures)
    - closures/rcft/*.py (RCFT closures)
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Import frozen contract constants
from .frozen_contract import (
    ALPHA,
    DEFAULT_THRESHOLDS,
    EPSILON,
    P_EXPONENT,
    TOL_SEAM,
    Regime,
    RegimeThresholds,
    check_seam_pass,
    classify_regime,
    compute_budget_delta_kappa,
    compute_seam_residual,
    cost_curvature,
    equator_phi,
    gamma_omega,
)

# Type alias for array types (mypy strict mode compliance)
FloatArray = NDArray[np.floating[Any]]

# =============================================================================
# DATA STRUCTURES
# =============================================================================


class ComputationMode(Enum):
    """Computation mode for the universal calculator."""

    MINIMAL = "minimal"  # Tier-1 kernel only
    STANDARD = "standard"  # Tier-1 + cost closures + regime
    FULL = "full"  # All tiers including GCD/RCFT
    KINEMATICS = "kinematics"  # Include kinematics-specific
    RCFT = "rcft"  # Include RCFT (fractal, recursive field)


@dataclass
class KernelInvariants:
    """Tier-1 Kernel Invariants (The Seven)."""

    omega: float  # Drift (1 - F)
    F: float  # Fidelity (weighted mean)
    S: float  # Entropy (Shannon binary)
    C: float  # Curvature (normalized dispersion)
    tau_R: float  # Return time
    kappa: float  # Log-integrity
    IC: float  # Integrity composite (exp(κ))

    def to_dict(self) -> dict[str, float | str | None]:
        """Convert to dict, handling infinity for JSON serialization."""
        tau_R_val: float | str | None
        if math.isinf(self.tau_R):
            tau_R_val = "INF_REC" if self.tau_R > 0 else "-INF"
        elif math.isnan(self.tau_R):
            tau_R_val = None
        else:
            tau_R_val = self.tau_R

        return {
            "omega": self.omega,
            "F": self.F,
            "S": self.S,
            "C": self.C,
            "tau_R": tau_R_val,
            "kappa": self.kappa,
            "IC": self.IC,
        }


@dataclass
class CostClosures:
    """Cost closure computations."""

    gamma_omega: float  # Drift cost Γ(ω)
    D_C: float  # Curvature cost
    D_omega: float  # Total drift cost (= gamma_omega)
    equator_phi: float  # Equator deviation (diagnostic)

    def to_dict(self) -> dict[str, float]:
        return {
            "gamma_omega": self.gamma_omega,
            "D_C": self.D_C,
            "D_omega": self.D_omega,
            "equator_phi": self.equator_phi,
        }


@dataclass
class SeamResult:
    """Seam accounting result."""

    delta_kappa_budget: float
    delta_kappa_ledger: float
    residual: float
    passed: bool
    failures: list[str]
    R_credit: float
    I_ratio: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "delta_kappa_budget": self.delta_kappa_budget,
            "delta_kappa_ledger": self.delta_kappa_ledger,
            "residual": self.residual,
            "passed": self.passed,
            "failures": self.failures,
            "R_credit": self.R_credit,
            "I_ratio": self.I_ratio,
        }


@dataclass
class GCDMetrics:
    """GCD (Generative Collapse Dynamics) metrics."""

    E_potential: float  # Energy potential
    E_collapse: float  # Collapse component
    E_entropy: float  # Entropy component
    E_curvature: float  # Curvature component
    Phi_collapse: float  # Collapse potential
    Phi_gen: float  # Generative flux
    R_resonance: float  # Field resonance
    energy_regime: str  # Energy-based regime

    def to_dict(self) -> dict[str, Any]:
        return {
            "E_potential": self.E_potential,
            "E_collapse": self.E_collapse,
            "E_entropy": self.E_entropy,
            "E_curvature": self.E_curvature,
            "Phi_collapse": self.Phi_collapse,
            "Phi_gen": self.Phi_gen,
            "R_resonance": self.R_resonance,
            "energy_regime": self.energy_regime,
        }


@dataclass
class RCFTMetrics:
    """RCFT (Recursive Collapse Field Theory) metrics."""

    D_fractal: float  # Fractal dimension
    fractal_regime: str  # Smooth/Wrinkled/Turbulent
    Psi_recursive: float  # Recursive field strength
    memory_depth: int  # Effective memory depth
    basin_strength: float  # Attractor basin strength

    def to_dict(self) -> dict[str, Any]:
        return {
            "D_fractal": self.D_fractal,
            "fractal_regime": self.fractal_regime,
            "Psi_recursive": self.Psi_recursive,
            "memory_depth": self.memory_depth,
            "basin_strength": self.basin_strength,
        }


@dataclass
class UncertaintyBounds:
    """Uncertainty bounds from delta-method propagation."""

    var_F: float
    var_omega: float
    var_S: float
    var_kappa: float
    var_C: float
    std_F: float
    std_omega: float
    std_S: float
    std_kappa: float
    std_C: float

    def to_dict(self) -> dict[str, float]:
        return {
            "var_F": self.var_F,
            "var_omega": self.var_omega,
            "var_S": self.var_S,
            "var_kappa": self.var_kappa,
            "var_C": self.var_C,
            "std_F": self.std_F,
            "std_omega": self.std_omega,
            "std_S": self.std_S,
            "std_kappa": self.std_kappa,
            "std_C": self.std_C,
        }


@dataclass
class SS1MTriad:
    """Human-verifiable checksum (mod-97 triads)."""

    C1: str
    C2: str
    C3: str
    full_hash: str

    def to_dict(self) -> dict[str, str]:
        return {
            "C1": self.C1,
            "C2": self.C2,
            "C3": self.C3,
            "full_hash": self.full_hash,
        }

    def __str__(self) -> str:
        return f"{self.C1}-{self.C2}-{self.C3}"


@dataclass
class UniversalResult:
    """Complete result from universal calculator."""

    # Metadata
    timestamp: str
    computation_mode: str
    input_hash: str

    # Core invariants
    kernel: KernelInvariants
    regime: str
    regime_enum: Regime

    # Cost closures
    costs: CostClosures | None = None

    # Seam accounting (requires prior state)
    seam: SeamResult | None = None

    # GCD metrics
    gcd: GCDMetrics | None = None

    # RCFT metrics
    rcft: RCFTMetrics | None = None

    # Uncertainty
    uncertainty: UncertaintyBounds | None = None

    # Checksum
    ss1m: SS1MTriad | None = None

    # Diagnostics
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: dict[str, Any] = {
            "metadata": {
                "timestamp": self.timestamp,
                "computation_mode": self.computation_mode,
                "input_hash": self.input_hash,
            },
            "kernel": self.kernel.to_dict(),
            "regime": self.regime,
        }

        if self.costs:
            result["costs"] = self.costs.to_dict()
        if self.seam:
            result["seam"] = self.seam.to_dict()
        if self.gcd:
            result["gcd"] = self.gcd.to_dict()
        if self.rcft:
            result["rcft"] = self.rcft.to_dict()
        if self.uncertainty:
            result["uncertainty"] = self.uncertainty.to_dict()
        if self.ss1m:
            result["ss1m"] = self.ss1m.to_dict()
        if self.diagnostics:
            result["diagnostics"] = self.diagnostics

        return result

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "UMCP Universal Calculator Result",
            "=" * 60,
            f"Timestamp: {self.timestamp}",
            f"Mode: {self.computation_mode}",
            "",
            "─── Tier-1 Kernel Invariants ───",
            f"  ω (Drift):      {self.kernel.omega:.6f}",
            f"  F (Fidelity):   {self.kernel.F:.6f}",
            f"  S (Entropy):    {self.kernel.S:.6f}",
            f"  C (Curvature):  {self.kernel.C:.6f}",
            f"  τ_R (Return):   {self.kernel.tau_R}",
            f"  κ (Log-Int):    {self.kernel.kappa:.6f}",
            f"  IC (Integrity): {self.kernel.IC:.6f}",
            "",
            f"─── Regime: {self.regime} ───",
        ]

        if self.costs:
            lines.extend(
                [
                    "",
                    "─── Cost Closures ───",
                    f"  Γ(ω):       {self.costs.gamma_omega:.6f}",
                    f"  D_C:        {self.costs.D_C:.6f}",
                    f"  Φ_eq:       {self.costs.equator_phi:.6f}",
                ]
            )

        if self.seam:
            lines.extend(
                [
                    "",
                    "─── Seam Accounting ───",
                    f"  Δκ_budget:  {self.seam.delta_kappa_budget:.6f}",
                    f"  Δκ_ledger:  {self.seam.delta_kappa_ledger:.6f}",
                    f"  Residual s: {self.seam.residual:.6f}",
                    f"  PASS:       {self.seam.passed}",
                ]
            )

        if self.gcd:
            lines.extend(
                [
                    "",
                    "─── GCD Metrics ───",
                    f"  E_potential:  {self.gcd.E_potential:.6f}",
                    f"  Φ_collapse:   {self.gcd.Phi_collapse:.6f}",
                    f"  Φ_gen:        {self.gcd.Phi_gen:.6f}",
                    f"  R_resonance:  {self.gcd.R_resonance:.6f}",
                    f"  Energy Regime: {self.gcd.energy_regime}",
                ]
            )

        if self.rcft:
            lines.extend(
                [
                    "",
                    "─── RCFT Metrics ───",
                    f"  D_fractal:     {self.rcft.D_fractal:.4f}",
                    f"  Fractal Regime: {self.rcft.fractal_regime}",
                    f"  Ψ_recursive:   {self.rcft.Psi_recursive:.6f}",
                    f"  Basin Strength: {self.rcft.basin_strength:.6f}",
                ]
            )

        if self.uncertainty:
            lines.extend(
                [
                    "",
                    "─── Uncertainty (1σ) ───",
                    f"  σ_F:     {self.uncertainty.std_F:.6f}",
                    f"  σ_ω:     {self.uncertainty.std_omega:.6f}",
                    f"  σ_κ:     {self.uncertainty.std_kappa:.6f}",
                ]
            )

        if self.ss1m:
            lines.extend(
                [
                    "",
                    "─── SS1M Checksum ───",
                    f"  Triad: {self.ss1m}",
                ]
            )

        lines.append("=" * 60)
        return "\n".join(lines)


# =============================================================================
# UNIVERSAL CALCULATOR
# =============================================================================


class UniversalCalculator:
    """
    Universal UMCP Calculator.

    Integrates all UMCP concepts into a single, unified interface.

    Example:
        >>> calc = UniversalCalculator()
        >>> result = calc.compute_all(
        ...     coordinates=[0.9, 0.85, 0.92],
        ...     weights=[0.5, 0.3, 0.2]
        ... )
        >>> print(result.summary())
    """

    def __init__(
        self,
        epsilon: float = EPSILON,
        p_exponent: int = P_EXPONENT,
        alpha: float = ALPHA,
        tol_seam: float = TOL_SEAM,
        thresholds: RegimeThresholds = DEFAULT_THRESHOLDS,
    ):
        """
        Initialize calculator with frozen contract constants.

        Args:
            epsilon: Guard band for numerical stability (default: 1e-8)
            p_exponent: Power exponent for Γ(ω) (default: 3)
            alpha: Curvature cost coefficient (default: 1.0)
            tol_seam: Seam residual tolerance (default: 0.005)
            thresholds: Regime classification thresholds
        """
        self.epsilon = epsilon
        self.p_exponent = p_exponent
        self.alpha = alpha
        self.tol_seam = tol_seam
        self.thresholds = thresholds

    def compute_all(
        self,
        coordinates: list[float] | FloatArray,
        weights: list[float] | FloatArray | None = None,
        tau_R: float | None = None,
        prior_kappa: float | None = None,
        prior_IC: float | None = None,
        R_credit: float = 0.1,
        trajectory: FloatArray | None = None,
        coord_variances: list[float] | FloatArray | None = None,
        mode: ComputationMode = ComputationMode.STANDARD,
    ) -> UniversalResult:
        """
        Compute all UMCP metrics in one call.

        Args:
            coordinates: Bounded coordinates c_i ∈ [0,1]
            weights: Channel weights (default: uniform)
            tau_R: Return time (default: compute if trajectory provided)
            prior_kappa: Previous κ for seam accounting
            prior_IC: Previous IC for seam accounting
            R_credit: Return credit for budget identity
            trajectory: Historical trajectory for τ_R and RCFT
            coord_variances: Coordinate variances for uncertainty propagation
            mode: Computation mode (MINIMAL, STANDARD, FULL, RCFT)

        Returns:
            UniversalResult with all computed metrics
        """
        # Convert to numpy arrays
        c = np.array(coordinates, dtype=np.float64)
        n = len(c)

        w = np.ones(n) / n if weights is None else np.array(weights, dtype=np.float64)

        # Validate weights
        if not np.isclose(w.sum(), 1.0, atol=1e-9):
            w = w / w.sum()  # Normalize

        # Clip coordinates to valid range
        c_clipped = np.clip(c, self.epsilon, 1 - self.epsilon)

        # Compute input hash
        input_hash = self._compute_hash(c, w)

        # Tier-1: Kernel invariants
        kernel = self._compute_kernel(c_clipped, w)

        # Default tau_R if not provided
        if tau_R is None:
            if trajectory is not None:
                tau_R = self._compute_tau_R_from_trajectory(trajectory, c_clipped)
            else:
                tau_R = float("inf")  # INF_REC

        # Update kernel with tau_R
        kernel = KernelInvariants(
            omega=kernel.omega,
            F=kernel.F,
            S=kernel.S,
            C=kernel.C,
            tau_R=tau_R,
            kappa=kernel.kappa,
            IC=kernel.IC,
        )

        # Classify regime
        regime_enum = classify_regime(
            omega=kernel.omega,
            F=kernel.F,
            S=kernel.S,
            C=kernel.C,
            integrity=kernel.IC,
            thresholds=self.thresholds,
        )

        # Initialize result
        result = UniversalResult(
            timestamp=datetime.now(UTC).isoformat(),
            computation_mode=mode.value,
            input_hash=input_hash,
            kernel=kernel,
            regime=regime_enum.value,
            regime_enum=regime_enum,
        )

        # Minimal mode stops here
        if mode == ComputationMode.MINIMAL:
            return result

        # Cost closures
        result.costs = self._compute_costs(kernel)

        # Seam accounting (if prior state provided)
        if prior_kappa is not None and prior_IC is not None:
            result.seam = self._compute_seam(
                kernel=kernel,
                costs=result.costs,
                prior_kappa=prior_kappa,
                prior_IC=prior_IC,
                R_credit=R_credit,
            )

        # Standard mode stops here
        if mode == ComputationMode.STANDARD:
            result.ss1m = self._compute_ss1m(result)
            return result

        # GCD metrics
        result.gcd = self._compute_gcd(kernel)

        # RCFT metrics (requires trajectory)
        if mode in (ComputationMode.FULL, ComputationMode.RCFT):
            if trajectory is not None:
                result.rcft = self._compute_rcft(trajectory, kernel)
            else:
                # Compute from single-point (limited)
                result.rcft = self._compute_rcft_single(kernel)

        # Uncertainty propagation
        if coord_variances is not None:
            result.uncertainty = self._compute_uncertainty(c_clipped, w, coord_variances)

        # Diagnostics
        result.diagnostics = self._compute_diagnostics(kernel, c_clipped, w)

        # SS1M checksum
        result.ss1m = self._compute_ss1m(result)

        return result

    def _compute_kernel(self, c: FloatArray, w: FloatArray) -> KernelInvariants:
        """Compute Tier-1 kernel invariants."""
        # Fidelity: F = Σ w_i c_i
        F = float(np.sum(w * c))

        # Drift: ω = 1 - F
        omega = 1.0 - F

        # Log-integrity: κ = Σ w_i ln(c_i)
        kappa = float(np.sum(w * np.log(c)))

        # Integrity: IC = exp(κ)
        IC = float(np.exp(kappa))

        # Entropy: S = -Σ w_i [c_i ln(c_i) + (1-c_i) ln(1-c_i)]
        S = float(-np.sum(w * (c * np.log(c) + (1 - c) * np.log(1 - c))))

        # Curvature: C = std(c) / 0.5 (normalized dispersion)
        C = float(np.std(c) / 0.5)
        C = min(C, 1.0)  # Cap at 1.0

        return KernelInvariants(
            omega=omega,
            F=F,
            S=S,
            C=C,
            tau_R=float("inf"),  # Will be updated
            kappa=kappa,
            IC=IC,
        )

    def _compute_costs(self, kernel: KernelInvariants) -> CostClosures:
        """Compute cost closures."""
        # Drift cost: Γ(ω) = ω^p / (1 - ω + ε)
        g_omega = gamma_omega(kernel.omega, self.p_exponent, self.epsilon)

        # Curvature cost: D_C = α·C
        d_c = cost_curvature(kernel.C, self.alpha)

        # Equator diagnostic
        phi_eq = equator_phi(kernel.omega, kernel.F, kernel.C)

        return CostClosures(
            gamma_omega=g_omega,
            D_C=d_c,
            D_omega=g_omega,
            equator_phi=phi_eq,
        )

    def _compute_seam(
        self,
        kernel: KernelInvariants,
        costs: CostClosures,
        prior_kappa: float,
        prior_IC: float,
        R_credit: float,
    ) -> SeamResult:
        """Compute seam accounting."""
        # Ledger change
        delta_kappa_ledger = kernel.kappa - prior_kappa
        I_ratio = kernel.IC / prior_IC if prior_IC > 0 else 0.0

        # Budget change
        delta_kappa_budget = compute_budget_delta_kappa(
            R=R_credit,
            tau_R=kernel.tau_R,
            D_omega=costs.D_omega,
            D_C=costs.D_C,
        )

        # Residual
        residual = compute_seam_residual(delta_kappa_budget, delta_kappa_ledger)

        # PASS/FAIL check
        passed, failures = check_seam_pass(
            residual=residual,
            tau_R=kernel.tau_R,
            I_ratio=I_ratio,
            delta_kappa=delta_kappa_ledger,
            tol_seam=self.tol_seam,
        )

        return SeamResult(
            delta_kappa_budget=delta_kappa_budget,
            delta_kappa_ledger=delta_kappa_ledger,
            residual=residual,
            passed=passed,
            failures=failures,
            R_credit=R_credit,
            I_ratio=I_ratio,
        )

    def _compute_gcd(self, kernel: KernelInvariants) -> GCDMetrics:
        """Compute GCD (Generative Collapse Dynamics) metrics."""
        # Energy potential: E = ω² + α·S + β·C²
        alpha, beta = 1.0, 0.5
        E_collapse = kernel.omega**2
        E_entropy = alpha * kernel.S
        E_curvature = beta * (kernel.C**2)
        E_potential = E_collapse + E_entropy + E_curvature

        # Collapse potential: Φ = ω · (1 + C)
        Phi_collapse = kernel.omega * (1 + kernel.C)

        # Generative flux: Φ_gen = Φ_collapse · (1 - S)
        Phi_gen = Phi_collapse * (1 - min(kernel.S, 1.0))

        # Field resonance: R = (1 - ω) · IC
        R_resonance = (1 - kernel.omega) * kernel.IC

        # Energy regime classification
        if E_potential < 0.01:
            energy_regime = "Low"
        elif E_potential < 0.05:
            energy_regime = "Medium"
        else:
            energy_regime = "High"

        return GCDMetrics(
            E_potential=E_potential,
            E_collapse=E_collapse,
            E_entropy=E_entropy,
            E_curvature=E_curvature,
            Phi_collapse=Phi_collapse,
            Phi_gen=Phi_gen,
            R_resonance=R_resonance,
            energy_regime=energy_regime,
        )

    def _compute_rcft(self, trajectory: FloatArray, kernel: KernelInvariants) -> RCFTMetrics:
        """Compute RCFT metrics from trajectory."""
        # Fractal dimension via box-counting
        D_fractal = self._box_counting_dimension(trajectory)

        # Fractal regime classification
        if D_fractal < 1.2:
            fractal_regime = "Smooth"
        elif D_fractal < 1.8:
            fractal_regime = "Wrinkled"
        else:
            fractal_regime = "Turbulent"

        # Recursive field: Ψ_r = Σ exp(-t/τ) · ω(t)
        n_points = len(trajectory)
        tau_decay = max(n_points // 4, 1)
        weights = np.exp(-np.arange(n_points) / tau_decay)
        Psi_recursive = float(np.sum(weights * np.abs(trajectory[:, 0] - 0.5)) / np.sum(weights))

        # Memory depth
        memory_depth = min(int(tau_decay * 2), n_points)

        # Basin strength: based on trajectory variance
        basin_strength = max(0.0, 1.0 - float(np.std(trajectory)))

        return RCFTMetrics(
            D_fractal=D_fractal,
            fractal_regime=fractal_regime,
            Psi_recursive=Psi_recursive,
            memory_depth=memory_depth,
            basin_strength=basin_strength,
        )

    def _compute_rcft_single(self, kernel: KernelInvariants) -> RCFTMetrics:
        """Compute RCFT metrics from single point (limited)."""
        # Estimate fractal dimension from curvature proxy
        D_fractal = 1.0 + kernel.C * 0.5

        if D_fractal < 1.2:
            fractal_regime = "Smooth"
        elif D_fractal < 1.8:
            fractal_regime = "Wrinkled"
        else:
            fractal_regime = "Turbulent"

        return RCFTMetrics(
            D_fractal=D_fractal,
            fractal_regime=fractal_regime,
            Psi_recursive=kernel.omega,
            memory_depth=1,
            basin_strength=1.0 - kernel.C,
        )

    def _box_counting_dimension(self, trajectory: FloatArray) -> float:
        """Compute box-counting fractal dimension."""
        if trajectory.ndim == 1:
            trajectory = trajectory.reshape(-1, 1)

        n_points, _n_dims = trajectory.shape
        if n_points < 3:
            return 1.0

        # Normalize trajectory
        extent = np.ptp(trajectory, axis=0)
        max_extent = np.max(extent)
        if max_extent < 1e-10:
            return 0.0

        trajectory_norm = (trajectory - trajectory.min(axis=0)) / max_extent

        # Box-counting
        eps_values = np.logspace(-2, 0, 10)
        box_counts = []

        for eps in eps_values:
            boxes = set()
            for point in trajectory_norm:
                box_idx = tuple((point // eps).astype(int))
                boxes.add(box_idx)
            box_counts.append(len(boxes))

        # Linear regression in log-log space
        log_eps = np.log(1 / eps_values)
        log_counts = np.log(box_counts)

        # Least squares fit
        A = np.vstack([log_eps, np.ones(len(log_eps))]).T
        slope, _ = np.linalg.lstsq(A, log_counts, rcond=None)[0]

        return float(np.clip(slope, 0.0, 3.0))

    def _compute_uncertainty(
        self, c: FloatArray, w: FloatArray, variances: list[float] | FloatArray
    ) -> UncertaintyBounds:
        """Compute uncertainty bounds via delta-method."""
        v = np.array(variances, dtype=np.float64)

        # Gradients
        grad_F = w
        grad_omega = -w
        grad_kappa = w / c

        # Entropy gradient: ∂S/∂c_i = w_i · ln((1-c_i)/c_i)
        grad_S = w * np.log((1 - c) / c)

        # Curvature gradient (approximate)
        n = len(c)
        c_mean = np.mean(c)
        grad_C = (c - c_mean) / (n * 0.5 * np.std(c) + 1e-10)

        # Variance propagation: Var(T) = Σ (∂T/∂c_i)² · Var(c_i)
        var_F = float(np.sum(grad_F**2 * v))
        var_omega = float(np.sum(grad_omega**2 * v))
        var_kappa = float(np.sum(grad_kappa**2 * v))
        var_S = float(np.sum(grad_S**2 * v))
        var_C = float(np.sum(grad_C**2 * v))

        return UncertaintyBounds(
            var_F=var_F,
            var_omega=var_omega,
            var_S=var_S,
            var_kappa=var_kappa,
            var_C=var_C,
            std_F=math.sqrt(var_F),
            std_omega=math.sqrt(var_omega),
            std_S=math.sqrt(var_S),
            std_kappa=math.sqrt(var_kappa),
            std_C=math.sqrt(var_C),
        )

    def _compute_tau_R_from_trajectory(
        self, trajectory: FloatArray, current: FloatArray, eta: float = 0.001, H_rec: int = 64
    ) -> float:
        """Compute return time from trajectory."""
        if trajectory.ndim == 1:
            trajectory = trajectory.reshape(-1, 1)
            current = current.reshape(1, -1)

        n_points = len(trajectory)
        for delta_t in range(1, min(H_rec + 1, n_points)):
            past = trajectory[n_points - delta_t - 1]
            distance = float(np.linalg.norm(current - past))
            if distance < eta:
                return float(delta_t)

        return float("inf")  # INF_REC

    def _compute_diagnostics(self, kernel: KernelInvariants, c: FloatArray, w: FloatArray) -> dict[str, Any]:
        """Compute diagnostic information."""
        return {
            "n_coordinates": len(c),
            "c_min": float(np.min(c)),
            "c_max": float(np.max(c)),
            "c_mean": float(np.mean(c)),
            "c_std": float(np.std(c)),
            "am_gm_gap": kernel.F - kernel.IC,  # Lemma 4
            "identity_check": abs(kernel.F - (1 - kernel.omega)) < 1e-10,  # F = 1 - ω
            "ic_exp_kappa_check": abs(kernel.IC - math.exp(kernel.kappa)) < 1e-10,
            "is_homogeneous": float(np.std(c)) < 1e-10,
        }

    def _compute_hash(self, c: FloatArray, w: FloatArray) -> str:
        """Compute SHA256 hash of inputs."""
        data = f"c={c.tobytes().hex()},w={w.tobytes().hex()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def _compute_ss1m(self, result: UniversalResult) -> SS1MTriad:
        """Compute SS1M human-verifiable checksum (mod-97 triads)."""
        # Concatenate key values
        data = f"{result.kernel.omega:.6f},{result.kernel.F:.6f},{result.kernel.IC:.6f},{result.regime}"
        full_hash = hashlib.sha256(data.encode()).hexdigest()

        # Extract three segments and compute mod-97
        h1 = int(full_hash[0:16], 16) % 97
        h2 = int(full_hash[16:32], 16) % 97
        h3 = int(full_hash[32:48], 16) % 97

        return SS1MTriad(
            C1=f"{h1:02d}",
            C2=f"{h2:02d}",
            C3=f"{h3:02d}",
            full_hash=full_hash,
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def compute_kernel(
    coordinates: list[float],
    weights: list[float] | None = None,
) -> KernelInvariants:
    """
    Quick computation of Tier-1 kernel invariants.

    Args:
        coordinates: Bounded coordinates c_i ∈ [0,1]
        weights: Channel weights (default: uniform)

    Returns:
        KernelInvariants with the seven core metrics
    """
    calc = UniversalCalculator()
    result = calc.compute_all(coordinates, weights, mode=ComputationMode.MINIMAL)
    return result.kernel


def compute_regime(
    coordinates: list[float],
    weights: list[float] | None = None,
) -> str:
    """
    Quick regime classification.

    Args:
        coordinates: Bounded coordinates c_i ∈ [0,1]
        weights: Channel weights (default: uniform)

    Returns:
        Regime string: STABLE, WATCH, COLLAPSE, or CRITICAL
    """
    calc = UniversalCalculator()
    result = calc.compute_all(coordinates, weights, mode=ComputationMode.MINIMAL)
    return result.regime


def compute_full(
    coordinates: list[float],
    weights: list[float] | None = None,
    trajectory: FloatArray | None = None,
) -> UniversalResult:
    """
    Full computation with all metrics.

    Args:
        coordinates: Bounded coordinates c_i ∈ [0,1]
        weights: Channel weights (default: uniform)
        trajectory: Historical trajectory for RCFT

    Returns:
        UniversalResult with all computed metrics
    """
    calc = UniversalCalculator()
    return calc.compute_all(coordinates, weights, trajectory=trajectory, mode=ComputationMode.FULL)


# =============================================================================
# CLI INTERFACE
# =============================================================================


def main() -> None:
    """Command-line interface for the universal calculator."""
    import argparse

    parser = argparse.ArgumentParser(description="UMCP Universal Calculator - Compute all metrics in one call")
    parser.add_argument(
        "-c",
        "--coordinates",
        type=str,
        required=True,
        help="Comma-separated coordinates (e.g., '0.9,0.85,0.92')",
    )
    parser.add_argument(
        "-w",
        "--weights",
        type=str,
        default=None,
        help="Comma-separated weights (default: uniform)",
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        choices=["minimal", "standard", "full", "rcft"],
        default="standard",
        help="Computation mode (default: standard)",
    )
    parser.add_argument(
        "--tau-R",
        type=float,
        default=None,
        help="Return time (default: INF_REC)",
    )
    parser.add_argument(
        "--prior-kappa",
        type=float,
        default=None,
        help="Prior κ for seam accounting",
    )
    parser.add_argument(
        "--prior-IC",
        type=float,
        default=None,
        help="Prior IC for seam accounting",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )

    args = parser.parse_args()

    # Parse coordinates
    coordinates = [float(x.strip()) for x in args.coordinates.split(",")]

    # Parse weights
    weights = None
    if args.weights:
        weights = [float(x.strip()) for x in args.weights.split(",")]

    # Map mode string to enum
    mode_map = {
        "minimal": ComputationMode.MINIMAL,
        "standard": ComputationMode.STANDARD,
        "full": ComputationMode.FULL,
        "rcft": ComputationMode.RCFT,
    }

    # Compute
    calc = UniversalCalculator()
    result = calc.compute_all(
        coordinates=coordinates,
        weights=weights,
        tau_R=args.tau_R,
        prior_kappa=args.prior_kappa,
        prior_IC=args.prior_IC,
        mode=mode_map[args.mode],
    )

    # Output
    if args.json:
        print(result.to_json())
    else:
        print(result.summary())


if __name__ == "__main__":
    main()

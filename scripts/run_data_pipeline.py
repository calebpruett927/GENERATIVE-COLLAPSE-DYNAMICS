#!/usr/bin/env python3
"""
UMCP Real Data Ingestion and Δκ Optimization

This script demonstrates how to:
1. Ingest real time-series data
2. Compute kernel invariants
3. Track Δκ evolution
4. Achieve neutral/positive Δκ through return dynamics

The Physics of Positive Δκ:
---------------------------
Δκ_budget = R·τ_R - (D_ω + D_C)

For Δκ > 0 (integrity gain):
  R·τ_R > D_ω + D_C

This requires:
  - τ_R finite (system actually returns to prior states)
  - Low ω (minimal drift from baseline)
  - Low C (minimal curvature/dispersion)
  - Sufficient R (return credit rate, typically 0.01)

Usage:
    python scripts/run_data_pipeline.py --data <csv_file>
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Contract parameters (from contract.yaml)
LAMBDA = 0.2  # Drift cost coefficient
ALPHA = 1.0  # Curvature cost coefficient
R = 0.01  # Return credit rate
ETA = 0.001  # Return threshold η
TOL_SEAM = 0.005  # Seam tolerance


@dataclass
class KernelState:
    """Tier-1 kernel invariants at a single timestep."""

    t: int
    omega: float  # Drift ω
    F: float  # Fidelity F = 1 - ω
    S: float  # Stiffness
    C: float  # Curvature
    kappa: float  # Log-integrity κ
    IC: float  # Integrity composite IC = exp(κ)
    tau_R: float  # Return time (inf = no return)

    def to_dict(self) -> dict[str, Any]:
        return {
            "t": self.t,
            "omega": self.omega,
            "F": self.F,
            "S": self.S,
            "C": self.C,
            "kappa": self.kappa,
            "IC": self.IC,
            "tau_R": self.tau_R if np.isfinite(self.tau_R) else "INF_REC",
        }


@dataclass
class BudgetResult:
    """Budget identity calculation result."""

    delta_kappa_ledger: float  # Observed κ(t1) - κ(t0)
    delta_kappa_budget: float  # R·τ_R - (D_ω + D_C)
    residual: float  # budget - ledger
    D_omega: float  # Drift cost
    D_C: float  # Curvature cost
    return_credit: float  # R·τ_R

    @property
    def is_positive(self) -> bool:
        return self.delta_kappa_ledger > 0

    @property
    def is_neutral(self) -> bool:
        return abs(self.delta_kappa_ledger) < TOL_SEAM


def load_weights(root: Path) -> np.ndarray:
    """Load weights from weights.csv."""
    weights_path = root / "weights.csv"
    with open(weights_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return np.array([float(r["weight"]) for r in rows])


def compute_kernel(
    coords: np.ndarray,
    weights: np.ndarray,
    history: list[np.ndarray],
    eta: float = ETA,
) -> KernelState:
    """
    Compute Tier-1 kernel invariants from coordinate vector.

    Args:
        coords: Current coordinate vector c_i(t) ∈ [0,1]^n
        weights: Weight vector w_i with Σw_i = 1
        history: List of prior coordinate vectors for return detection
        eta: Return threshold

    Returns:
        KernelState with all invariants
    """
    t = len(history)
    _ = len(coords)  # Coords dimension (unused but validated)

    # Clip to valid range (face policy)
    coords = np.clip(coords, 1e-10, 1.0)

    # ω: Drift (distance from unity)
    omega = np.mean(np.abs(1.0 - coords))

    # F: Fidelity
    F = 1.0 - omega

    # S: Stiffness (variance of coordinates)
    s_val = np.std(coords) / 0.5  # Normalized by max std on [0,1]
    s_val = min(float(s_val), 1.0)

    # C: Curvature (population std normalized)
    c_val = np.std(coords, ddof=0) / 0.5
    c_val = min(float(c_val), 1.0)

    # κ: Log-integrity
    log_coords = np.log(coords)
    kappa = np.dot(weights, log_coords)

    # IC: Integrity composite
    IC = np.exp(kappa)

    # τ_R: Return time detection
    tau_R = float("inf")
    if history:
        for lag in range(1, min(len(history) + 1, 64)):  # H_rec = 64
            past_idx = len(history) - lag
            if past_idx >= 0:
                past = history[past_idx]
                dist = np.linalg.norm(coords - past)
                if dist <= eta:
                    tau_R = float(lag)
                    break

    return KernelState(
        t=t,
        omega=float(omega),
        F=float(F),
        S=s_val,
        C=c_val,
        kappa=float(kappa),
        IC=float(IC),
        tau_R=tau_R,
    )


def compute_budget(
    kappa_t0: float,
    kappa_t1: float,
    omega: float,
    C: float,
    tau_R: float,
) -> BudgetResult:
    """
    Compute budget identity and residual.

    Δκ_budget = R·τ_R - (D_ω + D_C)
    residual s = Δκ_budget - Δκ_ledger
    """
    delta_kappa_ledger = kappa_t1 - kappa_t0

    D_omega = LAMBDA * omega
    D_C = ALPHA * C

    if np.isfinite(tau_R):
        return_credit = R * tau_R
    else:
        return_credit = 0.0  # Typed censoring: INF_REC → no credit

    delta_kappa_budget = return_credit - (D_omega + D_C)
    residual = delta_kappa_budget - delta_kappa_ledger

    return BudgetResult(
        delta_kappa_ledger=delta_kappa_ledger,
        delta_kappa_budget=delta_kappa_budget,
        residual=residual,
        D_omega=D_omega,
        D_C=D_C,
        return_credit=return_credit,
    )


def simulate_returning_dynamics(
    n_steps: int = 100,
    n_coords: int = 3,
    noise_level: float = 0.01,
    return_tendency: float = 0.8,
) -> list[np.ndarray]:
    """
    Simulate coordinate time series with return dynamics.

    For positive Δκ, we need:
    - Coordinates that return to prior values (low τ_R)
    - Low drift (stay near baseline)
    - Low curvature (homogeneous coordinates)

    Args:
        n_steps: Number of timesteps
        n_coords: Number of coordinates
        noise_level: Noise amplitude
        return_tendency: How strongly coords return to mean (0-1)
    """
    np.random.seed(42)

    # Start near 1.0 (high integrity baseline)
    baseline: np.ndarray = 0.95 * np.ones(n_coords)  # type: ignore[assignment]
    coords_history: list[np.ndarray] = []

    current: np.ndarray = baseline.copy()

    for _ in range(n_steps):
        # Mean-reverting dynamics (encourages return)
        noise: np.ndarray = noise_level * np.random.randn(n_coords)  # type: ignore[assignment]
        drift_to_baseline: np.ndarray = return_tendency * (baseline - current)  # type: ignore[assignment]

        current = current + drift_to_baseline + noise
        current = np.clip(current, 0.01, 0.99)

        coords_history.append(current.copy())

    return coords_history


def run_pipeline(
    data: list[np.ndarray],
    weights: np.ndarray,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """
    Run full UMCP pipeline on coordinate data.

    Returns ledger of kernel states and budget calculations.
    """
    history: list[np.ndarray] = []
    states: list[KernelState] = []
    budgets: list[BudgetResult] = []
    ledger: list[dict[str, Any]] = []

    for i, coords in enumerate(data):
        # Compute kernel state
        state = compute_kernel(coords, weights, history, eta=ETA)
        states.append(state)
        history.append(coords)

        # Compute budget if we have prior state
        if len(states) >= 2:
            prev = states[-2]
            budget = compute_budget(
                kappa_t0=prev.kappa,
                kappa_t1=state.kappa,
                omega=state.omega,
                C=state.C,
                tau_R=state.tau_R,
            )
            budgets.append(budget)

            entry: dict[str, Any] = {
                "t": state.t,
                "kappa": state.kappa,
                "IC": state.IC,
                "omega": state.omega,
                "tau_R": state.tau_R,
                "delta_kappa": budget.delta_kappa_ledger,
                "D_omega": budget.D_omega,
                "D_C": budget.D_C,
                "return_credit": budget.return_credit,
                "residual": budget.residual,
            }
            ledger.append(entry)

            if verbose and i % 10 == 0:
                dk = budget.delta_kappa_ledger
                sign = "+" if dk > 0 else ""
                print(f"  t={i:3d}: Δκ={sign}{dk:>8.5f}, τ_R={state.tau_R:>5.1f}, ω={state.omega:.4f}")

    return ledger


def analyze_results(ledger: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze Δκ distribution and return dynamics."""
    if not ledger:
        return {}

    delta_kappas = [e["delta_kappa"] for e in ledger]
    residuals = [e["residual"] for e in ledger]

    positive = sum(1 for dk in delta_kappas if dk > 0)
    neutral = sum(1 for dk in delta_kappas if abs(dk) < TOL_SEAM)
    negative = sum(1 for dk in delta_kappas if dk < 0)

    return {
        "total_steps": len(ledger),
        "cumulative_delta_kappa": sum(delta_kappas),
        "mean_delta_kappa": np.mean(delta_kappas),
        "std_delta_kappa": np.std(delta_kappas),
        "positive_steps": positive,
        "neutral_steps": neutral,
        "negative_steps": negative,
        "positive_ratio": positive / len(ledger),
        "mean_residual": np.mean(residuals),
        "max_abs_residual": max(abs(r) for r in residuals),
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run UMCP data pipeline")
    parser.add_argument("--data", default=None, help="Path to CSV data file")
    parser.add_argument("--simulate", action="store_true", help="Use simulated data")
    parser.add_argument("--steps", type=int, default=100, help="Simulation steps")
    parser.add_argument("--noise", type=float, default=0.01, help="Noise level")
    parser.add_argument(
        "--return-tendency", type=float, default=0.8, help="Return tendency (0-1, higher = more returns)"
    )
    args = parser.parse_args()

    # Find repo root
    root = Path.cwd()
    while root != root.parent:
        if (root / "pyproject.toml").exists():
            break
        root = root.parent

    print("=" * 70)
    print("UMCP Data Pipeline - Δκ Optimization")
    print("=" * 70)

    # Load weights
    weights = load_weights(root)
    print(f"Loaded weights: {weights}")
    print()

    # Get data
    if args.simulate or args.data is None:
        print(f"Simulating {args.steps} steps with return_tendency={args.return_tendency}")
        print(f"  noise_level={args.noise}")
        data = simulate_returning_dynamics(
            n_steps=args.steps,
            n_coords=len(weights),
            noise_level=args.noise,
            return_tendency=args.return_tendency,
        )
    else:
        # Load from CSV
        data_path = Path(args.data)
        with open(data_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Extract coordinate columns
        coord_cols = [c for c in rows[0].keys() if c.startswith("c_")]
        data: list[np.ndarray] = []
        for row in rows:
            coords = np.array([float(row[c]) for c in coord_cols])
            data.append(coords)
        print(f"Loaded {len(data)} rows from {data_path}")

    print()
    print("Running pipeline...")
    print("-" * 70)

    ledger = run_pipeline(data, weights, verbose=True)

    print("-" * 70)
    print()

    # Analyze
    analysis = analyze_results(ledger)

    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Total steps:           {analysis['total_steps']}")
    print(f"  Cumulative Δκ:         {analysis['cumulative_delta_kappa']:+.6f}")
    print(f"  Mean Δκ:               {analysis['mean_delta_kappa']:+.6f}")
    print(f"  Std Δκ:                {analysis['std_delta_kappa']:.6f}")
    print()
    print(f"  Positive Δκ steps:     {analysis['positive_steps']} ({analysis['positive_ratio']*100:.1f}%)")
    print(f"  Neutral Δκ steps:      {analysis['neutral_steps']}")
    print(f"  Negative Δκ steps:     {analysis['negative_steps']}")
    print()
    print(f"  Mean residual:         {analysis['mean_residual']:+.6f}")
    print(f"  Max |residual|:        {analysis['max_abs_residual']:.6f}")
    print()

    # Verdict
    if analysis["cumulative_delta_kappa"] > 0:
        print("  ✅ POSITIVE Δκ ACHIEVED - System gained integrity")
    elif abs(analysis["cumulative_delta_kappa"]) < TOL_SEAM * analysis["total_steps"]:
        print("  ⚖️  NEUTRAL Δκ - System maintained integrity")
    else:
        print("  ⚠️  NEGATIVE Δκ - System lost integrity")
        print("     To improve: increase return_tendency, decrease noise")

    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())

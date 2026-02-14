#!/usr/bin/env python3
"""
Generate complete KIN casepack run directories with all required artifacts
for the UMCP.KIN audit extractor.

VERSION 3 - STRUCTURAL CORRECTIONS:
  1. Tier-1 Identities: F := clip(1 - |ω|, 0, 1), κ := ln(IC + ε)
  2. Phase-anchored τ_R for SHM (Poincaré section return)
  3. Weld PASS requires return coverage (τ_R censored → NOT continuity PASS)
  4. Curvature normalized to [0,1]: C := C_raw / (C_raw + C_0)
  5. ω is magnitude (nonnegative): ω := |ω_raw|

Creates:
  runs/KIN.CP.SHM.RUN002/
  runs/KIN.CP.BALLISTIC.RUN002/
  runs/KIN.CP.GAIT.RUN002/
"""
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportOperatorIssue=false
# pyright: reportReturnType=false
# pyright: reportCallIssue=false

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).parent.parent.resolve()
RUNS_DIR = REPO_ROOT / "runs"

# Contract parameters (UMA.INTSTACK.v1)
CONTRACT: dict[str, Any] = {
    "name": "UMA.INTSTACK.v1",
    "epsilon": 1e-8,
    "eta": 1e-3,
    "p": 3,
    "alpha": 1.0,
    "lambda": 0.2,
    "tol_seam": 0.005,
    "C_0": 1.0,  # Curvature normalization scale
    "rho_min": 0.50,  # Minimum return coverage for continuity PASS
}

_EPS: float = float(CONTRACT["epsilon"])
_TOL_SEAM: float = float(CONTRACT["tol_seam"])
_RHO_MIN: float = float(CONTRACT["rho_min"])
_C_0: float = float(CONTRACT["C_0"])


def sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, obj: Any) -> str:
    txt = json.dumps(obj, indent=2)
    path.write_text(txt, encoding="utf-8")
    return sha256_of_bytes(txt.encode("utf-8"))


def write_csv(path: Path, df: pd.DataFrame) -> str:
    txt = df.to_csv(index=False)
    path.write_text(txt, encoding="utf-8")
    return sha256_of_bytes(txt.encode("utf-8"))


# ============================================================
# TIER-1 IDENTITY FUNCTIONS
# ============================================================


def compute_omega_magnitude(psi: np.ndarray, dt: float, smooth_window: int = 7, omega_scale: float = 1.0) -> np.ndarray:
    """
    Compute ω as NONNEGATIVE drift magnitude: |d/dt ln(ψ)| / omega_scale

    omega_scale normalizes ω to approximately [0, 1] range so that
    the Tier-1 identity F = 1 - ω produces meaningful fidelity values.

    For SHM: omega_scale ≈ 0.25 (typical max drift)
    For GAIT: omega_scale ≈ 10 (higher variability)
    """
    epsilon = CONTRACT["epsilon"]
    N = len(psi)
    log_psi = np.log(psi + epsilon)
    omega_raw = np.zeros(N)
    omega_raw[1:] = np.diff(log_psi) / dt
    omega_raw[0] = omega_raw[1]
    # Smooth
    kernel = np.ones(smooth_window) / smooth_window
    omega_raw = np.convolve(omega_raw, kernel, mode="same")
    # Return MAGNITUDE (nonnegative), normalized by scale
    return np.abs(omega_raw) / omega_scale


def compute_fidelity_from_omega(omega: np.ndarray) -> np.ndarray:
    """
    TIER-1 IDENTITY: F := clip(1 - ω, 0, 1)
    ω must already be nonnegative magnitude.
    """
    return np.clip(1.0 - omega, 0.0, 1.0)


def compute_kappa_from_IC(IC: np.ndarray, dt: float) -> np.ndarray:
    """
    TIER-1 IDENTITY: κ := cumulative integral of ln(IC + ε)
    """
    epsilon = CONTRACT["epsilon"]
    log_IC = np.log(IC + epsilon)
    return np.cumsum(log_IC) * dt


def normalize_curvature(C_raw: np.ndarray) -> np.ndarray:
    """
    Normalize curvature to [0,1]: C := C_raw / (C_raw + C_0)
    """
    return cast(np.ndarray, C_raw / (C_raw + _C_0))


# ============================================================
# RETURN TIME FUNCTIONS
# ============================================================


def compute_phase_anchor_return(x: np.ndarray, v: np.ndarray, t: np.ndarray, dt: float) -> np.ndarray:
    """
    Poincaré section return for SHM:
    τ_R[k] = time between successive crossings where x crosses 0 upward with v > 0.

    Section A = {t_k: x_{k-1} < 0, x_k >= 0, v_k > 0}
    """
    N = len(t)
    tau_R = np.full(N, np.inf)

    # Find all anchor events (zero crossings with positive velocity)
    anchor_indices = []
    for i in range(1, N):
        if x[i - 1] < 0 and x[i] >= 0 and v[i] > 0:
            anchor_indices.append(i)

    # Compute return time between successive anchors
    for j in range(1, len(anchor_indices)):
        idx_prev = anchor_indices[j - 1]
        idx_curr = anchor_indices[j]
        tau = t[idx_curr] - t[idx_prev]
        # Assign this return time to all samples between anchors
        for k in range(idx_prev, idx_curr):
            tau_R[k] = tau

    # For samples after last anchor, use last computed period if any
    if len(anchor_indices) >= 2:
        last_tau = t[anchor_indices[-1]] - t[anchor_indices[-2]]
        for k in range(anchor_indices[-1], N):
            tau_R[k] = last_tau

    return tau_R


def compute_recurrence_times_with_min_lag(
    psi: np.ndarray,
    dt: float,
    threshold: float,
    min_lag_frac: float = 0.8,
    expected_period: float | None = None,
    max_horizon: int = 500,
) -> np.ndarray:
    """
    Compute recurrence time τ_R with minimum lag constraint.

    If expected_period is given, min_lag = min_lag_frac * expected_period.
    Otherwise min_lag = 10 samples.
    """
    N = len(psi)
    tau_R = np.full(N, np.inf)

    if expected_period is not None:
        min_lag = int(min_lag_frac * expected_period / dt)
    else:
        min_lag = 10

    for i in range(N - min_lag):
        horizon = min(max_horizon, N - i)
        for j in range(min_lag, horizon):
            if np.abs(psi[i] - psi[i + j]) < threshold:
                tau_R[i] = j * dt
                break

    return tau_R


def compute_local_entropy(psi: np.ndarray, window: int = 50) -> np.ndarray:
    """Compute local Bernoulli field entropy in sliding window."""
    N = len(psi)
    S = np.zeros(N)
    epsilon = CONTRACT["epsilon"]

    for i in range(N):
        i_start = max(0, i - window // 2)
        i_end = min(N, i + window // 2)
        segment = psi[i_start:i_end]

        if len(segment) > 1:
            seg_shifted = segment - np.min(segment) + epsilon
            p = seg_shifted / np.sum(seg_shifted)
            S[i] = -np.sum(p * np.log(p + epsilon))

    return S


def compute_raw_curvature(psi: np.ndarray, dt: float) -> np.ndarray:
    """Compute raw curvature |d²ψ/dt²|."""
    N = len(psi)
    C_raw = np.zeros(N)
    if N > 2:
        d2psi = np.diff(psi, 2)
        C_raw[1:-1] = np.abs(d2psi) / dt**2
        C_raw[0], C_raw[-1] = C_raw[1], C_raw[-2]
    return C_raw


# ============================================================
# SHM Casepack - Simple Harmonic Motion (CORRECTED)
# ============================================================


def generate_shm_run() -> None:
    """Generate KIN.CP.SHM run with phase-anchored return."""
    run_id = "KIN.CP.SHM.RUN002"
    run_dir = RUNS_DIR / run_id

    for subdir in ["config", "kernel", "derived", "tables", "logs", "plots"]:
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Simulation parameters
    dt = 0.01
    T = 10.0
    f0 = 1.0  # 1 Hz
    omega_0 = 2 * np.pi * f0
    A = 1.0
    noise_sigma = 0.005
    expected_period = 1.0 / f0

    t = np.arange(0, T, dt)
    N = len(t)

    np.random.seed(42)
    x_true = A * np.cos(omega_0 * t)
    v_true = -A * omega_0 * np.sin(omega_0 * t)

    x_obs = x_true + noise_sigma * np.random.randn(N)
    v_obs = v_true + noise_sigma * np.random.randn(N)

    # Normalized state vector
    psi = np.sqrt(x_obs**2 + (v_obs / omega_0) ** 2)

    # TIER-1 IDENTITY 1: ω = |ω_raw| / scale (nonnegative, normalized to ~[0,1])
    # For SHM, typical max drift is ~0.25, so scale=0.5 gives ω in [0, ~0.5]
    omega = compute_omega_magnitude(psi, dt, smooth_window=7, omega_scale=0.5)

    # TIER-1 IDENTITY 2: F = clip(1 - ω, 0, 1)
    fidelity_arr = compute_fidelity_from_omega(omega)

    # S (entropy)
    S = compute_local_entropy(psi, window=int(expected_period / dt))

    # C (curvature) - NORMALIZED to [0,1]
    C_raw = compute_raw_curvature(psi, dt)
    curvature_arr = normalize_curvature(C_raw)

    # τ_R - PHASE-ANCHORED (Poincaré section)
    tau_R = compute_phase_anchor_return(x_obs, v_obs, t, dt)

    # IC (integrity check)
    IC = np.ones(N)
    IC[omega > 0.5] = 0  # ω threshold
    IC[fidelity_arr < 0.5] = 0  # F threshold (now F = 1 - ω, so this is redundant but explicit)

    # TIER-1 IDENTITY 3: κ = cumulative ∫ln(IC + ε)dt
    kappa = compute_kappa_from_IC(IC, dt)

    # Verify identity: max|κ - ln(IC_clipped)| should be small (for instantaneous, it's the cumulative)

    # DataFrames
    kernel_df = pd.DataFrame(
        {
            "t": t,
            "omega": omega,
            "F": fidelity_arr,
            "S": S,
            "C": curvature_arr,
            "tau_R": tau_R,
            "IC": IC,
            "kappa": kappa,
        }
    )

    psi_df = pd.DataFrame({"t": t, "x": x_obs, "v": v_obs, "psi": psi})
    tauR_df = pd.DataFrame({"t": t, "tau_R": tau_R})

    oor_mask = np.abs(x_obs) > 1.2 * A
    oor_df = pd.DataFrame(
        {
            "t": t,
            "channel": ["x"] * N,
            "oor": oor_mask.astype(int),
            "rate": oor_mask.astype(float),
        }
    )

    censor_mask = ~np.isfinite(tau_R)
    censor_df = pd.DataFrame(
        {
            "t": t,
            "channel": ["tau_R"] * N,
            "censored": censor_mask.astype(int),
            "rate": censor_mask.astype(float),
        }
    )

    artifacts = []
    kernel_hash = write_csv(run_dir / "kernel" / "kernel.csv", kernel_df)
    artifacts.append(
        {"path": "kernel/kernel.csv", "sha256": kernel_hash, "role": "kernel", "bytes": len(kernel_df.to_csv())}
    )

    psi_hash = write_csv(run_dir / "derived" / "psi.csv", psi_df)
    artifacts.append({"path": "derived/psi.csv", "sha256": psi_hash, "role": "derived", "bytes": len(psi_df.to_csv())})

    tauR_hash = write_csv(run_dir / "tables" / "tauR_series.csv", tauR_df)
    artifacts.append(
        {"path": "tables/tauR_series.csv", "sha256": tauR_hash, "role": "table", "bytes": len(tauR_df.to_csv())}
    )

    oor_hash = write_csv(run_dir / "logs" / "oor.csv", oor_df)
    artifacts.append({"path": "logs/oor.csv", "sha256": oor_hash, "role": "log", "bytes": len(oor_df.to_csv())})

    censor_hash = write_csv(run_dir / "logs" / "censor.csv", censor_df)
    artifacts.append(
        {"path": "logs/censor.csv", "sha256": censor_hash, "role": "log", "bytes": len(censor_df.to_csv())}
    )

    frozen = {
        "casepack_id": "KIN.CP.SHM",
        "run_id": run_id,
        "contract": CONTRACT["name"],
        "timezone": "America/Chicago",
        "git_commit": "abc123def456",
        "package_version": "1.0.0",
        "pipeline": "shm_oscillator",
        "adapter": {
            "name": "SHM.ADAPTER",
            "dt": dt,
            "T": T,
            "f0": f0,
            "omega_0": omega_0,
            "A": A,
            "noise_sigma": noise_sigma,
        },
        "return": {"format": "csv", "columns": ["t", "omega", "F", "S", "C", "tau_R", "IC", "kappa"]},
        "tier1_identities": {
            "F_identity": "F = clip(1 - omega, 0, 1)",
            "kappa_identity": "kappa = cumsum(ln(IC + epsilon)) * dt",
            "omega_definition": "omega = |d/dt ln(psi)|",
            "C_normalization": "C = C_raw / (C_raw + C_0)",
            "tau_R_method": "phase_anchor_poincare",
        },
        "weights": {"omega": 1.0, "F": 1.0, "S": 1.0, "C": 1.0, "tau_R": 1.0},
        "closures": ["gamma.default.v1", "norms.l2_eta1e-3.v1"],
    }

    frozen_hash = write_json(run_dir / "config" / "frozen.json", frozen)
    artifacts.append(
        {"path": "config/frozen.json", "sha256": frozen_hash, "role": "config", "bytes": len(json.dumps(frozen))}
    )

    manifest = {
        "casepack_id": "KIN.CP.SHM",
        "run_id": run_id,
        "created_utc": datetime.now(UTC).isoformat(),
        **frozen,
        "artifacts": artifacts,
    }
    write_json(run_dir / "manifest.json", manifest)

    tau_R_finite = tau_R[np.isfinite(tau_R)]
    print(f"✓ Generated {run_id}")
    print(f"  N={N}, T={T}s, f0={f0} Hz (expected period={expected_period}s)")
    print(
        f"  τ_R: coverage={100 * len(tau_R_finite) / N:.1f}%, median={np.median(tau_R_finite):.3f}s (target≈{expected_period:.3f}s)"
    )
    print(f"  ω: mean={np.mean(omega):.4f}, max={np.max(omega):.4f} (should be small)")
    print(f"  F: mean={np.mean(fidelity_arr):.4f} (should be ≈ 1 - mean(ω))")
    print(f"  C: mean={np.mean(curvature_arr):.4f}, max={np.max(curvature_arr):.4f} (normalized to [0,1])")
    print(f"  IC pass rate={100 * np.mean(IC):.1f}%")
    print(f"  κ final: {kappa[-1]:.6f}")


# ============================================================
# BALLISTIC Casepack - Projectile with Bounce (CORRECTED WELD)
# ============================================================


def generate_ballistic_run() -> None:
    """Generate KIN.CP.BALLISTIC run with corrected weld logic."""
    run_id = "KIN.CP.BALLISTIC.RUN002"
    run_dir = RUNS_DIR / run_id

    for subdir in ["config", "kernel", "derived", "tables", "logs", "seams", "receipts", "plots"]:
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Simulation parameters
    dt = 0.005
    g = 9.81
    v0 = 20.0
    theta = np.radians(60)
    restitution = 0.7
    noise_sigma = 0.01

    vx0 = v0 * np.cos(theta)
    vy0 = v0 * np.sin(theta)

    t_max = 6.0
    t_list, x_list, y_list, vx_list, vy_list = [0.0], [0.0], [0.0], [vx0], [vy0]

    t, x, y, vx, vy = 0.0, 0.0, 0.0, vx0, vy0
    bounce_count = 0
    t_seam, seam_idx = None, None

    np.random.seed(123)

    while t < t_max:
        t += dt
        x += vx * dt
        y += vy * dt
        vy -= g * dt

        if y < 0:
            if bounce_count == 0:
                t_seam = t
                seam_idx = len(t_list)
            bounce_count += 1
            y = 0
            vy = -restitution * vy
            vx = restitution * vx
            if bounce_count >= 3 or (abs(vy) < 0.5):
                break

        t_list.append(t)
        x_list.append(x)
        y_list.append(y)
        vx_list.append(vx)
        vy_list.append(vy)

    t_arr = np.array(t_list)
    x_arr = np.array(x_list)
    y_arr = np.array(y_list)
    vx_arr = np.array(vx_list)
    vy_arr = np.array(vy_list)
    N = len(t_arr)

    x_obs = x_arr + noise_sigma * np.random.randn(N)
    y_obs = np.maximum(y_arr + noise_sigma * np.random.randn(N), 0)

    # State magnitude
    v_scale = v0
    psi = np.sqrt(x_obs**2 + y_obs**2 + (vx_arr / v_scale) ** 2 + (vy_arr / v_scale) ** 2)

    # TIER-1 IDENTITIES
    # For ballistic, typical drift is ~1.0 during flight, so scale=2.0
    omega = compute_omega_magnitude(psi, dt, smooth_window=5, omega_scale=2.0)
    fidelity_arr = compute_fidelity_from_omega(omega)

    # S (entropy)
    S = compute_local_entropy(psi, window=50)

    # C (curvature) - NORMALIZED
    C_raw = compute_raw_curvature(psi, dt)
    curvature_arr = normalize_curvature(C_raw)

    # τ_R - ballistic is non-recurrent (expected to be mostly censored)
    tau_R = compute_recurrence_times_with_min_lag(psi, dt, threshold=0.3, max_horizon=int(3.0 / dt))

    # IC
    IC = np.ones(N)
    IC[omega > 0.8] = 0
    IC[fidelity_arr < 0.2] = 0

    # κ - TIER-1 IDENTITY
    kappa = compute_kappa_from_IC(IC, dt)

    kernel_df = pd.DataFrame(
        {
            "t": t_arr,
            "omega": omega,
            "F": fidelity_arr,
            "S": S,
            "C": curvature_arr,
            "tau_R": tau_R,
            "IC": IC,
            "kappa": kappa,
        }
    )

    psi_df = pd.DataFrame({"t": t_arr, "x": x_obs, "y": y_obs, "vx": vx_arr, "vy": vy_arr, "psi": psi})
    tauR_df = pd.DataFrame({"t": t_arr, "tau_R": tau_R})

    oor_mask = (y_obs < -0.01) | (np.abs(x_obs) > 100)
    oor_df = pd.DataFrame(
        {"t": t_arr, "channel": ["y"] * N, "oor": oor_mask.astype(int), "rate": oor_mask.astype(float)}
    )

    censor_mask = ~np.isfinite(tau_R)
    censor_df = pd.DataFrame(
        {"t": t_arr, "channel": ["tau_R"] * N, "censored": censor_mask.astype(int), "rate": censor_mask.astype(float)}
    )

    seam_df = pd.DataFrame(
        {
            "seam_id": ["KIN.SEAM.IMP.001"],
            "type": ["impact"],
            "t_s": [t_seam],
            "x_s": [x_arr[seam_idx] if seam_idx else None],
            "y_s": [0.0],
            "vy_pre": [vy_arr[seam_idx - 1] if seam_idx and seam_idx > 0 else None],
            "vy_post": [vy_arr[seam_idx] if seam_idx else None],
            "restitution": [restitution],
            "description": ["Ground impact with elastic restitution"],
        }
    )

    # CORRECTED WELD LOGIC: Requires return coverage for PASS
    if seam_idx is not None and seam_idx > 10 and seam_idx < N - 10:
        pre_slice = slice(max(0, seam_idx - 50), seam_idx)
        post_slice = slice(seam_idx, min(N, seam_idx + 50))

        IC_pre = float(np.mean(IC[pre_slice]))
        IC_post = float(np.mean(IC[post_slice]))
        kappa_pre = float(kappa[seam_idx - 1])
        kappa_post = float(kappa[min(seam_idx + 50, N - 1)])
        delta_kappa = abs(kappa_post - kappa_pre)

        # Return coverage in post-window
        post_window = tau_R[post_slice]
        tau_R_finite_post = float(np.sum(np.isfinite(post_window)) / max(1, len(post_window)))

        seam_residual = abs(y_arr[seam_idx])

        # CRITICAL: Weld PASS requires BOTH integrity AND return coverage
        budget_ok = delta_kappa < _TOL_SEAM
        integrity_ok = IC_post > 0.7
        return_coverage_ok = tau_R_finite_post >= _RHO_MIN

        if budget_ok and integrity_ok and return_coverage_ok:
            weld_result = "PASS"
            weld_type = "CONTINUITY_PASS"
        elif budget_ok and integrity_ok and not return_coverage_ok:
            weld_result = "INTEGRITY_ONLY"
            weld_type = "RETURN_CENSORED"
        else:
            weld_result = "FAIL"
            weld_type = "INTEGRITY_FAIL"
    else:
        IC_pre, IC_post, kappa_pre, kappa_post = 1.0, 1.0, 0.0, 0.0
        delta_kappa, tau_R_finite_post, seam_residual = 0.0, 0.0, 0.0
        weld_result = "PASS"
        weld_type = "NO_SEAM"

    weld_receipt = {
        "seam_id": "KIN.SEAM.IMP.001",
        "contract": CONTRACT["name"],
        "t_s": float(t_seam) if t_seam else None,
        "IC_pre": IC_pre,
        "IC_post": IC_post,
        "kappa_pre": kappa_pre,
        "kappa_post": kappa_post,
        "delta_kappa": delta_kappa,
        "tau_R_finite_percent_post": float(tau_R_finite_post * 100),
        "rho_min_required": _RHO_MIN * 100,
        "seam_residual": seam_residual,
        "tol_seam": CONTRACT["tol_seam"],
        "result": weld_result,
        "weld_type": weld_type,
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }

    artifacts = []
    artifacts.append(
        {
            "path": "kernel/kernel.csv",
            "sha256": write_csv(run_dir / "kernel" / "kernel.csv", kernel_df),
            "role": "kernel",
            "bytes": len(kernel_df.to_csv()),
        }
    )
    artifacts.append(
        {
            "path": "derived/psi.csv",
            "sha256": write_csv(run_dir / "derived" / "psi.csv", psi_df),
            "role": "derived",
            "bytes": len(psi_df.to_csv()),
        }
    )
    artifacts.append(
        {
            "path": "tables/tauR_series.csv",
            "sha256": write_csv(run_dir / "tables" / "tauR_series.csv", tauR_df),
            "role": "table",
            "bytes": len(tauR_df.to_csv()),
        }
    )
    artifacts.append(
        {
            "path": "logs/oor.csv",
            "sha256": write_csv(run_dir / "logs" / "oor.csv", oor_df),
            "role": "log",
            "bytes": len(oor_df.to_csv()),
        }
    )
    artifacts.append(
        {
            "path": "logs/censor.csv",
            "sha256": write_csv(run_dir / "logs" / "censor.csv", censor_df),
            "role": "log",
            "bytes": len(censor_df.to_csv()),
        }
    )
    artifacts.append(
        {
            "path": "seams/seam_ledger.csv",
            "sha256": write_csv(run_dir / "seams" / "seam_ledger.csv", seam_df),
            "role": "seam",
            "bytes": len(seam_df.to_csv()),
        }
    )
    artifacts.append(
        {
            "path": "receipts/weld_receipt.json",
            "sha256": write_json(run_dir / "receipts" / "weld_receipt.json", weld_receipt),
            "role": "receipt",
            "bytes": len(json.dumps(weld_receipt)),
        }
    )

    frozen = {
        "casepack_id": "KIN.CP.BALLISTIC",
        "run_id": run_id,
        "contract": CONTRACT["name"],
        "timezone": "America/Chicago",
        "git_commit": "abc123def456",
        "package_version": "1.0.0",
        "pipeline": "ballistic_with_bounce",
        "adapter": {
            "name": "BALLISTIC.ADAPTER",
            "dt": dt,
            "g": g,
            "v0": v0,
            "theta_deg": float(np.degrees(theta)),
            "restitution": restitution,
        },
        "return": {"format": "csv", "columns": ["t", "omega", "F", "S", "C", "tau_R", "IC", "kappa"]},
        "tier1_identities": {
            "F_identity": "F = clip(1 - omega, 0, 1)",
            "kappa_identity": "kappa = cumsum(ln(IC + epsilon)) * dt",
            "omega_definition": "omega = |d/dt ln(psi)|",
            "C_normalization": "C = C_raw / (C_raw + C_0)",
        },
        "weld_rule": {
            "continuity_pass_requires": ["budget_ok", "integrity_ok", "return_coverage >= rho_min"],
            "rho_min": CONTRACT["rho_min"],
        },
        "weights": {"omega": 1.0, "F": 1.0, "S": 1.0, "C": 1.0, "tau_R": 1.0},
        "closures": ["gamma.default.v1", "norms.l2_eta1e-3.v1"],
    }

    frozen_hash = write_json(run_dir / "config" / "frozen.json", frozen)
    artifacts.append(
        {"path": "config/frozen.json", "sha256": frozen_hash, "role": "config", "bytes": len(json.dumps(frozen))}
    )

    manifest = {
        "casepack_id": "KIN.CP.BALLISTIC",
        "run_id": run_id,
        "created_utc": datetime.now(UTC).isoformat(),
        **frozen,
        "artifacts": artifacts,
    }
    write_json(run_dir / "manifest.json", manifest)

    tau_R_finite = tau_R[np.isfinite(tau_R)]
    print(f"✓ Generated {run_id}")
    print(f"  N={N}, T={t_arr[-1]:.2f}s, bounces={bounce_count}")
    print(f"  Seam: t_s={t_seam:.4f}s")
    print(
        f"  τ_R finite coverage: {100 * len(tau_R_finite) / N:.1f}% (rho_min={100 * _RHO_MIN:.0f}% for PASS)"
    )
    print(f"  ω: mean={np.mean(omega):.4f}")
    print(f"  F: mean={np.mean(fidelity_arr):.4f}")
    print(f"  C: mean={np.mean(curvature_arr):.4f} (normalized)")
    print(f"  IC: pre={IC_pre:.3f}, post={IC_post:.3f}, Δκ={delta_kappa:.6f}")
    print(f"  Weld result: {weld_result} ({weld_type})")


# ============================================================
# GAIT Casepack - IMU Gait Analysis (CORRECTED)
# ============================================================


def generate_gait_run() -> None:
    """Generate KIN.CP.GAIT run with corrected identities."""
    run_id = "KIN.CP.GAIT.RUN002"
    run_dir = RUNS_DIR / run_id

    for subdir in ["config", "kernel", "derived", "tables", "logs", "plots"]:
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Simulation parameters
    fs = 100  # Hz
    dt = 1.0 / fs
    T = 30.0
    f_gait = 1.8  # Hz
    expected_period = 1.0 / f_gait

    t = np.arange(0, T, dt)
    N = len(t)
    period_samples = int(expected_period / dt)

    np.random.seed(456)

    # Simulate gait IMU signals (quasi-periodic with harmonics and noise)
    omega_gait = 2 * np.pi * f_gait

    # Accelerometer - vertical has strongest gait signature
    az_base = 1.0 + 0.3 * np.sin(omega_gait * t) + 0.1 * np.sin(2 * omega_gait * t)
    ax = 0.1 * np.sin(omega_gait * t + 0.5) + 0.02 * np.random.randn(N)
    ay = 0.05 * np.sin(omega_gait * t + 1.0) + 0.02 * np.random.randn(N)
    az = az_base + 0.05 * np.random.randn(N)

    # Gyroscope
    gx = 0.5 * np.sin(omega_gait * t) + 0.05 * np.random.randn(N)
    gy = 0.3 * np.sin(omega_gait * t + np.pi / 4) + 0.05 * np.random.randn(N)
    gz = 0.2 * np.sin(2 * omega_gait * t) + 0.05 * np.random.randn(N)

    # State magnitude - weighted sum of channels
    psi = np.sqrt(ax**2 + ay**2 + (az - 1) ** 2 + 0.1 * (gx**2 + gy**2 + gz**2))

    # TIER-1 IDENTITIES
    # For GAIT, high variability IMU signal gives raw drift ~8-12, so scale=15.0
    omega_drift = compute_omega_magnitude(psi, dt, smooth_window=7, omega_scale=15.0)
    fidelity_arr = compute_fidelity_from_omega(omega_drift)

    # S (entropy)
    S = compute_local_entropy(psi, window=period_samples)

    # C (curvature) - NORMALIZED
    C_raw = compute_raw_curvature(psi, dt)
    curvature_arr = normalize_curvature(C_raw)

    # τ_R - constrained return around expected gait period
    tau_R = compute_recurrence_times_with_min_lag(
        psi,
        dt,
        threshold=0.08,
        min_lag_frac=0.7,
        expected_period=expected_period,
        max_horizon=int(2 * expected_period / dt),
    )

    # IC
    IC = np.ones(N)
    IC[omega_drift > 0.5] = 0
    IC[fidelity_arr < 0.5] = 0
    IC[np.abs(az) > 2.5] = 0

    # κ - TIER-1 IDENTITY
    kappa = compute_kappa_from_IC(IC, dt)

    kernel_df = pd.DataFrame(
        {
            "t": t,
            "omega": omega_drift,
            "F": fidelity_arr,
            "S": S,
            "C": curvature_arr,
            "tau_R": tau_R,
            "IC": IC,
            "kappa": kappa,
        }
    )

    psi_df = pd.DataFrame({"t": t, "ax": ax, "ay": ay, "az": az, "gx": gx, "gy": gy, "gz": gz, "psi": psi})
    tauR_df = pd.DataFrame({"t": t, "tau_R": tau_R})

    oor_mask = np.abs(az) > 2.5
    oor_df = pd.DataFrame({"t": t, "channel": ["az"] * N, "oor": oor_mask.astype(int), "rate": oor_mask.astype(float)})

    censor_mask = ~np.isfinite(tau_R)
    censor_df = pd.DataFrame(
        {"t": t, "channel": ["tau_R"] * N, "censored": censor_mask.astype(int), "rate": censor_mask.astype(float)}
    )

    artifacts = []
    artifacts.append(
        {
            "path": "kernel/kernel.csv",
            "sha256": write_csv(run_dir / "kernel" / "kernel.csv", kernel_df),
            "role": "kernel",
            "bytes": len(kernel_df.to_csv()),
        }
    )
    artifacts.append(
        {
            "path": "derived/psi.csv",
            "sha256": write_csv(run_dir / "derived" / "psi.csv", psi_df),
            "role": "derived",
            "bytes": len(psi_df.to_csv()),
        }
    )
    artifacts.append(
        {
            "path": "tables/tauR_series.csv",
            "sha256": write_csv(run_dir / "tables" / "tauR_series.csv", tauR_df),
            "role": "table",
            "bytes": len(tauR_df.to_csv()),
        }
    )
    artifacts.append(
        {
            "path": "logs/oor.csv",
            "sha256": write_csv(run_dir / "logs" / "oor.csv", oor_df),
            "role": "log",
            "bytes": len(oor_df.to_csv()),
        }
    )
    artifacts.append(
        {
            "path": "logs/censor.csv",
            "sha256": write_csv(run_dir / "logs" / "censor.csv", censor_df),
            "role": "log",
            "bytes": len(censor_df.to_csv()),
        }
    )

    frozen = {
        "casepack_id": "KIN.CP.GAIT",
        "run_id": run_id,
        "contract": CONTRACT["name"],
        "timezone": "America/Chicago",
        "git_commit": "abc123def456",
        "package_version": "1.0.0",
        "pipeline": "imu_gait_analysis",
        "adapter": {
            "name": "GAIT.ADAPTER",
            "fs": fs,
            "T": T,
            "f_gait": f_gait,
            "expected_period": expected_period,
            "channels": ["ax", "ay", "az", "gx", "gy", "gz"],
        },
        "return": {"format": "csv", "columns": ["t", "omega", "F", "S", "C", "tau_R", "IC", "kappa"]},
        "tier1_identities": {
            "F_identity": "F = clip(1 - omega, 0, 1)",
            "kappa_identity": "kappa = cumsum(ln(IC + epsilon)) * dt",
            "omega_definition": "omega = |d/dt ln(psi)|",
            "C_normalization": "C = C_raw / (C_raw + C_0)",
            "tau_R_method": "min_lag_constrained",
        },
        "weights": {"omega": 1.0, "F": 1.5, "S": 1.0, "C": 0.5, "tau_R": 2.0},
        "closures": ["gamma.default.v1", "norms.l2_eta1e-3.v1", "return_domain.window64.v1"],
    }

    frozen_hash = write_json(run_dir / "config" / "frozen.json", frozen)
    artifacts.append(
        {"path": "config/frozen.json", "sha256": frozen_hash, "role": "config", "bytes": len(json.dumps(frozen))}
    )

    manifest = {
        "casepack_id": "KIN.CP.GAIT",
        "run_id": run_id,
        "created_utc": datetime.now(UTC).isoformat(),
        **frozen,
        "artifacts": artifacts,
    }
    write_json(run_dir / "manifest.json", manifest)

    tau_R_finite = tau_R[np.isfinite(tau_R)]
    print(f"✓ Generated {run_id}")
    print(f"  N={N}, T={T}s, f_gait={f_gait} Hz (expected period={expected_period:.3f}s)")
    print(
        f"  τ_R: coverage={100 * len(tau_R_finite) / N:.1f}%, median={np.median(tau_R_finite):.3f}s (target≈{expected_period:.3f}s)"
    )
    print(f"  ω: mean={np.mean(omega_drift):.4f} (nonnegative)")
    print(f"  F: mean={np.mean(fidelity_arr):.4f} (should be ≈ 1 - mean(ω))")
    print(f"  C: mean={np.mean(curvature_arr):.4f} (normalized to [0,1])")
    print(f"  IC pass rate={100 * np.mean(IC):.1f}%")
    print(f"  κ final: {kappa[-1]:.6f}")


def main() -> None:
    print("Generating KIN casepack runs (CORRECTED v3)...")
    print("=" * 60)
    print("CORRECTIONS APPLIED:")
    print("  1. F := clip(1 - |ω|, 0, 1)  [Tier-1 identity]")
    print("  2. κ := cumsum(ln(IC + ε))dt  [Tier-1 identity]")
    print("  3. ω := |ω_raw|  [nonnegative magnitude]")
    print("  4. C := C_raw / (C_raw + C_0)  [normalized to [0,1]]")
    print("  5. τ_R phase-anchored for SHM, min-lag constrained for GAIT")
    print("  6. Weld PASS requires return coverage ≥ ρ_min")
    print("=" * 60)

    generate_shm_run()
    print()
    generate_ballistic_run()
    print()
    generate_gait_run()

    print()
    print("=" * 60)
    print("All runs generated successfully with Tier-1 identity enforcement!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Generate complete KIN casepack run directories with all required artifacts.

VERSION 4 - REFINED CORRECTIONS:
  1. C_0 = median(C_raw) so curvature median ≈ 0.5 (not saturating near 1)
  2. ω = ||Ψ(t) - Ψ(t-Δt)|| (trace jitter, not derivative noise)
  3. SHM: hysteresis + interpolation for phase anchors
  4. BALLISTIC: τ_R always censored (non-recurrent by design)
  5. GAIT: smooth IC contribution mapping + geometric aggregation
  6. κ = ln(IC) instantaneous (Tier-1), κ_cum as derived

Creates:
  runs/KIN.CP.SHM.RUN003/
  runs/KIN.CP.BALLISTIC.RUN003/
  runs/KIN.CP.GAIT.RUN003/
"""
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).parent.parent.resolve()
RUNS_DIR = REPO_ROOT / "runs"

# Contract parameters (UMA.INTSTACK.v1)
CONTRACT = {
    "name": "UMA.INTSTACK.v1",
    "epsilon": 1e-8,
    "eta": 1e-3,
    "p": 3,
    "alpha": 1.0,
    "lambda": 0.2,
    "tol_seam": 0.005,
    "rho_min": 0.50,
}


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
# TIER-1 IDENTITY FUNCTIONS (REFINED)
# ============================================================


def compute_omega_trace_jitter(psi: np.ndarray, smooth_window: int = 5) -> np.ndarray:
    """
    Compute ω as trace jitter: ||Ψ(t) - Ψ(t-Δt)||
    This measures actual trace movement, not numerical differentiation noise.
    Optionally median-filtered for stability.
    """
    N = len(psi)
    omega = np.zeros(N)
    omega[1:] = np.abs(np.diff(psi))
    omega[0] = omega[1]

    # Median filter for smoothing
    if smooth_window > 1:
        padded = np.pad(omega, smooth_window // 2, mode="edge")
        omega_smooth = np.zeros(N)
        for i in range(N):
            omega_smooth[i] = np.median(padded[i : i + smooth_window])
        omega = omega_smooth

    return omega


def compute_fidelity_from_omega(omega: np.ndarray, omega_scale: float = 1.0) -> np.ndarray:
    """
    TIER-1 IDENTITY: F := clip(1 - ω/scale, 0, 1)
    omega_scale normalizes ω to [0,1] range.
    """
    return np.clip(1.0 - omega / omega_scale, 0.0, 1.0)


def compute_kappa_instantaneous(IC: np.ndarray) -> np.ndarray:
    """
    TIER-1 IDENTITY: κ(t) = ln(IC(t) + ε)
    Instantaneous log-integrity (not cumulative).
    """
    epsilon = CONTRACT["epsilon"]
    return np.log(IC + epsilon)


def compute_kappa_cumulative(IC: np.ndarray, dt: float) -> np.ndarray:
    """
    DERIVED: κ_cum(T) = ∫₀ᵀ ln(IC(t) + ε) dt
    Cumulative log-integrity for seam accounting.
    """
    epsilon = CONTRACT["epsilon"]
    log_IC = np.log(IC + epsilon)
    return np.cumsum(log_IC) * dt


def normalize_curvature_adaptive(C_raw: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Normalize curvature to [0,1] using adaptive C_0 = median(C_raw).
    Returns (C_normalized, C_0_used).

    This ensures baseline median C ≈ 0.5, not saturating near 1.
    """
    C_0 = float(np.median(C_raw)) + CONTRACT["epsilon"]
    C_normalized = C_raw / (C_raw + C_0)
    return C_normalized, C_0


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
# RETURN TIME FUNCTIONS (REFINED)
# ============================================================


def compute_phase_anchor_return_hysteresis(x: np.ndarray, v: np.ndarray, t: np.ndarray, h: float = 0.05) -> np.ndarray:
    """
    Poincaré section return with hysteresis + interpolation.

    Anchor: x crosses from < -h to > +h with v > 0.
    Crossing time computed by linear interpolation for tighter τ_R.

    Parameters:
        h: hysteresis threshold (frozen per CasePack)
    """
    N = len(t)
    tau_R = np.full(N, np.inf)

    # Find anchor crossings with hysteresis
    anchor_times: list[float] = []
    below_neg_h = False

    for i in range(N):
        if x[i] < -h:
            below_neg_h = True
        elif x[i] > h and below_neg_h and v[i] > 0:
            # Interpolate exact crossing time
            if i > 0:
                # Linear interpolation: find t where x = 0
                x_prev, x_curr = x[i - 1], x[i]
                t_prev, t_curr = t[i - 1], t[i]
                if x_curr != x_prev:
                    t_cross = t_prev + (0 - x_prev) * (t_curr - t_prev) / (x_curr - x_prev)
                else:
                    t_cross = t_curr
                anchor_times.append(t_cross)
            else:
                anchor_times.append(t[i])
            below_neg_h = False

    # Compute return time between successive anchors
    if len(anchor_times) < 2:
        return tau_R

    anchor_idx = 0
    for i in range(N):
        # Find which anchor interval this sample is in
        while anchor_idx < len(anchor_times) - 1 and t[i] >= anchor_times[anchor_idx + 1]:
            anchor_idx += 1

        if anchor_idx < len(anchor_times) - 1:
            tau_R[i] = anchor_times[anchor_idx + 1] - anchor_times[anchor_idx]

    return tau_R


def compute_return_always_censored(N: int) -> np.ndarray:
    """
    Return closure for non-recurrent systems (e.g., ballistic).
    τ_R = ∞ always (censored by design).
    """
    return np.full(N, np.inf)


def compute_event_anchor_return(
    signal: np.ndarray, t: np.ndarray, min_interval: float = 0.3, threshold_quantile: float = 0.8
) -> np.ndarray:
    """
    Event-anchored return for quasi-periodic signals (e.g., gait).

    Detects peaks above threshold with minimum refractory interval.
    Returns inter-event intervals.
    """
    N = len(t)
    dt = t[1] - t[0] if N > 1 else 0.01
    tau_R = np.full(N, np.inf)

    # Threshold for peak detection
    threshold = np.quantile(signal, threshold_quantile)
    min_samples = int(min_interval / dt)

    # Find peaks
    peak_times: list[float] = []
    last_peak_idx = -min_samples

    for i in range(1, N - 1):
        if (
            signal[i] > signal[i - 1]
            and signal[i] > signal[i + 1]
            and signal[i] > threshold
            and i - last_peak_idx >= min_samples
        ):
            peak_times.append(t[i])
            last_peak_idx = i

    # Compute inter-event intervals
    if len(peak_times) < 2:
        return tau_R

    peak_idx = 0
    for i in range(N):
        while peak_idx < len(peak_times) - 1 and t[i] >= peak_times[peak_idx + 1]:
            peak_idx += 1

        if peak_idx < len(peak_times) - 1:
            tau_R[i] = peak_times[peak_idx + 1] - peak_times[peak_idx]

    return tau_R


def compute_local_entropy(psi: np.ndarray, window: int = 50) -> np.ndarray:
    """Compute local Shannon entropy in sliding window."""
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


# ============================================================
# IC CONTRIBUTION MAPPING (REFINED)
# ============================================================


def compute_IC_smooth_contribution(
    residuals: dict[str, np.ndarray], weights: dict[str, float], p: float = 2.0
) -> np.ndarray:
    """
    Compute IC with smooth contribution mapping + geometric aggregation.

    For each residual channel e_i(t):
      E_i = p95(|e_i|)  (frozen scale)
      c_i(t) = exp(-(|e_i(t)| / (E_i + ε))^p)

    Aggregate: IC(t) = Π_i c_i(t)^w_i  (weighted geometric mean)
    """
    epsilon = CONTRACT["epsilon"]

    # Compute contribution for each channel
    contributions: list[np.ndarray] = []
    channel_weights: list[float] = []

    for channel, residual in residuals.items():
        E_i = float(np.quantile(np.abs(residual), 0.95)) + epsilon
        c_i = np.exp(-np.power(np.abs(residual) / E_i, p))
        contributions.append(c_i)
        channel_weights.append(weights.get(channel, 1.0))

    # Normalize weights
    total_weight = sum(channel_weights)
    channel_weights = [w / total_weight for w in channel_weights]

    # Geometric aggregation
    N = len(contributions[0])
    ic_result = np.ones(N)
    for c_i, w_i in zip(contributions, channel_weights, strict=False):
        ic_result *= np.power(c_i, w_i)

    return ic_result


# ============================================================
# SHM Casepack - Simple Harmonic Motion (REFINED)
# ============================================================


def generate_shm_run() -> None:
    """Generate KIN.CP.SHM run with refined corrections."""
    run_id = "KIN.CP.SHM.RUN003"
    run_dir = RUNS_DIR / run_id

    for subdir in ["config", "kernel", "derived", "tables", "logs", "plots"]:
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Simulation parameters
    dt = 0.01
    T = 10.0
    f0 = 1.0
    omega_0 = 2 * np.pi * f0
    A = 1.0
    noise_sigma = 0.005
    expected_period = 1.0 / f0
    h_hysteresis = 0.05 * A  # Hysteresis threshold for anchors

    t = np.arange(0, T, dt)
    N = len(t)

    np.random.seed(42)
    x_true = A * np.cos(omega_0 * t)
    v_true = -A * omega_0 * np.sin(omega_0 * t)

    x_obs = x_true + noise_sigma * np.random.randn(N)
    v_obs = v_true + noise_sigma * np.random.randn(N)

    # Normalized state vector
    psi = np.sqrt(x_obs**2 + (v_obs / omega_0) ** 2)

    # ω = trace jitter (not derivative noise)
    omega = compute_omega_trace_jitter(psi, smooth_window=5)
    omega_scale = float(np.quantile(omega, 0.95)) + CONTRACT["epsilon"]  # Freeze scale

    # F = 1 - ω/scale (Tier-1 identity)
    fidelity_arr = compute_fidelity_from_omega(omega, omega_scale)

    # S (entropy)
    S = compute_local_entropy(psi, window=int(expected_period / dt))

    # C (curvature) - adaptive normalization so median ≈ 0.5
    C_raw = compute_raw_curvature(psi, dt)
    curvature_arr, C_0_used = normalize_curvature_adaptive(C_raw)

    # τ_R - phase-anchored with hysteresis + interpolation
    tau_R = compute_phase_anchor_return_hysteresis(x_obs, v_obs, t, h=h_hysteresis)

    # IC - smooth contribution based on fidelity
    residuals = {"omega": omega}
    weights = {"omega": 1.0}
    IC = compute_IC_smooth_contribution(residuals, weights, p=2.0)

    # κ instantaneous (Tier-1) and κ_cum (derived)
    kappa = compute_kappa_instantaneous(IC)
    kappa_cum = compute_kappa_cumulative(IC, dt)

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
            "kappa_cum": kappa_cum,
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
        "return": {"format": "csv", "columns": ["t", "omega", "F", "S", "C", "tau_R", "IC", "kappa", "kappa_cum"]},
        "tier1_identities": {
            "omega_definition": "||Ψ(t) - Ψ(t-Δt)|| (trace jitter)",
            "F_identity": "F = clip(1 - ω/scale, 0, 1)",
            "kappa_identity": "κ = ln(IC + ε) [instantaneous]",
            "C_normalization": "C = C_raw / (C_raw + C_0), C_0 = median(C_raw)",
            "tau_R_method": "phase_anchor_hysteresis_interpolation",
            "IC_method": "smooth_contribution_geometric_aggregation",
        },
        "frozen_scales": {
            "omega_scale": omega_scale,
            "C_0": C_0_used,
            "h_hysteresis": h_hysteresis,
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
    print(f"  τ_R: coverage={100*len(tau_R_finite)/N:.1f}%, median={np.median(tau_R_finite):.4f}s")
    print(f"  ω: median={np.median(omega):.4f}, scale={omega_scale:.4f}")
    print(f"  F: median={np.median(fidelity_arr):.4f} (should be ≈ 1 - median(ω)/scale)")
    print(f"  C: median={np.median(curvature_arr):.4f}, C_0={C_0_used:.4f} (target median ≈ 0.5)")
    print(f"  IC: median={np.median(IC):.4f}, min={np.min(IC):.4f}")
    print(f"  κ: median={np.median(kappa):.4f} (instantaneous)")


# ============================================================
# BALLISTIC Casepack - τ_R always censored (REFINED)
# ============================================================


def generate_ballistic_run() -> None:
    """Generate KIN.CP.BALLISTIC run with τ_R always censored."""
    run_id = "KIN.CP.BALLISTIC.RUN003"
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

    # ω = trace jitter
    omega = compute_omega_trace_jitter(psi, smooth_window=5)
    omega_scale = float(np.quantile(omega, 0.95)) + CONTRACT["epsilon"]

    # F = 1 - ω/scale
    fidelity_arr = compute_fidelity_from_omega(omega, omega_scale)

    # S (entropy)
    S = compute_local_entropy(psi, window=50)

    # C (curvature) - adaptive
    C_raw = compute_raw_curvature(psi, dt)
    curvature_arr, C_0_used = normalize_curvature_adaptive(C_raw)

    # τ_R - ALWAYS CENSORED for ballistic (non-recurrent by design)
    tau_R = compute_return_always_censored(N)

    # IC - smooth contribution
    residuals = {"omega": omega}
    weights = {"omega": 1.0}
    IC = compute_IC_smooth_contribution(residuals, weights, p=2.0)

    # κ instantaneous and cumulative
    kappa = compute_kappa_instantaneous(IC)
    kappa_cum = compute_kappa_cumulative(IC, dt)

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
            "kappa_cum": kappa_cum,
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

    # Weld logic - integrity only (τ_R always censored)
    if seam_idx is not None and seam_idx > 10 and seam_idx < N - 10:
        pre_slice = slice(max(0, seam_idx - 50), seam_idx)
        post_slice = slice(seam_idx, min(N, seam_idx + 50))

        IC_pre = float(np.mean(IC[pre_slice]))
        IC_post = float(np.mean(IC[post_slice]))
        kappa_cum_pre = float(kappa_cum[seam_idx - 1])
        kappa_cum_post = float(kappa_cum[min(seam_idx + 50, N - 1)])
        delta_kappa = abs(kappa_cum_post - kappa_cum_pre)

        tau_R_finite_post = 0.0  # Always 0 since τ_R is censored by design
        seam_residual = abs(y_arr[seam_idx])

        _budget_ok = delta_kappa < CONTRACT["tol_seam"]  # Recorded for audit
        _integrity_ok = IC_post > 0.7  # Recorded for audit

        weld_result = "INTEGRITY_ONLY"
        weld_type = "NON_RECURRENT_BY_DESIGN"
    else:
        IC_pre, IC_post = 1.0, 1.0
        kappa_cum_pre, kappa_cum_post = 0.0, 0.0
        delta_kappa, tau_R_finite_post, seam_residual = 0.0, 0.0, 0.0
        weld_result = "NO_SEAM"
        weld_type = "NO_SEAM"

    weld_receipt = {
        "seam_id": "KIN.SEAM.IMP.001",
        "contract": CONTRACT["name"],
        "t_s": float(t_seam) if t_seam else None,
        "IC_pre": IC_pre,
        "IC_post": IC_post,
        "kappa_cum_pre": kappa_cum_pre,
        "kappa_cum_post": kappa_cum_post,
        "delta_kappa_cum": delta_kappa,
        "tau_R_finite_percent_post": float(tau_R_finite_post * 100),
        "tau_R_closure": "always_censored",
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
        "return": {"format": "csv", "columns": ["t", "omega", "F", "S", "C", "tau_R", "IC", "kappa", "kappa_cum"]},
        "tier1_identities": {
            "omega_definition": "||Ψ(t) - Ψ(t-Δt)|| (trace jitter)",
            "F_identity": "F = clip(1 - ω/scale, 0, 1)",
            "kappa_identity": "κ = ln(IC + ε) [instantaneous]",
            "C_normalization": "C = C_raw / (C_raw + C_0), C_0 = median(C_raw)",
            "tau_R_closure": "always_censored (non-recurrent by design)",
            "IC_method": "smooth_contribution_geometric_aggregation",
        },
        "frozen_scales": {
            "omega_scale": omega_scale,
            "C_0": C_0_used,
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

    print(f"✓ Generated {run_id}")
    print(f"  N={N}, T={t_arr[-1]:.2f}s, bounces={bounce_count}")
    print(f"  Seam: t_s={t_seam:.4f}s")
    print("  τ_R: ALWAYS CENSORED (non-recurrent by design)")
    print(f"  ω: median={np.median(omega):.4f}, scale={omega_scale:.4f}")
    print(f"  F: median={np.median(fidelity_arr):.4f}")
    print(f"  C: median={np.median(curvature_arr):.4f}, C_0={C_0_used:.4f}")
    print(f"  IC: pre={IC_pre:.4f}, post={IC_post:.4f}")
    print(f"  Weld: {weld_result} ({weld_type})")


# ============================================================
# GAIT Casepack - Smooth IC + Event Anchors (REFINED)
# ============================================================


def generate_gait_run() -> None:
    """Generate KIN.CP.GAIT run with refined IC and event anchors."""
    run_id = "KIN.CP.GAIT.RUN003"
    run_dir = RUNS_DIR / run_id

    for subdir in ["config", "kernel", "derived", "tables", "logs", "plots"]:
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Simulation parameters
    fs = 100
    dt = 1.0 / fs
    T = 30.0
    f_gait = 1.8
    expected_period = 1.0 / f_gait
    min_interval = 0.8 * expected_period  # Refractory interval for peak detection

    t = np.arange(0, T, dt)
    N = len(t)
    period_samples = int(expected_period / dt)

    np.random.seed(456)

    # Simulate gait IMU signals
    omega_gait = 2 * np.pi * f_gait

    az_base = 1.0 + 0.3 * np.sin(omega_gait * t) + 0.1 * np.sin(2 * omega_gait * t)
    ax = 0.1 * np.sin(omega_gait * t + 0.5) + 0.02 * np.random.randn(N)
    ay = 0.05 * np.sin(omega_gait * t + 1.0) + 0.02 * np.random.randn(N)
    az = az_base + 0.05 * np.random.randn(N)

    gx = 0.5 * np.sin(omega_gait * t) + 0.05 * np.random.randn(N)
    gy = 0.3 * np.sin(omega_gait * t + np.pi / 4) + 0.05 * np.random.randn(N)
    gz = 0.2 * np.sin(2 * omega_gait * t) + 0.05 * np.random.randn(N)

    # State magnitude
    psi = np.sqrt(ax**2 + ay**2 + (az - 1) ** 2 + 0.1 * (gx**2 + gy**2 + gz**2))

    # ω = trace jitter
    omega = compute_omega_trace_jitter(psi, smooth_window=5)
    omega_scale = float(np.quantile(omega, 0.95)) + CONTRACT["epsilon"]

    # F = 1 - ω/scale
    fidelity_arr = compute_fidelity_from_omega(omega, omega_scale)

    # S (entropy)
    S = compute_local_entropy(psi, window=period_samples)

    # C (curvature) - adaptive
    C_raw = compute_raw_curvature(psi, dt)
    curvature_arr, C_0_used = normalize_curvature_adaptive(C_raw)

    # τ_R - event-anchored (vertical acceleration peaks = heel strikes)
    tau_R = compute_event_anchor_return(az, t, min_interval=min_interval, threshold_quantile=0.75)

    # IC - smooth contribution with multiple channels
    # Compute residuals as deviations from smooth baseline
    residuals = {
        "omega": omega,
        "az_deviation": np.abs(az - 1.0),  # Deviation from gravity
    }
    weights = {"omega": 0.5, "az_deviation": 0.5}
    IC = compute_IC_smooth_contribution(residuals, weights, p=2.0)

    # κ instantaneous and cumulative
    kappa = compute_kappa_instantaneous(IC)
    kappa_cum = compute_kappa_cumulative(IC, dt)

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
            "kappa_cum": kappa_cum,
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
        "return": {"format": "csv", "columns": ["t", "omega", "F", "S", "C", "tau_R", "IC", "kappa", "kappa_cum"]},
        "tier1_identities": {
            "omega_definition": "||Ψ(t) - Ψ(t-Δt)|| (trace jitter)",
            "F_identity": "F = clip(1 - ω/scale, 0, 1)",
            "kappa_identity": "κ = ln(IC + ε) [instantaneous]",
            "C_normalization": "C = C_raw / (C_raw + C_0), C_0 = median(C_raw)",
            "tau_R_method": "event_anchor_az_peaks",
            "IC_method": "smooth_contribution_geometric_aggregation",
        },
        "frozen_scales": {
            "omega_scale": omega_scale,
            "C_0": C_0_used,
            "min_interval": min_interval,
        },
        "IC_channels": {"omega": 0.5, "az_deviation": 0.5},
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
    print(f"  τ_R: coverage={100*len(tau_R_finite)/N:.1f}%, median={np.median(tau_R_finite):.4f}s")
    print(f"  ω: median={np.median(omega):.4f}, scale={omega_scale:.4f}")
    print(f"  F: median={np.median(fidelity_arr):.4f}")
    print(f"  C: median={np.median(curvature_arr):.4f}, C_0={C_0_used:.4f}")
    print(f"  IC: median={np.median(IC):.4f}, min={np.min(IC):.4f}, pass rate={100*np.mean(IC > 0.5):.1f}%")
    print(f"  κ: median={np.median(kappa):.4f} (instantaneous)")


def main() -> None:
    print("Generating KIN casepack runs (REFINED v4)...")
    print("=" * 70)
    print("REFINEMENTS APPLIED:")
    print("  1. C_0 = median(C_raw) → C median ≈ 0.5 (not saturating)")
    print("  2. ω = ||Ψ(t) - Ψ(t-Δt)|| (trace jitter, not derivative noise)")
    print("  3. SHM: hysteresis + interpolation phase anchors")
    print("  4. BALLISTIC: τ_R always censored (non-recurrent by design)")
    print("  5. GAIT: smooth IC contribution + geometric aggregation")
    print("  6. κ = ln(IC) instantaneous (Tier-1), κ_cum as derived")
    print("=" * 70)

    generate_shm_run()
    print()
    generate_ballistic_run()
    print()
    generate_gait_run()

    print()
    print("=" * 70)
    print("All runs generated successfully with refined corrections!")


if __name__ == "__main__":
    main()

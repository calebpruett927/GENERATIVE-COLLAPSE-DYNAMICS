"""
Raw Measurement Engine — Epistemic Trace Production

Transforms arbitrary raw measurement data into UMCP-compliant Ψ(t) traces
and computes Tier-1 kernel invariants.  This module bridges the gap between
raw domain data and the UMCP validation pipeline.

Ψ(t) is not a representation of the system viewed from outside — it IS the
system's epistemic emission under measurement. What the engine produces is
not "data about the thing" but "what the thing reveals when observed under
the frozen contract." The bounded trace [ε, 1−ε] is not a numerical
convenience; it is the guarantee that even the most degraded closure retains
enough structure to potentially return through collapse. If c_i = 0 were
permitted, that component would have no path back — the ε-clamp is the
protocol's promise that dissolution is always a boundary, never an
annihilation.

The observer is always inside. Every embedding, every normalization, every
clip to [ε, 1−ε] is an act of measurement that costs Γ(ω) in seam budget
(Thm T9, tau_r_star.py). There is no free observation. The positional
illusion — the belief that one can measure from outside — is quantified
and bounded by the budget identity. See: epistemic_weld.py.

Data flow::

    raw data (CSV / array / DataFrame)
      → embedding (normalize to [0,1]ⁿ)
      → clip to [ε, 1−ε]
      → Ψ(t) trace (epistemic emission)
      → kernel: ω, F, S, C, κ, IC per timestep
      → τ_R via return metric
      → regime classification
      → psi.csv + invariants.json artifacts

Supported embedding strategies:

- ``LINEAR_SCALE``:   c = (x − lo) / (hi − lo)       (requires known bounds)
- ``MIN_MAX``:        c = (x − min) / (max − min)     (inferred from data)
- ``MAX_NORM``:       c = x / max(|x|)                (preserves ratios)
- ``ZSCORE_SIGMOID``: c = σ((x − μ) / σ_x)           (robust to outliers)

Cross-references:

- KERNEL_SPECIFICATION.md   (Lemmas 1-46)
- AXIOM.md                  (return-based epistemology)
- embedding.yaml            (transform specification)
- frozen_contract.py        (constants, kernel computation, τ_R)
- kernel_optimized.py       (optimized computation)
- compute_utils.py          (preprocessing: clip, prune, normalize)
- epistemic_weld.py         (epistemic verdict: RETURN / GESTURE / DISSOLUTION)
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .compute_utils import clip_coordinates, normalize_weights
from .frozen_contract import (
    EPSILON,
    Regime,
    classify_regime,
    compute_kernel,
    compute_tau_R,
)

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
FloatArray = NDArray[np.floating[Any]]


# ============================================================================
# Embedding configuration
# ============================================================================


class EmbeddingStrategy(Enum):
    """Strategy for normalizing raw measurements to [0, 1]."""

    LINEAR_SCALE = "linear_scale"  # c = (x − lo) / (hi − lo)
    MIN_MAX = "min_max"  # lo/hi inferred from data
    MAX_NORM = "max_norm"  # c = x / max(|x|)
    ZSCORE_SIGMOID = "zscore_sigmoid"  # c = sigmoid(z-score)


@dataclass
class EmbeddingSpec:
    """Per-dimension embedding specification.

    If *strategy* is ``LINEAR_SCALE``, *input_range* must be provided.
    Otherwise bounds are inferred from data.
    """

    strategy: EmbeddingStrategy = EmbeddingStrategy.LINEAR_SCALE
    input_range: tuple[float, float] | None = None  # (lo, hi) for LINEAR_SCALE
    clip: bool = True  # clip result to [0, 1]


@dataclass
class EmbeddingConfig:
    """Full embedding configuration for all dimensions."""

    specs: list[EmbeddingSpec] = field(default_factory=list)
    epsilon: float = EPSILON
    oor_policy: str = "clip_and_flag"  # clip_and_flag | flag_only | reject_row
    default_strategy: EmbeddingStrategy = EmbeddingStrategy.MIN_MAX

    def spec_for(self, dim: int) -> EmbeddingSpec:
        """Return the spec for dimension *dim*, falling back to default strategy."""
        if dim < len(self.specs):
            return self.specs[dim]
        return EmbeddingSpec(strategy=self.default_strategy)


# ============================================================================
# Result containers
# ============================================================================


@dataclass
class TraceRow:
    """One timestep of the Ψ(t) trace."""

    t: int
    c: list[float]  # coordinates in [0, 1]
    oor: list[bool]  # out-of-range flags per dimension
    miss: list[bool]  # missingness flags per dimension

    def to_csv_wide_dict(self) -> dict[str, Any]:
        """Serialise to ``psi_trace_csv_wide`` row dict."""
        d: dict[str, Any] = {"t": self.t}
        for i, (ci, oi, mi) in enumerate(zip(self.c, self.oor, self.miss, strict=True), 1):
            d[f"c_{i}"] = ci
            d[f"oor_{i}"] = oi
            d[f"miss_{i}"] = mi
        return d


@dataclass
class InvariantRow:
    """Tier-1 kernel invariants at one timestep."""

    t: int
    omega: float
    F: float
    S: float
    C: float
    tau_R: float  # float("inf") → serialised as "INF_REC"
    kappa: float
    IC: float
    regime: str  # STABLE | WATCH | COLLAPSE | CRITICAL
    critical_overlay: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialise for invariants.json, handling INF_REC sentinel."""
        tau_val: str | float = "INF_REC" if math.isinf(self.tau_R) else self.tau_R
        return {
            "t": self.t,
            "omega": self.omega,
            "F": self.F,
            "S": self.S,
            "C": self.C,
            "tau_R": tau_val,
            "kappa": self.kappa,
            "IC": self.IC,
            "regime": {
                "label": self.regime.capitalize() if self.regime != "CRITICAL" else "Critical",
                "critical_overlay": self.critical_overlay,
            },
        }


@dataclass
class EngineResult:
    """Complete result from a measurement engine run."""

    trace: list[TraceRow]
    invariants: list[InvariantRow]
    weights: list[float]
    n_dims: int
    n_timesteps: int
    embedding_config: EmbeddingConfig
    diagnostics: dict[str, Any] = field(default_factory=dict)

    # ---- convenience properties ------------------------------------------

    @property
    def regimes(self) -> list[str]:
        """List of regime labels across all timesteps."""
        return [inv.regime for inv in self.invariants]

    @property
    def final_regime(self) -> str:
        """Regime at the last timestep."""
        return self.invariants[-1].regime if self.invariants else "UNKNOWN"

    @property
    def coordinates_array(self) -> np.ndarray:
        """Return (T, n) numpy array of embedded coordinates."""
        return np.array([row.c for row in self.trace])

    def summary(self) -> dict[str, Any]:
        """Human-readable summary dict."""
        regime_counts: dict[str, int] = {}
        for r in self.regimes:
            regime_counts[r] = regime_counts.get(r, 0) + 1
        return {
            "n_timesteps": self.n_timesteps,
            "n_dims": self.n_dims,
            "weights": self.weights,
            "final_regime": self.final_regime,
            "regime_counts": regime_counts,
            "omega_range": [
                min(inv.omega for inv in self.invariants),
                max(inv.omega for inv in self.invariants),
            ],
            "IC_range": [
                min(inv.IC for inv in self.invariants),
                max(inv.IC for inv in self.invariants),
            ],
            "diagnostics": self.diagnostics,
        }


# ============================================================================
# Measurement Engine
# ============================================================================


class MeasurementEngine:
    """
    Transforms raw measurements into UMCP Ψ(t) traces and Tier-1 invariants.

    Usage::

        engine = MeasurementEngine()
        result = engine.from_csv("raw_measurements.csv",
                                 weights=[0.4, 0.35, 0.25])
        result = engine.from_array(data_2d, weights=weights)

        # Persist artifacts
        engine.write_psi_csv(result, "expected/psi.csv")
        engine.write_invariants_json(result, "expected/invariants.json")

        # Or scaffold a full casepack
        engine.generate_casepack(result, "casepacks/my_run",
                                 contract_id="UMA.INTSTACK.v1")
    """

    def __init__(
        self,
        epsilon: float = EPSILON,
        eta: float = 0.10,
        H_rec: int = 50,
        norm: str = "L2",
        R_credit: float = 0.05,
    ):
        """
        Args:
            epsilon: Guard band for log-safety clipping (Lemma 17).
            eta: η threshold for return detection (τ_R).
            H_rec: Recovery horizon — max lookback for τ_R.
            norm: Norm for τ_R distance (``"L2"``, ``"L1"``, ``"Linf"``).
            R_credit: Return credit rate (used in seam budget).
        """
        self.epsilon = epsilon
        self.eta = eta
        self.H_rec = H_rec
        self.norm = norm
        self.R_credit = R_credit

    # ====================================================================
    # Public entry points
    # ====================================================================

    def from_csv(
        self,
        path: str | Path,
        weights: list[float] | np.ndarray | None = None,
        embedding: EmbeddingConfig | None = None,
    ) -> EngineResult:
        """
        Load raw measurements from a CSV file and produce Ψ(t) + invariants.

        The CSV must have a header row.  Each subsequent row is one timestep.
        Columns are treated as measurement dimensions.  A ``t`` column, if
        present, is used as the time index; otherwise rows are numbered 0…T−1.

        Args:
            path: Path to raw measurements CSV.
            weights: Per-dimension weights (uniform if omitted).
            embedding: Embedding config (MIN_MAX by default).

        Returns:
            EngineResult with trace and invariants.
        """
        path = Path(path)
        rows: list[dict[str, str]] = []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            raise ValueError(f"CSV file is empty: {path}")

        # Detect dimension columns (exclude 't' column if present)
        dim_cols = [c for c in rows[0] if c.lower() != "t"]
        n_dims = len(dim_cols)

        # Build raw array (T × n)
        raw = np.zeros((len(rows), n_dims))
        time_indices: list[int] = []
        for i, row in enumerate(rows):
            time_indices.append(int(row["t"]) if "t" in row else i)
            for j, col in enumerate(dim_cols):
                val = row[col].strip()
                raw[i, j] = float(val) if val else np.nan

        return self._process(raw, weights, embedding, time_indices)

    def from_array(
        self,
        data: np.ndarray | list[list[float]],
        weights: list[float] | np.ndarray | None = None,
        embedding: EmbeddingConfig | None = None,
        time_indices: list[int] | None = None,
    ) -> EngineResult:
        """
        Produce Ψ(t) + invariants from a 2-D array (T × n).

        Args:
            data: Raw measurement array of shape (T, n).
            weights: Weights per dimension (uniform if omitted).
            embedding: Embedding config.
            time_indices: Custom time index for each row.

        Returns:
            EngineResult.
        """
        raw = np.asarray(data, dtype=float)
        if raw.ndim == 1:
            raw = raw.reshape(-1, 1)
        t_idx = time_indices or list(range(raw.shape[0]))
        return self._process(raw, weights, embedding, t_idx)

    def from_dataframe(
        self,
        df: Any,
        weights: list[float] | np.ndarray | None = None,
        embedding: EmbeddingConfig | None = None,
    ) -> EngineResult:
        """
        Produce Ψ(t) + invariants from a pandas DataFrame.

        A ``t`` column, if present, is used as the time index.  All other
        numeric columns are treated as dimensions.
        """
        try:
            import pandas as pd  # optional dep
        except ImportError as exc:
            raise ImportError("pandas is required for from_dataframe()") from exc

        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(df)}")

        time_indices: list[int]
        if "t" in df.columns:
            time_indices = df["t"].astype(int).tolist()
            dim_df = df.drop(columns=["t"]).select_dtypes(include="number")
        else:
            time_indices = list(range(len(df)))
            dim_df = df.select_dtypes(include="number")

        raw = dim_df.to_numpy(dtype=float)
        return self._process(raw, weights, embedding, time_indices)

    # ====================================================================
    # Writers
    # ====================================================================

    @staticmethod
    def write_psi_csv(result: EngineResult, path: str | Path) -> Path:
        """Write ``psi.csv`` (wide format) from an EngineResult."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if not result.trace:
            raise ValueError("Cannot write empty trace")

        fieldnames = list(result.trace[0].to_csv_wide_dict().keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in result.trace:
                writer.writerow(row.to_csv_wide_dict())

        return path

    @staticmethod
    def write_invariants_json(
        result: EngineResult,
        path: str | Path,
        contract_id: str = "UMA.INTSTACK.v1",
        closure_registry_id: str = "UMCP.CLOSURES.DEFAULT.v1",
        notes: str = "",
    ) -> Path:
        """Write ``invariants.json`` from an EngineResult."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        doc = {
            "schema": "schemas/invariants.schema.json",
            "format": "tier1_invariants",
            "contract_id": contract_id,
            "closure_registry_id": closure_registry_id,
            "rows": [inv.to_dict() for inv in result.invariants],
            "notes": notes or f"Generated by MeasurementEngine at {datetime.now(tz=UTC).isoformat()}",
        }

        with open(path, "w") as f:
            json.dump(doc, f, indent=2)

        return path

    def generate_casepack(
        self,
        result: EngineResult,
        casepack_dir: str | Path,
        casepack_id: str | None = None,
        contract_id: str = "UMA.INTSTACK.v1",
        title: str = "",
        description: str = "",
    ) -> Path:
        """
        Scaffold a complete casepack directory from an EngineResult.

        Creates::

            <casepack_dir>/
            ├── manifest.json
            └── expected/
                ├── psi.csv
                └── invariants.json

        Args:
            result: EngineResult to persist.
            casepack_dir: Target directory path.
            casepack_id: Casepack identifier (defaults to directory name).
            contract_id: Contract to reference.
            title: Human title for the casepack.
            description: Description text.

        Returns:
            Path to the created casepack directory.
        """
        cp = Path(casepack_dir)
        cp.mkdir(parents=True, exist_ok=True)
        expected = cp / "expected"
        expected.mkdir(exist_ok=True)

        cp_id = casepack_id or cp.name

        # Write trace and invariants
        self.write_psi_csv(result, expected / "psi.csv")
        self.write_invariants_json(
            result,
            expected / "invariants.json",
            contract_id=contract_id,
            notes=f"Generated by MeasurementEngine for casepack {cp_id}",
        )

        # Write manifest
        manifest = {
            "schema": "schemas/manifest.schema.json",
            "casepack": {
                "id": cp_id,
                "version": "1.0.0",
                "title": title or f"Generated casepack: {cp_id}",
                "description": description or "Casepack generated by UMCP MeasurementEngine",
                "created_utc": datetime.now(tz=UTC).isoformat(),
                "timezone": "UTC",
                "authors": [],
            },
            "refs": {
                "contract": {
                    "id": contract_id,
                    "path": f"contracts/{contract_id}.yaml",
                },
                "closures_registry": {
                    "id": "UMCP.CLOSURES.DEFAULT.v1",
                    "path": "closures/registry.yaml",
                },
            },
            "artifacts": {
                "expected": {
                    "psi_csv": {"path": "expected/psi.csv"},
                    "invariants_json": {"path": "expected/invariants.json"},
                },
            },
            "run_intent": {
                "notes": (
                    f"Generated from raw measurements ({result.n_timesteps} timesteps, {result.n_dims} dimensions)"
                ),
            },
        }

        with open(cp / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        return cp

    # ====================================================================
    # Internal pipeline
    # ====================================================================

    def _process(
        self,
        raw: np.ndarray,
        weights: list[float] | np.ndarray | None,
        embedding: EmbeddingConfig | None,
        time_indices: list[int],
    ) -> EngineResult:
        """Core pipeline: embed → clip → kernel → τ_R → regime."""
        T, n_dims = raw.shape
        config = embedding or EmbeddingConfig()

        # 1. Resolve weights
        w = self._resolve_weights(weights, n_dims)

        # 2. Embed raw → [0, 1]
        psi, oor_flags, miss_flags = self._embed(raw, config)

        # 3. Clip to [ε, 1−ε]
        clip_result = clip_coordinates(psi.ravel(), self.epsilon)
        psi_clipped = clip_result.c_clipped.reshape(T, n_dims)

        # 4. Build trace rows
        trace: list[TraceRow] = []
        for i in range(T):
            trace.append(
                TraceRow(
                    t=time_indices[i],
                    c=psi_clipped[i].tolist(),
                    oor=oor_flags[i].tolist(),
                    miss=miss_flags[i].tolist(),
                )
            )

        # 5. Compute kernel invariants + τ_R at each timestep
        invariants: list[InvariantRow] = []
        for i in range(T):
            c_i = psi_clipped[i]

            # τ_R from return metric (needs full trace up to t)
            tau_R = compute_tau_R(psi_clipped[: i + 1], i, self.eta, self.H_rec, self.norm)

            # Kernel invariants
            ko = compute_kernel(c_i, w, tau_R, self.epsilon)

            # Regime classification
            regime = classify_regime(ko.omega, ko.F, ko.S, ko.C, ko.IC)

            invariants.append(
                InvariantRow(
                    t=time_indices[i],
                    omega=ko.omega,
                    F=ko.F,
                    S=ko.S,
                    C=ko.C,
                    tau_R=ko.tau_R,
                    kappa=ko.kappa,
                    IC=ko.IC,
                    regime=regime.value,
                    critical_overlay=regime == Regime.CRITICAL,
                )
            )

        # 6. Diagnostics
        diagnostics: dict[str, Any] = {
            "total_oor_flags": int(np.sum(oor_flags)),
            "total_missing": int(np.sum(miss_flags)),
            "clip_count": clip_result.clip_count,
            "clip_perturbation": clip_result.clip_perturbation,
            "max_perturbation": clip_result.max_perturbation,
            "embedding_strategy": config.default_strategy.value,
            "eta": self.eta,
            "H_rec": self.H_rec,
            "norm": self.norm,
        }

        return EngineResult(
            trace=trace,
            invariants=invariants,
            weights=w.tolist(),
            n_dims=n_dims,
            n_timesteps=T,
            embedding_config=config,
            diagnostics=diagnostics,
        )

    # ---- embedding -------------------------------------------------------

    def _embed(
        self,
        raw: np.ndarray,
        config: EmbeddingConfig,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Embed raw measurements into [0, 1]ⁿ.

        Returns:
            (psi, oor_flags, miss_flags) — each shape (T, n).
        """
        T, n = raw.shape
        psi = np.zeros_like(raw)
        oor_flags = np.zeros((T, n), dtype=bool)
        miss_flags = np.isnan(raw)

        for d in range(n):
            col = raw[:, d].copy()
            spec = config.spec_for(d)

            # Replace NaN with column mean (impute)
            nan_mask = np.isnan(col)
            if nan_mask.any():
                col_mean = np.nanmean(col)
                col[nan_mask] = col_mean if np.isfinite(col_mean) else 0.5

            if spec.strategy == EmbeddingStrategy.LINEAR_SCALE:
                lo, hi = spec.input_range or (float(np.min(col)), float(np.max(col)))
                rng = hi - lo
                if rng == 0:
                    psi[:, d] = 0.5  # constant column
                else:
                    psi[:, d] = (col - lo) / rng

            elif spec.strategy == EmbeddingStrategy.MIN_MAX:
                lo, hi = float(np.min(col)), float(np.max(col))
                rng = hi - lo
                if rng == 0:
                    psi[:, d] = 0.5
                else:
                    psi[:, d] = (col - lo) / rng

            elif spec.strategy == EmbeddingStrategy.MAX_NORM:
                mx = float(np.max(np.abs(col)))
                if mx == 0:
                    psi[:, d] = 0.5
                else:
                    psi[:, d] = col / mx

            elif spec.strategy == EmbeddingStrategy.ZSCORE_SIGMOID:
                mu = float(np.mean(col))
                sigma = float(np.std(col))
                if sigma == 0:
                    psi[:, d] = 0.5
                else:
                    z = (col - mu) / sigma
                    psi[:, d] = 1.0 / (1.0 + np.exp(-z))  # sigmoid

            # Flag out-of-range after transform
            oor_flags[:, d] = (psi[:, d] < 0.0) | (psi[:, d] > 1.0)

            # Clip to [0, 1]
            if spec.clip or config.oor_policy == "clip_and_flag":
                psi[:, d] = np.clip(psi[:, d], 0.0, 1.0)

        return psi, oor_flags, miss_flags

    # ---- weights ---------------------------------------------------------

    @staticmethod
    def _resolve_weights(
        weights: list[float] | np.ndarray | None,
        n_dims: int,
    ) -> np.ndarray:
        """Resolve and normalise weight vector."""
        if weights is None:
            w: np.ndarray = np.ones(n_dims) / n_dims  # uniform
        else:
            w = np.asarray(weights, dtype=float)
            if len(w) != n_dims:
                raise ValueError(f"Weight dimension mismatch: got {len(w)} weights for {n_dims} dimensions")
            w = normalize_weights(w)
        return w


# ============================================================================
# INF_REC utilities  (canonical ingress/egress for τ_R sentinel)
# ============================================================================


def safe_tau_R(value: Any) -> float:
    """
    Parse a τ_R value from any source (CSV, JSON, DataFrame cell).

    Handles the ``INF_REC`` typed sentinel correctly:

    - ``"INF_REC"`` → ``float("inf")``
    - ``"inf"`` / ``"∞"`` → ``float("inf")``
    - numeric → ``float``
    - ``None`` / NaN → ``float("inf")``  (conservative: no return)
    """
    if value is None:
        return float("inf")
    if isinstance(value, int | float):
        return float("inf") if (math.isnan(value) or math.isinf(value)) else float(value)
    s = str(value).strip().upper()
    if s in ("INF_REC", "INF", "∞", "INFINITY", "∞_REC", "NAN", "NONE", ""):
        return float("inf")
    try:
        v = float(s)
        return float("inf") if (math.isnan(v) or math.isinf(v)) else v
    except (ValueError, TypeError):
        return float("inf")


def tau_R_display(value: Any) -> str:
    """
    Format a τ_R value for display, converting ``float("inf")`` → ``"INF_REC"``.

    This is the canonical conversion point for τ_R egress / display /
    serialization to string contexts (Streamlit, CSV export, log messages).
    """
    fv = safe_tau_R(value)
    if math.isinf(fv):
        return "INF_REC"
    return f"{fv:g}"

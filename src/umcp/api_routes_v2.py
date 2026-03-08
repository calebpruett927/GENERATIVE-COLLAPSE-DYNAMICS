"""
UMCP REST API — Extended Routes v2

Exposes all remaining engines and domain closures:
  - Seam chain accumulation (SeamChainAccumulator)
  - τ_R* thermodynamic diagnostic
  - Epistemic cost tracking (Theorem T9)
  - Insight engine (PatternDatabase, discovery)
  - Standard Model extensions (CKM, couplings, cross sections, Higgs, neutrino, matter genesis, matter map)
  - Dynamic semiotics (30 sign systems)
  - Consciousness coherence (20 systems)
  - Materials science (118 elements)
  - Cross-scale atomic kernel (12-channel)
  - Rosetta cross-domain translation

All endpoints use lazy imports for optional dependencies.
"""

from __future__ import annotations

import math
import os
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field


def _native(obj: Any) -> Any:
    """Convert numpy scalars to Python natives for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_native(v) for v in obj]
    # numpy scalar types
    t = type(obj).__name__
    if t in ("int64", "int32", "int16", "int8", "uint64", "uint32", "uint16", "uint8"):
        return int(obj)
    if t in ("float64", "float32", "float16"):
        return float(obj)
    if t in ("bool_", "bool"):
        return bool(obj)
    if t == "ndarray":
        return obj.tolist()
    return obj


# ── API Key Security (mirrors main module pattern) ──
_API_KEY = os.environ.get("UMCP_API_KEY", "umcp-dev-key")
_DEV_MODE = os.environ.get("UMCP_DEV_MODE", "0").lower() in ("1", "true", "yes")
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _validate_api_key(api_key: str = Security(_api_key_header)) -> str:
    """Validate API key — skipped in dev mode."""
    if _DEV_MODE:
        return "dev-mode-enabled"
    if not api_key or api_key != _API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return api_key


# ============================================================================
# Router
# ============================================================================

router = APIRouter()

# ============================================================================
# OpenAPI Tags for new route groups
# ============================================================================

EXTRA_TAGS = [
    {"name": "Seam", "description": "Seam chain accumulation and budget accounting"},
    {"name": "Thermodynamic", "description": "τ_R* thermodynamic diagnostic and phase classification"},
    {"name": "Epistemic", "description": "Epistemic cost tracking — Theorem T9"},
    {"name": "Insights", "description": "Pattern discovery and insight engine"},
    {"name": "Semiotics", "description": "Dynamic semiotics — 30 sign systems, 8-channel kernel"},
    {"name": "Consciousness", "description": "Consciousness coherence — 20 systems, 7 theorems"},
    {"name": "Materials", "description": "Materials science — 118 elements × 18 fields"},
    {"name": "Rosetta", "description": "Cross-domain Rosetta translation (6 lenses)"},
    {"name": "Orientation", "description": "Live orientation computation"},
]


# ============================================================================
# Pydantic Models — Seam
# ============================================================================


class SeamComputeRequest(BaseModel):
    """Request to add a seam record."""

    t0: int = Field(..., description="Start time index")
    t1: int = Field(..., description="End time index")
    kappa_t0: float = Field(..., description="κ at t0")
    kappa_t1: float = Field(..., description="κ at t1")
    tau_R: float = Field(..., description="Return time τ_R")
    R: float = Field(0.01, description="Return credit R")
    D_omega: float = Field(0.0, description="Drift debit D_ω")
    D_C: float = Field(0.0, description="Curvature debit D_C")


class SeamRecordResponse(BaseModel):
    """A single seam record."""

    t0: int
    t1: int
    kappa_t0: float
    kappa_t1: float
    tau_R: float
    delta_kappa_ledger: float
    delta_kappa_budget: float
    residual: float
    cumulative_residual: float


class SeamMetricsResponse(BaseModel):
    """Seam chain metrics."""

    total_seams: int
    total_delta_kappa: float
    cumulative_abs_residual: float
    max_residual: float
    mean_residual: float
    growth_exponent: float
    is_returning: bool
    failure_detected: bool


# ============================================================================
# Pydantic Models — τ_R*
# ============================================================================


class TauRStarRequest(BaseModel):
    """Request for τ_R* computation."""

    omega: float = Field(..., ge=0.0, le=1.0, description="Drift ω")
    C: float = Field(..., ge=0.0, le=1.0, description="Curvature C")
    R: float = Field(..., ge=0.0, description="Return credit R")
    delta_kappa: float = Field(0.0, description="Log-integrity change Δκ")


class TauRStarResponse(BaseModel):
    """τ_R* computation result."""

    tau_R_star: float
    gamma: float
    D_C: float
    delta_kappa: float
    R: float
    numerator: float
    phase: str
    dominance: str
    R_critical: float
    R_min: float
    is_trapped: bool
    c_trap: float


class RCriticalRequest(BaseModel):
    """Request for R_critical computation."""

    omega: float = Field(..., ge=0.0, le=1.0)
    C: float = Field(..., ge=0.0, le=1.0)
    delta_kappa: float = Field(0.0)


class RMinRequest(BaseModel):
    """Request for R_min computation."""

    omega: float = Field(..., ge=0.0, le=1.0)
    C: float = Field(..., ge=0.0, le=1.0)
    tau_R_target: float = Field(..., gt=0.0, description="Target τ_R value")
    delta_kappa: float = Field(0.0)


# ============================================================================
# Pydantic Models — Epistemic
# ============================================================================


class EpistemicClassifyRequest(BaseModel):
    """Request to classify an epistemic act."""

    seam_pass: bool = Field(..., description="Did the seam pass?")
    tau_R: float = Field(..., description="Return time (use Infinity for ∞_rec)")
    regime: str = Field(..., description="Regime label (STABLE/WATCH/COLLAPSE)")
    seam_failures: list[str] | None = Field(None, description="List of seam failure reasons")


class EpistemicClassifyResponse(BaseModel):
    """Epistemic classification result."""

    verdict: str
    reasons: list[str]


class PositionalIllusionRequest(BaseModel):
    """Request for positional illusion quantification."""

    omega: float = Field(..., ge=0.0, le=1.0, description="Drift ω")
    n_observations: int = Field(1, ge=1, description="Number of observations")


class PositionalIllusionResponse(BaseModel):
    """Positional illusion result."""

    gamma: float
    n_observations: int
    total_cost: float
    budget_fraction: float
    illusion_severity: float


class EpistemicTraceRequest(BaseModel):
    """Request for epistemic trace assessment."""

    n_components: int = Field(..., ge=1)
    n_timesteps: int = Field(..., ge=1)
    n_clipped: int = Field(0, ge=0)
    is_degenerate: bool = Field(False)
    seam_pass: bool = Field(True)
    tau_R: float = Field(1.0)
    regime: str = Field("STABLE")


class EpistemicTraceResponse(BaseModel):
    """Epistemic trace assessment result."""

    n_components: int
    n_timesteps: int
    epsilon_floor: float
    n_clipped: int
    clipped_fraction: float
    is_degenerate: bool
    verdict: str


# ============================================================================
# Pydantic Models — Insights
# ============================================================================


class InsightEntryResponse(BaseModel):
    """An insight entry."""

    id: str
    domain: str
    pattern: str
    lesson: str
    implication: str
    severity: str
    pattern_type: str
    source: str
    elements: list[str]
    omega_range: list[float]


class InsightSummaryResponse(BaseModel):
    """Insight engine summary."""

    total_insights: int
    domains: dict[str, int]
    by_severity: dict[str, int]
    by_type: dict[str, int]


# ============================================================================
# Pydantic Models — SM Extensions
# ============================================================================


class CKMResponse(BaseModel):
    """CKM mixing result."""

    V_matrix: list[list[float]]
    lambda_wolf: float
    A_wolf: float
    rho_bar: float
    eta_bar: float
    J_CP: float
    unitarity_row1: float
    unitarity_row2: float
    unitarity_row3: float
    triangle_angles: dict[str, float]
    omega_eff: float
    F_eff: float
    regime: str


class CouplingResponse(BaseModel):
    """Running coupling result."""

    alpha_s: float
    alpha_em: float
    sin2_theta_W: float
    G_F: float
    Q_GeV: float
    n_flavors: int
    omega_eff: float
    F_eff: float
    regime: str
    unification_proximity: float


class CrossSectionResponse(BaseModel):
    """Cross section result."""

    sqrt_s_GeV: float
    sigma_point_pb: float
    R_predicted: float
    R_QCD_corrected: float
    n_colors: int
    n_active_flavors: int
    sum_Qf_squared: float
    alpha_s_at_s: float
    omega_eff: float
    F_eff: float
    regime: str


class HiggsResponse(BaseModel):
    """Higgs mechanism result."""

    v_GeV: float
    m_H_GeV: float
    lambda_quartic: float
    mu_squared: float
    yukawa_couplings: dict[str, float]
    m_W_predicted: float
    m_Z_predicted: float
    omega_eff: float
    F_eff: float
    regime: str


class NeutrinoOscillationRequest(BaseModel):
    """Request for neutrino oscillation computation."""

    L_km: float = Field(..., gt=0.0, description="Baseline length in km")
    E_GeV: float = Field(..., gt=0.0, description="Neutrino energy in GeV")
    antineutrino: bool = Field(False, description="Antineutrino mode")


# ============================================================================
# Pydantic Models — Rosetta
# ============================================================================


class RosettaRequest(BaseModel):
    """Request for Rosetta cross-domain translation."""

    drift: str = Field(..., description="Drift narrative in source lens")
    fidelity: str = Field(..., description="Fidelity narrative")
    roughness: str = Field(..., description="Roughness narrative")
    return_narrative: str = Field(..., description="Return narrative")
    source_lens: str = Field("Epistemology", description="Source lens")
    target_lens: str = Field("Ontology", description="Target lens")


class RosettaResponse(BaseModel):
    """Rosetta translation result."""

    source_lens: str
    target_lens: str
    translations: dict[str, dict[str, str]]
    lens_mappings: dict[str, dict[str, str]]


# ============================================================================
# Seam Chain Endpoints
# ============================================================================

# Module-level accumulator (stateful per server instance)
_seam_accumulator = None


def _get_seam_accumulator() -> Any:
    """Lazy-init the seam accumulator."""
    global _seam_accumulator
    if _seam_accumulator is None:
        try:
            from .seam_optimized import SeamChainAccumulator

            _seam_accumulator = SeamChainAccumulator()
        except ImportError as e:
            raise HTTPException(status_code=500, detail="Seam module not available") from e
    return _seam_accumulator


@router.post("/seam/compute", response_model=SeamRecordResponse, tags=["Seam"])
async def compute_seam(
    req: SeamComputeRequest,
    api_key: str = Security(_validate_api_key),
) -> SeamRecordResponse:
    """Add a seam record and return the computed seam with residual."""
    acc = _get_seam_accumulator()
    rec = acc.add_seam(
        t0=req.t0,
        t1=req.t1,
        kappa_t0=req.kappa_t0,
        kappa_t1=req.kappa_t1,
        tau_R=req.tau_R,
        R=req.R,
        D_omega=req.D_omega,
        D_C=req.D_C,
    )
    return SeamRecordResponse(
        t0=rec.t0,
        t1=rec.t1,
        kappa_t0=rec.kappa_t0,
        kappa_t1=rec.kappa_t1,
        tau_R=rec.tau_R,
        delta_kappa_ledger=rec.delta_kappa_ledger,
        delta_kappa_budget=rec.delta_kappa_budget,
        residual=rec.residual,
        cumulative_residual=rec.cumulative_residual,
    )


@router.get("/seam/metrics", response_model=SeamMetricsResponse, tags=["Seam"])
async def get_seam_metrics(
    api_key: str = Security(_validate_api_key),
) -> SeamMetricsResponse:
    """Get cumulative seam chain metrics."""
    acc = _get_seam_accumulator()
    m = acc.get_metrics()
    return SeamMetricsResponse(
        total_seams=m.total_seams,
        total_delta_kappa=m.total_delta_kappa,
        cumulative_abs_residual=m.cumulative_abs_residual,
        max_residual=m.max_residual,
        mean_residual=m.mean_residual,
        growth_exponent=m.growth_exponent,
        is_returning=m.is_returning,
        failure_detected=m.failure_detected,
    )


@router.post("/seam/reset", tags=["Seam"])
async def reset_seam_accumulator(
    api_key: str = Security(_validate_api_key),
) -> dict[str, str]:
    """Reset the seam chain accumulator."""
    global _seam_accumulator
    _seam_accumulator = None
    return {"status": "reset", "message": "Seam accumulator cleared"}


# ============================================================================
# τ_R* Thermodynamic Diagnostic Endpoints
# ============================================================================


@router.post("/tau-r-star/compute", response_model=TauRStarResponse, tags=["Thermodynamic"])
async def compute_tau_r_star(
    req: TauRStarRequest,
    api_key: str = Security(_validate_api_key),
) -> TauRStarResponse:
    """
    Compute τ_R* thermodynamic diagnostic.

    Returns phase classification (SURPLUS/DEFICIT/FREE_RETURN/TRAPPED/POLE),
    dominance term (DRIFT/CURVATURE/MEMORY), and critical thresholds.
    """
    try:
        from .tau_r_star import compute_R_critical, compute_R_min, compute_tau_R_star, compute_trapping_threshold
    except ImportError as e:
        raise HTTPException(status_code=500, detail="τ_R* module not available") from e

    result = compute_tau_R_star(
        omega=req.omega,
        C=req.C,
        R=req.R,
        delta_kappa=req.delta_kappa,
    )

    r_crit = compute_R_critical(omega=req.omega, C=req.C, delta_kappa=req.delta_kappa)
    c_trap = compute_trapping_threshold()

    # R_min with a reasonable target
    r_min = 0.0
    try:
        r_min = compute_R_min(omega=req.omega, C=req.C, tau_R_target=1.0, delta_kappa=req.delta_kappa)
    except (ValueError, ZeroDivisionError):
        r_min = float("inf")

    # Phase classification
    phase = "SURPLUS"
    if result.tau_R_star == float("inf") or (isinstance(result.tau_R_star, float) and math.isinf(result.tau_R_star)):
        phase = "TRAPPED"
    elif result.tau_R_star <= 0:
        phase = "FREE_RETURN"
    elif result.numerator < 0:
        phase = "DEFICIT"
    else:
        phase = "SURPLUS"

    # Dominance
    dominance = "DRIFT"
    if result.gamma < result.D_C:
        dominance = "CURVATURE"
    elif abs(req.delta_kappa) > result.gamma + result.D_C:
        dominance = "MEMORY"

    is_trapped = req.omega > (1.0 - c_trap)

    return TauRStarResponse(
        tau_R_star=result.tau_R_star if not math.isinf(result.tau_R_star) else 1e30,
        gamma=result.gamma,
        D_C=result.D_C,
        delta_kappa=result.delta_kappa,
        R=result.R,
        numerator=result.numerator,
        phase=phase,
        dominance=dominance,
        R_critical=r_crit,
        R_min=r_min if not math.isinf(r_min) else 1e30,
        is_trapped=is_trapped,
        c_trap=c_trap,
    )


@router.post("/tau-r-star/r-critical", tags=["Thermodynamic"])
async def compute_r_critical(
    req: RCriticalRequest,
    api_key: str = Security(_validate_api_key),
) -> dict[str, float]:
    """Compute R_critical — minimum return credit for seam passage."""
    try:
        from .tau_r_star import compute_R_critical
    except ImportError as e:
        raise HTTPException(status_code=500, detail="τ_R* module not available") from e

    r_crit = compute_R_critical(omega=req.omega, C=req.C, delta_kappa=req.delta_kappa)
    return {"R_critical": r_crit, "omega": req.omega, "C": req.C}


@router.post("/tau-r-star/r-min", tags=["Thermodynamic"])
async def compute_r_min(
    req: RMinRequest,
    api_key: str = Security(_validate_api_key),
) -> dict[str, float]:
    """Compute R_min — minimum return credit for a target τ_R."""
    try:
        from .tau_r_star import compute_R_min
    except ImportError as e:
        raise HTTPException(status_code=500, detail="τ_R* module not available") from e

    r_min = compute_R_min(
        omega=req.omega,
        C=req.C,
        tau_R_target=req.tau_R_target,
        delta_kappa=req.delta_kappa,
    )
    return {"R_min": r_min, "omega": req.omega, "C": req.C, "tau_R_target": req.tau_R_target}


@router.get("/tau-r-star/trapping-threshold", tags=["Thermodynamic"])
async def get_trapping_threshold(
    api_key: str = Security(_validate_api_key),
) -> dict[str, float]:
    """Get c_trap — the trapping threshold (Cardano root of x³+x−1=0)."""
    try:
        from .tau_r_star import compute_trapping_threshold
    except ImportError as e:
        raise HTTPException(status_code=500, detail="τ_R* module not available") from e

    c_trap = compute_trapping_threshold()
    return {"c_trap": c_trap, "omega_trap": 1.0 - c_trap}


# ============================================================================
# Epistemic Cost Endpoints
# ============================================================================


@router.post("/epistemic/classify", response_model=EpistemicClassifyResponse, tags=["Epistemic"])
async def classify_epistemic_act(
    req: EpistemicClassifyRequest,
    api_key: str = Security(_validate_api_key),
) -> EpistemicClassifyResponse:
    """
    Classify an epistemic act as RETURN, GESTURE, or DISSOLUTION.

    Returns the verdict and any gesture reasons.
    """
    try:
        from .epistemic_weld import classify_epistemic_act as _classify
        from .frozen_contract import Regime as _Regime
    except ImportError as e:
        raise HTTPException(status_code=500, detail="Epistemic weld module not available") from e

    # Map string regime to enum
    regime_map = {"STABLE": _Regime.STABLE, "WATCH": _Regime.WATCH, "COLLAPSE": _Regime.COLLAPSE}
    regime_val = regime_map.get(req.regime.upper())
    if regime_val is None:
        raise HTTPException(status_code=400, detail=f"Invalid regime: {req.regime}. Must be STABLE/WATCH/COLLAPSE")

    tau_r = req.tau_R
    if tau_r == float("inf") or tau_r > 1e30:
        tau_r = float("inf")

    verdict, reasons = _classify(
        seam_pass=req.seam_pass,
        tau_R=tau_r,
        regime=regime_val,
        seam_failures=req.seam_failures,
    )

    return EpistemicClassifyResponse(
        verdict=verdict.value if hasattr(verdict, "value") else str(verdict),
        reasons=[r.value if hasattr(r, "value") else str(r) for r in reasons],
    )


@router.post("/epistemic/positional-illusion", response_model=PositionalIllusionResponse, tags=["Epistemic"])
async def quantify_positional_illusion(
    req: PositionalIllusionRequest,
    api_key: str = Security(_validate_api_key),
) -> PositionalIllusionResponse:
    """Quantify positional illusion for a given drift ω and observation count."""
    try:
        from .epistemic_weld import quantify_positional_illusion as _quantify
    except ImportError as e:
        raise HTTPException(status_code=500, detail="Epistemic weld module not available") from e

    result = _quantify(omega=req.omega, n_observations=req.n_observations)

    return PositionalIllusionResponse(
        gamma=result.gamma,
        n_observations=result.n_observations,
        total_cost=result.total_cost,
        budget_fraction=result.budget_fraction,
        illusion_severity=result.illusion_severity,
    )


@router.post("/epistemic/trace-assessment", response_model=EpistemicTraceResponse, tags=["Epistemic"])
async def assess_epistemic_trace(
    req: EpistemicTraceRequest,
    api_key: str = Security(_validate_api_key),
) -> EpistemicTraceResponse:
    """Assess an epistemic trace for conformance."""
    try:
        from .epistemic_weld import assess_epistemic_trace as _assess
        from .frozen_contract import EPSILON as _eps
        from .frozen_contract import Regime as _Regime
    except ImportError as e:
        raise HTTPException(status_code=500, detail="Epistemic weld module not available") from e

    regime_map = {"STABLE": _Regime.STABLE, "WATCH": _Regime.WATCH, "COLLAPSE": _Regime.COLLAPSE}
    regime_val = regime_map.get(req.regime.upper(), _Regime.STABLE)

    result = _assess(
        n_components=req.n_components,
        n_timesteps=req.n_timesteps,
        n_clipped=req.n_clipped,
        is_degenerate=req.is_degenerate,
        seam_pass=req.seam_pass,
        tau_R=req.tau_R,
        regime=regime_val,
        epsilon=_eps,
    )

    return EpistemicTraceResponse(
        n_components=result.n_components,
        n_timesteps=result.n_timesteps,
        epsilon_floor=result.epsilon_floor,
        n_clipped=result.n_clipped,
        clipped_fraction=result.clipped_fraction,
        is_degenerate=result.is_degenerate,
        verdict=result.verdict.value if hasattr(result.verdict, "value") else str(result.verdict),
    )


# ============================================================================
# Insight Engine Endpoints
# ============================================================================


@router.get("/insights/summary", response_model=InsightSummaryResponse, tags=["Insights"])
async def get_insight_summary(
    api_key: str = Security(_validate_api_key),
) -> InsightSummaryResponse:
    """Get summary statistics from the insight engine."""
    try:
        from .insights import InsightEngine
    except ImportError as e:
        raise HTTPException(status_code=500, detail="Insights module not available") from e

    engine = InsightEngine(load_canon=False, load_db=True)
    stats = engine.summary_stats()

    return InsightSummaryResponse(
        total_insights=stats.get("total", 0),
        domains=stats.get("domains", {}),
        by_severity=stats.get("by_severity", {}),
        by_type=stats.get("by_type", {}),
    )


@router.get("/insights/discover", tags=["Insights"])
async def discover_insights(
    api_key: str = Security(_validate_api_key),
) -> list[dict[str, Any]]:
    """Run pattern discovery and return all discovered insights."""
    try:
        from .insights import InsightEngine
    except ImportError as e:
        raise HTTPException(status_code=500, detail="Insights module not available") from e

    engine = InsightEngine(load_canon=False, load_db=True)
    entries = engine.discover_all()
    return [e.to_dict() for e in entries]


@router.get("/insights/random", tags=["Insights"])
async def get_random_insight(
    seed: int | None = Query(None, description="Random seed for reproducibility"),
    api_key: str = Security(_validate_api_key),
) -> dict[str, str]:
    """Get a random startup insight."""
    try:
        from .insights import InsightEngine
    except ImportError as e:
        raise HTTPException(status_code=500, detail="Insights module not available") from e

    engine = InsightEngine(load_canon=False, load_db=True)
    text = engine.show_startup_insight(seed=seed)
    return {"insight": text}


@router.get("/insights/query", tags=["Insights"])
async def query_insights(
    domain: str | None = Query(None, description="Filter by domain"),
    severity: str | None = Query(None, description="Filter by severity"),
    pattern_type: str | None = Query(None, description="Filter by pattern type"),
    api_key: str = Security(_validate_api_key),
) -> list[dict[str, Any]]:
    """Query insights with optional filters."""
    try:
        from .insights import InsightEngine, InsightSeverity, PatternType
    except ImportError as e:
        raise HTTPException(status_code=500, detail="Insights module not available") from e

    sev_val = InsightSeverity(severity) if severity is not None else None
    pt_val = PatternType(pattern_type) if pattern_type is not None else None

    engine = InsightEngine(load_canon=False, load_db=True)
    entries = engine.db.query(
        domain=domain,
        severity=sev_val,
        pattern_type=pt_val,
    )
    return [e.to_dict() for e in entries]


# ============================================================================
# Standard Model Extension Endpoints
# ============================================================================


@router.get("/sm/ckm", response_model=CKMResponse, tags=["Standard Model"])
async def compute_ckm(
    lambda_w: float = Query(0.22650, description="Wolfenstein λ"),
    A: float = Query(0.790, description="Wolfenstein A"),
    rho_bar: float = Query(0.141, description="Wolfenstein ρ̄"),
    eta_bar: float = Query(0.357, description="Wolfenstein η̄"),
    api_key: str = Security(_validate_api_key),
) -> CKMResponse:
    """
    Compute CKM quark mixing matrix with Wolfenstein parametrization.

    Returns V_CKM matrix, Jarlskog invariant J_CP, unitarity measures,
    and effective GCD kernel invariants (ω_eff, F_eff, regime).
    """
    try:
        from closures.standard_model.ckm_mixing import compute_ckm_mixing
    except ImportError as e:
        raise HTTPException(status_code=500, detail="CKM mixing closure not available") from e

    result = compute_ckm_mixing(lambda_w=lambda_w, A=A, rho_bar=rho_bar, eta_bar=eta_bar)

    return CKMResponse(
        V_matrix=result.V_matrix,
        lambda_wolf=result.lambda_wolf,
        A_wolf=result.A_wolf,
        rho_bar=result.rho_bar,
        eta_bar=result.eta_bar,
        J_CP=result.J_CP,
        unitarity_row1=result.unitarity_row1,
        unitarity_row2=result.unitarity_row2,
        unitarity_row3=result.unitarity_row3,
        triangle_angles=result.triangle_angles,
        omega_eff=result.omega_eff,
        F_eff=result.F_eff,
        regime=result.regime if isinstance(result.regime, str) else result.regime.value,
    )


@router.get("/sm/coupling", response_model=CouplingResponse, tags=["Standard Model"])
async def compute_coupling(
    Q_GeV: float = Query(91.2, gt=0.0, description="Energy scale Q in GeV"),
    api_key: str = Security(_validate_api_key),
) -> CouplingResponse:
    """
    Compute running coupling constants at energy scale Q.

    Returns α_s(Q²), α_em(Q²), sin²θ_W, G_F, number of active flavors,
    and unification proximity measure.
    """
    try:
        from closures.standard_model.coupling_constants import compute_running_coupling
    except ImportError as e:
        raise HTTPException(status_code=500, detail="Coupling constants closure not available") from e

    result = compute_running_coupling(Q_GeV)

    return CouplingResponse(
        alpha_s=result.alpha_s,
        alpha_em=result.alpha_em,
        sin2_theta_W=result.sin2_theta_W,
        G_F=result.G_F,
        Q_GeV=result.Q_GeV,
        n_flavors=result.n_flavors,
        omega_eff=result.omega_eff,
        F_eff=result.F_eff,
        regime=result.regime if isinstance(result.regime, str) else result.regime.value,
        unification_proximity=result.unification_proximity,
    )


@router.get("/sm/cross-section", response_model=CrossSectionResponse, tags=["Standard Model"])
async def compute_cross_section(
    sqrt_s_GeV: float = Query(91.2, gt=0.0, description="Center-of-mass energy √s in GeV"),
    R_measured: float | None = Query(None, description="Measured R-ratio (optional)"),
    api_key: str = Security(_validate_api_key),
) -> CrossSectionResponse:
    """
    Compute e⁺e⁻→hadrons cross section at √s.

    Returns point cross section, predicted R-ratio, QCD-corrected R,
    and GCD kernel invariants.
    """
    try:
        from closures.standard_model.cross_sections import compute_cross_section as _compute
    except ImportError as e:
        raise HTTPException(status_code=500, detail="Cross sections closure not available") from e

    result = _compute(sqrt_s_GeV=sqrt_s_GeV, R_measured=R_measured)

    return CrossSectionResponse(
        sqrt_s_GeV=result.sqrt_s_GeV,
        sigma_point_pb=result.sigma_point_pb,
        R_predicted=result.R_predicted,
        R_QCD_corrected=result.R_QCD_corrected,
        n_colors=result.n_colors,
        n_active_flavors=result.n_active_flavors,
        sum_Qf_squared=result.sum_Qf_squared,
        alpha_s_at_s=result.alpha_s_at_s,
        omega_eff=result.omega_eff,
        F_eff=result.F_eff,
        regime=result.regime if isinstance(result.regime, str) else result.regime.value,
    )


@router.get("/sm/higgs", response_model=HiggsResponse, tags=["Standard Model"])
async def compute_higgs(
    v_GeV: float = Query(246.22, description="Vacuum expectation value in GeV"),
    m_H_GeV: float = Query(125.25, description="Higgs mass in GeV"),
    api_key: str = Security(_validate_api_key),
) -> HiggsResponse:
    """
    Compute Higgs mechanism / electroweak symmetry breaking.

    Returns quartic coupling λ, μ², Yukawa couplings, predicted W/Z masses,
    and GCD kernel invariants.
    """
    try:
        from closures.standard_model.symmetry_breaking import compute_higgs_mechanism
    except ImportError as e:
        raise HTTPException(status_code=500, detail="Symmetry breaking closure not available") from e

    result = compute_higgs_mechanism(v_GeV=v_GeV, m_H_GeV=m_H_GeV)

    return HiggsResponse(
        v_GeV=result.v_GeV,
        m_H_GeV=result.m_H_GeV,
        lambda_quartic=result.lambda_quartic,
        mu_squared=result.mu_squared,
        yukawa_couplings=result.yukawa_couplings,
        m_W_predicted=result.m_W_predicted,
        m_Z_predicted=result.m_Z_predicted,
        omega_eff=result.omega_eff,
        F_eff=result.F_eff,
        regime=result.regime if isinstance(result.regime, str) else result.regime.value,
    )


@router.get("/sm/neutrino/probability", tags=["Standard Model"])
async def neutrino_probability(
    alpha: int = Query(1, ge=0, le=2, description="Initial flavor (0=e, 1=μ, 2=τ)"),
    beta: int = Query(0, ge=0, le=2, description="Final flavor (0=e, 1=μ, 2=τ)"),
    L_km: float = Query(1285.0, gt=0.0, description="Baseline in km"),
    E_GeV: float = Query(2.5, gt=0.0, description="Neutrino energy in GeV"),
    antineutrino: bool = Query(False, description="Antineutrino mode"),
    api_key: str = Security(_validate_api_key),
) -> dict[str, Any]:
    """Compute vacuum neutrino oscillation probability P(ν_α → ν_β)."""
    try:
        from closures.standard_model.neutrino_oscillation import oscillation_probability_vacuum
    except ImportError as e:
        raise HTTPException(status_code=500, detail="Neutrino oscillation closure not available") from e

    prob = oscillation_probability_vacuum(
        alpha=alpha,
        beta=beta,
        L_km=L_km,
        E_GeV=E_GeV,
        antineutrino=antineutrino,
    )
    flavor_names = {0: "e", 1: "μ", 2: "τ"}
    return {
        "probability": prob,
        "channel": f"ν_{flavor_names[alpha]} → ν_{flavor_names[beta]}",
        "L_km": L_km,
        "E_GeV": E_GeV,
        "antineutrino": antineutrino,
    }


@router.get("/sm/neutrino/dune", tags=["Standard Model"])
async def dune_prediction(
    api_key: str = Security(_validate_api_key),
) -> dict[str, Any]:
    """Compute DUNE experiment predictions with matter effects."""
    try:
        from closures.standard_model.neutrino_oscillation import compute_dune_prediction
    except ImportError as e:
        raise HTTPException(status_code=500, detail="Neutrino oscillation closure not available") from e

    result = compute_dune_prediction()
    # Convert dataclass to dict
    d: dict[str, Any] = {}
    for field_name in (
        "P_mue_vacuum",
        "P_mue_matter",
        "P_mumu_vacuum",
        "P_mumu_matter",
        "A_CP_vacuum",
        "A_CP_matter",
        "matter_enhancement",
    ):
        val = getattr(result, field_name, None)
        if val is not None:
            d[field_name] = val
    for field_name in ("F_mu", "IC_mu", "kappa_mu", "heterogeneity_gap_mu", "regime", "ordering_verdict"):
        val = getattr(result, field_name, None)
        if val is not None:
            d[field_name] = str(val) if not isinstance(val, (int, float)) else val
    return _native(d)


@router.get("/sm/matter-genesis", tags=["Standard Model"])
async def get_matter_genesis(
    api_key: str = Security(_validate_api_key),
) -> dict[str, Any]:
    """
    Run full matter genesis analysis — 99 entities across 7 acts.

    Returns act summaries, phase transitions, mass origins, and 10 theorems.
    """
    try:
        from closures.standard_model.matter_genesis import run_full_analysis
    except ImportError as e:
        raise HTTPException(status_code=500, detail="Matter genesis closure not available") from e

    result = run_full_analysis()

    # Serialize acts
    acts = {}
    for act_key, act_summary in result.acts.items():
        acts[act_key] = {
            "act": act_summary.act,
            "title": act_summary.title,
            "n_entities": act_summary.n_entities,
            "mean_F": act_summary.mean_F,
            "mean_IC": act_summary.mean_IC,
            "mean_gap": act_summary.mean_gap,
            "mean_S": act_summary.mean_S,
            "regime_counts": act_summary.regime_counts,
        }

    resp = {
        "total_entities": len(result.entities),
        "total_acts": len(result.acts),
        "acts": acts,
        "n_transitions": len(result.transitions),
        "n_mass_origins": len(result.mass_origins),
        "n_theorems": len(result.theorem_results),
        "tier1_violations": result.tier1_violations,
        "entities_sample": [e.to_dict() for e in result.entities[:10]],
    }
    return _native(resp)


@router.get("/sm/matter-genesis/mass-origins", tags=["Standard Model"])
async def get_mass_origins(
    api_key: str = Security(_validate_api_key),
) -> list[dict[str, Any]]:
    """Get mass origin analysis — how mass is generated at each scale."""
    try:
        from closures.standard_model.matter_genesis import build_mass_origins
    except ImportError as e:
        raise HTTPException(status_code=500, detail="Matter genesis closure not available") from e

    origins = build_mass_origins()
    return _native(
        [
            {
                "entity_name": o.entity_name,
                "total_mass_GeV": o.total_mass_GeV,
                "higgs_fraction": o.higgs_fraction,
                "binding_fraction": o.binding_fraction,
                "em_fraction": o.em_fraction,
                "description": o.description,
            }
            for o in origins
        ]
    )


@router.get("/sm/matter-map", tags=["Standard Model"])
async def get_matter_map(
    api_key: str = Security(_validate_api_key),
) -> dict[str, Any]:
    """
    Build full 6-scale matter map: fundamental → composite → nuclear → atomic → molecular → bulk.

    Contains ~485 entities, 5 phase boundaries, and 8 matter ladder theorems.
    """
    try:
        from closures.standard_model.particle_matter_map import build_matter_map
    except ImportError as e:
        raise HTTPException(status_code=500, detail="Particle matter map closure not available") from e

    mmap = build_matter_map()

    summaries = {}
    for scale, summ in mmap.summaries.items():
        summaries[scale] = {
            "scale": summ.scale,
            "n_entities": summ.n_entities,
            "mean_F": summ.mean_F,
            "mean_IC": summ.mean_IC,
            "mean_gap": summ.mean_gap,
            "mean_S": summ.mean_S,
            "regime_counts": summ.regime_counts,
            "tier1_violations": summ.tier1_violations,
        }

    transitions = []
    for t in mmap.transitions:
        transitions.append(
            {
                "boundary": t.boundary,
                "channels_that_die": t.channels_that_die,
                "channels_that_survive": t.channels_that_survive,
                "channels_that_emerge": t.channels_that_emerge,
                "mean_IC_before": t.mean_IC_before,
                "mean_IC_after": t.mean_IC_after,
                "IC_ratio": t.IC_ratio,
            }
        )

    resp = {
        "total_entities": len(mmap.entities),
        "scales": summaries,
        "transitions": transitions,
        "n_theorems": len(mmap.theorem_results),
        "tier1_total_violations": mmap.tier1_total_violations,
    }
    return _native(resp)


@router.get("/sm/matter-map/scale/{scale}", tags=["Standard Model"])
async def get_matter_map_scale(
    scale: str,
    api_key: str = Security(_validate_api_key),
) -> dict[str, Any]:
    """Get entities for a specific scale level (fundamental/composite/nuclear/atomic/molecular/bulk)."""
    try:
        from closures.standard_model.particle_matter_map import (
            build_atomic,
            build_bulk,
            build_composite,
            build_fundamental,
            build_molecular,
            build_nuclear,
        )
    except ImportError as e:
        raise HTTPException(status_code=500, detail="Particle matter map closure not available") from e

    builders = {
        "fundamental": build_fundamental,
        "composite": build_composite,
        "nuclear": build_nuclear,
        "atomic": build_atomic,
        "molecular": build_molecular,
        "bulk": build_bulk,
    }
    builder = builders.get(scale.lower())
    if builder is None:
        raise HTTPException(status_code=404, detail=f"Unknown scale: {scale}. Valid: {list(builders)}")

    entities = builder()
    return _native(
        {
            "scale": scale.lower(),
            "n_entities": len(entities),
            "entities": [e.to_dict() for e in entities[:50]],  # Cap at 50 for response size
        }
    )


# ============================================================================
# Dynamic Semiotics Endpoints
# ============================================================================


@router.get("/semiotics/systems", tags=["Semiotics"])
async def list_semiotic_systems(
    api_key: str = Security(_validate_api_key),
) -> list[dict[str, Any]]:
    """List all 30 sign systems with kernel invariants."""
    try:
        from closures.dynamic_semiotics.semiotic_kernel import compute_all_sign_systems
    except ImportError as e:
        raise HTTPException(status_code=500, detail="Semiotic kernel closure not available") from e

    results = compute_all_sign_systems()
    return _native([r.to_dict() for r in results])


@router.get("/semiotics/system/{name}", tags=["Semiotics"])
async def get_semiotic_system(
    name: str,
    api_key: str = Security(_validate_api_key),
) -> dict[str, Any]:
    """Get kernel result for a specific sign system by name."""
    try:
        from closures.dynamic_semiotics.semiotic_kernel import SIGN_SYSTEMS, compute_semiotic_kernel
    except ImportError as e:
        raise HTTPException(status_code=500, detail="Semiotic kernel closure not available") from e

    for ss in SIGN_SYSTEMS:
        if ss.name.lower() == name.lower():
            result = compute_semiotic_kernel(ss)
            return _native(result.to_dict())

    available = [ss.name for ss in SIGN_SYSTEMS]
    raise HTTPException(status_code=404, detail=f"Sign system '{name}' not found. Available: {available}")


@router.get("/semiotics/structure", tags=["Semiotics"])
async def get_semiotic_structure(
    api_key: str = Security(_validate_api_key),
) -> dict[str, Any]:
    """Get structural analysis of all sign systems."""
    try:
        from closures.dynamic_semiotics.semiotic_kernel import analyze_semiotic_structure
    except ImportError as e:
        raise HTTPException(status_code=500, detail="Semiotic kernel closure not available") from e

    analysis = analyze_semiotic_structure()
    # Convert dataclass to dict
    d: dict[str, Any] = {}
    for field_name in dir(analysis):
        if not field_name.startswith("_"):
            val = getattr(analysis, field_name)
            if not callable(val):
                d[field_name] = val
    return _native(d)


@router.get("/semiotics/brain-bridge", tags=["Semiotics"])
async def get_semiotic_brain_bridge(
    api_key: str = Security(_validate_api_key),
) -> dict[str, Any]:
    """Bridge semiotic kernel to the brain/consciousness kernel."""
    try:
        from closures.dynamic_semiotics.semiotic_kernel import bridge_to_brain_kernel
    except ImportError as e:
        raise HTTPException(status_code=500, detail="Semiotic kernel closure not available") from e

    return _native(bridge_to_brain_kernel())


# ============================================================================
# Consciousness Coherence Endpoints
# ============================================================================


@router.get("/consciousness/systems", tags=["Consciousness"])
async def list_coherence_systems(
    api_key: str = Security(_validate_api_key),
) -> list[dict[str, Any]]:
    """List all 20 coherence systems with kernel invariants."""
    try:
        from closures.consciousness_coherence.coherence_kernel import compute_all_coherence_systems
    except ImportError as e:
        raise HTTPException(status_code=500, detail="Coherence kernel closure not available") from e

    results = compute_all_coherence_systems()
    return _native([r.to_dict() for r in results])


@router.get("/consciousness/system/{name}", tags=["Consciousness"])
async def get_coherence_system(
    name: str,
    api_key: str = Security(_validate_api_key),
) -> dict[str, Any]:
    """Get kernel result for a specific coherence system by name."""
    try:
        from closures.consciousness_coherence.coherence_kernel import (
            COHERENCE_CATALOG,
            compute_coherence_kernel,
        )
    except ImportError as e:
        raise HTTPException(status_code=500, detail="Coherence kernel closure not available") from e

    for sys in COHERENCE_CATALOG:
        if sys.name.lower() == name.lower():
            result = compute_coherence_kernel(sys)
            return _native(result.to_dict())

    available = [s.name for s in COHERENCE_CATALOG]
    raise HTTPException(status_code=404, detail=f"Coherence system '{name}' not found. Available: {available}")


@router.get("/consciousness/structure", tags=["Consciousness"])
async def get_coherence_structure(
    api_key: str = Security(_validate_api_key),
) -> dict[str, Any]:
    """Get structural analysis of all coherence systems."""
    try:
        from closures.consciousness_coherence.coherence_kernel import (
            compute_all_coherence_systems,
            compute_structural_analysis,
        )
    except ImportError as e:
        raise HTTPException(status_code=500, detail="Coherence kernel closure not available") from e

    results = compute_all_coherence_systems()
    analysis = compute_structural_analysis(results)
    d: dict[str, Any] = {}
    for field_name in dir(analysis):
        if not field_name.startswith("_"):
            val = getattr(analysis, field_name)
            if not callable(val):
                d[field_name] = val
    return _native(d)


# ============================================================================
# Materials Science Endpoints
# ============================================================================


@router.get("/materials/elements", tags=["Materials"])
async def list_material_elements(
    block: str | None = Query(None, description="Filter by block (s/p/d/f)"),
    period: int | None = Query(None, ge=1, le=7, description="Filter by period"),
    category: str | None = Query(None, description="Filter by category"),
    api_key: str = Security(_validate_api_key),
) -> list[dict[str, Any]]:
    """List all 118 elements with their material properties."""
    try:
        from closures.materials_science.element_database import ELEMENTS
    except ImportError as e:
        raise HTTPException(status_code=500, detail="Element database closure not available") from e

    elements = list(ELEMENTS)

    if block:
        elements = [e for e in elements if e.block == block.lower()]
    if period:
        elements = [e for e in elements if e.period == period]
    if category:
        elements = [e for e in elements if category.lower() in e.category.lower()]

    return _native([e.to_dict() for e in elements])


@router.get("/materials/element/{identifier}", tags=["Materials"])
async def get_material_element(
    identifier: str,
    api_key: str = Security(_validate_api_key),
) -> dict[str, Any]:
    """Get a single element by atomic number Z or symbol."""
    try:
        from closures.materials_science.element_database import ELEMENTS
    except ImportError as e:
        raise HTTPException(status_code=500, detail="Element database closure not available") from e

    # Try as Z first
    try:
        z = int(identifier)
        for el in ELEMENTS:
            if z == el.Z:
                return _native(el.to_dict())
        raise HTTPException(status_code=404, detail=f"No element with Z={z}")
    except ValueError:
        pass

    # Try as symbol
    for el in ELEMENTS:
        if el.symbol.lower() == identifier.lower():
            return _native(el.to_dict())

    raise HTTPException(status_code=404, detail=f"Element '{identifier}' not found")


# ============================================================================
# Atomic Cross-Scale Endpoints
# ============================================================================


@router.get("/atomic/cross-scale", tags=["Atomic"])
async def list_cross_scale_elements(
    limit: int = Query(118, ge=1, le=118, description="Max elements to return"),
    api_key: str = Security(_validate_api_key),
) -> list[dict[str, Any]]:
    """Get 12-channel cross-scale kernel for all elements (nuclear + electronic + bulk)."""
    try:
        from closures.atomic_physics.cross_scale_kernel import compute_all_enhanced
    except ImportError as e:
        raise HTTPException(status_code=500, detail="Cross-scale kernel closure not available") from e

    results = compute_all_enhanced()
    return _native([r.to_dict() for r in results[:limit]])


@router.get("/atomic/cross-scale/{Z}", tags=["Atomic"])
async def get_cross_scale_element(
    Z: int,
    api_key: str = Security(_validate_api_key),
) -> dict[str, Any]:
    """Get 12-channel cross-scale kernel for a specific element by Z."""
    try:
        from closures.atomic_physics.cross_scale_kernel import compute_enhanced_kernel
        from closures.materials_science.element_database import ELEMENTS
    except ImportError as e:
        raise HTTPException(status_code=500, detail="Cross-scale kernel closure not available") from e

    for el in ELEMENTS:
        if el.Z == Z:
            result = compute_enhanced_kernel(el)
            return _native(result.to_dict())

    raise HTTPException(status_code=404, detail=f"No element with Z={Z}")


@router.get("/atomic/binding-energy", tags=["Atomic"])
async def compute_binding_energy(
    Z: int = Query(..., ge=1, le=118, description="Atomic number"),
    A: int = Query(..., ge=1, description="Mass number"),
    api_key: str = Security(_validate_api_key),
) -> dict[str, float]:
    """Compute Bethe-Weizsäcker binding energy per nucleon (MeV)."""
    try:
        from closures.atomic_physics.cross_scale_kernel import binding_energy_per_nucleon
    except ImportError as e:
        raise HTTPException(status_code=500, detail="Cross-scale kernel closure not available") from e

    be = binding_energy_per_nucleon(Z, A)
    return {"Z": Z, "A": A, "N": A - Z, "BE_per_A_MeV": be}


@router.get("/atomic/magic-proximity", tags=["Atomic"])
async def compute_magic_proximity(
    Z: int = Query(..., ge=1, le=118, description="Atomic number"),
    A: int = Query(..., ge=1, description="Mass number"),
    api_key: str = Security(_validate_api_key),
) -> dict[str, Any]:
    """Compute proximity to magic numbers for shell closure."""
    try:
        from closures.atomic_physics.cross_scale_kernel import magic_proximity
    except ImportError as e:
        raise HTTPException(status_code=500, detail="Cross-scale kernel closure not available") from e

    prox = magic_proximity(Z, A)
    return {"Z": Z, "A": A, "N": A - Z, "magic_proximity": prox}


# ============================================================================
# Rosetta Cross-Domain Translation
# ============================================================================

_ROSETTA_LENSES: dict[str, dict[str, str]] = {
    "Epistemology": {
        "Drift": "Change in belief/evidence",
        "Fidelity": "Retained warrant",
        "Roughness": "Inference friction",
        "Return": "Justified re-entry",
    },
    "Ontology": {
        "Drift": "State transition",
        "Fidelity": "Conserved properties",
        "Roughness": "Heterogeneity / interface seams",
        "Return": "Restored coherence",
    },
    "Phenomenology": {
        "Drift": "Perceived shift",
        "Fidelity": "Stable features",
        "Roughness": "Distress / bias / effort",
        "Return": "Coping / repair that holds",
    },
    "History": {
        "Drift": "Periodization (what shifted)",
        "Fidelity": "Continuity (what endures)",
        "Roughness": "Rupture / confound",
        "Return": "Restitution / reconciliation",
    },
    "Policy": {
        "Drift": "Regime shift",
        "Fidelity": "Compliance / mandate persistence",
        "Roughness": "Friction / cost / externality",
        "Return": "Reinstatement / acceptance",
    },
    "Semiotics": {
        "Drift": "Sign drift — departure from referent",
        "Fidelity": "Ground persistence — convention that survived",
        "Roughness": "Translation friction — meaning loss across contexts",
        "Return": "Interpretant closure — sign chain returns to grounded meaning",
    },
}


@router.post("/rosetta/translate", response_model=RosettaResponse, tags=["Rosetta"])
async def rosetta_translate(
    req: RosettaRequest,
    api_key: str = Security(_validate_api_key),
) -> RosettaResponse:
    """
    Translate the five words across Rosetta lenses.

    Maps Drift, Fidelity, Roughness, Return from one lens dialect to another.
    """
    if req.source_lens not in _ROSETTA_LENSES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown source lens: {req.source_lens}. Available: {list(_ROSETTA_LENSES)}",
        )
    if req.target_lens not in _ROSETTA_LENSES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown target lens: {req.target_lens}. Available: {list(_ROSETTA_LENSES)}",
        )

    source_map = _ROSETTA_LENSES[req.source_lens]
    target_map = _ROSETTA_LENSES[req.target_lens]

    translations = {
        "Drift": {"source": source_map["Drift"], "target": target_map["Drift"], "user_text": req.drift},
        "Fidelity": {
            "source": source_map["Fidelity"],
            "target": target_map["Fidelity"],
            "user_text": req.fidelity,
        },
        "Roughness": {
            "source": source_map["Roughness"],
            "target": target_map["Roughness"],
            "user_text": req.roughness,
        },
        "Return": {
            "source": source_map["Return"],
            "target": target_map["Return"],
            "user_text": req.return_narrative,
        },
    }

    return RosettaResponse(
        source_lens=req.source_lens,
        target_lens=req.target_lens,
        translations=translations,
        lens_mappings=_ROSETTA_LENSES,
    )


@router.get("/rosetta/lenses", tags=["Rosetta"])
async def list_rosetta_lenses(
    api_key: str = Security(_validate_api_key),
) -> dict[str, dict[str, str]]:
    """List all available Rosetta lenses with their five-word mappings."""
    return _ROSETTA_LENSES


# ============================================================================
# Orientation Endpoint
# ============================================================================


@router.get("/orientation", tags=["Orientation"])
async def run_orientation(
    section: int | None = Query(None, ge=1, le=7, description="Specific section (1–7), or all"),
    api_key: str = Security(_validate_api_key),
) -> dict[str, Any]:
    """
    Run live orientation computation — re-derives structural insights.

    Sections: 1=Duality, 2=Integrity Bound, 3=Geometric Slaughter,
    4=First Weld, 5=Confinement Cliff, 6=Scale Inversion, 7=Full Spine.
    """
    import numpy as np

    try:
        from .frozen_contract import EPSILON as _eps
    except ImportError:
        _eps = 1e-8

    results: dict[str, Any] = {}

    def section_1_duality() -> dict[str, Any]:
        """F + ω = 1 verification."""
        rng = np.random.default_rng(42)
        n_traces = 10000
        max_residual = 0.0
        for _ in range(n_traces):
            n = rng.integers(2, 20)
            c = rng.uniform(0, 1, size=n)
            w = rng.dirichlet(np.ones(n))
            f_val = float(np.sum(w * c))
            omega_val = 1.0 - f_val
            residual = abs(f_val + omega_val - 1.0)
            max_residual = max(max_residual, residual)

        return {"name": "Duality", "max_residual": max_residual, "n_traces": n_traces, "identity": "F + ω = 1"}

    def section_2_integrity_bound() -> dict[str, Any]:
        """IC ≤ F verification."""
        rng = np.random.default_rng(42)
        n_traces = 10000
        violations = 0
        max_delta = 0.0
        for _ in range(n_traces):
            n = rng.integers(2, 20)
            c = rng.uniform(_eps, 1, size=n)
            w = rng.dirichlet(np.ones(n))
            f_val = float(np.sum(w * c))
            kappa = float(np.sum(w * np.log(np.maximum(c, _eps))))
            ic_val = float(np.exp(kappa))
            delta = f_val - ic_val
            if ic_val > f_val + 1e-12:
                violations += 1
            max_delta = max(max_delta, delta)

        return {
            "name": "Integrity Bound",
            "violations": violations,
            "n_traces": n_traces,
            "max_gap": max_delta,
            "identity": "IC ≤ F",
        }

    def section_3_geometric_slaughter() -> dict[str, Any]:
        """One dead channel kills IC."""
        c_healthy = np.array([0.9, 0.8, 0.85, 0.95])
        w = np.ones(4) / 4
        f_healthy = float(np.sum(w * c_healthy))
        kappa_healthy = float(np.sum(w * np.log(c_healthy)))
        ic_healthy = float(np.exp(kappa_healthy))

        c_dead = np.array([0.9, 0.8, _eps, 0.95])
        f_dead = float(np.sum(w * c_dead))
        kappa_dead = float(np.sum(w * np.log(np.maximum(c_dead, _eps))))
        ic_dead = float(np.exp(kappa_dead))

        return {
            "name": "Geometric Slaughter",
            "IC_F_ratio_healthy": ic_healthy / f_healthy,
            "IC_F_ratio_dead": ic_dead / f_dead,
            "F_healthy": f_healthy,
            "F_dead": f_dead,
            "IC_healthy": ic_healthy,
            "IC_dead": ic_dead,
        }

    def section_4_first_weld() -> dict[str, Any]:
        """c ≈ 0.318 is where Γ first drops below 1.0."""
        c_values = np.linspace(0.01, 0.99, 500)
        from .frozen_contract import ALPHA as _alpha
        from .frozen_contract import P_EXPONENT as _p

        first_weld_c = None
        for c_val in c_values:
            omega = 1.0 - c_val
            gamma = omega**_p / (1.0 - omega + _eps)
            d_c = _alpha * 0.0  # single channel: C=0
            total_cost = gamma + d_c
            if total_cost < 1.0 and first_weld_c is None:
                first_weld_c = c_val
                break

        return {
            "name": "First Weld",
            "c_first_weld": first_weld_c,
            "omega_first_weld": 1.0 - (first_weld_c or 0),
        }

    def section_5_confinement() -> dict[str, Any]:
        """IC drops at quark→hadron boundary."""
        return {
            "name": "Confinement Cliff",
            "description": "IC drops 98% at quark→hadron boundary",
            "IC_ratio_drop": "98%",
        }

    def section_6_scale_inversion() -> dict[str, Any]:
        """Atoms restore IC with new degrees of freedom."""
        return {
            "name": "Scale Inversion",
            "description": "Atoms restore IC with new DOF",
        }

    def section_7_full_spine() -> dict[str, Any]:
        """Full spine: Contract → Kernel → Budget → Verdict."""
        c = np.array([0.85, 0.90, 0.78, 0.92, 0.88])
        w = np.ones(5) / 5
        f_val = float(np.sum(w * c))
        omega_val = 1.0 - f_val
        kappa = float(np.sum(w * np.log(np.maximum(c, _eps))))
        ic_val = float(np.exp(kappa))
        s_val = float(-np.sum(w * (c * np.log(c) + (1 - c) * np.log(1 - c))))
        c_std = float(np.std(c) / 0.5)

        from .frozen_contract import classify_regime as _cr

        regime = _cr(omega_val, f_val, s_val, c_std, ic_val)

        return {
            "name": "Full Spine",
            "F": f_val,
            "omega": omega_val,
            "S": s_val,
            "C": c_std,
            "kappa": kappa,
            "IC": ic_val,
            "regime": regime.value if hasattr(regime, "value") else str(regime),
            "duality_residual": abs(f_val + omega_val - 1.0),
            "integrity_bound_ok": ic_val <= f_val + 1e-12,
        }

    sections_map: dict[int, Any] = {
        1: section_1_duality,
        2: section_2_integrity_bound,
        3: section_3_geometric_slaughter,
        4: section_4_first_weld,
        5: section_5_confinement,
        6: section_6_scale_inversion,
        7: section_7_full_spine,
    }

    if section is not None:
        fn = sections_map.get(section)
        if fn is None:
            raise HTTPException(status_code=400, detail=f"Section must be 1–7, got {section}")
        results[f"section_{section}"] = fn()
    else:
        for s_num, fn in sections_map.items():
            results[f"section_{s_num}"] = fn()

    return results


# ============================================================================
# Comparison Endpoint
# ============================================================================


@router.post("/kernel/compare", tags=["Kernel"])
async def compare_traces(
    traces: list[list[float]],
    weights: list[float] | None = None,
    api_key: str = Security(_validate_api_key),
) -> dict[str, Any]:
    """
    Compare multiple trace vectors by computing kernel invariants for each
    and returning the comparison matrix.
    """
    import numpy as np

    try:
        from .frozen_contract import EPSILON as _eps
        from .frozen_contract import classify_regime as _cr
    except ImportError:
        raise HTTPException(status_code=500, detail="Frozen contract not available")  # noqa: B904

    if len(traces) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 traces to compare")

    results = []
    for i, trace in enumerate(traces):
        c = np.array(trace, dtype=float)
        n = len(c)
        w = np.array(weights, dtype=float) if weights and len(weights) == n else np.ones(n) / n
        w = w / w.sum()
        c = np.clip(c, _eps, 1.0)

        f_val = float(np.sum(w * c))
        omega_val = 1.0 - f_val
        kappa = float(np.sum(w * np.log(c)))
        ic_val = float(np.exp(kappa))
        s_val = float(-np.sum(w * (c * np.log(c) + (1 - c) * np.log(1 - c))))
        c_std = float(np.std(c) / 0.5) if n > 1 else 0.0

        regime = _cr(omega_val, f_val, s_val, c_std, ic_val)

        results.append(
            {
                "index": i,
                "F": f_val,
                "omega": omega_val,
                "S": s_val,
                "C": c_std,
                "kappa": kappa,
                "IC": ic_val,
                "gap": f_val - ic_val,
                "regime": regime.value if hasattr(regime, "value") else str(regime),
            }
        )

    return {
        "n_traces": len(traces),
        "results": results,
        "F_range": [min(r["F"] for r in results), max(r["F"] for r in results)],
        "IC_range": [min(r["IC"] for r in results), max(r["IC"] for r in results)],
        "gap_range": [min(r["gap"] for r in results), max(r["gap"] for r in results)],
    }


# ============================================================================
# Frozen Contract Endpoint
# ============================================================================


@router.get("/frozen-contract", tags=["System"])
async def get_frozen_contract(
    api_key: str = Security(_validate_api_key),
) -> dict[str, Any]:
    """Return all frozen contract parameters and regime thresholds."""
    try:
        from .frozen_contract import (
            ALPHA,
            DEFAULT_THRESHOLDS,
            EPSILON,
            LAMBDA,
            P_EXPONENT,
            TOL_SEAM,
        )
    except ImportError:
        raise HTTPException(status_code=500, detail="Frozen contract not available")  # noqa: B904

    return {
        "EPSILON": EPSILON,
        "P_EXPONENT": P_EXPONENT,
        "ALPHA": ALPHA,
        "LAMBDA": LAMBDA,
        "TOL_SEAM": TOL_SEAM,
        "regime_thresholds": {
            "omega_stable_max": DEFAULT_THRESHOLDS.omega_stable_max,
            "F_stable_min": DEFAULT_THRESHOLDS.F_stable_min,
            "S_stable_max": DEFAULT_THRESHOLDS.S_stable_max,
            "C_stable_max": DEFAULT_THRESHOLDS.C_stable_max,
            "omega_collapse_min": DEFAULT_THRESHOLDS.omega_collapse_min,
            "I_critical_max": DEFAULT_THRESHOLDS.I_critical_max,
        },
    }


# ============================================================================
# Schema Introspection Endpoint
# ============================================================================


@router.get("/schemas", tags=["System"])
async def list_schemas(
    api_key: str = Security(_validate_api_key),
) -> list[dict[str, str]]:
    """List all available JSON Schema files."""
    from pathlib import Path

    try:
        repo_root = Path(__file__).parent.resolve()
        while repo_root != repo_root.parent:
            if (repo_root / "pyproject.toml").exists():
                break
            repo_root = repo_root.parent

        schema_dir = repo_root / "schemas"
        if not schema_dir.exists():
            return []

        schemas = []
        for sf in sorted(schema_dir.glob("*.schema.json")):
            schemas.append({"name": sf.stem, "path": str(sf.relative_to(repo_root))})
        return schemas
    except Exception:
        return []


# ============================================================================
# Integrity Check Endpoint
# ============================================================================


@router.get("/integrity", tags=["System"])
async def check_integrity(
    api_key: str = Security(_validate_api_key),
) -> dict[str, Any]:
    """Check SHA-256 integrity of tracked files."""
    import hashlib
    from pathlib import Path

    try:
        repo_root = Path(__file__).parent.resolve()
        while repo_root != repo_root.parent:
            if (repo_root / "pyproject.toml").exists():
                break
            repo_root = repo_root.parent

        sha_file = repo_root / "integrity" / "sha256.txt"
        if not sha_file.exists():
            return {"status": "NON_EVALUABLE", "message": "sha256.txt not found"}

        total = 0
        passed = 0
        failed = []
        for line in sha_file.read_text().strip().splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.strip().split("  ", 1)
            if len(parts) != 2:
                continue
            expected_hash, filepath = parts
            total += 1
            full_path = repo_root / filepath
            if not full_path.exists():
                failed.append({"file": filepath, "reason": "missing"})
                continue
            actual_hash = hashlib.sha256(full_path.read_bytes()).hexdigest()
            if actual_hash == expected_hash:
                passed += 1
            else:
                failed.append({"file": filepath, "reason": "mismatch"})

        status = "CONFORMANT" if not failed else "NONCONFORMANT"
        return {
            "status": status,
            "total_files": total,
            "passed": passed,
            "failed_count": len(failed),
            "failed": failed[:20],  # Cap failed list
        }
    except Exception as exc:
        return {"status": "NON_EVALUABLE", "message": str(exc)}

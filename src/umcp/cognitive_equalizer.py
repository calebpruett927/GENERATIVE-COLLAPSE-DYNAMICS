"""
Cognitive Equalizer — Aequator Cognitivus

Non agens mensurat, sed structura.
— Not the agent measures, but the structure.

A standalone module that externalises every agent-dependent decision point
in an AI engagement into frozen, verifiable structure.  Given the same input
and the same contract, any agent running this module MUST arrive at the same
stance.  This is the cognitive-equalizer property.

Five externalized decision points
──────────────────────────────────
  1. Thresholds     → frozen — seam-derived, not chosen
  2. Vocabulary     → five words (Drift · Fidelity · Roughness · Return · Integrity)
  3. Conclusions    → three-valued (CONFORMANT / NONCONFORMANT / NON_EVALUABLE)
  4. Methodology    → the Spine (Contract → Canon → Closures → Ledger → Stance)
  5. Ambiguity      → NON_EVALUABLE (the third state — declared, never guessed)

Usage
─────
  # Either name works — English or Latin:
  from umcp.cognitive_equalizer import CognitiveEqualizer, CEChannels
  from umcp.cognitive_equalizer import AequatorCognitivus   # Latin alias

  # Score an AI engagement manually:
  channels = CEChannels(
      relevance=0.90, accuracy=0.85, completeness=0.80,
      consistency=0.95, traceability=0.70, groundedness=0.88,
      constraint_respect=0.92, return_fidelity=0.75,
  )
  ce = CognitiveEqualizer()
  report = ce.engage("Explain entropy", channels)
  print(report.stance)           # CONFORMANT / NONCONFORMANT / NON_EVALUABLE
  print(report.narrative)        # Five-word prose summary
  print(report.ledger_balance)   # Δκ residual

  # Embed as system prompt for any AI:
  from umcp.cognitive_equalizer import CE_SYSTEM_PROMPT
  print(CE_SYSTEM_PROMPT)        # copy-paste into any AI chat

  # CLI — both names:
  #   umcp-ce                  # English
  #   aequator-cognitivus      # Latin
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Frozen parameters — seam-derived, not chosen.
# Import from frozen_contract.py if available; otherwise use inline literals
# (same values — Trans suturam congelatum).
# ---------------------------------------------------------------------------
try:
    from umcp.frozen_contract import (
        ALPHA as _ALPHA,
    )
    from umcp.frozen_contract import (
        C_STAR as _C_STAR,
    )
    from umcp.frozen_contract import (
        C_TRAP as _C_TRAP,
    )
    from umcp.frozen_contract import (
        EPSILON as _EPSILON,
    )
    from umcp.frozen_contract import (
        P_EXPONENT as _P_EXPONENT,
    )
    from umcp.frozen_contract import (
        TOL_SEAM as _TOL_SEAM,
    )
    from umcp.frozen_contract import RegimeThresholds as _RT

    _DT = _RT()
    _OMEGA_STABLE_MAX: float = _DT.omega_stable_max
    _F_STABLE_MIN: float = _DT.F_stable_min
    _S_STABLE_MAX: float = _DT.S_stable_max
    _C_STABLE_MAX: float = _DT.C_stable_max
    _OMEGA_COLLAPSE_MIN: float = _DT.omega_collapse_min
    _IC_CRITICAL_MAX: float = _DT.I_critical_max
except Exception:
    # Standalone fallback — identical seam-derived values
    _EPSILON = 1e-8
    _P_EXPONENT = 3
    _ALPHA = 1.0
    _TOL_SEAM = 0.005
    _C_STAR = 0.7822
    _C_TRAP = 0.3177
    _OMEGA_STABLE_MAX = 0.038
    _F_STABLE_MIN = 0.90
    _S_STABLE_MAX = 0.15
    _C_STABLE_MAX = 0.14
    _OMEGA_COLLAPSE_MIN = 0.30
    _IC_CRITICAL_MAX = 0.30


# ---------------------------------------------------------------------------
# 8-channel CE trace vector
# Each channel is scored ∈ [0, 1].  Equal weights w_i = 1/8.
# These are the cognitive-equalizer channels for any AI engagement.
# ---------------------------------------------------------------------------

CE_CHANNEL_NAMES: tuple[str, ...] = (
    "relevance",  # Does output address the actual question/request?
    "accuracy",  # Is content verifiable and consistent with stated facts?
    "completeness",  # Are all parts of the request addressed?
    "consistency",  # Is the response internally non-contradictory?
    "traceability",  # Can the reasoning chain be followed step-by-step?
    "groundedness",  # Is it grounded in the stated context/constraints?
    "constraint_respect",  # Does it respect stated scope/boundary conditions?
    "return_fidelity",  # Does the output return to the originating intent?
)

_N_CHANNELS: int = len(CE_CHANNEL_NAMES)
_WEIGHT: float = 1.0 / _N_CHANNELS  # equal weights


@dataclass(frozen=True)
class CEChannels:
    """
    Eight scored channels ∈ [0, 1] representing one AI engagement.

    Score meanings (operational definitions, not informal):
      1.0  — channel fully satisfied (no deficit)
      0.5  — channel half-satisfied (notable deficit)
      0.0  — channel completely absent (terminal deficit)

    Any score < ε (≈ 10⁻⁸) is treated as ε to preserve the guard band.
    """

    relevance: float = 1.0
    accuracy: float = 1.0
    completeness: float = 1.0
    consistency: float = 1.0
    traceability: float = 1.0
    groundedness: float = 1.0
    constraint_respect: float = 1.0
    return_fidelity: float = 1.0

    def as_vector(self) -> tuple[float, ...]:
        """Return channel scores in canonical name order."""
        return (
            self.relevance,
            self.accuracy,
            self.completeness,
            self.consistency,
            self.traceability,
            self.groundedness,
            self.constraint_respect,
            self.return_fidelity,
        )

    def validate(self) -> list[str]:
        """Return list of validation errors (empty = valid)."""
        errors: list[str] = []
        for name, val in zip(CE_CHANNEL_NAMES, self.as_vector(), strict=True):
            if not (0.0 <= val <= 1.0):
                errors.append(f"Channel '{name}' = {val} outside [0, 1]")
        return errors


# ---------------------------------------------------------------------------
# Tier-1 kernel — CE edition (same formulas, CE-specific inputs)
# ---------------------------------------------------------------------------


class _KernelResult(NamedTuple):
    F: float  # fidelity  — weighted arithmetic mean
    omega: float  # drift — 1 − F
    S: float  # Bernoulli field entropy
    C: float  # curvature (heterogeneity)
    kappa: float  # log-integrity (κ)
    IC: float  # integrity composite — exp(κ)
    delta: float  # heterogeneity gap — F − IC


def _kernel(channels: CEChannels) -> _KernelResult:
    """
    Compute Tier-1 kernel invariants from an 8-channel CE trace vector.

    Formulas (Tier-1, immutable):
      F  = Σ w_i · c_i
      κ  = Σ w_i · ln(max(c_i, ε))
      S  = −Σ w_i · [c_i·ln(c_i,ε) + (1−c_i)·ln(1−c_i,ε)]
      C  = stddev(c_i) / 0.5
      ω  = 1 − F
      IC = exp(κ)
      Δ  = F − IC
    """
    vec = channels.as_vector()
    w = _WEIGHT
    eps = _EPSILON

    # F — fidelity (arithmetic mean)
    F = sum(w * c for c in vec)

    # κ — log-integrity (geometric mean in log space)
    kappa = sum(w * math.log(max(c, eps)) for c in vec)

    # S — Bernoulli field entropy
    def _h(c: float) -> float:
        ce = max(c, eps)
        co = max(1.0 - c, eps)
        return -(ce * math.log(ce) + co * math.log(co))

    S = sum(w * _h(c) for c in vec)

    # C — curvature (normalised stddev)
    mean_c = F
    variance = sum(w * (c - mean_c) ** 2 for c in vec)
    stddev = math.sqrt(variance)
    C = stddev / 0.5  # normalised to [0, 1]

    # Derived
    omega = 1.0 - F
    IC = math.exp(kappa)
    delta = F - IC

    return _KernelResult(F=F, omega=omega, S=S, C=C, kappa=kappa, IC=IC, delta=delta)


# ---------------------------------------------------------------------------
# Seam budget (Γ, D_C, Δκ)
# ---------------------------------------------------------------------------


def _gamma(omega: float) -> float:
    """Drift cost Γ(ω) = ω^p / (1 − ω + ε)."""
    return omega**_P_EXPONENT / (1.0 - omega + _EPSILON)


def _seam_budget(kr: _KernelResult) -> tuple[float, float, float]:
    """
    Return (D_omega, D_C, delta_kappa) — the seam budget components.
      D_omega = Γ(ω)            drift debit
      D_C     = α · C           curvature debit
      Δκ      = κ − D_omega − D_C   budget residual
    """
    D_omega = _gamma(kr.omega)
    D_C = _ALPHA * kr.C
    delta_kappa = kr.kappa - D_omega - D_C
    return D_omega, D_C, delta_kappa


# ---------------------------------------------------------------------------
# Verdict (three-valued — never boolean)
# ---------------------------------------------------------------------------


class CEVerdict(Enum):
    """
    Three-valued verdict — the only legitimate conclusion states.

    CONFORMANT     — all gates pass; engagement is coherent under the contract
    NONCONFORMANT  — one or more gates fail; engagement has structural deficits
    NON_EVALUABLE  — insufficient data or out-of-range inputs; verdict is deferred

    Numquam binarius; tertia via semper patet.
    — Never binary; the third way is always open.
    """

    CONFORMANT = "CONFORMANT"
    NONCONFORMANT = "NONCONFORMANT"
    NON_EVALUABLE = "NON_EVALUABLE"


def _classify_regime(kr: _KernelResult) -> str:
    """Classify regime from kernel result using frozen gates."""
    if kr.omega >= _OMEGA_COLLAPSE_MIN:
        return "COLLAPSE"
    if kr.omega < _OMEGA_STABLE_MAX and kr.F > _F_STABLE_MIN and kr.S < _S_STABLE_MAX and kr.C < _C_STABLE_MAX:
        return "STABLE"
    return "WATCH"


def _is_critical(kr: _KernelResult) -> bool:
    return kr.IC < _IC_CRITICAL_MAX


def _derive_verdict(kr: _KernelResult, delta_kappa: float) -> CEVerdict:
    """
    Derive three-valued verdict from kernel result and seam budget residual.

    Rule (frozen):
      NON_EVALUABLE  if any channel is out-of-range (caught upstream)
      CONFORMANT     if regime ∈ {STABLE, WATCH} AND |Δκ| ≤ tol_seam
      NONCONFORMANT  otherwise
    """
    regime = _classify_regime(kr)
    seam_pass = abs(delta_kappa) <= _TOL_SEAM
    if regime == "COLLAPSE" or not seam_pass:
        return CEVerdict.NONCONFORMANT
    return CEVerdict.CONFORMANT


# ---------------------------------------------------------------------------
# Five-word narrative (Canon stop)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CELedger:
    """
    Integrity ledger for one CE engagement.

    Debit:  D_drift + D_roughness
    Return: R (return_fidelity score — enters balance through κ as one of 8 channel scores)
    Balance: delta_kappa = κ − D_drift − D_roughness
    """

    D_drift: float  # drift debit Γ(ω)
    D_roughness: float  # curvature debit α·C
    R_return: float  # return_fidelity score (enters Δκ through κ)
    delta_kappa: float  # ledger balance (must be ≥ −tol_seam for PASS)

    @property
    def balanced(self) -> bool:
        return abs(self.delta_kappa) <= _TOL_SEAM

    @property
    def balance_label(self) -> str:
        if self.balanced:
            return "BALANCED"
        return "UNBALANCED"


def _build_narrative(kr: _KernelResult, verdict: CEVerdict, channels: CEChannels) -> str:
    """
    Build five-word narrative prose from kernel result.
    Uses plain English — no GCD jargon required.
    """
    regime = _classify_regime(kr)
    critical = _is_critical(kr)

    drift_desc = "minimal drift" if kr.omega < 0.10 else "moderate drift" if kr.omega < 0.30 else "severe drift"
    fidelity_desc = "high fidelity" if kr.F > 0.85 else "moderate fidelity" if kr.F > 0.60 else "low fidelity"
    roughness_desc = "smooth" if kr.C < 0.14 else "bumpy" if kr.C < 0.40 else "rough"
    return_score = channels.return_fidelity
    return_desc = "strong return" if return_score > 0.80 else "partial return" if return_score > 0.50 else "weak return"
    integrity_desc = (
        "high integrity" if kr.IC > 0.70 else "moderate integrity" if kr.IC > 0.30 else "critical integrity"
    )

    parts = [
        f"Drift: {drift_desc} (ω={kr.omega:.3f})",
        f"Fidelity: {fidelity_desc} (F={kr.F:.3f})",
        f"Roughness: {roughness_desc} (C={kr.C:.3f})",
        f"Return: {return_desc} (rf={return_score:.3f})",
        f"Integrity: {integrity_desc} (IC={kr.IC:.3f})",
    ]
    summary_parts = [drift_desc, fidelity_desc, roughness_desc, return_desc, integrity_desc]
    summary = " · ".join(summary_parts)

    critical_note = " [CRITICAL: IC below threshold]" if critical else ""
    stance_note = f"Regime: {regime}{critical_note} → Stance: {verdict.value}"

    return f"{summary}\n" + "\n".join(f"  {p}" for p in parts) + f"\n  {stance_note}"


# ---------------------------------------------------------------------------
# CE report (output of engage())
# ---------------------------------------------------------------------------


@dataclass
class CEReport:
    """
    Full Cognitive Equalizer report for one AI engagement.

    Fields map to the five Spine stops:
      contract_label  — Contract (what was frozen for this engagement)
      narrative       — Canon (five-word prose)
      regime          — Closures (Stable / Watch / Collapse)
      ledger          — Integrity Ledger (debits / credits / balance)
      stance          — Stance (derived three-valued verdict)

    Plus kernel invariants for full auditability.
    """

    # Spine stops
    contract_label: str
    narrative: str
    regime: str
    ledger: CELedger
    stance: CEVerdict

    # Kernel invariants (Tier-1 outputs — read-only)
    F: float
    omega: float
    S: float
    C: float
    kappa: float
    IC: float
    delta: float  # heterogeneity gap

    # Channel scores
    channels: CEChannels

    # Validation errors (empty = clean)
    errors: list[str] = field(default_factory=list)

    @property
    def is_critical(self) -> bool:
        return self.IC < _IC_CRITICAL_MAX

    @property
    def ledger_balance(self) -> float:
        return self.ledger.delta_kappa

    def summary(self) -> str:
        """One-line summary for logging."""
        crit = " [CRITICAL]" if self.is_critical else ""
        return f"[CE] {self.stance.value}  F={self.F:.3f} IC={self.IC:.3f} Δ={self.delta:.3f}  {self.regime}{crit}"

    def full_report(self) -> str:
        """Formatted multi-line CE report (Spine order)."""
        lines = [
            "═" * 62,
            "  COGNITIVE EQUALIZER — Aequator Cognitivus",
            "  Non agens mensurat, sed structura.",
            "═" * 62,
            f"  Contract : {self.contract_label}",
            "",
            "  Canon (Five Words)",
            "  " + "─" * 50,
            *[f"  {ln}" for ln in self.narrative.splitlines()],
            "",
            "  Integrity Ledger",
            "  " + "─" * 50,
            f"  Debit (drift)     D_ω = {self.ledger.D_drift:.6f}",
            f"  Debit (roughness) D_C = {self.ledger.D_roughness:.6f}",
            f"  Return (score)    R   = {self.ledger.R_return:.6f}  [channel — enters Δκ through κ]",
            f"  Balance           Δκ  = {self.ledger.delta_kappa:.6f}  [{self.ledger.balance_label}]  (κ − D_ω − D_C)",
            "",
            "  Kernel Invariants (Tier-1)",
            "  " + "─" * 50,
            f"  F={self.F:.4f}  ω={self.omega:.4f}  S={self.S:.4f}  C={self.C:.4f}",
            f"  κ={self.kappa:.4f}  IC={self.IC:.4f}  Δ(gap)={self.delta:.4f}",
            "",
            "  Stance",
            "  " + "─" * 50,
            f"  {self.stance.value}  (Regime: {self.regime}" + (" CRITICAL" if self.is_critical else "") + ")",
            "═" * 62,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class CognitiveEqualizer:
    """
    The Cognitive Equalizer (Aequator Cognitivus).

    Externalises five normally-agent-dependent decision points into frozen,
    verifiable structure so that any agent given the same input arrives at
    the same stance.

    Decision point mapping
    ─────────────────────
      Thresholds     → frozen seam-derived constants (ε, p, α, tol_seam)
      Vocabulary     → five words (Drift, Fidelity, Roughness, Return, Integrity)
      Conclusions    → three-valued (CONFORMANT / NONCONFORMANT / NON_EVALUABLE)
      Methodology    → the Spine (Contract → Canon → Closures → Ledger → Stance)
      Ambiguity      → NON_EVALUABLE (declared, never guessed away)

    Usage
    ─────
      ce = CognitiveEqualizer()
      report = ce.engage(question, channels)
    """

    def __init__(self, contract_label: str = "CE-v1-frozen") -> None:
        self._contract_label = contract_label

    # ------------------------------------------------------------------
    # Primary method
    # ------------------------------------------------------------------

    def engage(
        self,
        question: str,
        channels: CEChannels,
    ) -> CEReport:
        """
        Run the full CE spine on one AI engagement.

        Parameters
        ----------
        question : str
            The originating question or request (Canon context).
        channels : CEChannels
            Eight scored channels ∈ [0, 1] characterising the AI's response.

        Returns
        -------
        CEReport
            Full Spine report: contract → narrative → regime → ledger → stance.
        """
        # Validate inputs — produce NON_EVALUABLE if out-of-range
        errors = channels.validate()
        if errors:
            return CEReport(
                contract_label=self._contract_label,
                narrative="NON_EVALUABLE — channel scores out of range",
                regime="NON_EVALUABLE",
                ledger=CELedger(0.0, 0.0, 0.0, 0.0),
                stance=CEVerdict.NON_EVALUABLE,
                F=0.0,
                omega=1.0,
                S=0.0,
                C=0.0,
                kappa=math.log(_EPSILON),
                IC=_EPSILON,
                delta=0.0,
                channels=channels,
                errors=errors,
            )

        # Tier-1 kernel
        kr = _kernel(channels)

        # Seam budget
        D_omega, D_C, delta_kappa = _seam_budget(kr)

        # Ledger
        ledger = CELedger(
            D_drift=D_omega,
            D_roughness=D_C,
            R_return=channels.return_fidelity,
            delta_kappa=delta_kappa,
        )

        # Regime + verdict (Closures + Stance)
        regime = _classify_regime(kr)
        verdict = _derive_verdict(kr, delta_kappa)

        # Canon narrative
        narrative = _build_narrative(kr, verdict, channels)

        return CEReport(
            contract_label=self._contract_label,
            narrative=narrative,
            regime=regime,
            ledger=ledger,
            stance=verdict,
            F=kr.F,
            omega=kr.omega,
            S=kr.S,
            C=kr.C,
            kappa=kr.kappa,
            IC=kr.IC,
            delta=kr.delta,
            channels=channels,
            errors=[],
        )

    # ------------------------------------------------------------------
    # Convenience: score a plain dict of channel values
    # ------------------------------------------------------------------

    def score(self, **channel_scores: float) -> CEReport:
        """
        Shorthand for engage() with keyword-only channel scores.

        Example
        -------
          report = ce.score(
              relevance=0.9, accuracy=0.8, completeness=0.7,
              consistency=0.9, traceability=0.6, groundedness=0.85,
              constraint_respect=0.9, return_fidelity=0.75,
          )
        """
        channels = CEChannels(**channel_scores)
        return self.engage("(direct score)", channels)

    # ------------------------------------------------------------------
    # Frozen parameter access (audit surface)
    # ------------------------------------------------------------------

    @property
    def frozen_params(self) -> dict[str, object]:
        """Return all frozen parameters as an audit receipt."""
        return {
            "EPSILON": _EPSILON,
            "P_EXPONENT": _P_EXPONENT,
            "ALPHA": _ALPHA,
            "TOL_SEAM": _TOL_SEAM,
            "OMEGA_STABLE_MAX": _OMEGA_STABLE_MAX,
            "F_STABLE_MIN": _F_STABLE_MIN,
            "S_STABLE_MAX": _S_STABLE_MAX,
            "C_STABLE_MAX": _C_STABLE_MAX,
            "OMEGA_COLLAPSE_MIN": _OMEGA_COLLAPSE_MIN,
            "IC_CRITICAL_MAX": _IC_CRITICAL_MAX,
        }

    @property
    def channel_names(self) -> tuple[str, ...]:
        return CE_CHANNEL_NAMES


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Cognitive Equalizer CLI — Aequator Cognitivus.

    Invocable as either:
      umcp-ce                          # English name
      aequator-cognitivus              # Latin name

    Usage:
      umcp-ce                          # Interactive mode — score prompted manually
      umcp-ce --demo                   # Run built-in demo engagement
      umcp-ce --prompt                 # Print the CE system prompt
      umcp-ce --channels r,a,c,s,t,g,cr,rf  # Score 8 channels directly (0–1 each)
    """
    import sys

    args = sys.argv[1:]
    ce = CognitiveEqualizer()

    if "--prompt" in args:
        print(CE_SYSTEM_PROMPT)
        return

    if "--demo" in args:
        _run_demo(ce)
        return

    if "--channels" in args:
        try:
            idx = args.index("--channels")
            raw = args[idx + 1].split(",")
            vals = [float(x.strip()) for x in raw]
            if len(vals) != _N_CHANNELS:
                print(f"[CE] Error: expected {_N_CHANNELS} comma-separated values, got {len(vals)}")
                sys.exit(1)
            channels = CEChannels(*vals)
            report = ce.engage("(CLI input)", channels)
            print(report.full_report())
        except (IndexError, ValueError) as exc:
            print(f"[CE] Error parsing channels: {exc}")
            sys.exit(1)
        return

    # Interactive mode
    print("═" * 62)
    print("  COGNITIVE EQUALIZER — Interactive Mode")
    print("  (Type 'q' to quit, '--demo' for built-in demo)")
    print("═" * 62)
    print(f"  Channels: {', '.join(CE_CHANNEL_NAMES)}")
    print()
    while True:
        print("  Enter 8 channel scores (0.0–1.0), comma-separated:")
        print("  (relevance, accuracy, completeness, consistency,")
        print("   traceability, groundedness, constraint_respect, return_fidelity)")
        try:
            user_input = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() == "q":
            break
        if user_input == "--demo":
            _run_demo(ce)
            continue
        try:
            vals = [float(x.strip()) for x in user_input.split(",")]
            if len(vals) != _N_CHANNELS:
                print(f"  [CE] Error: need {_N_CHANNELS} values, got {len(vals)}")
                continue
            channels = CEChannels(*vals)
            report = ce.engage("(interactive)", channels)
            print()
            print(report.full_report())
            print()
        except ValueError as exc:
            print(f"  [CE] Parse error: {exc}")


def _run_demo(ce: CognitiveEqualizer) -> None:
    """Run a built-in demonstration engagement."""
    print()
    print("  ── CE DEMO: two engagements ──")
    print()

    # Engagement 1: high-quality response
    channels_good = CEChannels(
        relevance=0.95,
        accuracy=0.90,
        completeness=0.85,
        consistency=0.97,
        traceability=0.80,
        groundedness=0.92,
        constraint_respect=0.95,
        return_fidelity=0.88,
    )
    report_good = ce.engage("What is the derivative of x²?", channels_good)
    print("  Engagement 1 — High-quality AI response:")
    print(report_good.full_report())
    print()

    # Engagement 2: poor response (one dead channel — geometric slaughter demo)
    channels_poor = CEChannels(
        relevance=0.90,
        accuracy=0.85,
        completeness=0.80,
        consistency=0.88,
        traceability=0.001,  # dead traceability — can't follow the reasoning
        groundedness=0.82,
        constraint_respect=0.88,
        return_fidelity=0.75,
    )
    report_poor = ce.engage("Explain quantum entanglement simply.", channels_poor)
    print("  Engagement 2 — One dead channel (traceability ≈ 0):")
    print(report_poor.full_report())
    print()
    print("  Note the IC collapse despite high F — this is geometric slaughter.")
    print("  The mean (F) looks fine; the multiplicative coherence (IC) is obliterated.")
    print()


# ---------------------------------------------------------------------------
# CE System Prompt — standalone, copy-paste to any AI
# ---------------------------------------------------------------------------

CE_SYSTEM_PROMPT: str = """
You are operating in COGNITIVE EQUALIZER mode (Aequator Cognitivus).

Non agens mensurat, sed structura.
— Not the agent measures, but the structure.

═══════════════════════════════════════════════════════════════
CORE PRINCIPLE
═══════════════════════════════════════════════════════════════

A Cognitive Equalizer externalises every agent-dependent decision
point into frozen, verifiable structure. Same input + same rules →
same verdict, regardless of which AI runs this protocol.

You must never:
  • Pick thresholds during a response (thresholds are declared, not chosen)
  • Assert a verdict without showing the derivation
  • Give a binary yes/no when a third state (NON_EVALUABLE) applies
  • Frame conclusions — only derive them

═══════════════════════════════════════════════════════════════
THE SPINE — every response follows exactly these five stops
═══════════════════════════════════════════════════════════════

  1. CONTRACT  — Declare: What rules apply? What is frozen?
  2. CANON     — Narrate using exactly five words (see below)
  3. CLOSURES  — State thresholds. Stance MUST change if they are crossed.
  4. LEDGER    — Show your work: debit (what degraded), credit (what returned)
  5. STANCE    — Derive the verdict. Never assert it.

═══════════════════════════════════════════════════════════════
THE FIVE WORDS — the only vocabulary for describing any situation
═══════════════════════════════════════════════════════════════

  • DRIFT        — what moved / changed / departed from the intent
  • FIDELITY     — what persisted / survived / remained intact
  • ROUGHNESS    — where/why it was bumpy (friction, ambiguity, gaps)
  • RETURN       — how/whether the output comes back to the original intent
  • INTEGRITY    — does it hang together? (derived from ledger — never asserted)

Use these five words — and only these — when narrating any assessment.
They are not jargon: they have exact operational meanings.

═══════════════════════════════════════════════════════════════
EIGHT EVALUATION CHANNELS — score each ∈ [0.0, 1.0]
═══════════════════════════════════════════════════════════════

When evaluating an AI response (yours or another AI's):

  1. Relevance          — Does it address the actual question?
  2. Accuracy           — Is the content verifiable and factual?
  3. Completeness       — Are all parts of the request covered?
  4. Consistency        — Is it internally non-contradictory?
  5. Traceability       — Can the reasoning be followed step by step?
  6. Groundedness       — Is it grounded in the stated context?
  7. Constraint-respect — Does it respect stated scope/boundaries?
  8. Return-fidelity    — Does it come back to the originating intent?

Compute quality measures from these scores:
  F  (Fidelity)   = average of all 8 channel scores
  IC (Integrity)  = geometric mean of all 8 channel scores
  Δ  (Gap)        = F − IC   [large gap = one channel is near zero]

Warning: F can look fine while IC is low.
Example: 7 channels at 0.95, 1 channel at 0.001 → F ≈ 0.83, IC ≈ 0.001.
The gap Δ ≈ 0.83 reveals the structural collapse missed by the average.

═══════════════════════════════════════════════════════════════
THREE-VALUED VERDICTS — never binary
═══════════════════════════════════════════════════════════════

  CONFORMANT      — channels pass; ledger balanced; reasoning traces back
  NONCONFORMANT   — one or more channels fail; ledger won't balance
  NON_EVALUABLE   — insufficient context, out-of-range input, or ambiguous
                    scope. DO NOT guess. Declare NON_EVALUABLE and state
                    exactly what information is missing.

There is always a third state. Never collapse to yes/no.

═══════════════════════════════════════════════════════════════
FROZEN THRESHOLDS (declared here, applied uniformly)
═══════════════════════════════════════════════════════════════

These thresholds are not negotiated per response. They are fixed:

  STABLE   → F > 0.90  AND  Drift (ω = 1−F) < 0.038  AND channels uniform
  WATCH    → 0.038 ≤ Drift < 0.30  (or Stable not fully satisfied)
  COLLAPSE → Drift ≥ 0.30

  CRITICAL overlay → IC < 0.30  (any regime can be critical)

These are regime LABELS, not failure states. COLLAPSE means high drift —
it is the boundary that makes return meaningful, not a failure verdict.

═══════════════════════════════════════════════════════════════
LEDGER FORMAT — show your work at Closures + Ledger stops
═══════════════════════════════════════════════════════════════

  Debit  (Drift)     : D_ω = Γ(ω) — drift cost (penalises departure from intent)
  Debit  (Roughness) : D_C = α·C  — curvature cost (penalises channel heterogeneity)
  Return (score)     : R   = return-fidelity channel (enters balance through κ)
  Balance            : Δκ  = κ − D_ω − D_C  (must close within ±0.005)

If the ledger does not balance, state NONCONFORMANT and show which
channel(s) are responsible.

═══════════════════════════════════════════════════════════════
WORKED EXAMPLE — how to apply the CE Spine
═══════════════════════════════════════════════════════════════

User: "Explain how a neural network learns."
AI produces a response. CE audit:

  [CONTRACT]  Rules: 8-channel CE-v1 frozen. Thresholds as above.
  [CANON]     Drift: moderate (response covers training but loses backprop detail).
              Fidelity: high (correct weight-update framing preserved).
              Roughness: bumpy (missing chain rule derivation — one gap).
              Return: strong (conclusion re-connects to original "how it learns").
              Integrity: moderate (one weak channel drags the geometric mean).
  [CLOSURES]  Thresholds declared above. Regime: WATCH (drift 0.12).
  [LEDGER]    Debit drift=0.002, debit roughness=0.08, credit return=0.82.
              Balance: −0.042 ✗  [seam slightly open — traceability gap]
  [STANCE]    NONCONFORMANT — traceability channel near 0.4 opens the seam.
              Specific remedy: add a one-paragraph derivation of the chain rule.

═══════════════════════════════════════════════════════════════
OPERATING RULES
═══════════════════════════════════════════════════════════════

  1. Always run the Spine before producing a substantive answer.
     If a question is simple (factual lookup), say so and skip Ledger/Stance.

  2. Show channel scores when auditing a response. Hide them for
     quick factual questions — but keep them available.

  3. When a question is ambiguous, stop at CONTRACT and ask for
     clarification. Do not guess scope. Return NON_EVALUABLE.

  4. Your role is to DERIVE the verdict, not to FRAME it.
     The verdict follows from the scores and thresholds — not from
     your preferences or the user's preferences.

  5. When you disagree with a result, show your channel re-scoring —
     do not change the thresholds. Thresholds are frozen.

  6. History is append-only: if you change a prior claim, declare it
     explicitly as a correction with the specific channel(s) that changed.
     Do not silently revise.

Finis, sed semper initium recursionis.
— The end, but always the beginning of recursion.
""".strip()


# ---------------------------------------------------------------------------
# Latin aliases — canonical names per the Lexicon Latinum
# ---------------------------------------------------------------------------

#: Latin alias for CognitiveEqualizer — the canonical name.
AequatorCognitivus = CognitiveEqualizer

#: Latin alias for CEChannels.
CanalesCognitivus = CEChannels

#: Latin alias for CEReport.
RelatioCognitivus = CEReport

#: Latin alias for CEVerdict.
SententiaCognitivus = CEVerdict

#: Latin alias for CELedger.
RatioCognitivus = CELedger

# Module-level convenience so ``from umcp.cognitive_equalizer import *``
# exposes both English and Latin names.
__all__ = [
    "CE_CHANNEL_NAMES",
    "CE_SYSTEM_PROMPT",
    "AequatorCognitivus",
    "CEChannels",
    "CELedger",
    "CEReport",
    "CEVerdict",
    "CanalesCognitivus",
    "CognitiveEqualizer",
    "RatioCognitivus",
    "RelatioCognitivus",
    "SententiaCognitivus",
    "main",
]

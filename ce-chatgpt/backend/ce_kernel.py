from __future__ import annotations

import math
from typing import Any

# Frozen parameters (from GCD contract)
EPSILON = 1e-8
P_EXPONENT = 3
ALPHA = 1.0
TOL_SEAM = 0.005

CHANNELS = [
    "relevance",
    "accuracy",
    "completeness",
    "consistency",
    "traceability",
    "groundedness",
    "constraint_respect",
    "return_fidelity",
]

# Regime gates (example thresholds)
REGIME_THRESHOLDS = {"stable": {"omega": 0.038, "F": 0.90, "S": 0.15, "C": 0.14}, "collapse": {"omega": 0.30}}


def kernel_invariants(channels: dict[str, float]) -> dict[str, float]:
    w = 1.0 / 8
    c = [max(EPSILON, min(1.0, channels[k])) for k in CHANNELS]
    F = sum(c) * w
    kappa = sum(w * math.log(x) for x in c)
    IC = math.exp(kappa)
    omega = 1.0 - F
    S = -sum(w * (x * math.log(x) + (1 - x) * math.log(1 - x)) for x in c)
    C = (math.sqrt(sum((x - F) ** 2 for x in c) / 8)) / 0.5
    delta = F - IC
    return {"F": F, "kappa": kappa, "IC": IC, "omega": omega, "S": S, "C": C, "delta": delta}


def classify_regime(inv: dict[str, float]) -> str:
    if (
        inv["omega"] < REGIME_THRESHOLDS["stable"]["omega"]
        and inv["F"] > REGIME_THRESHOLDS["stable"]["F"]
        and inv["S"] < REGIME_THRESHOLDS["stable"]["S"]
        and inv["C"] < REGIME_THRESHOLDS["stable"]["C"]
    ):
        return "CONFORMANT"
    elif inv["omega"] >= REGIME_THRESHOLDS["collapse"]["omega"]:
        return "NONCONFORMANT"
    else:
        return "NON_EVALUABLE"


def auto_score_channels(response: str, user_message: str) -> dict[str, float]:
    # Simple heuristics for demonstration (replace with LLM or advanced logic as needed)
    scores = {}
    # Relevance: does response mention key terms from user_message?
    scores["relevance"] = 0.95 if any(word in response for word in user_message.split()[:3]) else 0.7
    # Accuracy: placeholder, assume high
    scores["accuracy"] = 0.9 if len(response) > 10 else 0.6
    # Completeness: longer responses are more complete
    scores["completeness"] = min(1.0, len(response) / 200)
    # Consistency: always high in this stub
    scores["consistency"] = 0.95
    # Traceability: if response cites sources or steps
    scores["traceability"] = 0.95 if ("http" in response or "step" in response) else 0.7
    # Groundedness: if response is not generic
    scores["groundedness"] = 0.9 if len(set(response.split())) > 10 else 0.6
    # Constraint respect: always high in this stub
    scores["constraint_respect"] = 0.95
    # Return-fidelity: if response restates or answers the question
    scores["return_fidelity"] = 0.9 if user_message.split("?")[0] in response else 0.7
    # Clamp and fill missing
    for k in CHANNELS:
        scores[k] = max(0.0, min(1.0, scores.get(k, 0.8)))
    return scores


def run_ce_audit(response: str, user_message: str, ce_mode: str) -> dict[str, Any]:
    channels = auto_score_channels(response, user_message)
    inv = kernel_invariants(channels)
    regime = classify_regime(inv)
    audit = {
        "channels": channels,
        "invariants": inv,
        "regime": regime,
        "mode": ce_mode,
        "summary": f"Drift: {inv['omega']:.3f}, Fidelity: {inv['F']:.3f}, Roughness: {inv['C']:.3f}, Return: {inv['IC']:.3f}, Integrity: {regime}",
    }
    if ce_mode == "full":
        audit["full_report"] = {"spine": ["CONTRACT", "CANON", "CLOSURES", "LEDGER", "STANCE"], "details": inv}
    return audit

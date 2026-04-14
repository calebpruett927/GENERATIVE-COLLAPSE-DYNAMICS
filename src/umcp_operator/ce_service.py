# CE audit service for GCD operator backend


def compute_ce_audit(answer: str) -> dict[str, float]:
    # Deterministic heuristic: score based on answer length and keywords
    length = len(answer)
    has_contract = "contract" in answer.lower()
    has_session = "session" in answer.lower()
    has_working = "working set" in answer.lower()
    base = min(1.0, 0.5 + 0.005 * min(length, 100))
    return {
        "relevance": base + 0.1 if has_contract else base,
        "accuracy": base + 0.1 if has_session else base,
        "completeness": base,
        "consistency": base,
        "traceability": base + 0.1 if has_working else base,
        "groundedness": base,
        "constraintRespect": base,
        "returnFidelity": base,
    }

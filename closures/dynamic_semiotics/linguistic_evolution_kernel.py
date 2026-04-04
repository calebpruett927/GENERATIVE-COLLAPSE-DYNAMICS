import numpy as np

from umcp.frozen_contract import EPSILON
from umcp.kernel_optimized import compute_kernel_outputs

LINGUISTIC_STATES = {
    "STABLE": {"phonology": 0.9, "semantics": 0.9, "syntax": 0.9},
    "DRIFT": {"phonology": 0.6, "semantics": 0.5, "syntax": 0.8},
    "PIDGIN": {"phonology": 0.4, "semantics": 0.7, "syntax": 0.2},
    "DEAD": {"phonology": EPSILON, "semantics": EPSILON, "syntax": EPSILON},
}


def analyze_language_state(state: str):
    data = LINGUISTIC_STATES[state]
    c = np.array([data["phonology"], data["semantics"], data["syntax"]])
    c = np.clip(c, EPSILON, 1.0 - EPSILON)
    w = np.ones(3) / 3
    out = compute_kernel_outputs(c, w)
    return out

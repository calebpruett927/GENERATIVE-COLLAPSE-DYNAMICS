import numpy as np

from umcp.frozen_contract import EPSILON
from umcp.kernel_optimized import compute_kernel_outputs

COMPLEXITY_CLASSES = {
    "P": {"halting_prob": 1.0, "circuit_depth": 0.9, "kolmogorov": 0.8},
    "NP": {"halting_prob": 1.0, "circuit_depth": 0.2, "kolmogorov": 0.3},
    "EXPTIME": {"halting_prob": 0.5, "circuit_depth": 0.1, "kolmogorov": 0.2},
    "RE": {"halting_prob": EPSILON, "circuit_depth": 1.0, "kolmogorov": 1.0},
    "R": {"halting_prob": 1.0, "circuit_depth": 0.5, "kolmogorov": 0.5},
}


def analyze_complexity_class(name: str):
    data = COMPLEXITY_CLASSES[name]
    c = np.array([data["halting_prob"], data["circuit_depth"], data["kolmogorov"]])
    c = np.clip(c, EPSILON, 1.0 - EPSILON)
    w = np.ones(3) / 3
    out = compute_kernel_outputs(c, w)
    return out

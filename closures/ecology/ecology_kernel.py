import numpy as np

from umcp.frozen_contract import EPSILON
from umcp.kernel_optimized import compute_kernel_outputs

TROPHIC_STATES = {
    "PRISTINE": {"biomass": 0.9, "diversity": 0.9, "connectivity": 0.9},
    "KEYSTONE_LOSS": {"biomass": 0.8, "diversity": 0.7, "connectivity": 0.2},
    "CASCADE": {"biomass": 0.4, "diversity": 0.3, "connectivity": EPSILON},
    "EXTINCTION": {"biomass": EPSILON, "diversity": EPSILON, "connectivity": EPSILON},
}


def analyze_ecology_state(state: str):
    data = TROPHIC_STATES[state]
    c = np.array([data["biomass"], data["diversity"], data["connectivity"]])
    c = np.clip(c, EPSILON, 1.0 - EPSILON)
    w = np.ones(3) / 3
    out = compute_kernel_outputs(c, w)
    return out

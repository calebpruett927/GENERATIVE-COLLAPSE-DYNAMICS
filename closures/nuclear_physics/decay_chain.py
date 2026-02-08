"""Decay Chain Closure — NUC.INTSTACK.v1

Traverses a multi-step radioactive decay cascade and computes
aggregate chain metrics.  The paradigm case is U-238 → Pb-206
(8α + 6β⁻, 14 steps, 32 nucleons shed, 10 protons shed).

Each step is specified as a dict with {isotope, Z, A, decay_mode,
half_life_s, Q_MeV}.  The closure computes cumulative metrics and
classifies the chain's overall temporal regime.

UMCP integration:
  The chain is treated as a single "collapse cascade" whose total
  timescale is dominated by the slowest step (bottleneck).
  Chain budget: ΔZ = Z_parent − Z_endpoint, ΔA = A_parent − A_endpoint

Regime classification (chain_dynamics):
  ZeroStep:   single stable nucleus (no chain)
  Dominated:  ≤ 3 steps
  Cascade:    4–10 steps
  DeepChain:  > 10 steps

Cross-references:
  Contract:  contracts/NUC.INTSTACK.v1.yaml
  Sources:   Bateman equations; IAEA Nuclear Data Services
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import NamedTuple


class ChainRegime(StrEnum):
    """Regime based on chain depth."""

    ZERO_STEP = "ZeroStep"
    DOMINATED = "Dominated"
    CASCADE = "Cascade"
    DEEP_CHAIN = "DeepChain"


class ChainStep(NamedTuple):
    """A single step in a decay chain."""

    isotope: str  # e.g. "U-238"
    Z: int  # Atomic number
    A: int  # Mass number
    decay_mode: str  # "alpha", "beta_minus", "beta_plus", etc.
    half_life_s: float  # Half-life in seconds
    Q_MeV: float  # Release energy for this step (MeV)


class DecayChainResult(NamedTuple):
    """Result of decay chain computation."""

    chain_length: int  # Number of decay steps
    total_nucleons_shed: int  # A_parent − A_endpoint
    total_protons_shed: int  # Z_parent − Z_endpoint
    total_neutrons_shed: int  # N_parent − N_endpoint
    alpha_count: int  # Number of α decays
    beta_minus_count: int  # Number of β⁻ decays
    beta_plus_count: int  # Number of β⁺/EC decays
    endpoint_isotope: str  # Final stable isotope
    endpoint_Z: int
    endpoint_A: int
    bottleneck_step: str  # Isotope with longest T½
    bottleneck_half_life_s: float
    total_Q_MeV: float  # Total energy released (MeV)
    log10_bottleneck_tau: float  # log₁₀(τ_bottleneck / s)
    regime: str


LN2 = math.log(2)


def _classify_regime(n_steps: int) -> ChainRegime:
    """Classify chain dynamics regime."""
    if n_steps == 0:
        return ChainRegime.ZERO_STEP
    if n_steps <= 3:
        return ChainRegime.DOMINATED
    if n_steps <= 10:
        return ChainRegime.CASCADE
    return ChainRegime.DEEP_CHAIN


def compute_decay_chain(
    steps: list[ChainStep] | list[dict[str, object]],
) -> DecayChainResult:
    """Compute aggregate decay chain metrics.

    Parameters
    ----------
    steps : list[ChainStep] | list[dict]
        Ordered list of decay steps from parent to endpoint.
        Each step describes one decay event.  The first step's
        isotope is the parent; the endpoint is the daughter of
        the last step.

    Returns
    -------
    DecayChainResult
    """
    if not steps:
        return DecayChainResult(
            chain_length=0,
            total_nucleons_shed=0,
            total_protons_shed=0,
            total_neutrons_shed=0,
            alpha_count=0,
            beta_minus_count=0,
            beta_plus_count=0,
            endpoint_isotope="(none)",
            endpoint_Z=0,
            endpoint_A=0,
            bottleneck_step="(none)",
            bottleneck_half_life_s=0.0,
            total_Q_MeV=0.0,
            log10_bottleneck_tau=0.0,
            regime=ChainRegime.ZERO_STEP.value,
        )

    # Normalize to ChainStep namedtuples
    normalized: list[ChainStep] = []
    for s in steps:
        if isinstance(s, dict):
            normalized.append(
                ChainStep(
                    isotope=str(s["isotope"]),
                    Z=int(s["Z"]),  # type: ignore[arg-type]
                    A=int(s["A"]),  # type: ignore[arg-type]
                    decay_mode=str(s["decay_mode"]),
                    half_life_s=float(s["half_life_s"]),  # type: ignore[arg-type]
                    Q_MeV=float(s["Q_MeV"]),  # type: ignore[arg-type]
                )
            )
        else:
            normalized.append(s)  # type: ignore[arg-type]

    parent = normalized[0]

    # Compute endpoint from the last step's decay
    last = normalized[-1]
    if last.decay_mode == "alpha":
        end_Z = last.Z - 2
        end_A = last.A - 4
    elif last.decay_mode == "beta_minus":
        end_Z = last.Z + 1
        end_A = last.A
    elif last.decay_mode in ("beta_plus", "electron_capture"):
        end_Z = last.Z - 1
        end_A = last.A
    else:
        end_Z = last.Z
        end_A = last.A

    # Counters
    alpha_ct = sum(1 for s in normalized if s.decay_mode == "alpha")
    beta_m_ct = sum(1 for s in normalized if s.decay_mode == "beta_minus")
    beta_p_ct = sum(1 for s in normalized if s.decay_mode in ("beta_plus", "electron_capture"))
    total_Q = sum(s.Q_MeV for s in normalized)

    # Bottleneck (longest half-life)
    bottleneck = max(normalized, key=lambda s: s.half_life_s)
    bn_tau = bottleneck.half_life_s / LN2 if bottleneck.half_life_s < float("inf") else float("inf")
    log10_bn_tau = (
        math.log10(bn_tau) if 0 < bn_tau < float("inf") else (float("inf") if bn_tau == float("inf") else float("-inf"))
    )

    n_steps = len(normalized)
    regime = _classify_regime(n_steps)

    # Derive endpoint name from Z
    _element_symbols = {
        82: "Pb",
        83: "Bi",
        84: "Po",
        86: "Rn",
        88: "Ra",
        90: "Th",
        92: "U",
        81: "Tl",
        85: "At",
        87: "Fr",
        89: "Ac",
        91: "Pa",
    }
    end_sym = _element_symbols.get(end_Z, f"Z{end_Z}")
    end_iso = f"{end_sym}-{end_A}"

    return DecayChainResult(
        chain_length=n_steps,
        total_nucleons_shed=parent.A - end_A,
        total_protons_shed=parent.Z - end_Z,
        total_neutrons_shed=(parent.A - parent.Z) - (end_A - end_Z),
        alpha_count=alpha_ct,
        beta_minus_count=beta_m_ct,
        beta_plus_count=beta_p_ct,
        endpoint_isotope=end_iso,
        endpoint_Z=end_Z,
        endpoint_A=end_A,
        bottleneck_step=bottleneck.isotope,
        bottleneck_half_life_s=bottleneck.half_life_s,
        total_Q_MeV=round(total_Q, 4),
        log10_bottleneck_tau=round(log10_bn_tau, 4) if log10_bn_tau != float("inf") else float("inf"),
        regime=regime.value,
    )


# ── Self-test: U-238 → Pb-206 natural decay series ──────────────
if __name__ == "__main__":
    # Uranium-238 decay chain (8α + 6β⁻ = 14 steps)
    u238_chain: list[ChainStep] = [
        ChainStep("U-238", 92, 238, "alpha", 1.41e17, 4.270),
        ChainStep("Th-234", 90, 234, "beta_minus", 2.08e6, 0.273),
        ChainStep("Pa-234", 91, 234, "beta_minus", 6.70e1, 2.197),
        ChainStep("U-234", 92, 234, "alpha", 7.74e12, 4.858),
        ChainStep("Th-230", 90, 230, "alpha", 2.38e12, 4.770),
        ChainStep("Ra-226", 88, 226, "alpha", 5.05e10, 4.871),
        ChainStep("Rn-222", 86, 222, "alpha", 3.30e5, 5.590),
        ChainStep("Po-218", 84, 218, "alpha", 1.86e2, 6.115),
        ChainStep("Pb-214", 82, 214, "beta_minus", 1.61e3, 1.024),
        ChainStep("Bi-214", 83, 214, "beta_minus", 1.19e3, 3.272),
        ChainStep("Po-214", 84, 214, "alpha", 1.64e-4, 7.833),
        ChainStep("Pb-210", 82, 210, "beta_minus", 7.01e8, 0.064),
        ChainStep("Bi-210", 83, 210, "beta_minus", 4.33e5, 1.163),
        ChainStep("Po-210", 84, 210, "alpha", 1.20e7, 5.407),
    ]

    result = compute_decay_chain(u238_chain)
    print(f"U-238 → {result.endpoint_isotope}")
    print(
        f"  Steps:    {result.chain_length}  (α={result.alpha_count}, "
        f"β⁻={result.beta_minus_count}, β⁺={result.beta_plus_count})"
    )
    print(
        f"  Shed:     ΔA={result.total_nucleons_shed}, ΔZ={result.total_protons_shed}, ΔN={result.total_neutrons_shed}"
    )
    print(f"  Q_total:  {result.total_Q_MeV:.3f} MeV")
    print(f"  Bottleneck: {result.bottleneck_step} — log₁₀(τ/s)={result.log10_bottleneck_tau}")
    print(f"  Regime:   {result.regime}")

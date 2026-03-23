"""Return Rope — Adaptive return cycle for traversing unknown territories.

The return rope formalizes the iterative exploration-return pattern
where each "parse" through unknown territory is a collapse-return cycle.
The rope contracts on successful return, building grip with each
iteration.

Mathematical formulation
------------------------

**Contraction rule** (per parse):

    τ_R^(k+1) = τ_R^(k) · (1 − IC^(k) · ω^(k))

**Contraction bound** — since IC ≤ F = 1 − ω (integrity bound):

    IC · ω  ≤  (1 − ω) · ω  ≤  1/4      (maximum at ω = 1/2)

So the rope contracts by at most 25% per parse — no catastrophic
over-tightening.

**Grip** (cumulative compounding):

    grip_k  =  1 − ∏_{i=1}^{k}  (1 − IC^(i) · ω^(i))

Each parse contributes IC·ω to closing the remaining gap between
grip and 1.  The product of return quality (IC) and exploration
depth (ω) is the compounding factor.

**Gesture detection**:

    If IC^(k) < ε  →  τ_R = ∞_rec  (rope snaps; territory is a gesture)

**Convergence**:

    |τ_R^(k+1) − τ_R^(k)| < tol_seam  →  territory is grounded

Axiom-0 constraint: *solum quod redit, reale est* — only parses
that return (IC ≥ ε) contribute to grip.  A snapped rope is an
honest verdict, not a failure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from umcp.frozen_contract import EPSILON, TOL_SEAM
from umcp.kernel_optimized import compute_kernel_outputs

# ─────────────────────────────────────────────────────────────────
# Result containers (immutable)
# ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ParseResult:
    """Immutable result of one exploration parse."""

    iteration: int
    trace: tuple[float, ...]
    weights: tuple[float, ...]
    label: str

    # Tier-1 kernel invariants
    F: float
    omega: float
    IC: float
    kappa: float
    S: float
    C: float
    heterogeneity_gap: float
    regime: str

    # Rope dynamics
    tau_r: float  # Rope length after this parse
    grip: float  # Cumulative grip after this parse
    contraction: float  # IC · ω — contraction factor for this parse
    is_gesture: bool  # True if IC < ε (rope snapped)


# ─────────────────────────────────────────────────────────────────
# Rope state (mutable, evolves with parses)
# ─────────────────────────────────────────────────────────────────


@dataclass
class RopeState:
    """Current state of the return rope."""

    iteration: int = 0
    tau_r: float = 1.0
    grip: float = 0.0
    converged: bool = False
    snapped: bool = False
    parses: list[ParseResult] = field(default_factory=list)

    @property
    def ic_sequence(self) -> list[float]:
        """IC values across all parses."""
        return [p.IC for p in self.parses]

    @property
    def tau_sequence(self) -> list[float]:
        """Rope length after each parse."""
        return [p.tau_r for p in self.parses]

    @property
    def grip_sequence(self) -> list[float]:
        """Grip after each parse."""
        return [p.grip for p in self.parses]

    @property
    def contraction_sequence(self) -> list[float]:
        """Contraction factor at each parse."""
        return [p.contraction for p in self.parses]


# ─────────────────────────────────────────────────────────────────
# The Return Rope
# ─────────────────────────────────────────────────────────────────


class ReturnRope:
    """Adaptive return rope for traversing unknown territories.

    Each parse through unknown territory is a collapse-return cycle.
    The rope contracts on successful return, building grip with each
    iteration.

    The contraction is bounded: IC · ω ≤ 1/4, so the rope can lose
    at most 25% of its length per parse.

    Parameters
    ----------
    initial_tau : float
        Starting rope length (default 1.0).
    convergence_tol : float
        Rope converges when |Δτ| < this tolerance (default tol_seam).
    epsilon : float
        Guard band.  IC < epsilon triggers gesture detection.
    """

    def __init__(
        self,
        initial_tau: float = 1.0,
        convergence_tol: float = TOL_SEAM,
        gesture_threshold: float = EPSILON**0.25,
        epsilon: float = EPSILON,
    ) -> None:
        self._initial_tau = initial_tau
        self._convergence_tol = convergence_tol
        self._gesture_threshold = gesture_threshold
        self._epsilon = epsilon
        self._state = RopeState(tau_r=initial_tau)
        self._gap_product = 1.0  # Running ∏(1 − IC_i · ω_i)

    # ── properties ────────────────────────────────────────────

    @property
    def state(self) -> RopeState:
        """Current rope state."""
        return self._state

    @property
    def initial_tau(self) -> float:
        """Initial rope length."""
        return self._initial_tau

    # ── core operation ────────────────────────────────────────

    def parse(
        self,
        trace: list[float] | tuple[float, ...] | np.ndarray,
        weights: list[float] | tuple[float, ...] | np.ndarray | None = None,
        label: str = "",
    ) -> ParseResult:
        """Execute one exploration parse.

        Computes kernel on the trace vector, contracts the rope based
        on IC · ω, and updates grip.

        Parameters
        ----------
        trace : array-like
            Channel values in [0, 1].
        weights : array-like or None
            Simplex weights (default: uniform 1/n).
        label : str
            Human-readable label for this parse.

        Returns
        -------
        ParseResult
            Immutable result with kernel invariants and rope dynamics.

        Raises
        ------
        RuntimeError
            If the rope has already snapped (gesture detected).
        """
        if self._state.snapped:
            msg = (
                "Rope has snapped (gesture detected at parse "
                f"{self._state.iteration}). "
                "Call reset() or create a new ReturnRope."
            )
            raise RuntimeError(msg)

        # Normalize inputs
        c = np.asarray(trace, dtype=np.float64)
        n = len(c)
        w = np.full(n, 1.0 / n) if weights is None else np.asarray(weights, dtype=np.float64)

        # Compute kernel
        kernel = compute_kernel_outputs(c, w, self._epsilon)

        F = kernel["F"]
        omega = kernel["omega"]
        IC = kernel["IC"]
        kappa = kernel["kappa"]
        S = kernel["S"]
        C = kernel["C"]
        gap = kernel["heterogeneity_gap"]
        regime = kernel["regime"]

        # Gesture detection: IC < gesture_threshold → rope snaps
        is_gesture = bool(self._gesture_threshold > IC)

        if is_gesture:
            self._state.snapped = True
            contraction = 0.0
            new_tau = float("inf")  # ∞_rec
            new_grip = self._state.grip  # Grip freezes
        else:
            # Contraction factor: IC · ω
            # Bounded by (1−ω)·ω ≤ 1/4 since IC ≤ F = 1−ω
            contraction = IC * omega

            # Contract rope
            new_tau = self._state.tau_r * (1.0 - contraction)

            # Update grip: grip = 1 − ∏(1 − IC_i · ω_i)
            self._gap_product *= 1.0 - contraction
            new_grip = 1.0 - self._gap_product

        # Convergence check (needs ≥ 2 parses)
        iteration = self._state.iteration + 1
        converged = not is_gesture and iteration >= 2 and abs(new_tau - self._state.tau_r) < self._convergence_tol

        result = ParseResult(
            iteration=iteration,
            trace=tuple(float(x) for x in c),
            weights=tuple(float(x) for x in w),
            label=label,
            F=F,
            omega=omega,
            IC=IC,
            kappa=kappa,
            S=S,
            C=C,
            heterogeneity_gap=gap,
            regime=regime,
            tau_r=new_tau,
            grip=new_grip,
            contraction=contraction,
            is_gesture=is_gesture,
        )

        # Mutate state
        self._state.iteration = iteration
        self._state.tau_r = new_tau
        self._state.grip = new_grip
        self._state.converged = converged
        self._state.parses.append(result)

        return result

    # ── multi-parse convenience ───────────────────────────────

    def explore(
        self,
        traces: list[list[float] | tuple[float, ...] | np.ndarray],
        weights: list[float] | tuple[float, ...] | np.ndarray | None = None,
        labels: list[str] | None = None,
    ) -> list[ParseResult]:
        """Execute a sequence of parses.

        Parameters
        ----------
        traces : list of array-like
            One trace vector per parse.
        weights : array-like or None
            Shared weights for all parses (default: uniform).
        labels : list of str or None
            One label per parse (default: "parse_1", "parse_2", ...).

        Returns
        -------
        list[ParseResult]
            Results for each parse, in order.
        """
        if labels is None:
            labels = [f"parse_{i + 1}" for i in range(len(traces))]

        results: list[ParseResult] = []
        for i, trace in enumerate(traces):
            lbl = labels[i] if i < len(labels) else f"parse_{i + 1}"
            result = self.parse(trace, weights=weights, label=lbl)
            results.append(result)
            if self._state.snapped:
                break  # Rope snapped — stop exploration
        return results

    # ── reset ─────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset rope to initial state."""
        self._state = RopeState(tau_r=self._initial_tau)
        self._gap_product = 1.0

    # ── summary ───────────────────────────────────────────────

    def summary(self) -> dict[str, Any]:
        """Summary statistics of the exploration."""
        if not self._state.parses:
            return {
                "parses": 0,
                "grip": 0.0,
                "tau_r": self._initial_tau,
                "converged": False,
                "snapped": False,
            }

        n = self._state.iteration
        return {
            "parses": n,
            "initial_tau": self._initial_tau,
            "final_tau": self._state.tau_r,
            "contraction_ratio": (self._state.tau_r / self._initial_tau if not self._state.snapped else float("inf")),
            "grip": self._state.grip,
            "mean_ic": sum(p.IC for p in self._state.parses) / n,
            "mean_omega": sum(p.omega for p in self._state.parses) / n,
            "mean_contraction": sum(p.contraction for p in self._state.parses) / n,
            "max_contraction": max(p.contraction for p in self._state.parses),
            "converged": self._state.converged,
            "snapped": self._state.snapped,
            "gesture_count": sum(1 for p in self._state.parses if p.is_gesture),
        }


# ─────────────────────────────────────────────────────────────────
# Contraction bound proof (standalone, for orientation / testing)
# ─────────────────────────────────────────────────────────────────


def contraction_bound_proof(n_samples: int = 10_000) -> dict[str, float]:
    """Empirically verify the contraction bound IC · ω ≤ 1/4.

    Samples random trace vectors and checks that the product of
    IC and ω never exceeds 1/4.  The theoretical maximum is at
    ω = 1/2 (equivalently F = 1/2) with homogeneous channels.

    Returns
    -------
    dict
        max_product, theoretical_max (0.25), and margin.
    """
    rng = np.random.default_rng(42)
    max_product = 0.0

    for _ in range(n_samples):
        n_ch = rng.integers(2, 16)
        c = rng.uniform(0.0, 1.0, size=n_ch)
        w = np.full(n_ch, 1.0 / n_ch)
        kernel = compute_kernel_outputs(c, w, EPSILON)
        product = kernel["IC"] * kernel["omega"]
        if product > max_product:
            max_product = product

    return {
        "max_product": max_product,
        "theoretical_max": 0.25,
        "margin": 0.25 - max_product,
        "bound_holds": max_product <= 0.25 + 1e-12,
    }


def grip_convergence_analysis(
    ic_omega_constant: float = 0.1,
    n_parses: int = 50,
) -> dict[str, Any]:
    """Analyze grip convergence for a constant IC·ω contraction.

    Shows how grip approaches 1.0 as a geometric series:
        grip_k = 1 − (1 − c)^k  where c = IC · ω

    Returns
    -------
    dict
        Grip sequence, half-life (parses to reach grip=0.5),
        and 90%-life (parses to grip=0.9).
    """
    c = ic_omega_constant
    grips = [1.0 - (1.0 - c) ** k for k in range(1, n_parses + 1)]

    half_life = math.ceil(math.log(0.5) / math.log(1.0 - c)) if c > 0 else float("inf")
    ninety_life = math.ceil(math.log(0.1) / math.log(1.0 - c)) if c > 0 else float("inf")

    return {
        "ic_omega": c,
        "grip_sequence": grips,
        "final_grip": grips[-1],
        "half_life": half_life,
        "ninety_life": ninety_life,
    }

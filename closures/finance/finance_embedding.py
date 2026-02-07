"""
Financial Metric Embedding

Maps raw financial data to UMCP coordinates in [0,1]^4.

Embedding rules (frozen under FINANCE.INTSTACK.v1):
    c₁ = min(revenue / revenue_target, 1.0)         — Revenue performance
    c₂ = min(expense_budget / expenses, 1.0)         — Expense control
    c₃ = (revenue - cogs) / revenue                  — Gross margin
    c₄ = min(cashflow / cashflow_target, 1.0)        — Cash flow health

All coordinates are clipped to [ε, 1-ε] after embedding (Lemma 17).

Cross-references:
    - KERNEL_SPECIFICATION.md (Lemma 1: range bounds)
    - contracts/FINANCE.INTSTACK.v1.yaml
    - src/umcp/frozen_contract.py (EPSILON)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np  # pyright: ignore[reportMissingImports]

EPSILON: float = 1e-8

# Default weights: revenue-weighted for business priority
DEFAULT_WEIGHTS: list[float] = [0.30, 0.25, 0.25, 0.20]

COORDINATE_NAMES: list[str] = [
    "revenue_performance",
    "expense_control",
    "gross_margin",
    "cashflow_health",
]


@dataclass(frozen=True)
class FinanceTargets:
    """Frozen financial targets for embedding.

    These are Tier-0 declarations — they must be frozen before kernel compute.
    Changing targets requires a new contract variant.
    """

    revenue_target: float  # Monthly revenue target ($)
    expense_budget: float  # Monthly expense budget ($)
    cashflow_target: float  # Monthly cash flow target ($)

    def to_dict(self) -> dict[str, float]:
        return {
            "revenue_target": self.revenue_target,
            "expense_budget": self.expense_budget,
            "cashflow_target": self.cashflow_target,
        }


@dataclass(frozen=True)
class FinanceRecord:
    """One month of raw financial data.

    These are Tier-0 observables — raw measurements with units (USD).
    """

    month: str  # YYYY-MM format
    revenue: float  # Monthly revenue ($)
    expenses: float  # Monthly total expenses ($)
    cogs: float  # Cost of goods sold ($)
    cashflow: float  # Operating cash flow ($)

    def to_dict(self) -> dict[str, Any]:
        return {
            "month": self.month,
            "revenue": self.revenue,
            "expenses": self.expenses,
            "cogs": self.cogs,
            "cashflow": self.cashflow,
        }


@dataclass(frozen=True)
class EmbeddedFinance:
    """Embedded financial coordinates in [0,1]^4.

    Result of Tier-0 embedding: raw → Ψ(t) ∈ [ε, 1-ε]^4.
    """

    c: np.ndarray  # Coordinates [c1, c2, c3, c4]
    oor_flags: list[bool]  # Out-of-range flags per dimension
    month: str  # Source month


def embed_finance(
    record: FinanceRecord,
    targets: FinanceTargets,
    epsilon: float = EPSILON,
) -> EmbeddedFinance:
    """
    Embed raw financial data into UMCP coordinates.

    This is the Tier-0 embedding N_K: x(t) → Ψ(t) ∈ [0,1]^4.

    Args:
        record: Raw financial record
        targets: Frozen financial targets
        epsilon: Guard band for log-safety clipping

    Returns:
        EmbeddedFinance with coordinates and OOR flags
    """
    oor_flags: list[bool] = []

    # c₁: Revenue performance = min(revenue / target, 1.0)
    c1_raw = record.revenue / targets.revenue_target if targets.revenue_target > 0 else 0.0
    oor_flags.append(c1_raw > 1.0 or c1_raw < 0.0)
    c1 = min(max(c1_raw, 0.0), 1.0)

    # c₂: Expense control = min(budget / expenses, 1.0)
    # Under budget → 1.0 (perfect), over budget → scales down proportionally
    c2_raw = targets.expense_budget / record.expenses if record.expenses > 0 else 1.0
    oor_flags.append(c2_raw < 1.0)  # Over budget
    c2 = min(max(c2_raw, 0.0), 1.0)

    # c₃: Gross margin = (revenue - cogs) / revenue
    c3_raw = (record.revenue - record.cogs) / record.revenue if record.revenue > 0 else 0.0
    oor_flags.append(c3_raw < 0.0 or c3_raw > 1.0)
    c3 = min(max(c3_raw, 0.0), 1.0)

    # c₄: Cash flow health = min(cashflow / target, 1.0)
    c4_raw = record.cashflow / targets.cashflow_target if targets.cashflow_target > 0 else 0.0
    oor_flags.append(c4_raw > 1.0 or c4_raw < 0.0)
    c4 = min(max(c4_raw, 0.0), 1.0)

    # Clip to [ε, 1-ε] for log-safety (Lemma 17)
    coords = np.array([c1, c2, c3, c4])
    coords_clipped = np.clip(coords, epsilon, 1.0 - epsilon)

    return EmbeddedFinance(
        c=coords_clipped,
        oor_flags=oor_flags,
        month=record.month,
    )

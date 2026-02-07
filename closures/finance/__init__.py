"""
Finance Domain Closures

Embedding and cost closures for business financial continuity validation
under UMCP FINANCE.INTSTACK.v1 contract.

Coordinates:
    c₁ = revenue_performance  = min(revenue / revenue_target, 1.0)
    c₂ = expense_control      = 1.0 - min(expenses / expense_budget, 1.0)
    c₃ = gross_margin          = (revenue - cogs) / revenue
    c₄ = cashflow_health       = min(cashflow / cashflow_target, 1.0)

Default weights: [0.30, 0.25, 0.25, 0.20]

Cross-references:
    - contracts/FINANCE.INTSTACK.v1.yaml
    - closures/registry.yaml (finance section)
    - src/umcp/finance_cli.py
"""

from __future__ import annotations

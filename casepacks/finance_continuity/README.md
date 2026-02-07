# Financial Continuity CasePack

**Contract**: `FINANCE.INTSTACK.v1`  
**Cadence**: Monthly  
**Coordinates**: Revenue Performance, Expense Control, Gross Margin, Cash Flow Health

## What This CasePack Demonstrates

A 12-month financial trajectory showing a business that:

1. **Months 0-3**: Operates near targets (STABLE regime)
2. **Months 4-5**: Drifts from targets (WATCH regime) 
3. **Month 6**: Financial stress deepens (CRITICAL — IC < 0.10)
4. **Month 7**: Peak stress (COLLAPSE — ω ≥ 0.30)
5. **Month 8**: Recovers toward targets (WATCH, τ_R = 4)
6. **Months 9-11**: Re-stabilizes (STABLE regime, τ_R = 1)

This is the axiom in action: the business's financial health *collapsed* and then *returned*. The return is certified by seam accounting — not by narrative ("we feel better about Q4") but by demonstrated re-entry into the η-neighborhood of prior stable states.

## Embedding

| Coordinate | Formula | Weight |
|------------|---------|--------|
| c₁ (Revenue) | `min(revenue / 200000, 1.0)` | 0.30 |
| c₂ (Expenses) | `min(150000 / expenses, 1.0)` | 0.25 |
| c₃ (Margin) | `(revenue - cogs) / revenue` | 0.25 |
| c₄ (Cash Flow) | `min(cashflow / 40000, 1.0)` | 0.20 |

## Running

```bash
# Generate expected outputs from raw measurements
python casepacks/finance_continuity/generate_expected.py

# Validate the casepack
umcp validate casepacks/finance_continuity

# Use the finance CLI for ongoing recording
umcp-finance record --month 2026-01 --revenue 215000 --expenses 148000 --cogs 116000 --cashflow 50000
umcp-finance analyze
umcp-finance report
umcp-finance dashboard
```

## Regime Interpretation

| Regime | Financial Meaning |
|--------|-------------------|
| **STABLE** | Operating within targets. Low drift (ω < 0.18), high fidelity (F > 0.80). |
| **WATCH** | One or more metrics drifting (0.18 ≤ ω < 0.30). Review recommended. |
| **COLLAPSE** | Significant deviation (ω ≥ 0.30). Urgent action required. |
| **CRITICAL** | Integrity below 0.10. Financial health not demonstrable from data. |

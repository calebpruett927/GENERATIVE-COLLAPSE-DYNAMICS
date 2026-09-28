# Structura Reditus Tableau control-surface export

This integration prepares repository run artifacts for Tableau without changing their evidentiary or authority status.

## Burden

Expose the active GCD/UMCP run archive in a form that Tableau can query while preserving distinctions required by Structura Reditus:

- current vs archived execution,
- run identity and frozen contract,
- source-declared kernel values,
- typed or source-specific return tokens,
- missingness,
- seam evidence,
- receipts,
- casepack identity,
- source regime vs final stance.

The exporter is a **derived restatement**. It creates no new evidence and performs no authority promotion.

## Main dashboard tables

- `dashboard_run_summary.csv`: one row per run, with source/run identity plus latest and mean diagnostics and event counts.
- `kernel_observations.csv`: one source-declared kernel row per time point.
- `return_observations.csv`: return values and token classes.
- `missingness_events.csv`: censor and OOR records.
- `seam_events.csv`: seam ledger records, explicitly without derived weld.
- `contract_fields.csv`: flattened manifest and frozen-contract fields.
- `receipt_fields.csv`: receipt fields preserved as source receipts only.
- `casepacks.csv`: recursively discovered casepack manifests.

## Boundary

The exporter does not reinterpret historical/run-local uses of reserved symbols as current canon. It does not turn `inf` into `INF_REC`; it marks that value as a source infinity token. It does not turn a regime into `CONFORMANT`, `NONCONFORMANT`, or `NON_EVALUABLE`. It does not infer weld from seam or receipt presence.

## Build

```bash
python integrations/tableau/export_dataset.py \
  --repo-root . \
  --output-dir artifacts/tableau
python integrations/tableau/validate_export.py artifacts/tableau
```

The GitHub Actions workflow `Build Structura Tableau Dataset` performs the same build from the checked-out repository state and uploads the validated bundle as an artifact.

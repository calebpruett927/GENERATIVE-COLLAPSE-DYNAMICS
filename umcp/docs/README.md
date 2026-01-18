# UMCP runtime docs (`umcp/docs/`)

This directory contains the runnable documentation for **UMCP-Metadata-Runnable-Code**. The intended reader is someone who wants to verify UMCP artifacts mechanically (schemas, contracts, closures, CasePacks, receipts) and reproduce the minimal runs described by the manuscript.

Design goal: reduce misimplementation by making the runnable path obvious, pinned, and auditable.

## Start here

- `quickstart.md`  
  Clone → validate → run a minimal CasePack → inspect invariants and receipts.

## Reference docs

- `validator.md`  
  What the validator checks, what outputs look like, and how failures are classified.

- `casepacks.md`  
  CasePack layout rules, what “L0” means mechanically, and how to add new runnable examples.

- `closures.md`  
  Closure registry discipline (pinning non-kernel degrees of freedom) and cookbook growth rules.

- `contracts.md`  
  Contract immutability, version governance, and what must be pinned for interpretability.

## Growth rule

When the manuscript needs clarification, prefer adding runnable artifacts (CasePacks + validator rules) over narrative expansion. This docs folder should always reflect the runnable truth.

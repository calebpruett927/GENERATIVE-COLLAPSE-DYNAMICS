# Contracts changelog

This changelog documents *additions* of new contract versions. Contracts are immutable once released; do not edit old entries, append new ones.

## UMA.INTSTACK

### UMA.INTSTACK.v1 (contract id: `UMA.INTSTACK.v1`)
- File: `contracts/UMA.INTSTACK.v1.yaml`
- Contract version field: `1.0.0`
- Purpose:
  - Freeze embedding interval, face, OOR policy, epsilon
  - Reserve Tier-1 kernel symbol set
  - Pin typed-censoring enums used across receipts and validators
  - Pin seam tolerances (`tol_seam`, `tol_id`) and frozen parameters (`p`, `alpha`, `lambda`, `eta`)

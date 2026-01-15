UMCP Contracts

This folder contains contract documents that define the frozen, machine-checkable execution boundary for UMCP runs.

A contract is not a narrative. It is an enforceable specification that the validator and runtime use to determine:
- embedding bounds and OOR policy (Tier-0)
- which Tier-1 kernel symbols are reserved and how they are computed
- typed-censoring domains (e.g., τ_R special values and run_status enum)
- frozen defaults and tolerances required for weld/receipt correctness

Immutability and versioning

Published contracts should be treated as immutable. If you need a semantic change:
- create a new contract id/version file (e.g., UMA.INTSTACK.v2.yaml)
- do not modify UMA.INTSTACK.v1.yaml in-place once it is referenced by published CasePacks
- update references in new CasePacks to the new contract

Schema

All contract files in this folder must validate against:
- schemas/contract.schema.json

Naming convention

- contracts/<CONTRACT_ID>.yaml
Example:
- contracts/UMA.INTSTACK.v1.yaml

Operational note

The contract’s role is to freeze what must not drift across implementations. Anything not frozen here must be explicitly frozen via closures and the closure registry, or it is not eligible for “conformant” claims.

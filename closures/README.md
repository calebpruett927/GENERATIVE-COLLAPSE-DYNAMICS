UMCP Closures

This folder contains the explicit closure set required to make UMCP runs executable and auditable.

Closures are where you freeze the “missing pieces” that cannot be derived from Tier-1 identities alone:
- Γ(ω; p, ε) form (drift dissipation closure used in seam budget accounting)
- Return-domain generator D_θ (how candidate return times are selected)
- Norm choice ‖·‖ and neighborhood tolerance η (how return proximity is measured)

Immutability and versioning

Published closure sets should be treated as immutable. If you need a semantic change:
- create a new registry id and new closure filenames (do not modify in place)
- update new CasePacks to point at the new registry id + new closure files

Schema

All closure documents in this folder must validate against:
- schemas/closures.schema.json

Registry

The active closure set is pinned by:
- closures/registry.yaml

That registry references the specific closure files used for a run. A run is not conformant if the closure registry is missing, ambiguous, or references non-existent files.

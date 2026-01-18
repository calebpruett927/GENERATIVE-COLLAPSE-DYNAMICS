# CasePacks — runnable publication units

A CasePack is the preferred growth surface for UMCP: a runnable and auditable unit that demonstrates:

- `x → Ψ` mapping (with flags),
- Tier-1 invariant computation under a pinned contract,
- regime labeling from kernel gates,
- receipt emission (SS1m always; seam only when claiming continuity).

## Minimal L0 CasePack layout

```text
casepacks/<domain>_l0/
  manifest.json
  expected/
    psi.csv
    invariants.json
    ss1m_receipt.json

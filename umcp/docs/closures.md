
---

UMCP-Metadata-Runnable-Code/umcp/docs/closures.md
```markdown
# Closures and closure registries

Closures pin non-kernel degrees of freedom so that runs are comparable and (when needed) weld-ready.

A **closures registry** is a single identifier that resolves to a concrete set of closure definitions, such as:
- Γ form (for `D_ω`)
- return-domain generator (for `τ_R`)
- norm and `η` used by the return test
- curvature neighborhood method (if not fully fixed by contract)

## Core discipline

- Closures do not redefine Tier-1 kernel symbols.
- Closures must be explicit, versioned, and discoverable via registry id.
- If a closure change affects meaning or any computed quantity, it must be represented by a new `closures_registry_id`.

## What a closure entry must include

Minimum closure metadata:
- what it is (family name + math form)
- what it pins (parameters + discrete policy choices)
- what it changes (which computed quantities)
- governance impact (what forces new IDs/variants)

## Registry discipline (why it exists)

The registry prevents silent drift.

Two runs are comparable only if:
- contract pins are compatible (or explicitly governed), and
- the closures registry id is identical (or explicitly governed).

If the closures registry changes:
- the comparison must be governed as an identity/variant boundary, and
- if continuity is claimed across the change, a seam receipt is required.

## Cookbook growth rule (recommended)

When you add a closure option that a reader might implement differently:
1) add a versioned closure file,
2) add or version a registry file that pins the closure selection,
3) add a tiny CasePack that exercises the change,
4) add a falsifier/validator rule if it is a common silent-drift vector.

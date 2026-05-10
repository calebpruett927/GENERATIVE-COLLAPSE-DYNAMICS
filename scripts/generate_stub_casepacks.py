#!/usr/bin/env python3
"""Generate casepack stubs for the 6 missing-coverage domains.

Per casepacks/TAXONOMY.md "Coverage Gap": these domains have closure code
and tests but no packaged demonstration unit. This script writes minimal
Tier-1 stubs (single Stable-regime row) so each registered domain has a
casepack of record under closures/full/<domain>/.

Each stub is intentionally minimal:
- 3 channels at c=0.99 → Stable regime (matches hello_world pattern)
- Domain-specific contract used when available; falls back to UMA otherwise
- README explicitly flags the casepack as a stub pending domain-specific data
"""

from __future__ import annotations

import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CASEPACKS = REPO / "casepacks" / "closures" / "full"

# Stub specifications. Each entry: domain, contract_id, contract_path, canon_path,
# title, one-line description.
STUBS = [
    {
        "domain": "nuclear_physics",
        "contract_id": "NUC.INTSTACK.v1",
        "contract_path": "contracts/NUC.INTSTACK.v1.yaml",
        "canon_path": "canon/nuc_anchors.yaml",
        "title": "Nuclear Physics Demonstration Stub",
        "description": (
            "Minimal Tier-1 demonstration unit for the nuclear_physics domain. "
            "Full closure code lives in closures/nuclear_physics/ (binding energy, "
            "decay chains, QGP/RHIC); this casepack is a placeholder packaging "
            "demonstration pending domain-specific raw measurements."
        ),
    },
    {
        "domain": "consciousness_coherence",
        "contract_id": "CONS.INTSTACK.v1",
        "contract_path": "contracts/CONS.INTSTACK.v1.yaml",
        "canon_path": "canon/cons_anchors.yaml",
        "title": "Consciousness Coherence Demonstration Stub",
        "description": (
            "Minimal Tier-1 demonstration unit for the consciousness_coherence "
            "domain. Full closure code lives in closures/consciousness_coherence/ "
            "(coherence kernel across 20 systems, altered states, neural correlates); "
            "this casepack is a placeholder packaging demonstration."
        ),
    },
    {
        "domain": "standard_model",
        "contract_id": "SM.INTSTACK.v1",
        "contract_path": "contracts/SM.INTSTACK.v1.yaml",
        "canon_path": "canon/sm_anchors.yaml",
        "title": "Standard Model Demonstration Stub",
        "description": (
            "Minimal Tier-1 demonstration unit for the standard_model domain. "
            "Full closure code lives in closures/standard_model/ (subatomic kernel, "
            "27 proven theorems, CKM/PMNS mixing, matter genesis); this casepack "
            "is a placeholder packaging demonstration."
        ),
    },
    {
        "domain": "ecology",
        "contract_id": "UMA.INTSTACK.v1",
        "contract_path": "contracts/UMA.INTSTACK.v1.yaml",
        "canon_path": "canon/anchors.yaml",
        "title": "Ecology Demonstration Stub",
        "description": (
            "Minimal Tier-1 demonstration unit for the ecology domain (uses universal "
            "UMA contract pending a dedicated ECOL contract). Full closure code lives "
            "in closures/ecology/; this casepack is a placeholder packaging demonstration."
        ),
    },
    {
        "domain": "immunology",
        "contract_id": "UMA.INTSTACK.v1",
        "contract_path": "contracts/UMA.INTSTACK.v1.yaml",
        "canon_path": "canon/imm_anchors.yaml",
        "title": "Immunology Demonstration Stub",
        "description": (
            "Minimal Tier-1 demonstration unit for the immunology domain (uses universal "
            "UMA contract pending a dedicated IMM contract). Full closure code lives in "
            "closures/immunology/; this casepack is a placeholder packaging demonstration."
        ),
    },
    {
        "domain": "information_theory",
        "contract_id": "UMA.INTSTACK.v1",
        "contract_path": "contracts/UMA.INTSTACK.v1.yaml",
        "canon_path": "canon/anchors.yaml",
        "title": "Information Theory Demonstration Stub",
        "description": (
            "Minimal Tier-1 demonstration unit for the information_theory domain (uses "
            "universal UMA contract pending a dedicated INFO contract). Full closure "
            "code lives in closures/information_theory/; this casepack is a placeholder."
        ),
    },
]

# Tier-1 invariants for the canonical Stable row (c = 0.99 across 3 channels).
# Verified against hello_world: F = 0.99, IC = exp(-0.01005...) = 0.99, S = 0.05600...
C_VAL = 0.99
F_VAL = C_VAL  # uniform channels: F = c
KAPPA = math.log(max(C_VAL, 1e-8))
IC_VAL = math.exp(KAPPA)
# Bernoulli field entropy: -[c ln(c) + (1-c) ln(1-c)] for c in (0,1)
S_VAL = -(C_VAL * math.log(C_VAL) + (1 - C_VAL) * math.log(1 - C_VAL))


def build_manifest(spec: dict[str, str]) -> dict:
    return {
        "schema": "schemas/manifest.schema.json",
        "casepack": {
            "id": f"{spec['domain']}_stub",
            "version": "0.1.0",
            "title": spec["title"],
            "description": spec["description"],
            "created_utc": "2026-05-10T00:00:00Z",
            "timezone": "America/Chicago",
            "authors": ["Clement Paulus"],
        },
        "refs": {
            "canon_anchors": {"path": spec["canon_path"]},
            "contract": {"id": spec["contract_id"], "path": spec["contract_path"]},
            "closures_registry": {"id": "UMCP.CLOSURES.DEFAULT.v1", "path": "closures/registry.yaml"},
        },
        "artifacts": {
            "raw_measurements": {"path": "raw_measurements.csv", "format": "csv"},
            "expected": {
                "psi_csv": {"path": "expected/psi.csv"},
                "invariants_json": {"path": "expected/invariants.json"},
            },
        },
        "run_intent": {
            "notes": (
                f"Stub demonstration unit for the {spec['domain']} domain. Three uniform "
                "channels at c=0.99 yield a Stable regime under canon thresholds. Pending "
                "replacement with domain-specific raw measurements drawn from the closure "
                "code in closures/" + spec["domain"] + "/."
            ),
        },
    }


def build_invariants(spec: dict[str, str]) -> dict:
    return {
        "schema": "schemas/invariants.schema.json",
        "format": "tier1_invariants",
        "contract_id": spec["contract_id"],
        "closure_registry_id": "UMCP.CLOSURES.DEFAULT.v1",
        "canon": {
            "pre_doi": "10.5281/zenodo.17756705",
            "post_doi": "10.5281/zenodo.18072852",
            "weld_id": "W-2025-12-31-PHYS-COHERENCE",
            "timezone": "America/Chicago",
        },
        "rows": [
            {
                "t": 0,
                "omega": round(1.0 - F_VAL, 12),
                "F": F_VAL,
                "S": S_VAL,
                "C": 0.0,
                "tau_R": "INF_REC",
                "kappa": KAPPA,
                "IC": IC_VAL,
                "regime": {"label": "Stable", "critical_overlay": False},
                "kernel_optional": {"IC_min": IC_VAL},
            }
        ],
        "notes": f"Stub demonstration row for the {spec['domain']} domain (uniform 3-channel Stable regime).",
    }


def build_readme(spec: dict[str, str]) -> str:
    return f"""# {spec["domain"]}_stub

**{spec["title"]}**

> ⚠️ **Stub demonstration unit.** This casepack provides minimal packaging
> coverage for the `{spec["domain"]}` domain so that every registered Tier-2
> domain has a casepack of record under `closures/full/`. The Tier-1 row is
> a uniform 3-channel Stable-regime demonstration; it does not represent
> domain-specific empirical data.

## Status

- **Casepack ID**: `{spec["domain"]}_stub`
- **Contract**: `{spec["contract_id"]}`
- **Canon anchors**: `{spec["canon_path"]}`
- **Status**: CONFORMANT (structural stub)
- **Replacement target**: domain-specific raw measurements from `closures/{spec["domain"]}/`

## What it validates

The stub validates the casepack packaging contract: schema, semantic rules,
Tier-1 identities (F + ω = 1, IC ≤ F, IC = exp(κ)), and CONFORMANT verdict
through `umcp validate`. It does **not** demonstrate the rich domain-specific
phenomena that live in the closure code — those will arrive when the stub
is welded into a full demonstration unit.

## How to validate

```bash
umcp validate casepacks/closures/full/{spec["domain"]}
```

## Where the real domain lives

Closure code, theorems, and tests for `{spec["domain"]}` live at:

- Closures: `closures/{spec["domain"]}/`
- Tests: `tests/test_*{spec["domain"].split("_")[0]}*` (and related)
- Full description: `casepacks/TAXONOMY.md` (Coverage Gap section)

## Lineage

- Created: 2026-05-10 (Phase 4 reorg follow-up)
- Reason: TAXONOMY.md Coverage Gap — `{spec["domain"]}` domain had closure
  code and tests but no packaged demonstration unit.
"""


def write_stub(spec: dict[str, str]) -> None:
    cp_dir = CASEPACKS / spec["domain"]
    expected = cp_dir / "expected"
    cp_dir.mkdir(parents=True, exist_ok=True)
    expected.mkdir(parents=True, exist_ok=True)

    # manifest.json
    (cp_dir / "manifest.json").write_text(json.dumps(build_manifest(spec), indent=2) + "\n")

    # raw_measurements.csv (single row, 3 channels at 9.9 → embed to 0.99 under [0, 10])
    (cp_dir / "raw_measurements.csv").write_text(
        "t,x1_si,x2_si,x3_si,units,notes\n"
        '0,9.9,9.9,9.9,arbitrary,"Stub: 3 uniform channels embed to c_k = 0.99 under bounds [0,10]."\n'
    )

    # expected/psi.csv
    (expected / "psi.csv").write_text(
        "t,c_1,c_2,c_3,oor_1,oor_2,oor_3,miss_1,miss_2,miss_3\n0,0.99,0.99,0.99,false,false,false,false,false,false\n"
    )

    # expected/invariants.json
    (expected / "invariants.json").write_text(json.dumps(build_invariants(spec), indent=2) + "\n")

    # README.md
    (cp_dir / "README.md").write_text(build_readme(spec))

    print(f"  ✓ {cp_dir.relative_to(REPO)}")


def main() -> None:
    print("Generating 6 casepack stubs for missing-coverage domains:")
    for spec in STUBS:
        write_stub(spec)
    print(f"\nDone. {len(STUBS)} stubs written under {CASEPACKS.relative_to(REPO)}/")


if __name__ == "__main__":
    main()

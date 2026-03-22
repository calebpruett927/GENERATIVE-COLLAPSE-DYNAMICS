"""
Site Data Extractor — Reads frozen validation artifacts into unified SiteData.

Reads:
    - ledger/return_log.csv       (append-only validation log)
    - canon/<domain>_anchors.yaml (domain anchor definitions)
    - closures/<domain>/          (closure module metadata)
    - casepacks/*/manifest.json   (casepack manifests referencing the domain)
    - contracts/*.yaml            (contract definitions)
    - outputs/                    (invariants, regimes, welds)

Emits:
    SiteData — a frozen dataclass containing everything a static site needs
    to render pages without touching the Python/C compute stack.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def _repo_root() -> Path:
    """Resolve repository root from this file's location."""
    # src/umcp/hcg/extractor.py → repo root is 3 levels up
    p = Path(__file__).resolve().parent.parent.parent.parent
    if (p / "pyproject.toml").exists():
        return p
    # Fallback: cwd
    return Path.cwd()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LedgerRow:
    """One append-only row from return_log.csv."""

    timestamp: str
    run_status: str
    F: float
    omega: float
    kappa: float
    IC: float
    C: float
    S: float
    tau_R: str  # may be "" or "INF_REC"
    delta_kappa: str
    chain_hash: str


@dataclass(frozen=True)
class KernelSnapshot:
    """Tier-1 invariants at a single point in time."""

    F: float
    omega: float
    S: float
    C: float
    kappa: float
    IC: float
    regime: str
    heterogeneity_gap: float  # F - IC


@dataclass
class DomainAnchors:
    """Parsed canon anchor data for one domain."""

    domain_id: str
    domain_name: str
    version: str
    hierarchy: str
    channels: list[Any] | dict[str, Any]
    anchors: dict[str, Any]
    raw: dict[str, Any]


@dataclass
class CasepackSummary:
    """Lightweight casepack manifest summary."""

    name: str
    path: str
    contract_ref: str
    status: str  # last known validation status
    description: str
    title: str = ""
    authors: list[str] = field(default_factory=list)
    version: str = ""
    artifacts: dict[str, Any] = field(default_factory=dict)
    run_intent: str = ""
    canon_ref: str = ""


@dataclass
class SiteData:
    """Everything a static site generator needs for one domain.

    By the time this object is constructed, all computation is frozen.
    The rendering layer reads this — it never invokes the kernel.
    """

    domain: str
    domain_display: str
    anchors: DomainAnchors | None
    ledger_rows: list[LedgerRow]
    latest_snapshot: KernelSnapshot | None
    casepacks: list[CasepackSummary]
    contract: dict[str, Any] | None
    closure_modules: list[str]
    theorem_count: int
    entity_count: int
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


def _safe_float(val: str, default: float = 0.0) -> float:
    """Parse float, returning *default* for empty / non-numeric."""
    if not val or val.strip() == "":
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def read_ledger(root: Path | None = None) -> list[LedgerRow]:
    """Parse the append-only ledger into typed rows."""
    root = root or _repo_root()
    ledger_path = root / "ledger" / "return_log.csv"
    if not ledger_path.exists():
        return []

    rows: list[LedgerRow] = []
    with open(ledger_path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header is None:
            return []
        for line in reader:
            if len(line) < 8:
                continue
            rows.append(
                LedgerRow(
                    timestamp=line[0],
                    run_status=line[1],
                    F=_safe_float(line[2]),
                    omega=_safe_float(line[3]),
                    kappa=_safe_float(line[4]),
                    IC=_safe_float(line[5]),
                    C=_safe_float(line[6]),
                    S=_safe_float(line[7]),
                    tau_R=line[8] if len(line) > 8 else "",
                    delta_kappa=line[9] if len(line) > 9 else "",
                    chain_hash=line[10] if len(line) > 10 else "",
                )
            )
    return rows


def read_anchors(domain: str, root: Path | None = None) -> DomainAnchors | None:
    """Load canon/<domain>_anchors.yaml, returning None if absent."""
    if yaml is None:
        return None
    root = root or _repo_root()

    # Map domain names to anchor file prefixes
    prefix_map: dict[str, str] = {
        "gcd": "gcd",
        "rcft": "rcft",
        "kinematics": "kin",
        "weyl": "weyl",
        "security": "sec",
        "astronomy": "astro",
        "nuclear_physics": "nuc",
        "quantum_mechanics": "qm",
        "finance": "finance",
        "atomic_physics": "atom",
        "materials_science": "matl",
        "everyday_physics": "evday",
        "evolution": "evo",
        "dynamic_semiotics": "semiotics",
        "consciousness_coherence": "cons",
        "continuity_theory": "ct",
        "awareness_cognition": "awc",
        "standard_model": "sm",
        "clinical_neuroscience": "clin",
        "spacetime_memory": "st",
    }

    prefix = prefix_map.get(domain, domain)
    anchor_path = root / "canon" / f"{prefix}_anchors.yaml"
    if not anchor_path.exists():
        return None

    with open(anchor_path, encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh)

    raw: dict[str, Any] = loaded if isinstance(loaded, dict) else {}

    return DomainAnchors(
        domain_id=raw.get("id", f"UMCP.CANON.{domain.upper()}.v1"),
        domain_name=raw.get("domain", domain.replace("_", " ").title()),
        version=raw.get("version", "1.0.0"),
        hierarchy=raw.get("hierarchy", f"GCD > {domain.upper()} (Tier-2)"),
        channels=raw.get("channels", []),
        anchors=raw.get("anchors", {}),
        raw=raw,
    )


def read_casepacks(domain: str, root: Path | None = None) -> list[CasepackSummary]:
    """Find casepacks whose manifest references this domain."""
    root = root or _repo_root()
    casepacks_dir = root / "casepacks"
    if not casepacks_dir.exists():
        return []

    # Explicit mapping for domains whose casepack dir names don't contain
    # the full domain slug. Maps domain → set of known casepack dir names.
    _DOMAIN_CASEPACK_MAP: dict[str, set[str]] = {
        "standard_model": {"confinement_T3"},
        "nuclear_physics": {"nuclear_chain"},
        "clinical_neuroscience": {"clinical_neuro_states"},
        "consciousness_coherence": {"ricp_v2_consciousness", "consciousness_kappa_72"},
        "dynamic_semiotics": {"semiotics_kernel"},
    }

    # Contract prefix → domain mapping (from anchor prefix map)
    _CONTRACT_PREFIX_DOMAIN: dict[str, str] = {
        "SM": "standard_model",
        "NUC": "nuclear_physics",
        "CLIN": "clinical_neuroscience",
        "CONS": "consciousness_coherence",
        "SEM": "dynamic_semiotics",
        "GCD": "gcd",
        "RCFT": "rcft",
        "KIN": "kinematics",
        "WEYL": "weyl",
        "SECURITY": "security",
        "ASTRO": "astronomy",
        "QM": "quantum_mechanics",
        "FINANCE": "finance",
        "ATOM": "atomic_physics",
        "MATL": "materials_science",
        "EVDAY": "everyday_physics",
        "EVO": "evolution",
        "CT": "continuity_theory",
        "AWC": "awareness_cognition",
        "ST": "spacetime_memory",
    }

    explicit_names = _DOMAIN_CASEPACK_MAP.get(domain, set())

    results: list[CasepackSummary] = []
    for child in sorted(casepacks_dir.iterdir()):
        manifest = child / "manifest.json"
        if not manifest.exists():
            continue
        try:
            with open(manifest, encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue

        # Parse nested manifest structure (casepack.*, refs.*)
        cp_block = data.get("casepack", {})
        refs_block = data.get("refs", {})
        contract_block = refs_block.get("contract", {})
        canon_block = refs_block.get("canon_anchors", {})

        # Extract fields — fall back to top-level for flat manifests
        name = cp_block.get("id", data.get("name", child.name))
        title = cp_block.get("title", data.get("title", ""))
        desc = cp_block.get("description", data.get("description", ""))
        version = cp_block.get("version", data.get("version", ""))
        authors = cp_block.get("authors", data.get("authors", []))
        contract_id = contract_block.get("id", "") if isinstance(contract_block, dict) else str(contract_block)
        contract_path = contract_block.get("path", "") if isinstance(contract_block, dict) else ""
        canon_path = (
            canon_block.get("path", "") if isinstance(canon_block, dict) else str(canon_block) if canon_block else ""
        )
        artifacts = data.get("artifacts", {})
        run_intent_block = data.get("run_intent", {})
        run_intent = run_intent_block.get("notes", "") if isinstance(run_intent_block, dict) else str(run_intent_block)

        # Build contract_ref from best available data
        contract_ref = contract_id or contract_path or data.get("contract", "")

        # Domain matching — check multiple signals
        domain_lower = domain.lower()
        domain_upper = domain.upper().replace("_", "")

        # Check contract prefix → domain
        contract_domain = ""
        if contract_id:
            prefix = contract_id.split(".")[0] if "." in contract_id else ""
            contract_domain = _CONTRACT_PREFIX_DOMAIN.get(prefix, "")

        matched = (
            child.name in explicit_names
            or domain_lower in child.name.lower()
            or contract_domain == domain
            or domain_lower in desc.lower()
            or domain_lower in title.lower()
            or domain_upper in contract_ref.upper().replace(".", "").replace("_", "")
        )

        if matched:
            results.append(
                CasepackSummary(
                    name=name,
                    path=str(child.relative_to(root)),
                    contract_ref=contract_ref,
                    status=data.get("status", "validated"),
                    description=desc,
                    title=title,
                    authors=authors if isinstance(authors, list) else [],
                    version=version,
                    artifacts=artifacts,
                    run_intent=run_intent,
                    canon_ref=canon_path,
                )
            )

    # If domain is "gcd" and no results, include all casepacks
    if domain == "gcd" and not results:
        for child in sorted(casepacks_dir.iterdir()):
            manifest = child / "manifest.json"
            if not manifest.exists():
                continue
            try:
                with open(manifest, encoding="utf-8") as fh:
                    data = json.load(fh)
            except (json.JSONDecodeError, OSError):
                continue
            cp_block = data.get("casepack", {})
            refs_block = data.get("refs", {})
            contract_block = refs_block.get("contract", {})
            canon_block = refs_block.get("canon_anchors", {})
            run_intent_block = data.get("run_intent", {})
            results.append(
                CasepackSummary(
                    name=cp_block.get("id", data.get("name", child.name)),
                    path=str(child.relative_to(root)),
                    contract_ref=contract_block.get("id", "")
                    if isinstance(contract_block, dict)
                    else str(contract_block),
                    status=data.get("status", "validated"),
                    description=cp_block.get("description", data.get("description", "")),
                    title=cp_block.get("title", data.get("title", "")),
                    authors=cp_block.get("authors", []),
                    version=cp_block.get("version", ""),
                    artifacts=data.get("artifacts", {}),
                    run_intent=run_intent_block.get("notes", "") if isinstance(run_intent_block, dict) else "",
                    canon_ref=canon_block.get("path", "") if isinstance(canon_block, dict) else "",
                )
            )

    return results


def read_contract(domain: str, root: Path | None = None) -> dict[str, Any] | None:
    """Load the first contract matching the domain."""
    if yaml is None:
        return None
    root = root or _repo_root()
    contracts_dir = root / "contracts"
    if not contracts_dir.exists():
        return None

    domain_upper = domain.upper().replace("_", "")
    for yf in sorted(contracts_dir.glob("*.yaml")):
        if domain_upper in yf.stem.upper() or domain.lower() in yf.stem.lower():
            try:
                with open(yf, encoding="utf-8") as fh:
                    loaded = yaml.safe_load(fh)
                    return loaded if isinstance(loaded, dict) else None
            except Exception:
                continue
    return None


def _count_closure_files(domain: str, root: Path | None = None) -> tuple[list[str], int, int]:
    """Count closure modules, theorems, and entities for a domain."""
    root = root or _repo_root()
    closure_dir = root / "closures" / domain
    if not closure_dir.exists():
        return [], 0, 0

    modules: list[str] = []
    theorem_count = 0
    entity_count = 0

    for py_file in sorted(closure_dir.glob("*.py")):
        if py_file.name.startswith("__"):
            continue
        modules.append(py_file.stem)

        # Quick heuristic scan for theorem/entity counts
        try:
            text = py_file.read_text(encoding="utf-8")
            # Count theorem definitions (T-XX- patterns or THEOREM entries)
            theorem_count += text.count('"T-')
            theorem_count += text.count("'T-")
            # Count entity entries (dict entries in ENTITIES-like structures)
            entity_count += text.count('"name":')
            entity_count += text.count("'name':")
        except OSError:
            continue

    return modules, theorem_count, entity_count


@dataclass
class TheoremInfo:
    """Extracted theorem metadata from a closure file."""

    theorem_id: str  # e.g. "T1", "T-MS-1", "TCN1"
    name: str  # e.g. "Spin-Statistics Kernel Theorem"
    source_module: str  # e.g. "particle_physics_formalism"
    function_name: str  # e.g. "theorem_T1_spin_statistics"


@dataclass
class EntityInfo:
    """Extracted entity metadata from a closure file."""

    name: str
    source_module: str


def scan_closure_theorems(domain: str, root: Path | None = None) -> list[TheoremInfo]:
    """Scan closure files for theorem function definitions."""
    import re

    root = root or _repo_root()
    closure_dir = root / "closures" / domain
    if not closure_dir.exists():
        return []

    results: list[TheoremInfo] = []
    seen: set[str] = set()

    for py_file in sorted(closure_dir.glob("*.py")):
        if py_file.name.startswith("__"):
            continue
        try:
            text = py_file.read_text(encoding="utf-8")
        except OSError:
            continue

        module = py_file.stem

        # Pattern 1: "# THEOREM T-XX-N: NAME" heading
        for m in re.finditer(r"^# THEOREM\s+(T[\w-]+\d+):\s*(.+)$", text, re.MULTILINE):
            tid = m.group(1).strip()
            tname = m.group(2).strip()
            if tid not in seen:
                seen.add(tid)
                # Find the function below this heading
                fn_match = re.search(r"^def (theorem_\w+)\(", text[m.end() :], re.MULTILINE)
                fn_name = fn_match.group(1) if fn_match else ""
                results.append(
                    TheoremInfo(
                        theorem_id=tid,
                        name=tname,
                        source_module=module,
                        function_name=fn_name,
                    )
                )

        # Pattern 2: def theorem_Txx_name() with docstring "Txx: Name."
        for m in re.finditer(
            r'^def (theorem_(\w+))\([^)]*\)[^:]*:\s*\n\s+"""([^"]+)',
            text,
            re.MULTILINE,
        ):
            fn_name = m.group(1)
            tid_raw = m.group(2)
            docline = m.group(3).strip().rstrip(".")
            # Extract theorem ID from docline (e.g., "T1: Spin-Statistics...")
            doc_match = re.match(r"(T[\w-]*\d+):\s*(.+)", docline)
            if doc_match:
                tid = doc_match.group(1)
                tname = doc_match.group(2)
            else:
                tid = tid_raw.replace("_", "-").upper()
                tname = docline
            if tid not in seen:
                seen.add(tid)
                results.append(
                    TheoremInfo(
                        theorem_id=tid,
                        name=tname,
                        source_module=module,
                        function_name=fn_name,
                    )
                )

    return results


def scan_closure_entities(domain: str, root: Path | None = None) -> list[EntityInfo]:
    """Scan closure files for entity-like data structures."""
    import re

    root = root or _repo_root()
    closure_dir = root / "closures" / domain
    if not closure_dir.exists():
        return []

    results: list[EntityInfo] = []
    seen: set[str] = set()

    for py_file in sorted(closure_dir.glob("*.py")):
        if py_file.name.startswith("__"):
            continue
        try:
            text = py_file.read_text(encoding="utf-8")
        except OSError:
            continue

        module = py_file.stem

        # Extract "name": "value" patterns from catalog-like structures
        for m in re.finditer(r"""["']name["']\s*:\s*["']([^"']+)["']""", text):
            name = m.group(1)
            if name not in seen and len(name) > 1:
                seen.add(name)
                results.append(EntityInfo(name=name, source_module=module))

    return results


def _latest_kernel_snapshot(rows: list[LedgerRow]) -> KernelSnapshot | None:
    """Derive latest KernelSnapshot from the most recent CONFORMANT ledger row."""
    conformant = [r for r in rows if r.run_status == "CONFORMANT"]
    if not conformant:
        return None
    last = conformant[-1]
    gap = last.F - last.IC

    # Derive regime from gates
    if last.omega >= 0.30:
        regime = "COLLAPSE"
    elif last.omega < 0.038 and last.F > 0.90 and last.S < 0.15 and last.C < 0.14:
        regime = "STABLE"
    else:
        regime = "WATCH"

    return KernelSnapshot(
        F=last.F,
        omega=last.omega,
        S=last.S,
        C=last.C,
        kappa=last.kappa,
        IC=last.IC,
        regime=regime,
        heterogeneity_gap=gap,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Human-readable display names
DOMAIN_DISPLAY: dict[str, str] = {
    "gcd": "Generative Collapse Dynamics",
    "rcft": "Recursive Collapse Field Theory",
    "kinematics": "Kinematics",
    "weyl": "WEYL Cosmology",
    "security": "Security & Audit",
    "astronomy": "Astronomy",
    "nuclear_physics": "Nuclear Physics",
    "quantum_mechanics": "Quantum Mechanics",
    "finance": "Finance",
    "atomic_physics": "Atomic Physics",
    "materials_science": "Materials Science",
    "everyday_physics": "Everyday Physics",
    "evolution": "Evolution & Neuroscience",
    "dynamic_semiotics": "Dynamic Semiotics",
    "consciousness_coherence": "Consciousness Coherence",
    "continuity_theory": "Continuity Theory",
    "awareness_cognition": "Awareness & Cognition",
    "standard_model": "Standard Model",
    "clinical_neuroscience": "Clinical Neuroscience",
    "spacetime_memory": "Spacetime Memory",
}


def extract_domain_data(
    domain: str,
    root: Path | None = None,
) -> SiteData:
    """Extract all data needed to build a static site for *domain*.

    This is the single entry-point for the rendering layer.  It reads
    frozen artifacts only — no kernel computation happens here.
    """
    root = root or _repo_root()

    ledger_rows = read_ledger(root)
    anchors = read_anchors(domain, root)
    casepacks = read_casepacks(domain, root)
    contract = read_contract(domain, root)
    modules, theorems, entities = _count_closure_files(domain, root)
    snapshot = _latest_kernel_snapshot(ledger_rows)

    return SiteData(
        domain=domain,
        domain_display=DOMAIN_DISPLAY.get(domain, domain.replace("_", " ").title()),
        anchors=anchors,
        ledger_rows=ledger_rows,
        latest_snapshot=snapshot,
        casepacks=casepacks,
        contract=contract,
        closure_modules=modules,
        theorem_count=theorems,
        entity_count=entities,
        metadata={
            "repo_root": str(root),
            "domain_count": len(DOMAIN_DISPLAY),
            "total_ledger_rows": len(ledger_rows),
            "conformant_count": sum(1 for r in ledger_rows if r.run_status == "CONFORMANT"),
        },
    )

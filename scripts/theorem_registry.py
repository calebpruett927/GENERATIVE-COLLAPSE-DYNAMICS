"""Theorem Registry — Execute all theorems and build a master registry.

Imports every theorem function across all 26 theorem-bearing files in
closures/, executes each one, captures the TheoremResult, and serializes
a comprehensive registry with:

    - tag (unique identifier)
    - domain
    - source file
    - name (full human-readable)
    - statement (mathematical claim)
    - n_tests, n_passed, n_failed, pass_rate
    - verdict (PROVEN / FALSIFIED)
    - details (domain-specific diagnostics)
    - identity_checks (which Tier-1 identities are verified)
    - archetype (auto-classified theorem pattern)

Usage:
    python scripts/theorem_registry.py              # Full registry + summary
    python scripts/theorem_registry.py --json       # JSON output to stdout
    python scripts/theorem_registry.py --csv        # CSV summary to stdout

Collapsus generativus est; solum quod redit, reale est.
"""

from __future__ import annotations

import csv
import importlib
import inspect
import io
import json
import re
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ═══════════════════════════════════════════════════════════════════
# PATH SETUP
# ═══════════════════════════════════════════════════════════════════
_WORKSPACE = Path(__file__).resolve().parents[1]
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))
if str(_WORKSPACE / "src") not in sys.path:
    sys.path.insert(0, str(_WORKSPACE / "src"))

# ═══════════════════════════════════════════════════════════════════
# REGISTRY ENTRY
# ═══════════════════════════════════════════════════════════════════


@dataclass
class TheoremEntry:
    """A single theorem in the master registry."""

    tag: str
    domain: str
    source_file: str
    function_name: str
    name: str
    statement: str
    n_tests: int
    n_passed: int
    n_failed: int
    pass_rate: float
    verdict: str
    archetype: str
    identity_checks: list[str]
    details: dict[str, Any] = field(default_factory=dict)
    execution_time_ms: float = 0.0
    error: str | None = None


# ═══════════════════════════════════════════════════════════════════
# THEOREM SOURCES — all 26 theorem-bearing files
# ═══════════════════════════════════════════════════════════════════

# (module_path, domain_name)
_THEOREM_SOURCES: list[tuple[str, str]] = [
    # Dedicated theorem files (16)
    ("closures.gcd.kernel_structural_theorems", "gcd"),
    ("closures.gcd.universal_regime_calibration", "gcd"),
    ("closures.standard_model.particle_physics_formalism", "standard_model"),
    ("closures.standard_model.neutrino_oscillation", "standard_model"),
    ("closures.astronomy.astronomy_theorems", "astronomy"),
    ("closures.finance.finance_theorems", "finance"),
    ("closures.consciousness_coherence.consciousness_theorems", "consciousness_coherence"),
    ("closures.nuclear_physics.nuclear_theorems", "nuclear_physics"),
    ("closures.everyday_physics.everyday_physics_theorems", "everyday_physics"),
    ("closures.everyday_physics.epistemic_coherence", "everyday_physics"),
    ("closures.materials_science.materials_theorems", "materials_science"),
    ("closures.evolution.evolution_theorems", "evolution"),
    ("closures.rcft.rcft_theorems", "rcft"),
    ("closures.weyl.weyl_theorems", "weyl"),
    ("closures.clinical_neuroscience.neurocognitive_theorems", "clinical_neuroscience"),
    ("closures.kinematics.kinematics_theorems", "kinematics"),
    ("closures.awareness_cognition.awareness_theorems", "awareness_cognition"),
    ("closures.spacetime_memory.spacetime_theorems", "spacetime_memory"),
    ("closures.dynamic_semiotics.semiotic_theorems", "dynamic_semiotics"),
    ("closures.continuity_theory.continuity_theorems", "continuity_theory"),
    # Other theorem-bearing files (6)
    ("closures.quantum_mechanics.ters_near_field", "quantum_mechanics"),
    ("closures.quantum_mechanics.atom_dot_mi_transition", "quantum_mechanics"),
    ("closures.quantum_mechanics.double_slit_interference", "quantum_mechanics"),
    ("closures.quantum_mechanics.muon_laser_decay", "quantum_mechanics"),
    ("closures.atomic_physics.recursive_instantiation", "atomic_physics"),
    ("closures.unified_minimal_structure", "unified_minimal_structure"),
    ("closures.standard_model.sm_extended_theorems", "standard_model"),
]


# ═══════════════════════════════════════════════════════════════════
# ARCHETYPE CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════

_IDENTITY_KEYWORDS = [
    "F+ω=1",
    "F + ω = 1",
    "F+omega=1",
    "duality",
    "IC≤F",
    "IC ≤ F",
    "IC<=F",
    "integrity bound",
    "IC=exp(κ)",
    "IC = exp(κ)",
    "IC=exp(kappa)",
    "log-integrity",
    "tier-1",
    "Tier-1",
    "tier 1",
]

_ARCHETYPE_RULES: list[tuple[str, list[str]]] = [
    ("identity_verification", ["F+ω=1", "IC≤F", "IC=exp", "Tier-1", "tier-1", "duality", "identities"]),
    ("geometric_slaughter", ["geometric slaughter", "dead channel", "IC kill", "one dead", "IC drop"]),
    ("ordering_hierarchy", ["ordering", "hierarchy", "monoton", "stratif", "ranking", "gradient"]),
    ("regime_classification", ["regime", "Stable", "Watch", "Collapse", "regime distribution"]),
    ("phase_transition", ["confinement", "phase", "transition", "boundary", "cliff", "collapse detection"]),
    ("correlation_structure", ["correlat", "Spearman", "Pearson", "anticorrelat", "coupling"]),
    ("cross_scale", ["cross-scale", "universal", "cross-domain", "scale invariance"]),
    ("composition", ["composition", "compos", "cascade", "multiplicat"]),
    ("extremal_entity", ["unique", "sole", "singular", "peak", "floor", "lowest", "highest"]),
    ("coverage_completeness", ["coverage", "count", "span", "all.*entities", "total"]),
]


def _classify_archetype(name: str, statement: str) -> str:
    """Classify a theorem into an archetype based on name + statement."""
    text = f"{name} {statement}".lower()
    for archetype, keywords in _ARCHETYPE_RULES:
        for kw in keywords:
            if kw.lower() in text:
                return archetype
    return "domain_specific"


def _extract_identity_checks(name: str, statement: str, details: dict[str, Any]) -> list[str]:
    """Determine which Tier-1 identities this theorem verifies."""
    text = f"{name} {statement} {json.dumps(details, default=str)}".lower()
    checks = []
    if any(k.lower() in text for k in ["f+ω=1", "f + ω = 1", "f+omega=1", "duality", "f + omega"]):
        checks.append("F+ω=1")
    if any(k.lower() in text for k in ["ic≤f", "ic ≤ f", "ic<=f", "integrity bound", "ic_le_f"]):
        checks.append("IC≤F")
    if any(k.lower() in text for k in ["ic=exp(κ)", "ic=exp(kappa)", "ic = exp", "log-integrity", "exp_kappa"]):
        checks.append("IC=exp(κ)")
    # Check details keys for identity evidence
    detail_str = str(details).lower()
    if ("duality_residual" in detail_str or "f_plus_omega" in detail_str) and "F+ω=1" not in checks:
        checks.append("F+ω=1")
    if (
        "ic_le_f" in detail_str or "ic_bound" in detail_str or "integrity_bound" in detail_str
    ) and "IC≤F" not in checks:
        checks.append("IC≤F")
    if ("exp_kappa" in detail_str or "ic_vs_exp" in detail_str) and "IC=exp(κ)" not in checks:
        checks.append("IC=exp(κ)")
    return checks


def _extract_tag(name: str) -> str:
    """Extract the tag prefix from a theorem name like 'T-KS-1: ...'."""
    # Try patterns: "T-XX-N:", "TN:", "T-XX-N ", "TXX1"
    m = re.match(r"^(T[\w-]+\d+)\b", name.replace(": ", " ").replace(":", " "))
    if m:
        return m.group(1).rstrip("-").rstrip("_")
    # Direct tag patterns
    m = re.match(r"^([A-Z]{2,}\d+)\b", name)
    if m:
        return m.group(1)
    # Fallback: first word/token
    token = name.split(":")[0].strip() if ":" in name else name.split()[0] if name else "UNKNOWN"
    return token


# ═══════════════════════════════════════════════════════════════════
# DISCOVERY AND EXECUTION
# ═══════════════════════════════════════════════════════════════════


def _is_theorem_function(name: str, obj: object) -> bool:
    """Check if a module-level function is a theorem function."""
    if not callable(obj) or not inspect.isfunction(obj):
        return False
    if name.startswith("_"):
        return False
    # Must start with theorem_ or prove_
    if not (name.startswith("theorem_") or name.startswith("prove_")):
        return False
    # Check return annotation if available
    hints = getattr(obj, "__annotations__", {})
    ret = hints.get("return", None)
    if ret is not None:
        ret_name = getattr(ret, "__name__", str(ret))
        if "TheoremResult" in ret_name:
            return True
    # If no annotation, accept based on naming convention
    return True


def discover_and_execute(
    sources: list[tuple[str, str]] | None = None,
    *,
    verbose: bool = False,
) -> list[TheoremEntry]:
    """Import all theorem modules, discover theorem functions, execute them."""
    if sources is None:
        sources = _THEOREM_SOURCES

    entries: list[TheoremEntry] = []
    total_time = 0.0

    for module_path, domain in sources:
        if verbose:
            print(f"\n{'─' * 60}")
            print(f"Loading {module_path} ({domain})")

        try:
            mod = importlib.import_module(module_path)
        except Exception as exc:
            if verbose:
                print(f"  ERROR importing: {exc}")
            entries.append(
                TheoremEntry(
                    tag="IMPORT_ERROR",
                    domain=domain,
                    source_file=module_path.replace(".", "/") + ".py",
                    function_name="<import>",
                    name=f"Import Error: {module_path}",
                    statement="",
                    n_tests=0,
                    n_passed=0,
                    n_failed=0,
                    pass_rate=0.0,
                    verdict="ERROR",
                    archetype="error",
                    identity_checks=[],
                    error=str(exc),
                )
            )
            continue

        # Discover theorem functions
        funcs = [
            (fname, fobj)
            for fname, fobj in inspect.getmembers(mod)
            if _is_theorem_function(fname, fobj) and fobj.__module__ == mod.__name__
        ]

        if verbose:
            print(f"  Found {len(funcs)} theorem functions")

        for fname, fobj in sorted(funcs, key=lambda x: x[0]):
            t0 = time.perf_counter()
            try:
                result = fobj()
                dt = (time.perf_counter() - t0) * 1000.0
                total_time += dt

                # Extract from TheoremResult (consistent interface)
                r_name = getattr(result, "name", fname)
                r_statement = getattr(result, "statement", "")
                r_ntests = getattr(result, "n_tests", 0)
                r_npassed = getattr(result, "n_passed", 0)
                r_nfailed = getattr(result, "n_failed", 0)
                r_verdict = getattr(result, "verdict", "UNKNOWN")
                r_details = getattr(result, "details", {})
                r_pass_rate = r_npassed / r_ntests if r_ntests > 0 else 0.0

                tag = _extract_tag(r_name)
                archetype = _classify_archetype(r_name, r_statement)
                id_checks = _extract_identity_checks(r_name, r_statement, r_details)

                entry = TheoremEntry(
                    tag=tag,
                    domain=domain,
                    source_file=module_path.replace(".", "/") + ".py",
                    function_name=fname,
                    name=r_name,
                    statement=r_statement,
                    n_tests=r_ntests,
                    n_passed=r_npassed,
                    n_failed=r_nfailed,
                    pass_rate=r_pass_rate,
                    verdict=r_verdict,
                    archetype=archetype,
                    identity_checks=id_checks,
                    details=r_details,
                    execution_time_ms=dt,
                )
                entries.append(entry)

                if verbose:
                    status = "✓" if r_verdict == "PROVEN" else "✗"
                    print(f"  {status} {tag:<12s} {r_verdict:<10s} {r_npassed}/{r_ntests} ({dt:.0f}ms) [{archetype}]")

            except Exception as exc:
                dt = (time.perf_counter() - t0) * 1000.0
                total_time += dt
                if verbose:
                    print(f"  ✗ {fname}: ERROR — {exc}")
                entries.append(
                    TheoremEntry(
                        tag=f"ERR-{fname}",
                        domain=domain,
                        source_file=module_path.replace(".", "/") + ".py",
                        function_name=fname,
                        name=fname,
                        statement="",
                        n_tests=0,
                        n_passed=0,
                        n_failed=0,
                        pass_rate=0.0,
                        verdict="ERROR",
                        archetype="error",
                        identity_checks=[],
                        error=traceback.format_exc(),
                        execution_time_ms=dt,
                    )
                )

    if verbose:
        n_proven = sum(1 for e in entries if e.verdict == "PROVEN")
        n_total = len(entries)
        print(f"\n{'═' * 60}")
        print(f"REGISTRY: {n_total} theorems | {n_proven} PROVEN | {total_time:.0f}ms total")

    return entries


# ═══════════════════════════════════════════════════════════════════
# SERIALIZATION
# ═══════════════════════════════════════════════════════════════════


def _entry_to_dict(e: TheoremEntry) -> dict[str, Any]:
    """Convert entry to a JSON-safe dict (details serialized)."""
    d = {
        "tag": e.tag,
        "domain": e.domain,
        "source_file": e.source_file,
        "function_name": e.function_name,
        "name": e.name,
        "statement": e.statement,
        "n_tests": e.n_tests,
        "n_passed": e.n_passed,
        "n_failed": e.n_failed,
        "pass_rate": round(e.pass_rate, 4),
        "verdict": e.verdict,
        "archetype": e.archetype,
        "identity_checks": e.identity_checks,
        "execution_time_ms": round(e.execution_time_ms, 1),
    }
    if e.error:
        d["error"] = e.error
    # Include select detail keys (skip large arrays)
    if e.details:
        summary_details: dict[str, Any] = {}
        for k, v in e.details.items():
            if (
                isinstance(v, (int, float, str, bool))
                or (isinstance(v, dict) and len(str(v)) < 500)
                or (isinstance(v, (list, tuple)) and len(v) <= 20)
            ):
                summary_details[k] = v
        d["details_summary"] = summary_details
    return d


def to_json(entries: list[TheoremEntry]) -> str:
    """Serialize registry to JSON."""
    data = [_entry_to_dict(e) for e in entries]

    def _default(o: Any) -> Any:
        if hasattr(o, "__float__"):
            v = float(o)
            if v == v:  # handle NaN
                return v
            return None
        return str(o)

    return json.dumps(data, indent=2, default=_default, ensure_ascii=False)


def to_csv(entries: list[TheoremEntry]) -> str:
    """Serialize registry to CSV summary."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "tag",
            "domain",
            "name",
            "verdict",
            "n_tests",
            "n_passed",
            "pass_rate",
            "archetype",
            "identity_checks",
            "time_ms",
        ]
    )
    for e in entries:
        writer.writerow(
            [
                e.tag,
                e.domain,
                e.name,
                e.verdict,
                e.n_tests,
                e.n_passed,
                f"{e.pass_rate:.4f}",
                e.archetype,
                "|".join(e.identity_checks),
                f"{e.execution_time_ms:.0f}",
            ]
        )
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════


def main() -> None:
    """Run the full registry and print results."""
    import argparse

    parser = argparse.ArgumentParser(description="Theorem Registry")
    parser.add_argument("--json", action="store_true", help="Output JSON to stdout")
    parser.add_argument("--csv", action="store_true", help="Output CSV to stdout")
    parser.add_argument("--save", type=str, default=None, help="Save JSON registry to file")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    verbose = args.verbose or not (args.json or args.csv)
    entries = discover_and_execute(verbose=verbose)

    if args.json:
        print(to_json(entries))
    elif args.csv:
        print(to_csv(entries))
    else:
        # Summary
        n = len(entries)
        proven = [e for e in entries if e.verdict == "PROVEN"]
        falsified = [e for e in entries if e.verdict == "FALSIFIED"]
        errors = [e for e in entries if e.verdict == "ERROR"]

        print(f"\n{'═' * 70}")
        print(f"THEOREM REGISTRY — {n} theorems discovered")
        print(f"{'═' * 70}")
        print(f"  PROVEN:    {len(proven)}")
        print(f"  FALSIFIED: {len(falsified)}")
        print(f"  ERRORS:    {len(errors)}")
        print(f"  Total tests: {sum(e.n_tests for e in entries)}")
        print(f"  Total pass:  {sum(e.n_passed for e in entries)}")

        # Per-domain
        domains = sorted({e.domain for e in entries})
        print(f"\n{'─' * 70}")
        print(f"{'Domain':<30s} {'Thms':>5s} {'Proven':>7s} {'Tests':>6s} {'Pass':>6s}")
        print(f"{'─' * 70}")
        for d in domains:
            de = [e for e in entries if e.domain == d]
            dp = sum(1 for e in de if e.verdict == "PROVEN")
            dt = sum(e.n_tests for e in de)
            dpassed = sum(e.n_passed for e in de)
            print(f"  {d:<28s} {len(de):>5d} {dp:>7d} {dt:>6d} {dpassed:>6d}")

        # Archetype distribution
        archetypes = sorted({e.archetype for e in entries})
        print(f"\n{'─' * 70}")
        print("Archetype Distribution:")
        for a in archetypes:
            count = sum(1 for e in entries if e.archetype == a)
            pct = count / n * 100
            print(f"  {a:<30s} {count:>4d} ({pct:>5.1f}%)")

        # Identity verification coverage
        id_checks = ["F+ω=1", "IC≤F", "IC=exp(κ)"]
        print(f"\n{'─' * 70}")
        print("Tier-1 Identity Verification Coverage:")
        for ic in id_checks:
            count = sum(1 for e in entries if ic in e.identity_checks)
            pct = count / n * 100
            print(f"  {ic:<15s} verified by {count:>4d} theorems ({pct:>5.1f}%)")

        if falsified:
            print(f"\n{'─' * 70}")
            print("FALSIFIED theorems:")
            for e in falsified:
                print(f"  {e.tag:<15s} {e.domain:<25s} {e.n_passed}/{e.n_tests}")

        if errors:
            print(f"\n{'─' * 70}")
            print("ERRORS:")
            for e in errors:
                print(f"  {e.tag:<15s} {e.domain:<25s} {(e.error or '')[:80]}")

    if args.save:
        p = Path(args.save)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(to_json(entries), encoding="utf-8")
        print(f"\nSaved to {p}")


if __name__ == "__main__":
    main()

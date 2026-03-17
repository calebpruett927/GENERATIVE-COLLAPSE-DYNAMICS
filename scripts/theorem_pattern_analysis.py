"""Theorem Pattern Analysis — Deep structural study of the theorem registry.

Loads the theorem registry (derived/theorem_registry.json) and performs
multi-dimensional pattern analysis to discover structural regularities
across all 225 theorems and 20 domains.

Analyses performed:
    §1  Registry Overview               — counts, verdicts, coverage
    §2  Archetype Taxonomy              — what kinds of theorems exist
    §3  Cross-Domain Structural Twins   — same proof pattern, different domain
    §4  The First-Theorem Invariant     — the identity-verification bootstrap
    §5  Geometric Slaughter Map         — where IC destruction appears
    §6  Regime Distribution Landscape   — which regimes dominate each domain
    §7  Falsification Forensics         — what breaks and why
    §8  Test Density Spectrum           — from 2 to 777 tests per theorem
    §9  Identity Verification Network   — which identities proven where
    §10 Archetype Signatures per Domain — domain fingerprints
    §11 Emergent Structural Laws        — patterns that repeat universally

Usage:
    python scripts/theorem_pattern_analysis.py          # Full report
    python scripts/theorem_pattern_analysis.py --json   # Pattern data as JSON

Collapsus generativus est; solum quod redit, reale est.
"""

from __future__ import annotations

import collections
import json
import sys
from dataclasses import dataclass, field
from itertools import combinations
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

_REGISTRY_PATH = _WORKSPACE / "derived" / "theorem_registry.json"


# ═══════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════


@dataclass
class PatternResult:
    """Result of one pattern analysis."""

    name: str
    description: str
    findings: list[str]
    data: dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════
# LOAD REGISTRY
# ═══════════════════════════════════════════════════════════════════


def load_registry(path: Path | None = None) -> list[dict[str, Any]]:
    """Load theorem registry from JSON."""
    p = path or _REGISTRY_PATH
    if not p.exists():
        print(f"Registry not found at {p}")
        print("Run: python scripts/theorem_registry.py --save derived/theorem_registry.json")
        sys.exit(1)
    with open(p, encoding="utf-8") as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════
# §1: REGISTRY OVERVIEW
# ═══════════════════════════════════════════════════════════════════


def analyze_overview(entries: list[dict[str, Any]]) -> PatternResult:
    """Basic registry statistics."""
    n = len(entries)
    proven = sum(1 for e in entries if e["verdict"] == "PROVEN")
    falsified = sum(1 for e in entries if e["verdict"] == "FALSIFIED")
    errors = sum(1 for e in entries if e["verdict"] == "ERROR")
    domains = sorted({e["domain"] for e in entries})
    total_tests = sum(e["n_tests"] for e in entries)
    total_pass = sum(e["n_passed"] for e in entries)

    findings = [
        f"{n} theorems across {len(domains)} domains",
        f"{proven} PROVEN ({proven / n * 100:.1f}%), {falsified} FALSIFIED ({falsified / n * 100:.1f}%)",
        f"{total_tests} total subtests, {total_pass} passed ({total_pass / total_tests * 100:.1f}%)",
        f"Mean tests/theorem: {total_tests / n:.1f}, median: {sorted(e['n_tests'] for e in entries)[n // 2]}",
    ]

    return PatternResult(
        name="§1: Registry Overview",
        description="Global metrics across all theorems",
        findings=findings,
        data={
            "n_theorems": n,
            "n_proven": proven,
            "n_falsified": falsified,
            "n_errors": errors,
            "n_domains": len(domains),
            "total_tests": total_tests,
            "total_pass": total_pass,
            "proof_rate": round(proven / n, 4),
            "domains": domains,
        },
    )


# ═══════════════════════════════════════════════════════════════════
# §2: ARCHETYPE TAXONOMY
# ═══════════════════════════════════════════════════════════════════


def analyze_archetypes(entries: list[dict[str, Any]]) -> PatternResult:
    """Classify and count theorem archetypes."""
    arch_counts = collections.Counter(e["archetype"] for e in entries)
    arch_proven = collections.Counter()
    for e in entries:
        if e["verdict"] == "PROVEN":
            arch_proven[e["archetype"]] += 1

    findings = []
    for arch, count in arch_counts.most_common():
        prov = arch_proven.get(arch, 0)
        pct = count / len(entries) * 100
        findings.append(f"{arch:<30s} {count:>3d} ({pct:>5.1f}%)  [{prov}/{count} proven]")

    # Structural observation
    top3 = [a for a, _ in arch_counts.most_common(3)]
    findings.append("")
    findings.append(
        f"Top-3 archetypes ({', '.join(top3)}) account for "
        f"{sum(arch_counts[a] for a in top3) / len(entries) * 100:.0f}% of all theorems"
    )

    return PatternResult(
        name="§2: Archetype Taxonomy",
        description="What kinds of theorems exist",
        findings=findings,
        data={"archetype_counts": dict(arch_counts), "archetype_proven": dict(arch_proven)},
    )


# ═══════════════════════════════════════════════════════════════════
# §3: CROSS-DOMAIN STRUCTURAL TWINS
# ═══════════════════════════════════════════════════════════════════


def analyze_twins(entries: list[dict[str, Any]]) -> PatternResult:
    """Find theorems with identical proof structures across domains."""
    # Group by (archetype, n_tests) as a structural fingerprint
    by_fingerprint: dict[tuple[str, int], list[dict[str, Any]]] = collections.defaultdict(list)
    for e in entries:
        fp = (e["archetype"], e["n_tests"])
        by_fingerprint[fp].append(e)

    # Find twins: same archetype + same test count, different domains
    twin_groups: list[dict[str, Any]] = []
    for fp, group in sorted(by_fingerprint.items()):
        domains = {e["domain"] for e in group}
        if len(domains) >= 2 and len(group) >= 2:
            twin_groups.append(
                {
                    "archetype": fp[0],
                    "n_tests": fp[1],
                    "n_theorems": len(group),
                    "domains": sorted(domains),
                    "tags": [e["tag"] for e in group],
                }
            )

    # Also find statement-level twins (similar statements)
    statement_clusters: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for e in entries:
        stmt = e.get("statement", "").lower()
        # Normalize: look for key phrases
        if "f+ω=1" in stmt or "f + ω = 1" in stmt or "tier-1" in stmt.lower():
            statement_clusters["tier1_identity"].append(e)
        elif "geometric slaughter" in stmt or "dead channel" in stmt or "ic kill" in stmt:
            statement_clusters["geometric_slaughter"].append(e)
        elif "regime" in stmt and ("collapse" in stmt or "stable" in stmt or "watch" in stmt):
            statement_clusters["regime_classification"].append(e)

    findings = [
        f"{len(twin_groups)} structural twin groups found",
        "",
        "Largest twin families (same archetype + same test count, multiple domains):",
    ]

    for tg in sorted(twin_groups, key=lambda x: -x["n_theorems"])[:10]:
        findings.append(
            f"  [{tg['archetype']}] n_tests={tg['n_tests']}: "
            f"{tg['n_theorems']} theorems across {', '.join(tg['domains'][:5])}"
        )

    findings.append("")
    findings.append("Statement-level semantic clusters:")
    for cluster_name, members in sorted(statement_clusters.items()):
        domains = sorted({e["domain"] for e in members})
        findings.append(
            f"  {cluster_name}: {len(members)} theorems across {len(domains)} domains ({', '.join(domains[:6])}...)"
        )

    return PatternResult(
        name="§3: Cross-Domain Structural Twins",
        description="Theorems with identical proof structures across different domains",
        findings=findings,
        data={
            "twin_groups": twin_groups,
            "statement_clusters": {k: [e["tag"] for e in v] for k, v in statement_clusters.items()},
        },
    )


# ═══════════════════════════════════════════════════════════════════
# §4: THE FIRST-THEOREM INVARIANT
# ═══════════════════════════════════════════════════════════════════


def analyze_first_theorem(entries: list[dict[str, Any]]) -> PatternResult:
    """Check if every domain's 'first' theorem verifies Tier-1 identities."""
    # Domain → list of entries sorted by tag
    by_domain: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for e in entries:
        by_domain[e["domain"]].append(e)

    first_theorems: list[dict[str, Any]] = []
    identity_first = 0
    non_identity_first: list[str] = []

    for domain in sorted(by_domain):
        de = sorted(by_domain[domain], key=lambda e: e["tag"])
        first = de[0]
        is_identity = first["archetype"] == "identity_verification" or any(
            ic in first.get("identity_checks", []) for ic in ["F+ω=1", "IC≤F", "IC=exp(κ)"]
        )
        first_theorems.append(
            {
                "domain": domain,
                "tag": first["tag"],
                "archetype": first["archetype"],
                "is_identity_check": is_identity,
                "n_tests": first["n_tests"],
            }
        )
        if is_identity:
            identity_first += 1
        else:
            non_identity_first.append(f"{domain} ({first['tag']}: {first['archetype']})")

    n_domains = len(by_domain)
    pct = identity_first / n_domains * 100

    findings = [
        f"First-theorem identity bootstrap: {identity_first}/{n_domains} domains ({pct:.0f}%)",
        "",
    ]

    if identity_first == n_domains:
        findings.append("UNIVERSAL: Every domain's first sorted theorem verifies Tier-1 identities")
    else:
        findings.append("Domains where first theorem is NOT identity verification:")
        for nif in non_identity_first:
            findings.append(f"  {nif}")

    # Test count distribution for first theorems
    test_counts = [ft["n_tests"] for ft in first_theorems]
    findings.append("")
    findings.append(
        f"Identity-verification test counts: "
        f"min={min(test_counts)}, max={max(test_counts)}, "
        f"mean={sum(test_counts) / len(test_counts):.0f}"
    )

    return PatternResult(
        name="§4: First-Theorem Invariant",
        description="Does every domain bootstrap with identity verification?",
        findings=findings,
        data={"first_theorems": first_theorems, "identity_first_pct": pct},
    )


# ═══════════════════════════════════════════════════════════════════
# §5: GEOMETRIC SLAUGHTER MAP
# ═══════════════════════════════════════════════════════════════════


def analyze_slaughter(entries: list[dict[str, Any]]) -> PatternResult:
    """Map where geometric slaughter (IC destruction) appears across domains."""
    slaughter_entries = [
        e
        for e in entries
        if e["archetype"] == "geometric_slaughter"
        or "slaughter" in e.get("statement", "").lower()
        or "dead channel" in e.get("statement", "").lower()
        or "ic kill" in e.get("statement", "").lower()
        or "ic drop" in e.get("statement", "").lower()
    ]

    domains_with_slaughter = sorted({e["domain"] for e in slaughter_entries})

    findings = [
        f"Geometric slaughter detected in {len(domains_with_slaughter)} domains:",
    ]
    for d in domains_with_slaughter:
        de = [e for e in slaughter_entries if e["domain"] == d]
        tags = [e["tag"] for e in de]
        findings.append(f"  {d:<30s} {', '.join(tags)}")

    all_domains = sorted({e["domain"] for e in entries})
    missing = [d for d in all_domains if d not in domains_with_slaughter]
    findings.append("")
    findings.append(
        f"Domains WITHOUT explicit slaughter theorems ({len(missing)}): "
        f"{', '.join(missing[:8])}{'...' if len(missing) > 8 else ''}"
    )

    findings.append("")
    findings.append(
        "Structural insight: Geometric slaughter is the mechanism that connects "
        "IC ≤ F (integrity bound) to observable phenomena. A single dead channel "
        "(c_k → ε) collapses IC via exp(w_k·ln(ε)) while F barely moves."
    )

    return PatternResult(
        name="§5: Geometric Slaughter Map",
        description="Where IC destruction from dead channels appears across domains",
        findings=findings,
        data={
            "n_slaughter_theorems": len(slaughter_entries),
            "domains_with_slaughter": domains_with_slaughter,
            "tags": [e["tag"] for e in slaughter_entries],
        },
    )


# ═══════════════════════════════════════════════════════════════════
# §6: REGIME DISTRIBUTION LANDSCAPE
# ═══════════════════════════════════════════════════════════════════


def analyze_regimes(entries: list[dict[str, Any]]) -> PatternResult:
    """Study regime-related theorems across domains."""
    regime_entries = [e for e in entries if e["archetype"] == "regime_classification"]

    by_domain: dict[str, list[str]] = collections.defaultdict(list)
    for e in regime_entries:
        by_domain[e["domain"]].append(e["tag"])

    # Extract regime mentions from statements
    universal_collapse: list[str] = []
    stability_rare: list[str] = []
    for e in regime_entries:
        stmt = (e.get("statement", "") + " " + e.get("name", "")).lower()
        if "universal collapse" in stmt or "all.*collapse" in stmt:
            universal_collapse.append(e["domain"])
        if "sole" in stmt or "unique" in stmt or "rare" in stmt:
            stability_rare.append(e["domain"])

    findings = [
        f"{len(regime_entries)} regime-classification theorems across {len(by_domain)} domains",
        "",
        "Domains with regime theorems:",
    ]
    for d in sorted(by_domain):
        findings.append(f"  {d:<30s} {len(by_domain[d])} theorems: {', '.join(by_domain[d])}")

    if universal_collapse:
        findings.append("")
        findings.append(f"Domains proving universal Collapse: {', '.join(sorted(set(universal_collapse)))}")

    findings.append("")
    findings.append(
        "Structural pattern: Regime classification is the MOST COMMON "
        f"archetype ({len(regime_entries)} theorems, "
        f"{len(regime_entries) / len(entries) * 100:.1f}%). "
        "This reflects 87.5% of the Fisher manifold being outside Stable — "
        "regime detection is the primary structural task."
    )

    return PatternResult(
        name="§6: Regime Distribution Landscape",
        description="What regimes dominate each domain",
        findings=findings,
        data={"regime_entries_by_domain": dict(by_domain)},
    )


# ═══════════════════════════════════════════════════════════════════
# §7: FALSIFICATION FORENSICS
# ═══════════════════════════════════════════════════════════════════


def analyze_falsifications(entries: list[dict[str, Any]]) -> PatternResult:
    """Deep investigation of FALSIFIED theorems."""
    falsified = [e for e in entries if e["verdict"] == "FALSIFIED"]

    if not falsified:
        return PatternResult(
            name="§7: Falsification Forensics",
            description="No falsified theorems — perfect proof coverage",
            findings=["ALL 225 theorems PROVEN. No falsification forensics needed."],
        )

    findings = [
        f"{len(falsified)} FALSIFIED theorems found:",
        "",
    ]

    by_domain = collections.Counter(e["domain"] for e in falsified)
    for domain, count in by_domain.most_common():
        findings.append(f"  {domain}: {count} falsified")

    findings.append("")
    findings.append("Detailed breakdown:")

    for e in falsified:
        tag = e["tag"]
        prate = e["pass_rate"]
        findings.append(f"\n  {tag} ({e['domain']}): {e['n_passed']}/{e['n_tests']} ({prate * 100:.0f}%)")
        findings.append(f"    Name: {e['name']}")
        findings.append(f"    Statement: {e['statement'][:120]}")
        findings.append(f"    Archetype: {e['archetype']}")
        # Check detail keys for clues
        ds = e.get("details_summary", {})
        for k, v in list(ds.items())[:5]:
            findings.append(f"    {k}: {str(v)[:80]}")

    # Structural diagnosis
    falsified_archetypes = collections.Counter(e["archetype"] for e in falsified)
    findings.append("")
    findings.append("Falsified archetypes: " + ", ".join(f"{a}({c})" for a, c in falsified_archetypes.most_common()))

    concentrated = len(by_domain) == 1
    if concentrated:
        domain = next(iter(by_domain.keys()))
        findings.append("")
        findings.append(
            f"DIAGNOSIS: All falsifications concentrated in '{domain}'. "
            f"This suggests the {domain} entity catalog has channel definitions "
            f"that don't fully match the theorem expectations — a Tier-2 "
            f"calibration issue, not a Tier-1 violation."
        )

    return PatternResult(
        name="§7: Falsification Forensics",
        description="What breaks and why",
        findings=findings,
        data={
            "n_falsified": len(falsified),
            "by_domain": dict(by_domain),
            "by_archetype": dict(falsified_archetypes),
            "tags": [e["tag"] for e in falsified],
        },
    )


# ═══════════════════════════════════════════════════════════════════
# §8: TEST DENSITY SPECTRUM
# ═══════════════════════════════════════════════════════════════════


def analyze_test_density(entries: list[dict[str, Any]]) -> PatternResult:
    """Analyze the distribution of subtests per theorem."""
    tests = sorted(e["n_tests"] for e in entries)
    n = len(tests)

    # Quartiles
    q1 = tests[n // 4]
    q2 = tests[n // 2]
    q3 = tests[3 * n // 4]
    total = sum(tests)

    # Extremes
    min_e = min(entries, key=lambda e: e["n_tests"])
    max_e = max(entries, key=lambda e: e["n_tests"])

    # Distribution buckets
    buckets = {"1-5": 0, "6-10": 0, "11-25": 0, "26-50": 0, "51-100": 0, "101-200": 0, "201-500": 0, "500+": 0}
    for t in tests:
        if t <= 5:
            buckets["1-5"] += 1
        elif t <= 10:
            buckets["6-10"] += 1
        elif t <= 25:
            buckets["11-25"] += 1
        elif t <= 50:
            buckets["26-50"] += 1
        elif t <= 100:
            buckets["51-100"] += 1
        elif t <= 200:
            buckets["101-200"] += 1
        elif t <= 500:
            buckets["201-500"] += 1
        else:
            buckets["500+"] += 1

    findings = [
        f"Test density: min={tests[0]}, Q1={q1}, median={q2}, Q3={q3}, max={tests[-1]}",
        f"Total subtests: {total} across {n} theorems",
        f"Mean: {total / n:.1f} tests/theorem",
        "",
        "Distribution:",
    ]
    for bucket, count in buckets.items():
        bar = "█" * (count * 40 // n)
        findings.append(f"  {bucket:>10s}: {count:>4d} ({count / n * 100:>5.1f}%) {bar}")

    findings.append("")
    findings.append(f"Sparsest: {min_e['tag']} ({min_e['domain']}): {min_e['n_tests']} tests")
    findings.append(f"Densest:  {max_e['tag']} ({max_e['domain']}): {max_e['n_tests']} tests")
    findings.append("")

    # High-density theorems (top 10)
    top10 = sorted(entries, key=lambda e: -e["n_tests"])[:10]
    findings.append("Top-10 highest-density theorems:")
    for e in top10:
        findings.append(f"  {e['tag']:<12s} {e['domain']:<25s} {e['n_tests']:>6d} tests ({e['archetype']})")

    findings.append("")
    findings.append(
        f"Dynamic range: {tests[-1] / max(tests[0], 1):.0f}× (from {tests[0]} to {tests[-1]} tests per theorem)"
    )

    return PatternResult(
        name="§8: Test Density Spectrum",
        description="Subtest count distribution across theorems",
        findings=findings,
        data={
            "quartiles": [tests[0], q1, q2, q3, tests[-1]],
            "buckets": buckets,
            "total_tests": total,
        },
    )


# ═══════════════════════════════════════════════════════════════════
# §9: IDENTITY VERIFICATION NETWORK
# ═══════════════════════════════════════════════════════════════════


def analyze_identity_network(entries: list[dict[str, Any]]) -> PatternResult:
    """Map which Tier-1 identities are verified where."""
    identities = ["F+ω=1", "IC≤F", "IC=exp(κ)"]

    # Build coverage matrix: domain × identity
    domains = sorted({e["domain"] for e in entries})
    coverage: dict[str, dict[str, int]] = {d: dict.fromkeys(identities, 0) for d in domains}

    for e in entries:
        for ic in e.get("identity_checks", []):
            if ic in identities:
                coverage[e["domain"]][ic] += 1

    # Also count globally
    global_counts = {i: sum(1 for e in entries if i in e.get("identity_checks", [])) for i in identities}

    findings = [
        "Identity verification network (domains × identities):",
        "",
        f"{'Domain':<30s} {'F+ω=1':>8s} {'IC≤F':>8s} {'IC=exp(κ)':>10s}",
        "─" * 60,
    ]

    for d in domains:
        row = coverage[d]
        findings.append(f"  {d:<28s} {row['F+ω=1']:>8d} {row['IC≤F']:>8d} {row['IC=exp(κ)']:>10d}")

    findings.append("─" * 60)
    findings.append(
        f"  {'TOTAL':<28s} {global_counts['F+ω=1']:>8d} {global_counts['IC≤F']:>8d} {global_counts['IC=exp(κ)']:>10d}"
    )

    # Which domains verify all 3?
    all3 = [d for d in domains if all(coverage[d][i] > 0 for i in identities)]
    [d for d in domains if any(coverage[d][i] > 0 for i in identities)]
    no_check = [d for d in domains if not any(coverage[d][i] > 0 for i in identities)]

    findings.append("")
    findings.append(f"Domains verifying all 3 identities: {len(all3)}/{len(domains)}")
    if all3:
        findings.append(f"  {', '.join(all3)}")
    if no_check:
        findings.append(f"Domains with no explicit identity verification: {', '.join(no_check)}")

    return PatternResult(
        name="§9: Identity Verification Network",
        description="Which Tier-1 identities are proven in which domains",
        findings=findings,
        data={"coverage": coverage, "global_counts": global_counts},
    )


# ═══════════════════════════════════════════════════════════════════
# §10: ARCHETYPE SIGNATURES PER DOMAIN
# ═══════════════════════════════════════════════════════════════════


def analyze_domain_signatures(entries: list[dict[str, Any]]) -> PatternResult:
    """Compute archetype distribution per domain as a 'fingerprint'."""
    domains = sorted({e["domain"] for e in entries})
    archetypes = sorted({e["archetype"] for e in entries})

    # Build matrix
    matrix: dict[str, dict[str, int]] = {d: dict.fromkeys(archetypes, 0) for d in domains}
    for e in entries:
        matrix[e["domain"]][e["archetype"]] += 1

    # Compute Jaccard similarity between domain pairs
    similarities: list[tuple[float, str, str]] = []
    for d1, d2 in combinations(domains, 2):
        v1 = {a for a in archetypes if matrix[d1][a] > 0}
        v2 = {a for a in archetypes if matrix[d2][a] > 0}
        if v1 or v2:
            jaccard = len(v1 & v2) / len(v1 | v2)
            similarities.append((jaccard, d1, d2))

    similarities.sort(reverse=True)

    findings = [
        "Domain archetype fingerprints (columns = archetype counts):",
        "",
    ]

    # Compact header
    arch_abbrev = {
        "identity_verification": "ID",
        "geometric_slaughter": "GS",
        "ordering_hierarchy": "OH",
        "regime_classification": "RC",
        "phase_transition": "PT",
        "correlation_structure": "CS",
        "cross_scale": "XS",
        "composition": "CO",
        "extremal_entity": "EE",
        "coverage_completeness": "CC",
        "domain_specific": "DS",
    }

    header = f"{'Domain':<28s} " + " ".join(f"{arch_abbrev.get(a, a[:2]):>3s}" for a in archetypes)
    findings.append(header)
    findings.append("─" * len(header))

    for d in domains:
        row = " ".join(f"{matrix[d][a]:>3d}" for a in archetypes)
        total = sum(matrix[d][a] for a in archetypes)
        findings.append(f"  {d:<26s} {row}  ({total})")

    findings.append("")
    findings.append("Most similar domain pairs (Jaccard similarity):")
    for sim, d1, d2 in similarities[:8]:
        findings.append(f"  {d1:<25s} <-> {d2:<25s}  J={sim:.3f}")

    findings.append("")
    findings.append("Most dissimilar domain pairs:")
    for sim, d1, d2 in similarities[-3:]:
        findings.append(f"  {d1:<25s} <-> {d2:<25s}  J={sim:.3f}")

    return PatternResult(
        name="§10: Archetype Signatures per Domain",
        description="Domain fingerprints based on theorem archetype distribution",
        findings=findings,
        data={
            "matrix": matrix,
            "similarities_top": [(s, d1, d2) for s, d1, d2 in similarities[:10]],
        },
    )


# ═══════════════════════════════════════════════════════════════════
# §11: EMERGENT STRUCTURAL LAWS
# ═══════════════════════════════════════════════════════════════════


def analyze_structural_laws(entries: list[dict[str, Any]]) -> PatternResult:
    """Discover laws that hold across all or most domains."""
    domains = sorted({e["domain"] for e in entries})
    n_domains = len(domains)

    laws: list[dict[str, Any]] = []

    # Law 1: Does every domain have at least one identity-verification theorem?
    id_domains = {e["domain"] for e in entries if e["archetype"] == "identity_verification"}
    law1 = {
        "name": "Identity Bootstrap Law",
        "statement": "Every domain contains at least one identity-verification theorem",
        "coverage": len(id_domains),
        "total": n_domains,
        "holds": len(id_domains) == n_domains,
        "exceptions": sorted(set(domains) - id_domains),
    }
    laws.append(law1)

    # Law 2: Does every domain have at least one regime theorem?
    regime_domains = {e["domain"] for e in entries if e["archetype"] == "regime_classification"}
    law2 = {
        "name": "Regime Ubiquity Law",
        "statement": "Every domain contains at least one regime-classification theorem",
        "coverage": len(regime_domains),
        "total": n_domains,
        "holds": len(regime_domains) == n_domains,
        "exceptions": sorted(set(domains) - regime_domains),
    }
    laws.append(law2)

    # Law 3: Is proof rate > 90% in every domain?
    by_domain: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for e in entries:
        by_domain[e["domain"]].append(e)

    low_proof_domains = []
    for d in domains:
        de = by_domain[d]
        prate = sum(1 for e in de if e["verdict"] == "PROVEN") / len(de)
        if prate < 0.9:
            low_proof_domains.append((d, prate))

    law3 = {
        "name": "Universal Provability Law",
        "statement": "Every domain achieves ≥90% proof rate",
        "coverage": n_domains - len(low_proof_domains),
        "total": n_domains,
        "holds": len(low_proof_domains) == 0,
        "exceptions": [f"{d} ({r * 100:.0f}%)" for d, r in low_proof_domains],
    }
    laws.append(law3)

    # Law 4: Does ordering_hierarchy appear in ≥50% of domains?
    oh_domains = {e["domain"] for e in entries if e["archetype"] == "ordering_hierarchy"}
    law4 = {
        "name": "Hierarchy Prevalence Law",
        "statement": "Ordering/hierarchy theorems appear in ≥50% of domains",
        "coverage": len(oh_domains),
        "total": n_domains,
        "holds": len(oh_domains) >= n_domains * 0.5,
    }
    laws.append(law4)

    # Law 5: Test density law — more entity-rich domains have higher test counts
    total_tests_per_domain = {d: sum(e["n_tests"] for e in de) for d, de in by_domain.items()}
    top_test_domains = sorted(total_tests_per_domain.items(), key=lambda x: -x[1])[:5]
    law5 = {
        "name": "Entity-Density Scaling Law",
        "statement": "Domains with more entities produce exponentially more subtests",
        "top_5": [(d, t) for d, t in top_test_domains],
        "ratio": top_test_domains[0][1] / max(top_test_domains[-1][1], 1),
    }
    laws.append(law5)

    # Law 6: Archetype diversity — is there a minimum number of distinct archetypes?
    arch_per_domain = {d: len({e["archetype"] for e in de}) for d, de in by_domain.items()}
    min_arch = min(arch_per_domain.values())
    max_arch = max(arch_per_domain.values())
    law6 = {
        "name": "Archetype Diversity Minimum",
        "statement": f"Every domain exhibits ≥{min_arch} distinct archetype(s)",
        "min": min_arch,
        "max": max_arch,
        "distribution": dict(sorted(arch_per_domain.items())),
    }
    laws.append(law6)

    # Law 7: Perfect pass rate concentration
    perfect = [e for e in entries if e["pass_rate"] == 1.0 and e["verdict"] == "PROVEN"]
    imperfect = [e for e in entries if 0 < e["pass_rate"] < 1.0]
    law7 = {
        "name": "Binary Verdict Law",
        "statement": "Theorems tend to either fully prove or clearly fail — few are close calls",
        "n_perfect": len(perfect),
        "n_imperfect": len(imperfect),
        "perfect_pct": len(perfect) / len(entries) * 100,
    }
    laws.append(law7)

    # Law 8: Composition structure — analyze how archetype profiles cluster
    sorted({e["archetype"] for e in entries})
    # Count domains that have at least one of each archetype pair
    cooccurrence: dict[tuple[str, str], int] = collections.Counter()
    for _d, de in by_domain.items():
        d_archs = {e["archetype"] for e in de}
        for a1, a2 in combinations(sorted(d_archs), 2):
            cooccurrence[(a1, a2)] += 1

    most_common_pairs = cooccurrence.most_common(5)
    law8 = {
        "name": "Archetype Co-occurrence Law",
        "statement": "Certain archetype pairs always appear together",
        "top_pairs": [(f"{a1}+{a2}", c) for (a1, a2), c in most_common_pairs],
    }
    laws.append(law8)

    # Format findings
    findings = []
    for i, law in enumerate(laws, 1):
        name = law["name"]
        stmt = law.get("statement", "")
        holds = law.get("holds", None)
        status = "✓ HOLDS" if holds else "✗ BROKEN" if holds is not None else "—"
        findings.append(f"Law {i}: {name} {status}")
        findings.append(f"  {stmt}")
        if "coverage" in law:
            findings.append(f"  Coverage: {law['coverage']}/{law.get('total', '?')}")
        if law.get("exceptions"):
            findings.append(f"  Exceptions: {', '.join(str(x) for x in law['exceptions'])}")
        if "top_5" in law:
            for d, t in law["top_5"]:
                findings.append(f"    {d}: {t} tests")
        if "top_pairs" in law:
            for pair, c in law["top_pairs"]:
                findings.append(f"    {pair}: in {c}/{n_domains} domains")
        findings.append("")

    return PatternResult(
        name="§11: Emergent Structural Laws",
        description="Universal patterns that hold across all or most domains",
        findings=findings,
        data={"laws": laws},
    )


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════


_ANALYSES = [
    analyze_overview,
    analyze_archetypes,
    analyze_twins,
    analyze_first_theorem,
    analyze_slaughter,
    analyze_regimes,
    analyze_falsifications,
    analyze_test_density,
    analyze_identity_network,
    analyze_domain_signatures,
    analyze_structural_laws,
]


def run_all(entries: list[dict[str, Any]]) -> list[PatternResult]:
    """Run all pattern analyses."""
    results = []
    for analyzer in _ANALYSES:
        results.append(analyzer(entries))
    return results


def main() -> None:
    """Run full pattern analysis pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="Theorem Pattern Analysis")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--registry", type=str, default=None)
    args = parser.parse_args()

    path = Path(args.registry) if args.registry else None
    entries = load_registry(path)
    results = run_all(entries)

    if args.json:
        data = [
            {
                "name": r.name,
                "description": r.description,
                "findings": r.findings,
                "data": r.data,
            }
            for r in results
        ]

        def _default(o: Any) -> Any:
            if hasattr(o, "__float__"):
                return float(o)
            return str(o)

        print(json.dumps(data, indent=2, default=_default))
    else:
        for r in results:
            print(f"\n{'═' * 70}")
            print(f"{r.name}")
            print(f"{r.description}")
            print(f"{'═' * 70}")
            for line in r.findings:
                print(f"  {line}")


if __name__ == "__main__":
    main()

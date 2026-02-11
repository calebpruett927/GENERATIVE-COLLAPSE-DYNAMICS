#!/usr/bin/env python3
"""Discover insights from materials-science closures and generate report.

Usage::

    python scripts/discover_insights.py              # Full discovery + report
    python scripts/discover_insights.py --startup     # Show one rotating insight
    python scripts/discover_insights.py --save        # Discover + persist to YAML
    python scripts/discover_insights.py --philosophy  # Show engine philosophy

This script scans all 8 materials-science closures, detects patterns across
the periodic table, correlates quantities between closures, identifies regime
boundaries, and builds a lessons-learned database that grows with each run.

Part of the UMCP Lessons-Learned Insight Engine.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repo root on sys.path
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.umcp.insights import InsightEngine


def main() -> None:
    parser = argparse.ArgumentParser(
        description="UMCP Materials-Science Insight Discovery Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Every anomalous ω_eff is not a failure — it is a signal\n"
            "pointing toward physics the model has not yet learned.\n\n"
            '"What returns through collapse is real."'
        ),
    )
    parser.add_argument(
        "--startup",
        action="store_true",
        help="Show a single rotating insight (changes daily)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Persist discovered insights to lessons_db.yaml",
    )
    parser.add_argument(
        "--philosophy",
        action="store_true",
        help="Display the engine's guiding philosophy",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Filter report to a single domain",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Deterministic seed for --startup (default: date-based)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print summary statistics as JSON",
    )

    args = parser.parse_args()

    # ── Philosophy ──────────────────────────────────────────────
    if args.philosophy:
        print(InsightEngine.philosophy())
        return

    # ── Initialize engine ───────────────────────────────────────
    engine = InsightEngine(load_canon=True, load_db=True)
    canon_count = engine.db.count()

    # ── Startup insight ─────────────────────────────────────────
    if args.startup:
        # Quick mode: show one insight, no discovery
        print(engine.show_startup_insight(seed=args.seed))
        return

    # ── Discovery ───────────────────────────────────────────────
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║     UMCP — Materials-Science Insight Discovery Engine       ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    print(f"  Loaded {canon_count} canonical lessons from anchors/database.")
    print()

    print("  ▶ Discovering periodic trends...")
    pt = engine.discover_periodic_trends()
    print(f"    Found {len(pt)} new periodic-trend insights.")

    print("  ▶ Discovering regime boundaries...")
    rb = engine.discover_regime_boundaries()
    print(f"    Found {len(rb)} new regime-boundary insights.")

    print("  ▶ Discovering cross-correlations...")
    cc = engine.discover_cross_correlations()
    print(f"    Found {len(cc)} new cross-correlation insights.")

    print("  ▶ Discovering universality signatures...")
    us = engine.discover_universality_signatures()
    print(f"    Found {len(us)} new universality insights.")

    total_new = len(pt) + len(rb) + len(cc) + len(us)
    print()
    print(f"  Total: {engine.db.count()} insights ({canon_count} canon + {total_new} discovered)")
    print()

    # ── Stats ───────────────────────────────────────────────────
    if args.stats:
        import json

        stats = engine.summary_stats()
        print(json.dumps(stats, indent=2))
        print()

    # ── Save ────────────────────────────────────────────────────
    if args.save:
        engine.save()
        print(f"  ✓ Saved {engine.db.count()} insights to lessons_db.yaml")
        print()

    # ── Report ──────────────────────────────────────────────────
    if args.domain:
        # Filtered report
        entries = engine.db.query(domain=args.domain)
        if not entries:
            print(f"  No insights found for domain '{args.domain}'.")
            print(f"  Available domains: {', '.join(engine.db.domains())}")
        else:
            for e in entries:
                print(f"\n  [{e.id}] {e.severity.value} — {e.pattern}")
                print(f"    {e.lesson}")
                print(f"    → {e.implication}")
    else:
        print(engine.full_report())


if __name__ == "__main__":
    main()

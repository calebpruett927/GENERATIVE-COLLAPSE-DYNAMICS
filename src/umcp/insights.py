"""Lessons-Learned Insight Engine — dynamic pattern discovery and wisdom extraction.

Transforms the UMCP materials-science framework from a pure validator into a
living knowledge system that discovers, stores, and reports patterns across
closure domains.

Architecture:
    InsightEntry        Typed dataclass for a single pattern / lesson
    PatternDatabase     Load from YAML ↔ discover from closures ↔ query
    InsightEngine       Orchestrator: scan, correlate, report, startup banner

Usage::

    from umcp.insights import InsightEngine
    engine = InsightEngine()
    engine.discover_all()           # Run all closures, detect patterns
    engine.show_startup_insight()   # Print a rotating insight
    engine.full_report()            # Print all lessons + new discoveries

Design principle: ω_eff is NOT a failure metric — it is a *signal*
that maps the boundary of each model's domain of validity.  Every
Anomalous regime classification is a lesson about physics.
"""

from __future__ import annotations

import hashlib
import math
import random
import textwrap
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
#  Optional YAML dependency
# ---------------------------------------------------------------------------
try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_LESSONS_DB = _REPO_ROOT / "closures" / "materials_science" / "lessons_db.yaml"
_CANON_MATL = _REPO_ROOT / "canon" / "matl_anchors.yaml"

# ---------------------------------------------------------------------------
#  Severity levels for insights
# ---------------------------------------------------------------------------


class InsightSeverity(str, Enum):
    """How significant the pattern is."""

    FUNDAMENTAL = "Fundamental"  # Core physics principle revealed
    STRUCTURAL = "Structural"  # Architecture / model boundary
    EMPIRICAL = "Empirical"  # Data-driven observation
    CURIOUS = "Curious"  # Interesting but not yet actionable


class PatternType(str, Enum):
    """Category of discovered pattern."""

    PERIODIC_TREND = "PeriodicTrend"
    REGIME_BOUNDARY = "RegimeBoundary"
    CROSS_CORRELATION = "CrossCorrelation"
    UNIVERSALITY = "Universality"
    MODEL_LIMITATION = "ModelLimitation"
    PHYSICAL_INSIGHT = "PhysicalInsight"


# ---------------------------------------------------------------------------
#  InsightEntry — single pattern / lesson
# ---------------------------------------------------------------------------


@dataclass
class InsightEntry:
    """A single discovered insight, lesson, or pattern."""

    id: str
    domain: str
    pattern: str
    lesson: str
    implication: str
    severity: InsightSeverity = InsightSeverity.EMPIRICAL
    pattern_type: PatternType = PatternType.PHYSICAL_INSIGHT
    source: str = "canon"  # "canon" | "discovered" | "cross-closure"
    elements: list[str] = field(default_factory=list)
    omega_range: tuple[float, float] = (0.0, 1.0)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for YAML output."""
        return {
            "id": self.id,
            "domain": self.domain,
            "pattern": self.pattern,
            "lesson": self.lesson,
            "implication": self.implication,
            "severity": self.severity.value,
            "pattern_type": self.pattern_type.value,
            "source": self.source,
            "elements": self.elements,
            "omega_range": list(self.omega_range),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> InsightEntry:
        """Deserialize from YAML / dict."""
        return cls(
            id=d["id"],
            domain=d.get("domain", "General"),
            pattern=d.get("pattern", ""),
            lesson=d.get("lesson", ""),
            implication=d.get("implication", ""),
            severity=InsightSeverity(d.get("severity", InsightSeverity.EMPIRICAL.value)),
            pattern_type=PatternType(d.get("pattern_type", PatternType.PHYSICAL_INSIGHT.value)),
            source=d.get("source", "canon"),
            elements=d.get("elements", []),
            omega_range=(d.get("omega_range", [0.0, 1.0])[0], d.get("omega_range", [0.0, 1.0])[1]),
        )


# ---------------------------------------------------------------------------
#  PatternDatabase — load, query, persist
# ---------------------------------------------------------------------------


class PatternDatabase:
    """In-memory pattern store backed by YAML on disk."""

    def __init__(self) -> None:
        self.entries: list[InsightEntry] = []
        self._seen_ids: set[str] = set()

    def add(self, entry: InsightEntry) -> bool:
        """Add entry if its id is new. Returns True if added."""
        if entry.id in self._seen_ids:
            return False
        self.entries.append(entry)
        self._seen_ids.add(entry.id)
        return True

    def query(
        self,
        *,
        domain: str | None = None,
        severity: InsightSeverity | None = None,
        pattern_type: PatternType | None = None,
        source: str | None = None,
    ) -> list[InsightEntry]:
        """Filter entries by optional criteria."""
        result = self.entries
        if domain is not None:
            result = [e for e in result if e.domain == domain]
        if severity is not None:
            result = [e for e in result if e.severity == severity]
        if pattern_type is not None:
            result = [e for e in result if e.pattern_type == pattern_type]
        if source is not None:
            result = [e for e in result if e.source == source]
        return result

    def count(self) -> int:
        return len(self.entries)

    # -- I/O ----------------------------------------------------------------

    def load_yaml(self, path: Path | None = None) -> int:
        """Load entries from a YAML lessons database. Returns count loaded."""
        if yaml is None:
            return 0
        path = path or _LESSONS_DB
        if not path.exists():
            return 0
        with open(path) as f:
            data = yaml.safe_load(f)
        if not data or "insights" not in data:
            return 0
        count = 0
        for d in data["insights"]:
            if self.add(InsightEntry.from_dict(d)):
                count += 1
        return count

    def load_canon(self, path: Path | None = None) -> int:
        """Load lessons_learned entries from canon/matl_anchors.yaml."""
        if yaml is None:
            return 0
        path = path or _CANON_MATL
        if not path.exists():
            return 0
        with open(path) as f:
            data = yaml.safe_load(f)
        if not data or "lessons_learned" not in data:
            return 0
        count = 0
        for d in data["lessons_learned"]:
            entry = InsightEntry(
                id=d["id"],
                domain=d.get("domain", "General"),
                pattern=d.get("pattern", ""),
                lesson=d.get("lesson", ""),
                implication=d.get("implication", ""),
                severity=InsightSeverity.FUNDAMENTAL,
                pattern_type=PatternType.PHYSICAL_INSIGHT,
                source="canon",
            )
            if self.add(entry):
                count += 1
        return count

    def save_yaml(self, path: Path | None = None) -> None:
        """Persist all entries to YAML."""
        if yaml is None:
            return
        path = path or _LESSONS_DB
        data = {
            "version": "1.0",
            "description": "UMCP Materials-Science Lessons-Learned Database",
            "insight_count": len(self.entries),
            "insights": [e.to_dict() for e in self.entries],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, width=100)

    def domains(self) -> list[str]:
        """Return sorted list of unique domains."""
        return sorted({e.domain for e in self.entries})


# ---------------------------------------------------------------------------
#  Periodic-table element data (Z → symbol)
# ---------------------------------------------------------------------------

_Z_TO_SYMBOL: dict[int, str] = {
    3: "Li",
    4: "Be",
    11: "Na",
    12: "Mg",
    13: "Al",
    14: "Si",
    19: "K",
    20: "Ca",
    22: "Ti",
    23: "V",
    24: "Cr",
    25: "Mn",
    26: "Fe",
    27: "Co",
    28: "Ni",
    29: "Cu",
    30: "Zn",
    31: "Ga",
    32: "Ge",
    33: "As",
    37: "Rb",
    38: "Sr",
    40: "Zr",
    41: "Nb",
    42: "Mo",
    44: "Ru",
    45: "Rh",
    46: "Pd",
    47: "Ag",
    48: "Cd",
    49: "In",
    50: "Sn",
    55: "Cs",
    56: "Ba",
    72: "Hf",
    73: "Ta",
    74: "W",
    75: "Re",
    76: "Os",
    77: "Ir",
    78: "Pt",
    79: "Au",
    82: "Pb",
    83: "Bi",
}


# ---------------------------------------------------------------------------
#  InsightEngine — orchestrator
# ---------------------------------------------------------------------------


class InsightEngine:
    """Main insight engine: discover patterns, manage database, format reports.

    The engine treats every anomaly as information, every boundary as a lesson,
    and every cross-correlation as a bridge between sub-disciplines.

    Philosophy: Science is not a fixed collection of facts but a living process
    where each measurement — each collapse and return — teaches something about
    the boundary conditions of our models.  What returns through collapse is real;
    what doesn't return tells us where to look next.
    """

    def __init__(self, *, load_canon: bool = True, load_db: bool = True) -> None:
        self.db = PatternDatabase()
        if load_canon:
            self.db.load_canon()
        if load_db:
            self.db.load_yaml()

    # -- Discovery ----------------------------------------------------------

    def discover_periodic_trends(self) -> list[InsightEntry]:
        """Scan cohesive closure across Z, detect ω_eff trend-breaks."""
        try:
            from closures.materials_science.cohesive_energy import (
                compute_cohesive_energy,
            )
        except ImportError:
            return []

        insights: list[InsightEntry] = []

        # Scan 3d transition metals
        _3d = [22, 23, 24, 25, 26, 27, 28, 29, 30]  # Ti→Zn
        omegas_3d: list[tuple[str, float]] = []
        for z in _3d:
            sym = _Z_TO_SYMBOL.get(z, "")
            if not sym:
                continue
            try:
                r = compute_cohesive_energy(z, symbol=sym)
                omegas_3d.append((sym, r.omega_eff))
            except Exception:
                continue

        if len(omegas_3d) >= 5:
            # Find elements with ω > 0.40 (Anomalous boundary)
            anomalous = [(s, w) for s, w in omegas_3d if w > 0.40]
            precise = [(s, w) for s, w in omegas_3d if w < 0.10]

            if anomalous and precise:
                entry = InsightEntry(
                    id=f"DIS-PT-3D-{_hash_short(str(anomalous))}",
                    domain="Cohesive Energy",
                    pattern=(
                        f"3d series: {len(precise)} elements Precise (ω<0.10), {len(anomalous)} Anomalous (ω>0.40)"
                    ),
                    lesson=(
                        f"Elements {', '.join(s for s, _ in anomalous)} resist the "
                        f"Friedel-EAM model while {', '.join(s for s, _ in precise)} "
                        f"are well-captured. The boundary between success and failure "
                        f"lies at the magnetic exchange threshold — strong spin "
                        f"alignment adds a cohesive channel the band model cannot see."
                    ),
                    implication="Magnetic metals need explicit exchange treatment",
                    severity=InsightSeverity.STRUCTURAL,
                    pattern_type=PatternType.REGIME_BOUNDARY,
                    source="discovered",
                    elements=[s for s, _ in omegas_3d],
                    omega_range=(
                        min(w for _, w in omegas_3d),
                        max(w for _, w in omegas_3d),
                    ),
                )
                insights.append(entry)
                self.db.add(entry)

        # Scan 4d transition metals
        _4d = [40, 41, 42, 44, 45, 46, 47, 48]  # Zr→Cd
        omegas_4d: list[tuple[str, float]] = []
        for z in _4d:
            sym = _Z_TO_SYMBOL.get(z, "")
            if not sym:
                continue
            try:
                r = compute_cohesive_energy(z, symbol=sym)
                omegas_4d.append((sym, r.omega_eff))
            except Exception:
                continue

        if len(omegas_4d) >= 4:
            avg_4d = sum(w for _, w in omegas_4d) / len(omegas_4d)
            if avg_4d < 0.15:
                entry = InsightEntry(
                    id=f"DIS-PT-4D-{_hash_short(str(omegas_4d))}",
                    domain="Cohesive Energy",
                    pattern=(f"4d series average ω = {avg_4d:.3f} — systematically better than 3d series"),
                    lesson=(
                        "4d metals have weaker exchange interactions (broader d-bands, "
                        "larger orbital extent) → Friedel-EAM model works better. "
                        "Magnetic perturbations are smaller for 4d/5d than 3d."
                    ),
                    implication=("Bandwidth → exchange hierarchy: broader bands = weaker magnetism = lower ω_eff"),
                    severity=InsightSeverity.EMPIRICAL,
                    pattern_type=PatternType.PERIODIC_TREND,
                    source="discovered",
                    elements=[s for s, _ in omegas_4d],
                    omega_range=(
                        min(w for _, w in omegas_4d),
                        max(w for _, w in omegas_4d),
                    ),
                )
                insights.append(entry)
                self.db.add(entry)

        return insights

    def discover_regime_boundaries(self) -> list[InsightEntry]:
        """Detect where regime flips happen across each closure."""
        insights: list[InsightEntry] = []

        # -- Magnetic: find Tc boundary by scanning T --
        try:
            from closures.materials_science.magnetic_properties import (
                compute_magnetic_properties,
            )

            # Iron: detailed T scan
            ferro_temps: list[float] = []
            para_temps: list[float] = []
            for T in range(100, 1500, 50):
                r = compute_magnetic_properties(0, symbol="Fe", T_K=float(T))
                if r.M_total_B > 0:
                    ferro_temps.append(float(T))
                else:
                    para_temps.append(float(T))

            if ferro_temps and para_temps:
                boundary = max(ferro_temps)
                entry = InsightEntry(
                    id="DIS-RB-FE-CURIE",
                    domain="Magnetic Properties",
                    pattern=(f"Fe: ferromagnetic → paramagnetic boundary at T ≈ {boundary:.0f} K (reference: 1043 K)"),
                    lesson=(
                        "The RCFT critical exponent β = 5/6 predicts M(T) = "
                        "M_sat·|1−T/T_c|^(5/6). The magnetic-to-paramagnetic "
                        "regime boundary IS the Curie temperature — the engine "
                        "discovers T_c by computing where M vanishes."
                    ),
                    implication="RCFT β exponent generates verifiable phase boundaries",
                    severity=InsightSeverity.FUNDAMENTAL,
                    pattern_type=PatternType.REGIME_BOUNDARY,
                    source="discovered",
                    elements=["Fe"],
                )
                insights.append(entry)
                self.db.add(entry)

        except ImportError:
            pass

        # -- BCS: find type-I/type-II boundary --
        try:
            from closures.materials_science.bcs_superconductivity import (
                compute_bcs_superconductivity,
            )

            type1: list[str] = []
            type2: list[str] = []
            normal: list[str] = []
            for sym in ["Nb", "Pb", "Al", "Sn", "V", "Ta", "In"]:
                try:
                    r_bcs = compute_bcs_superconductivity(0, 0, symbol=sym)
                    if hasattr(r_bcs, "sc_type"):
                        if r_bcs.sc_type == "TypeI":
                            type1.append(sym)
                        elif r_bcs.sc_type == "TypeII":
                            type2.append(sym)
                        else:
                            normal.append(sym)
                except Exception:
                    continue

            if type1 or type2:
                entry = InsightEntry(
                    id="DIS-RB-SC-TYPE",
                    domain="Superconductivity",
                    pattern=(f"Type-I: [{', '.join(type1)}], Type-II: [{', '.join(type2)}]"),
                    lesson=(
                        "The GL parameter κ = λ_L/ξ₀ divides superconductors "
                        "at κ = 1/√2. Type-I (κ < 1/√2) expel flux completely; "
                        "Type-II (κ > 1/√2) admit quantized vortices. The BCS "
                        "closure discovers this boundary from Θ_D and λ_ep alone."
                    ),
                    implication=("Two material parameters predict macroscopic flux behavior"),
                    severity=InsightSeverity.STRUCTURAL,
                    pattern_type=PatternType.REGIME_BOUNDARY,
                    source="discovered",
                    elements=type1 + type2,
                )
                insights.append(entry)
                self.db.add(entry)

        except ImportError:
            pass

        return insights

    def discover_cross_correlations(self) -> list[InsightEntry]:
        """Find correlations between quantities from different closures."""
        insights: list[InsightEntry] = []

        # -- ΘD vs surface energy --
        try:
            from closures.materials_science.debye_thermal import (
                compute_debye_thermal,
            )
            from closures.materials_science.surface_catalysis import (
                compute_surface_catalysis,
            )

            pairs: list[tuple[str, float, float]] = []
            for sym in ["Cu", "Ag", "Au", "Pt", "Ni", "Al", "W", "Mo"]:
                try:
                    d = compute_debye_thermal(300.0, symbol=sym)
                    s = compute_surface_catalysis(0, symbol=sym)
                    pairs.append((sym, d.Theta_D_K, s.gamma_J_m2))
                except Exception:
                    continue

            if len(pairs) >= 4:
                # Check Pearson correlation
                thetas = [p[1] for p in pairs]
                gammas = [p[2] for p in pairs]
                r_val = _pearson(thetas, gammas)

                if abs(r_val) > 0.5:
                    direction = "positive" if r_val > 0 else "negative"
                    entry = InsightEntry(
                        id=f"DIS-CC-DEBYE-SURF-{_hash_short(str(pairs))}",
                        domain="Cross-Domain",
                        pattern=(f"Θ_D vs γ_surface: Pearson r = {r_val:.3f} ({direction} correlation)"),
                        lesson=(
                            "Debye temperature (lattice stiffness) and surface "
                            "energy (bond-breaking cost) share a common ancestor: "
                            "the interatomic bond strength. Stiff lattices have "
                            "strong bonds → both high Θ_D and high γ_s. This is "
                            "the same collapse channel viewed from two perspectives."
                        ),
                        implication=(
                            "Surface and thermal properties are different projections of the same cohesive landscape"
                        ),
                        severity=InsightSeverity.STRUCTURAL,
                        pattern_type=PatternType.CROSS_CORRELATION,
                        source="discovered",
                        elements=[p[0] for p in pairs],
                    )
                    insights.append(entry)
                    self.db.add(entry)

        except ImportError:
            pass

        # -- Cohesive energy vs elastic modulus --
        try:
            from closures.materials_science.cohesive_energy import (
                compute_cohesive_energy,
            )
            from closures.materials_science.elastic_moduli import (
                compute_elastic_moduli,
            )

            pairs2: list[tuple[str, float, float]] = []
            for sym, z in [
                ("Cu", 29),
                ("Ag", 47),
                ("Au", 79),
                ("Al", 13),
                ("W", 74),
                ("Mo", 42),
                ("Ni", 28),
                ("Fe", 26),
            ]:
                try:
                    c = compute_cohesive_energy(z, symbol=sym)
                    e = compute_elastic_moduli(c.E_coh_eV, c.r0_A, symbol=sym)
                    if c.E_coh_eV > 0 and e.K_GPa > 0:
                        pairs2.append((sym, c.E_coh_eV, e.K_GPa))
                except Exception:
                    continue

            if len(pairs2) >= 4:
                e_vals = [p[1] for p in pairs2]
                k_vals = [p[2] for p in pairs2]
                r_val2 = _pearson(e_vals, k_vals)

                if abs(r_val2) > 0.5:
                    entry = InsightEntry(
                        id=f"DIS-CC-COH-ELAST-{_hash_short(str(pairs2))}",
                        domain="Cross-Domain",
                        pattern=(f"E_coh vs K_bulk: Pearson r = {r_val2:.3f} — strongly correlated"),
                        lesson=(
                            "Cohesive energy is the depth of the interatomic "
                            "potential well; bulk modulus K is its curvature. "
                            "For the same functional form V(r), deeper wells have "
                            "steeper walls: E_coh and K are not independent — they "
                            "are the 0th and 2nd moments of the same binding curve."
                        ),
                        implication=("Binding depth and curvature are coupled moments of the collapse potential"),
                        severity=InsightSeverity.FUNDAMENTAL,
                        pattern_type=PatternType.CROSS_CORRELATION,
                        source="discovered",
                        elements=[p[0] for p in pairs2],
                    )
                    insights.append(entry)
                    self.db.add(entry)

        except ImportError:
            pass

        return insights

    def discover_universality_signatures(self) -> list[InsightEntry]:
        """Detect RCFT exponents appearing across domains."""
        insights: list[InsightEntry] = []

        # Collect all ω_eff values across closures for shared elements
        try:
            from closures.materials_science import (
                compute_cohesive_energy,
                compute_magnetic_properties,
                compute_surface_catalysis,
            )

            multi_omega: dict[str, dict[str, float]] = {}
            for sym, z in [("Fe", 26), ("Cu", 29), ("Ni", 28), ("Al", 13), ("Au", 79), ("Pt", 78), ("W", 74)]:
                omegas: dict[str, float] = {}
                try:
                    rc = compute_cohesive_energy(z, symbol=sym)
                    omegas["cohesive"] = rc.omega_eff
                except Exception:
                    pass
                try:
                    rm = compute_magnetic_properties(0, symbol=sym, T_K=300)
                    omegas["magnetic"] = rm.omega_eff
                except Exception:
                    pass
                try:
                    rs = compute_surface_catalysis(0, symbol=sym)
                    omegas["surface"] = rs.omega_eff
                except Exception:
                    pass
                if len(omegas) >= 2:
                    multi_omega[sym] = omegas

            if multi_omega:
                # Find elements where ω is high in one domain but low in another
                divergent: list[str] = []
                for sym, omegas in multi_omega.items():
                    vals = list(omegas.values())
                    spread = max(vals) - min(vals)
                    if spread > 0.30:
                        divergent.append(sym)

                if divergent:
                    entry = InsightEntry(
                        id=f"DIS-UNI-OMEGA-SPREAD-{_hash_short(str(divergent))}",
                        domain="Cross-Domain",
                        pattern=(f"Elements with divergent ω across closures: {', '.join(divergent)}"),
                        lesson=(
                            "When ω_eff is high for one property but low for another "
                            "in the same element, the model captures one collapse "
                            "channel but not another. This is NOT a failure — it maps "
                            "which physics (band, exchange, surface) each model "
                            "includes. The divergence IS the lesson."
                        ),
                        implication=("Closure-specific ω divergence maps model completeness"),
                        severity=InsightSeverity.FUNDAMENTAL,
                        pattern_type=PatternType.UNIVERSALITY,
                        source="discovered",
                        elements=divergent,
                    )
                    insights.append(entry)
                    self.db.add(entry)

        except ImportError:
            pass

        return insights

    def discover_all(self) -> list[InsightEntry]:
        """Run all discovery passes. Returns newly discovered insights."""
        all_new: list[InsightEntry] = []
        all_new.extend(self.discover_periodic_trends())
        all_new.extend(self.discover_regime_boundaries())
        all_new.extend(self.discover_cross_correlations())
        all_new.extend(self.discover_universality_signatures())
        return all_new

    # -- Reports ------------------------------------------------------------

    def show_startup_insight(self, *, seed: int | None = None) -> str:
        """Return a formatted startup insight for terminal display.

        Selects a random insight from the database (deterministic if seed given).
        """
        if not self.db.entries:
            return ""

        rng = random.Random(seed if seed is not None else _daily_seed())
        entry = rng.choice(self.db.entries)

        width = 72
        sep = "═" * width
        lines = [
            "",
            f"  ╔{sep}╗",
            f"  ║{'UMCP — Lesson of the Day':^{width}}║",
            f"  ╠{sep}╣",
        ]
        # Wrap pattern
        for line in textwrap.wrap(f"  {entry.pattern}", width - 4):
            lines.append(f"  ║ {line:<{width - 2}} ║")
        lines.append(f"  ╟{'─' * width}╢")
        # Wrap lesson
        for line in textwrap.wrap(entry.lesson, width - 4):
            lines.append(f"  ║ {line:<{width - 2}} ║")
        lines.append(f"  ╟{'─' * width}╢")
        # Implication
        imp_text = f"→ {entry.implication}"
        for line in textwrap.wrap(imp_text, width - 4):
            lines.append(f"  ║ {line:<{width - 2}} ║")
        # Source + domain
        tag = f"[{entry.id}]  domain: {entry.domain}  source: {entry.source}"
        lines.append(f"  ╟{'─' * width}╢")
        lines.append(f"  ║ {tag:<{width - 2}} ║")
        lines.append(f"  ╚{sep}╝")
        lines.append("")

        return "\n".join(lines)

    def full_report(self) -> str:
        """Return a full-text report of all insights grouped by domain."""
        if not self.db.entries:
            return "No insights in database."

        lines: list[str] = []
        lines.append("=" * 80)
        lines.append("  UMCP MATERIALS-SCIENCE — LESSONS-LEARNED REPORT")
        lines.append(f"  {self.db.count()} insights across {len(self.db.domains())} domains")
        lines.append("=" * 80)

        for domain in self.db.domains():
            entries = self.db.query(domain=domain)
            lines.append("")
            lines.append(f"  ┌── {domain} ({len(entries)} insights) ──")
            for e in entries:
                lines.append("  │")
                lines.append(f"  │  [{e.id}]  {e.severity.value} / {e.pattern_type.value}")
                lines.append(f"  │  Pattern: {e.pattern}")
                for wl in textwrap.wrap(e.lesson, 68):
                    lines.append(f"  │    {wl}")
                lines.append(f"  │  → {e.implication}")
                if e.elements:
                    lines.append(f"  │  Elements: {', '.join(e.elements)}")
            lines.append(f"  └{'─' * 50}")

        lines.append("")
        lines.append("  " + "─" * 50)
        lines.append('  "What returns through collapse is real."')
        lines.append("  Every anomalous ω_eff is not a failure — it is a signal")
        lines.append("  pointing toward physics the model has not yet learned.")
        lines.append("  " + "─" * 50)
        lines.append("")

        return "\n".join(lines)

    def summary_stats(self) -> dict[str, Any]:
        """Return summary statistics for dashboard / API consumption."""
        by_domain: dict[str, int] = {}
        by_severity: dict[str, int] = {}
        by_source: dict[str, int] = {}

        for e in self.db.entries:
            by_domain[e.domain] = by_domain.get(e.domain, 0) + 1
            by_severity[e.severity.value] = by_severity.get(e.severity.value, 0) + 1
            by_source[e.source] = by_source.get(e.source, 0) + 1

        return {
            "total_insights": self.db.count(),
            "domains": by_domain,
            "by_severity": by_severity,
            "by_source": by_source,
            "domain_list": self.db.domains(),
        }

    def save(self, path: Path | None = None) -> None:
        """Persist database to YAML."""
        self.db.save_yaml(path)

    # -- Philosophical output -----------------------------------------------

    @staticmethod
    def philosophy() -> str:
        """Return the guiding philosophy of the insight engine."""
        return textwrap.dedent("""\
        ┌─────────────────────────────────────────────────────────┐
        │  The Insight Engine's Philosophy                        │
        │                                                         │
        │  Science is not a warehouse of facts but a living       │
        │  process of collapse and return.  Each model — Friedel, │
        │  BCS, broken-bond — illuminates one collapse channel    │
        │  while leaving others in shadow.                        │
        │                                                         │
        │  ω_eff measures the shadow.  When it's small, the       │
        │  model captures the essential physics.  When it's       │
        │  large, the model is teaching us what it cannot see.    │
        │                                                         │
        │  The "Anomalous" regime is not failure — it is the      │
        │  most honest scientific statement a model can make:     │
        │  "There is physics here I do not contain."              │
        │                                                         │
        │  Every lesson in this database was discovered, not      │
        │  assumed.  Every pattern emerged from computation, not  │
        │  from authority.  The engine learns because it measures │
        │  what returns through collapse — and what doesn't.      │
        │                                                         │
        │  What returns through collapse is real.                 │
        └─────────────────────────────────────────────────────────┘
        """)


# ---------------------------------------------------------------------------
#  Utility helpers
# ---------------------------------------------------------------------------


def _hash_short(s: str) -> str:
    """Short deterministic hash for deduplication IDs."""
    return hashlib.sha256(s.encode()).hexdigest()[:8]


def _daily_seed() -> int:
    """Deterministic seed from date so the same insight shows all day."""
    import datetime

    d = datetime.date.today()
    return d.year * 10000 + d.month * 100 + d.day


def _pearson(x: list[float], y: list[float]) -> float:
    """Pearson correlation coefficient."""
    n = len(x)
    if n < 2:
        return 0.0
    mx = sum(x) / n
    my = sum(y) / n
    sx = math.sqrt(sum((xi - mx) ** 2 for xi in x) / n)
    sy = math.sqrt(sum((yi - my) ** 2 for yi in y) / n)
    if sx == 0 or sy == 0:
        return 0.0
    return sum((xi - mx) * (yi - my) for xi, yi in zip(x, y, strict=False)) / (n * sx * sy)

"""
UMCP Novel Outputs Module - Rich Export Formats and Visualizations

This module provides diverse output formats for UMCP validation results,
enabling integration with CI/CD pipelines, documentation, and reporting systems.

Available Generators:
  - SVG Badges: Dynamic status badges (shields.io style)
  - Mermaid Diagrams: Provenance graphs, regime flow charts
  - ASCII Art: Terminal-friendly regime gauges and sparklines
  - Markdown Reports: Publication-ready summaries
  - LaTeX Tables: Academic paper integration
  - JSON-LD: Linked data for semantic web
  - SARIF: Static analysis results format (GitHub integration)
  - JUnit XML: Test runner compatibility
  - HTML Cards: Embeddable status widgets

Cross-references:
  - src/umcp/validator.py (validation engine)
  - src/umcp/api_umcp.py (REST endpoints)
  - PUBLICATION_INFRASTRUCTURE.md (publication formats)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, ClassVar

try:
    from . import __version__
except ImportError:
    __version__ = "1.5.0"


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class RegimeState:
    """Regime classification state."""

    omega: float  # Overlap fraction
    F: float  # Freshness
    S: float  # Seam residual
    C: float  # Curvature
    regime: str  # STABLE, WATCH, COLLAPSE, CRITICAL

    @classmethod
    def from_values(cls, omega: float, F: float, S: float, C: float) -> RegimeState:
        """Create with automatic regime classification."""
        if abs(S) > 0.01:
            regime = "CRITICAL"
        elif omega < 0.1 or omega > 0.9:
            regime = "COLLAPSE"
        elif 0.3 <= omega <= 0.7:
            regime = "STABLE"
        else:
            regime = "WATCH"
        return cls(omega=omega, F=F, S=S, C=C, regime=regime)


@dataclass
class ValidationSummary:
    """Validation result summary for output generation."""

    status: str  # CONFORMANT, NONCONFORMANT, NON_EVALUABLE
    errors: int
    warnings: int
    casepack_id: str | None = None
    contract_id: str | None = None
    run_id: str | None = None
    timestamp: str | None = None
    regime: RegimeState | None = None
    hash: str | None = None


# ============================================================================
# SVG Badge Generator (Shields.io Style)
# ============================================================================


class BadgeGenerator:
    """Generate SVG badges for validation status."""

    # Color schemes
    COLORS: ClassVar[dict[str, str]] = {
        "CONFORMANT": "#4c1",  # Green
        "NONCONFORMANT": "#e05d44",  # Red
        "NON_EVALUABLE": "#9f9f9f",  # Gray
        "STABLE": "#2e7d32",  # Dark green
        "WATCH": "#f9a825",  # Amber
        "COLLAPSE": "#d84315",  # Deep orange
        "CRITICAL": "#b71c1c",  # Dark red
    }

    ICONS: ClassVar[dict[str, str]] = {
        "CONFORMANT": "✓",
        "NONCONFORMANT": "✗",
        "NON_EVALUABLE": "?",
        "STABLE": "◉",
        "WATCH": "◐",
        "COLLAPSE": "◯",
        "CRITICAL": "⚠",
    }

    @classmethod
    def status_badge(
        cls,
        status: str,
        label: str = "UMCP",
        style: str = "flat",
    ) -> str:
        """Generate validation status badge SVG."""
        color = cls.COLORS.get(status, "#9f9f9f")
        icon = cls.ICONS.get(status, "")

        # Calculate widths
        label_width = len(label) * 7 + 10
        status_width = len(status) * 7 + 20
        total_width = label_width + status_width

        if style == "flat":
            return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{total_width}" height="20">
  <linearGradient id="b" x2="0" y2="100%">
    <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
    <stop offset="1" stop-opacity=".1"/>
  </linearGradient>
  <clipPath id="a">
    <rect width="{total_width}" height="20" rx="3" fill="#fff"/>
  </clipPath>
  <g clip-path="url(#a)">
    <rect width="{label_width}" height="20" fill="#555"/>
    <rect x="{label_width}" width="{status_width}" height="20" fill="{color}"/>
    <rect width="{total_width}" height="20" fill="url(#b)"/>
  </g>
  <g fill="#fff" text-anchor="middle" font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="11">
    <text x="{label_width/2}" y="15" fill="#010101" fill-opacity=".3">{label}</text>
    <text x="{label_width/2}" y="14">{label}</text>
    <text x="{label_width + status_width/2}" y="15" fill="#010101" fill-opacity=".3">{icon} {status}</text>
    <text x="{label_width + status_width/2}" y="14">{icon} {status}</text>
  </g>
</svg>"""
        else:
            # Flat-square style
            return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{total_width}" height="20">
  <rect width="{label_width}" height="20" fill="#555"/>
  <rect x="{label_width}" width="{status_width}" height="20" fill="{color}"/>
  <g fill="#fff" text-anchor="middle" font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="11">
    <text x="{label_width/2}" y="14">{label}</text>
    <text x="{label_width + status_width/2}" y="14">{icon} {status}</text>
  </g>
</svg>"""

    @classmethod
    def regime_badge(cls, regime: RegimeState) -> str:
        """Generate regime state badge."""
        color = cls.COLORS.get(regime.regime, "#9f9f9f")
        icon = cls.ICONS.get(regime.regime, "")

        # Mini gauge representation
        omega_pct = int(regime.omega * 100)

        return f"""<svg xmlns="http://www.w3.org/2000/svg" width="180" height="20">
  <rect width="50" height="20" fill="#555"/>
  <rect x="50" width="60" height="20" fill="{color}"/>
  <rect x="110" width="70" height="20" fill="#333"/>
  <g fill="#fff" text-anchor="middle" font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="11">
    <text x="25" y="14">Regime</text>
    <text x="80" y="14">{icon} {regime.regime}</text>
    <text x="145" y="14">ω={omega_pct}%</text>
  </g>
</svg>"""

    @classmethod
    def tests_badge(cls, passed: int, total: int) -> str:
        """Generate tests passed badge."""
        pct = (passed / total * 100) if total > 0 else 0
        color = "#4c1" if pct == 100 else "#f9a825" if pct >= 80 else "#e05d44"

        return f"""<svg xmlns="http://www.w3.org/2000/svg" width="120" height="20">
  <rect width="50" height="20" fill="#555"/>
  <rect x="50" width="70" height="20" fill="{color}"/>
  <g fill="#fff" text-anchor="middle" font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="11">
    <text x="25" y="14">tests</text>
    <text x="85" y="14">{passed}/{total} ({pct:.0f}%)</text>
  </g>
</svg>"""


# ============================================================================
# Mermaid Diagram Generator
# ============================================================================


class MermaidGenerator:
    """Generate Mermaid diagrams for UMCP structures."""

    @staticmethod
    def provenance_graph(
        casepack_id: str,
        contract_id: str,
        closures: list[str],
        artifacts: list[dict[str, str]],
    ) -> str:
        """Generate provenance flowchart."""
        lines = [
            "```mermaid",
            "flowchart TD",
            f'    CP["{casepack_id}"]',
            f'    CT["{contract_id}"]',
            "    CP --> CT",
        ]

        # Closures
        for i, closure in enumerate(closures):
            lines.append(f'    CL{i}["{closure}"]')
            lines.append(f"    CT --> CL{i}")

        # Artifacts
        for i, artifact in enumerate(artifacts):
            role = artifact.get("role", "file")
            path = artifact.get("path", f"artifact_{i}")
            icon = {"data": "📊", "receipt": "🧾", "log": "📋", "seam": "🔗"}.get(role, "📄")
            lines.append(f'    A{i}["{icon} {path}"]')
            lines.append(f"    CL{min(i, len(closures)-1)} --> A{i}")

        lines.append("```")
        return "\n".join(lines)

    @staticmethod
    def regime_state_diagram() -> str:
        """Generate regime state transition diagram."""
        return """```mermaid
stateDiagram-v2
    [*] --> STABLE : ω ∈ [0.3, 0.7]
    STABLE --> WATCH : ω drift
    WATCH --> STABLE : recovery
    WATCH --> COLLAPSE : ω < 0.1 or > 0.9
    COLLAPSE --> [*] : terminate
    
    STABLE --> CRITICAL : |s| > 0.01
    WATCH --> CRITICAL : |s| > 0.01
    CRITICAL --> [*] : abort
    
    note right of STABLE
        Optimal operating regime
        Near-equilibrium conditions
    end note
    
    note right of COLLAPSE
        Near-boundary regime
        Saturation effects
    end note
```"""

    @staticmethod
    def validation_sequence(steps: list[tuple[str, str, str]]) -> str:
        """Generate validation sequence diagram.

        Args:
            steps: List of (actor, action, result) tuples
        """
        lines = [
            "```mermaid",
            "sequenceDiagram",
            "    participant U as User",
            "    participant V as Validator",
            "    participant C as Contract",
            "    participant K as Kernel",
        ]

        for actor, action, result in steps:
            lines.append(f"    {actor}->>+{action}: validate")
            lines.append(f"    {action}-->>-{actor}: {result}")

        lines.append("```")
        return "\n".join(lines)


# ============================================================================
# ASCII Art Generator (Terminal-Friendly)
# ============================================================================


class ASCIIGenerator:
    """Generate ASCII art for terminal output."""

    @staticmethod
    def regime_gauge(omega: float, width: int = 40) -> str:
        """Generate ASCII gauge for omega value."""
        # Regime zones
        zones = [
            (0.0, 0.1, "COLLAPSE", "░"),
            (0.1, 0.3, "WATCH", "▒"),
            (0.3, 0.7, "STABLE", "█"),
            (0.7, 0.9, "WATCH", "▒"),
            (0.9, 1.0, "COLLAPSE", "░"),
        ]

        # Build gauge
        gauge = ""
        for start, end, _name, char in zones:
            segment_width = int((end - start) * width)
            gauge += char * segment_width

        # Add pointer
        pointer_pos = int(omega * width)
        pointer_pos = max(0, min(pointer_pos, width - 1))

        lines = [
            "┌" + "─" * width + "┐",
            "│" + gauge + "│",
            "│" + " " * pointer_pos + "▼" + " " * (width - pointer_pos - 1) + "│",
            "│" + " " * pointer_pos + f"ω={omega:.3f}" + " " * max(0, width - pointer_pos - 10) + "│",
            "└" + "─" * width + "┘",
            " 0.0       COLLAPSE  WATCH  STABLE  WATCH  COLLAPSE       1.0",
        ]
        return "\n".join(lines)

    @staticmethod
    def sparkline(values: list[float], width: int = 20) -> str:
        """Generate ASCII sparkline for time series."""
        if not values:
            return "─" * width

        # Normalize values
        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val if max_val != min_val else 1.0

        # Characters for different heights
        chars = " ▁▂▃▄▅▆▇█"

        # Sample if too many values
        if len(values) > width:
            step = len(values) / width
            sampled = [values[int(i * step)] for i in range(width)]
        else:
            sampled = values

        # Build sparkline
        spark = ""
        for v in sampled:
            normalized = (v - min_val) / range_val
            char_idx = int(normalized * (len(chars) - 1))
            spark += chars[char_idx]

        return spark

    @staticmethod
    def validation_box(summary: ValidationSummary) -> str:
        """Generate ASCII box with validation summary."""
        status_icon = {"CONFORMANT": "✓", "NONCONFORMANT": "✗", "NON_EVALUABLE": "?"}.get(summary.status, "?")

        lines = [
            "╔════════════════════════════════════════════╗",
            "║  UMCP Validation Result                    ║",
            "╠════════════════════════════════════════════╣",
            f"║  Status: {status_icon} {summary.status:<30} ║",
            f"║  Errors: {summary.errors:<34} ║",
            f"║  Warnings: {summary.warnings:<32} ║",
        ]

        if summary.casepack_id:
            lines.append(f"║  Casepack: {summary.casepack_id:<32} ║")
        if summary.contract_id:
            lines.append(f"║  Contract: {summary.contract_id:<32} ║")
        if summary.regime:
            lines.append(f"║  Regime: {summary.regime.regime:<34} ║")
            lines.append(f"║    ω = {summary.regime.omega:.6f}                          ║")

        lines.append("╚════════════════════════════════════════════╝")
        return "\n".join(lines)

    @staticmethod
    def kernel_hud(omega: float, F: float, kappa: float, S: float) -> str:
        """Generate Heads-Up Display for kernel invariants."""
        return f"""
┌─────────────────────────────────────────────────────────────┐
│                    UMCP KERNEL HUD                          │
├─────────────────────────────────────────────────────────────┤
│   ω (Overlap)      │ F (Freshness)     │ κ (Curvature)      │
│   {omega:>12.6f}     │ {F:>12.6f}      │ {kappa:>12.6f}      │
├─────────────────────────────────────────────────────────────┤
│   Identity Check: F ≈ 1-ω  │  |{F:.6f} - {1-omega:.6f}| = {abs(F - (1-omega)):.2e}     │
│   Seam Residual:  |s| = {abs(S):.6f}  {"✓ PASS" if abs(S) <= 0.01 else "✗ FAIL":<8}                       │
└─────────────────────────────────────────────────────────────┘
"""


# ============================================================================
# Markdown Report Generator
# ============================================================================


class MarkdownGenerator:
    """Generate Markdown reports for documentation."""

    @staticmethod
    def validation_report(
        summary: ValidationSummary,
        details: dict[str, Any] | None = None,
    ) -> str:
        """Generate full validation report in Markdown."""
        status_emoji = {"CONFORMANT": "✅", "NONCONFORMANT": "❌", "NON_EVALUABLE": "⚠️"}.get(summary.status, "❓")

        lines = [
            "# UMCP Validation Report",
            "",
            f"**Generated**: {summary.timestamp or datetime.now(UTC).isoformat()}",
            f"**Validator**: UMCP v{__version__}",
            "",
            "## Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Status | {status_emoji} {summary.status} |",
            f"| Errors | {summary.errors} |",
            f"| Warnings | {summary.warnings} |",
        ]

        if summary.casepack_id:
            lines.append(f"| Casepack | `{summary.casepack_id}` |")
        if summary.contract_id:
            lines.append(f"| Contract | `{summary.contract_id}` |")
        if summary.hash:
            lines.append(f"| Hash | `{summary.hash[:16]}...` |")

        if summary.regime:
            lines.extend(
                [
                    "",
                    "## Regime Classification",
                    "",
                    "| Invariant | Value | Status |",
                    "|-----------|-------|--------|",
                    f"| ω (Overlap) | {summary.regime.omega:.6f} | {'✅' if 0.3 <= summary.regime.omega <= 0.7 else '⚠️'} |",
                    f"| F (Freshness) | {summary.regime.F:.6f} | {'✅' if 0.3 <= summary.regime.F <= 0.7 else '⚠️'} |",
                    f"| S (Seam) | {summary.regime.S:.6f} | {'✅' if abs(summary.regime.S) <= 0.01 else '❌'} |",
                    f"| κ (Curvature) | {summary.regime.C:.6f} | ℹ️ |",
                    f"| **Regime** | **{summary.regime.regime}** | |",
                ]
            )

        if details:
            lines.extend(
                [
                    "",
                    "## Details",
                    "",
                    "```json",
                    json.dumps(details, indent=2, default=str),
                    "```",
                ]
            )

        return "\n".join(lines)

    @staticmethod
    def changelog_entry(
        version: str,
        changes: list[tuple[str, str]],  # (type, description)
    ) -> str:
        """Generate changelog entry."""
        date = datetime.now(UTC).strftime("%Y-%m-%d")
        lines = [
            f"## [{version}] - {date}",
            "",
        ]

        # Group by type
        grouped: dict[str, list[str]] = {}
        for change_type, desc in changes:
            if change_type not in grouped:
                grouped[change_type] = []
            grouped[change_type].append(desc)

        type_headers = {
            "added": "### Added",
            "changed": "### Changed",
            "fixed": "### Fixed",
            "removed": "### Removed",
            "security": "### Security",
        }

        for change_type, descriptions in grouped.items():
            header = type_headers.get(change_type, f"### {change_type.title()}")
            lines.append(header)
            for desc in descriptions:
                lines.append(f"- {desc}")
            lines.append("")

        return "\n".join(lines)


# ============================================================================
# LaTeX Table Generator
# ============================================================================


class LaTeXGenerator:
    """Generate LaTeX tables for academic papers."""

    @staticmethod
    def invariants_table(
        rows: list[dict[str, Any]],
        caption: str = "UMCP Kernel Invariants",
        label: str = "tab:invariants",
    ) -> str:
        """Generate LaTeX table for kernel invariants."""
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{" + caption + "}",
            r"\label{" + label + "}",
            r"\begin{tabular}{lccccc}",
            r"\toprule",
            r"$t$ & $\omega$ & $F$ & $\kappa$ & $s$ & Regime \\",
            r"\midrule",
        ]

        for row in rows:
            t = row.get("t", 0)
            omega = row.get("omega", 0)
            F = row.get("F", 0)
            kappa = row.get("kappa", 0)
            s = row.get("s", row.get("S", 0))
            regime = row.get("regime", "UNKNOWN")

            lines.append(f"{t} & {omega:.4f} & {F:.4f} & {kappa:.4f} & {s:.6f} & {regime} \\\\")

        lines.extend(
            [
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table}",
            ]
        )

        return "\n".join(lines)

    @staticmethod
    def regime_distribution(
        counts: dict[str, int],
        caption: str = "Regime Distribution",
    ) -> str:
        """Generate LaTeX table for regime distribution."""
        total = sum(counts.values())

        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{" + caption + "}",
            r"\begin{tabular}{lrr}",
            r"\toprule",
            r"Regime & Count & Percentage \\",
            r"\midrule",
        ]

        for regime, count in sorted(counts.items()):
            pct = (count / total * 100) if total > 0 else 0
            lines.append(f"{regime} & {count} & {pct:.1f}\\% \\\\")

        lines.extend(
            [
                r"\midrule",
                f"Total & {total} & 100.0\\% \\\\",
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table}",
            ]
        )

        return "\n".join(lines)


# ============================================================================
# JSON-LD Generator (Linked Data)
# ============================================================================


class JSONLDGenerator:
    """Generate JSON-LD for semantic web integration."""

    @staticmethod
    def validation_result(summary: ValidationSummary) -> dict[str, Any]:
        """Generate JSON-LD for validation result."""
        return {
            "@context": {
                "@vocab": "https://umcp.io/schema/",
                "schema": "http://schema.org/",
                "prov": "http://www.w3.org/ns/prov#",
            },
            "@type": "ValidationResult",
            "@id": f"urn:umcp:validation:{summary.run_id or hashlib.sha256(str(summary).encode()).hexdigest()[:8]}",
            "status": summary.status,
            "errors": summary.errors,
            "warnings": summary.warnings,
            "dateCreated": summary.timestamp or datetime.now(UTC).isoformat(),
            "prov:wasGeneratedBy": {
                "@type": "prov:Activity",
                "prov:used": {
                    "@type": "Contract",
                    "@id": f"urn:umcp:contract:{summary.contract_id}",
                }
                if summary.contract_id
                else None,
            },
            "schema:isPartOf": {
                "@type": "CasePack",
                "@id": f"urn:umcp:casepack:{summary.casepack_id}",
            }
            if summary.casepack_id
            else None,
        }


# ============================================================================
# SARIF Generator (GitHub Integration)
# ============================================================================


class SARIFGenerator:
    """Generate SARIF for GitHub code scanning integration."""

    @staticmethod
    def from_validation(
        summary: ValidationSummary,
        issues: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Generate SARIF report from validation issues."""
        results = []

        for issue in issues:
            result = {
                "ruleId": issue.get("code", "UMCP001"),
                "level": "error" if issue.get("severity") == "ERROR" else "warning",
                "message": {
                    "text": issue.get("message", "Validation issue"),
                },
                "locations": [
                    {
                        "physicalLocation": {
                            "artifactLocation": {
                                "uri": issue.get("file", "unknown"),
                            },
                            "region": {
                                "startLine": issue.get("line", 1),
                            },
                        },
                    }
                ],
            }
            results.append(result)

        return {
            "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
            "version": "2.1.0",
            "runs": [
                {
                    "tool": {
                        "driver": {
                            "name": "UMCP Validator",
                            "version": __version__,
                            "informationUri": "https://github.com/calebpruett927/UMCP-Metadata-Runnable-Code",
                            "rules": [
                                {
                                    "id": "UMCP001",
                                    "name": "ValidationError",
                                    "shortDescription": {"text": "UMCP validation error"},
                                },
                            ],
                        },
                    },
                    "results": results,
                }
            ],
        }


# ============================================================================
# JUnit XML Generator (CI Integration)
# ============================================================================


class JUnitGenerator:
    """Generate JUnit XML for CI/CD integration."""

    @staticmethod
    def from_validation(
        summary: ValidationSummary,
        test_cases: list[dict[str, Any]] | None = None,
    ) -> str:
        """Generate JUnit XML from validation result."""
        timestamp = summary.timestamp or datetime.now(UTC).isoformat()

        # Default test cases if not provided
        if test_cases is None:
            test_cases = [
                {
                    "name": "validation_status",
                    "passed": summary.status == "CONFORMANT",
                    "message": f"Status: {summary.status}",
                },
            ]

        tests = len(test_cases)
        failures = sum(1 for tc in test_cases if not tc.get("passed", True))

        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<testsuite name="UMCP Validation" tests="{tests}" failures="{failures}" timestamp="{timestamp}">',
        ]

        for tc in test_cases:
            name = tc.get("name", "test")
            passed = tc.get("passed", True)
            message = tc.get("message", "")

            if passed:
                lines.append(f'  <testcase name="{name}" classname="umcp.validation"/>')
            else:
                lines.append(f'  <testcase name="{name}" classname="umcp.validation">')
                lines.append(f'    <failure message="{message}"/>')
                lines.append("  </testcase>")

        lines.append("</testsuite>")
        return "\n".join(lines)


# ============================================================================
# HTML Widget Generator
# ============================================================================


class HTMLGenerator:
    """Generate HTML widgets for embedding."""

    @staticmethod
    def status_card(summary: ValidationSummary) -> str:
        """Generate embeddable HTML status card."""
        status_color = {
            "CONFORMANT": "#4caf50",
            "NONCONFORMANT": "#f44336",
            "NON_EVALUABLE": "#9e9e9e",
        }.get(summary.status, "#9e9e9e")

        regime_info = ""
        if summary.regime:
            regime_color = {
                "STABLE": "#2e7d32",
                "WATCH": "#f9a825",
                "COLLAPSE": "#d84315",
                "CRITICAL": "#b71c1c",
            }.get(summary.regime.regime, "#666")

            regime_info = f"""
            <div style="margin-top:10px; padding:8px; background:#f5f5f5; border-radius:4px;">
                <span style="font-size:12px; color:#666;">Regime:</span>
                <span style="background:{regime_color}; color:white; padding:2px 8px; border-radius:4px; font-size:12px; margin-left:8px;">
                    {summary.regime.regime}
                </span>
                <span style="font-size:11px; color:#888; margin-left:10px;">ω={summary.regime.omega:.4f}</span>
            </div>
            """

        return f"""
<div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; max-width:300px; border:1px solid #ddd; border-radius:8px; overflow:hidden; box-shadow:0 2px 4px rgba(0,0,0,0.1);">
    <div style="background:{status_color}; color:white; padding:12px 16px;">
        <div style="font-size:14px; font-weight:600;">UMCP Validation</div>
        <div style="font-size:20px; font-weight:700; margin-top:4px;">{summary.status}</div>
    </div>
    <div style="padding:12px 16px;">
        <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
            <span style="color:#666;">Errors</span>
            <span style="font-weight:600; color:{'#f44336' if summary.errors > 0 else '#4caf50'};">{summary.errors}</span>
        </div>
        <div style="display:flex; justify-content:space-between;">
            <span style="color:#666;">Warnings</span>
            <span style="font-weight:600; color:{'#ff9800' if summary.warnings > 0 else '#4caf50'};">{summary.warnings}</span>
        </div>
        {regime_info}
        <div style="margin-top:12px; font-size:10px; color:#999;">
            {summary.timestamp or datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")}
        </div>
    </div>
</div>
"""

    @staticmethod
    def regime_gauge_html(omega: float) -> str:
        """Generate HTML gauge for omega value."""
        # Determine regime and color
        if omega < 0.1 or omega > 0.9:
            regime, color = "COLLAPSE", "#d84315"
        elif 0.3 <= omega <= 0.7:
            regime, color = "STABLE", "#4caf50"
        else:
            regime, color = "WATCH", "#ff9800"

        pct = omega * 100

        return f"""
<div style="font-family:sans-serif; max-width:200px;">
    <div style="background:#eee; border-radius:4px; height:20px; position:relative; overflow:hidden;">
        <div style="background:linear-gradient(to right, #d84315 10%, #ff9800 30%, #4caf50 50%, #ff9800 70%, #d84315 90%); height:100%; opacity:0.3;"></div>
        <div style="position:absolute; left:{pct}%; top:0; width:3px; height:100%; background:{color}; transform:translateX(-50%);"></div>
    </div>
    <div style="display:flex; justify-content:space-between; font-size:10px; color:#666; margin-top:2px;">
        <span>0</span>
        <span style="color:{color}; font-weight:bold;">ω={omega:.3f} ({regime})</span>
        <span>1</span>
    </div>
</div>
"""


# ============================================================================
# Convenience Functions
# ============================================================================


def generate_all_badges(summary: ValidationSummary) -> dict[str, str]:
    """Generate all available badges for a validation result."""
    badges = {
        "status": BadgeGenerator.status_badge(summary.status),
        "status_flat_square": BadgeGenerator.status_badge(summary.status, style="flat-square"),
    }

    if summary.regime:
        badges["regime"] = BadgeGenerator.regime_badge(summary.regime)

    return badges


def generate_full_report(
    summary: ValidationSummary,
    details: dict[str, Any] | None = None,
    format: str = "markdown",
) -> str:
    """Generate full report in specified format."""
    if format == "markdown":
        return MarkdownGenerator.validation_report(summary, details)
    elif format == "html":
        return HTMLGenerator.status_card(summary)
    elif format == "ascii":
        return ASCIIGenerator.validation_box(summary)
    elif format == "junit":
        return JUnitGenerator.from_validation(summary)
    elif format == "json-ld":
        return json.dumps(JSONLDGenerator.validation_result(summary), indent=2)
    else:
        raise ValueError(f"Unknown format: {format}")


# ============================================================================
# CLI Integration
# ============================================================================


def main() -> None:
    """Demo the output generators."""
    # Create sample data
    regime = RegimeState.from_values(omega=0.45, F=0.55, S=0.002, C=0.001)
    summary = ValidationSummary(
        status="CONFORMANT",
        errors=0,
        warnings=2,
        casepack_id="UMCP-REF-E2E-0001",
        contract_id="UMA.INTSTACK.v1",
        run_id="demo-run-001",
        timestamp=datetime.now(UTC).isoformat(),
        regime=regime,
        hash="abc123def456",
    )

    print("=" * 60)
    print("UMCP Output Generators Demo")
    print("=" * 60)

    print("\n### ASCII Validation Box ###")
    print(ASCIIGenerator.validation_box(summary))

    print("\n### ASCII Regime Gauge ###")
    print(ASCIIGenerator.regime_gauge(regime.omega))

    print("\n### ASCII Kernel HUD ###")
    print(ASCIIGenerator.kernel_hud(regime.omega, regime.F, 0.001, regime.S))

    print("\n### Sparkline Demo ###")
    values = [0.2, 0.3, 0.5, 0.7, 0.6, 0.8, 0.9, 0.7, 0.5, 0.4]
    print(f"Values: {values}")
    print(f"Sparkline: {ASCIIGenerator.sparkline(values)}")

    print("\n### Markdown Report Preview ###")
    md = MarkdownGenerator.validation_report(summary)
    print(md[:500] + "...")

    print("\n### Mermaid State Diagram ###")
    print(MermaidGenerator.regime_state_diagram())

    print("\n### LaTeX Table ###")
    rows = [
        {"t": 0, "omega": 0.45, "F": 0.55, "kappa": 0.001, "s": 0.002, "regime": "STABLE"},
        {"t": 1, "omega": 0.52, "F": 0.48, "kappa": 0.002, "s": 0.001, "regime": "STABLE"},
    ]
    print(LaTeXGenerator.invariants_table(rows))


if __name__ == "__main__":
    main()

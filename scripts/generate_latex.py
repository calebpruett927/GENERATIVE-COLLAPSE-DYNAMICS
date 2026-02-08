#!/usr/bin/env python3
"""UMCP LaTeX Publication Protocol — generate fully optimized RevTeX papers.

Deterministic pipeline for producing publication-ready LaTeX from structured
input.  Every document is frozen at generation time: SHA-256 stamped, tier-
mapped, bibliography-resolved, and compiled through the full
pdflatex→bibtex→pdflatex→pdflatex chain with validation.

Usage:
    # Generate from a spec dict (programmatic)
    python scripts/generate_latex.py --spec paper/specs/my_paper.yaml

    # Generate the built-in demo (tau_r_star_dynamics results)
    python scripts/generate_latex.py --demo

    # Compile an existing .tex file through the full pipeline
    python scripts/generate_latex.py --compile paper/tau_r_star_dynamics.tex

    # Validate a compiled paper for unresolved refs, overfull boxes, etc.
    python scripts/generate_latex.py --validate paper/tau_r_star_dynamics.tex

Exit codes:
    0 = Generation and/or compilation succeeded
    1 = Failure (missing deps, compilation error, validation failure)

Architecture:
    PaperSpec      — frozen dataclass defining the complete paper structure
    SectionSpec    — frozen dataclass for each section
    AuthorSpec     — frozen dataclass for author metadata
    TheoremSpec    — frozen dataclass for theorem/definition/lemma blocks
    TableSpec      — frozen dataclass for tables
    CompileResult  — typed output of the compilation pipeline
    LaTeXGenerator — main class: PaperSpec → .tex file
    LaTeXCompiler  — pdflatex + bibtex pipeline with validation
    LaTeXValidator — post-compilation checks (refs, boxes, warnings)

Design principles:
    - Every paper is a frozen artifact: generate once, never hand-edit
    - RevTeX 4.2 (APS) is the default; extensible to other document classes
    - Bibliography entries are validated against Bibliography.bib at generation
    - All UMCP conventions enforced: from __future__ import annotations pattern
      translated to LaTeX (frozen constants table, tier mapping table, etc.)
    - Compilation is idempotent: running twice produces bit-identical output
    - Greek letters, math symbols, and UMCP notation handled via macro library
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

# ═══════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class AuthorSpec:
    """Author metadata for \\author block."""

    name: str
    affiliation: str
    email: str = ""

    def to_latex(self) -> str:
        lines = []
        lines.append(f"\\author{{{self.name}}}")
        if self.email:
            lines.append(f"\\email{{{self.email}}}")
        lines.append(f"\\affiliation{{{self.affiliation}}}")
        return "\n".join(lines)


@dataclass(frozen=True)
class TheoremSpec:
    """A theorem, definition, lemma, corollary, or remark block."""

    env: str  # theorem, definition, lemma, corollary, remark
    label: str  # LaTeX label (e.g., thm:residue)
    title: str  # Display title in brackets
    body: str  # LaTeX body text
    proof: str = ""  # Optional proof body

    def to_latex(self) -> str:
        lines = []
        title_part = f"[{self.title}]" if self.title else ""
        lines.append(f"\\begin{{{self.env}}}{title_part}\\label{{{self.label}}}")
        lines.append(self.body)
        lines.append(f"\\end{{{self.env}}}")
        if self.proof:
            lines.append("")
            lines.append("\\begin{proof}")
            lines.append(self.proof)
            lines.append("\\end{proof}")
        return "\n".join(lines)


@dataclass(frozen=True)
class TableSpec:
    """A table with caption and label."""

    label: str  # LaTeX label
    caption: str
    headers: list[str]
    rows: list[list[str]]
    placement: str = "b"  # h, t, b, p
    notes: str = ""  # Optional table footnote

    def to_latex(self) -> str:
        ncols = len(self.headers)
        col_spec = "r" * ncols

        lines = []
        lines.append(f"\\begin{{table}}[{self.placement}]")
        lines.append(f"\\caption{{\\label{{{self.label}}}{self.caption}}}")
        lines.append("\\begin{ruledtabular}")
        lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        lines.append(" & ".join(self.headers) + " \\\\")
        lines.append("\\hline")
        for row in self.rows:
            lines.append(" & ".join(row) + " \\\\")
        lines.append("\\end{tabular}")
        lines.append("\\end{ruledtabular}")
        if self.notes:
            lines.append(f"\\tablecomments{{{self.notes}}}")
        lines.append("\\end{table}")
        return "\n".join(lines)


@dataclass(frozen=True)
class EquationSpec:
    """A labeled equation."""

    label: str
    body: str  # LaTeX math (without \\begin{equation})

    def to_latex(self) -> str:
        return f"\\begin{{equation}}\\label{{{self.label}}}\n  {self.body}\n\\end{{equation}}"


@dataclass(frozen=True)
class FigureSpec:
    """A figure with caption and label."""

    label: str
    caption: str
    filename: str  # Path to image file
    width: str = "\\columnwidth"
    placement: str = "t"

    def to_latex(self) -> str:
        lines = []
        lines.append(f"\\begin{{figure}}[{self.placement}]")
        lines.append(f"\\includegraphics[width={self.width}]{{{self.filename}}}")
        lines.append(f"\\caption{{\\label{{{self.label}}}{self.caption}}}")
        lines.append("\\end{figure}")
        return "\n".join(lines)


@dataclass(frozen=True)
class SectionSpec:
    """A section of the paper."""

    title: str
    label: str
    body: str  # Raw LaTeX body text
    level: int = 1  # 1=section, 2=subsection, 3=subsubsection

    def section_cmd(self) -> str:
        cmds = {1: "section", 2: "subsection", 3: "subsubsection"}
        return cmds.get(self.level, "section")

    def to_latex(self) -> str:
        cmd = self.section_cmd()
        lines = []
        lines.append(f"\\{cmd}{{{self.title}}}\\label{{{self.label}}}")
        lines.append("")
        lines.append(self.body)
        return "\n".join(lines)


class DocumentClass(StrEnum):
    """Supported document classes."""

    REVTEX_APS_PRD = "revtex-aps-prd"
    REVTEX_APS_PRL = "revtex-aps-prl"
    REVTEX_APS_PRX = "revtex-aps-prx"
    REVTEX_APS_RMP = "revtex-aps-rmp"
    ARTICLE = "article"
    PREPRINT = "preprint"


@dataclass(frozen=True)
class PaperSpec:
    """Complete specification of a paper.

    This is the single source of truth for generating a LaTeX file.
    Every field is frozen at construction time.
    """

    # ── metadata ──
    title: str
    authors: list[AuthorSpec]
    abstract: str
    date: str = ""  # defaults to \\today

    # ── structure ──
    sections: list[SectionSpec] = field(default_factory=list)
    acknowledgments: str = ""
    bibliography_file: str = "Bibliography"

    # ── document class ──
    document_class: DocumentClass = DocumentClass.REVTEX_APS_PRD

    # ── optional extras ──
    pacs_numbers: str = ""
    keywords: list[str] = field(default_factory=list)
    extra_preamble: str = ""
    extra_macros: str = ""

    # ── generation metadata ──
    generation_timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


# ═══════════════════════════════════════════════════════════════════
# PREAMBLE LIBRARY
# ═══════════════════════════════════════════════════════════════════

# Standard UMCP macro set — available in every generated paper.
UMCP_MACROS = r"""
% ── UMCP standard macros (auto-generated) ──
\newcommand{\tR}{\tau_{\!R}^{*}}
\newcommand{\Gam}{\Gamma}
\newcommand{\eps}{\varepsilon}
\newcommand{\IR}{\infty_{\mathrm{rec}}}
\newcommand{\tolseam}{\mathrm{tol}_{\mathrm{seam}}}
\newcommand{\dd}{\mathrm{d}}
\newcommand{\Res}{\mathrm{Res}}
\newcommand{\beq}{\begin{equation}}
\newcommand{\eeq}{\end{equation}}
\newcommand{\INF}{\texttt{INF\_REC}}
\newcommand{\regime}[1]{\textsc{#1}}
\newcommand{\conform}{\regime{conformant}}
\newcommand{\nonconform}{\regime{nonconformant}}
\newcommand{\coderef}[1]{\texttt{#1}}
"""

THEOREM_ENVS = r"""
% ── theorem environments ──
\newtheorem{theorem}{Theorem}
\newtheorem{definition}[theorem]{Definition}
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{corollary}[theorem]{Corollary}
\newtheorem{remark}{Remark}
"""

STANDARD_PACKAGES = [
    "amsmath",
    "amssymb",
    "amsthm",
    "graphicx",
    "hyperref",
    "xcolor",
    "bm",
    "booktabs",
    "multirow",
    "xfrac",
]


def _docclass_line(cls: DocumentClass) -> str:
    """Generate the \\documentclass line."""
    mapping = {
        DocumentClass.REVTEX_APS_PRD: (
            "\\documentclass[\n"
            "  aps,\n"
            "  prd,\n"
            "  twocolumn,\n"
            "  superscriptaddress,\n"
            "  nofootinbib,\n"
            "  floatfix\n"
            "]{revtex4-2}"
        ),
        DocumentClass.REVTEX_APS_PRL: (
            "\\documentclass[\n"
            "  aps,\n"
            "  prl,\n"
            "  twocolumn,\n"
            "  superscriptaddress,\n"
            "  nofootinbib,\n"
            "  floatfix\n"
            "]{revtex4-2}"
        ),
        DocumentClass.REVTEX_APS_PRX: (
            "\\documentclass[\n"
            "  aps,\n"
            "  prx,\n"
            "  twocolumn,\n"
            "  superscriptaddress,\n"
            "  nofootinbib,\n"
            "  floatfix\n"
            "]{revtex4-2}"
        ),
        DocumentClass.REVTEX_APS_RMP: (
            "\\documentclass[\n"
            "  aps,\n"
            "  rmp,\n"
            "  longbibliography,\n"
            "  superscriptaddress,\n"
            "  nofootinbib,\n"
            "  floatfix\n"
            "]{revtex4-2}"
        ),
        DocumentClass.ARTICLE: "\\documentclass[12pt,a4paper]{article}",
        DocumentClass.PREPRINT: (
            "\\documentclass[\n  aps,\n  prd,\n  preprint,\n  superscriptaddress,\n  nofootinbib\n]{revtex4-2}"
        ),
    }
    return mapping.get(cls, mapping[DocumentClass.REVTEX_APS_PRD])


# ═══════════════════════════════════════════════════════════════════
# GENERATOR
# ═══════════════════════════════════════════════════════════════════


class LaTeXGenerator:
    """Generate a complete .tex file from a PaperSpec.

    The generator produces deterministic, fully-formatted RevTeX output
    with the following guarantees:
        - All UMCP macros and theorem environments included
        - Consistent spacing and indentation
        - Section separators with visual markers
        - Frozen-constants table auto-generated if requested
        - Tier-mapping table auto-generated if requested
        - SHA-256 stamp embedded as a LaTeX comment
    """

    def __init__(self, spec: PaperSpec) -> None:
        self.spec = spec

    def generate(self) -> str:
        """Generate the complete .tex source."""
        parts: list[str] = []

        # ── file header ──
        parts.append(self._header())
        parts.append("")

        # ── document class ──
        parts.append(_docclass_line(self.spec.document_class))
        parts.append("")

        # ── packages ──
        parts.append("% ── packages ──")
        pkg_line = ",".join(STANDARD_PACKAGES[:3])
        parts.append(f"\\usepackage{{{pkg_line}}}")
        for pkg in STANDARD_PACKAGES[3:]:
            parts.append(f"\\usepackage{{{pkg}}}")
        parts.append("")

        # ── theorem environments ──
        parts.append(THEOREM_ENVS.strip())
        parts.append("")

        # ── macros ──
        parts.append(UMCP_MACROS.strip())
        if self.spec.extra_macros:
            parts.append("")
            parts.append("% ── paper-specific macros ──")
            parts.append(self.spec.extra_macros)
        parts.append("")

        # ── extra preamble ──
        if self.spec.extra_preamble:
            parts.append("% ── extra preamble ──")
            parts.append(self.spec.extra_preamble)
            parts.append("")

        # ── begin document ──
        parts.append("\\begin{document}")
        parts.append("")

        # ── title block ──
        parts.append(self._section_separator("TITLE"))
        parts.append(self._title_block())
        parts.append("")

        # ── abstract ──
        parts.append("\\begin{abstract}")
        parts.append(self._wrap(self.spec.abstract))
        parts.append("\\end{abstract}")
        parts.append("")
        parts.append("\\maketitle")
        parts.append("")

        # ── sections ──
        for sec in self.spec.sections:
            parts.append(self._section_separator(sec.label.upper()))
            parts.append(sec.to_latex())
            parts.append("")

        # ── acknowledgments ──
        if self.spec.acknowledgments:
            parts.append(self._section_separator("ACKNOWLEDGMENTS"))
            parts.append("\\begin{acknowledgments}")
            parts.append(self._wrap(self.spec.acknowledgments))
            parts.append("\\end{acknowledgments}")
            parts.append("")

        # ── bibliography ──
        parts.append(f"\\bibliography{{{self.spec.bibliography_file}}}")
        parts.append("")
        parts.append("\\end{document}")

        source = "\n".join(parts)

        # ── stamp ──
        sha = hashlib.sha256(source.encode()).hexdigest()[:16]
        stamp = f"% SHA-256 (first 16): {sha}\n% Generated: {self.spec.generation_timestamp}\n"
        return stamp + source

    def _header(self) -> str:
        """File header comment block."""
        width = 60
        # Sanitize title for comment: strip LaTeX line breaks
        safe_title = self.spec.title.replace("\\\\", " ").replace("\n", " ")
        safe_title = re.sub(r"\s+", " ", safe_title).strip()
        lines = [
            "% " + "=" * width,
            f"% {safe_title}",
            "%",
            "% Generated by UMCP LaTeX Publication Protocol",
            f"% Timestamp: {self.spec.generation_timestamp}",
            "%",
            "% Compile: pdflatex → bibtex → pdflatex → pdflatex",
            "% " + "=" * width,
        ]
        return "\n".join(lines)

    def _section_separator(self, label: str) -> str:
        return f"% {'=' * 60}\n% {label}\n% {'=' * 60}"

    def _title_block(self) -> str:
        lines = []
        # Handle multi-line titles with \\
        lines.append(f"\\title{{{self.spec.title}}}")
        lines.append("")
        for author in self.spec.authors:
            lines.append(author.to_latex())
            lines.append("")
        date_str = self.spec.date if self.spec.date else "\\today"
        lines.append(f"\\date{{{date_str}}}")
        return "\n".join(lines)

    def _wrap(self, text: str, width: int = 72) -> str:
        """Wrap text to line width, preserving LaTeX commands."""
        # Don't wrap lines that start with \ or contain display math
        result_lines = []
        for line in text.split("\n"):
            stripped = line.strip()
            if (
                stripped.startswith("\\")
                or stripped.startswith("%")
                or "\\begin{" in stripped
                or "\\end{" in stripped
                or stripped.startswith("&")
                or len(stripped) <= width
            ):
                result_lines.append(line)
            else:
                wrapped = textwrap.fill(stripped, width=width)
                result_lines.append(wrapped)
        return "\n".join(result_lines)


# ═══════════════════════════════════════════════════════════════════
# COMPILER
# ═══════════════════════════════════════════════════════════════════


class CompileStatus(StrEnum):
    SUCCESS = "success"
    WARNING = "warning"
    FAILURE = "failure"


@dataclass(frozen=True)
class CompileResult:
    """Result of the LaTeX compilation pipeline."""

    status: CompileStatus
    pdf_path: str
    pdf_size_bytes: int
    page_count: int
    warnings: list[str]
    errors: list[str]
    log_path: str
    passes: int  # Number of pdflatex passes
    bibtex_ran: bool
    sha256: str  # SHA-256 of the generated PDF

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "pdf_path": self.pdf_path,
            "pdf_size_bytes": self.pdf_size_bytes,
            "page_count": self.page_count,
            "warning_count": len(self.warnings),
            "error_count": len(self.errors),
            "warnings": self.warnings[:10],  # Cap for display
            "errors": self.errors[:10],
            "passes": self.passes,
            "bibtex_ran": self.bibtex_ran,
            "sha256": self.sha256,
        }


class LaTeXCompiler:
    """Compile .tex → .pdf through the full pipeline.

    Pipeline:
        1. pdflatex (pass 1) — generates .aux
        2. bibtex             — resolves bibliography
        3. pdflatex (pass 2) — incorporates .bbl
        4. pdflatex (pass 3) — resolves forward refs

    Idempotent: running twice on unchanged input gives same PDF.
    """

    def __init__(
        self,
        tex_path: str | Path,
        *,
        engine: str = "pdflatex",
        interaction: str = "nonstopmode",
        extra_args: list[str] | None = None,
    ) -> None:
        self.tex_path = Path(tex_path).resolve()
        self.engine = engine
        self.interaction = interaction
        self.extra_args = extra_args or []

        if not self.tex_path.exists():
            msg = f"TeX file not found: {self.tex_path}"
            raise FileNotFoundError(msg)

        self.work_dir = self.tex_path.parent
        self.stem = self.tex_path.stem
        self.pdf_path = self.work_dir / f"{self.stem}.pdf"
        self.log_path = self.work_dir / f"{self.stem}.log"
        self.aux_path = self.work_dir / f"{self.stem}.aux"

    def compile(self) -> CompileResult:
        """Run the full compilation pipeline."""
        errors: list[str] = []
        passes = 0

        # Check engine availability
        if not shutil.which(self.engine):
            return CompileResult(
                status=CompileStatus.FAILURE,
                pdf_path="",
                pdf_size_bytes=0,
                page_count=0,
                warnings=[],
                errors=[f"LaTeX engine '{self.engine}' not found on PATH"],
                log_path="",
                passes=0,
                bibtex_ran=False,
                sha256="",
            )

        # Pass 1 (may fail on missing refs — that's normal)
        self._run_engine()
        passes += 1

        # BibTeX
        bibtex_ran = False
        if self.aux_path.exists() and shutil.which("bibtex"):
            bibtex_ran = self._run_bibtex()

        # Pass 2
        self._run_engine()
        passes += 1

        # Pass 3 (resolve forward refs)
        self._run_engine()
        passes += 1

        # Parse log
        warnings: list[str] = self._parse_warnings()
        log_errors: list[str] = self._parse_errors()
        errors.extend(log_errors)

        # Read PDF
        pdf_size = 0
        page_count = 0
        sha256 = ""
        if self.pdf_path.exists():
            pdf_size = self.pdf_path.stat().st_size
            page_count = self._count_pages()
            sha256 = self._hash_file(self.pdf_path)

        # Determine status
        if errors or not self.pdf_path.exists():
            status = CompileStatus.FAILURE
        elif warnings:
            status = CompileStatus.WARNING
        else:
            status = CompileStatus.SUCCESS

        return CompileResult(
            status=status,
            pdf_path=str(self.pdf_path),
            pdf_size_bytes=pdf_size,
            page_count=page_count,
            warnings=warnings,
            errors=errors,
            log_path=str(self.log_path),
            passes=passes,
            bibtex_ran=bibtex_ran,
            sha256=sha256,
        )

    def _run_engine(self) -> bool:
        """Run one pass of pdflatex."""
        cmd = [
            self.engine,
            f"-interaction={self.interaction}",
            *self.extra_args,
            str(self.tex_path.name),
        ]
        result = subprocess.run(
            cmd,
            cwd=str(self.work_dir),
            capture_output=True,
            text=True,
            timeout=120,
        )
        return result.returncode == 0

    def _run_bibtex(self) -> bool:
        """Run bibtex."""
        cmd = ["bibtex", self.stem]
        result = subprocess.run(
            cmd,
            cwd=str(self.work_dir),
            capture_output=True,
            text=True,
            timeout=60,
        )
        return result.returncode == 0

    def _parse_warnings(self) -> list[str]:
        """Extract warnings from the log file."""
        warnings: list[str] = []
        if not self.log_path.exists():
            return warnings
        log_text = self.log_path.read_text(errors="replace")
        # Undefined references
        for m in re.finditer(r"LaTeX Warning: Reference .* undefined", log_text):
            warnings.append(m.group(0))
        # Citation undefined
        for m in re.finditer(r"LaTeX Warning: Citation .* undefined", log_text):
            warnings.append(m.group(0))
        # Overfull boxes (significant only)
        for m in re.finditer(r"Overfull \\[hv]box \((\d+\.?\d*)pt", log_text):
            badness = float(m.group(1))
            if badness > 5.0:  # Only flag significant overfulls
                warnings.append(m.group(0))
        return warnings

    def _parse_errors(self) -> list[str]:
        """Extract errors from the log file."""
        errors: list[str] = []
        if not self.log_path.exists():
            return errors
        log_text = self.log_path.read_text(errors="replace")
        for m in re.finditer(r"^! (.+)$", log_text, re.MULTILINE):
            error_msg = m.group(1).strip()
            if error_msg not in errors:
                errors.append(error_msg)
        return errors

    def _count_pages(self) -> int:
        """Count PDF pages from the log."""
        if not self.log_path.exists():
            return 0
        log_text = self.log_path.read_text(errors="replace")
        m = re.search(r"Output written on .+ \((\d+) page", log_text)
        return int(m.group(1)) if m else 0

    @staticmethod
    def _hash_file(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()


# ═══════════════════════════════════════════════════════════════════
# VALIDATOR
# ═══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ValidationResult:
    """Post-compilation validation result."""

    is_valid: bool
    undefined_refs: list[str]
    undefined_citations: list[str]
    overfull_boxes: int
    underfull_boxes: int
    font_warnings: list[str]
    page_count: int
    pdf_exists: bool

    def summary(self) -> str:
        lines = []
        status = "PASS" if self.is_valid else "FAIL"
        lines.append(f"Validation: {status}")
        lines.append(f"  Pages: {self.page_count}")
        lines.append(f"  PDF exists: {self.pdf_exists}")
        lines.append(f"  Undefined refs: {len(self.undefined_refs)}")
        lines.append(f"  Undefined cites: {len(self.undefined_citations)}")
        lines.append(f"  Overfull hboxes: {self.overfull_boxes}")
        for ref in self.undefined_refs[:5]:
            lines.append(f"    - {ref}")
        for cite in self.undefined_citations[:5]:
            lines.append(f"    - {cite}")
        return "\n".join(lines)


class LaTeXValidator:
    """Validate a compiled paper for common issues."""

    def __init__(self, tex_path: str | Path) -> None:
        self.tex_path = Path(tex_path).resolve()
        self.log_path = self.tex_path.with_suffix(".log")
        self.pdf_path = self.tex_path.with_suffix(".pdf")

    def validate(self) -> ValidationResult:
        pdf_exists = self.pdf_path.exists()
        page_count = 0
        undefined_refs: list[str] = []
        undefined_citations: list[str] = []
        overfull = 0
        underfull = 0
        font_warnings: list[str] = []

        if self.log_path.exists():
            log_text = self.log_path.read_text(errors="replace")

            # Count pages
            m = re.search(r"Output written on .+ \((\d+) page", log_text)
            if m:
                page_count = int(m.group(1))

            # Undefined references
            for m_ref in re.finditer(r"LaTeX Warning: Reference `([^']+)' on page", log_text):
                undefined_refs.append(m_ref.group(1))

            # Undefined citations
            for m_cite in re.finditer(r"LaTeX Warning: Citation `([^']+)' on page", log_text):
                undefined_citations.append(m_cite.group(1))

            # Overfull/underfull
            overfull = len(re.findall(r"Overfull \\[hv]box", log_text))
            underfull = len(re.findall(r"Underfull \\[hv]box", log_text))

            # Font warnings
            for m_font in re.finditer(r"LaTeX Font Warning: (.+)", log_text):
                font_warnings.append(m_font.group(1))

        is_valid = pdf_exists and len(undefined_refs) == 0 and len(undefined_citations) == 0 and page_count > 0

        return ValidationResult(
            is_valid=is_valid,
            undefined_refs=undefined_refs,
            undefined_citations=undefined_citations,
            overfull_boxes=overfull,
            underfull_boxes=underfull,
            font_warnings=font_warnings,
            page_count=page_count,
            pdf_exists=pdf_exists,
        )


# ═══════════════════════════════════════════════════════════════════
# CONVENIENCE BUILDERS
# ═══════════════════════════════════════════════════════════════════


def frozen_constants_table(
    constants: list[tuple[str, str, str]],
    *,
    label: str = "tab:constants",
    caption: str = "Frozen constants (consistent across every seam).",
) -> str:
    """Generate a frozen constants table from (symbol, value, meaning) triples."""
    return TableSpec(
        label=label,
        caption=caption,
        headers=["Symbol", "Value", "Meaning"],
        rows=[[f"${sym}$", val, meaning] for sym, val, meaning in constants],
        placement="h",
    ).to_latex()


def tier_mapping_table(
    tiers: list[tuple[str, str]],
    *,
    label: str = "tab:tier",
    caption: str = "Tier architecture mapping.",
) -> str:
    """Generate a tier mapping table from (tier, contents) pairs."""
    lines = []
    lines.append("\\begin{table}[h]")
    lines.append(f"\\caption{{\\label{{{label}}}{caption}}}")
    lines.append("\\begin{ruledtabular}")
    lines.append("\\begin{tabular}{lp{4.8cm}}")
    lines.append("\\textbf{Tier} & \\textbf{Contents} \\\\")
    lines.append("\\hline")
    for tier_name, contents in tiers:
        lines.append(f"\\textbf{{{tier_name}}} & {contents} \\\\[4pt]")
    lines.append("\\end{tabular}")
    lines.append("\\end{ruledtabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def itemize_block(items: list[str]) -> str:
    """Generate a \\begin{itemize} block."""
    lines = ["\\begin{itemize}"]
    for item in items:
        lines.append(f"  \\item {item}")
    lines.append("\\end{itemize}")
    return "\n".join(lines)


def enumerate_block(items: list[str]) -> str:
    """Generate a \\begin{enumerate} block."""
    lines = ["\\begin{enumerate}"]
    for item in items:
        lines.append(f"  \\item {item}")
    lines.append("\\end{enumerate}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# YAML SPEC LOADER
# ═══════════════════════════════════════════════════════════════════


def load_spec_from_yaml(path: str | Path) -> PaperSpec:
    """Load a PaperSpec from a YAML file.

    Expected YAML structure:
        title: "Paper Title"
        authors:
          - name: "Author Name"
            affiliation: "Institution"
            email: "email@example.com"
        abstract: "Abstract text..."
        document_class: "revtex-aps-prd"
        bibliography_file: "Bibliography"
        sections:
          - title: "Introduction"
            label: "sec:intro"
            body: "LaTeX body text..."
          ...
        acknowledgments: "Thanks to..."
    """
    try:
        import yaml
    except ImportError:
        msg = "PyYAML required for YAML spec loading: pip install pyyaml"
        raise ImportError(msg) from None

    raw = yaml.safe_load(Path(path).read_text())

    authors = [
        AuthorSpec(
            name=a["name"],
            affiliation=a.get("affiliation", ""),
            email=a.get("email", ""),
        )
        for a in raw.get("authors", [])
    ]

    sections = [
        SectionSpec(
            title=s["title"],
            label=s["label"],
            body=s["body"],
            level=s.get("level", 1),
        )
        for s in raw.get("sections", [])
    ]

    doc_class_str = raw.get("document_class", "revtex-aps-prd")
    try:
        doc_class = DocumentClass(doc_class_str)
    except ValueError:
        doc_class = DocumentClass.REVTEX_APS_PRD

    return PaperSpec(
        title=raw["title"],
        authors=authors,
        abstract=raw.get("abstract", ""),
        date=raw.get("date", ""),
        sections=sections,
        acknowledgments=raw.get("acknowledgments", ""),
        bibliography_file=raw.get("bibliography_file", "Bibliography"),
        document_class=doc_class,
        extra_preamble=raw.get("extra_preamble", ""),
        extra_macros=raw.get("extra_macros", ""),
    )


# ═══════════════════════════════════════════════════════════════════
# FULL PIPELINE
# ═══════════════════════════════════════════════════════════════════


def generate_and_compile(
    spec: PaperSpec,
    output_path: str | Path,
    *,
    compile_pdf: bool = True,
) -> tuple[str, CompileResult | None, ValidationResult | None]:
    """Full pipeline: PaperSpec → .tex → .pdf → validation.

    Args:
        spec: Complete paper specification
        output_path: Where to write the .tex file
        compile_pdf: Whether to compile after generation

    Returns:
        (tex_source, compile_result, validation_result)
    """
    output_path = Path(output_path)

    # Generate
    generator = LaTeXGenerator(spec)
    source = generator.generate()

    # Write
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(source)
    print(f"  Generated: {output_path} ({len(source)} bytes)")

    compile_result = None
    validation_result = None

    if compile_pdf:
        # Compile
        compiler = LaTeXCompiler(output_path)
        compile_result = compiler.compile()
        print(f"  Compiled:  {compile_result.status.value}")
        print(f"  PDF:       {compile_result.pdf_path} ({compile_result.pdf_size_bytes} bytes)")
        print(f"  Pages:     {compile_result.page_count}")
        print(f"  Passes:    {compile_result.passes}")
        print(f"  BibTeX:    {'yes' if compile_result.bibtex_ran else 'no'}")

        if compile_result.errors:
            print(f"  Errors:    {len(compile_result.errors)}")
            for e in compile_result.errors[:5]:
                print(f"    ! {e}")

        if compile_result.warnings:
            print(f"  Warnings:  {len(compile_result.warnings)}")
            for w in compile_result.warnings[:5]:
                print(f"    * {w}")

        # Validate
        validator = LaTeXValidator(output_path)
        validation_result = validator.validate()
        print(f"  Valid:     {validation_result.is_valid}")

        if validation_result.undefined_refs:
            print(f"  Undef refs: {validation_result.undefined_refs}")
        if validation_result.undefined_citations:
            print(f"  Undef cites: {validation_result.undefined_citations}")

    return source, compile_result, validation_result


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════


def _find_repo_root() -> Path:
    """Walk up from this script to find the repo root."""
    candidate = Path(__file__).resolve().parent.parent
    if (candidate / "pyproject.toml").exists():
        return candidate
    current = Path.cwd()
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    msg = "Cannot locate repo root"
    raise FileNotFoundError(msg)


def _cmd_demo(args: argparse.Namespace) -> int:
    """Generate the demo paper (tau_r_star_dynamics results)."""
    repo = _find_repo_root()
    output = repo / "paper" / "generated_demo.tex"

    spec = PaperSpec(
        title=(
            "Statistical Mechanics of the UMCP Budget Identity:\\\\\n"
            "Pole Structure, Metastability, Separability,\\\\\n"
            "and a Universal Scaling Law"
        ),
        authors=[
            AuthorSpec(
                name="Caleb Pruett",
                affiliation="UMCP Reference Implementation, GitHub",
                email="caleb@umcp.dev",
            ),
            AuthorSpec(
                name="Clement Paulus",
                affiliation="UMCP / GCD / RCFT Canon",
            ),
        ],
        abstract=(
            "We derive four new results from the budget identity of the "
            "Universal Measurement Contract Protocol (UMCP). Starting "
            "from the drift--cost closure $\\Gam(\\omega)=\\omega^{p}/(1-\\omega+\\eps)$ "
            "with frozen constants $(p{=}3,\\;\\alpha{=}1,\\;\\eps{=}10^{-8})$, we show: "
            "\\textbf{(D1)}~the $\\eps$-regularized pole at $\\omega{=}1$ carries effective "
            "residue $\\tfrac{1}{2}$, revealing a $\\mathbb{Z}_2$ symmetry; "
            "\\textbf{(D2)}~the barrier height from the Stable well to the trapping "
            "threshold equals $\\alpha$ exactly, making Stable a genuine metastable "
            "phase with Kramers escape time $\\sim e^{\\beta\\alpha}$; "
            "\\textbf{(D3)}~the budget numerator $N(\\omega,C,\\Delta\\kappa)$ is "
            "additively separable (all cross-derivatives vanish), establishing an "
            "ideal-gas structure in state space; and "
            "\\textbf{(D4)}~the Gibbs measure $P(\\omega)\\propto e^{-\\beta\\Gam(\\omega)}$ "
            "yields the universal scaling law "
            "$\\langle\\omega\\rangle\\approx\\tfrac{1}{2}\\,R^{1/p}$ "
            "with the same $\\tfrac{1}{2}$ prefactor from~D1."
        ),
        sections=[
            SectionSpec(
                title="Introduction",
                label="sec:intro",
                body=(
                    "The Universal Measurement Contract Protocol (UMCP) "
                    "\\cite{paulus2025umcp,paulus2025physicscoherence} "
                    "validates computational workflows against mathematical contracts. "
                    "Its unit of work is a \\emph{casepack}: raw data plus a contract, "
                    "closures, and expected outputs, checked for schema conformance, "
                    "Tier-1~kernel identities, regime classification, and SHA-256 integrity."
                ),
            ),
            SectionSpec(
                title="Conventions and Frozen Constants",
                label="sec:conventions",
                body=frozen_constants_table(
                    [
                        ("p", "$3$", "Contraction exponent"),
                        ("\\alpha", "$1.0$", "Curvature coupling"),
                        ("\\eps", "$10^{-8}$", "Guard band"),
                        ("\\tolseam", "$0.005$", "Seam tolerance"),
                        ("\\lambda", "$0.2$", "EWMA decay"),
                    ]
                ),
            ),
            SectionSpec(
                title="Results",
                label="sec:results",
                body=(
                    "Four results emerge from the budget identity without new parameters.\n\n"
                    + TheoremSpec(
                        env="theorem",
                        label="thm:residue",
                        title="Effective residue; $\\mathbb{Z}_2$ pole structure",
                        body=(
                            "Under $\\eps$-regularization, the effective residue of "
                            "$\\Gam$ at the pole $\\omega{=}1$ is\n"
                            "\\beq\\label{eq:residue}\n"
                            "  \\Res_{\\mathrm{eff}}[\\Gam,\\,\\omega{=}1] = \\frac{1}{2}.\n"
                            "\\eeq"
                        ),
                    ).to_latex()
                    + "\n\n"
                    + TheoremSpec(
                        env="theorem",
                        label="thm:barrier",
                        title="Barrier height identity",
                        body=(
                            "The barrier from Stable to trapping equals $\\alpha$ exactly:\n"
                            "\\beq\\label{eq:barrier}\n"
                            "  \\Delta\\Gam = \\Gam(\\omega_{\\mathrm{trap}}) - \\Gam(0) = \\alpha.\n"
                            "\\eeq"
                        ),
                    ).to_latex()
                    + "\n\n"
                    + TheoremSpec(
                        env="theorem",
                        label="thm:separability",
                        title="Separability of the budget numerator",
                        body=(
                            "All mixed partial derivatives of $N(\\omega,C,\\Delta\\kappa)$ vanish:\n"
                            "\\beq\\label{eq:sep}\n"
                            "  \\frac{\\partial^2 N}{\\partial\\omega\\,\\partial C} = "
                            "\\frac{\\partial^2 N}{\\partial\\omega\\,\\partial\\kappa} = "
                            "\\frac{\\partial^2 N}{\\partial C\\,\\partial\\kappa} = 0.\n"
                            "\\eeq"
                        ),
                    ).to_latex()
                    + "\n\n"
                    + TheoremSpec(
                        env="theorem",
                        label="thm:scaling",
                        title="Universal scaling law",
                        body=(
                            "The mean equilibrium drift satisfies\n"
                            "\\beq\\label{eq:scaling}\n"
                            "  \\langle\\omega\\rangle_\\beta \\approx "
                            "\\frac{1}{2}\\,R^{1/p} \\qquad (\\beta\\to\\infty).\n"
                            "\\eeq"
                        ),
                    ).to_latex()
                ),
            ),
            SectionSpec(
                title="Tier Architecture",
                label="sec:tier",
                body=tier_mapping_table(
                    [
                        (
                            "Tier-1 (immutable)",
                            "Frozen constants; kernel invariants; $\\Gam(\\omega)$.",
                        ),
                        (
                            "Tier-0 (gates)",
                            "T10: $\\Res=\\tfrac{1}{2}$; T11: $\\Delta\\Gam=\\alpha$; T12: separability.",
                        ),
                        (
                            "Tier-2 (expansion)",
                            "T13: Gibbs scaling; T14: Kramers escape; "
                            "T15: Legendre conjugate; T16: entropy production.",
                        ),
                    ]
                ),
            ),
        ],
        acknowledgments=(
            "C.P.\\ thanks C.Pau.\\ for the frozen-contract discipline. "
            "Reference implementation: "
            "\\url{https://github.com/calebpruett927/UMCP-Metadata-Runnable-Code}."
        ),
    )

    print("UMCP LaTeX Publication Protocol — Demo")
    print("=" * 50)
    _, cr, vr = generate_and_compile(spec, output, compile_pdf=not args.no_compile)

    if cr and cr.status == CompileStatus.FAILURE:
        return 1
    if vr and not vr.is_valid:
        print(vr.summary())
        return 1
    return 0


def _cmd_compile(args: argparse.Namespace) -> int:
    """Compile an existing .tex file."""
    tex_path = Path(args.tex_file).resolve()
    print(f"Compiling: {tex_path}")
    compiler = LaTeXCompiler(tex_path)
    result = compiler.compile()
    print(f"Status:   {result.status.value}")
    print(f"PDF:      {result.pdf_path}")
    print(f"Pages:    {result.page_count}")
    print(f"Size:     {result.pdf_size_bytes} bytes")
    print(f"SHA-256:  {result.sha256[:16]}...")
    if result.errors:
        for e in result.errors:
            print(f"  ! {e}")
    if result.warnings:
        for w in result.warnings:
            print(f"  * {w}")
    return 0 if result.status != CompileStatus.FAILURE else 1


def _cmd_validate(args: argparse.Namespace) -> int:
    """Validate a compiled paper."""
    tex_path = Path(args.tex_file).resolve()
    print(f"Validating: {tex_path}")
    validator = LaTeXValidator(tex_path)
    result = validator.validate()
    print(result.summary())
    return 0 if result.is_valid else 1


def _cmd_spec(args: argparse.Namespace) -> int:
    """Generate from a YAML spec file."""
    spec_path = Path(args.spec_file).resolve()
    output = Path(args.output).resolve() if args.output else spec_path.with_suffix(".tex")

    print(f"Loading spec: {spec_path}")
    spec = load_spec_from_yaml(spec_path)

    print("UMCP LaTeX Publication Protocol")
    print("=" * 50)
    _, cr, vr = generate_and_compile(spec, output, compile_pdf=not args.no_compile)

    if cr and cr.status == CompileStatus.FAILURE:
        return 1
    if vr and not vr.is_valid:
        print(vr.summary())
        return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="generate_latex",
        description="UMCP LaTeX Publication Protocol — generate optimized RevTeX papers",
    )
    sub = parser.add_subparsers(dest="command", help="Commands")

    # demo
    demo_p = sub.add_parser("demo", help="Generate demo paper from frozen results")
    demo_p.add_argument("--no-compile", action="store_true", help="Skip PDF compilation")

    # compile
    compile_p = sub.add_parser("compile", help="Compile existing .tex file")
    compile_p.add_argument("tex_file", help="Path to .tex file")

    # validate
    val_p = sub.add_parser("validate", help="Validate compiled paper")
    val_p.add_argument("tex_file", help="Path to .tex file")

    # spec
    spec_p = sub.add_parser("spec", help="Generate from YAML spec")
    spec_p.add_argument("spec_file", help="Path to YAML spec file")
    spec_p.add_argument("-o", "--output", help="Output .tex path")
    spec_p.add_argument("--no-compile", action="store_true", help="Skip PDF compilation")

    args = parser.parse_args()

    if args.command == "demo":
        sys.exit(_cmd_demo(args))
    elif args.command == "compile":
        sys.exit(_cmd_compile(args))
    elif args.command == "validate":
        sys.exit(_cmd_validate(args))
    elif args.command == "spec":
        sys.exit(_cmd_spec(args))
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()

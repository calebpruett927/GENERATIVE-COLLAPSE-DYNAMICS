"""Tests for the UMCP LaTeX Publication Protocol (generate_latex.py).

Covers: data models, generator output, compiler integration, validator,
YAML spec loading, and convenience builders.
"""

from __future__ import annotations

import importlib
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

# Import the module under test
gen = importlib.import_module("scripts.generate_latex")

AuthorSpec = gen.AuthorSpec
TheoremSpec = gen.TheoremSpec
TableSpec = gen.TableSpec
EquationSpec = gen.EquationSpec
FigureSpec = gen.FigureSpec
SectionSpec = gen.SectionSpec
DocumentClass = gen.DocumentClass
PaperSpec = gen.PaperSpec
LaTeXGenerator = gen.LaTeXGenerator
LaTeXCompiler = gen.LaTeXCompiler
LaTeXValidator = gen.LaTeXValidator
CompileStatus = gen.CompileStatus
frozen_constants_table = gen.frozen_constants_table
tier_mapping_table = gen.tier_mapping_table
itemize_block = gen.itemize_block
enumerate_block = gen.enumerate_block
generate_and_compile = gen.generate_and_compile


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture()
def minimal_spec() -> PaperSpec:
    """A minimal PaperSpec with one author and one section."""
    return PaperSpec(
        title="Test Paper",
        authors=[
            AuthorSpec(
                name="Test Author",
                affiliation="Test University",
                email="test@test.edu",
            )
        ],
        abstract="This is a test abstract.",
        sections=[
            SectionSpec(
                title="Introduction",
                label="sec:intro",
                body="Test body content.",
            )
        ],
    )


@pytest.fixture()
def repo_root() -> Path:
    """Path to the repo root."""
    return Path(__file__).resolve().parent.parent


# ═══════════════════════════════════════════════════════════════════
# Data Model Tests
# ═══════════════════════════════════════════════════════════════════


class TestAuthorSpec:
    """Tests for AuthorSpec data model."""

    def test_to_latex_with_email(self) -> None:
        author = AuthorSpec(name="A. Name", affiliation="Dept", email="a@b.c")
        tex = author.to_latex()
        assert "\\author{A. Name}" in tex
        assert "\\email{a@b.c}" in tex
        assert "\\affiliation{Dept}" in tex

    def test_to_latex_without_email(self) -> None:
        author = AuthorSpec(name="A. Name", affiliation="Dept")
        tex = author.to_latex()
        assert "\\author{A. Name}" in tex
        assert "\\email" not in tex
        assert "\\affiliation{Dept}" in tex

    def test_frozen(self) -> None:
        author = AuthorSpec(name="X", affiliation="Y")
        with pytest.raises(AttributeError):
            author.name = "Z"  # type: ignore[misc]


class TestTheoremSpec:
    """Tests for TheoremSpec data model."""

    def test_basic_theorem(self) -> None:
        thm = TheoremSpec(
            env="theorem",
            label="thm:test",
            title="Test Theorem",
            body="Statement.",
        )
        tex = thm.to_latex()
        assert "\\begin{theorem}[Test Theorem]\\label{thm:test}" in tex
        assert "Statement." in tex
        assert "\\end{theorem}" in tex
        assert "\\begin{proof}" not in tex

    def test_theorem_with_proof(self) -> None:
        thm = TheoremSpec(
            env="theorem",
            label="thm:p",
            title="",
            body="Statement.",
            proof="QED.",
        )
        tex = thm.to_latex()
        assert "\\begin{proof}" in tex
        assert "QED." in tex
        assert "\\end{proof}" in tex

    def test_env_types(self) -> None:
        for env in ("theorem", "definition", "lemma", "corollary", "remark"):
            thm = TheoremSpec(env=env, label=f"x:{env}", title="T", body="B")
            assert f"\\begin{{{env}}}" in thm.to_latex()

    def test_no_title_bracket(self) -> None:
        thm = TheoremSpec(env="lemma", label="lem:x", title="", body="B")
        tex = thm.to_latex()
        assert "\\begin{lemma}\\label{lem:x}" in tex
        assert "[]" not in tex


class TestTableSpec:
    """Tests for TableSpec data model."""

    def test_basic_table(self) -> None:
        tbl = TableSpec(
            label="tab:test",
            caption="Test Table",
            headers=["Col A", "Col B"],
            rows=[["1", "2"], ["3", "4"]],
        )
        tex = tbl.to_latex()
        assert "\\begin{table}" in tex
        assert "\\label{tab:test}" in tex
        assert "Test Table" in tex
        assert "Col A & Col B" in tex
        assert "1 & 2" in tex
        assert "\\end{table}" in tex

    def test_table_with_notes(self) -> None:
        tbl = TableSpec(
            label="tab:n",
            caption="C",
            headers=["X"],
            rows=[["1"]],
            notes="Some notes.",
        )
        tex = tbl.to_latex()
        assert "Some notes." in tex


class TestEquationSpec:
    """Tests for EquationSpec."""

    def test_basic_equation(self) -> None:
        eq = EquationSpec(label="eq:test", body="x = y + z")
        tex = eq.to_latex()
        assert "\\begin{equation}\\label{eq:test}" in tex
        assert "x = y + z" in tex
        assert "\\end{equation}" in tex


class TestFigureSpec:
    """Tests for FigureSpec."""

    def test_basic_figure(self) -> None:
        fig = FigureSpec(
            label="fig:test",
            caption="A figure",
            filename="plot.pdf",
        )
        tex = fig.to_latex()
        assert "\\begin{figure}" in tex
        assert "\\includegraphics" in tex
        assert "plot.pdf" in tex
        assert "\\label{fig:test}" in tex


class TestSectionSpec:
    """Tests for SectionSpec."""

    def test_section_levels(self) -> None:
        for level, cmd in [(1, "section"), (2, "subsection"), (3, "subsubsection")]:
            sec = SectionSpec(title="T", label="s:t", body="B", level=level)
            assert f"\\{cmd}{{T}}" in sec.to_latex()

    def test_label_included(self) -> None:
        sec = SectionSpec(title="T", label="sec:mine", body="B")
        assert "\\label{sec:mine}" in sec.to_latex()


class TestDocumentClass:
    """Tests for DocumentClass enum."""

    def test_all_values(self) -> None:
        expected = {
            "revtex-aps-prd",
            "revtex-aps-prl",
            "revtex-aps-prx",
            "revtex-aps-rmp",
            "article",
            "preprint",
        }
        actual = {dc.value for dc in DocumentClass}
        assert actual == expected


# ═══════════════════════════════════════════════════════════════════
# Generator Tests
# ═══════════════════════════════════════════════════════════════════


class TestLaTeXGenerator:
    """Tests for LaTeXGenerator."""

    def test_generate_produces_string(self, minimal_spec: PaperSpec) -> None:
        gen_inst = LaTeXGenerator(minimal_spec)
        tex = gen_inst.generate()
        assert isinstance(tex, str)
        assert len(tex) > 100

    def test_document_structure(self, minimal_spec: PaperSpec) -> None:
        tex = LaTeXGenerator(minimal_spec).generate()
        assert "\\begin{document}" in tex
        assert "\\end{document}" in tex
        assert "\\maketitle" in tex

    def test_title_and_author(self, minimal_spec: PaperSpec) -> None:
        tex = LaTeXGenerator(minimal_spec).generate()
        assert "\\title{Test Paper}" in tex
        assert "\\author{Test Author}" in tex
        assert "\\email{test@test.edu}" in tex

    def test_abstract(self, minimal_spec: PaperSpec) -> None:
        tex = LaTeXGenerator(minimal_spec).generate()
        assert "\\begin{abstract}" in tex
        assert "This is a test abstract." in tex
        assert "\\end{abstract}" in tex

    def test_umcp_macros_included(self, minimal_spec: PaperSpec) -> None:
        tex = LaTeXGenerator(minimal_spec).generate()
        assert "\\newcommand{\\tR}" in tex
        assert "\\newcommand{\\Gam}" in tex
        assert "\\newcommand{\\eps}" in tex
        assert "\\newcommand{\\beq}" in tex

    def test_theorem_envs_included(self, minimal_spec: PaperSpec) -> None:
        tex = LaTeXGenerator(minimal_spec).generate()
        assert "\\newtheorem{theorem}" in tex
        assert "\\newtheorem{definition}" in tex

    def test_sha256_stamp(self, minimal_spec: PaperSpec) -> None:
        tex = LaTeXGenerator(minimal_spec).generate()
        assert "% SHA-256 (first 16):" in tex

    def test_generation_timestamp(self, minimal_spec: PaperSpec) -> None:
        tex = LaTeXGenerator(minimal_spec).generate()
        assert "% Generated:" in tex

    def test_bibliography_line(self, minimal_spec: PaperSpec) -> None:
        tex = LaTeXGenerator(minimal_spec).generate()
        assert "\\bibliography{Bibliography}" in tex

    def test_revtex_docclass(self, minimal_spec: PaperSpec) -> None:
        tex = LaTeXGenerator(minimal_spec).generate()
        assert "revtex4-2" in tex

    def test_packages(self, minimal_spec: PaperSpec) -> None:
        tex = LaTeXGenerator(minimal_spec).generate()
        assert "amsmath" in tex
        assert "hyperref" in tex
        assert "graphicx" in tex

    def test_section_appears(self, minimal_spec: PaperSpec) -> None:
        tex = LaTeXGenerator(minimal_spec).generate()
        assert "\\section{Introduction}" in tex
        assert "Test body content." in tex

    def test_extra_macros(self) -> None:
        spec = PaperSpec(
            title="T",
            authors=[AuthorSpec(name="A", affiliation="B")],
            abstract="C",
            extra_macros="\\newcommand{\\mycommand}{42}",
        )
        tex = LaTeXGenerator(spec).generate()
        assert "\\newcommand{\\mycommand}{42}" in tex

    def test_acknowledgments(self) -> None:
        spec = PaperSpec(
            title="T",
            authors=[AuthorSpec(name="A", affiliation="B")],
            abstract="C",
            acknowledgments="Thank you.",
        )
        tex = LaTeXGenerator(spec).generate()
        assert "\\begin{acknowledgments}" in tex
        assert "Thank you." in tex

    def test_deterministic(self, minimal_spec: PaperSpec) -> None:
        """Same spec → same output (ignoring timestamp)."""
        # We need a spec with fixed timestamp for determinism
        spec = PaperSpec(
            title="Det",
            authors=[AuthorSpec(name="A", affiliation="B")],
            abstract="C",
            generation_timestamp="2025-01-01T00:00:00Z",
        )
        tex1 = LaTeXGenerator(spec).generate()
        tex2 = LaTeXGenerator(spec).generate()
        assert tex1 == tex2


# ═══════════════════════════════════════════════════════════════════
# Convenience Builder Tests
# ═══════════════════════════════════════════════════════════════════


class TestConvenienceBuilders:
    """Tests for frozen_constants_table, tier_mapping_table, etc."""

    def test_frozen_constants_table(self) -> None:
        tex = frozen_constants_table(
            [
                ("\\alpha", "1.0", "Coupling"),
                ("\\eps", "1e-8", "Guard"),
            ]
        )
        assert "\\begin{table}" in tex
        assert "Coupling" in tex
        assert "$\\alpha$" in tex

    def test_tier_mapping_table(self) -> None:
        tex = tier_mapping_table(
            [
                ("Tier-1", "Constants"),
                ("Tier-0", "Gates"),
            ]
        )
        assert "Tier-1" in tex
        assert "Constants" in tex

    def test_itemize_block(self) -> None:
        tex = itemize_block(["First", "Second"])
        assert "\\begin{itemize}" in tex
        assert "\\item First" in tex
        assert "\\item Second" in tex
        assert "\\end{itemize}" in tex

    def test_enumerate_block(self) -> None:
        tex = enumerate_block(["A", "B"])
        assert "\\begin{enumerate}" in tex
        assert "\\item A" in tex
        assert "\\end{enumerate}" in tex


# ═══════════════════════════════════════════════════════════════════
# Compiler Tests (unit-level, mocked subprocess)
# ═══════════════════════════════════════════════════════════════════


class TestLaTeXCompiler:
    """Tests for LaTeXCompiler — subprocess calls are mocked."""

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            LaTeXCompiler(tmp_path / "nonexistent.tex")

    def test_missing_engine(self, tmp_path: Path) -> None:
        tex = tmp_path / "test.tex"
        tex.write_text("\\documentclass{article}\\begin{document}Hello\\end{document}")
        with patch("shutil.which", return_value=None):
            compiler = LaTeXCompiler(tex)
            result = compiler.compile()
            assert result.status == CompileStatus.FAILURE
            assert "not found" in result.errors[0]


# ═══════════════════════════════════════════════════════════════════
# Validator Tests
# ═══════════════════════════════════════════════════════════════════


class TestLaTeXValidator:
    """Tests for LaTeXValidator."""

    def test_no_log_file(self, tmp_path: Path) -> None:
        tex = tmp_path / "test.tex"
        tex.write_text("content")
        v = LaTeXValidator(tex)
        result = v.validate()
        assert not result.is_valid
        assert not result.pdf_exists

    def test_clean_log(self, tmp_path: Path) -> None:
        tex = tmp_path / "test.tex"
        tex.write_text("content")
        log = tmp_path / "test.log"
        log.write_text("Output written on test.pdf (3 pages)\n")
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.5 fake")
        v = LaTeXValidator(tex)
        result = v.validate()
        assert result.is_valid
        assert result.page_count == 3
        assert result.pdf_exists
        assert len(result.undefined_refs) == 0

    def test_undefined_ref(self, tmp_path: Path) -> None:
        tex = tmp_path / "test.tex"
        tex.write_text("content")
        log = tmp_path / "test.log"
        log.write_text(
            "LaTeX Warning: Reference `thm:missing' on page 2 undefined.\nOutput written on test.pdf (1 page)\n"
        )
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.5 fake")
        v = LaTeXValidator(tex)
        result = v.validate()
        assert not result.is_valid
        assert "thm:missing" in result.undefined_refs

    def test_undefined_citation(self, tmp_path: Path) -> None:
        tex = tmp_path / "test.tex"
        tex.write_text("content")
        log = tmp_path / "test.log"
        log.write_text(
            "LaTeX Warning: Citation `nobody2025' on page 1 undefined.\nOutput written on test.pdf (1 page)\n"
        )
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.5 fake")
        v = LaTeXValidator(tex)
        result = v.validate()
        assert not result.is_valid
        assert "nobody2025" in result.undefined_citations

    def test_overfull_count(self, tmp_path: Path) -> None:
        tex = tmp_path / "test.tex"
        tex.write_text("content")
        log = tmp_path / "test.log"
        log.write_text(
            "Overfull \\hbox (3pt too wide)\n"
            "Overfull \\hbox (1pt too wide)\n"
            "Underfull \\vbox detected\n"
            "Output written on test.pdf (1 page)\n"
        )
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.5 fake")
        v = LaTeXValidator(tex)
        result = v.validate()
        assert result.overfull_boxes == 2
        assert result.underfull_boxes == 1

    def test_summary_output(self, tmp_path: Path) -> None:
        tex = tmp_path / "test.tex"
        tex.write_text("content")
        log = tmp_path / "test.log"
        log.write_text("Output written on test.pdf (2 pages)\n")
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.5 fake")
        v = LaTeXValidator(tex)
        result = v.validate()
        summary = result.summary()
        assert "PASS" in summary
        assert "Pages: 2" in summary


# ═══════════════════════════════════════════════════════════════════
# YAML Spec Loader Tests
# ═══════════════════════════════════════════════════════════════════


class TestYAMLSpecLoader:
    """Tests for load_spec_from_yaml."""

    def test_load_minimal_yaml(self, tmp_path: Path) -> None:
        yaml_content = textwrap.dedent("""\
            title: "Test Title"
            authors:
              - name: "Author A"
                affiliation: "Univ"
            abstract: "Abstract text"
            sections:
              - title: "Intro"
                label: "sec:intro"
                body: "Body text"
        """)
        yaml_file = tmp_path / "spec.yaml"
        yaml_file.write_text(yaml_content)
        spec = gen.load_spec_from_yaml(yaml_file)
        assert spec.title == "Test Title"
        assert len(spec.authors) == 1
        assert spec.authors[0].name == "Author A"
        assert len(spec.sections) == 1
        assert spec.sections[0].label == "sec:intro"

    def test_load_with_document_class(self, tmp_path: Path) -> None:
        yaml_content = textwrap.dedent("""\
            title: "T"
            authors:
              - name: "A"
                affiliation: "B"
            document_class: "preprint"
        """)
        yaml_file = tmp_path / "spec.yaml"
        yaml_file.write_text(yaml_content)
        spec = gen.load_spec_from_yaml(yaml_file)
        assert spec.document_class == DocumentClass.PREPRINT

    def test_load_unknown_docclass_defaults(self, tmp_path: Path) -> None:
        yaml_content = textwrap.dedent("""\
            title: "T"
            authors:
              - name: "A"
                affiliation: "B"
            document_class: "unknown-class"
        """)
        yaml_file = tmp_path / "spec.yaml"
        yaml_file.write_text(yaml_content)
        spec = gen.load_spec_from_yaml(yaml_file)
        assert spec.document_class == DocumentClass.REVTEX_APS_PRD


# ═══════════════════════════════════════════════════════════════════
# Integration: Generate → File
# ═══════════════════════════════════════════════════════════════════


class TestGenerateToFile:
    """Test writing generated LaTeX to a file."""

    def test_write_to_file(self, minimal_spec: PaperSpec, tmp_path: Path) -> None:
        output = tmp_path / "paper.tex"
        _, _, _ = generate_and_compile(minimal_spec, output, compile_pdf=False)
        assert output.exists()
        content = output.read_text()
        assert "\\begin{document}" in content
        assert "\\end{document}" in content

    def test_creates_parent_dirs(self, minimal_spec: PaperSpec, tmp_path: Path) -> None:
        output = tmp_path / "deep" / "nested" / "paper.tex"
        generate_and_compile(minimal_spec, output, compile_pdf=False)
        assert output.exists()

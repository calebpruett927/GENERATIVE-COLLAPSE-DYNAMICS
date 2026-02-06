"""
Tests for outputs.py - Output formatters.

These tests verify that output formatters produce valid
structured output in various formats.
"""

from umcp.outputs import (
    ASCIIGenerator,
    BadgeGenerator,
    HTMLGenerator,
    JSONLDGenerator,
    JUnitGenerator,
    LaTeXGenerator,
    MarkdownGenerator,
    MermaidGenerator,
    RegimeState,
    SARIFGenerator,
    ValidationSummary,
    generate_all_badges,
)


class TestRegimeState:
    """Tests for the RegimeState dataclass."""

    def test_regime_creation(self):
        """RegimeState can be created."""
        regime = RegimeState(omega=0.5, F=0.85, S=0.001, C=0.1, regime="STABLE")
        assert regime.regime == "STABLE"
        assert regime.F == 0.85

    def test_regime_from_values(self):
        """RegimeState.from_values classifies correctly."""
        regime = RegimeState.from_values(omega=0.5, F=0.85, S=0.001, C=0.1)
        assert regime.regime == "STABLE"

    def test_regime_critical(self):
        """High S triggers CRITICAL."""
        regime = RegimeState.from_values(omega=0.5, F=0.85, S=0.05, C=0.1)
        assert regime.regime == "CRITICAL"


class TestValidationSummary:
    """Tests for the ValidationSummary dataclass."""

    def test_summary_creation(self):
        """ValidationSummary can be created."""
        summary = ValidationSummary(
            status="CONFORMANT",
            errors=0,
            warnings=0,
        )
        assert summary.status == "CONFORMANT"
        assert summary.errors == 0

    def test_summary_with_optional(self):
        """ValidationSummary with optional fields."""
        regime = RegimeState.from_values(omega=0.5, F=0.85, S=0.001, C=0.1)
        summary = ValidationSummary(
            status="CONFORMANT",
            errors=0,
            warnings=1,
            casepack_id="gcd_complete",
            regime=regime,
        )
        assert summary.warnings == 1
        assert summary.regime is not None


class TestBadgeGenerator:
    """Tests for the BadgeGenerator class."""

    def test_badge_generator_class_methods(self):
        """BadgeGenerator has class methods."""
        assert hasattr(BadgeGenerator, "status_badge")

    def test_status_badge(self):
        """Status badge can be generated."""
        svg = BadgeGenerator.status_badge("CONFORMANT")
        assert "<svg" in svg
        assert "CONFORMANT" in svg

    def test_badge_colors(self):
        """Badge colors are defined."""
        assert "CONFORMANT" in BadgeGenerator.COLORS
        assert "NONCONFORMANT" in BadgeGenerator.COLORS


class TestASCIIGenerator:
    """Tests for the ASCIIGenerator class."""

    def test_ascii_generator_creation(self):
        """ASCIIGenerator can be created."""
        gen = ASCIIGenerator()
        assert gen is not None

    def test_ascii_regime_gauge(self):
        """ASCIIGenerator has regime_gauge method."""
        output = ASCIIGenerator.regime_gauge(omega=0.5)
        assert isinstance(output, str)
        assert len(output) > 0
        assert "ω=" in output

    def test_ascii_sparkline(self):
        """ASCIIGenerator has sparkline method."""
        output = ASCIIGenerator.sparkline([0.5, 0.6, 0.7, 0.8])
        assert isinstance(output, str)


class TestMarkdownGenerator:
    """Tests for the MarkdownGenerator class."""

    def test_markdown_generator_creation(self):
        """MarkdownGenerator can be created."""
        gen = MarkdownGenerator()
        assert gen is not None


class TestMermaidGenerator:
    """Tests for the MermaidGenerator class."""

    def test_mermaid_generator_creation(self):
        """MermaidGenerator can be created."""
        gen = MermaidGenerator()
        assert gen is not None

    def test_mermaid_has_diagram_method(self):
        """MermaidGenerator has diagram method."""
        gen = MermaidGenerator()
        assert hasattr(gen, "regime_state_diagram")


class TestLaTeXGenerator:
    """Tests for the LaTeXGenerator class."""

    def test_latex_generator_creation(self):
        """LaTeXGenerator can be created."""
        gen = LaTeXGenerator()
        assert gen is not None


class TestJSONLDGenerator:
    """Tests for the JSONLDGenerator class."""

    def test_jsonld_generator_creation(self):
        """JSONLDGenerator can be created."""
        gen = JSONLDGenerator()
        assert gen is not None


class TestSARIFGenerator:
    """Tests for the SARIFGenerator class."""

    def test_sarif_generator_creation(self):
        """SARIFGenerator can be created."""
        gen = SARIFGenerator()
        assert gen is not None


class TestJUnitGenerator:
    """Tests for the JUnitGenerator class."""

    def test_junit_generator_creation(self):
        """JUnitGenerator can be created."""
        gen = JUnitGenerator()
        assert gen is not None


class TestHTMLGenerator:
    """Tests for the HTMLGenerator class."""

    def test_html_generator_creation(self):
        """HTMLGenerator can be created."""
        gen = HTMLGenerator()
        assert gen is not None


class TestGenerateFunctions:
    """Tests for module-level generate functions."""

    def test_generate_all_badges(self):
        """generate_all_badges produces dict of badges."""
        summary = ValidationSummary(
            status="CONFORMANT",
            errors=0,
            warnings=0,
        )
        badges = generate_all_badges(summary)
        assert isinstance(badges, dict)
        # Should have at least status badge
        assert len(badges) > 0

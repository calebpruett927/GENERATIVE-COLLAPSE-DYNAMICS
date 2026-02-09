"""
Test suite for RCFT Tier-2 layering and hierarchy validation

Validates:
- Tier hierarchy: UMCP (Tier-0) → GCD (Tier-1) → RCFT (Tier-2)
- Tier-1 symbols remain frozen and unchanged
- RCFT augments without overriding
- Contract inheritance chain works correctly
- RCFT casepack validates with all tiers
- Multi-tier regime classifications coexist
"""

import json
from pathlib import Path

import pytest
import yaml


@pytest.fixture
def uma_contract():
    """Load UMA contract (implicit Tier-0)."""
    path = Path("contracts/UMA.INTSTACK.v1.yaml")
    with open(path) as f:
        full = yaml.safe_load(f)
        return full["contract"]


@pytest.fixture
def gcd_contract():
    """Load GCD contract (Tier-1)."""
    path = Path("contracts/GCD.INTSTACK.v1.yaml")
    with open(path) as f:
        full = yaml.safe_load(f)
        return full["contract"]


@pytest.fixture
def rcft_contract():
    """Load RCFT contract (Tier-2)."""
    path = Path("contracts/RCFT.INTSTACK.v1.yaml")
    with open(path) as f:
        full = yaml.safe_load(f)
        return full["contract"]


@pytest.fixture
def gcd_canon():
    """Load GCD canon."""
    path = Path("canon/gcd_anchors.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


@pytest.fixture
def rcft_canon():
    """Load RCFT canon."""
    path = Path("canon/rcft_anchors.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


class TestTierHierarchy:
    """Tests for tier hierarchy structure."""

    def test_gcd_is_tier1(self, gcd_contract, gcd_canon):
        """GCD should be Tier-1."""
        assert gcd_contract["tier_level"] == 1
        assert gcd_canon["scope"]["tier"] == 1

    def test_rcft_is_tier2(self, rcft_contract, rcft_canon):
        """RCFT should be Tier-2."""
        assert rcft_contract["tier_level"] == 2
        assert rcft_canon["scope"]["tier"] == 2

    def test_gcd_extends_uma(self, gcd_contract):
        """GCD should extend UMA.INTSTACK.v1."""
        assert gcd_contract["parent_contract"] == "UMA.INTSTACK.v1"

    def test_rcft_extends_gcd(self, rcft_contract):
        """RCFT should extend GCD.INTSTACK.v1."""
        assert rcft_contract["parent_contract"] == "GCD.INTSTACK.v1"

    def test_tier_chain(self, uma_contract, gcd_contract, rcft_contract):
        """Full chain should be UMA → GCD → RCFT."""
        # UMA has no parent (Tier-0 / base)
        assert "parent_contract" not in uma_contract or uma_contract.get("parent_contract") is None

        # GCD extends UMA
        assert gcd_contract["parent_contract"] == "UMA.INTSTACK.v1"

        # RCFT extends GCD
        assert rcft_contract["parent_contract"] == "GCD.INTSTACK.v1"


class TestTier1FrozenSymbols:
    """Tests that Tier-1 symbols remain frozen in Tier-2."""

    def test_gcd_defines_tier1_symbols(self, gcd_contract):
        """GCD should define Tier-1 symbols."""
        reserved = gcd_contract["tier_1_kernel"]["reserved_symbols"]

        # Core Tier-1
        assert "omega" in reserved
        assert "F" in reserved
        assert "S" in reserved
        assert "C" in reserved

        # GCD Tier-1 extensions (using Greek symbols)
        assert "E_potential" in reserved
        assert "Φ_collapse" in reserved
        assert "Φ_gen" in reserved
        assert "R" in reserved

    def test_rcft_lists_frozen_tier1(self, rcft_canon):
        """RCFT canon should explicitly list frozen Tier-1 symbols."""
        frozen = rcft_canon["tier_hierarchy"]["tier_1_frozen_symbols"]

        # All GCD Tier-1 symbols should be in frozen list
        assert "ω" in frozen or "omega" in str(frozen)
        assert "F" in frozen
        assert "S" in frozen
        assert "C" in frozen
        assert "E_potential" in frozen
        assert "Φ_collapse" in frozen or "Phi_collapse" in str(frozen)
        assert "Φ_gen" in frozen or "Phi_gen" in str(frozen)
        assert "R" in frozen

    def test_rcft_does_not_redefine_tier1(self, rcft_canon):
        """RCFT Tier-2 extensions should not redefine Tier-1 symbols."""
        tier2_symbols = [s["symbol"] for s in rcft_canon["tier_2_extensions"]["reserved_symbols"]]
        frozen_tier1 = rcft_canon["tier_hierarchy"]["tier_1_frozen_symbols"]

        # No overlap between Tier-2 and frozen Tier-1
        for symbol in tier2_symbols:
            assert symbol not in frozen_tier1, f"Tier-2 symbol {symbol} conflicts with frozen Tier-1"

    def test_rcft_principles_acknowledge_freeze(self, rcft_canon):
        """RCFT principles should state "Tier-2 augments, never overrides"."""
        principles = rcft_canon["principles"]
        p0 = next(p for p in principles if p["id"] == "P-RCFT-0")

        assert "augments" in p0["statement"].lower()
        assert "override" in p0["statement"].lower()


class TestAugmentationNotReplacement:
    """Tests that RCFT augments GCD without replacing functionality."""

    def test_rcft_inherits_all_gcd_closures(self, rcft_contract):
        """RCFT contract should reference all GCD closures."""
        closures = rcft_contract["closures"]
        # Closures are a dict with closure names as keys

        # All 4 GCD closures should be present
        assert "energy_potential" in closures
        assert "entropic_collapse" in closures
        assert "generative_flux" in closures
        assert "field_resonance" in closures

    def test_rcft_adds_new_closures(self, rcft_contract):
        """RCFT should add 3 new closures without removing GCD closures."""
        closures = rcft_contract["closures"]
        # Closures are a dict with closure names as keys

        # New RCFT closures
        assert "fractal_dimension" in closures
        assert "recursive_field" in closures
        assert "resonance_pattern" in closures

        # Total should be 7 (4 GCD + 3 RCFT)
        assert len(closures) == 7

    def test_rcft_inherits_gcd_regimes(self, rcft_contract):
        """RCFT should inherit all GCD regime classifications."""
        regimes = rcft_contract["regime_classification"]

        # GCD regimes should all be present (using actual regime names)
        assert "energy" in regimes
        assert "collapse" in regimes
        assert "flux" in regimes
        assert "resonance" in regimes

    def test_rcft_adds_new_regimes(self, rcft_contract):
        """RCFT should add new regime classifications."""
        regimes = rcft_contract["regime_classification"]

        # New RCFT regimes (using actual regime names)
        assert "fractal" in regimes
        assert "recursive" in regimes
        assert "pattern" in regimes


class TestMultiTierValidation:
    """Tests that multi-tier system validates correctly."""

    def test_rcft_casepack_exists(self):
        """RCFT casepack should exist."""
        manifest = Path("casepacks/rcft_complete/manifest.json")
        assert manifest.exists()

    def test_rcft_casepack_references_rcft_contract(self):
        """RCFT casepack should reference RCFT.INTSTACK.v1."""
        manifest_path = Path("casepacks/rcft_complete/manifest.json")
        with open(manifest_path) as f:
            manifest = json.load(f)

        assert manifest["refs"]["contract"]["id"] == "RCFT.INTSTACK.v1"
        assert "rcft" in manifest["casepack"]["id"].lower()

    def test_rcft_casepack_has_all_tier_outputs(self):
        """RCFT casepack should have outputs for all tiers."""
        manifest_path = Path("casepacks/rcft_complete/manifest.json")
        with open(manifest_path) as f:
            manifest = json.load(f)

        expected_outputs = manifest["artifacts"]["expected"]

        # Tier-0/1 outputs
        assert "invariants_json" in expected_outputs

        # GCD Tier-1 outputs
        assert "gcd_energy_json" in expected_outputs
        assert "gcd_collapse_json" in expected_outputs
        assert "gcd_flux_json" in expected_outputs
        assert "gcd_resonance_json" in expected_outputs

        # RCFT Tier-2 outputs
        assert "rcft_fractal_json" in expected_outputs
        assert "rcft_recursive_json" in expected_outputs
        assert "rcft_pattern_json" in expected_outputs

    def test_rcft_receipt_validates_all_tiers(self):
        """RCFT receipt should validate identities from all tiers."""
        receipt_path = Path("casepacks/rcft_complete/expected/seam_receipt.json")
        with open(receipt_path) as f:
            receipt = json.load(f)

        assert receipt["contract"] == "RCFT.INTSTACK.v1"
        assert receipt["tier"] == 2
        assert receipt["parent_contract"] == "GCD.INTSTACK.v1"

        # Should have Tier-1 regimes
        assert "tier_1_regimes" in receipt
        tier1_regimes = receipt["tier_1_regimes"]
        assert "gcd_energy" in tier1_regimes
        assert "gcd_resonance" in tier1_regimes

        # Should have Tier-2 regimes
        assert "tier_2_regimes" in receipt
        tier2_regimes = receipt["tier_2_regimes"]
        assert "rcft_fractal" in tier2_regimes
        assert "rcft_recursive" in tier2_regimes

        # Tier hierarchy should be validated
        assert receipt["tier_hierarchy_validated"] is True
        assert receipt["tier_1_frozen_symbols_unchanged"] is True


class TestRegistryIntegration:
    """Tests that registry integrates all tiers correctly."""

    def test_registry_has_gcd_section(self):
        """Registry should have GCD extension section."""
        registry_path = Path("closures/registry.yaml")
        with open(registry_path) as f:
            registry = yaml.safe_load(f)

        assert "extensions" in registry["registry"]
        assert "gcd" in registry["registry"]["extensions"]

        gcd_closures = registry["registry"]["extensions"]["gcd"]
        assert len(gcd_closures) == 5  # Updated: 4 original + momentum_flux

    def test_registry_has_rcft_section(self):
        """Registry should have RCFT extension section."""
        registry_path = Path("closures/registry.yaml")
        with open(registry_path) as f:
            registry = yaml.safe_load(f)

        assert "rcft" in registry["registry"]["extensions"]

        rcft_closures = registry["registry"]["extensions"]["rcft"]
        assert len(rcft_closures) == 5  # 3 original + attractor_basin + active_matter

    def test_registry_closure_paths_resolve(self):
        """All closure paths in registry should resolve."""
        registry_path = Path("closures/registry.yaml")
        with open(registry_path) as f:
            registry = yaml.safe_load(f)

        extensions = registry["registry"]["extensions"]

        # Check GCD closures
        for closure in extensions["gcd"]:
            path = Path(closure["path"])
            assert path.exists(), f"GCD closure not found: {path}"

        # Check RCFT closures
        for closure in extensions["rcft"]:
            path = Path(closure["path"])
            assert path.exists(), f"RCFT closure not found: {path}"


class TestZeroEntropyAcrossTiers:
    """Tests zero entropy state across all tiers."""

    def test_tier1_zero_entropy_values(self):
        """GCD Tier-1 zero entropy values."""
        invariants_path = Path("casepacks/rcft_complete/expected/invariants.json")
        with open(invariants_path) as f:
            inv = json.load(f)

        # Access values from rows array (new format)
        row = inv["rows"][0]
        assert row["omega"] == 0.0
        assert row["F"] == 1.0
        assert row["S"] == 0.0
        assert row["C"] == 0.0

    def test_tier1_gcd_zero_entropy_values(self):
        """GCD extensions at zero entropy."""
        energy_path = Path("casepacks/rcft_complete/expected/gcd_energy.json")
        with open(energy_path) as f:
            energy = json.load(f)

        assert energy["E_potential"] == 0.0
        assert energy["regime"] == "Low"

    def test_tier2_rcft_zero_entropy_values(self):
        """RCFT Tier-2 values at zero entropy."""
        fractal_path = Path("casepacks/rcft_complete/expected/rcft_fractal.json")
        recursive_path = Path("casepacks/rcft_complete/expected/rcft_recursive.json")

        with open(fractal_path) as f:
            fractal = json.load(f)
        assert fractal["D_fractal"] == 0.0  # Point trajectory
        assert fractal["regime"] == "Smooth"

        with open(recursive_path) as f:
            recursive = json.load(f)
        assert recursive["Psi_recursive"] == 0.0  # No memory
        assert recursive["regime"] == "Dormant"

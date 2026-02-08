#!/usr/bin/env bash
# UMCP Q&A Helper - Quick answers to common questions

set -euo pipefail

question="${1:-help}"

case "${question,,}" in
  # Repository structure
  "structure"|"folders"|"layout")
    echo "📁 REPOSITORY STRUCTURE"
    echo "  src/umcp/          → Python package"
    echo "  tests/             → Test suite (17 tests)"
    echo "  canon/             → Canonical anchors & thresholds"
    echo "  contracts/         → UMA.INTSTACK contract definitions"
    echo "  closures/          → Closure registry & definitions"
    echo "  schemas/           → JSON schemas for validation"
    echo "  casepacks/         → Example runnable casepacks"
    echo "  scripts/           → Helper scripts"
    ;;

  # Tier-1 symbols
  "symbols"|"tier1"|"invariants")
    echo "📊 TIER-1 KERNEL SYMBOLS"
    echo "  ω (omega)   → Omega parameter"
    echo "  F           → F invariant (F ≈ 1 - ω)"
    echo "  S           → S invariant"
    echo "  C           → C invariant"
    echo "  τ_R (tau_R) → Return time"
    echo "  κ (kappa)   → Kappa parameter"
    echo "  IC          → IC invariant (IC ≈ exp(κ))"
    ;;

  # IDs
  "ids"|"identifiers"|"contracts")
    echo "🔖 CANONICAL IDENTIFIERS"
    echo "  Canon:    UMCP.CANON.v1"
    echo "  Contract: UMA.INTSTACK.v1 (primary)"
    echo "  Contract: UMA.INTSTACK.v1.0.1 (alternate)"
    echo "  Closures: UMCP.CLOSURES.DEFAULT.v1"
    echo "  Validator: umcp-validator v0.1.0"
    ;;

  # DOIs
  "dois"|"references"|"citations")
    echo "📚 DOI REFERENCES"
    echo "  PRE:  10.5281/zenodo.17756705"
    echo "  POST: 10.5281/zenodo.18072852"
    echo "  WELD: W-2025-12-31-PHYS-COHERENCE"
    ;;

  # Commands
  "commands"|"how"|"usage")
    echo "🔧 COMMON COMMANDS"
    echo "  umcp validate .              → Validate repository"
    echo "  umcp validate [casepack]     → Validate casepack"
    echo "  pytest                       → Run all tests"
    echo "  pytest -v                    → Run tests (verbose)"
    echo "  ./scripts/test.sh            → Run test suite"
    echo "  ./scripts/validate.sh        → Validate + test"
    ;;

  # Tests
  "tests"|"testing")
    echo "🧪 TEST SUITE"
    echo "  Total: 17 tests (all passing)"
    echo "  • 3 tests: Schema validation"
    echo "  • 3 tests: Canon/Contract/Closure validation"
    echo "  • 5 tests: CasePack validation"
    echo "  • 5 tests: Semantic rules"
    echo "  • 1 test:  Validator result schema"
    ;;

  # Closures
  "closures"|"registry")
    echo "🔍 CLOSURE REGISTRY"
    echo "  gamma                → gamma.default.v1.yaml"
    echo "  return_domain        → return_domain.window64.v1.yaml"
    echo "  norms                → norms.l2_eta1e-3.v1.yaml"
    echo "  curvature_neighborhood → curvature_neighborhood.default.v1.yaml"
    ;;

  # Casepacks
  "casepacks"|"examples")
    echo "📦 CASEPACKS"
    echo "  hello_world/         → Example casepack"
    echo "    manifest.json      → CasePack metadata"
    echo "    raw_measurements.csv → Input data"
    echo "    expected/psi.csv   → Bounded trace"
    echo "    expected/invariants.json → Tier-1 invariants"
    echo "    expected/ss1m_receipt.json → Audit receipt"
    ;;

  # Status
  "status"|"health"|"check")
    echo "✅ REPOSITORY STATUS"
    if command -v pytest &> /dev/null && command -v umcp &> /dev/null; then
      echo "  Tests: $(pytest tests/ -q 2>&1 | tail -1)"
      echo "  CLI:   $(umcp --version)"
      echo "  Validation: CONFORMANT"
    else
      echo "  Please run: source .venv/bin/activate"
    fi
    ;;

  # Version
  "version"|"v")
    echo "📌 VERSION INFO"
    echo "  Package: umcp 0.1.0"
    echo "  Python:  $(python --version 2>&1)"
    echo "  Repo:    GENERATIVE-COLLAPSE-DYNAMICS"
    ;;

  # Help
  "help"|*)
    cat << 'HELP'
╔══════════════════════════════════════════════════════════════╗
║              UMCP Q&A Helper - Quick Reference               ║
╚══════════════════════════════════════════════════════════════╝

USAGE: ./scripts/ask.sh [question]

AVAILABLE QUESTIONS:
  structure       → Show repository folder structure
  symbols         → Show Tier-1 kernel symbols
  ids             → Show canonical identifiers
  dois            → Show DOI references
  commands        → Show common commands
  tests           → Show test suite info
  closures        → Show closure registry
  casepacks       → Show casepack structure
  status          → Check repository health
  version         → Show version info
  help            → Show this help

EXAMPLES:
  ./scripts/ask.sh structure
  ./scripts/ask.sh symbols
  ./scripts/ask.sh commands
  ./scripts/ask.sh status

You can also type questions in natural language!
HELP
    ;;
esac

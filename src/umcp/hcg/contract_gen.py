"""
Contract Page Generator — Renders contract YAML as browsable markdown.

Reads the frozen contract definition for a domain and produces a
structured markdown page detailing the contract's configuration:
embedding parameters, reserved symbols, frozen parameters, and
regime thresholds.

Usage:
    from umcp.hcg.contract_gen import generate_contract_markdown
    from umcp.hcg.extractor import extract_domain_data

    data = extract_domain_data("standard_model")
    md = generate_contract_markdown(data)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from umcp.hcg.extractor import SiteData


def generate_contract_markdown(data: SiteData) -> str:
    """Generate markdown content for a domain's contract page."""
    sections: list[str] = []
    contract = data.contract

    sections.append(f"# Contract — {data.domain_display}\n")
    sections.append(
        "> The contract defines the rules *before* evidence. "
        "All thresholds, embedding parameters, and reserved symbols "
        "are frozen here.\n"
    )

    if not contract:
        sections.append("*No contract found for this domain.*\n")
        return "\n".join(sections)

    # Contract identity
    cblock = contract.get("contract", contract)
    contract_id = cblock.get("id", "")
    version = cblock.get("version", "")
    parent = cblock.get("parent_contract", "")
    tier = cblock.get("tier_level", "")

    sections.append("## Identity\n")
    sections.append("| Field | Value |")
    sections.append("|-------|-------|")
    if contract_id:
        sections.append(f"| **Contract ID** | `{contract_id}` |")
    if version:
        sections.append(f"| **Version** | {version} |")
    if parent:
        sections.append(f"| **Parent Contract** | `{parent}` |")
    if tier:
        sections.append(f"| **Tier Level** | {tier} |")
    sections.append("")

    # Embedding
    embedding = cblock.get("embedding", {})
    if embedding:
        sections.append("## Embedding Configuration\n")
        sections.append("| Parameter | Value |")
        sections.append("|-----------|-------|")
        for key, val in embedding.items():
            sections.append(f"| `{key}` | `{val}` |")
        sections.append("")

    # Reserved symbols
    kernel = cblock.get("tier_1_kernel", {})
    symbols = kernel.get("reserved_symbols", [])
    if symbols:
        sections.append("## Reserved Symbols (Tier-1)\n")
        sections.append(
            "These symbols are frozen within a run. Any Tier-2 code that redefines them is automatic nonconformance.\n"
        )
        # Separate GCD invariants from domain-specific
        gcd_symbols = {"omega", "F", "S", "C", "tau_R", "kappa", "IC", "IC_min"}
        domain_symbols = [s for s in symbols if s not in gcd_symbols]
        gcd_found = [s for s in symbols if s in gcd_symbols]

        if gcd_found:
            sections.append("**GCD Kernel Invariants** (inherited):\n")
            for s in gcd_found:
                sections.append(f"- `{s}`")
            sections.append("")

        if domain_symbols:
            sections.append("**Domain-Specific Symbols**:\n")
            for s in domain_symbols:
                sections.append(f"- `{s}`")
            sections.append("")

    # Frozen parameters
    frozen = kernel.get("frozen_parameters", {})
    if frozen:
        sections.append("## Frozen Parameters\n")
        sections.append("*Consistent across the seam — same rules both sides of every collapse-return boundary.*\n")
        sections.append("| Parameter | Value |")
        sections.append("|-----------|-------|")
        for key, val in frozen.items():
            sections.append(f"| `{key}` | `{val}` |")
        sections.append("")

    # Weights policy
    wp = kernel.get("weights_policy", "")
    if wp:
        sections.append(f"**Weights Policy**: `{wp}`\n")

    # Regime thresholds
    _render_thresholds(cblock, sections)

    # Footer
    sections.append("---\n")
    sections.append(f"*Contract frozen by the Headless Contract Gateway (HCG) · Domain: {data.domain} · UMCP v2.3.1*\n")

    return "\n".join(sections)


def _render_thresholds(cblock: dict[str, Any], sections: list[str]) -> None:
    """Render regime thresholds if present."""
    thresholds = cblock.get("regime_thresholds", cblock.get("thresholds", {}))
    if not thresholds:
        return

    sections.append("## Regime Thresholds\n")
    sections.append("| Gate | Condition |")
    sections.append("|------|-----------|")
    for gate, condition in thresholds.items():
        sections.append(f"| `{gate}` | `{condition}` |")
    sections.append("")

---
title: "Contract — Evolution & Neuroscience"
description: "Frozen contract for Evolution & Neuroscience"
domain: evolution
pageType: contract
---

# Contract — Evolution & Neuroscience

> The contract defines the rules *before* evidence. All thresholds, embedding parameters, and reserved symbols are frozen here.

## Identity

| Field | Value |
|-------|-------|
| **Contract ID** | `EVO.INTSTACK.v1` |
| **Version** | 1.0.0 |
| **Parent Contract** | `GCD.INTSTACK.v1` |
| **Tier Level** | 2 |

## Embedding Configuration

| Parameter | Value |
|-----------|-------|
| `interval` | `[0.0, 1.0]` |
| `face` | `pre_clip` |
| `epsilon` | `1e-08` |
| `channels` | 8 |
| `weights` | Equal: w_i = 1/8 |

## Reserved Symbols (Tier-1)

**GCD Kernel Invariants** (inherited):

- `F`, `S`, `C`, `IC`, `IC_min`, `omega`, `tau_R`, `kappa`

**Domain-Specific Symbols**:

- `genetic_diversity` — normalized heterozygosity (H_e / 0.95)
- `morphological_fitness` — cell type count / 32
- `reproductive_success` — log₁₀(R₀) / log₁₀(R₀_max)
- `metabolic_efficiency` — ATP yield / 32
- `immune_competence` — immune layers / 4
- `environmental_breadth` — n_habitats / 14
- `behavioral_complexity` — log₂(ethogram_size) / log₂(10000)
- `lineage_persistence` — geological_Ma / 3800 Ma

## 8-Channel Evolution Kernel

| Channel | Formula | Measures |
|---------|---------|----------|
| `genetic_diversity` | H_e / 0.95 | Population heterozygosity |
| `morphological_fitness` | cell_type_count / 32 | Structural complexity |
| `reproductive_success` | log₁₀(R₀) / log₁₀(R₀_max) | Net reproductive rate |
| `metabolic_efficiency` | ATP_yield / 32 | Energy extraction efficiency |
| `immune_competence` | immune_layers / 4 | Immune system layering |
| `environmental_breadth` | n_habitats / 14 | Niche breadth |
| `behavioral_complexity` | log₂(ethogram) / log₂(10000) | Behavioral repertoire |
| `lineage_persistence` | geological_Ma / 3800 | Deep-time survival |

## Evolutionary Axioms

- **AX-EV0**: Natural selection is the collapse operator (F = fitness, ω = mutation)
- **AX-EV1**: Extinction is generative collapse (mass extinctions → adaptive radiation)
- **AX-EV2**: Lineage persistence measures return fidelity

## Dataset

**40 organisms** spanning: Prokaryotes (4), Unicellular eukaryotes (3), Plants (3), Invertebrates (5), Fish (3), Amphibians (2), Reptiles (3), Birds (3), Mammals (6), Extinct (4), and additional lineages.

---

*Contract frozen by the Headless Contract Gateway (HCG) · Domain: evolution · UMCP v2.3.1*

*No contract found for this domain.*

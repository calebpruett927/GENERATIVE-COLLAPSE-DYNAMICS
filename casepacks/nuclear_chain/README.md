# Nuclear Chain Casepack

**Contract**: NUC.INTSTACK.v1  
**Experiments**: 30  
**Domain**: Nuclear Physics  
**Canon anchors**: `canon/nuc_anchors.yaml`

## Overview

This casepack maps nuclear structure and decay dynamics into the GCD invariant
framework.  The fundamental insight: **the iron peak (Ni-62, BE/A = 8.7945
MeV/nucleon) is the collapse attractor**, with both fusion (light side) and
decay/fission (heavy side) converging on it.

## Frozen Choices (per expert review 2026-02-08)

| Choice | Frozen Value | Alternative | Rationale |
|--------|-------------|-------------|-----------|
| Lifetime | Mean lifetime τ = T½/ln(2) | Half-life T½ | τ is the 1/e constant; T½ is the median |
| Peak reference | Ni-62 (8.7945 MeV/nucleon) | Fe-56 (8.7903) | AME2020 data; Ni-62 is unambiguous max |
| Decay scaling | Geiger-Nuttall: log₁₀(T½) = a/√Q_α + b | 1/ΔBE | Gamow tunneling derivation |
| Q-value | Q_α = M_parent − M_daughter − M_He4 | ΔBE/A | Precise decay energy |
| SEMF set | a_V=15.67, a_S=17.23, a_C=0.714, a_A=23.29, a_P=11.2 | Various fits | von Weizsäcker 1935 |

## Subdomains (6)

| # | Subdomain | Experiments | Closure |
|---|-----------|-------------|---------|
| 1 | Nuclide binding | NUC01–NUC10 | `nuclide_binding.py` |
| 2 | Alpha decay | NUC11–NUC17 | `alpha_decay.py` |
| 3 | Decay chains | NUC18–NUC20 | `decay_chain.py` |
| 4 | Fissility | NUC21–NUC24 | `fissility.py` |
| 5 | Shell structure | NUC25–NUC27 | `shell_structure.py` |
| 6 | Double-sided collapse | NUC28–NUC30 | `double_sided_collapse.py` |

## Key Experiments

- **NUC06** (Ni-62): The peak — ω_eff = 0, F_eff = 1, Stable
- **NUC17** (Bi-209): Quasi-stable, τ ≈ 8.7×10²⁶ s — Eternal decay regime
- **NUC18** (U-238 chain): 14 steps, 8α + 6β⁻, → Pb-206
- **NUC28** (H-1): Light extreme — ω_eff = 1.0 (no binding), Collapse
- **NUC29** (Ni-62): At peak — ω_eff = 0.0, AtPeak regime

## Regenerating Expected Outputs

```bash
cd casepacks/nuclear_chain
python generate_expected.py
```

## References

- von Weizsäcker (1935), Z. Phys. 96, 431
- Bohr & Wheeler (1939), Phys. Rev. 56, 426
- Geiger & Nuttall (1911), Phil. Mag. 22, 613
- Gamow (1928), Z. Phys. 51, 204
- Wang et al. (2021), AME2020, Chinese Physics C 45, 030003

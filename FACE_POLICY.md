# Face Policy Admissibility (Boundary Governance)

**Version**: 1.0.0  
**Status**: Protocol Foundation  
**Domain**: Boundary handling, clipping, and OOR (out-of-range) governance  
**Last Updated**: 2026-01-21

---

## Overview

**Definition**: Because the trace lives in a bounded cube **Ψ(t) ∈ [0,1]ⁿ**, boundary behavior is real: coordinates can saturate at the faces. A **face policy** specifies how boundary values and clipping are handled and how modifications are flagged.

**Why it matters**: Face policy is **load-bearing** wherever return neighborhoods and log terms are evaluated. Therefore admissible face policies must be restricted and audit-visible.

---

## Admissible Face Policy (Definition 1)

A face policy `face` is **admissible** only if it satisfies:

| Criterion | Definition | Example |
|-----------|------------|---------|
| **(i) Locality** | Operates per-sample at time t (no dependence on future samples) | Clip c_i(t) to [ε, 1-ε] using only c_i(t) |
| **(ii) Determinism** | Replayable from declared rules (no hidden stochasticity) | Fixed threshold ε = 1e-8 |
| **(iii) Idempotence** | Applying policy twice yields same result as once | clip(clip(x)) = clip(x) |
| **(iv) Audit visibility** | Any modification produces explicit flag under frozen OOR policy | `oor_flags` column in trace.csv |
| **(v) No silent smoothing** | No implicit filtering/interpolation unless declared at Tier-0 and frozen | No moving averages unless in embedding.yaml |

**Rule**: Any proposed new face policy must (i) satisfy admissibility criteria and (ii) be exported with an OOR/face diff report (counts, channels, indices). Policies that depend on future samples or silently smooth are **non-admissible** for kernel-grade claims.

---

## Minimal Taxonomy (Admissible Classes)

The following classes are **admissible** when they satisfy Definition 1:

### 1. Pre-clip (clip-and-flag)

**Operation**: Clip to [a, b] (and to [ε, 1-ε] for log-safety) with explicit flags

```yaml
face_policy: pre_clip
epsilon: 1.0e-8
oor_policy: clip_and_flag
```

**Behavior**:
- Before kernel computation, clip all c_i(t) to [ε, 1-ε]
- Set `oor_flags[t]` = True if any component was clipped
- Record original values in `observables.yaml` metadata

**Use case**: Default for most UMCP contracts, ensures log(c_i) is well-defined

**Files**:
- Policy declaration: [contract.yaml](contract.yaml)
- Implementation: [src/umcp/validator.py](src/umcp/validator.py)
- Flags: `derived/trace.csv` (oor_flags column)

---

### 2. Mask-and-type

**Operation**: Mark coordinates or rows as typed outcomes for return eligibility (without altering values silently)

```yaml
face_policy: mask_and_type
typed_censoring:
  tau_R_infinite_policy: "R*tau_R := 0"
```

**Behavior**:
- Identify samples where c_i(t) saturates at boundary
- Type τ_R = ∞_rec for non-returning trajectories
- Adjust return credit R via declared typed censoring rule
- Do NOT alter trace values themselves

**Use case**: Handling τ_R = ∞_rec in seam budget without corrupting kernel outputs

**Files**:
- Policy declaration: [contracts/UMA.INTSTACK.v1.yaml](contracts/UMA.INTSTACK.v1.yaml)
- Implementation: [src/umcp/validator.py](src/umcp/validator.py) (return domain logic)

---

### 3. Postselected/Transmit Domain Policies

**Operation**: Restrict eligibility via D_θ with explicit anchor rule; must not use lookahead

```yaml
return:
  domain_type: fixed
  D_theta:
    - condition: "omega < 0.3"
      label: "admissible"
```

**Behavior**:
- Define return domain D_θ(t) using only past/present data
- Exclude future samples from eligibility determination
- Flag excluded samples explicitly in return candidate set U_θ(t)

**Use case**: Restricting return neighborhoods to stable regimes only

**Files**:
- Domain specification: [canon/anchors.yaml](canon/anchors.yaml)
- Implementation: [closures/return_domain.window64.v1.yaml](closures/return_domain.window64.v1.yaml)

---

## Anti-Example: Non-Admissible Policies

### ❌ Future-dependent smoothing

```python
# NON-ADMISSIBLE: Uses future samples
c_smoothed[t] = np.mean(c[t-5:t+5])  # Violates locality
```

**Why rejected**: Breaks locality criterion (i), depends on future samples, prevents replay

---

### ❌ Silent interpolation

```python
# NON-ADMISSIBLE: Modifies without flagging
if c[t] == 0:
    c[t] = 0.5 * (c[t-1] + c[t+1])  # No flag raised
```

**Why rejected**: Violates audit visibility (iv), silently alters trace

---

### ❌ Stochastic adjustment

```python
# NON-ADMISSIBLE: Non-deterministic
c[t] += np.random.normal(0, 0.01)  # Violates determinism
```

**Why rejected**: Breaks determinism (ii), cannot be replayed from contract

---

### ❌ Non-idempotent clipping

```python
# NON-ADMISSIBLE: Not idempotent
def bad_clip(x):
    return x * 0.99  # Applying twice gives 0.99^2 * x
```

**Why rejected**: Violates idempotence (iii), repeated application changes result

---

## OOR/Face Diff Report (Required Output)

When face policy is applied, export a diff report:

```json
{
  "face_policy": "pre_clip",
  "epsilon": 1e-8,
  "oor_report": {
    "total_samples": 1000,
    "samples_clipped": 23,
    "channels_affected": [0, 2, 5],
    "clipping_events": [
      {"t": 47, "channel": 0, "original": 1.0000001, "clipped": 0.99999999},
      {"t": 128, "channel": 2, "original": -0.0000001, "clipped": 1e-8}
    ],
    "max_deviation": 1.1e-7,
    "verification": "All clipped values within epsilon tolerance"
  }
}
```

**Files**: Export to `derived/oor_report.json` in CasePack

---

## Contract Integration

Face policy must be declared in contract frozen params:

```yaml
contract:
  id: UMA.INTSTACK.v1
  frozen_params:
    face_policy: pre_clip
    epsilon: 1.0e-8
    oor_policy: clip_and_flag
    embedding_range: [0, 1]
```

**Immutability**: Once contract is frozen, face policy **cannot be changed** without creating new contract version (weld event).

---

## Verification Checklist

Before accepting a new face policy:

- [ ] **Locality**: Does it operate on t-only data?
- [ ] **Determinism**: Can it be replayed from frozen contract?
- [ ] **Idempotence**: Does applying twice = applying once?
- [ ] **Audit visibility**: Are all modifications flagged?
- [ ] **No silent smoothing**: Is any filtering declared in Tier-0?
- [ ] **OOR report**: Does it export clipping event metadata?
- [ ] **Contract frozen**: Is policy declared in `frozen_params`?

**If any answer is NO, the policy is non-admissible.**

---

## Implementation Status

### Current UMCP Support

- ✅ **pre_clip**: Fully implemented in [src/umcp/validator.py](src/umcp/validator.py)
- ✅ **oor_flags**: Exported in `derived/trace.csv`
- ✅ **typed_censoring**: Supported for τ_R = ∞_rec
- ✅ **Contract declaration**: `face_policy` field in UMA.INTSTACK.v1
- ⚠️ **OOR diff report**: Partial (logged but not exported to JSON)

### Future Work

- [ ] Export `derived/oor_report.json` in all CasePacks
- [ ] Add mask-and-type policy as alternative to pre_clip
- [ ] Implement postselected domain policies with explicit D_θ
- [ ] Create face policy validation tests
- [ ] Document face policy in receipts (ss1m.json)

---

## Summary

Face policy governs **boundary behavior** in the bounded trace cube. Admissible policies must be:

1. **Local** (no lookahead)
2. **Deterministic** (replayable)
3. **Idempotent** (stable under reapplication)
4. **Audit-visible** (explicit flags)
5. **No silent smoothing** (declared filtering only)

**Default policy**: `pre_clip` with `oor_policy: clip_and_flag` and `epsilon: 1e-8`

**Anti-pattern**: Silent interpolation, stochastic adjustment, future-dependent smoothing

**Rule**: If unsure whether a policy is admissible, default to `pre_clip` and consult manuscript §2.3.

---

**See Also**:
- [AXIOM.md](AXIOM.md) - Core axiom: "What returns through collapse is real"
- [contracts/UMA.INTSTACK.v1.yaml](contracts/UMA.INTSTACK.v1.yaml) - Face policy contract specification
- [UHMP.md](UHMP.md) - Universal Hash Manifest Protocol
- [SYMBOL_INDEX.md](SYMBOL_INDEX.md) - Authoritative symbol table

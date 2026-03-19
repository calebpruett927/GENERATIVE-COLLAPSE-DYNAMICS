---
name: Bug Report
about: Report a validation failure, identity mismatch, or unexpected result
title: '[BUG] '
labels: bug
assignees: ''

---

## Environment

- **UMCP Version**: <!-- e.g., 2.2.3 — run `python -c "import umcp; print(umcp.__version__)"` -->
- **Python Version**: <!-- e.g., 3.12.1 -->
- **OS**: <!-- e.g., Ubuntu 24.04, macOS 15, Windows 11 -->
- **Install method**: <!-- pip install umcp / pip install -e ".[all]" / other -->

## Contract & Regime (if applicable)

- **Contract**: <!-- e.g., UMA.INTSTACK.v1 -->
- **Casepack**: <!-- e.g., casepacks/hello_world -->
- **Regime**: <!-- Stable / Watch / Collapse / NON_EVALUABLE / unknown -->
- **Verdict**: <!-- CONFORMANT / NONCONFORMANT / NON_EVALUABLE -->

## Describe the Bug

<!-- A clear, concise description of what went wrong. -->

## Steps to Reproduce

```bash
# Minimal commands to reproduce the issue:
pip install -e ".[all]"
umcp validate <casepack>
```

## Expected Behavior

<!-- What you expected to happen. -->

## Actual Behavior

<!-- What actually happened. Include error messages and tracebacks. -->

## Identity Check (if relevant)

<!-- If this is a kernel/math issue, verify: -->
- [ ] F + ω = 1? (residual: )
- [ ] IC ≤ F? (margin: )
- [ ] IC ≈ exp(κ)? (diff: )
- [ ] `umcp integrity` passes?

## Additional Context

<!-- Paste validation output, JSON reports, or screenshots. -->

# UMCP Commit Protocol

Every commit to this repository **must** pass through the pre-commit protocol before push.
This ensures CI (`.github/workflows/validate.yml`) will never see ruff, formatting, integrity,
test, or validation failures.

## Quick Reference

```bash
# Run the full protocol (auto-fixes + validates):
python scripts/pre_commit_protocol.py

# Dry-run (report only, no modifications):
python scripts/pre_commit_protocol.py --check
```

Exit code 0 = safe to commit. Exit code 1 = failures remain.

## Protocol Steps

| Step | What it does | Blocking? | Auto-fixable? |
|------|-------------|-----------|---------------|
| 1. `ruff format` | Format all Python files | Yes | Yes |
| 2. `ruff check --fix` | Lint + auto-fix | Yes | Partially |
| 3. `mypy src/umcp` | Type-check core library | No* | No |
| 4. `git add -A` | Stage all changes | Yes | N/A |
| 5. `update_integrity.py` | Regenerate SHA256 checksums | Yes | Yes |
| 6. `pytest` | Full test suite (1060+ tests) | Yes | No |
| 7. `umcp validate .` | CONFORMANT required | Yes | No |

*\*mypy is non-blocking because CI runs it with `continue-on-error: true`.*

## When to Run

**Always** before `git commit`. The protocol should be the **last thing** you run
before committing. It handles:

- Formatting fixes (so `ruff format --check` passes in CI)
- Lint auto-fixes (so `ruff check` passes in CI)
- Integrity checksum regeneration (so tests checking SHA256 don't fail)
- Full test + validation pass

## What Gets Updated Automatically

When you modify tracked files, the protocol updates:

| File | What changes |
|------|-------------|
| `integrity/sha256.txt` | SHA256 checksums of all tracked source files |
| Any `.py` file | Formatting adjustments (whitespace, quotes, trailing commas) |
| Any `.py` file | Auto-fixable lint issues (unused imports, f-string cleanup) |

## Files That May Need Manual Updates

When adding **new domains, closures, or casepacks**, you must also update:

| Change type | Files to update |
|-------------|----------------|
| New closure | `closures/registry.yaml`, `closures/<domain>/__init__.py` |
| New contract | `contracts/` (YAML), reference in manifest |
| New casepack | `casepacks/<name>/manifest.json`, `expected/invariants.json` |
| New canon anchor | `canon/<name>_anchors.yaml`, `canon/README.md` |
| New test file | `tests/` (follows `test_<NN>_<subject>.py` convention) |
| Version bump | `src/umcp/__init__.py` (`__version__`), `CHANGELOG.md` |

## CI Pipeline Mirror

The protocol mirrors what CI runs:

```
CI Job: lint
  ├── ruff format --check .      ← Protocol Step 1
  ├── ruff check .               ← Protocol Step 2
  └── mypy src/umcp              ← Protocol Step 3

CI Job: test
  ├── update_integrity.py        ← Protocol Step 5
  └── pytest                     ← Protocol Step 6

CI Job: umcp-validate
  ├── umcp validate .            ← Protocol Step 7
  └── umcp validate --strict casepacks/UMCP-REF-E2E-0001
```

## Common Failure Patterns

| Failure | Cause | Fix |
|---------|-------|-----|
| `ruff format` | New file not formatted | Auto-fixed by protocol |
| `F541 f-string without placeholders` | Using `f"..."` with no `{...}` | Auto-fixed by `--fix` |
| `E402 import not at top` | `sys.path` manipulation before imports | Add file to `per-file-ignores` in `pyproject.toml` |
| `F841 unused variable` | Assigned but never read | Prefix with `_` or remove assignment |
| Integrity mismatch | Modified file without re-checksumming | Auto-fixed by protocol |
| NONCONFORMANT | Schema violation in invariants | Fix `expected/invariants.json` |

## Per-File Ignores (pyproject.toml)

Scripts that manipulate `sys.path` before imports need E402 ignored:

```toml
[tool.ruff.lint.per-file-ignores]
"scripts/**" = ["E501", "E741", "E402", "SIM108", "SIM118", "UP038"]
```

## Example Workflow

```bash
# 1. Make your changes
vim closures/astronomy/new_closure.py

# 2. Run the protocol
python scripts/pre_commit_protocol.py

# 3. If all green, commit and push
git commit -m "feat: add new closure"
git push origin main
```

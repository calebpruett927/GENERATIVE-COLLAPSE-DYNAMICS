# AI Workflow Protocol — GCD Repository

> **This file is committed to the repo so ALL agents in ALL sessions follow these rules.**
> It supplements `AGENTS.md`, `CLAUDE.md`, and `.github/copilot-instructions.md`.

## Task Sizing & Autonomy Rules

Before starting, classify each task:

| Size | Criteria | Autonomy |
|------|----------|----------|
| **Small** | Single-file edit, registry fix, freeze/integrity update, test fix, doc correction | **Full autonomy** — execute without stopping |
| **Medium** | New closure file, multi-file edits in one domain, theorem additions | **Guided** — show plan, execute on approval |
| **Large** | New domain, cross-domain changes, kernel/protocol changes, architecture | **Collaborative** — plan together, execute stepwise with check-ins |

## Three-Batch Cycle

Tasks are processed in **batches of 3**. Within each batch:

**Batch 1 — Research (read-only)**
- Parallel file reads + grep/regex searches only
- Map dependencies, identify touch points
- Classify each task as Small / Medium / Large
- Output: summary of what needs to change and where

**Batch 2 — Plan + Auto-Execute Small Tasks**
- Create `manage_todo_list` with all items across the 3 tasks
- **Small tasks**: execute immediately and autonomously — no pause needed
- **Medium/Large tasks**: present plan, wait for user approval before executing

**Batch 3 — Execute Remaining (one item at a time)**
- Medium/Large items: mark in-progress → do it → mark completed
- After each edit: targeted validation (e.g., `pytest tests/test_XXX.py -x`)
- Full validation (`pre_commit_protocol.py`) only at final commit step
- If all 3 tasks were Small, this batch is skip — already done in Batch 2

### Batch Grouping Rules
- When given N tasks, group into ceil(N/3) batches of 3
- Within a batch, order: Small first, then Medium, then Large
- A single request with 1-3 items = 1 batch
- If a Small task fails validation, escalate it to Medium (show user)

### Incremental Validation Rules
- Single closure change → `pytest tests/test_XXX.py -x` (specific test file)
- Core module change → `pytest tests/test_000*.py tests/test_001*.py -x` (identity tests first)
- Multi-file change → `umcp validate casepacks/hello_world --strict`
- Pre-commit only → `python scripts/pre_commit_protocol.py`
- NEVER run full test suite mid-task

### Context Budget Rules
- Avoid reading >3 large files in one response
- Prefer `grep_search` over `read_file` when looking for specific patterns
- Use subagents for deep multi-file research to keep main context clean
- Keep terminal output short: use `head`, `tail`, `grep` to filter

## Connection Stability Rules (CRITICAL)

- **Commit after every meaningful unit** — never batch multiple files to one commit at the end
- **Use subagents for research** — keeps main thread lighter, prevents timeouts
- **Write session memory checkpoints** — if connection drops, read `/memories/session/` to resume
- **Max 5 sequential tool calls** before producing output — if more needed, split into subagent
- **Never stop mid-task without committing** — if work is done but not pushed, always push before responding

## Post-Edit Commit Protocol (MANDATORY — prevents CI failures)

After ANY file edit, before committing, run ALL of these in order:

1. `python scripts/update_integrity.py` — regenerate SHA-256 checksums
2. Check freeze drift: `sha256sum closures/registry.yaml` and compare to `freeze/freeze_manifest.json`
   - If `closures/registry.yaml` changed: update BOTH `freeze/freeze_manifest.json` AND `freeze/closures_registry.yaml.sha256`
3. `python scripts/repo_health_check.py` — must show PASS
4. Then: `git add -A && git commit && git push`

### Files That Trigger Freeze Updates

| If you edit... | You MUST also update... |
|----------------|------------------------|
| `closures/registry.yaml` | `freeze/freeze_manifest.json` AND `freeze/closures_registry.yaml.sha256` |
| Any file listed in `freeze/freeze_manifest.json` | The corresponding hash in `freeze/freeze_manifest.json` |

### Test Count Update Protocol

When adding test files:

1. `python -m pytest --collect-only 2>&1 | tail -1` → get new test count
2. `ls tests/test_*.py | wc -l` → get new file count
3. `grep -rl 'OLD_COUNT' --include='*.md' --include='*.astro'` → find all references
4. Also update URL-encoded badge: `OLD%2CCOUNT` → `NEW%2CCOUNT`
5. Also update "N test files" and "test_NNN" range references in docs
6. Update `.github/copilot-instructions.md` test table with new entries

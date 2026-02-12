# Archive

Superseded, stale, or unreferenced files moved here to keep the active
codebase clean while preserving continuity.  Every file remains in git
history and can be restored with `git checkout <commit> -- <path>`.

## Contents

| Directory | What | Why archived |
|-----------|------|--------------|
| `contracts/` | UMA.INTSTACK v1.0.1, v2 | Superseded — all casepacks use v1 |
| `scripts/` | Old run generators (v1-v4), shell wrappers | Superseded by v5, tasks.json, pre_commit_protocol |
| `runs/` | KIN.CP.*.RUN001-003 | Superseded by RUN004 (latest, frozen) |
| `artifacts/` | Validator baselines, test landscape, regenerable outputs | Stale snapshots, no code references |
| `images/` | Root-level PNGs (architecture, workflow, benchmark) | Unreferenced by any doc or code |
| `docs/` | umcp_geometric_structure.png | Unreferenced |
| `examples/` | Dashboard screenshots (10 PNGs) | Unreferenced by any doc |
| `web/` | Empty web scaffold (tsconfig + empty src/) | Never completed |
| `poetry.lock` | Poetry lockfile | Project uses pip/setuptools, not Poetry |

## Restoring a file

```bash
# Copy back from archive
cp archive/scripts/generate_kin_runs_v4.py scripts/

# Or restore from git history
git log --oneline -- scripts/generate_kin_runs_v4.py
git checkout <commit>^ -- scripts/generate_kin_runs_v4.py
```

## Date archived

2026-02-12 — commit that moved files to archive.

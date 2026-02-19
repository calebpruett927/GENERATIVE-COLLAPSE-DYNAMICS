# Archive

Superseded, stale, or unreferenced files moved here to keep the active
codebase clean while preserving continuity.  Every file remains in git
history and can be restored with `git checkout <commit> -- <path>`.

## Version Context

| Item | Active (root) | Archived |
|------|---------------|----------|
| Package version | v2.1.3 | v0.1.0 – v1.5.0 era artifacts |
| Contract | UMA.INTSTACK.v1 (v1.0.0) | v1.0.1 (patch), v2 (draft, never adopted) |
| Kinematics runs | RUN004 (frozen, in `runs/`) | RUN001–003 (superseded) |
| Run generators | v5 (`scripts/`) | v1–v4 (`archive/scripts/`) |
| Test landscape | `artifacts/test_landscape_profile.json` (uses `integrity_excess`) | `archive/artifacts/test_landscape_profile.json` (uses deprecated `amgm_excess`) |

## Contents

| Directory | What | Why archived |
|-----------|------|--------------|
| `contracts/` | UMA.INTSTACK v1.0.1, v2 | Superseded — all casepacks use v1 (v1.0.0) |
| `scripts/` | Old run generators (v1–v4), shell wrappers | Superseded by v5, tasks.json, pre_commit_protocol |
| `runs/` | KIN.CP.*.RUN001–003 (9 run directories) | Superseded by RUN004 (latest, frozen) |
| `artifacts/` | Validator baselines, old test landscape, regenerable outputs | Stale snapshots; root `artifacts/` has current version |
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

2026-02-12 — initial archive commit (v2.1.0 release).
2026-02-19 — labels updated for v2.1.3 freeze.

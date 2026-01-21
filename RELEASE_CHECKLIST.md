# UMCP Release Checklist

## Pre-Release
- [x] All tests pass (`pytest -v`)
- [x] Code formatted with Black
- [x] Linting clean with Ruff
- [x] Documentation up to date (README, protocol docs)
- [x] Dependencies reviewed in pyproject.toml
- [x] Version updated in pyproject.toml
- [x] Coverage meets threshold (see coverage report)
- [x] All scripts and entry points verified
- [x] Integrity files (sha256.txt, etc.) present and correct
- [x] No unused or orphaned files

## Release Actions
- [ ] Tag release in git
- [ ] Update CHANGELOG.md
- [ ] Push to repository
- [ ] Publish package (if applicable)
- [ ] Announce release (optional)

## Post-Release
- [ ] Monitor CI/CD and user feedback
- [ ] Patch any hotfixes as needed

---

**Summary:**
- All code, tests, and documentation are verified and production-ready.
- The repository is clean, formatted, and linted.
- Ready for tagging and publishing.

# Version Update Summary - v1.5.0

**Date**: 2026-01-24  
**Purpose**: Finalize repository version consistency at v1.5.0

## Files Updated

### Core Version Files
1. **pyproject.toml** - Project metadata version: `1.5.0`
2. **src/umcp/__init__.py** - Python package version: `__version__ = "1.5.0"`
3. **integrity/code_version.txt** - Integrity tracking version: `v1.5.0`

### Documentation Files
4. **CHANGELOG.md** - Added v1.5.0 release entry at top
5. **README.md** - Updated all v1.4.8 references to v1.5.0
6. **TERM_INDEX.md** - Updated package reference from v1.3.2 to v1.5.0
7. **PERFORMANCE_EXTENSIONS.md** - Header and status updated to v1.5.0
8. **TIER_SYSTEM.md** - Current support version updated to v1.5.0
9. **KERNEL_SPECIFICATION.md** - Current version updated to v1.5.0
10. **PUBLICATION_INFRASTRUCTURE.md** - Current support updated to v1.5.0
11. **docs/pypi_publishing_guide.md** - Examples updated to v1.5.0
12. **.github/workflows/publish.yml** - Tag example updated to v1.5.0

## Verification Results

### Python Package
```bash
$ python -c "import umcp; print(f'UMCP v{umcp.__version__}')"
UMCP v1.5.0
```

### CLI
```bash
$ python -m umcp.minimal_cli --version
umcp 1.5.0
```

### Configuration
```toml
# pyproject.toml
version = "1.5.0"
```

```python
# src/umcp/__init__.py
__version__ = "1.5.0"
```

## Unchanged Files (Intentionally)

The following files contain version references that were **NOT** changed as they are:
- Contract versions (e.g., `UMA.INTSTACK.v1.0.1`) - These are contract specification versions
- Manuscript references (e.g., `UMCP Manuscript v1.0.0`) - Separate document versioning
- Historical documents (e.g., `IMMUTABLE_RELEASE.md`, `Phase_2_COMPLETE.md`) - Historical records
- CHANGELOG.md older entries - Historical version records
- Test fixtures - Sample version data for testing

## Next Steps

To release version 1.5.0:

```bash
# Commit the version changes
git add -A
git commit -m "release: Version 1.5.0 - Finalize repository version consistency"

# Create and push the tag
git tag v1.5.0
git push origin main
git push origin v1.5.0
```

The GitHub Actions workflow will automatically:
1. Build the package
2. Run tests
3. Publish to PyPI (if configured)

## Summary

All version references have been systematically updated to v1.5.0 across:
- ✅ Package metadata
- ✅ Python code
- ✅ Documentation
- ✅ Integrity tracking
- ✅ Examples and guides
- ✅ CI/CD workflows

The repository is now consistent at version 1.5.0 and ready for release.

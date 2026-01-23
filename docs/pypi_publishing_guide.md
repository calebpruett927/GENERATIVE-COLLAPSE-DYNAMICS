# PyPI Publishing Guide

## Current Configuration

The repository uses **token-based authentication** for PyPI publishing.

### How It Works

1. When you push a tag like `v1.4.0`, GitHub Actions triggers
2. The workflow builds the package (`python -m build`)
3. It publishes to PyPI using the `PYPI_PUBLISH_TOKEN` secret

### Secret Configuration

The `PYPI_PUBLISH_TOKEN` secret must be configured at:
- **URL**: https://github.com/calebpruett927/UMCP-Metadata-Runnable-Code/settings/secrets/actions
- **Name**: `PYPI_PUBLISH_TOKEN`
- **Value**: Your PyPI API token from https://pypi.org/manage/account/token/

### VS Code Warning

The warning `Context access might be invalid: PYPI_PUBLISH_TOKEN` is **harmless**:
- ✅ It's just a static analysis linter that can't verify secrets exist
- ✅ The workflow will work correctly if the secret is configured
- ✅ The warning does NOT affect CI/CD functionality

## Alternative: Trusted Publishing (OIDC)

For enhanced security without managing tokens, you can switch to **Trusted Publishing**:

### Steps to Enable Trusted Publishing

1. **On PyPI** (https://pypi.org/manage/project/umcp/settings/publishing/):
   - Add publisher: `calebpruett927/UMCP-Metadata-Runnable-Code`
   - Workflow: `publish.yml`
   - Environment: (leave blank)

2. **In the workflow** (`.github/workflows/publish.yml`):
   ```yaml
   - name: Publish to PyPI
     uses: pypa/gh-action-pypi-publish@release/v1
     # No password needed - uses OIDC via id-token: write permission
   ```

3. **Remove the token**:
   - Delete the `password: ${{ secrets.PYPI_PUBLISH_TOKEN }}` line
   - Delete the secret from GitHub repo settings (optional)

### Benefits of Trusted Publishing

- ✅ No token to manage or rotate
- ✅ More secure (short-lived OIDC tokens)
- ✅ No GitHub Actions linter warnings
- ✅ PyPI's recommended approach

## Recommendation

Since you already have the token configured and it works, you can:
- **Keep current setup** - ignore the VS Code warning (it's harmless)
- **Switch to Trusted Publishing** - follow the steps above for better security

Both work correctly. The warning is cosmetic and won't affect deployments.

# PyPI Publishing Guide

## Current Configuration

The repository uses **token-based authentication** for PyPI publishing.

### How It Works

1. When you push a tag like `v1.4.0`, GitHub Actions triggers
2. The workflow builds the package (`python -m build`)
3. It publishes to PyPI using the `PYPI_PUBLISH_TOKEN` secret

### Secret Configuration

**REQUIRED**: The `PYPI_PUBLISH_TOKEN` secret must be configured in GitHub repository settings.

#### Step-by-Step Setup

1. **Create PyPI API token**:
   - Go to https://pypi.org/manage/account/token/
   - Click "Add API token"
   - Token name: `UMCP GitHub Actions`
   - Scope: Select "Project: umcp" (or "Entire account")
   - Click "Create token"
   - **COPY THE TOKEN** (shown only once!)

2. **Add secret to GitHub**:
   - Go to https://github.com/calebpruett927/UMCP-Metadata-Runnable-Code/settings/secrets/actions
   - Click "New repository secret"
   - Name: `PYPI_PUBLISH_TOKEN`
   - Value: Paste the PyPI token
   - Click "Add secret"

### VS Code Warning - HOW TO RESOLVE

**The Issue:**
```
Context access might be invalid: PYPI_PUBLISH_TOKEN
```

**Why It Happens:**
- VS Code's GitHub Actions linter performs static analysis
- It cannot verify GitHub repository secrets (they're stored securely on GitHub servers)
- This is a **linter limitation**, not an actual error

**How to Resolve:**

**Option 1: Ignore It (Recommended if token is configured)**
- ✅ The warning is harmless and cosmetic
- ✅ Your workflow will work correctly if the secret exists in GitHub
- ✅ CI/CD is NOT affected by this warning
- Just ensure the secret is configured in GitHub repo settings (step 2 above)

**Option 2: Switch to Trusted Publishing (Removes Warning Permanently)**
- See "Alternative: Trusted Publishing (OIDC)" section below
- This eliminates tokens entirely and the warning disappears

**Verification:**
To confirm the secret is configured:
```bash
# You cannot read secrets locally, but you can verify the workflow works:
git tag v1.4.8-test
git push origin v1.4.8-test
# Watch: https://github.com/calebpruett927/UMCP-Metadata-Runnable-Code/actions
# If successful, delete test tag: git push --delete origin v1.4.8-test
```

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

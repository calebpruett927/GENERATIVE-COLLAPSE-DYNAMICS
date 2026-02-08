# GitHub Secrets Configuration

## Required Secrets

### PYPI_PUBLISH_TOKEN

**Purpose**: Authenticates PyPI package publishing in the `publish.yml` workflow.

**Status**: ⚠️ MUST BE CONFIGURED before publishing to PyPI

**Setup Instructions**:

1. **Create PyPI API Token**:
   - Visit: https://pypi.org/manage/account/token/
   - Create a new token with scope: "Project: umcp"
   - Copy the token (shown only once!)

2. **Add to GitHub Repository**:
   - Visit: https://github.com/calebpruett927/GENERATIVE-COLLAPSE-DYNAMICS/settings/secrets/actions
   - Click "New repository secret"
   - Name: `PYPI_PUBLISH_TOKEN`
   - Value: Paste the PyPI token
   - Save

**VS Code Warning**:
If you see `Context access might be invalid: PYPI_PUBLISH_TOKEN` in VS Code:
- ✅ **This is normal and harmless**
- ✅ VS Code's linter cannot verify GitHub secrets exist
- ✅ Your workflow will work correctly if the secret is configured in GitHub
- ✅ This does NOT indicate an actual error

**Verification**:
The secret is configured correctly if:
- You can see it listed at: https://github.com/calebpruett927/GENERATIVE-COLLAPSE-DYNAMICS/settings/secrets/actions
- The `publish.yml` workflow succeeds when you push a version tag

## Alternative: Trusted Publishing (Recommended)

To eliminate secrets entirely and remove VS Code warnings:

1. **Configure on PyPI**:
   - Go to: https://pypi.org/manage/project/umcp/settings/publishing/
   - Add publisher: `calebpruett927/GENERATIVE-COLLAPSE-DYNAMICS`
   - Workflow: `publish.yml`

2. **Update Workflow**:
   - Remove `password: ${{ secrets.PYPI_PUBLISH_TOKEN }}` from `publish.yml`
   - Keep `id-token: write` permission (already present)
   - OIDC authentication happens automatically

3. **Benefits**:
   - ✅ No tokens to manage or rotate
   - ✅ More secure (short-lived credentials)
   - ✅ No VS Code warnings
   - ✅ PyPI's recommended method

See [docs/pypi_publishing_guide.md](../docs/pypi_publishing_guide.md) for detailed instructions.

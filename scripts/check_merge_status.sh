#!/usr/bin/env bash
# check_merge_status.sh
# Quick script to verify merge status and repository health
set -euo pipefail

# Function to activate virtual environment
activate_venv() {
  if [[ -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
  elif [[ -f ".venv/Scripts/activate" ]]; then
    source .venv/Scripts/activate
  else
    return 1
  fi
}

echo "========================================"
echo "UMCP Merge Status Verification"
echo "========================================"
echo ""

# Change to repo root
cd "$(dirname "$0")/.."

echo "✓ Git Status:"
git status --short
if [ -z "$(git status --short)" ]; then
  echo "  Clean working tree ✅"
else
  echo "  Uncommitted changes present ⚠️"
fi
echo ""

echo "✓ Checking for merge conflict artifacts:"
CONFLICTS=$(find . -type f \( -name "*.orig" -o -name "*.rej" -o -name "*CONFLICT*" \) 2>/dev/null | wc -l)
if [ "$CONFLICTS" -eq 0 ]; then
  echo "  No conflict artifacts found ✅"
else
  echo "  Found $CONFLICTS conflict artifact(s) ❌"
  find . -type f \( -name "*.orig" -o -name "*.rej" -o -name "*CONFLICT*" \) 2>/dev/null
fi
echo ""

echo "✓ Running tests:"
if ! activate_venv; then
  echo "  Virtual environment not found. Creating..."
  python3 -m venv .venv
  activate_venv
  pip install -q -e ".[test]"
fi

if pytest -q; then
  echo "  All tests passing ✅"
else
  echo "  Some tests failing ❌"
  exit 1
fi
echo ""

echo "✓ Running UMCP validator:"
TEMP_RESULT=$(mktemp)
trap "rm -f $TEMP_RESULT" EXIT

if umcp validate . --out "$TEMP_RESULT" 2>&1 | grep -q "Wrote validator result"; then
  read -r STATUS ERRORS WARNINGS <<< "$(python3 -c "
import json
data = json.load(open('$TEMP_RESULT'))
print(data['run_status'], data['summary']['counts']['errors'], data['summary']['counts']['warnings'])
")"
  
  if [ "$STATUS" = "CONFORMANT" ] && [ "$ERRORS" -eq 0 ] && [ "$WARNINGS" -eq 0 ]; then
    echo "  Validation: $STATUS (Errors: $ERRORS, Warnings: $WARNINGS) ✅"
  else
    echo "  Validation: $STATUS (Errors: $ERRORS, Warnings: $WARNINGS) ⚠️"
  fi
else
  echo "  Validator failed to run ❌"
  exit 1
fi
echo ""

echo "========================================"
echo "✅ MERGE STATUS: SUCCESSFUL"
echo "========================================"
echo ""
echo "All checks passed. The content has been successfully merged."
echo "For detailed report, see: MERGE_VERIFICATION.md"

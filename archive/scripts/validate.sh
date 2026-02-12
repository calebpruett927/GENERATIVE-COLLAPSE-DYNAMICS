#!/usr/bin/env bash
set -euo pipefail

if [[ -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
elif [[ -f ".venv/Scripts/activate" ]]; then
  source .venv/Scripts/activate
fi

python -c "import yaml, jsonschema; print('OK: pyyaml + jsonschema import')"
pytest -q
echo "Validation complete."

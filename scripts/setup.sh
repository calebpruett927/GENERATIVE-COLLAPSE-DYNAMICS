#!/usr/bin/env bash
set -euo pipefail

python -m venv .venv

if [[ -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
elif [[ -f ".venv/Scripts/activate" ]]; then
  source .venv/Scripts/activate
fi

python -m pip install -U pip
python -m pip install -e ".[test]"
echo "Setup complete. Activate your venv and run: pytest"

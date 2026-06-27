#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -x ".venv/bin/python" ]; then
  echo "Virtual environment not found. Run scripts/setup_local.sh first." >&2
  exit 1
fi

.venv/bin/python -m streamlit run ui/app.py

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

UV_CMD=()
if command -v uv >/dev/null 2>&1; then
  UV_CMD=(uv)
elif python -m uv --version >/dev/null 2>&1; then
  UV_CMD=(python -m uv)
fi

if [ "${#UV_CMD[@]}" -gt 0 ]; then
  if [ ! -x ".venv/bin/python" ]; then
    "${UV_CMD[@]}" venv --python 3.12 .venv
  fi
  "${UV_CMD[@]}" pip install --python .venv/bin/python -r requirements.txt
else
  echo "Warning: uv is not installed; falling back to python -m venv + pip. RDKit may not install on Python 3.14." >&2
  if [ ! -x ".venv/bin/python" ]; then
    python -m venv .venv
  fi

  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt
fi

echo "Local environment is ready. Activate it with: source .venv/bin/activate"

#!/usr/bin/env bash
set -euo pipefail

# Optional: pass a .env path as first arg; otherwise relies on current env
ENV_FILE=${1:-}

if [[ -n "${ENV_FILE}" ]]; then
  if [[ ! -f "${ENV_FILE}" ]]; then
    echo "Env file not found: ${ENV_FILE}" >&2
    exit 1
  fi
  set -a; source "${ENV_FILE}"; set +a
fi

# Use prod proxy smoke (base or ft:... via MODEL env)
uv run python examples/qwen_coder/infer_prod_proxy.py



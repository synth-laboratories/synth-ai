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

uvx synth-ai train \
  --type sft \
  --config examples/qwen_coder/configs/coder_lora_30b.toml \
  --dataset examples/qwen_coder/ft_data/coder_sft.small.jsonl \
  --env-file "${ENV_FILE:-}"



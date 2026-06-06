#!/usr/bin/env bash
# Crafter code-policy DEO hillclimb (1 candidate) on slot1 via crafter_runs/crafter_deo_run.py
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export EVALS_ROOT="${EVALS_ROOT:-$REPO_ROOT/../evals}"
export SYNTH_WORKSPACE_ROOT="${SYNTH_WORKSPACE_ROOT:-$REPO_ROOT/..}"

if [[ -z "${SYNTH_API_KEY:-}" ]] && [[ -f "$REPO_ROOT/.env" ]]; then
  SYNTH_API_KEY="$(grep -E '^SYNTH_API_KEY=' "$REPO_ROOT/.env" | head -1 | cut -d= -f2- | tr -d '\r')"
  if [[ -n "$SYNTH_API_KEY" ]]; then
    export SYNTH_API_KEY
  fi
fi

uv sync --group dev

args=(--use-default-slot1)
if [[ -n "${OUTPUT_ROOT:-}" ]]; then
  args+=(--output-root "$OUTPUT_ROOT")
fi

exec uv run python crafter_runs/crafter_deo_run.py "${args[@]}" "$@"

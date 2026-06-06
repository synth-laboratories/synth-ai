#!/usr/bin/env bash
# ReportBench README smoke on slot1 (Codex gpt-5.4-mini worker, gpt-5.3-codex-spark judge).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

EVALS_ROOT="${EVALS_ROOT:-$REPO_ROOT/../evals}"
SYNTH_WORKSPACE_ROOT="${SYNTH_WORKSPACE_ROOT:-$REPO_ROOT/..}"
# Default: readme_runs/runs/<UTC>_slot1/ (see readme_runs/README.md)
OUTPUT_ROOT="${OUTPUT_ROOT:-}"

export EVALS_ROOT
export SYNTH_WORKSPACE_ROOT

if [[ -z "${SYNTH_API_KEY:-}" ]] && [[ -f "$REPO_ROOT/.env" ]]; then
  SYNTH_API_KEY="$(grep -E '^SYNTH_API_KEY=' "$REPO_ROOT/.env" | head -1 | cut -d= -f2- | tr -d '\r')"
  if [[ -n "$SYNTH_API_KEY" ]]; then
    export SYNTH_API_KEY
  fi
fi

uv sync --group dev

args=(--use-default-slot1)
if [[ -n "$OUTPUT_ROOT" ]]; then
  args+=(--output-root "$OUTPUT_ROOT")
fi

exec uv run python readme_runs/readme_smoke.py "${args[@]}" "$@"

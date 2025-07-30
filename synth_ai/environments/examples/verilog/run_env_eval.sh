#!/usr/bin/env bash
# Evaluation runner for Verilog environment.
set -euo pipefail

if [[ "${1:-}" == "--info" ]]; then
  echo "verilog : --model_name <MODEL> --episodes <N> [--debug]"
  exit 0
fi

MODEL_NAME="gpt-4.1-nano"
FORMAT_MODEL_NAME="$MODEL_NAME"
INSTANCES=1
DEBUG_MODE="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_name)
      MODEL_NAME="$2"; FORMAT_MODEL_NAME="$2"; shift 2;;
    --episodes)
      INSTANCES="$2"; shift 2;;
    --debug)
      DEBUG_MODE="true"; shift 1;;
    *) echo "Unknown arg: $1" >&2; exit 1;;
  esac
done

uv run -- python - <<EOF "$MODEL_NAME" "$FORMAT_MODEL_NAME" "$INSTANCES" "$DEBUG_MODE"
import sys, asyncio
from synth_env.examples.verilog.agent_demos.test_synth_react import eval_verilog_react

model_name, fmt_name, n_inst, debug_flag = sys.argv[1:]
asyncio.run(
    eval_verilog_react(
        model_name=model_name,
        formatting_model_name=fmt_name,
        n_instances=int(n_inst),
        debug_mode=(debug_flag.lower() == 'true'),
    )
)
EOF 
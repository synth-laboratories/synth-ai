#!/usr/bin/env bash
# Evaluation runner for Sokoban environment.
# Supports basic args and executes the async evaluation defined in the ReAct demo.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${1:-}" == "--info" ]]; then
  echo "sokoban : --model_name <MODEL> --episodes <N> --max_steps <STEPS> [--seed <SEED>]"
  exit 0
fi

# Defaults
MODEL_NAME="gpt-4.1-nano"
FORMAT_MODEL_NAME="$MODEL_NAME"
EPISODES=3  # per difficulty (the underlying eval uses 3)
MAX_STEPS=20
SEED=0

# Simple arg parsing
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_name)
      MODEL_NAME="$2"; FORMAT_MODEL_NAME="$2"; shift 2;;
    --episodes)
      EPISODES="$2"; shift 2;;
    --max_steps)
      MAX_STEPS="$2"; shift 2;;
    --seed)
      SEED="$2"; shift 2;;
    *)
      echo "Unknown arg: $1" >&2; exit 1;;
  esac
done

# Execute the evaluation via Python (async) using uv
uv run -- python - <<PY "$MODEL_NAME" "$FORMAT_MODEL_NAME" "$EPISODES" "$MAX_STEPS" "$SEED"
import sys, asyncio
from synth_env.examples.sokoban.agent_demos.test_synth_react_locally import eval_react_sokoban

model_name, fmt_name, episodes, max_steps, seed = sys.argv[1:]

# NOTE: The current eval_react_sokoban internally runs 3 episodes per difficulty.
# Until it is parameterised, we ignore episodes / max_steps / seed and simply pass model args.
asyncio.run(eval_react_sokoban(model_name=model_name, formatting_model_name=fmt_name))
PY 
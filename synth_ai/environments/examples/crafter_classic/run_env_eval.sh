#!/usr/bin/env bash
# Evaluation runner for Crafter environment.
set -euo pipefail

if [[ "${1:-}" == "--info" ]]; then
  echo "crafter : --model_name <MODEL> --episodes <N> [--modes easy,hard] [--seed <SEED>]"
  exit 0
fi

MODEL_NAME="gpt-4.1-nano"
FORMAT_MODEL_NAME="$MODEL_NAME"
EPISODES=3
MODES="easy,hard"
SEED=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_name)
      MODEL_NAME="$2"; FORMAT_MODEL_NAME="$2"; shift 2;;
    --episodes)
      EPISODES="$2"; shift 2;;
    --modes)
      MODES="$2"; shift 2;;
    --seed)
      SEED="$2"; shift 2;;
    *) echo "Unknown arg: $1" >&2; exit 1;;
  esac
done

IFS=',' read -ra MODE_LIST <<< "$MODES"
PY_MODE_LIST="["$(printf "'%s'," "${MODE_LIST[@]}")"]"
PY_MODE_LIST=${PY_MODE_LIST/,]/]}

uv run -- python - <<EOF "$MODEL_NAME" "$FORMAT_MODEL_NAME" "$EPISODES" "$PY_MODE_LIST"
import sys, asyncio, ast
from synth_env.examples.crafter_classic.agent_demos.test_synth_react import eval_react_crafter

model_name, fmt_name, episodes, modes_literal = sys.argv[1:]
episodes = int(episodes)
modes = ast.literal_eval(modes_literal)

asyncio.run(
    eval_react_crafter(
        model_name=model_name,
        formatting_model_name=fmt_name,
        modes=modes,
        n_instances_per_mode=episodes,
    )
)
EOF 
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export TASK_APP_URL=${TASK_APP_URL:-http://localhost:8001}
export SIMPLE_CFG=examples/warming_up_to_rl/configs/eval_stepwise_simple.toml
export COMPLEX_CFG=examples/warming_up_to_rl/configs/eval_stepwise_complex.toml

if [[ ! -f "$SIMPLE_CFG" || ! -f "$COMPLEX_CFG" ]]; then
  echo "Missing eval TOMLs under examples/warming_up_to_rl/configs/." >&2
  exit 1
fi

# Fail fast if health not OK
set +e
HEALTH=$(curl -s -H "X-API-Key: ${ENVIRONMENT_API_KEY:-}" "${TASK_APP_URL}/health")
set -e
if [[ -z "$HEALTH" ]]; then
  echo "Task app not reachable at ${TASK_APP_URL}. Start it first (scripts/run_crafter_server.sh)." >&2
  exit 1
fi

# Run evals
echo "Running simple stepwise eval..."
uv run python examples/warming_up_to_rl/run_eval.py --toml "$SIMPLE_CFG" --use-rollout | tee /tmp/eval_simple.out

echo "Running complex stepwise eval..."
uv run python examples/warming_up_to_rl/run_eval.py --toml "$COMPLEX_CFG" --use-rollout | tee /tmp/eval_complex.out

echo "Done. Outputs: /tmp/eval_simple.out, /tmp/eval_complex.out"

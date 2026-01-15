#!/bin/bash
# Run eval jobs for both opencode and codex agents, saving traces and artifacts

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default values
SEEDS=${SEEDS:-3}
MODEL=${MODEL:-gpt-4o-mini}
TIMEOUT=${TIMEOUT:-180}
LOCAL=${LOCAL:-false}

echo "============================================================"
echo "Running eval jobs for both opencode and codex agents"
echo "============================================================"
echo "Seeds: $SEEDS"
echo "Model: $MODEL"
echo "Timeout: ${TIMEOUT}s"
echo "Local mode: $LOCAL"
echo ""

# Create artifacts directory
ARTIFACTS_DIR="$SCRIPT_DIR/artifacts"
mkdir -p "$ARTIFACTS_DIR"

# Run opencode agent
echo ""
echo "============================================================"
echo "Running OPencode agent eval"
echo "============================================================"
if [ "$LOCAL" = "true" ]; then
    uv run python run_eval.py --local --seeds "$SEEDS" --model "$MODEL" --agent opencode --timeout "$TIMEOUT"
else
    uv run python run_eval.py --seeds "$SEEDS" --model "$MODEL" --agent opencode --timeout "$TIMEOUT"
fi

# Run codex agent
echo ""
echo "============================================================"
echo "Running Codex agent eval"
echo "============================================================"
if [ "$LOCAL" = "true" ]; then
    uv run python run_eval.py --local --seeds "$SEEDS" --model "$MODEL" --agent codex --timeout "$TIMEOUT"
else
    uv run python run_eval.py --seeds "$SEEDS" --model "$MODEL" --agent codex --timeout "$TIMEOUT"
fi

echo ""
echo "============================================================"
echo "All eval jobs completed!"
echo "Traces and artifacts saved to: $ARTIFACTS_DIR"
echo "============================================================"

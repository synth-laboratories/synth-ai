#!/bin/bash
# Run eval jobs for both opencode and codex agents, saving traces and artifacts

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default values
SEEDS=${SEEDS:-3}
MODEL=${MODEL:-gpt-4o-mini}
TIMEOUT=${TIMEOUT:-180}

echo "============================================================"
echo "Running eval jobs for both opencode and codex agents"
echo "============================================================"
echo "Seeds: $SEEDS"
echo "Model: $MODEL"
echo "Timeout: ${TIMEOUT}s"
echo "Backend: ${SYNTH_BACKEND_URL:-https://api.usesynth.ai}"
echo ""

# Create artifacts directory
ARTIFACTS_DIR="$SCRIPT_DIR/artifacts"
mkdir -p "$ARTIFACTS_DIR"

# Run opencode agent
echo ""
echo "============================================================"
echo "Running OpenCode agent eval"
echo "============================================================"
uv run python run_eval.py --seeds "$SEEDS" --model "$MODEL" --agent opencode --timeout "$TIMEOUT"

# Run codex agent
echo ""
echo "============================================================"
echo "Running Codex agent eval"
echo "============================================================"
uv run python run_eval.py --seeds "$SEEDS" --model "$MODEL" --agent codex --timeout "$TIMEOUT"

echo ""
echo "============================================================"
echo "All eval jobs completed!"
echo "Traces and artifacts saved to: $ARTIFACTS_DIR"
echo "============================================================"

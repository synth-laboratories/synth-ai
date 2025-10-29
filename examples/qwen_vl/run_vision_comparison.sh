#!/bin/bash
# Compare Qwen-VL (via synth) vs gpt-5-nano (via OpenAI) on Crafter

set -e

SEEDS=10
STEPS=20
OUTPUT_DIR="examples/qwen_vl/temp/comparison"

echo "======================================"
echo "Vision Model Comparison on Crafter"
echo "======================================"
echo ""
echo "Running $SEEDS episodes, $STEPS steps each"
echo ""

# Check API keys
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY not set"
    exit 1
fi

if [ -z "$SYNTH_API_KEY" ]; then
    echo "Error: SYNTH_API_KEY not set"
    exit 1
fi

# Run gpt-5-nano
echo "======================================"
echo "1. Running gpt-5-nano (OpenAI)"
echo "======================================"
uv run python examples/qwen_vl/crafter_gpt5nano_agent.py \
    --model gpt-5-nano \
    --seeds $SEEDS \
    --steps $STEPS \
    --output-dir "$OUTPUT_DIR/gpt5nano"

echo ""
echo "======================================"
echo "2. Running Qwen3-VL-8B (synth-ai)"
echo "======================================"
uv run python examples/qwen_vl/crafter_qwen_vl_agent.py \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --seeds $SEEDS \
    --steps $STEPS \
    --output-dir "$OUTPUT_DIR/qwen3vl"

echo ""
echo "======================================"
echo "Results Summary"
echo "======================================"
echo ""
echo "gpt-5-nano (OpenAI):"
cat "$OUTPUT_DIR/gpt5nano/gpt5nano_summary.json" | python -m json.tool
echo ""
echo "Qwen3-VL-8B (synth-ai):"
cat "$OUTPUT_DIR/qwen3vl/qwen_vl_summary.json" | python -m json.tool
echo ""
echo "Frames saved in:"
echo "  - $OUTPUT_DIR/gpt5nano/gpt5nano_frames/"
echo "  - $OUTPUT_DIR/qwen3vl/qwen_vl_frames/"

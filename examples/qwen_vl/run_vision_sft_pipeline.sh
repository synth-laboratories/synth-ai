#!/bin/bash
# Complete pipeline: Collect vision traces → Filter → Train SFT
# Uses synth-ai CLI tools for data collection and processing

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Allow callers to override root paths, otherwise derive them relative to this script.
SYNTH_DIR="${SYNTH_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
DEFAULT_MONOREPO_DIR="$(cd "$SYNTH_DIR/.." && pwd)/monorepo"
MONOREPO_DIR="${MONOREPO_DIR:-$DEFAULT_MONOREPO_DIR}"

if [ ! -d "$SYNTH_DIR" ]; then
    echo "Error: synth-ai repository not found at: $SYNTH_DIR"
    exit 1
fi

if [ ! -d "$MONOREPO_DIR" ]; then
    echo "Warning: MONOREPO_DIR not found at: $MONOREPO_DIR"
    echo "         Set MONOREPO_DIR to a valid path if you plan to run the optional training step."
fi

# Configuration
MODEL="gpt-5-nano"
PROVIDER="openai"
NUM_EPISODES=100
OUTPUT_DIR="traces/gpt5nano_vision"

echo "======================================"
echo "Vision SFT Pipeline for Crafter"
echo "======================================"
echo ""
echo "Model: $MODEL"
echo "Provider: $PROVIDER"
echo "Episodes: $NUM_EPISODES"
echo "Output: $OUTPUT_DIR"
echo ""

# Check API keys
if [ "$PROVIDER" = "openai" ]; then
    if [ -z "$OPENAI_API_KEY" ]; then
        echo "Error: OPENAI_API_KEY not set"
        exit 1
    fi
    echo "✓ OpenAI API key found"
elif [ "$PROVIDER" = "synth" ]; then
    if [ -z "$SYNTH_API_KEY" ]; then
        echo "Error: SYNTH_API_KEY not set"
        exit 1
    fi
    echo "✓ Synth API key found"
fi

if [ -z "$BACKEND_BASE_URL" ]; then
    echo "Warning: BACKEND_BASE_URL not set, using default"
    export BACKEND_BASE_URL="https://synth-backend-dev-docker.onrender.com/api"
fi

echo ""

# Step 1: Collect traces
echo "======================================"
echo "STEP 1: Collect Vision Traces"
echo "======================================"
echo ""
echo "Running $NUM_EPISODES episodes with $MODEL..."
echo "This will take ~30-60 minutes"
echo ""

cd "$SYNTH_DIR"

uvx synth-ai eval \
    --config examples/qwen_vl/configs/eval_${PROVIDER}_${MODEL/\//_}_vision.toml \
    --output-dir "$OUTPUT_DIR" \
    || {
        # Fallback to gpt5nano config if custom config not found
        uvx synth-ai eval \
            --config examples/qwen_vl/configs/eval_gpt5nano_vision.toml \
            --output-dir "$OUTPUT_DIR"
    }

echo ""
echo "✅ Trace collection complete!"
echo ""

# Step 2: Filter and export to SFT format
echo "======================================"
echo "STEP 2: Filter & Export to SFT JSONL"
echo "======================================"
echo ""

uvx synth-ai filter \
    --config examples/qwen_vl/configs/filter_vision_sft.toml \
    --input-db "$OUTPUT_DIR/rollouts.db" \
    --output-dir "$OUTPUT_DIR/sft"

echo ""
echo "✅ Filtering complete!"
echo ""

# Show dataset stats
echo "======================================"
echo "Dataset Statistics"
echo "======================================"
echo ""

if [ -f "$OUTPUT_DIR/sft/filter_stats.json" ]; then
    cat "$OUTPUT_DIR/sft/filter_stats.json" | python3 -m json.tool
else
    echo "Train samples: $(wc -l < "$OUTPUT_DIR/sft/train.jsonl")"
    echo "Val samples: $(wc -l < "$OUTPUT_DIR/sft/val.jsonl")"
fi

echo ""

# Step 3: Train SFT (optional - user can run this separately)
echo "======================================"
echo "STEP 3: Train Vision SFT (Optional)"
echo "======================================"
echo ""
echo "To train the model, run:"
echo ""
echo "  cd $MONOREPO_DIR"
echo "  uvx synth-ai train \\"
echo "    --type sft \\"
echo "    --config configs/vision_sft/crafter_qwen3vl_8b_gpt5nano.toml \\"
echo "    --dataset $SYNTH_DIR/$OUTPUT_DIR/sft/train.jsonl \\"
echo "    --eval-dataset $SYNTH_DIR/$OUTPUT_DIR/sft/val.jsonl \\"
echo "    --env-file backend/.env.dev"
echo ""

read -p "Run training now? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Starting SFT training..."
    echo ""
    
    if [ ! -d "$MONOREPO_DIR" ]; then
        echo "Error: MONOREPO_DIR not found. Set MONOREPO_DIR to your monorepo path before running training."
        exit 1
    fi
    
    cd "$MONOREPO_DIR"
    
    uvx synth-ai train \
        --type sft \
        --config configs/vision_sft/crafter_qwen3vl_8b_gpt5nano.toml \
        --dataset "$SYNTH_DIR/$OUTPUT_DIR/sft/train.jsonl" \
        --eval-dataset "$SYNTH_DIR/$OUTPUT_DIR/sft/val.jsonl" \
        --env-file backend/.env.dev
    
    echo ""
    echo "✅ Training complete!"
else
    echo ""
    echo "Skipping training. You can run it later using the command above."
fi

echo ""
echo "======================================"
echo "Pipeline Complete!"
echo "======================================"
echo ""
echo "📂 Outputs:"
echo "  - Raw traces: $OUTPUT_DIR/rollouts.db"
echo "  - SFT train: $OUTPUT_DIR/sft/train.jsonl"
echo "  - SFT val: $OUTPUT_DIR/sft/val.jsonl"
echo "  - Stats: $OUTPUT_DIR/sft/filter_stats.json"
echo ""
echo "🚀 Next steps:"
echo "  1. Train SFT model (see command above)"
echo "  2. Evaluate trained model"
echo "  3. Fine-tune with RL"
echo ""

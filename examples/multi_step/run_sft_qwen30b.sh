#!/bin/bash
# Run SFT for Qwen3-Coder-30B with LoRA on Crafter data

# Usage:
#   ./run_sft_qwen30b.sh <dataset_path> [env_file]
#
# Example:
#   ./run_sft_qwen30b.sh examples/multi_step/ft_data/crafter_traces.jsonl
#   ./run_sft_qwen30b.sh examples/multi_step/ft_data/crafter_traces.jsonl backend/.env.dev

set -e

DATASET_PATH="${1:-examples/sft/ft_data/crafter_traces.jsonl}"
ENV_FILE="${2:-backend/.env.dev}"

if [ ! -f "$DATASET_PATH" ]; then
    echo "Error: Dataset not found at $DATASET_PATH"
    echo "Usage: $0 <dataset_path> [env_file]"
    exit 1
fi

if [ ! -f "$ENV_FILE" ]; then
    echo "Error: Env file not found at $ENV_FILE"
    echo "Usage: $0 <dataset_path> [env_file]"
    exit 1
fi

echo "🚀 Starting SFT training for Qwen3-Coder-30B with LoRA"
echo "   Model: Qwen/Qwen3-Coder-30B-A3B-Instruct"
echo "   Dataset: $DATASET_PATH"
echo "   Config: examples/multi_step/configs/crafter_sft_qwen30b_lora.toml"
echo "   GPUs: 4x H200"
echo "   LoRA: r=16, alpha=32, all-linear"
echo ""

uvx synth-ai train \
  --type sft \
  --config examples/multi_step/configs/crafter_sft_qwen30b_lora.toml \
  --dataset "$DATASET_PATH" \
  --env-file "$ENV_FILE"

echo ""
echo "✅ SFT training job submitted!"
echo "   Monitor progress in your Synth dashboard"


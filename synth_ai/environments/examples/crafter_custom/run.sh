#!/bin/bash
# Convenience script for running custom Crafter evaluations

# Default values
MODEL="${MODEL:-gpt-4.1-nano}"
EPISODES="${EPISODES:-3}"
MAX_TURNS="${MAX_TURNS:-20}"
WORLD_CONFIG="${WORLD_CONFIG:-normal}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -e|--episodes)
            EPISODES="$2"
            shift 2
            ;;
        -t|--max-turns)
            MAX_TURNS="$2"
            shift 2
            ;;
        -w|--world-config)
            WORLD_CONFIG="$2"
            shift 2
            ;;
        --evaluate-traces)
            EVAL_TRACES="--evaluate-traces"
            shift
            ;;
        --analyze-traces)
            ANALYZE_TRACES="--analyze-traces"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [-m MODEL] [-e EPISODES] [-t MAX_TURNS] [-w WORLD_CONFIG] [--evaluate-traces] [--analyze-traces]"
            exit 1
            ;;
    esac
done

echo "Running Custom Crafter Evaluation"
echo "================================"
echo "Model: $MODEL"
echo "Episodes: $EPISODES"
echo "Max Turns: $MAX_TURNS"
echo "World Config: $WORLD_CONFIG"
echo

python -m synth_ai.environments.examples.crafter_custom.agent_demos.test_crafter_custom_agent \
    --model "$MODEL" \
    --episodes "$EPISODES" \
    --max-turns "$MAX_TURNS" \
    --world-config "$WORLD_CONFIG" \
    $EVAL_TRACES $ANALYZE_TRACES
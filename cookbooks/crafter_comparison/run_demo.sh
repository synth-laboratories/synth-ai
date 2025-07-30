#!/bin/bash

# Run Crafter experiments comparing gpt-4o-mini and gpt-4.1-mini

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Change to the synth-ai root directory
cd "$SCRIPT_DIR/../.."

# Run with 5 episodes per model, 50 max turns, with timeouts
python cookbooks/crafter_comparison/compare_models.py \
    --episodes 5 \
    --max-turns 50 \
    --difficulty easy \
    --models gpt-4o-mini gpt-4.1-mini \
    --base-seed 1000 \
    --turn-timeout 20.0 \
    --episode-timeout 180.0
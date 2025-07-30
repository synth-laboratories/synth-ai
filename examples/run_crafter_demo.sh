#!/bin/bash

# Run a Crafter agent demo with Gemini
# This script demonstrates a reactive agent in the Crafter environment

echo "🚀 Starting Crafter agent demo with Gemini 1.5 Flash..."
echo "Make sure the synth-ai service is running: uvx synth-ai serve"
echo ""

uv run python -m synth_ai.environments.examples.crafter_classic.agent_demos.test_crafter_react_agent --model gemini-1.5-flash
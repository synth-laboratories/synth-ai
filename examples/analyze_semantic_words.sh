#!/bin/bash

# Run Crafter agent and analyze semantic map word distribution
# This script demonstrates semantic analysis of agent observations
# Output: Markdown tables and JSON data (no plotting dependencies)

echo "🔍 Analyzing semantic map words from Crafter agent..."
echo "Make sure the synth-ai service is running: uvx synth-ai serve"
echo ""

cd synth_ai/environments/examples/crafter_classic/agent_demos/

# Run the semantic analysis (markdown output only)
python analyze_semantic_words_markdown.py --model gemini-1.5-flash --episodes 3 --max-turns 30

echo ""
echo "✅ Analysis complete! Check the generated markdown report and JSON files."
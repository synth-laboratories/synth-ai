#!/bin/bash
# Run judge model comparison integration test
# This will test all 4 models and save results to .txt files

set -e

cd "$(dirname "$0")/.."

echo "🧪 Running Judge Model Comparison Test"
echo "========================================"
echo ""
echo "This will test:"
echo "  - groq (qwen/qwen3-32b)"
echo "  - gpt-5-nano"
echo "  - gpt-5-mini"
echo "  - gpt-5"
echo ""
echo "Each model will evaluate 10 traces."
echo "Results will be saved to: tests/integration/judge_model_results/"
echo ""
echo "Estimated time: ~5-10 minutes"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Load environment variables
set -a
source .env
set +a

# Run the test
echo ""
echo "Starting comparison tests..."
echo ""

uv run pytest tests/integration/test_judge_models_comparison.py::test_judge_model_comparison -v -s

echo ""
echo "✅ Tests complete!"
echo ""
echo "📁 Results saved to: tests/integration/judge_model_results/"
echo ""
echo "View summary:"
echo "  cat tests/integration/judge_model_results/00_comparison_summary.txt"
echo ""
echo "View individual results:"
echo "  ls tests/integration/judge_model_results/*.txt"



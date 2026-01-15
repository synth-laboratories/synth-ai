#!/bin/bash
# Quick test script to validate both local and Daytona modes

set -e

echo "=========================================="
echo "Testing EngineBench Eval - Both Modes"
echo "=========================================="
echo ""

# Test 1: Local Mode
echo "=== TEST 1: LOCAL MODE (2 seeds) ==="
uv run python demos/engine_bench/run_eval.py --local --seeds 2 --model gpt-4o-mini --timeout 120
echo ""
echo "✅ Local mode test completed"
echo ""

# Test 2: Daytona Mode (if API key is set)
if [ -z "$DAYTONA_API_KEY" ]; then
    echo "=== TEST 2: DAYTONA MODE ==="
    echo "⚠️  DAYTONA_API_KEY not set, skipping Daytona test"
    echo "To run Daytona test, set: export DAYTONA_API_KEY=your_key"
    echo ""
else
    echo "=== TEST 2: DAYTONA MODE (2 seeds) ==="
    uv run python demos/engine_bench/run_eval.py --daytona --seeds 2 --model gpt-4o-mini --timeout 120
    echo ""
    echo "✅ Daytona mode test completed"
    echo ""
fi

echo "=========================================="
echo "All tests completed!"
echo "=========================================="

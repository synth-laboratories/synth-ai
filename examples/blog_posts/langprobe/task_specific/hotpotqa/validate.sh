#!/bin/bash
# Validate HotpotQA GEPA and MIPRO adapters

set -e

TASK_APP_URL="${1:-http://127.0.0.1:8110}"
BUDGET="${2:-10}"

echo "=== HotpotQA Validation ==="
echo "Task App URL: $TASK_APP_URL"
echo "Budget: $BUDGET"
echo ""

# Check if task app is running
echo "Checking if task app is running..."
if ! curl -s "$TASK_APP_URL/health" > /dev/null 2>&1; then
    echo "❌ Task app not running at $TASK_APP_URL"
    echo ""
    echo "Please start it first:"
    echo "  python -m examples.task_apps.gepa_benchmarks.hotpotqa_task_app --port 8110"
    exit 1
fi
echo "✅ Task app is running"
echo ""

# Test GEPA
echo "=== Testing Synth GEPA ==="
python3 -m examples.blog_posts.langprobe.task_specific.hotpotqa.run_synth_gepa_hotpotqa \
  --task-app-url "$TASK_APP_URL" \
  --rollout-budget "$BUDGET" || {
    echo "❌ GEPA validation failed"
    exit 1
}
echo "✅ GEPA validation passed"
echo ""

# Test MIPRO
echo "=== Testing Synth MIPRO ==="
python3 -m examples.blog_posts.langprobe.task_specific.hotpotqa.run_synth_mipro_hotpotqa \
  --task-app-url "$TASK_APP_URL" \
  --rollout-budget "$BUDGET" || {
    echo "❌ MIPRO validation failed"
    exit 1
}
echo "✅ MIPRO validation passed"
echo ""

echo "🎉 All validations passed!"


#!/bin/bash
# Run unified comparison of all frameworks on Iris

cd "$(dirname "$0")/../../../../.." || exit 1

BUDGET=${1:-100}

echo "Running comparison with budget: $BUDGET"
python3 -m examples.blog_posts.langprobe.task_specific.iris.run_comparison_iris --rollout-budget "$BUDGET"


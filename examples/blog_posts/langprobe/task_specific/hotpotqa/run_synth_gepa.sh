#!/bin/bash
# Run Synth GEPA on HotpotQA

cd "$(dirname "$0")/../../../../.." || exit 1

python3 -m examples.blog_posts.langprobe.task_specific.hotpotqa.run_synth_gepa_hotpotqa "$@"


#!/bin/bash
# Quick runner for DSPy GEPA on Iris with modest budget

cd "$(dirname "$0")/../../../../.." || exit 1
python3 -m examples.blog_posts.langprobe.task_specific.iris.run_dspy_gepa_iris "$@"

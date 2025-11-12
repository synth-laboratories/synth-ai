#!/bin/bash
# Quick runner for DSPy MIPROv2 on Iris with modest budget

cd "$(dirname "$0")/../../../../.." || exit 1
python3 -m examples.blog_posts.langprobe.task_specific.iris.run_dspy_miprov2_iris "$@"

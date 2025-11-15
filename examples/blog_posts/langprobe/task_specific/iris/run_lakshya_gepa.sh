#!/bin/bash
# Quick runner for Lakshya's GEPA on Iris with modest budget

cd "$(dirname "$0")/../../../../.." || exit 1
python3 -m examples.blog_posts.langprobe.task_specific.iris.run_lakshya_gepa_iris "$@"


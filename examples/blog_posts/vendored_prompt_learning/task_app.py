"""Task app helper for HeartDisease benchmark.

This module provides a simple way to create an in-process task app
for the HeartDisease classification benchmark.
"""

from __future__ import annotations

from pathlib import Path
import sys

# Add examples directory to path
REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_TASK_APPS = REPO_ROOT / "examples" / "task_apps" / "other_langprobe_benchmarks"
if str(EXAMPLES_TASK_APPS) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_TASK_APPS))

from heartdisease_task_app import build_config

__all__ = ["build_config"]


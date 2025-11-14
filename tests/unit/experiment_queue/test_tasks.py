from __future__ import annotations

import os
from pathlib import Path

import importlib
import stat

import pytest

from synth_ai.experiment_queue import tasks as queue_tasks


def _reload_tasks() -> None:
    """Reload tasks module to reset cached state."""
    importlib.reload(queue_tasks)
    # Reset any cached state that might interfere
    if hasattr(queue_tasks, '_default_train_cmd_cache'):
        delattr(queue_tasks, '_default_train_cmd_cache')



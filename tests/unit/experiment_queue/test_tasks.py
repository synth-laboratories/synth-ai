import importlib

from synth_ai.core.experiment_queue import tasks as queue_tasks


def _reload_tasks() -> None:
    """Reload tasks module to reset cached state."""
    importlib.reload(queue_tasks)
    # Reset any cached state that might interfere
    if hasattr(queue_tasks, "_default_train_cmd_cache"):
        delattr(queue_tasks, "_default_train_cmd_cache")

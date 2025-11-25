"""Task App SDK - in-process, deployed, and Modal task apps.

This module provides APIs for working with task apps:
- InProcessTaskApp: Local development with auto Cloudflare tunnel
- TaskAppConfig: Task app configuration
- create_task_app: Factory function

Example:
    from synth_ai.sdk.task_apps import InProcessTaskApp
    
    with InProcessTaskApp(app) as task_app:
        print(f"Task app URL: {task_app.url}")
        # Run training with task_app.url
"""

from __future__ import annotations

# Re-export from existing locations
from synth_ai.task.in_process import InProcessTaskApp
from synth_ai.task.server import TaskAppConfig, create_task_app

__all__ = [
    "InProcessTaskApp",
    "TaskAppConfig",
    "create_task_app",
]


"""Task app registry entry for the math demo Modal deployment."""

from __future__ import annotations

from contextlib import suppress
from importlib import import_module

from synth_ai.sdk.task.apps import ModalDeploymentConfig, TaskAppEntry, register_task_app

try:
    from synth_ai.sdk.task.apps.math_single_step import (  # type: ignore[unresolved-import]
        build_config as base_build_config,  # type: ignore[import-untyped]
    )
except ModuleNotFoundError:
    base_module = import_module("examples.rl.task_app.math_single_step")
    base_build_config = base_module.build_config

DEMO_MODAL_CONFIG = ModalDeploymentConfig(
    app_name="hendrycks-math-task-app",
    pip_packages=(
        "fastapi>=0.110.0",
        "uvicorn>=0.23.0",
        "pydantic>=2.6.0",
        "httpx>=0.24.0",
        "numpy>=1.24.0",
        "aiohttp>=3.8.0",
        "datasets>=2.16.0",
        "synth-ai",
    ),
)


def build_config():
    """Reuse the shared math single-step TaskAppConfig."""

    return base_build_config()


def register_demo_entry() -> None:
    entry = TaskAppEntry(
        app_id="hendrycks-math-demo",
        description="Demo math task app (Modal-focused) shipping with synth-ai demos.",
        config_factory=build_config,
        modal=DEMO_MODAL_CONFIG,
    )
    with suppress(ValueError):
        register_task_app(entry=entry)


register_demo_entry()


__all__ = ["DEMO_MODAL_CONFIG", "build_config", "register_demo_entry"]

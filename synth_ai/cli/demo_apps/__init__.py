"""Namespace for demo task apps (math, crafter, etc.)."""

import contextlib

# Ensure registry entries are loaded for CLI discovery.
with contextlib.suppress(Exception):  # pragma: no cover - optional on downstream installs
    from synth_ai.cli.demo_apps.math import task_app_entry  # noqa: F401

with contextlib.suppress(Exception):  # pragma: no cover - optional on downstream installs
    from synth_ai.cli.demo_apps.crafter import grpo_crafter_task_app  # noqa: F401

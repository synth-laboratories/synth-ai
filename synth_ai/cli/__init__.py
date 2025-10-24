"""CLI subcommands for Synth AI.

This package hosts modular commands (watch, traces, recent, calc, status)
and exposes a top-level Click group named `cli` compatible with the
pyproject entry point `synth_ai.cli:cli`.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any, cast

# Load environment variables from a local .env if present (repo root)
try:
    from dotenv import find_dotenv, load_dotenv

    # Source .env early so CLI subcommands inherit config; do not override shell
    load_dotenv(find_dotenv(usecwd=True), override=False)
except Exception:
    # dotenv is optional at runtime; proceed if unavailable
    pass

try:
    from synth_ai.cli._typer_patch import patch_typer_make_metavar

    patch_typer_make_metavar()
except Exception:
    pass


from synth_ai.cli.root import cli  # new canonical CLI entrypoint

# Register subcommands from this package onto the group
# Deprecated/legacy commands intentionally not registered: watch/experiments, balance, calc,
# man, recent, status, traces
try:
    from synth_ai.cli import demo as _demo

    _demo.register(cli)
except Exception:
    pass
try:
    from synth_ai.cli import turso as _turso

    _turso.register(cli)
except Exception:
    pass
try:
    _train_module = cast(Any, importlib.import_module("synth_ai.api.train"))
    _train_register = cast(Callable[[Any], None], _train_module.register)
    _train_register(cli)
except Exception:
    pass


# Import task_app_group conditionally
try:
    from synth_ai.cli.task_apps import task_app_group
    cli.add_command(task_app_group, name="task-app")
except Exception:
    # Task app functionality not available
    pass


try:
    # Make task_apps import more robust to handle missing optional dependencies
    import importlib
    task_apps_module = importlib.import_module('synth_ai.cli.task_apps')
    task_apps_module.register(cli)
except (ImportError, ModuleNotFoundError, TypeError, RuntimeError) as e:
    # Task apps module not available (missing optional dependencies)
    # This is expected - silently skip
    pass

# Register TUI command - make import completely isolated
def _register_tui_command():
    """Register TUI command only when called, not during CLI startup."""
    try:
        # Import TUI only when the command is actually used
        from synth_ai.cli.tui import register as tui_register
        tui_register(cli)
    except Exception:
        # TUI not available - this is expected if dependencies are missing
        pass

# Add TUI command as a lazy-registered command
try:
    # Try to import and register immediately for normal cases
    from synth_ai.cli.tui import register as tui_register
    tui_register(cli)
except Exception:
    # If that fails, add a lazy registration that will only happen when called
    # For now, just skip - the command won't be available but CLI won't crash
    pass

# Add task app commands if available
try:
    if 'task_app_group' in locals() and hasattr(task_app_group, 'commands'):
        cli.add_command(task_app_group.commands["serve"], name="serve")
        cli.add_command(task_app_group.commands["deploy"], name="deploy")
        cli.add_command(task_app_group.commands["modal-serve"], name="modal-serve")
except Exception:
    # Task app commands not available
    pass
# Top-level 'info' alias removed; use `synth-ai task-app info` instead

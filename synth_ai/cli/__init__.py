"""CLI subcommands for Synth AI.

This package hosts modular commands (watch, traces, recent, calc, status)
and exposes a top-level Click group named `cli` compatible with the
pyproject entry point `synth_ai.cli:cli`.
"""

from __future__ import annotations

import importlib

# Load environment variables from a local .env if present (repo root)
try:
    from dotenv import find_dotenv, load_dotenv

    # Source .env early so CLI subcommands inherit config; do not override shell
    load_dotenv(find_dotenv(usecwd=True), override=False)
except Exception:
    # dotenv is optional at runtime; proceed if unavailable
    pass

try:
    from ._typer_patch import patch_typer_make_metavar

    patch_typer_make_metavar()
except Exception:
    pass


from .root import cli  # new canonical CLI entrypoint

# Register subcommands from this package onto the group
# Deprecated/legacy commands intentionally not registered: watch/experiments, balance, calc,
# man, recent, status, traces
try:
    from . import demo as _demo

    _demo.register(cli)
except Exception:
    pass
try:
    from . import turso as _turso

    _turso.register(cli)
except Exception:
    pass
try:
    _train_module = importlib.import_module("synth_ai.api.train")
    _train_register = _train_module.register
    _train_register(cli)
except Exception:
    pass


from .task_apps import task_app_group

cli.add_command(task_app_group, name="task-app")


try:
    from . import task_apps as _task_apps

    _task_apps.register(cli)
except Exception:
    pass

cli.add_command(task_app_group.commands["serve"], name="serve")
cli.add_command(task_app_group.commands["deploy"], name="deploy")

cli.add_command(task_app_group.commands["modal-serve"], name="modal-serve")
# Top-level 'info' alias removed; use `synth-ai task-app info` instead

"""CLI subcommands for Synth AI.

This package hosts modular commands (watch, traces, recent, calc, status)
and exposes a top-level Click group named `cli` compatible with the
pyproject entry point `synth_ai.cli:cli`.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable
from typing import Any

# Load environment variables from a local .env if present (repo root)
try:
    from dotenv import find_dotenv, load_dotenv

    # Source .env early so CLI subcommands inherit config; do not override shell
    load_dotenv(find_dotenv(usecwd=True), override=False)
except Exception:
    # dotenv is optional at runtime; proceed if unavailable
    pass

def _callable_from(module: Any, attr: str) -> Callable[..., Any] | None:
    candidate = getattr(module, attr, None)
    return candidate if callable(candidate) else None


def _maybe_import(module_path: str) -> Any | None:
    try:
        return importlib.import_module(module_path)
    except Exception:
        return None


def _maybe_call(module_path: str, attr: str, *args: Any, **kwargs: Any) -> None:
    module = _maybe_import(module_path)
    if not module:
        return
    fn = _callable_from(module, attr)
    if fn:
        fn(*args, **kwargs)


# Apply Typer patch if available
_maybe_call("synth_ai.cli._typer_patch", "patch_typer_make_metavar")


_cli_module = _maybe_import("synth_ai.cli.root")
if not _cli_module:
    raise ImportError("synth_ai.cli.root is required for CLI entrypoint")
cli = _cli_module.cli  # type: ignore[attr-defined]

# Register core commands implemented as standalone modules
try:
    from synth_ai.cli.setup import setup_cmd

    cli.add_command(setup_cmd, name="setup")
except Exception:
    pass


# Register optional subcommands packaged under synth_ai.cli.*
for _module_path in ("synth_ai.cli.commands.demo", "synth_ai.cli.commands.status", "synth_ai.cli.turso"):
    module = _maybe_import(_module_path)
    if not module:
        continue
    sub_name = _module_path.rsplit(".", 1)[-1]
    setattr(sys.modules[__name__], sub_name, module)
    fn = _callable_from(module, "register")
    if fn:
        fn(cli)

# Register help command
_maybe_call("synth_ai.cli.commands.help.core", "register", cli)

# Train CLI lives under synth_ai.api.train
_maybe_call("synth_ai.api.train", "register", cli)

# Task app group/commands are optional and have richer API surface
_task_apps_module = _maybe_import("synth_ai.cli.task_apps")
if _task_apps_module:
    task_app_group = getattr(_task_apps_module, "task_app_group", None)
    if task_app_group is not None:
        cli.add_command(task_app_group, name="task-app")
        # Expose common aliases when present
        commands = getattr(task_app_group, "commands", None)
        if isinstance(commands, dict):
            for alias, name in (("serve", "serve"), ("deploy", "deploy"), ("modal-serve", "modal-serve")):
                command = commands.get(name)
                if command is not None:
                    cli.add_command(command, name=alias)
    register_task_apps = _callable_from(_task_apps_module, "register")
    if register_task_apps:
        register_task_apps(cli)

# Top-level 'info' alias removed; use `synth-ai task-app info` instead

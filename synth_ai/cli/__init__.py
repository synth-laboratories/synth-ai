"""CLI subcommands for Synth AI.

This package hosts modular commands and exposes a top-level Click group
named `cli` compatible with the pyproject entry point `synth_ai.cli:cli`.
"""

import importlib
import sys
from collections.abc import Callable
from typing import Any

from synth_ai.cli.agents import claude_cmd, codex_cmd, opencode_cmd
from synth_ai.cli.demos.demo import demo_cmd
from synth_ai.cli.deploy import deploy_cmd
from synth_ai.cli.infra.mcp import mcp_cmd
from synth_ai.cli.infra.modal_app import modal_app_cmd
from synth_ai.cli.infra.setup import setup_cmd
from synth_ai.cli.task_apps import task_app_cmd
from synth_ai.cli.training.train_cfg import train_cfg_cmd

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
_maybe_call("synth_ai.cli._internal.typer_patch", "patch_typer_make_metavar")


_cli_module = _maybe_import("synth_ai.cli.root")
if not _cli_module:
    raise ImportError("synth_ai.cli.root is required for CLI entrypoint")
cli = _cli_module.cli  # type: ignore[attr-defined]



# Register core commands implemented as standalone modules

cli.add_command(claude_cmd, name="claude")
cli.add_command(codex_cmd, name="codex")
cli.add_command(demo_cmd, name="demo")
cli.add_command(deploy_cmd, name="deploy")
cli.add_command(mcp_cmd, name="mcp")
cli.add_command(modal_app_cmd, name="modal-app")
cli.add_command(opencode_cmd, name="opencode")
cli.add_command(setup_cmd, name="setup")
cli.add_command(task_app_cmd, name="task-app")
cli.add_command(train_cfg_cmd, name="train-cfg")


# Register optional subcommands packaged under synth_ai.cli.*
for _module_path in ("synth_ai.cli.commands.demo", "synth_ai.cli.infra.turso"):
    module = _maybe_import(_module_path)
    if not module:
        continue
    sub_name = _module_path.rsplit(".", 1)[-1]
    setattr(sys.modules[__name__], sub_name, module)
    fn = _callable_from(module, "register")
    if fn:
        fn(cli)

# Smoke command registration (CLI-only helper)
try:
    from synth_ai.cli.commands.smoke import register as register_smoke

    register_smoke(cli)
except Exception:
    pass

# Register help command
_maybe_call("synth_ai.cli.commands.help.core", "register", cli)

# Register scan command
_maybe_call("synth_ai.cli.commands.scan", "register", cli)

# Train CLI lives under synth_ai.sdk.api.train
_maybe_call("synth_ai.sdk.api.train", "register", cli)

# Task app group/commands are optional and have richer API surface
_task_apps_module = _maybe_import("synth_ai.cli.task_apps")
#if _task_apps_module:
task_app_group = getattr(_task_apps_module, "task_app_group", None)
if task_app_group is not None:
    cli.add_command(task_app_group, name="task-app-group")
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

# Experiment queue commands
_maybe_call("synth_ai.cli.utils.experiments", "register", cli)
_maybe_call("synth_ai.cli.utils.queue", "register", cli)

# Artifacts commands
_maybe_call("synth_ai.cli.commands.artifacts", "register", cli)

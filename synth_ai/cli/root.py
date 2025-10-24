#!/usr/bin/env python3
"""
Canonical CLI entrypoint for Synth AI (moved from synth_ai/cli.py).
"""

from __future__ import annotations

import click
from synth_ai.cli.setup import setup
from synth_ai.cli.task_app_deploy import deploy_command as main_deploy_command

try:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _pkg_version

    try:
        __pkg_version__ = _pkg_version("synth-ai")
    except PackageNotFoundError:
        try:
            from synth_ai import __version__ as __pkg_version__  # type: ignore
        except Exception:
            __pkg_version__ = "unknown"
except Exception:
    try:
        from synth_ai import __version__ as __pkg_version__  # type: ignore
    except Exception:
        __pkg_version__ = "unknown"

@click.group(
    help=f"Synth AI v{__pkg_version__} - Software for aiding the best and multiplying the will."
)
@click.version_option(version=__pkg_version__, prog_name="synth-ai")
def cli():
    """Top-level command group for Synth AI."""


@cli.command(name="setup")
def setup_command():
    """Perform SDK handshake and persist keys to user_config.json."""
    code = setup()
    if code:
        raise click.exceptions.Exit(code)


cli.add_command(main_deploy_command, name="deploy")

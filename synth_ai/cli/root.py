#!/usr/bin/env python3
"""
Canonical CLI entrypoint for Synth AI (moved from synth_ai/cli.py).
"""

from __future__ import annotations

import click

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
    """Perform SDK handshake and write keys to .env."""
    from synth_ai.cli.demo.common import forward_to_core

    forward_to_core(["demo.setup"])

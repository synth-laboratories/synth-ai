#!/usr/bin/env python3
"""
RL demo command group - deprecated.

The demo apps have been removed. These commands are no longer functional.
"""

from __future__ import annotations

import click


def _deprecated_message() -> None:
    """Show deprecation message."""
    click.echo("⚠️  Demo apps have been removed from synth-ai.")
    click.echo("")
    click.echo("For examples of task apps, see:")
    click.echo("  - https://docs.usesynth.ai/guides/local-api")
    click.echo("  - The examples/ directory in this repository")


def register(cli):
    @cli.group("rl_demo")
    def rl_demo():
        """[DEPRECATED] RL Demo commands have been removed."""
        _deprecated_message()

    @rl_demo.command("setup")
    def rl_setup():
        _deprecated_message()

    @rl_demo.command("deploy")
    def rl_deploy():
        _deprecated_message()

    @rl_demo.command("configure")
    def rl_configure():
        _deprecated_message()

    @rl_demo.command("init")
    def rl_init():
        _deprecated_message()

    @rl_demo.command("run")
    def rl_run():
        _deprecated_message()

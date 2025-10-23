from __future__ import annotations

import click

from .run import run_job


def run_configure() -> int:
    return run_job(
        config=None,
        batch_size=None,
        group_size=None,
        model=None,
        timeout=600,
        dry_run=False,
    )


def register(group):
    @group.command("configure")
    def demo_configure():
        code = run_configure()
        if code:
            raise click.exceptions.Exit(code)

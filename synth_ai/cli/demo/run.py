from __future__ import annotations

import click

from .common import forward_to_core


def register(group):
    @group.command("run")
    @click.option("--batch-size", type=int, default=None)
    @click.option("--group-size", type=int, default=None)
    @click.option("--model", type=str, default=None)
    @click.option("--timeout", type=int, default=600)
    def demo_run(
        batch_size: int | None,
        group_size: int | None,
        model: str | None,
        timeout: int,
    ):
        args = ["run"]
        if batch_size is not None:
            args.extend(["--batch-size", str(batch_size)])
        if group_size is not None:
            args.extend(["--group-size", str(group_size)])
        if model:
            args.extend(["--model", model])
        if timeout:
            args.extend(["--timeout", str(timeout)])
        forward_to_core(args)

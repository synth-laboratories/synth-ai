from __future__ import annotations

import click

from .common import forward_to_core


def register(group):
    @group.command("init")
    @click.option("--template", type=str, default=None, help="Template id to instantiate")
    @click.option("--dest", type=str, default=None, help="Destination directory for files")
    @click.option("--force", is_flag=True, help="Overwrite existing files in destination")
    def demo_init(template: str | None, dest: str | None, force: bool):
        args: list[str] = ["demo.init"]
        if template:
            args.extend(["--template", template])
        if dest:
            args.extend(["--dest", dest])
        if force:
            args.append("--force")
        forward_to_core(args)

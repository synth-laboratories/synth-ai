import shutil
from pathlib import Path
from typing import Literal, TypeAlias, get_args

import click

from synth_ai.core.paths import REPO_ROOT

DemoType: TypeAlias = Literal[
    "mipro",
    "sft",
    "rl"
]


@click.command()
@click.argument(
    "demo_type",
    type=click.Choice(get_args(DemoType)),
)
def demo_cmd(demo_type: DemoType) -> None:
    src = REPO_ROOT / "synth_ai" / "demos" / demo_type
    if not src.exists():
        raise click.ClickException(f"Demo source directory not found: {src}")
    dst = Path.cwd() / f"demo_{src.name}"
    if dst.exists():
        if not click.confirm(f"Destination already exists: {dst}. Overwrite?", abort=True):
            click.echo("Aborted.")
            return
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    click.echo(f"Copied {demo_type.upper()} demo to your CWD: {dst}")

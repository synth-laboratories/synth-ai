from pathlib import Path

import click

from synth_ai.cli.lib.train_cfgs import find_train_cfgs_in_cwd, validate_train_cfg
from synth_ai.core.paths import print_paths_formatted


@click.command()
@click.argument(
    "action",
    type=click.Choice(["check", "list"]),
    metavar="[ACTION]"
)
@click.argument(
    "path",
    type=click.Path(path_type=Path, exists=True),
    metavar="[PATH]",
    required=False
)
def train_cfg_cmd(
    action: str,
    path: Path | None,
) -> None:
    try:
        match action:
            case "check":
                if path is None:
                    raise click.ClickException("PATH is required for 'check'")
                validate_train_cfg(path)
            case "list":
                print_paths_formatted(find_train_cfgs_in_cwd())
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

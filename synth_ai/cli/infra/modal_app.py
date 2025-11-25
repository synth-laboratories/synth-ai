from pathlib import Path

import click

from synth_ai.cli.lib.apps.modal_app import find_modal_apps_in_cwd, validate_modal_app
from synth_ai.core.apps.common import run_ruff_check
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
def modal_app_cmd(
    action: str,
    path: Path | None,
) -> None:
    try:
        match action:
            case "check":
                if path is None:
                    raise click.ClickException("PATH is required for 'check'")
                validate_modal_app(path)
                raise SystemExit(run_ruff_check(path))
            case "list":
                print_paths_formatted(find_modal_apps_in_cwd())
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

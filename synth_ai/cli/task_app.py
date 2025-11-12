from pathlib import Path

import click
from synth_ai.utils.apps import run_ruff_check, validate_task_app


@click.command()
@click.argument(
    "action",
    type=click.Choice(["check"]),
    metavar="[ACTION]"

)
@click.argument(
    "task_app_path",
    type=click.Path(path_type=Path, exists=False),
    metavar="[PATH]"
)
@click.option(
    "--fix",
    is_flag=True,
    default=False,
    help="Pass --fix through to Ruff so autofixable lint issues get patched.",
)
def task_app_cmd(
    action: str,
    task_app_path: Path,
    fix: bool,
) -> None:
    try:
        match action:
            case "check":
                validate_task_app(task_app_path)
                raise SystemExit(run_ruff_check(task_app_path, fix))
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

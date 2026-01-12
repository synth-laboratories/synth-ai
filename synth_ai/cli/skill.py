"""Skill command.

Installs synth-ai packaged OpenCode skills into the user's local OpenCode skills directory.

Example:
  uvx synth-ai skill list
  uvx synth-ai skill install synth-api
"""

from __future__ import annotations

from pathlib import Path

import click


@click.group()
def skill() -> None:
    """Manage packaged OpenCode skills (list/install)."""


@skill.command("list")
def list_() -> None:
    """List packaged skills shipped with synth-ai."""

    from synth_ai.sdk.opencode_skills import list_packaged_opencode_skill_names

    for name in list_packaged_opencode_skill_names():
        click.echo(name)


@skill.command("install")
@click.argument("name", type=str, required=False, default="all")
@click.option(
    "--dir",
    "dest_dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Destination OpenCode skills directory (defaults to ~/.config/opencode/skill).",
)
@click.option("--force", is_flag=True, help="Overwrite existing skill files.")
def install(name: str, dest_dir: Path | None, force: bool) -> None:
    """Install a packaged skill into the local OpenCode skills directory."""

    from synth_ai.sdk.opencode_skills import (
        default_opencode_global_skills_dir,
        install_all_packaged_opencode_skills,
        install_packaged_opencode_skill,
        list_packaged_opencode_skill_names,
    )

    if dest_dir is None:
        dest_dir = default_opencode_global_skills_dir()

    if name == "all":
        paths = install_all_packaged_opencode_skills(dest_skills_dir=dest_dir, force=force)
        if not paths:
            raise click.ClickException("No packaged skills found.")
        click.echo(str(dest_dir))
        return

    available = set(list_packaged_opencode_skill_names())
    if name not in available:
        raise click.ClickException(
            f"Unknown skill '{name}'. Available: {', '.join(sorted(available))}"
        )

    out = install_packaged_opencode_skill(skill_name=name, dest_skills_dir=dest_dir, force=force)
    click.echo(str(out))

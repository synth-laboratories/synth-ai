from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import click
from synth_ai.cli.lib.task_app_env import ensure_env_credentials, hydrate_user_environment
from synth_ai.cli.lib.task_app_state import persist_env_api_key
from synth_ai.cli.lib.user_config import load_user_config, update_user_config

from . import task_app
from .utils import mask_value


@dataclass(slots=True)
class KeySpec:
    name: str
    description: str
    secret: bool = True
    allow_modal_secret: bool = False
    allow_modal_app: bool = False
    modal_secret_pattern: str | None = None
    optional: bool = False


def resolve_env(
    *,
    config_path: Path | None,
    explicit_env_paths: Iterable[str],
    required_keys: list[KeySpec],
) -> tuple[Path, dict[str, str]]:
    """Resolve required environment values without relying on .env files."""

    supplied_paths = list(explicit_env_paths)
    if supplied_paths:
        raise click.ClickException("The --env-file option is no longer supported.")

    hydrate_user_environment(override=False)
    ensure_env_credentials(require_synth=False)
    user_config = load_user_config()

    resolved: dict[str, str] = {}
    for spec in required_keys:
        value = _resolve_key(spec, user_config)
        if value:
            resolved[spec.name] = value
    return Path.cwd(), resolved


def _resolve_key(spec: KeySpec, user_config: dict[str, str]) -> str:
    existing = (os.environ.get(spec.name) or user_config.get(spec.name) or "").strip()
    if existing:
        os.environ[spec.name] = existing
        return existing

    value: str | None = None

    if spec.allow_modal_secret and _prompt_yes_no(f"Fetch {spec.name} from Modal secrets?", default=True):
        value = _fetch_modal_secret(spec)
    if not value and spec.allow_modal_app and _prompt_yes_no(f"Select Modal app for {spec.name}?", default=True):
        value = _fetch_modal_app(spec)

    if not value and not spec.optional:
        prompt = spec.description or spec.name
        value = click.prompt(prompt, hide_input=spec.secret, default="", show_default=False).strip()
        if not value:
            raise click.ClickException(f"{spec.name} is required.")

    if value:
        os.environ[spec.name] = value
        update_user_config({spec.name: value})
        if spec.name == "ENVIRONMENT_API_KEY":
            persist_env_api_key(value)
    elif spec.optional:
        value = ""
    return value or ""


def _fetch_modal_secret(spec: KeySpec) -> str | None:
    try:
        names = task_app.list_modal_secrets(spec.modal_secret_pattern)
    except click.ClickException as exc:
        click.echo(str(exc))
        return None
    if not names:
        click.echo("No Modal secrets matched")
        return None
    click.echo(task_app.format_modal_secrets(names))
    idx = click.prompt("Select secret (0 to cancel)", type=int, default=0)
    if idx <= 0 or idx > len(names):
        return None
    name = names[idx - 1]
    value = task_app.get_modal_secret_value(name)
    if value:
        click.echo(f"Selected secret {name} ({mask_value(value)})")
    return value


def _fetch_modal_app(spec: KeySpec) -> str | None:
    try:
        apps = task_app.list_modal_apps("task-app")
    except click.ClickException as exc:
        click.echo(str(exc))
        return None
    if not apps:
        click.echo("No Modal apps matched")
        return None
    click.echo(task_app.format_modal_apps(apps))
    idx = click.prompt("Select app (0 to cancel)", type=int, default=0)
    if idx <= 0 or idx > len(apps):
        return None
    url = apps[idx - 1].url
    if url:
        click.echo(f"Selected Modal app URL: {url}")
    return url


def _prompt_yes_no(message: str, *, default: bool) -> bool:
    default_token = "y" if default else "n"
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        response = click.prompt(f"{message} {suffix}", default=default_token, show_default=False)
        normalized = str(response).strip().lower()
        if normalized in {"", "y", "yes", "1", "true", "t"}:
            return True
        if normalized in {"n", "no", "0", "false", "f"}:
            return False
        click.echo("Invalid input; enter 'y' or 'n'.")


__all__ = [
    "KeySpec",
    "resolve_env",
]

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click

from synth_ai.core.env import resolve_env_var
from synth_ai.core.telemetry import log_info

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
    required_keys: list[KeySpec],
    config_toml_path: str | None = None,
) -> tuple[Path, dict[str, str]]:
    """Resolve environment values from process env or user config."""
    ctx: dict[str, Any] = {
        "config_path": str(config_path) if config_path else None,
        "config_toml_path": config_toml_path,
        "required_key_count": len(required_keys),
    }
    log_info("resolve_env invoked", ctx=ctx)
    resolved: dict[str, str] = {}
    for spec in required_keys:
        value = _resolve_key(spec)
        if value:
            resolved[spec.name] = value
    return Path.cwd(), resolved


def _resolve_key(spec: KeySpec) -> str:
    """Resolve a key value without interactive prompts.

    Priority:
    1. Environment variable
    2. Secrets helper (resolve_env_var)

    Fails hard if not found (no interactive prompts).
    """
    # Priority: existing environment variable
    env_val = os.environ.get(spec.name)
    # Allow common aliases in env to satisfy required keys without extra prompts
    if not env_val and spec.name == "ENVIRONMENT_API_KEY":
        for alias in ("dev_environment_api_key", "DEV_ENVIRONMENT_API_KEY"):
            alt = os.environ.get(alias)
            if alt:
                env_val = alt
                os.environ[spec.name] = alt
                click.echo(f"Found {spec.name} via {alias}: {mask_value(alt)}")
                break
    if not env_val and spec.name == "TASK_APP_URL":
        for alias in ("TASK_APP_BASE_URL",):
            alt = os.environ.get(alias)
            if alt:
                env_val = alt
                os.environ[spec.name] = alt
                click.echo(f"Found {spec.name} via {alias}: {mask_value(alt)}")
                break
    if env_val:
        click.echo(f"Found {spec.name} in current sources: {mask_value(env_val)}")
        os.environ[spec.name] = env_val
        return env_val

    # Try secrets helper
    resolve_env_var(spec.name)
    resolved_value = os.environ.get(spec.name)
    if resolved_value:
        click.echo(f"Found {spec.name} via user config: {mask_value(resolved_value)}")
        os.environ[spec.name] = resolved_value
        return resolved_value

    # Not found - fail hard with informative message
    if spec.optional:
        return ""

    # Build helpful error message
    raise click.ClickException(
        f"❌ Missing required credential: {spec.name}\n\n"
        f"  Description: {spec.description}\n"
        f"  Options:\n"
        f"  1. Set environment variable: export {spec.name}=<value>\n"
        f"  2. Run `synth-ai setup` to store credentials in ~/.synth-ai"
    )


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
    idx = click.prompt("Select secret (0 to cancel)", type=int)
    if idx == 0:
        return None
    if idx < 1 or idx > len(names):
        click.echo("Invalid selection")
        return None
    name = names[idx - 1]
    value = task_app.get_modal_secret_value(name)
    os.environ[spec.name] = value
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
    idx = click.prompt("Select app (0 to cancel)", type=int)
    if idx == 0:
        return None
    if idx < 1 or idx > len(apps):
        click.echo("Invalid selection")
        return None
    url = apps[idx - 1].url
    os.environ[spec.name] = url
    return url


def _raise_abort(spec: KeySpec) -> None:
    raise click.ClickException(f"Missing required value for {spec.name}")


__all__ = [
    "KeySpec",
    "resolve_env",
]

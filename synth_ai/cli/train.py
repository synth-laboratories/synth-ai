from __future__ import annotations

import os
from pathlib import Path

import click
from synth_ai._utils.base_url import get_backend_from_env
from synth_ai.api.train.cli import handle_rl, handle_sft
from synth_ai.api.train.config_finder import discover_configs, prompt_for_config
from synth_ai.api.train.env_resolver import KeySpec, resolve_env
from synth_ai.api.train.utils import ensure_api_base, mask_value


def _default_backend() -> str:
    explicit = os.getenv("BACKEND_BASE_URL", "").strip()
    if explicit:
        return explicit
    base, _ = get_backend_from_env()
    return f"{base}/api" if not base.endswith("/api") else base


@click.command("train")
@click.option(
    "--config",
    "config_paths",
    multiple=True,
    type=click.Path(),
    help="Path to training TOML (repeatable)",
)
@click.option(
    "--type",
    "train_type",
    type=click.Choice(["rl", "sft"]),
    default=None,
    help="Training workflow (rl or sft). If omitted, you will be prompted.",
)
@click.option("--task-url", default=None, help="Override task app base URL (RL only)")
@click.option(
    "--dataset",
    "dataset_path",
    type=click.Path(),
    default=None,
    help="Override dataset JSONL path (SFT)",
)
@click.option("--backend", default=_default_backend, help="Backend base URL")
@click.option("--model", default=None, help="Override model identifier")
@click.option(
    "--allow-experimental",
    "allow_experimental",
    is_flag=True,
    flag_value=True,
    default=None,
    help="Allow experimental models (overrides SDK_EXPERIMENTAL env)",
)
@click.option(
    "--no-allow-experimental",
    "allow_experimental",
    is_flag=True,
    flag_value=False,
    help="Disallow experimental models (overrides SDK_EXPERIMENTAL env)",
)
@click.option("--idempotency", default=None, help="Idempotency-Key header for job creation")
@click.option("--dry-run", is_flag=True, hidden=True, help="Deprecated: no-op")
@click.option("--poll/--no-poll", default=True, help="Poll job status until terminal state")
@click.option(
    "--poll-timeout", default=3600.0, type=float, help="Maximum seconds to poll before timing out"
)
@click.option("--poll-interval", default=15.0, type=float, help="Seconds between poll attempts")
@click.option(
    "--examples",
    "examples_limit",
    type=int,
    default=None,
    help="Limit SFT training to the first N examples",
)
def train_command(
    config_paths: tuple[str, ...],
    train_type: str | None,
    task_url: str | None,
    dataset_path: str | None,
    backend: str,
    model: str | None,
    allow_experimental: bool | None,
    idempotency: str | None,
    dry_run: bool,
    poll: bool,
    poll_timeout: float,
    poll_interval: float,
    examples_limit: int | None,
) -> None:
    """Interactive launcher for RL / SFT jobs."""

    if train_type is None:
        all_candidates = discover_configs(list(config_paths), requested_type=None)
        if not all_candidates:
            raise click.ClickException("No training configs found. Pass --config to specify explicitly.")

        available_types = sorted({c.train_type for c in all_candidates if c.train_type in {"rl", "sft"}})

        if not available_types:
            train_type = click.prompt("Training type", type=click.Choice(["rl", "sft"]))
        elif len(available_types) == 1:
            train_type = available_types[0]
            click.echo(f"Detected training type: {train_type}")
        else:
            train_type = click.prompt(
                "Training type",
                type=click.Choice(available_types),
            )

        candidates = [c for c in all_candidates if c.train_type == train_type]
        if not candidates:
            raise click.ClickException(
                f"No configs marked for '{train_type}'. Pass --config to select manually."
            )
    else:
        candidates = discover_configs(
            list(config_paths),
            requested_type=train_type,
        )

    if not candidates:
        raise click.ClickException("No training configs matched the requested type.")
    selection = prompt_for_config(
        candidates,
        requested_type=train_type,
        allow_autoselect=bool(config_paths),
    )

    cfg_type = selection.train_type
    if cfg_type and cfg_type != train_type:
        click.echo(
            f"[WARN] Config {selection.path} is tagged as '{cfg_type}'. Continuing as '{train_type}'."
        )

    cfg_path = selection.path
    click.echo(f"Using config: {cfg_path} ({train_type})")

    required_keys: list[KeySpec] = []
    if train_type == "rl":
        required_keys.append(KeySpec("SYNTH_API_KEY", "Synth API key for backend"))
        required_keys.append(
            KeySpec(
                "ENVIRONMENT_API_KEY",
                "Environment API key for task app",
                allow_modal_secret=True,
                modal_secret_pattern="env",
            )
        )
        required_keys.append(
            KeySpec(
                "TASK_APP_URL",
                "Task app base URL",
                secret=False,
                allow_modal_app=True,
                optional=bool(task_url),
            )
        )
    else:
        required_keys.append(KeySpec("SYNTH_API_KEY", "Synth API key for backend"))

    _, env_values = resolve_env(
        config_path=cfg_path,
        explicit_env_paths=(),
        required_keys=required_keys,
    )

    click.echo("Environment credentials loaded.")

    synth_key = env_values.get("SYNTH_API_KEY") or os.environ.get("SYNTH_API_KEY")
    if not synth_key:
        raise click.ClickException("SYNTH_API_KEY required")

    backend_base = ensure_api_base(backend)
    click.echo(f"Backend base: {backend_base} (key {mask_value(synth_key)})")

    if train_type == "rl":
        handle_rl(
            cfg_path=cfg_path,
            backend_base=backend_base,
            synth_key=synth_key,
            task_url_override=task_url,
            model_override=model,
            idempotency=idempotency,
            allow_experimental=allow_experimental,
            dry_run=dry_run,
            poll=poll,
            poll_timeout=poll_timeout,
            poll_interval=poll_interval,
        )
    else:
        dataset_override_path = Path(dataset_path).expanduser().resolve() if dataset_path else None
        handle_sft(
            cfg_path=cfg_path,
            backend_base=backend_base,
            synth_key=synth_key,
            dataset_override=dataset_override_path,
            allow_experimental=allow_experimental,
            dry_run=dry_run,
            poll=poll,
            poll_timeout=poll_timeout,
            poll_interval=poll_interval,
            examples_limit=examples_limit,
        )


def register(cli: click.Group) -> None:
    cli.add_command(train_command)


__all__ = ["register", "train_command"]

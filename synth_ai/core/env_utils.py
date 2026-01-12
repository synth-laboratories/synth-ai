import json
import os
from pathlib import Path

import click

from synth_ai.core.user_config import load_user_env


def get_synth_and_env_keys() -> tuple[str, str]:
    load_user_env(override=False)
    synth_api_key = os.environ.get("SYNTH_API_KEY")
    env_api_key = os.environ.get("ENVIRONMENT_API_KEY")
    if not synth_api_key:
        raise RuntimeError(
            "SYNTH_API_KEY not in process environment or ~/.synth-ai config. "
            "Either run synth-ai setup to load automatically or manually set it in your shell."
        )
    if not env_api_key:
        raise RuntimeError(
            "ENVIRONMENT_API_KEY not in process environment or ~/.synth-ai config. "
            "Either run synth-ai setup to load automatically or manually set it in your shell."
        )
    return synth_api_key, env_api_key


def mask_str(input: str, position: int = 3) -> str:
    return input[:position] + "..." + input[-position:] if len(input) > position * 2 else "***"


def ensure_env_var(key: str, expected_value: str) -> None:
    actual_value = os.getenv(key)
    if expected_value != actual_value:
        raise ValueError(f"Expected: {key}={expected_value}\nActual: {key}={actual_value}")


def resolve_env_var(key: str, override_process_env: bool = False) -> str:
    """Resolve an environment variable from available sources.

    Non-interactive: uses first available option or raises error.
    Never prompts - fails hard if value cannot be found.
    """
    env_value = os.getenv(key)
    if env_value is not None and not override_process_env:
        click.echo(f"Using {key}={mask_str(env_value)} from process environment")
        return env_value

    applied = load_user_env(override=override_process_env)
    config_value = applied.get(key)

    if override_process_env and config_value is not None:
        value = config_value
        source = "synth config"
    elif env_value is not None:
        value = env_value
        source = "process environment"
    elif config_value is not None:
        value = config_value
        source = "synth config"
    else:
        raise click.ClickException(
            f"❌ Missing required environment variable: {key}\n\n"
            f"  Options:\n"
            f"  1. Set environment variable: export {key}=<value>\n"
            f"  2. Run `synth-ai setup` to store credentials in ~/.synth-ai\n\n"
            f"  Searched for {key} in:\n"
            f"    - Process environment\n"
            f"    - ~/.synth-ai/*.json config files"
        )

    os.environ[key] = value
    ensure_env_var(key, value)
    click.echo(f"Loaded {key}={mask_str(value)} from {source}")
    return value


def write_env_var_to_json(
    key: str,
    value: str,
    output_file_path: str | Path,
) -> None:
    path = Path(output_file_path).expanduser()
    if path.exists() and not path.is_file():
        raise RuntimeError(f"{path} exists and is not a file")

    data: dict[str, str] = {}

    if path.is_file():
        try:
            with path.open("r", encoding="utf-8") as handle:
                existing = json.load(handle)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON in {path}: {exc}") from exc
        except OSError as exc:
            raise RuntimeError(f"Failed to read {path}: {exc}") from exc

        if not isinstance(existing, dict):
            raise RuntimeError(f"Expected JSON object in {path}")

        for existing_key, existing_value in existing.items():
            if existing_key == key:
                continue
            data[str(existing_key)] = (
                existing_value if isinstance(existing_value, str) else str(existing_value)
            )

    data[key] = value

    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except OSError as exc:
        raise RuntimeError(f"Failed to write {path}: {exc}") from exc

    print(f"Wrote {key}={mask_str(value)} to {path}")

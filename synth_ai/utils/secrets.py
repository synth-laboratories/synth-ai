import click
import json
import os
from pathlib import Path


def mask_str(input: str, position: int = 4) -> str:
    return input[:position] + "..." + input[-position:] if len(input) > position * 2 else "***"


def get_env_file_paths(base_dir: str | Path = '.') -> list[Path]:
    base = Path(base_dir).resolve()
    return [path for path in base.rglob(".env*") if path.is_file()]


def get_synth_config_file_paths() -> list[Path]:
    dir = Path.home() / ".synth-ai"
    if not dir.exists():
        return []
    return [path for path in dir.glob("*.json") if path.is_file()]


def filter_env_files_by_key(key: str, paths: list[Path]) -> list[tuple[Path, str]]:
    matches: list[tuple[Path, str]] = []
    for path in paths:
        try:
            with path.open('r', encoding="utf-8") as file:
                for line in file:
                    stripped = line.strip()
                    if not stripped or stripped.startswith('#'):
                        continue
                    if stripped.split('=', 1)[0].strip() == key:
                        value = stripped \
                            .split('=', 1)[1] \
                            .strip() \
                            .strip('"') \
                            .strip("'")
                        matches.append((path, value))
                        break
        except (OSError, UnicodeDecodeError):
            continue
    return matches


def filter_json_files_by_key(key: str, paths: list[Path]) -> list[tuple[Path, str]]:
    matches: list[tuple[Path, str]] = []
    for path in paths:
        try:
            with path.open('r', encoding="utf-8") as file:
                data = json.load(file)
                if key in data and isinstance(data[key], str):
                    matches.append((path, data[key]))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
    return matches


def convert_abspath_to_relpath(path: Path, base_dir: Path | str = '.') -> str:
    try:
        return str(path.resolve().relative_to(Path(base_dir).resolve()))
    except ValueError:
        return path.name


def load_secret_value(secret_key: str) -> str:
    output = ""
    session_value = os.getenv(secret_key)
    env_file_paths = filter_env_files_by_key(secret_key, get_env_file_paths())
    synth_file_paths = filter_json_files_by_key(secret_key, get_synth_config_file_paths())
    if not session_value and len(env_file_paths) == 0 and len(synth_file_paths) == 0:
        click.echo(f"Failed to find {secret_key}")
        return ""
    
    options: list[tuple[str, str]] = []
    if (session_value):
        options.append((f"(session)  {mask_str(session_value)}", session_value))
    for path, value in env_file_paths:
        label = f"({convert_abspath_to_relpath(path)})  {mask_str(value)}"
        options.append((label, value))
    for path, value in synth_file_paths:
        label = f"({path})  {mask_str(value)}"
        options.append((label, value))
    if len(options) == 1:
        label, output = options[0]
        click.echo(f"Loaded {secret_key}={mask_str(output)} via {label}")
        return output
    
    click.echo(f"\nFound the following options for {secret_key}")
    for i, (label, _) in enumerate(options, start=1):
        click.echo(f" [{i}] {label}")
    click.echo()
    index = click.prompt(
        "Which do you want to load?",
        type=click.IntRange(1, len(options)),
        show_choices=False,
        default=1
    )

    label, output = options[index - 1]
    click.echo(f"Loaded {secret_key}={mask_str(output)}")
    return output

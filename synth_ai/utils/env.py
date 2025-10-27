import json
import os
from pathlib import Path

import click


def _strip_inline_comment(value: str) -> str:
    in_single = False
    in_double = False
    escaped = False
    for idx, char in enumerate(value):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == "'" and not in_double:
            in_single = not in_single
            continue
        if char == '"' and not in_single:
            in_double = not in_double
            continue
        if char == "#" and not in_single and not in_double:
            return value[:idx].rstrip()
    return value.rstrip()


def _parse_env_assignment(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith('#'):
        return None
    if stripped.lower().startswith("export "):
        stripped = stripped[7:].lstrip()
    if '=' not in stripped:
        return None
    key_part, value_part = stripped.split('=', 1)
    key = key_part.strip()
    if not key:
        return None
    value_candidate = _strip_inline_comment(value_part.strip())
    if not value_candidate:
        return key, ""
    if (
        len(value_candidate) >= 2
        and value_candidate[0] in {'"', "'"}
        and value_candidate[-1] == value_candidate[0]
    ):
        value = value_candidate[1:-1]
    else:
        value = value_candidate
    return key, value


def mask_str(input: str, position: int = 3) -> str:
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
                    parsed = _parse_env_assignment(line)
                    if parsed is None:
                        continue
                    parsed_key, value = parsed
                    if parsed_key == key:
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


def resolve_env_var(key: str) -> None:
    env_value = os.getenv(key)
    if env_value is not None:
        click.echo(f"Using ${key}={mask_str(env_value)} from process environment")
        return

    value: str = ""

    env_file_paths = filter_env_files_by_key(key, get_env_file_paths())
    synth_file_paths = filter_json_files_by_key(key, get_synth_config_file_paths())
    if len(env_file_paths) == 0 and len(synth_file_paths) == 0:
        click.echo(f"Failed to find {key}")
        return

    options: list[tuple[str, str]] = []
    for path, value in env_file_paths:
        label = f"({convert_abspath_to_relpath(path)})  {mask_str(value)}"
        options.append((label, value))
    for path, value in synth_file_paths:
        label = f"({path})  {mask_str(value)}"
        options.append((label, value))
    
    click.echo(f"\nFound the following options for {key}")
    for i, (label, _) in enumerate(options, start=1):
        click.echo(f" [{i}] {label}")
    click.echo()

    while True:
        try:
            index = click.prompt(
                "Which do you want to load into process environment?",
                default=1,
                type=int,
                show_choices=False
            )
            if 1 <= index <= len(options):
                _, value = options[index - 1]
                break
            click.echo(f"Invalid selection. Enter a number between 1 and {len(options)}.")
        except click.Abort:
            return
        except Exception:
            click.echo("Invalid input. Please enter a valid number.")
    
    os.environ[key] = value
    click.echo(f"Loaded {key}={mask_str(value)} into process environment")
    return

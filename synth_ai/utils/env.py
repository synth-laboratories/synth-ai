import json
import os
import string
from pathlib import Path

import click

from .paths import get_env_file_paths, get_home_config_file_paths

_ENV_SAFE_CHARS = set(string.ascii_letters + string.digits + "_-./:@+=")


def _format_env_value(value: str) -> str:
    if value == "":
        return '""'
    if all(char in _ENV_SAFE_CHARS for char in value):
        return value
    return json.dumps(value)


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
        if char == '#' and not in_single and not in_double:
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


def _prompt_manual_env_value(key: str) -> str:
    while True:
        value = click.prompt(
            f"Enter value for {key}",
            hide_input=False,
            default="",
            show_default=False,
            type=str,
        ).strip()
        if value:
            return value
        if click.confirm("Save empty value?", default=False):
            return ""
        click.echo("Empty value discarded; enter a value or confirm empty to continue")


def mask_str(input: str, position: int = 3) -> str:
    return input[:position] + "..." + input[-position:] if len(input) > position * 2 else "***"


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


def ensure_env_var(key: str, expected_value: str) -> None:
    actual_value = os.getenv(key)
    if expected_value != actual_value:
        raise ValueError(f"Expected: {key}={expected_value}\nActual: {key}={actual_value}")


def resolve_env_var(
    key: str,
    override_process_env: bool = False
) -> str:
    env_value = os.getenv(key)
    if env_value is not None and not override_process_env:
        click.echo(f"Using {key}={mask_str(env_value)} from process environment")
        return env_value

    value: str = ""

    env_file_paths = filter_env_files_by_key(key, get_env_file_paths())
    synth_file_paths = filter_json_files_by_key(key, get_home_config_file_paths(".synth-ai"))

    options: list[tuple[str, str]] = []
    if env_value is not None:
        if not override_process_env:
            return env_value
        options.append((f"(process environment)  {mask_str(env_value)}", env_value))
    for path, value in env_file_paths:
        resolved_path = path.resolve()
        try:
            rel_path = str(resolved_path.relative_to(Path.cwd()))
        except ValueError:
            rel_path = str(resolved_path)
        label = f"({rel_path})  {mask_str(value)}"
        options.append((label, value))
    for path, value in synth_file_paths:
        label = f"({path})  {mask_str(value)}"
        options.append((label, value))

    if options:
        click.echo(f"\nFound the following options for {key}")
        for i, (label, _) in enumerate(options, start=1):
            click.echo(f" [{i}] {label}")
        click.echo(" [m] Enter value manually")
        click.echo()

        while True:
            try:
                choice = click.prompt(
                    "Select an option",
                    default=1,
                    type=str,
                    show_choices=False,
                ).strip()
            except click.Abort:
                raise
            if choice.lower() == 'm':
                value = _prompt_manual_env_value(key)
                break

            try:
                index = int(choice)
            except ValueError:
                click.echo('Invalid selection. Enter a number or "m".')
                continue

            if 1 <= index <= len(options):
                _, value = options[index - 1]
                break

            click.echo(f"Invalid selection. Enter a number between 1 and {len(options)} or 'm'.")

    else:
        print(f"No value found for {key}")
        value = _prompt_manual_env_value(key)
    
    os.environ[key] = value
    ensure_env_var(key, value)
    print(f"Loaded {key}={mask_str(value)} into process environment")
    return value


def write_env_var_to_dotenv(
    key: str,
    value: str,
    output_file_path: str | Path | None = None,
    print_msg: bool = True,
    mask_msg: bool = True
) -> None:
    path = Path(".env") if output_file_path is None else Path(output_file_path)
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)

    encoded_value = _format_env_value(value)

    lines: list[str] = []
    key_written = False

    if path.is_file():
        try:
            with path.open('r', encoding="utf-8") as handle:
                lines = handle.readlines()
        except OSError as exc:
            raise RuntimeError(f"Failed to read {path}: {exc}") from exc

        for index, line in enumerate(lines):
            parsed = _parse_env_assignment(line)
            if parsed is None or parsed[0] != key:
                continue

            leading_len = len(line) - len(line.lstrip(' \t'))
            leading = line[:leading_len]
            stripped = line.lstrip()
            has_export = stripped.lower().startswith('export ')
            newline = '\n' if line.endswith('\n') else ''
            prefix = 'export ' if has_export else ''
            lines[index] = f"{leading}{prefix}{key}={encoded_value}{newline}"
            key_written = True
            break

    if not key_written:
        if lines and not lines[-1].endswith('\n'):
            lines[-1] = f"{lines[-1]}\n"
        lines.append(f"{key}={encoded_value}\n")

    try:
        with path.open('w', encoding="utf-8") as handle:
            handle.writelines(lines)
    except OSError as exc:
        raise RuntimeError(f"Failed to write {path}: {exc}") from exc

    if print_msg:
        print(f"Wrote {key}={mask_str(value) if mask_msg else value} to {path.resolve()}")


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
            with path.open('r', encoding="utf-8") as handle:
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
        with path.open('w', encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.write('\n')
    except OSError as exc:
        raise RuntimeError(f"Failed to write {path}: {exc}") from exc

    print(f"Wrote {key}={mask_str(value)} to {path}")

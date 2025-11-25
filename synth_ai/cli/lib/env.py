import json
import os
import string
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import click
from dotenv import find_dotenv, load_dotenv

from synth_ai.core.paths import REPO_ROOT, get_env_file_paths, get_home_config_file_paths

_ENV_SAFE_CHARS = set(string.ascii_letters + string.digits + "_-./:@+=")


def get_synth_and_env_keys(env_file: Path | None) -> tuple[str, str]:
    file_synth_api_key = None
    file_env_api_key = None
    if env_file is not None:
        file_synth_api_key = read_env_var_from_file("SYNTH_API_KEY", env_file)
        file_env_api_key = read_env_var_from_file("ENVIRONMENT_API_KEY", env_file)
    env_synth_api_key = os.environ.get("SYNTH_API_KEY")
    env_env_api_key = os.environ.get("ENVIRONMENT_API_KEY")
    synth_api_key = file_synth_api_key or env_synth_api_key
    env_api_key = file_env_api_key or env_env_api_key
    if not synth_api_key:
        raise RuntimeError("SYNTH_API_KEY not in process environment. Either run synth-ai setup to load automatically or manually load to process environment or pass .env via synth-ai deploy --env .env")
    if not env_api_key:
        raise RuntimeError("ENVIRONMENT_API_KEY not in process environment. Either run synth-ai setup to load automatically or manually load to process environment or pass .env via synth-ai deploy --env .env")
    return synth_api_key, env_api_key


def load_env_file(
    env_file: Optional[str] = None,
    required_vars: Optional[Sequence[str]] = None,
) -> Tuple[Optional[str], List[str]]:
    """Load a .env file (if found) and validate that required variables exist."""
    env_path_str = env_file or find_dotenv(usecwd=True)
    env_path: Optional[Path] = None
    if env_path_str:
        candidate = Path(env_path_str).expanduser()
        if candidate.exists():
            env_path = candidate.resolve()

    if env_path:
        load_dotenv(env_path, override=False)
        click.secho(f"✓ Loaded environment from {env_path}", err=True)

    missing: List[str] = []
    if required_vars:
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            click.secho(
                f"⚠️  Missing environment variables: {', '.join(missing)}",
                err=True,
                fg="yellow",
            )
            if env_path:
                click.secho(
                    f"   Check {env_path} for KEY=value formatting (no spaces around '=')",
                    err=True,
                    fg="yellow",
                )

    return (str(env_path) if env_path else None, missing)


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


def read_env_var_from_file(key: str, path: Path) -> str | None:
    try:
        with path.open('r', encoding="utf-8") as f:
            for line in f:
                parsed = _parse_env_assignment(line)
                if parsed is None:
                    continue
                parsed_key, value = parsed
                if parsed_key == key:
                    return value
    except (OSError, UnicodeDecodeError):
        return None
    return None


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
    """Resolve an environment variable from available sources.
    
    Non-interactive: uses first available option or raises error.
    Never prompts - fails hard if value cannot be found.
    """
    env_value = os.getenv(key)
    if env_value is not None and not override_process_env:
        click.echo(f"Using {key}={mask_str(env_value)} from process environment")
        return env_value

    # Get all env files with the key, then prioritize:
    # 1. Repo root .env (not .env.example)
    # 2. Other .env files (excluding .example files)
    # 3. .env.example files as last resort
    all_env_files = filter_env_files_by_key(key, get_env_file_paths())
    synth_file_paths = filter_json_files_by_key(key, get_home_config_file_paths(".synth-ai"))

    # Sort env files by priority: repo root .env first, then exclude .example files
    repo_root_env = None
    regular_env_files = []
    example_env_files = []
    
    repo_root_env_path = REPO_ROOT / ".env"
    for path, value in all_env_files:
        resolved = path.resolve()
        if resolved == repo_root_env_path:
            repo_root_env = (path, value)
        elif ".example" in resolved.name.lower():
            example_env_files.append((path, value))
        else:
            regular_env_files.append((path, value))
    
    # Priority order: process env > repo root .env > regular .env files > example .env files > synth files
    if env_value is not None and override_process_env:
        value = env_value
        source = "process environment"
    elif repo_root_env:
        _, value = repo_root_env
        source = f".env file ({REPO_ROOT / '.env'})"
    elif regular_env_files:
        _, value = regular_env_files[0]
        resolved_path = regular_env_files[0][0].resolve()
        try:
            rel_path = str(resolved_path.relative_to(Path.cwd()))
        except ValueError:
            rel_path = str(resolved_path)
        source = f".env file ({rel_path})"
    elif example_env_files:
        # Only use example files as absolute last resort, and warn
        _, value = example_env_files[0]
        resolved_path = example_env_files[0][0].resolve()
        try:
            rel_path = str(resolved_path.relative_to(Path.cwd()))
        except ValueError:
            rel_path = str(resolved_path)
        click.echo(f"⚠️  Warning: Using example .env file ({rel_path}). Consider using {REPO_ROOT / '.env'} instead.", err=True)
        source = f".env.example file ({rel_path})"
    elif synth_file_paths:
        _, value = synth_file_paths[0]
        source = f"synth config ({synth_file_paths[0][0]})"
    else:
        # No value found - fail hard (no prompting)
        raise click.ClickException(
            f"❌ Missing required environment variable: {key}\n\n"
            f"  Options:\n"
            f"  1. Set environment variable: export {key}=<value>\n"
            f"  2. Add to .env file: {key}=<value>\n"
            f"  3. Use --env-file to specify a .env file path\n"
            f"  4. Add env_file_path to your TOML config: [prompt_learning]\n"
            f"     env_file_path = \"/path/to/.env\"\n\n"
            f"  Searched for {key} in:\n"
            f"    - Process environment\n"
            f"    - .env files in current directory and subdirectories\n"
            f"    - ~/.synth-ai/*.json config files"
        )
    
    os.environ[key] = value
    ensure_env_var(key, value)
    click.echo(f"Loaded {key}={mask_str(value)} from {source}")
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

from __future__ import annotations

import importlib
import os
from collections.abc import Callable, Iterable, MutableMapping
from dataclasses import dataclass
from pathlib import Path

import click

from . import task_app
from .utils import REPO_ROOT, mask_value, read_env_file, write_env_value


def _load_saved_env_path() -> Path | None:
    try:
        module = importlib.import_module("synth_ai.demos.demo_task_apps.core")
        loader = module.load_env_file_path
        saved_path = loader()
        if saved_path:
            return Path(saved_path)
    except Exception:
        return None
    return None


@dataclass(slots=True)
class KeySpec:
    name: str
    description: str
    secret: bool = True
    allow_modal_secret: bool = False
    allow_modal_app: bool = False
    modal_secret_pattern: str | None = None
    optional: bool = False


class EnvResolver:
    def __init__(self, initial_candidates: list[Path]) -> None:
        if not initial_candidates:
            raise click.ClickException("No .env candidates discovered")
        self._candidates = initial_candidates
        self._current = initial_candidates[0]
        self._cache: MutableMapping[Path, dict[str, str]] = {}

    @property
    def current_path(self) -> Path:
        return self._current

    def select_new_env(self) -> None:
        path = prompt_for_env(self._candidates, current=self._current)
        self._current = path
        if path not in self._candidates:
            self._candidates.append(path)

    def get_value(self, key: str) -> str | None:
        cache = self._cache.get(self._current)
        if cache is None:
            cache = read_env_file(self._current)
            self._cache[self._current] = cache
        return cache.get(key)

    def set_value(self, key: str, value: str) -> None:
        cache = self._cache.setdefault(self._current, {})
        cache[key] = value
        write_env_value(self._current, key, value)


def _collect_default_candidates(config_path: Path | None) -> list[Path]:
    candidates: list[Path] = []
    cwd = Path.cwd()

    # Prioritize CWD env files
    cwd_env = cwd / ".env"
    if cwd_env.exists():
        candidates.append(cwd_env.resolve())

    # Search for additional .env files in CWD subdirectories
    for sub in cwd.glob("**/.env"):
        try:
            resolved = sub.resolve()
        except Exception:
            continue
        if resolved in candidates:
            continue
        # avoid nested venv caches
        if any(part in {".venv", "node_modules", "__pycache__"} for part in resolved.parts):
            continue
        if len(candidates) >= 20:
            break
        candidates.append(resolved)

    # Then config path env file
    if config_path:
        cfg_env = config_path.parent / ".env"
        if cfg_env.exists():
            candidates.append(cfg_env.resolve())

    # Then repo env files
    repo_env = REPO_ROOT / ".env"
    if repo_env.exists():
        candidates.append(repo_env.resolve())
    examples_env = REPO_ROOT / "examples" / ".env"
    if examples_env.exists():
        candidates.append(examples_env.resolve())

    # Search shallow depth for additional .env files in examples
    for sub in (REPO_ROOT / "examples").glob("**/.env"):
        try:
            resolved = sub.resolve()
        except Exception:
            continue
        if resolved in candidates:
            continue
        # avoid nested venv caches
        if any(part in {".venv", "node_modules", "__pycache__"} for part in resolved.parts):
            continue
        if len(candidates) >= 20:
            break
        candidates.append(resolved)

    deduped: list[Path] = []
    for path in candidates:
        if path not in deduped:
            deduped.append(path)
    return deduped


def prompt_for_env(candidates: list[Path], *, current: Path | None = None) -> Path:
    options = list(dict.fromkeys(candidates))  # preserve order, dedupe
    click.echo("Select an .env file:")
    for idx, path in enumerate(options, start=1):
        marker = " (current)" if current and path == current else ""
        click.echo(f"  {idx}) {path}{marker}")
    click.echo("  m) Enter path manually")
    click.echo("  0) Abort")

    choice = click.prompt("Choice", default=None)
    if choice is None:
        raise click.ClickException("Selection required")
    choice = choice.strip().lower()
    if choice == "0":
        raise click.ClickException("Aborted by user")
    if choice in {"m", "manual"}:
        manual = click.prompt("Enter path to .env", type=str).strip()
        path = Path(manual).expanduser().resolve()
        if not path.exists():
            raise click.ClickException(f"Env file not found: {path}")
        return path
    try:
        idx = int(choice)
    except ValueError as exc:
        raise click.ClickException("Invalid selection") from exc
    if idx < 1 or idx > len(options):
        raise click.ClickException("Invalid selection")
    return options[idx - 1]


def resolve_env(
    *,
    config_path: Path | None,
    explicit_env_paths: Iterable[str],
    required_keys: list[KeySpec],
) -> tuple[Path, dict[str, str]]:
    provided = [Path(p).expanduser().resolve() for p in explicit_env_paths]
    if provided:
        for path in provided:
            if not path.exists():
                raise click.ClickException(f"Env file not found: {path}")
        resolver = EnvResolver(provided)
    else:
        saved_path = _load_saved_env_path()
        if saved_path and saved_path.exists():
            click.echo(f"Using .env file: {saved_path}")
            resolver = EnvResolver([saved_path])
        else:
            resolver = EnvResolver(_collect_default_candidates(config_path))
            resolver.select_new_env()

    # Preload selected .env keys into process env so downstream lookups succeed
    try:
        current_env_map = read_env_file(resolver.current_path)
        for k in (
            "SYNTH_API_KEY",
            "ENVIRONMENT_API_KEY",
            "dev_environment_api_key",
            "DEV_ENVIRONMENT_API_KEY",
            "TASK_APP_URL",
            "TASK_APP_BASE_URL",
        ):
            v = current_env_map.get(k)
            if v and not os.environ.get(k):
                os.environ[k] = v
    except Exception:
        pass

    resolved: dict[str, str] = {}
    for spec in required_keys:
        value = _resolve_key(resolver, spec)
        if value:
            resolved[spec.name] = value
    return resolver.current_path, resolved


def _resolve_key(resolver: EnvResolver, spec: KeySpec) -> str:
    while True:
        # Priority: existing environment variable
        env_val = os.environ.get(spec.name) or resolver.get_value(spec.name)
        # Allow common aliases in .env to satisfy required keys without extra prompts
        if not env_val and spec.name == "ENVIRONMENT_API_KEY":
            for alias in ("dev_environment_api_key", "DEV_ENVIRONMENT_API_KEY"):
                alt = resolver.get_value(alias)
                if alt:
                    env_val = alt
                    os.environ[spec.name] = alt
                    click.echo(f"Found {spec.name} via {alias}: {mask_value(alt)}")
                    break
        if not env_val and spec.name == "TASK_APP_URL":
            for alias in ("TASK_APP_BASE_URL",):
                alt = resolver.get_value(alias)
                if alt:
                    env_val = alt
                    os.environ[spec.name] = alt
                    click.echo(f"Found {spec.name} via {alias}: {mask_value(alt)}")
                    break
        if env_val:
            click.echo(f"Found {spec.name} in current sources: {mask_value(env_val)}")
            # Automatically use and persist the value (no prompt)
            _maybe_persist(resolver, spec, env_val)
            os.environ[spec.name] = env_val
            return env_val
        options: list[tuple[str, Callable[[], str | None]]] = []

        def _enter_manual() -> str:
            prompt = f"Enter {spec.description}" if spec.description else f"Enter {spec.name}"
            value = click.prompt(prompt, hide_input=spec.secret).strip()
            if not value:
                raise click.ClickException(f"{spec.name} cannot be empty")
            _maybe_persist(resolver, spec, value)
            os.environ[spec.name] = value
            return value

        options.append(("Enter manually", _enter_manual))

        def _pick_env() -> str | None:
            resolver.select_new_env()
            return None

        options.append(("Choose another .env", _pick_env))

        if spec.allow_modal_secret:
            options.append(("Fetch from Modal secret", lambda: _fetch_modal_secret(resolver, spec)))
        if spec.allow_modal_app:
            options.append(("Choose Modal app URL", lambda: _fetch_modal_app(spec)))
        if spec.optional:
            options.append(("Skip", lambda: ""))
        options.append(("Abort", lambda: (_raise_abort(spec))))

        click.echo(f"Resolve {spec.name}:")
        for idx, (label, _) in enumerate(options, start=1):
            click.echo(f"  {idx}) {label}")
        choice = click.prompt("Choice", type=int)
        if choice < 1 or choice > len(options):
            click.echo("Invalid selection; try again")
            continue
        action = options[choice - 1][1]
        result = action()
        if result is None:
            # e.g. user switched env; restart loop
            continue
        return result


def _maybe_persist(resolver: EnvResolver, spec: KeySpec, value: str) -> None:
    # Automatically save (no prompt)
    resolver.set_value(spec.name, value)
    click.echo(f"Saved {spec.name} to {resolver.current_path}")


def _prompt_yes_no(message: str, *, default: bool) -> bool:
    """Prompt the user for a yes/no answer, accepting numeric variants."""

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


def _fetch_modal_secret(resolver: EnvResolver, spec: KeySpec) -> str | None:
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
    _maybe_persist(resolver, spec, value)
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

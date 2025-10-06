from __future__ import annotations

import ast
import contextlib
import functools
import hashlib
import importlib
import importlib.util
import inspect
import os
import signal
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import click
from synth_ai.task.apps import ModalDeploymentConfig, TaskAppConfig, TaskAppEntry, registry
from synth_ai.task.server import run_task_app

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_IGNORE_DIRS = {
    ".git",
    "__pycache__",
    "node_modules",
    "venv",
    ".venv",
    "build",
    "dist",
    ".mypy_cache",
    ".pytest_cache",
}

DEFAULT_SEARCH_RELATIVE = (
    Path("."),
    Path("examples"),
    Path("synth_ai"),
)


@dataclass
class AppChoice:
    app_id: str
    label: str
    path: Path
    source: str
    description: str | None = None
    aliases: tuple[str, ...] = ()
    entry: TaskAppEntry | None = None
    entry_loader: Callable[[], TaskAppEntry] | None = None
    modal_script: Path | None = None
    lineno: int | None = None

    def ensure_entry(self) -> TaskAppEntry:
        if self.entry is not None:
            return self.entry
        if self.entry_loader is None:
            raise click.ClickException(f"Unable to load task app '{self.app_id}' from {self.path}")
        entry = self.entry_loader()
        self.entry = entry
        return entry


def _should_ignore_path(path: Path) -> bool:
    return any(part in DEFAULT_IGNORE_DIRS for part in path.parts)


def _candidate_search_roots() -> list[Path]:
    roots: list[Path] = []
    env_paths = os.environ.get("SYNTH_TASK_APP_SEARCH_PATH")
    if env_paths:
        for chunk in env_paths.split(os.pathsep):
            if chunk:
                roots.append(Path(chunk).expanduser())

    cwd = Path.cwd().resolve()
    roots.append(cwd)

    for rel in DEFAULT_SEARCH_RELATIVE:
        try:
            candidate = (cwd / rel).resolve()
        except Exception:
            continue
        roots.append(candidate)
        if REPO_ROOT not in (None, candidate):
            try:
                repo_candidate = (REPO_ROOT / rel).resolve()
            except Exception:
                repo_candidate = None
            if repo_candidate:
                roots.append(repo_candidate)

    roots.append(REPO_ROOT)

    seen: set[Path] = set()
    ordered: list[Path] = []
    for root in roots:
        try:
            resolved = root.resolve()
        except Exception:
            continue
        if resolved in seen or not resolved.exists():
            continue
        seen.add(resolved)
        ordered.append(resolved)
    return ordered


class _TaskAppConfigVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.matches: list[tuple[str, int]] = []

    def visit_Call(self, node: ast.Call) -> None:  # noqa: D401
        if _is_task_app_config_call(node):
            app_id = _extract_app_id(node)
            if app_id:
                self.matches.append((app_id, getattr(node, "lineno", 0)))
        self.generic_visit(node)


def _is_task_app_config_call(node: ast.Call) -> bool:
    func = node.func
    if isinstance(func, ast.Name) and func.id == "TaskAppConfig":
        return True
    if isinstance(func, ast.Attribute) and func.attr == "TaskAppConfig":
        return True
    return False


def _extract_app_id(node: ast.Call) -> str | None:
    for kw in node.keywords:
        if kw.arg == "app_id" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
            return kw.value.value
    if node.args:
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            return first.value
    return None


class _ModalAppVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.app_aliases: set[str] = set()
        self.modal_aliases: set[str] = set()
        self.matches: list[tuple[str, int]] = []

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: D401
        if node.module == "modal":
            for alias in node.names:
                if alias.name == "App":
                    self.app_aliases.add(alias.asname or alias.name)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:  # noqa: D401
        for alias in node.names:
            if alias.name == "modal":
                self.modal_aliases.add(alias.asname or alias.name)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: D401
        func = node.func
        if isinstance(func, ast.Name) and func.id in self.app_aliases:
            name = _extract_modal_app_name(node)
            if name:
                self.matches.append((name, getattr(node, "lineno", 0)))
        elif isinstance(func, ast.Attribute):
            if isinstance(func.value, ast.Name) and func.value.id in self.modal_aliases and func.attr == "App":
                name = _extract_modal_app_name(node)
                if name:
                    self.matches.append((name, getattr(node, "lineno", 0)))
        self.generic_visit(node)


def _extract_modal_app_name(node: ast.Call) -> str | None:
    for kw in node.keywords:
        if kw.arg in {"name", "app_name"} and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
            return kw.value.value
    if node.args:
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            return first.value
    return None


@functools.lru_cache(maxsize=1)
def _collect_task_app_choices() -> list[AppChoice]:
    choices: list[AppChoice] = []
    with contextlib.suppress(Exception):
        import synth_ai.demos.demo_task_apps  # noqa: F401
    choices.extend(_collect_registered_choices())
    choices.extend(_collect_scanned_task_configs())
    choices.extend(_collect_modal_scripts())

    unique: dict[tuple[str, Path], AppChoice] = {}
    ordered: list[AppChoice] = []
    for choice in choices:
        key = (choice.app_id, choice.path.resolve())
        if key in unique:
            existing = unique[key]
            if existing.source == "registered" and choice.source != "registered":
                continue
            if choice.source == "registered" and existing.source != "registered":
                unique[key] = choice
                idx = ordered.index(existing)
                ordered[idx] = choice
            continue
        unique[key] = choice
        ordered.append(choice)
    return ordered


def _collect_registered_choices() -> list[AppChoice]:
    result: list[AppChoice] = []
    for entry in registry.list():
        module_name = entry.config_factory.__module__
        module = sys.modules.get(module_name)
        if module is None:
            module = importlib.import_module(module_name)
        module_file = getattr(module, "__file__", None)
        path = Path(module_file).resolve() if module_file else REPO_ROOT
        result.append(
            AppChoice(
                app_id=entry.app_id,
                label=entry.app_id,
                path=path,
                source="registered",
                description=entry.description,
                aliases=tuple(entry.aliases),
                entry=entry,
            )
        )
    return result


def _collect_scanned_task_configs() -> list[AppChoice]:
    results: list[AppChoice] = []
    seen: set[tuple[str, Path]] = set()
    for root in _candidate_search_roots():
        if not root.exists() or not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            if not path.is_file():
                continue
            if _should_ignore_path(path):
                continue
            try:
                source = path.read_text(encoding="utf-8")
            except Exception:
                continue
            try:
                tree = ast.parse(source, filename=str(path))
            except SyntaxError:
                continue
            visitor = _TaskAppConfigVisitor()
            visitor.visit(tree)
            for app_id, lineno in visitor.matches:
                key = (app_id, path.resolve())
                if key in seen:
                    continue
                seen.add(key)
                results.append(
                    AppChoice(
                        app_id=app_id,
                        label=app_id,
                        path=path.resolve(),
                        source="discovered",
                        description=f"TaskAppConfig in {path.name} (line {lineno})",
                        entry_loader=lambda p=path.resolve(), a=app_id: _load_entry_from_path(p, a),
                        lineno=lineno,
                    )
                )
    return results


def _collect_modal_scripts() -> list[AppChoice]:
    results: list[AppChoice] = []
    seen: set[tuple[str, Path]] = set()
    for root in _candidate_search_roots():
        if not root.exists() or not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            if not path.is_file():
                continue
            if _should_ignore_path(path):
                continue
            try:
                source = path.read_text(encoding="utf-8")
            except Exception:
                continue
            try:
                tree = ast.parse(source, filename=str(path))
            except SyntaxError:
                continue
            visitor = _ModalAppVisitor()
            visitor.visit(tree)
            for app_name, lineno in visitor.matches:
                key = (app_name, path.resolve())
                if key in seen:
                    continue
                seen.add(key)
                results.append(
                    AppChoice(
                        app_id=app_name,
                        label=app_name,
                        path=path.resolve(),
                        source="modal-script",
                        description=f"Modal App '{app_name}' in {path.name} (line {lineno})",
                        modal_script=path.resolve(),
                        lineno=lineno,
                    )
                )
    return results


def _choice_matches_identifier(choice: AppChoice, identifier: str) -> bool:
    ident = identifier.strip()
    if not ident:
        return False
    if ident == choice.app_id or ident == choice.label:
        return True
    if ident in choice.aliases:
        return True
    return False


def _choice_has_modal_support(choice: AppChoice) -> bool:
    if choice.modal_script:
        return True
    try:
        entry = choice.ensure_entry()
    except click.ClickException:
        return False
    return entry.modal is not None


def _choice_has_local_support(choice: AppChoice) -> bool:
    if choice.modal_script:
        return False
    try:
        choice.ensure_entry()
    except click.ClickException:
        return False
    return True


def _format_choice(choice: AppChoice, index: int | None = None) -> str:
    prefix = f"[{index}] " if index is not None else ""
    rel_path: str
    try:
        rel_path = str(choice.path.relative_to(REPO_ROOT))
    except Exception:
        rel_path = str(choice.path)
    details = choice.description or f"Located at {rel_path}"
    return f"{prefix}{choice.app_id} ({choice.source}) – {details}"


def _prompt_user_for_choice(choices: list[AppChoice]) -> AppChoice:
    click.echo("Select a task app:")
    for idx, choice in enumerate(choices, start=1):
        click.echo(_format_choice(choice, idx))
    response = click.prompt("Enter choice", default="1", type=str).strip() or "1"
    if not response.isdigit():
        raise click.ClickException("Selection must be a number")
    index = int(response)
    if not 1 <= index <= len(choices):
        raise click.ClickException("Selection out of range")
    return choices[index - 1]


def _select_app_choice(app_id: str | None, purpose: str) -> AppChoice:
    choices = _collect_task_app_choices()
    if purpose == "serve":
        filtered = [c for c in choices if not c.modal_script]
    elif purpose in {"deploy", "modal-serve"}:
        filtered = []
        for choice in choices:
            if choice.modal_script or _choice_has_modal_support(choice):
                filtered.append(choice)
    else:
        filtered = choices

    if not filtered:
        raise click.ClickException("No task apps discovered for this command.")

    if app_id:
        matches = [c for c in filtered if _choice_matches_identifier(c, app_id)]
        if not matches:
            available = ", ".join(sorted({c.app_id for c in filtered}))
            raise click.ClickException(f"Task app '{app_id}' not found. Available: {available}")
        if len(matches) == 1:
            return matches[0]
        # Prefer entries with modal support when required
        if purpose in {"deploy", "modal-serve"}:
            modal_matches = [c for c in matches if _choice_has_modal_support(c)]
            if len(modal_matches) == 1:
                return modal_matches[0]
            if modal_matches:
                matches = modal_matches
        return _prompt_user_for_choice(matches)

    if len(filtered) == 1:
        choice = filtered[0]
        click.echo(_format_choice(choice))
        return choice

    return _prompt_user_for_choice(filtered)


def _load_entry_from_path(path: Path, app_id: str) -> TaskAppEntry:
    resolved = path.resolve()
    module_name = f"_synth_task_app_{hashlib.md5(str(resolved).encode(), usedforsecurity=False).hexdigest()}"
    spec = importlib.util.spec_from_file_location(module_name, str(resolved))
    if spec is None or spec.loader is None:
        raise click.ClickException(f"Unable to load Python module from {resolved}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise click.ClickException(f"Failed to import {resolved}: {exc}") from exc

    config_obj: TaskAppConfig | None = None
    factory_callable: Callable[[], TaskAppConfig] | None = None

    for attr_name in dir(module):
        try:
            attr = getattr(module, attr_name)
        except Exception:
            continue
        if isinstance(attr, TaskAppConfig) and attr.app_id == app_id:
            config_obj = attr
            factory_callable = lambda cfg=attr: cfg
            break

    if factory_callable is None:
        for attr_name in dir(module):
            if attr_name.startswith("_"):
                continue
            try:
                attr = getattr(module, attr_name)
            except Exception:
                continue
            if not callable(attr):
                continue
            try:
                sig = inspect.signature(attr)
            except (TypeError, ValueError):
                continue
            has_required = False
            for param in sig.parameters.values():
                if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD) and param.default is inspect._empty:
                    has_required = True
                    break
            if has_required:
                continue
            try:
                result = attr()
            except Exception:
                continue
            if isinstance(result, TaskAppConfig) and result.app_id == app_id:
                def _factory() -> TaskAppConfig:
                    return attr()  # type: ignore[call-arg]
                factory_callable = _factory
                config_obj = result
                break

    if factory_callable is None or config_obj is None:
        raise click.ClickException(
            f"Could not locate TaskAppConfig for '{app_id}' in {resolved}."
        )

    modal_cfg: ModalDeploymentConfig | None = None
    for attr_name in dir(module):
        try:
            attr = getattr(module, attr_name)
        except Exception:
            continue
        if isinstance(attr, ModalDeploymentConfig):
            modal_cfg = attr
            break

    description = inspect.getdoc(module) or f"Discovered task app in {resolved.name}"
    env_files: Iterable[str] = getattr(module, "ENV_FILES", ())  # type: ignore[arg-type]

    entry = TaskAppEntry(
        app_id=app_id,
        description=description,
        config_factory=factory_callable,
        aliases=(),
        env_files=tuple(str(Path(p)) for p in env_files if p),
        modal=modal_cfg,
    )
    return entry


def _resolve_env_paths_for_script(script_path: Path, explicit: Sequence[str]) -> list[Path]:
    if explicit:
        resolved: list[Path] = []
        for candidate in explicit:
            p = Path(candidate).expanduser()
            if not p.exists():
                raise click.ClickException(f"Env file not found: {p}")
            resolved.append(p)
        return resolved

    script_dir = script_path.parent.resolve()
    fallback_order = [
        script_dir / ".env",
        REPO_ROOT / "examples" / "rl" / ".env",
        REPO_ROOT / "examples" / "warming_up_to_rl" / ".env",
        REPO_ROOT / ".env",
    ]
    resolved = [p for p in fallback_order if p.exists()]
    if resolved:
        return resolved
    created = _interactive_create_env(script_dir)
    if created is None:
        raise click.ClickException("Env file required (--env-file) for this task app")
    return [created]


def _run_modal_script(
    script_path: Path,
    modal_cli: str,
    command: str,
    env_paths: Sequence[Path],
    *,
    modal_name: str | None = None,
    dry_run: bool = False,
) -> None:
    modal_path = shutil.which(modal_cli)
    if modal_path is None:
        raise click.ClickException(f"Modal CLI not found (looked for '{modal_cli}')")

    env_paths_list = [Path(p).resolve() for p in env_paths]
    path_strings = [str(p) for p in env_paths_list]
    _load_env_files_into_process(path_strings)
    _ensure_env_values(env_paths_list, script_path.parent)
    _load_env_values(env_paths_list)

    cmd = [modal_path, command, str(script_path)]
    if modal_name:
        cmd.extend(["--name", modal_name])
    if dry_run:
        click.echo("Dry run: " + " ".join(cmd))
        return
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise click.ClickException(f"modal {command} failed with exit code {exc.returncode}") from exc


def _preflight_env_key() -> None:
    try:
        raw_backend = os.environ.get("BACKEND_BASE_URL") or os.environ.get("SYNTH_BASE_URL") or "http://localhost:8000/api"
        backend_base = raw_backend.rstrip('/')
        if not backend_base.endswith('/api'):
            backend_base = backend_base + '/api'
        synth_key = os.environ.get("SYNTH_API_KEY") or ""
        env_api_key = (
            os.environ.get("ENVIRONMENT_API_KEY")
            or os.environ.get("dev_environment_api_key")
            or os.environ.get("DEV_ENVIRONMENT_API_KEY")
            or ""
        )
        if synth_key and env_api_key:
            import base64
            import httpx

            click.echo(f"[preflight] backend={backend_base}")
            with httpx.Client(timeout=15.0, headers={"Authorization": f"Bearer {synth_key}"}) as c:
                click.echo("[preflight] fetching public key…")
                rpk = c.get(f"{backend_base.rstrip('/')}/v1/crypto/public-key")
                pk = (rpk.json() or {}).get("public_key") if rpk.status_code == 200 else None
            if pk:
                try:
                    from nacl.public import PublicKey, SealedBox

                    pub = PublicKey(base64.b64decode(pk, validate=True))
                    sb = SealedBox(pub)
                    ct_b64 = base64.b64encode(sb.encrypt(env_api_key.encode('utf-8'))).decode()
                    payload = {"name": "ENVIRONMENT_API_KEY", "ciphertext_b64": ct_b64}
                    with httpx.Client(timeout=15.0, headers={"Authorization": f"Bearer {synth_key}", "Content-Type": "application/json"}) as c:
                        click.echo("[preflight] upserting env key…")
                        up = c.post(f"{backend_base.rstrip('/')}/v1/env-keys", json=payload)
                        click.echo(f"[preflight] upsert status={up.status_code}")
                        click.echo("[preflight] verifying env key presence…")
                        ver = c.get(f"{backend_base.rstrip('/')}/v1/env-keys/verify")
                        if ver.status_code == 200 and (ver.json() or {}).get("present"):
                            click.echo("✅ ENVIRONMENT_API_KEY upserted and verified in backend")
                        else:
                            click.echo("[WARN] ENVIRONMENT_API_KEY verification failed; proceeding anyway")
                except Exception:
                    click.echo("[WARN] Failed to encrypt/upload ENVIRONMENT_API_KEY; proceeding anyway")
    except Exception:
        click.echo("[WARN] Backend preflight for ENVIRONMENT_API_KEY failed; proceeding anyway")


def _run_modal_with_entry(
    entry: TaskAppEntry,
    modal_cfg: ModalDeploymentConfig,
    modal_cli: str,
    modal_name: str | None,
    env_paths: list[Path],
    command: str,
    *,
    dry_run: bool = False,
) -> None:
    modal_path = shutil.which(modal_cli)
    if modal_path is None:
        raise click.ClickException(f"Modal CLI not found (looked for '{modal_cli}')")

    env_paths_list = [Path(p).resolve() for p in env_paths]
    dotenv_paths = [str(p) for p in env_paths_list]
    _load_env_files_into_process(dotenv_paths)
    fallback_dir = env_paths_list[0].parent if env_paths_list else Path.cwd()
    _ensure_env_values(env_paths_list, fallback_dir)
    _load_env_values(env_paths_list)
    _preflight_env_key()

    script_path = _write_modal_entrypoint(
        entry,
        modal_cfg,
        modal_name,
        dotenv_paths=dotenv_paths,
    )
    cmd = [modal_path, command, str(script_path)]

    if dry_run:
        click.echo("Dry run: " + " ".join(cmd))
        script_path.unlink(missing_ok=True)
        return

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise click.ClickException(f"modal {command} failed with exit code {exc.returncode}") from exc
    finally:
        script_path.unlink(missing_ok=True)




def _load_env_values(paths: list[Path], *, allow_empty: bool = False) -> dict[str, str]:
    values: dict[str, str] = {}
    for p in paths:
        try:
            content = p.read_text(encoding="utf-8")
        except FileNotFoundError:
            continue
        for line in content.splitlines():
            if not line or line.lstrip().startswith('#') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            if key and key not in values:
                values[key.strip()] = value.strip()
    if not allow_empty and not values:
        raise click.ClickException("No environment values found")
    os.environ.update({k: v for k, v in values.items() if k and v})
    return values
def _interactive_create_env(target_dir: Path) -> Path | None:
    env_path = (target_dir / ".env").resolve()
    if env_path.exists():
        existing = _parse_env_file(env_path)
        env_api = (existing.get("ENVIRONMENT_API_KEY") or "").strip()
        if env_api:
            return env_path
        click.echo(f"Existing {env_path} is missing ENVIRONMENT_API_KEY. Let's update it.")
        return _interactive_fill_env(env_path)

    click.echo("No .env found for this task app. Let's create one.")
    return _interactive_fill_env(env_path)


def _parse_env_file(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line or line.lstrip().startswith('#') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            data[key.strip()] = value.strip()
    except FileNotFoundError:
        pass
    return data


def _interactive_fill_env(env_path: Path) -> Path | None:
    existing = _parse_env_file(env_path) if env_path.exists() else {}

    def _prompt(label: str, *, default: str = "", required: bool) -> str | None:
        while True:
            try:
                value = click.prompt(label, default=default, show_default=bool(default) or not required).strip()
            except (click.exceptions.Abort, EOFError, KeyboardInterrupt):
                click.echo("Aborted env creation.")
                return None
            if value or not required:
                return value
            click.echo("This field is required.")

    env_default = existing.get("ENVIRONMENT_API_KEY", "").strip()
    env_api_key = _prompt("ENVIRONMENT_API_KEY", default=env_default, required=True)
    if env_api_key is None:
        return None
    synth_default = existing.get("SYNTH_API_KEY", "").strip()
    openai_default = existing.get("OPENAI_API_KEY", "").strip()
    synth_key = _prompt("SYNTH_API_KEY (optional)", default=synth_default, required=False) or ""
    openai_key = _prompt("OPENAI_API_KEY (optional)", default=openai_default, required=False) or ""

    lines = [
        f"ENVIRONMENT_API_KEY={env_api_key}",
        f"SYNTH_API_KEY={synth_key}",
        f"OPENAI_API_KEY={openai_key}",
    ]
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    click.echo(f"Wrote credentials to {env_path}")
    return env_path


def _ensure_env_values(env_paths: list[Path], fallback_dir: Path) -> None:
    if (os.environ.get("ENVIRONMENT_API_KEY") or "").strip():
        return
    target = env_paths[0] if env_paths else (fallback_dir / ".env").resolve()
    result = _interactive_fill_env(target)
    if result is None:
        raise click.ClickException("ENVIRONMENT_API_KEY required to continue")
    # After generating .env, load it and override any previously-empty values
    _load_env_values([result])
    if not (os.environ.get("ENVIRONMENT_API_KEY") or "").strip():
        raise click.ClickException("Failed to load ENVIRONMENT_API_KEY from generated .env")


def _deploy_entry(
    entry: TaskAppEntry,
    modal_name: str | None,
    dry_run: bool,
    modal_cli: str,
    env_file: Sequence[str],
) -> None:
    modal_cfg = entry.modal
    if modal_cfg is None:
        raise click.ClickException(f"Task app '{entry.app_id}' does not define Modal deployment settings")

    env_paths = _determine_env_files(entry, env_file)
    click.echo('Using env file(s): ' + ', '.join(str(p) for p in env_paths))
    _run_modal_with_entry(entry, modal_cfg, modal_cli, modal_name, env_paths, command="deploy", dry_run=dry_run)


def _modal_serve_entry(
    entry: TaskAppEntry,
    modal_name: str | None,
    modal_cli: str,
    env_file: Sequence[str],
) -> None:
    modal_cfg = entry.modal
    if modal_cfg is None:
        raise click.ClickException(f"Task app '{entry.app_id}' does not define Modal deployment settings")

    env_paths = _determine_env_files(entry, env_file)
    click.echo('Using env file(s): ' + ', '.join(str(p) for p in env_paths))
    _run_modal_with_entry(entry, modal_cfg, modal_cli, modal_name, env_paths, command="serve")

@click.group(
    name='task-app',
    help='Utilities for serving and deploying Synth task apps.'
)
def task_app_group() -> None:
    pass


@task_app_group.command('list')
def list_apps() -> None:
    """List registered task apps."""

    entries = registry.list()
    if not entries:
        click.echo("No task apps registered.")
        return
    for entry in entries:
        aliases = f" (aliases: {', '.join(entry.aliases)})" if entry.aliases else ""
        click.echo(f"- {entry.app_id}{aliases}: {entry.description}")
def _load_env_files_into_process(paths: Sequence[str]) -> None:
    for p in paths:
        try:
            txt = Path(p).expanduser().read_text()
        except Exception:
            continue
        for line in txt.splitlines():
            if not line or line.startswith('#') or '=' not in line:
                continue
            k, v = line.split('=', 1)
            key = k.strip()
            val = v.strip().strip('"').strip("'")
            # Load into process, but allow overriding if the current value is empty
            if key:
                current = os.environ.get(key)
                if current is None or not str(current).strip():
                    os.environ[key] = val



@click.command('serve')
@click.argument('app_id', type=str, required=False)
@click.option('--host', default='0.0.0.0', show_default=True)
@click.option('--port', default=8001, show_default=True, type=int)
@click.option('--env-file', multiple=True, type=click.Path(), help='Extra .env files to load')
@click.option('--reload/--no-reload', 'reload_flag', default=False, help='Enable uvicorn auto-reload')
@click.option('--force/--no-force', 'force', default=False, help='Kill any process already bound to the selected port before starting')
@click.option('--trace', 'trace_dir', type=click.Path(), default=None, help='Enable tracing and write SFT JSONL files to this directory')
@click.option('--trace-db', 'trace_db', type=click.Path(), default=None, help='Override local trace DB path (maps to SQLD_DB_PATH)')
def serve_command(
    app_id: str | None,
    host: str,
    port: int,
    env_file: Sequence[str],
    reload_flag: bool,
    force: bool,
    trace_dir: str | None,
    trace_db: str | None,
) -> None:
    choice = _select_app_choice(app_id, purpose="serve")
    entry = choice.ensure_entry()
    _serve_entry(entry, host, port, env_file, reload_flag, force, trace_dir=trace_dir, trace_db=trace_db)


@task_app_group.command('serve')
@click.argument('app_id', type=str, required=False)
@click.option('--host', default='0.0.0.0', show_default=True)
@click.option('--port', default=8001, show_default=True, type=int)
@click.option('--env-file', multiple=True, type=click.Path(), help='Extra .env files to load')
@click.option('--reload/--no-reload', 'reload_flag', default=False, help='Enable uvicorn auto-reload')
@click.option('--force/--no-force', 'force', default=False, help='Kill any process already bound to the selected port before starting')
@click.option('--trace', 'trace_dir', type=click.Path(), default=None, help='Enable tracing and write SFT JSONL files to this directory')
@click.option('--trace-db', 'trace_db', type=click.Path(), default=None, help='Override local trace DB path (maps to SQLD_DB_PATH)')
def serve_task_group(
    app_id: str | None,
    host: str,
    port: int,
    env_file: Sequence[str],
    reload_flag: bool,
    force: bool,
    trace_dir: str | None,
    trace_db: str | None,
) -> None:
    choice = _select_app_choice(app_id, purpose="serve")
    entry = choice.ensure_entry()
    _serve_entry(entry, host, port, env_file, reload_flag, force, trace_dir=trace_dir, trace_db=trace_db)

def _determine_env_files(entry: TaskAppEntry, user_env_files: Sequence[str]) -> list[Path]:
    resolved: list[Path] = []
    for candidate in user_env_files:
        p = Path(candidate).expanduser()
        if not p.exists():
            raise click.ClickException(f"Env file not found: {p}")
        resolved.append(p)
    if resolved:
        return resolved

    defaults = [Path(path).expanduser() for path in (entry.env_files or []) if Path(path).expanduser().exists()]
    if defaults:
        return defaults

    env_candidates = sorted(REPO_ROOT.glob('**/*.env'))
    if not env_candidates:
        raise click.ClickException('No env file found. Pass --env-file explicitly.')

    click.echo('Select env file to load:')
    for idx, path in enumerate(env_candidates, start=1):
        click.echo(f"  {idx}) {path}")
    choice = click.prompt('Enter choice', type=click.IntRange(1, len(env_candidates)))
    return [env_candidates[choice - 1]]


def _ensure_port_free(port: int, host: str, *, force: bool) -> None:
    import os
    import socket
    import subprocess
    import time

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        in_use = s.connect_ex((host, port)) == 0
    if not in_use:
        return

    try:
        out = subprocess.run(["lsof", "-ti", f"TCP:{port}"], capture_output=True, text=True, check=False)
        pids = [pid for pid in out.stdout.strip().splitlines() if pid]
    except FileNotFoundError:
        pids = []

    if not force:
        message = f"Port {port} appears to be in use"
        if pids:
            message += f" (PIDs: {', '.join(pids)})"
        raise click.ClickException(message)

    for pid in pids:
        try:
            os.kill(int(pid), signal.SIGTERM)
        except Exception as exc:
            raise click.ClickException(f'Failed to terminate PID {pid}: {exc}')

    time.sleep(0.5)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        still_in_use = s.connect_ex((host, port)) == 0

    if still_in_use:
        for pid in pids:
            try:
                os.kill(int(pid), signal.SIGKILL)
            except Exception as exc:
                raise click.ClickException(f'Failed to force terminate PID {pid}: {exc}')
        time.sleep(0.5)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        in_use_after = s.connect_ex((host, port)) == 0
    if in_use_after:
        raise click.ClickException(f'Port {port} is still in use after attempting to terminate processes.')

def _serve_entry(
    entry: TaskAppEntry,
    host: str,
    port: int,
    env_file: Sequence[str],
    reload_flag: bool,
    force: bool,
    *,
    trace_dir: str | None = None,
    trace_db: str | None = None,
) -> None:
    env_files = list(entry.env_files)
    env_files.extend(env_file)

    trace_enabled = trace_dir is not None or trace_db is not None
    if trace_enabled:
        os.environ['TASKAPP_TRACING_ENABLED'] = '1'
        if trace_dir is not None:
            dir_path = Path(trace_dir).expanduser()
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                raise click.ClickException(f"Failed to create trace directory {dir_path}: {exc}") from exc
            os.environ['TASKAPP_SFT_OUTPUT_DIR'] = str(dir_path)
            click.echo(f"Tracing enabled. SFT JSONL will be written to {dir_path}")
        if trace_db is not None:
            db_path = Path(trace_db).expanduser()
            os.environ['SQLD_DB_PATH'] = str(db_path)
            os.environ.pop('TURSO_LOCAL_DB_URL', None)
            click.echo(f"Tracing DB path set to {db_path}")
        from synth_ai.tracing_v3.config import CONFIG as TRACE_CONFIG
        # recompute db_url based on current environment
        new_db_url = os.getenv('TURSO_LOCAL_DB_URL') or TRACE_CONFIG.db_url
        TRACE_CONFIG.db_url = new_db_url
        if new_db_url:
            os.environ['TURSO_LOCAL_DB_URL'] = new_db_url
            click.echo(f"Tracing DB URL resolved to {new_db_url}")
    elif os.getenv('TASKAPP_TRACING_ENABLED'):
        click.echo("Tracing enabled via environment variables")

    _ensure_port_free(port, host, force=force)

    _preflight_env_key()

    run_task_app(
        entry.config_factory,
        host=host,
        port=port,
        reload=reload_flag,
        env_files=env_files,
    )


@task_app_group.command('deploy')
@click.argument("app_id", type=str, required=False)
@click.option("--name", "modal_name", default=None, help="Override Modal app name")
@click.option("--dry-run", is_flag=True, help="Print modal deploy command without executing")
@click.option("--modal-cli", default="modal", help="Path to modal CLI executable")
@click.option('--env-file', multiple=True, type=click.Path(), help='Env file to load into the container (can be repeated)')
def deploy_app(app_id: str | None, modal_name: str | None, dry_run: bool, modal_cli: str, env_file: Sequence[str]) -> None:
    """Deploy a task app to Modal."""

    choice = _select_app_choice(app_id, purpose="deploy")

    if choice.modal_script:
        env_paths = _resolve_env_paths_for_script(choice.modal_script, env_file)
        click.echo('Using env file(s): ' + ', '.join(str(p) for p in env_paths))
        _run_modal_script(choice.modal_script, modal_cli, "deploy", env_paths, modal_name=modal_name, dry_run=dry_run)
        return

    entry = choice.ensure_entry()
    _deploy_entry(entry, modal_name, dry_run, modal_cli, env_file)

@task_app_group.command('modal-serve')
@click.argument('app_id', type=str, required=False)
@click.option('--modal-cli', default='modal', help='Path to modal CLI executable')
@click.option('--name', 'modal_name', default=None, help='Override Modal app name (optional)')
@click.option('--env-file', multiple=True, type=click.Path(), help='Env file to load into the container (can be repeated)')
def modal_serve_app(app_id: str | None, modal_cli: str, modal_name: str | None, env_file: Sequence[str]) -> None:
    choice = _select_app_choice(app_id, purpose="modal-serve")

    if choice.modal_script:
        env_paths = _resolve_env_paths_for_script(choice.modal_script, env_file)
        click.echo('Using env file(s): ' + ', '.join(str(p) for p in env_paths))
        _run_modal_script(choice.modal_script, modal_cli, "serve", env_paths, modal_name=modal_name)
        return

    entry = choice.ensure_entry()
    _modal_serve_entry(entry, modal_name, modal_cli, env_file)


def _write_modal_entrypoint(
    entry: TaskAppEntry,
    modal_cfg: ModalDeploymentConfig,
    override_name: str | None,
    *,
    dotenv_paths: Sequence[str] | None = None,
) -> Path:
    modal_name = override_name or modal_cfg.app_name

    module_name = entry.config_factory.__module__
    dotenv_paths = [str(Path(path)) for path in (dotenv_paths or [])]

    pip_packages = list(modal_cfg.pip_packages)

    local_dirs = [(str(Path(src)), dst) for src, dst in modal_cfg.extra_local_dirs]
    secret_names = list(modal_cfg.secret_names)
    volume_mounts = [(name, mount) for name, mount in modal_cfg.volume_mounts]

    script = f"""from __future__ import annotations

import importlib
import sys
sys.path.insert(0, '/opt/synth_ai_repo')

from modal import App, Image, Secret, Volume, asgi_app

from synth_ai.task.apps import registry
from synth_ai.task.server import create_task_app

ENTRY_ID = {entry.app_id!r}
MODAL_APP_NAME = {modal_name!r}
MODULE_NAME = {module_name!r}
DOTENV_PATHS = {dotenv_paths!r}

image = Image.debian_slim(python_version={modal_cfg.python_version!r})

pip_packages = {pip_packages!r}
if pip_packages:
    image = image.pip_install(*pip_packages)

local_dirs = {local_dirs!r}
for local_src, remote_dst in local_dirs:
    image = image.add_local_dir(local_src, remote_dst)

secrets = {secret_names!r}
secret_objs = [Secret.from_name(name) for name in secrets]

if DOTENV_PATHS:
    secret_objs.extend(Secret.from_dotenv(path) for path in DOTENV_PATHS)

volume_mounts = {volume_mounts!r}
volume_map = {{}}
for vol_name, mount_path in volume_mounts:
    volume_map[mount_path] = Volume.from_name(vol_name, create_if_missing=True)

importlib.import_module(MODULE_NAME)

entry = registry.get(ENTRY_ID)
modal_cfg = entry.modal
if modal_cfg is None:
    raise RuntimeError("Modal configuration missing for task app {entry.app_id}")

app = App(MODAL_APP_NAME)

@app.function(
    image=image,
    timeout={modal_cfg.timeout},
    memory={modal_cfg.memory},
    cpu={modal_cfg.cpu},
    min_containers={modal_cfg.min_containers},
    max_containers={modal_cfg.max_containers},
    secrets=secret_objs,
    volumes=volume_map,
)
@asgi_app()
def fastapi_app():
    config = entry.config_factory()
    return create_task_app(config)
"""

    tmp = tempfile.NamedTemporaryFile("w", suffix=f"_{entry.app_id}_modal.py", delete=False)
    tmp.write(script)
    tmp.flush()
    tmp.close()
    return Path(tmp.name)


def register(cli: click.Group) -> None:
    cli.add_command(serve_command)
    cli.add_command(task_app_group)

from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib
import importlib.util
import inspect
import os
import sys
import types
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import click
from click.exceptions import Abort

try:
    _task_apps_module = importlib.import_module("synth_ai.sdk.task.apps")
    ModalDeploymentConfig = _task_apps_module.ModalDeploymentConfig
    TaskAppConfig = _task_apps_module.TaskAppConfig
    TaskAppEntry = _task_apps_module.TaskAppEntry
    registry = _task_apps_module.registry
except Exception:
    class _UnavailableTaskAppType:  # pragma: no cover - used when optional deps missing
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("Task app registry is unavailable in this environment")

    ModalDeploymentConfig = TaskAppConfig = TaskAppEntry = _UnavailableTaskAppType  # type: ignore[assignment]
    registry: dict[str, Any] = {}

REPO_ROOT = Path(__file__).resolve().parents[3]

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


def _temporary_sys_path(paths: Sequence[Path]):
    """Context manager to prepend entries to sys.path temporarily."""

    @contextlib.contextmanager
    def _manager() -> Iterator[None]:
        added: list[str] = []
        for p in paths:
            try:
                resolved = str(p.resolve())
            except Exception:
                continue
            if resolved in sys.path:
                continue
            sys.path.insert(0, resolved)
            added.append(resolved)
        try:
            yield None
        finally:
            for entry in added:
                with contextlib.suppress(ValueError):
                    sys.path.remove(entry)

    return _manager()


def _possible_module_names(path: Path, module_search_roots: Sequence[Path]) -> list[tuple[str, Path]]:
    candidates: list[tuple[str, Path]] = []
    for root in module_search_roots:
        try:
            resolved_root = root.resolve()
        except Exception:
            continue
        if not resolved_root.exists():
            continue
        with contextlib.suppress(ValueError):
            relative = path.resolve().relative_to(resolved_root)
        stem = relative.with_suffix("")
        parts = list(stem.parts)
        if not parts:
            continue
        module_name = ".".join(parts)
        if module_name:
            candidates.append((module_name, resolved_root))
    return candidates


def _ensure_parent_namespace(module_name: str, search_root: Path) -> None:
    """Ensure namespace packages exist for dotted module names."""

    parts = module_name.split(".")
    for depth in range(1, len(parts)):
        parent_name = ".".join(parts[:depth])
        if parent_name in sys.modules:
            continue
        parent_module = types.ModuleType(parent_name)
        candidate_dir = search_root.joinpath(*parts[:depth])
        try:
            resolved = candidate_dir.resolve()
        except Exception:
            resolved = search_root.resolve()
        parent_module.__path__ = [str(resolved)]  # type: ignore[attr-defined]
        sys.modules[parent_name] = parent_module


def _should_ignore_path(path: Path) -> bool:
    return any(part in DEFAULT_IGNORE_DIRS for part in path.parts)


def _candidate_search_roots() -> list[Path]:
    roots: list[Path] = []

    try:
        demo_module = importlib.import_module("synth_ai.cli.demo_apps.demo_task_apps.core")
    except Exception:
        demo_module = None
    if demo_module:
        load_demo_dir = getattr(demo_module, "load_demo_dir", None)
        if callable(load_demo_dir):
            try:
                demo_dir = load_demo_dir()
            except Exception:
                demo_dir = None
            if demo_dir:
                demo_path = Path(demo_dir)
                if demo_path.exists() and demo_path.is_dir():
                    roots.append(demo_path.resolve())

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


def _eval_config_sort_key(path: Path) -> tuple[int, int, int, str]:
    name = path.name.lower()
    parent_names = {p.name.lower() for p in path.parents}
    in_configs = 0 if "configs" in parent_names else 1
    in_examples = 0 if "examples" in parent_names else 1
    starts_eval = 0 if name.startswith("eval") else 1
    return (in_configs, in_examples, starts_eval, str(path))


def discover_eval_config_paths() -> list[Path]:
    candidates: list[Path] = []
    seen: set[Path] = set()
    for root in _candidate_search_roots():
        if not root.exists() or not root.is_dir():
            continue
        try:
            root = root.resolve()
        except Exception:
            continue
        for path in root.rglob("*.toml"):
            if not path.is_file() or _should_ignore_path(path):
                continue
            name_lower = path.name.lower()
            if "eval" not in name_lower and "evaluation" not in name_lower:
                continue
            try:
                resolved = path.resolve()
            except Exception:
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            candidates.append(resolved)
    candidates.sort(key=_eval_config_sort_key)
    return candidates


class _TaskAppConfigVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.matches: list[tuple[str, int]] = []

    def visit_Call(self, node: ast.Call) -> None:  # noqa: D401
        if _is_task_app_config_call(node):
            app_id = _extract_app_id(node)
            if app_id:
                self.matches.append((app_id, getattr(node, "lineno", 0)))
        elif _is_register_task_app_call(node):
            app_id = _extract_register_app_id(node)
            if app_id:
                self.matches.append((app_id, getattr(node, "lineno", 0)))
        self.generic_visit(node)


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
            if (
                isinstance(func.value, ast.Name)
                and func.value.id in self.modal_aliases
                and func.attr == "App"
            ):
                name = _extract_modal_app_name(node)
                if name:
                    self.matches.append((name, getattr(node, "lineno", 0)))
        self.generic_visit(node)


def _is_task_app_config_call(node: ast.Call) -> bool:
    func = node.func
    return (isinstance(func, ast.Name) and func.id == "TaskAppConfig") or (
        isinstance(func, ast.Attribute) and func.attr == "TaskAppConfig"
    )


def _extract_app_id(node: ast.Call) -> str | None:
    for kw in node.keywords:
        if (
            kw.arg == "app_id"
            and isinstance(kw.value, ast.Constant)
            and isinstance(kw.value.value, str)
        ):
            return kw.value.value
    if node.args:
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            return first.value
    return None


def _is_register_task_app_call(node: ast.Call) -> bool:
    func = node.func
    return (isinstance(func, ast.Name) and func.id == "register_task_app") or (
        isinstance(func, ast.Attribute) and func.attr == "register_task_app"
    )


def _extract_register_app_id(node: ast.Call) -> str | None:
    for kw in node.keywords:
        if kw.arg == "entry" and isinstance(kw.value, ast.Call):
            entry_call = kw.value
            if isinstance(entry_call.func, ast.Name) and entry_call.func.id == "TaskAppEntry":
                for entry_kw in entry_call.keywords:
                    if (
                        entry_kw.arg == "app_id"
                        and isinstance(entry_kw.value, ast.Constant)
                        and isinstance(entry_kw.value.value, str)
                    ):
                        return entry_kw.value.value
    return None


def _extract_modal_app_name(node: ast.Call) -> str | None:
    if node.args:
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            return first.value
    for kw in node.keywords:
        if kw.arg == "name" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
            return kw.value.value
    return None


def _collect_registered_choices() -> list[AppChoice]:
    result: list[AppChoice] = []
    for entry in registry.list():  # type: ignore[attr-defined]
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
        try:
            root_resolved = root.resolve()
        except Exception:
            continue
        for path in root.rglob("*.py"):
            if not path.is_file() or _should_ignore_path(path):
                continue
            try:
                source = path.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=str(path))
            except Exception:
                continue
            visitor = _TaskAppConfigVisitor()
            visitor.visit(tree)
            for app_id, lineno in visitor.matches:
                key = (app_id, path.resolve())
                if key in seen:
                    continue
                seen.add(key)

                def _loader(p: Path = path.resolve(), a: str = app_id, roots: tuple[Path, ...] = (root_resolved,)):
                    return _load_entry_from_path(p, a, module_search_roots=roots)

                results.append(
                    AppChoice(
                        app_id=app_id,
                        label=app_id,
                        path=path.resolve(),
                        source="discovered",
                        description=f"TaskAppConfig in {path.name} (line {lineno})",
                        entry_loader=_loader,
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
            if not path.is_file() or _should_ignore_path(path):
                continue
            try:
                source = path.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=str(path))
            except Exception:
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


def _app_choice_sort_key(choice: AppChoice) -> tuple[int, int, int, int, int, str, str]:
    demo_rank = 1
    try:
        demo_module = importlib.import_module("synth_ai.cli.demo_apps.demo_task_apps.core")
    except Exception:
        demo_module = None
    if demo_module:
        load_demo_dir = getattr(demo_module, "load_demo_dir", None)
        if callable(load_demo_dir):
            try:
                demo_dir = load_demo_dir()
            except Exception:
                demo_dir = None
            if demo_dir:
                demo_path = Path(demo_dir).resolve()
                if choice.path.is_relative_to(demo_path):
                    demo_rank = 0

    cwd_rank = 1
    try:
        cwd = Path.cwd().resolve()
        if choice.path.is_relative_to(cwd):
            try:
                rel_path = choice.path.relative_to(cwd)
                if len(rel_path.parts) <= 2:
                    cwd_rank = 0
            except Exception:
                pass
    except Exception:
        pass

    modal_rank = 1 if choice.modal_script else 0
    name = choice.path.name.lower()
    if name.endswith("_task_app.py") or name.endswith("task_app.py"):
        file_rank = 0
    elif name.endswith("_app.py") or "task_app" in name:
        file_rank = 1
    elif name.endswith(".py"):
        file_rank = 2
    else:
        file_rank = 3

    directory_rank = 0 if choice.path.parent.name.lower() in {"task_app", "task_apps"} else 1

    return (
        demo_rank,
        cwd_rank,
        modal_rank,
        file_rank,
        directory_rank,
        choice.app_id,
        str(choice.path),
    )


def _choice_matches_identifier(choice: AppChoice, identifier: str) -> bool:
    ident = identifier.strip()
    if not ident:
        return False
    return ident == choice.app_id or ident == choice.label or ident in choice.aliases


def _choice_has_modal_support(choice: AppChoice) -> bool:
    if choice.modal_script:
        return True
    try:
        entry = choice.ensure_entry()
    except click.ClickException:
        return _has_modal_support_in_file(choice.path)
    return entry.modal is not None  # type: ignore[attr-defined]


def _choice_has_local_support(choice: AppChoice) -> bool:
    if choice.modal_script:
        return False
    try:
        choice.ensure_entry()
    except click.ClickException:
        return False
    return True


def _has_modal_support_in_file(path: Path) -> bool:
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except Exception:
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _is_register_task_app_call(node):
            for kw in node.keywords:
                if kw.arg == "entry" and isinstance(kw.value, ast.Call):
                    entry_call = kw.value
                    if (
                        isinstance(entry_call.func, ast.Name)
                        and entry_call.func.id == "TaskAppEntry"
                    ):
                        for entry_kw in entry_call.keywords:
                            if entry_kw.arg == "modal" and isinstance(entry_kw.value, ast.Call):
                                modal_call = entry_kw.value
                                if (
                                    isinstance(modal_call.func, ast.Name)
                                    and modal_call.func.id == "ModalDeploymentConfig"
                                ):
                                    return True
    return False


def _extract_modal_config_from_file(path: Path) -> ModalDeploymentConfig | None:
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except Exception:
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _is_register_task_app_call(node):
            for kw in node.keywords:
                if kw.arg == "entry" and isinstance(kw.value, ast.Call):
                    entry_call = kw.value
                    if (
                        isinstance(entry_call.func, ast.Name)
                        and entry_call.func.id == "TaskAppEntry"
                    ):
                        for entry_kw in entry_call.keywords:
                            if entry_kw.arg == "modal" and isinstance(entry_kw.value, ast.Call):
                                modal_call = entry_kw.value
                                if (
                                    isinstance(modal_call.func, ast.Name)
                                    and modal_call.func.id == "ModalDeploymentConfig"
                                ):
                                    return _build_modal_config_from_ast(modal_call)
    return None


def _build_modal_config_from_ast(modal_call: ast.Call) -> ModalDeploymentConfig | None:
    try:
        kwargs = {}
        for kw in modal_call.keywords:
            if kw.arg and isinstance(kw.value, ast.Constant):
                kwargs[kw.arg] = kw.value.value
            elif kw.arg == "pip_packages" and isinstance(kw.value, ast.List | ast.Tuple):
                packages = []
                for elt in kw.value.elts:
                    if isinstance(elt, ast.Constant):
                        packages.append(elt.value)
                kwargs[kw.arg] = tuple(packages)
            elif kw.arg == "extra_local_dirs" and isinstance(kw.value, ast.List | ast.Tuple):
                dirs = []
                for elt in kw.value.elts:
                    if isinstance(elt, ast.List | ast.Tuple) and len(elt.elts) == 2:
                        src = elt.elts[0].value if isinstance(elt.elts[0], ast.Constant) else None
                        dst = elt.elts[1].value if isinstance(elt.elts[1], ast.Constant) else None
                        if src and dst:
                            dirs.append((src, dst))
                kwargs[kw.arg] = tuple(dirs)
            elif kw.arg == "secret_names" and isinstance(kw.value, ast.List | ast.Tuple):
                secrets = []
                for elt in kw.value.elts:
                    if isinstance(elt, ast.Constant):
                        secrets.append(elt.value)
                kwargs[kw.arg] = tuple(secrets)
            elif kw.arg == "volume_mounts" and isinstance(kw.value, ast.List | ast.Tuple):
                mounts = []
                for elt in kw.value.elts:
                    if isinstance(elt, ast.List | ast.Tuple) and len(elt.elts) == 2:
                        name = elt.elts[0].value if isinstance(elt.elts[0], ast.Constant) else None
                        mount = elt.elts[1].value if isinstance(elt.elts[1], ast.Constant) else None
                        if name and mount:
                            mounts.append((name, mount))
                kwargs[kw.arg] = tuple(mounts)
        return ModalDeploymentConfig(**kwargs)
    except Exception:
        return None


def _format_choice(choice: AppChoice, index: int | None = None) -> str:
    prefix = f"[{index}] " if index is not None else ""
    try:
        mtime = choice.path.stat().st_mtime
        modified_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        details = f"Modified: {modified_str}"
    except Exception:
        details = choice.description or "No timestamp available"
    return f"{prefix}{choice.app_id} ({choice.source}) – {details}"


def _prompt_user_for_choice(choices: list[AppChoice]) -> AppChoice:
    click.echo("Select a task app:")
    for idx, choice in enumerate(choices, start=1):
        click.echo(_format_choice(choice, idx))
    try:
        response = click.prompt("Enter choice", default="1", type=str).strip() or "1"
    except (Abort, EOFError, KeyboardInterrupt) as exc:
        raise click.ClickException("Task app selection cancelled by user") from exc
    if not response.isdigit():
        raise click.ClickException("Selection must be a number")
    index = int(response)
    if not 1 <= index <= len(choices):
        raise click.ClickException("Selection out of range")
    return choices[index - 1]


def _collect_task_app_choices() -> list[AppChoice]:
    registry.clear()
    choices: list[AppChoice] = []
    with contextlib.suppress(Exception):
        importlib.import_module("synth_ai.cli.demo_apps.demo_task_apps")
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
    ordered.sort(key=_app_choice_sort_key)
    return ordered


def select_app_choice(app_id: str | None, purpose: str) -> AppChoice:
    choices = _collect_task_app_choices()
    if purpose in {"serve", "eval"}:
        filtered = [c for c in choices if _choice_has_local_support(c)]
    elif purpose in {"deploy", "modal-serve"}:
        filtered = [c for c in choices if _choice_has_modal_support(c)]
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
        if purpose in {"deploy", "modal-serve"}:
            modal_matches = [c for c in matches if _choice_has_modal_support(c)]
            if len(modal_matches) == 1:
                return modal_matches[0]
            if modal_matches:
                matches = modal_matches
        filtered = matches

    filtered.sort(key=_app_choice_sort_key)
    if len(filtered) == 1:
        choice = filtered[0]
        click.echo(_format_choice(choice))
        return choice
    return _prompt_user_for_choice(filtered)


def _import_task_app_module(
    resolved: Path,
    module_name: str,
    *,
    namespace_root: Path | None,
    sys_path_roots: Sequence[Path],
    ensure_namespace: bool = True,
) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, str(resolved))
    if spec is None or spec.loader is None:
        raise click.ClickException(f"Unable to load Python module from {resolved}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    with _temporary_sys_path(sys_path_roots):
        if ensure_namespace and namespace_root is not None and "." in module_name:
            _ensure_parent_namespace(module_name, namespace_root)

        registry.clear()

        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise

    return module


def _load_entry_from_path(
    path: Path, app_id: str, module_search_roots: Sequence[Path] | None = None
) -> TaskAppEntry:
    resolved = path.resolve()
    search_roots: list[Path] = []
    seen_roots: set[Path] = set()

    def _append_root(candidate: Path) -> None:
        try:
            resolved_root = candidate.resolve()
        except Exception:
            return
        if resolved_root in seen_roots:
            return
        seen_roots.add(resolved_root)
        search_roots.append(resolved_root)

    for root in module_search_roots or []:
        _append_root(root)
    _append_root(resolved.parent)
    _append_root(REPO_ROOT)

    last_error: Exception | None = None
    module: types.ModuleType | None = None

    for module_name, namespace_root in _possible_module_names(resolved, search_roots):
        try:
            module = _import_task_app_module(
                resolved,
                module_name,
                namespace_root=namespace_root,
                sys_path_roots=search_roots,
                ensure_namespace=True,
            )
            break
        except Exception as exc:
            last_error = exc
            continue

    if module is None:
        hashed_name = f"_synth_task_app_{hashlib.md5(str(resolved).encode(), usedforsecurity=False).hexdigest()}"
        try:
            module = _import_task_app_module(
                resolved,
                hashed_name,
                namespace_root=None,
                sys_path_roots=search_roots,
                ensure_namespace=False,
            )
        except Exception as exc:
            detail = last_error or exc
            raise click.ClickException(f"Failed to import {resolved}: {detail}") from detail

    config_obj: TaskAppConfig | None = None
    factory_callable: Callable[[], TaskAppConfig] | None = None

    for attr_name in dir(module):
        try:
            attr = getattr(module, attr_name)
        except Exception:
            continue
        if isinstance(attr, TaskAppConfig) and attr.app_id == app_id:

            def _return_config(cfg: TaskAppConfig = attr) -> TaskAppConfig:
                return cfg

            factory_callable = _return_config
            config_obj = attr
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
            has_required = any(
                param.kind
                in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                and param.default is inspect._empty
                for param in sig.parameters.values()
            )
            if has_required:
                continue
            try:
                result = attr()
            except Exception:
                continue
            if isinstance(result, TaskAppConfig) and result.app_id == app_id:

                def _factory_noargs(func: Callable[[], TaskAppConfig] = attr) -> TaskAppConfig:
                    return func()

                factory_callable = _factory_noargs
                config_obj = result
                break

    if factory_callable is None or config_obj is None:
        try:
            entry = registry.get(app_id)
            if entry is None:
                raise KeyError(f"TaskApp '{app_id}' not found in registry")
            return entry
        except KeyError as exc:
            raise click.ClickException(
                f"Could not locate TaskAppConfig for '{app_id}' in {resolved}."
            ) from exc

    modal_cfg: ModalDeploymentConfig | None = None
    for attr_name in dir(module):
        try:
            attr = getattr(module, attr_name)
        except Exception:
            continue
        if isinstance(attr, ModalDeploymentConfig):
            modal_cfg = attr
            break

    if modal_cfg is None:
        modal_cfg = _extract_modal_config_from_file(resolved)

    env_files: Iterable[str] = getattr(module, "ENV_FILES", ())  # type: ignore[arg-type]

    return TaskAppEntry(
        app_id=app_id,
        description=inspect.getdoc(module) or f"Discovered task app in {resolved.name}",
        config_factory=factory_callable,
        aliases=(),
        env_files=tuple(str(Path(p)) for p in env_files if p),
        modal=modal_cfg,
    )


__all__ = [
    "AppChoice",
    "discover_eval_config_paths",
    "select_app_choice",
]

from __future__ import annotations

import ast
import contextlib
import functools
import hashlib
import importlib
import importlib.util
import inspect
import os
import json
import signal
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
import types
from typing import Any, Callable, Iterable, Sequence, Iterator, cast
try:  # Python 3.11+
    import tomllib as _toml
except Exception:  # pragma: no cover - fallback
    _toml = None  # type: ignore
import uuid

import click
from synth_ai.task.apps import ModalDeploymentConfig, TaskAppConfig, TaskAppEntry, registry
from synth_ai.task.server import run_task_app, create_task_app

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
    """Return potential module names based on candidate roots."""

    candidates: list[tuple[str, Path]] = []
    for root in module_search_roots:
        try:
            resolved_root = root.resolve()
        except Exception:
            continue
        if not resolved_root.exists() or not path.is_relative_to(resolved_root):
            continue
        relative = path.relative_to(resolved_root)
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

    parts = module_name.split('.')
    for depth in range(1, len(parts)):
        parent_name = '.'.join(parts[:depth])
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
    """Only search for task apps in the current working directory and subdirectories."""
    roots: list[Path] = []
    
    # Allow explicit search paths via environment variable
    env_paths = os.environ.get("SYNTH_TASK_APP_SEARCH_PATH")
    if env_paths:
        for chunk in env_paths.split(os.pathsep):
            if chunk:
                roots.append(Path(chunk).expanduser())

    # Always include current working directory
    cwd = Path.cwd().resolve()
    roots.append(cwd)

    for rel in DEFAULT_SEARCH_RELATIVE:
        try:
            candidate = (cwd / rel).resolve()
        except Exception:
            continue
        roots.append(candidate)

    # Remove duplicates while preserving order
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


def _discover_eval_config_paths() -> list[Path]:
    """Find candidate eval TOML files near the current working directory."""

    candidates: list[Path] = []
    seen: set[Path] = set()
    search_roots = _candidate_search_roots()
    for root in search_roots:
        if not root.exists() or not root.is_dir():
            continue
        try:
            root_resolved = root.resolve()
        except Exception:
            continue
        for path in root.rglob("*.toml"):
            if not path.is_file():
                continue
            if _should_ignore_path(path):
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


def _is_register_task_app_call(node: ast.Call) -> bool:
    func = node.func
    if isinstance(func, ast.Name) and func.id == "register_task_app":
        return True
    if isinstance(func, ast.Attribute) and func.attr == "register_task_app":
        return True
    return False


def _extract_register_app_id(node: ast.Call) -> str | None:
    # Look for entry=TaskAppEntry(app_id="...", ...)
    for kw in node.keywords:
        if kw.arg == "entry" and isinstance(kw.value, ast.Call):
            entry_call = kw.value
            if isinstance(entry_call.func, ast.Name) and entry_call.func.id == "TaskAppEntry":
                for entry_kw in entry_call.keywords:
                    if entry_kw.arg == "app_id" and isinstance(entry_kw.value, ast.Constant) and isinstance(entry_kw.value.value, str):
                        return entry_kw.value.value
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


def _collect_task_app_choices() -> list[AppChoice]:
    # Clear registry to avoid duplicate registration errors
    registry.clear()
    
    choices: list[AppChoice] = []
    with contextlib.suppress(Exception):
        import synth_ai.demos.demo_task_apps  # noqa: F401
    # Only use discovered task apps, not registered ones (since we moved them to examples)
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
        try:
            root_resolved = root.resolve()
        except Exception:
            continue
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
                        entry_loader=lambda p=path.resolve(), a=app_id, roots=(root_resolved,): _load_entry_from_path(p, a, module_search_roots=roots),
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


def _app_choice_sort_key(choice: AppChoice) -> tuple[int, int, int, str, str]:
    """Ranking heuristic so wrapper-style task apps surface first."""

    modal_rank = 1 if choice.modal_script else 0

    name = choice.path.name.lower()
    file_rank = 3
    if name.endswith("_task_app.py") or name.endswith("task_app.py"):
        file_rank = 0
    elif name.endswith("_app.py") or "task_app" in name:
        file_rank = 1
    elif name.endswith(".py"):
        file_rank = 2

    directory_rank = 0 if choice.path.parent.name.lower() in {"task_app", "task_apps"} else 1

    return (modal_rank, file_rank, directory_rank, choice.app_id, str(choice.path))


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
        # If we can't load the entry, try to detect Modal support via AST parsing
        return _has_modal_support_in_file(choice.path)
    return entry.modal is not None


def _has_modal_support_in_file(path: Path) -> bool:
    """Detect if a file has Modal deployment support by parsing the AST."""
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        
        # Look for ModalDeploymentConfig in register_task_app calls
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if _is_register_task_app_call(node):
                    # Check if the entry has modal=ModalDeploymentConfig(...)
                    for kw in node.keywords:
                        if kw.arg == "entry" and isinstance(kw.value, ast.Call):
                            entry_call = kw.value
                            if isinstance(entry_call.func, ast.Name) and entry_call.func.id == "TaskAppEntry":
                                for entry_kw in entry_call.keywords:
                                    if entry_kw.arg == "modal" and isinstance(entry_kw.value, ast.Call):
                                        modal_call = entry_kw.value
                                        if isinstance(modal_call.func, ast.Name) and modal_call.func.id == "ModalDeploymentConfig":
                                            return True
    except Exception:
        pass
    return False


def _extract_modal_config_from_file(path: Path) -> ModalDeploymentConfig | None:
    """Extract ModalDeploymentConfig from a file by parsing the AST."""
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        
        # Look for ModalDeploymentConfig in register_task_app calls
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if _is_register_task_app_call(node):
                    # Check if the entry has modal=ModalDeploymentConfig(...)
                    for kw in node.keywords:
                        if kw.arg == "entry" and isinstance(kw.value, ast.Call):
                            entry_call = kw.value
                            if isinstance(entry_call.func, ast.Name) and entry_call.func.id == "TaskAppEntry":
                                for entry_kw in entry_call.keywords:
                                    if entry_kw.arg == "modal" and isinstance(entry_kw.value, ast.Call):
                                        modal_call = entry_kw.value
                                        if isinstance(modal_call.func, ast.Name) and modal_call.func.id == "ModalDeploymentConfig":
                                            # Extract the arguments to ModalDeploymentConfig
                                            return _build_modal_config_from_ast(modal_call)
    except Exception:
        pass
    return None


def _build_modal_config_from_ast(modal_call: ast.Call) -> ModalDeploymentConfig | None:
    """Build a ModalDeploymentConfig from an AST Call node."""
    try:
        # Extract keyword arguments
        kwargs = {}
        for kw in modal_call.keywords:
            if kw.arg and isinstance(kw.value, ast.Constant):
                kwargs[kw.arg] = kw.value.value
            elif kw.arg == "pip_packages" and isinstance(kw.value, (ast.List, ast.Tuple)):
                # Handle pip_packages list/tuple
                packages = []
                for elt in kw.value.elts:
                    if isinstance(elt, ast.Constant):
                        packages.append(elt.value)
                kwargs[kw.arg] = tuple(packages)
            elif kw.arg == "extra_local_dirs" and isinstance(kw.value, (ast.List, ast.Tuple)):
                # Handle extra_local_dirs list/tuple of tuples
                dirs = []
                for elt in kw.value.elts:
                    if isinstance(elt, (ast.List, ast.Tuple)) and len(elt.elts) == 2:
                        src = elt.elts[0].value if isinstance(elt.elts[0], ast.Constant) else None
                        dst = elt.elts[1].value if isinstance(elt.elts[1], ast.Constant) else None
                        if src and dst:
                            dirs.append((src, dst))
                kwargs[kw.arg] = tuple(dirs)
            elif kw.arg == "secret_names" and isinstance(kw.value, (ast.List, ast.Tuple)):
                # Handle secret_names list/tuple
                secrets = []
                for elt in kw.value.elts:
                    if isinstance(elt, ast.Constant):
                        secrets.append(elt.value)
                kwargs[kw.arg] = tuple(secrets)
            elif kw.arg == "volume_mounts" and isinstance(kw.value, (ast.List, ast.Tuple)):
                # Handle volume_mounts list/tuple of tuples
                mounts = []
                for elt in kw.value.elts:
                    if isinstance(elt, (ast.List, ast.Tuple)) and len(elt.elts) == 2:
                        name = elt.elts[0].value if isinstance(elt.elts[0], ast.Constant) else None
                        mount = elt.elts[1].value if isinstance(elt.elts[1], ast.Constant) else None
                        if name and mount:
                            mounts.append((name, mount))
                kwargs[kw.arg] = tuple(mounts)
        
        # Create ModalDeploymentConfig with extracted arguments
        from synth_ai.task.apps import ModalDeploymentConfig
        return ModalDeploymentConfig(**kwargs)
    except Exception:
        return None


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
    if purpose in {"serve", "eval"}:
        filtered = [c for c in choices if not c.modal_script]
    elif purpose in {"deploy", "modal-serve"}:
        filtered = []
        for choice in choices:
            if choice.modal_script or _choice_has_modal_support(choice):
                filtered.append(choice)
    else:
        filtered = choices

    filtered.sort(key=_app_choice_sort_key)

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
        if ensure_namespace and namespace_root is not None and '.' in module_name:
            _ensure_parent_namespace(module_name, namespace_root)

        # Clear registry before importing to avoid duplicate registration errors
        registry.clear()

        try:
            spec.loader.exec_module(module)
        except Exception:
            # Remove partially-imported module to avoid reuse
            sys.modules.pop(module_name, None)
            raise

    return module


def _load_entry_from_path(path: Path, app_id: str, module_search_roots: Sequence[Path] | None = None) -> TaskAppEntry:
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
        except Exception as exc:  # pragma: no cover - best-effort fallbacks
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
        except Exception as exc:  # pragma: no cover - propagate meaningful error
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
                # Bind attr to a local and close over it without exposing parameters
                _bound_func: Callable[[], TaskAppConfig] = cast(Callable[[], TaskAppConfig], attr)  # type: ignore[assignment]
                def _factory_noargs() -> TaskAppConfig:
                    return _bound_func()
                factory_callable = _factory_noargs
                config_obj = result
                break

    # If no TaskAppConfig found directly, check if it was registered via register_task_app
    if factory_callable is None or config_obj is None:
        try:
            # Check if the app was registered in the registry
            entry = registry.get(app_id)
            return entry
        except KeyError:
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
    
    # If no ModalDeploymentConfig found, try to detect it via AST parsing
    if modal_cfg is None:
        modal_cfg = _extract_modal_config_from_file(resolved)

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

    # Always prompt for env file selection instead of auto-loading defaults
    script_dir = script_path.parent.resolve()
    cwd = Path.cwd()
    
    # Look for env files in current working directory first, then repo root
    env_candidates = []
    
    # Add CWD env files first (prioritized)
    cwd_env_files = sorted(cwd.glob('**/*.env'))
    env_candidates.extend(cwd_env_files)
    
    # Add repo root env files
    repo_env_files = sorted(REPO_ROOT.glob('**/*.env'))
    # Avoid duplicates
    for repo_file in repo_env_files:
        if repo_file not in env_candidates:
            env_candidates.append(repo_file)
    
    if not env_candidates:
        created = _interactive_create_env(script_dir)
        if created is None:
            raise click.ClickException("Env file required (--env-file) for this task app")
        return [created]

    click.echo('Select env file to load:')
    for idx, path in enumerate(env_candidates, start=1):
        click.echo(f"  {idx}) {path}")
    choice = click.prompt('Enter choice', type=click.IntRange(1, len(env_candidates)))
    return [env_candidates[choice - 1]]


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


def _preflight_env_key(crash_on_failure: bool = False) -> None:
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
                            # Show first and last 5 chars of the API key for verification
                            key_preview = f"{env_api_key[:5]}...{env_api_key[-5:]}" if len(env_api_key) > 10 else env_api_key
                            click.echo(f"✅ ENVIRONMENT_API_KEY upserted and verified in backend ({key_preview})")
                        else:
                            error_msg = "ENVIRONMENT_API_KEY verification failed"
                            if crash_on_failure:
                                raise click.ClickException(f"[CRITICAL] {error_msg}")
                            click.echo(f"[WARN] {error_msg}; proceeding anyway")
                except Exception as e:
                    error_msg = f"Failed to encrypt/upload ENVIRONMENT_API_KEY: {e}"
                    if crash_on_failure:
                        raise click.ClickException(f"[CRITICAL] {error_msg}")
                    click.echo(f"[WARN] {error_msg}; proceeding anyway")
    except Exception as e:
        error_msg = f"Backend preflight for ENVIRONMENT_API_KEY failed: {e}"
        if crash_on_failure:
            raise click.ClickException(f"[CRITICAL] {error_msg}")
        click.echo(f"[WARN] {error_msg}; proceeding anyway")


def _run_modal_with_entry(
    entry: TaskAppEntry,
    modal_cfg: ModalDeploymentConfig,
    modal_cli: str,
    modal_name: str | None,
    env_paths: list[Path],
    command: str,
    *,
    dry_run: bool = False,
    original_path: Path | None = None,
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
    _preflight_env_key(crash_on_failure=True)

    script_path = _write_modal_entrypoint(
        entry,
        modal_cfg,
        modal_name,
        dotenv_paths=dotenv_paths,
        original_path=original_path,
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
    original_path: Path | None = None,
) -> None:
    modal_cfg = entry.modal
    if modal_cfg is None:
        raise click.ClickException(f"Task app '{entry.app_id}' does not define Modal deployment settings")

    env_paths = _determine_env_files(entry, env_file)
    click.echo('Using env file(s): ' + ', '.join(str(p) for p in env_paths))
    _run_modal_with_entry(entry, modal_cfg, modal_cli, modal_name, env_paths, command="deploy", dry_run=dry_run, original_path=original_path)


def _modal_serve_entry(
    entry: TaskAppEntry,
    modal_name: str | None,
    modal_cli: str,
    env_file: Sequence[str],
    original_path: Path | None = None,
) -> None:
    modal_cfg = entry.modal
    if modal_cfg is None:
        raise click.ClickException(f"Task app '{entry.app_id}' does not define Modal deployment settings")

    env_paths = _determine_env_files(entry, env_file)
    click.echo('Using env file(s): ' + ', '.join(str(p) for p in env_paths))
    _run_modal_with_entry(entry, modal_cfg, modal_cli, modal_name, env_paths, command="serve", original_path=original_path)

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

    # Always prompt for env file selection instead of auto-loading defaults
    # Look for env files in current working directory first, then repo root
    cwd = Path.cwd()
    env_candidates = []
    
    # Add CWD env files first (prioritized)
    cwd_env_files = sorted(cwd.glob('**/*.env'))
    env_candidates.extend(cwd_env_files)
    
    # Add repo root env files
    repo_env_files = sorted(REPO_ROOT.glob('**/*.env'))
    # Avoid duplicates
    for repo_file in repo_env_files:
        if repo_file not in env_candidates:
            env_candidates.append(repo_file)
    
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
    _deploy_entry(entry, modal_name, dry_run, modal_cli, env_file, original_path=choice.path)

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
    _modal_serve_entry(entry, modal_name, modal_cli, env_file, original_path=choice.path)


def _write_modal_entrypoint(
    entry: TaskAppEntry,
    modal_cfg: ModalDeploymentConfig,
    override_name: str | None,
    *,
    dotenv_paths: Sequence[str] | None = None,
    original_path: Path | None = None,
) -> Path:
    modal_name = override_name or modal_cfg.app_name

    # For dynamically discovered apps, import the module by its package path
    # Compute the module name relative to the mounted repo root (/opt/synth_ai_repo)
    remote_file_str: str | None = None
    if original_path:
        try:
            # Build lookup of local->remote mounts
            mount_map: list[tuple[Path, Path]] = [
                (Path(local).resolve(), Path(remote)) for (local, remote) in modal_cfg.extra_local_dirs
            ]
            orig = Path(original_path).resolve()
            for local_src, remote_dst in mount_map:
                with contextlib.suppress(Exception):
                    if orig.is_relative_to(local_src):  # py311+
                        remote_file_str = str((remote_dst / orig.relative_to(local_src)).resolve())
                        break
                try:
                    rel = orig.relative_to(local_src)
                    remote_file_str = str((remote_dst / rel).resolve())
                    break
                except Exception:
                    pass
        except Exception:
            remote_file_str = None
    module_name = entry.config_factory.__module__

    # Prefer a guaranteed mount for the discovered file to avoid package import issues
    guaranteed_file_str: str | None = None
    if original_path:
        guaranteed_file_str = str((Path("/opt/synth_ai_repo/__local_task_app__") / Path(original_path).stem).with_suffix('.py'))
    
    dotenv_paths = [str(Path(path)) for path in (dotenv_paths or [])]

    pip_packages = list(modal_cfg.pip_packages)
    # Ensure synth-ai (matching host version if available) is installed in the container
    synth_pkg = "synth-ai"
    try:
        import synth_ai as _host_synth
        host_ver = getattr(_host_synth, "__version__", None)
        if host_ver:
            synth_pkg = f"synth-ai=={host_ver}"
    except Exception:
        pass
    if not any(str(p).startswith("synth-ai") for p in pip_packages):
        pip_packages.insert(0, synth_pkg)

    local_dirs = [(str(Path(src)), dst) for src, dst in modal_cfg.extra_local_dirs]
    # Also mount the host synth_ai source if available to ensure latest code is used
    try:
        import synth_ai as _host_synth
        host_synth_dir = Path(_host_synth.__file__).resolve().parent
        # host_synth_dir points to .../synth_ai; mount that directory
        sy_dst = "/opt/synth_ai_repo/synth_ai"
        candidate = (str(host_synth_dir), sy_dst)
        if candidate not in local_dirs:
            local_dirs.insert(0, candidate)
    except Exception:
        pass
    # Ensure the discovered app directory is mounted, regardless of modal_cfg
    if original_path:
        discovered_dir = str(Path(original_path).resolve().parent)
        mount_dst = "/opt/synth_ai_repo/__local_task_app__"
        if (discovered_dir, mount_dst) not in local_dirs:
            local_dirs.append((discovered_dir, mount_dst))
    secret_names = list(modal_cfg.secret_names)
    volume_mounts = [(name, mount) for name, mount in modal_cfg.volume_mounts]

    script = f"""from __future__ import annotations

import importlib
import importlib.util
import sys
import os
import shutil
import tempfile
from pathlib import Path as _Path
import fnmatch
sys.path.insert(0, '/opt/synth_ai_repo')

from modal import App, Image, Secret, Volume, asgi_app

 # Defer importing synth_ai until inside fastapi_app to avoid local import errors

ENTRY_ID = {entry.app_id!r}
MODAL_APP_NAME = {modal_name!r}
MODULE_NAME = {module_name!r}
MODULE_FILE = {guaranteed_file_str or remote_file_str!r}
DOTENV_PATHS = {dotenv_paths!r}

image = Image.debian_slim(python_version={modal_cfg.python_version!r})

pip_packages = {pip_packages!r}
if pip_packages:
    image = image.pip_install(*pip_packages)

local_dirs = {local_dirs!r}

def _copy_tree_filtered(src_dir: str) -> str:
    src = _Path(src_dir)
    temp_dir = _Path(tempfile.mkdtemp(prefix='synth_mount_'))

    exclude_dirs = {'.cache', '.git', '__pycache__'}
    exclude_globs = ['*.db', '*.db-journal', '*-wal', '*-shm']

    for root, dirs, files in os.walk(src):
        rel_root = _Path(root).relative_to(src)
        # filter dirs in-place
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        # ensure target directory exists
        target_dir = (temp_dir / rel_root)
        target_dir.mkdir(parents=True, exist_ok=True)
        # copy files with filtering
        for name in files:
            if any(fnmatch.fnmatch(name, pat) for pat in exclude_globs):
                continue
            src_file = _Path(root) / name
            dst_file = target_dir / name
            try:
                shutil.copy2(src_file, dst_file)
            except Exception:
                # ignore problematic files
                continue
    return str(temp_dir)

for local_src, remote_dst in local_dirs:
    safe_src = _copy_tree_filtered(local_src)
    image = image.add_local_dir(safe_src, remote_dst)

secrets = {secret_names!r}
secret_objs = [Secret.from_name(name) for name in secrets]

if DOTENV_PATHS:
    secret_objs.extend(Secret.from_dotenv(path) for path in DOTENV_PATHS)

volume_mounts = {volume_mounts!r}
volume_map = {{}}
for vol_name, mount_path in volume_mounts:
    volume_map[mount_path] = Volume.from_name(vol_name, create_if_missing=True)

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
    # Import the module to trigger registration (inside container)
    import os
    # Prefer mounted source over any preinstalled site-packages version
    import sys as _sys
    for k in list(_sys.modules.keys()):
        if k == 'synth_ai' or k.startswith('synth_ai.'):
            _sys.modules.pop(k, None)
    import importlib as _importlib
    _importlib.invalidate_caches()
    try:
        if MODULE_FILE and os.path.exists(MODULE_FILE):
            spec = importlib.util.spec_from_file_location(MODULE_NAME or 'task_app_module', MODULE_FILE)
            if not spec or not spec.loader:
                raise RuntimeError("Failed to prepare spec for: " + str(MODULE_FILE))
            mod = importlib.util.module_from_spec(spec)
            sys.modules[MODULE_NAME or 'task_app_module'] = mod
            spec.loader.exec_module(mod)
        else:
            try:
                importlib.import_module(MODULE_NAME)
            except Exception:
                fallback_file = '/opt/synth_ai_repo/__local_task_app__/' + (MODULE_NAME.split('.')[-1] if MODULE_NAME else 'task_app') + '.py'
                if os.path.exists(fallback_file):
                    spec = importlib.util.spec_from_file_location(MODULE_NAME or 'task_app_module', fallback_file)
                    if not spec or not spec.loader:
                        raise RuntimeError("Failed to prepare fallback spec for: " + str(fallback_file))
                    mod = importlib.util.module_from_spec(spec)
                    sys.modules[MODULE_NAME or 'task_app_module'] = mod
                    spec.loader.exec_module(mod)
                else:
                    raise
    except Exception as e:
        raise RuntimeError("Task app import failed: " + str(e))

    # Get the entry from registry (now that it's registered)
    from synth_ai.task.apps import registry
    from synth_ai.task.server import create_task_app
    entry = registry.get(ENTRY_ID)
    cfg = entry.modal
    if cfg is None:
        raise RuntimeError("Modal configuration missing for task app " + ENTRY_ID)
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
    cli.add_command(eval_command)


@click.command("eval")
@click.argument("app_id", type=str, required=False)
@click.option("--config", type=click.Path(), default=None, help="Path to eval TOML (short schema)")
@click.option("--url", "task_app_url", type=str, default=None, help="Base URL of a running task app (skip in-process server)")
@click.option("--seeds", default="0,1,2,3,4", help="Comma-separated seeds/indices to evaluate")
@click.option("--split", default="train", show_default=True, help="Dataset split to use")
@click.option("--model", default=None, help="Model identifier (prompted if omitted)")
@click.option('--env-file', multiple=True, type=click.Path(), help='Env file(s) for keys')
def eval_command(app_id: str | None, config: str | None, task_app_url: str | None, seeds: str, split: str, model: str | None, env_file: Sequence[str]) -> None:
    """Run local rollouts against a task app using in-process ASGI and summarize results."""
    cfg: dict[str, Any] = {}
    config_path: Path | None = None
    if config:
        config_path = Path(config)
    else:
        auto_configs = _discover_eval_config_paths()
        if auto_configs:
            config_path = auto_configs[0]
            click.echo(f"Using eval config: {config_path}")

    if config_path:
        if _toml is None:
            raise click.ClickException("TOML parser not available; use Python 3.11+ or install tomli")
        if not config_path.exists():
            raise click.ClickException(f"Eval config not found: {config_path}")
        try:
            data = config_path.read_bytes()
            parsed = _toml.loads(data.decode("utf-8"))
            if isinstance(parsed, dict):
                section = parsed.get("eval")
                if isinstance(section, dict):
                    cfg = dict(section)
                else:
                    cfg = dict(parsed)
        except Exception as exc:
            raise click.ClickException(f"Failed to parse TOML '{config_path}': {exc}")

    app_id = app_id or (cfg.get("app_id") if isinstance(cfg.get("app_id"), str) else None)  # type: ignore

    # Determine selection params (CLI takes precedence; TOML only fills unset model/seeds/env)
    if cfg.get("model") and not model:
        model = str(cfg["model"])  # type: ignore[index]
    if cfg.get("seeds") and seeds == "0,1,2,3,4":
        val = cfg["seeds"]
        if isinstance(val, list):
            try:
                seeds = ",".join(str(int(x)) for x in val)
            except Exception:
                pass
        elif isinstance(val, str):
            seeds = val
        elif isinstance(val, int):
            seeds = str(val)
    if cfg.get("env_file") and not env_file:
        ef = cfg["env_file"]
        if isinstance(ef, str):
            env_file = (ef,)  # type: ignore[assignment]
        elif isinstance(ef, list):
            env_file = tuple(str(x) for x in ef)  # type: ignore[assignment]

    entry: TaskAppEntry | None = None
    if task_app_url is None:
        choice = _select_app_choice(app_id, purpose="eval")
        entry = choice.ensure_entry()

    env_paths: list[Path] = []
    if entry is not None:
        env_paths = _determine_env_files(entry, env_file)
    else:
        if not env_file:
            raise click.ClickException("--env-file is required when using --url")
        for candidate in env_file:
            p = Path(candidate).expanduser()
            if not p.exists():
                raise click.ClickException(f"Env file not found: {p}")
            env_paths.append(p)

    click.echo('Using env file(s): ' + ', '.join(str(p) for p in env_paths))
    _load_env_files_into_process([str(Path(p)) for p in env_paths])

    if task_app_url is None:
        config = entry.config_factory()  # type: ignore[union-attr]
        # Help the type checker; runtime check also enforced in server.run_task_app
        if not isinstance(config, TaskAppConfig):
            raise click.ClickException("Invalid task app: config_factory did not return TaskAppConfig")
        app = create_task_app(config)

    # Determine supported models
    supported: list[str] = []
    if task_app_url is None:
        try:
            supported = list((config.base_task_info.inference or {}).get("models") or [])  # type: ignore[union-attr]
        except Exception:
            supported = []
    else:
        try:
            import httpx as _hx
            headers = {}
            api_key = (os.environ.get("ENVIRONMENT_API_KEY") or "").strip()
            if api_key:
                headers["X-API-Key"] = api_key
            with _hx.Client(base_url=task_app_url, headers=headers, timeout=15.0) as c:
                info = c.get("/info").json()
            inf = info.get("inference") if isinstance(info, dict) else None
            if isinstance(inf, dict):
                m = inf.get("models")
                if isinstance(m, list):
                    supported = [str(x) for x in m]
                if not supported:
                    providers = inf.get("providers")
                    if isinstance(providers, list):
                        if "openai" in providers:
                            supported.append("gpt-5")
                        if "groq" in providers:
                            supported.append("groq:llama-3.1-70b-versatile")
                        supported.append("synth:qwen-0.6b")
        except Exception:
            supported = []
    if not supported:
        # Only fall back to local config-derived providers when running in-process
        if task_app_url is None:
            try:
                providers = list((config.base_task_info.inference or {}).get("providers") or [])  # type: ignore[union-attr]
            except Exception:
                providers = []
            if "openai" in providers:
                supported.append("gpt-5")
            if "groq" in providers:
                supported.append("groq:llama-3.1-70b-versatile")
        # Always include a local synth model option for smoke tests
        supported.append("synth:qwen-0.6b")

    selected_model = model
    if not selected_model:
        if not supported:
            raise click.ClickException("No supported models; supply --model or add base_task_info.inference.models")
        click.echo("Select model to evaluate:")
        for idx, m in enumerate(supported, start=1):
            click.echo(f"  {idx}) {m}")
        choice_idx = click.prompt('Enter choice', type=click.IntRange(1, len(supported)))
        selected_model = supported[choice_idx - 1]

    try:
        seed_values = [int(s.strip()) for s in seeds.split(',') if s.strip()]
    except Exception:
        raise click.ClickException("Invalid --seeds; expected comma-separated integers")

    import httpx
    headers = {}
    api_key = (os.environ.get("ENVIRONMENT_API_KEY") or "").strip()
    if api_key:
        headers["X-API-Key"] = api_key

    successes = 0
    failures = 0
    # Aggregate outcome stats across successful seeds
    outcome_sum: float = 0.0
    outcome_count: int = 0
    outcome_correct: int = 0
    if task_app_url is None:
        transport = httpx.ASGITransport(app=app)  # type: ignore[name-defined]
        # Newer httpx types consider ASGITransport under httpx._transports; cast to satisfy type checker
        client = httpx.Client(transport=cast(Any, transport), base_url="http://eval.local", timeout=60.0, headers=headers)
    else:
        client = httpx.Client(base_url=task_app_url, timeout=60.0, headers=headers)
    with client as client:
        try:
            client.get("/task_info")
        except Exception:
            pass
        # Precompute optional policy overrides from TOML
        policy_overrides: dict[str, Any] = {}
        try:
            # Accept [eval.policy] table or top-level keys for convenience
            if isinstance(cfg.get("policy"), dict):
                policy_overrides.update(dict(cfg["policy"]))
            # Back-compat: allow temperature/max_tokens at top level
            for k in ("temperature", "max_tokens", "reasoning_effort", "system_hint", "tool_choice"):
                if k in cfg and k not in policy_overrides:
                    policy_overrides[k] = cfg.get(k)
        except Exception:
            policy_overrides = {}

        for seed_val in seed_values:
            body = {
                "run_id": str(uuid.uuid4()),
                "env": {"config": {"split": split, "index": seed_val}, "seed": seed_val},
                "policy": {"policy_name": selected_model, "config": {"model": selected_model, **policy_overrides}},
                "ops": [],
            }
            try:
                resp = client.post("/rollout", json=body)
                ok = 200 <= resp.status_code < 300
                if ok:
                    successes += 1
                else:
                    failures += 1

                # Print summary with any available metrics/tool calls
                summary = [f"seed={seed_val}", f"status={resp.status_code}"]
                try:
                    data = resp.json()
                except Exception:
                    data = None
                if isinstance(data, dict):
                    metrics = data.get("metrics") if isinstance(data.get("metrics"), dict) else None
                    if metrics:
                        mean_return = metrics.get("mean_return") or metrics.get("total_reward")
                        outcome = metrics.get("outcome_score")
                        if mean_return is not None:
                            summary.append(f"mean_return={mean_return}")
                        if outcome is not None:
                            summary.append(f"outcome={outcome}")
                            # Aggregate outcome stats
                            try:
                                val = float(outcome)
                                outcome_sum += val
                                outcome_count += 1
                                if val >= 0.5:
                                    outcome_correct += 1
                            except Exception:
                                pass
                    # Try to infer tool call count from first trajectory step
                    trajs = data.get("trajectories") if isinstance(data.get("trajectories"), list) else None
                    if trajs:
                        first = trajs[0] if trajs else None
                        steps = first.get("steps") if isinstance(first, dict) else None
                        if isinstance(steps, list) and steps:
                            step0 = steps[0]
                            tool_calls = step0.get("tool_calls") or step0.get("tools") or []
                            if isinstance(tool_calls, list):
                                summary.append(f"tool_calls={len(tool_calls)}")
                    click.echo(" ".join(summary))
                    # Print the full response JSON (trace, trajectories, metrics)
                    try:
                        click.echo(json.dumps(data, indent=2))
                    except Exception:
                        pass
                else:
                    click.echo(" ".join(summary))
            except Exception as exc:
                failures += 1
                click.echo(f"seed={seed_val} error={exc}")

    click.echo(f"Eval complete: {successes} ok, {failures} failed; model={selected_model}, split={split}")
    # Print outcome summary if any successes
    if outcome_count > 0:
        mean_outcome = outcome_sum / float(outcome_count)
        frac_right = outcome_correct / float(outcome_count)
        click.echo(f"Outcome summary: correct={outcome_correct}/{outcome_count} ({frac_right:.2%}), mean_outcome={mean_outcome:.3f}")


def register_eval(cli: click.Group) -> None:
    cli.add_command(eval_command)

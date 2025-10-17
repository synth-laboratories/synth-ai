from __future__ import annotations

import ast
import asyncio
import contextlib
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import types
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

try:  # Python 3.11+
    import tomllib as _toml
except Exception:  # pragma: no cover - fallback
    _toml = None  # type: ignore
import uuid

import click

from synth_ai.config.base_url import PROD_BASE_URL_DEFAULT
from synth_ai.task.apps import ModalDeploymentConfig, TaskAppConfig, TaskAppEntry, registry
from synth_ai.task.server import create_task_app, run_task_app

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


def _possible_module_names(
    path: Path, module_search_roots: Sequence[Path]
) -> list[tuple[str, Path]]:
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
    """Only search for task apps in the current working directory and subdirectories."""
    roots: list[Path] = []

    # Prioritize demo directory if it exists
    try:
        from synth_ai.demos.demo_task_apps.core import load_demo_dir

        demo_dir = load_demo_dir()
        if demo_dir:
            demo_path = Path(demo_dir)
            if demo_path.exists() and demo_path.is_dir():
                roots.append(demo_path.resolve())
    except Exception:
        pass

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
            root = root.resolve()
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
    # Look for entry=TaskAppEntry(app_id="...", ...)
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


def _extract_modal_app_name(node: ast.Call) -> str | None:
    for kw in node.keywords:
        if (
            kw.arg in {"name", "app_name"}
            and isinstance(kw.value, ast.Constant)
            and isinstance(kw.value.value, str)
        ):
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
                        entry_loader=lambda p=path.resolve(),
                        a=app_id,
                        roots=(root_resolved,): _load_entry_from_path(
                            p, a, module_search_roots=roots
                        ),
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


def _app_choice_sort_key(choice: AppChoice) -> tuple[int, int, int, int, int, str, str]:
    """Ranking heuristic so wrapper-style task apps surface first."""

    # Prioritize apps in the current working directory (demo or otherwise)
    cwd_rank = 1
    try:
        cwd = Path.cwd().resolve()
        if choice.path.is_relative_to(cwd):
            # Check if this is directly in CWD (not in subdirectories like examples/)
            try:
                rel_path = choice.path.relative_to(cwd)
                # If it's in the immediate directory or one level deep, prioritize it
                if len(rel_path.parts) <= 2:
                    cwd_rank = 0
            except Exception:
                pass
    except Exception:
        pass

    # Further prioritize apps in the demo directory if one is set
    demo_rank = 1
    try:
        from synth_ai.demos.demo_task_apps.core import load_demo_dir

        demo_dir = load_demo_dir()
        if demo_dir:
            demo_path = Path(demo_dir).resolve()
            if choice.path.is_relative_to(demo_path):
                demo_rank = 0
    except Exception:
        pass

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
            if isinstance(node, ast.Call) and _is_register_task_app_call(node):
                # Check if the entry has modal=ModalDeploymentConfig(...)
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
            if isinstance(node, ast.Call) and _is_register_task_app_call(node):
                # Check if the entry has modal=ModalDeploymentConfig(...)
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
            elif kw.arg == "pip_packages" and isinstance(kw.value, ast.List | ast.Tuple):
                # Handle pip_packages list/tuple
                packages = []
                for elt in kw.value.elts:
                    if isinstance(elt, ast.Constant):
                        packages.append(elt.value)
                kwargs[kw.arg] = tuple(packages)
            elif kw.arg == "extra_local_dirs" and isinstance(kw.value, ast.List | ast.Tuple):
                # Handle extra_local_dirs list/tuple of tuples
                dirs = []
                for elt in kw.value.elts:
                    if isinstance(elt, ast.List | ast.Tuple) and len(elt.elts) == 2:
                        src = elt.elts[0].value if isinstance(elt.elts[0], ast.Constant) else None
                        dst = elt.elts[1].value if isinstance(elt.elts[1], ast.Constant) else None
                        if src and dst:
                            dirs.append((src, dst))
                kwargs[kw.arg] = tuple(dirs)
            elif kw.arg == "secret_names" and isinstance(kw.value, ast.List | ast.Tuple):
                # Handle secret_names list/tuple
                secrets = []
                for elt in kw.value.elts:
                    if isinstance(elt, ast.Constant):
                        secrets.append(elt.value)
                kwargs[kw.arg] = tuple(secrets)
            elif kw.arg == "volume_mounts" and isinstance(kw.value, ast.List | ast.Tuple):
                # Handle volume_mounts list/tuple of tuples
                mounts = []
                for elt in kw.value.elts:
                    if isinstance(elt, ast.List | ast.Tuple) and len(elt.elts) == 2:
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
    # Get file modification timestamp
    try:
        from datetime import datetime

        mtime = choice.path.stat().st_mtime
        modified_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        details = f"Modified: {modified_str}"
    except Exception:
        # Fallback if timestamp unavailable
        details = choice.description or "No timestamp available"
    # Format: single line with timestamp
    main_line = f"{prefix}{choice.app_id} ({choice.source}) – {details}"
    return main_line


def _prompt_user_for_choice(choices: list[AppChoice]) -> AppChoice:
    click.echo("Select a task app:")
    for idx, choice in enumerate(choices, start=1):
        click.echo(_format_choice(choice, idx))
    try:
        response = click.prompt("Enter choice", default="1", type=str).strip() or "1"
    except (click.exceptions.Abort, EOFError, KeyboardInterrupt) as exc:
        raise click.ClickException("Task app selection cancelled by user") from exc
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
        if ensure_namespace and namespace_root is not None and "." in module_name:
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

            def _return_config(cfg: TaskAppConfig = attr) -> TaskAppConfig:
                return cfg

            factory_callable = _return_config
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
                if (
                    param.kind
                    in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                    and param.default is inspect._empty
                ):
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
                bound_func: Callable[[], TaskAppConfig] = cast(Callable[[], TaskAppConfig], attr)  # type: ignore[assignment]

                def _factory_noargs(
                    func: Callable[[], TaskAppConfig] = bound_func,
                ) -> TaskAppConfig:
                    return func()

                factory_callable = _factory_noargs
                config_obj = result
                break

    # If no TaskAppConfig found directly, check if it was registered via register_task_app
    if factory_callable is None or config_obj is None:
        try:
            # Check if the app was registered in the registry
            entry = registry.get(app_id)
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
    cwd_env_files = sorted(cwd.glob("**/*.env"))
    env_candidates.extend(cwd_env_files)

    # Add repo root env files
    repo_env_files = sorted(REPO_ROOT.glob("**/*.env"))
    # Avoid duplicates
    for repo_file in repo_env_files:
        if repo_file not in env_candidates:
            env_candidates.append(repo_file)

    if not env_candidates:
        created = _interactive_create_env(script_dir)
        if created is None:
            raise click.ClickException("Env file required (--env-file) for this task app")
        return [created]

    click.echo("Select env file to load:")
    for idx, path in enumerate(env_candidates, start=1):
        click.echo(f"  {idx}) {path.resolve()}")
    choice = click.prompt("Enter choice", type=click.IntRange(1, len(env_candidates)), default=1)
    return [env_candidates[choice - 1]]


def _modal_command_prefix(modal_cli: str) -> list[str]:
    """Resolve a command prefix for invoking the Modal CLI within the active environment."""
    if modal_cli == "modal" and importlib.util.find_spec("modal") is not None:
        return [sys.executable, "-m", "synth_ai.cli._modal_wrapper"]

    modal_path = shutil.which(modal_cli)
    if modal_path is not None:
        return [modal_path]

    if modal_cli == "modal":
        raise click.ClickException(
            "Modal CLI not found. Install the 'modal' package in this environment or pass "
            "--modal-cli with an explicit path."
        )
    raise click.ClickException(f"Modal CLI not found (looked for '{modal_cli}')")


def _build_modal_app_wrapper(original_script: Path) -> tuple[Path, Path]:
    source_dir = original_script.parent.resolve()
    repo_root = REPO_ROOT
    synth_src = (repo_root / "synth_ai").resolve()
    temp_root = Path(tempfile.mkdtemp(prefix="synth_modal_app_"))

    wrapper_source = textwrap.dedent(
        f"""
        from importlib import util as _util
        from pathlib import Path as _Path
        import sys as _sys

        _source_dir = _Path({str(source_dir)!r}).resolve()
        _module_path = _source_dir / {original_script.name!r}
        _package_name = _source_dir.name
        _repo_root = _Path({str(repo_root)!r}).resolve()
        _synth_dir = _repo_root / "synth_ai"

        for _path in (str(_source_dir), str(_source_dir.parent), str(_repo_root)):
            if _path not in _sys.path:
                _sys.path.insert(0, _path)

        _spec = _util.spec_from_file_location("_synth_modal_target", str(_module_path))
        if _spec is None or _spec.loader is None:
            raise SystemExit("Unable to load modal task app from {original_script}")
        _module = _util.module_from_spec(_spec)
        _sys.modules.setdefault("_synth_modal_target", _module)
        _spec.loader.exec_module(_module)

        try:
            from modal import App as _ModalApp
            from modal import Image as _ModalImage
        except Exception:
            _ModalApp = None  # type: ignore[assignment]
            _ModalImage = None  # type: ignore[assignment]

        def _apply_local_mounts(image):
            if _ModalImage is None or not isinstance(image, _ModalImage):
                return image
            mounts = [
                (str(_source_dir), f"/root/{{_package_name}}"),
                (str(_synth_dir), "/root/synth_ai"),
            ]
            for local_path, remote_path in mounts:
                try:
                    image = image.add_local_dir(local_path, remote_path=remote_path)
                except Exception:
                    pass
            return image

        if hasattr(_module, "image"):
            _module.image = _apply_local_mounts(getattr(_module, "image"))

        _candidate = getattr(_module, "app", None)
        if _ModalApp is None or not isinstance(_candidate, _ModalApp):
            candidate_modal_app = getattr(_module, "modal_app", None)
            if _ModalApp is not None and isinstance(candidate_modal_app, _ModalApp):
                _candidate = candidate_modal_app
                setattr(_module, "app", _candidate)

        if _ModalApp is not None and not isinstance(_candidate, _ModalApp):
            raise SystemExit(
                "Modal task app must expose an 'app = modal.App(...)' (or modal_app) attribute."
            )

        for remote_path in ("/root/synth_ai", f"/root/{{_package_name}}"):
            if remote_path not in _sys.path:
                _sys.path.insert(0, remote_path)

        globals().update({{k: v for k, v in vars(_module).items() if not k.startswith("__")}})
        app = getattr(_module, "app")
        """
    ).strip()

    wrapper_path = temp_root / "__modal_wrapper__.py"
    wrapper_path.write_text(wrapper_source + "\n", encoding="utf-8")
    return wrapper_path, temp_root



def _run_modal_script(
    script_path: Path,
    modal_cli: str,
    command: str,
    env_paths: Sequence[Path],
    *,
    modal_name: str | None = None,
    dry_run: bool = False,
) -> None:
    env_paths_list = [Path(p).resolve() for p in env_paths]
    path_strings = [str(p) for p in env_paths_list]
    _load_env_files_into_process(path_strings)
    _ensure_env_values(env_paths_list, script_path.parent)
    _load_env_values(env_paths_list)
    # Ensure ENVIRONMENT_API_KEY is uploaded to backend for this org (matches registry path behavior)
    try:
        _preflight_env_key(env_paths_list, crash_on_failure=True)
    except Exception as _pf_err:
        raise click.ClickException(str(_pf_err))

    proc_env = os.environ.copy()
    pythonpath_entries: list[str] = []
    script_dir = script_path.parent.resolve()
    pythonpath_entries.append(str(script_dir))
    if (script_dir / "__init__.py").exists():
        # Script lives inside a package; ensure the parent package directory is importable.
        pythonpath_entries.append(str(script_dir.parent.resolve()))
    pythonpath_entries.append(str(REPO_ROOT))
    existing_pp = proc_env.get("PYTHONPATH")
    if existing_pp:
        pythonpath_entries.append(existing_pp)
    unique_paths = list(dict.fromkeys(pythonpath_entries))
    proc_env["PYTHONPATH"] = os.pathsep.join(unique_paths)

    wrapper_info: tuple[Path, Path] | None = None
    target_script = script_path
    if command in {"serve", "deploy"}:
        wrapper_path, temp_root = _build_modal_app_wrapper(script_path)
        wrapper_info = (wrapper_path, temp_root)
        target_script = wrapper_path

        # Ensure the wrapper has access to the Synth AI source for intra-repo imports
        if "PYTHONPATH" in proc_env:
            proc_env["PYTHONPATH"] = os.pathsep.join(
                [str(REPO_ROOT)] + proc_env["PYTHONPATH"].split(os.pathsep)
            )
        else:
            proc_env["PYTHONPATH"] = str(REPO_ROOT)

    cmd = [*_modal_command_prefix(modal_cli), command, str(target_script)]
    if modal_name and command == "deploy":
        cmd.extend(["--name", modal_name])
    if dry_run:
        click.echo("Dry run: " + " ".join(cmd))
        return
    try:
        # Stream output live for better diagnostics
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=proc_env,
        )
        task_app_url = None
        assert proc.stdout is not None
        for line in proc.stdout:
            click.echo(line, nl=False)
            if task_app_url is None and ("modal.run" in line and "=>" in line):
                parts = line.split("=>")
                if len(parts) >= 2:
                    task_app_url = parts[-1].strip()
                    if task_app_url and env_paths_list:
                        env_file = env_paths_list[0]
                        _save_to_env_file(env_file, "TASK_APP_BASE_URL", task_app_url)
                        click.echo(f"\n✓ Task app URL: {task_app_url}\n")
        rc = proc.wait()
        if rc != 0:
            raise subprocess.CalledProcessError(rc, cmd)
    except subprocess.CalledProcessError as exc:
        raise click.ClickException(
            f"modal {command} failed with exit code {exc.returncode}"
        ) from exc
    finally:
        if wrapper_info is not None:
            wrapper_path, temp_root = wrapper_info
            try:
                wrapper_path.unlink(missing_ok=True)
            except Exception:
                pass
            shutil.rmtree(temp_root, ignore_errors=True)


def _preflight_env_key(env_paths: Sequence[Path] | None = None, *, crash_on_failure: bool = False) -> None:
    try:
        raw_backend = (
            os.environ.get("BACKEND_BASE_URL")
            or os.environ.get("SYNTH_BASE_URL")
            or f"{PROD_BASE_URL_DEFAULT}/api"
        )
        backend_base = raw_backend.rstrip("/")
        if not backend_base.endswith("/api"):
            backend_base = backend_base + "/api"
        synth_key = os.environ.get("SYNTH_API_KEY") or ""
        env_api_key = (
            os.environ.get("ENVIRONMENT_API_KEY") or os.environ.get("DEV_ENVIRONMENT_API_KEY") or ""
        ).strip()

        def _preview(value: str) -> str:
            if len(value) <= 10:
                return value
            return f"{value[:6]}...{value[-4:]}"

        minted = False
        if not env_api_key:
            try:
                from synth_ai.learning.rl.secrets import mint_environment_api_key

                env_api_key = mint_environment_api_key()
                os.environ["ENVIRONMENT_API_KEY"] = env_api_key
                os.environ.setdefault("DEV_ENVIRONMENT_API_KEY", env_api_key)
                minted = True
                click.echo(
                    f"[preflight] minted ENVIRONMENT_API_KEY ({_preview(env_api_key)})"
                )
            except Exception as mint_err:
                if crash_on_failure:
                    raise click.ClickException(
                        f"[CRITICAL] Failed to mint ENVIRONMENT_API_KEY: {mint_err}"
                    ) from mint_err
                click.echo(
                    f"[WARN] Failed to mint ENVIRONMENT_API_KEY automatically ({mint_err}); proceeding without upload"
                )

        if env_api_key and not os.environ.get("ENVIRONMENT_API_KEY"):
            os.environ["ENVIRONMENT_API_KEY"] = env_api_key
        if env_api_key and not os.environ.get("DEV_ENVIRONMENT_API_KEY"):
            os.environ["DEV_ENVIRONMENT_API_KEY"] = env_api_key

        if minted:
            _persist_env_api_key(env_api_key, env_paths)

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

                    # Decode public key and build sealed box
                    pk_bytes = base64.b64decode(pk, validate=True)
                    pub = PublicKey(pk_bytes)
                    sb = SealedBox(pub)

                    # Encrypt plaintext key
                    ct_b64 = base64.b64encode(sb.encrypt(env_api_key.encode("utf-8"))).decode()
                    payload = {"name": "ENVIRONMENT_API_KEY", "ciphertext_b64": ct_b64}

                    # Emit diagnostic logging (safe previews + hashes only)
                    try:
                        import hashlib as _hash

                        # Backend URL context
                        click.echo(f"[preflight] posting to {backend_base.rstrip('/')}/v1/env-keys")

                        # Public key diagnostics
                        pk_sha256 = _hash.sha256(pk_bytes).hexdigest()
                        click.echo(
                            f"[preflight] public_key: b64_len={len(pk)} sha256={pk_sha256} head={pk[:16]} tail={pk[-16:]}"
                        )

                        # Plaintext diagnostics (never print full secret)
                        _plain = env_api_key
                        _plen = len(_plain)
                        _ppref = (_plain[:6] + "…") if _plen > 10 else _plain
                        _psuf = ("…" + _plain[-4:]) if _plen > 10 else ""
                        _has_ws = any(ch.isspace() for ch in _plain)
                        click.echo(
                            f"[preflight] plaintext: len={_plen} preview={_ppref}{_psuf} has_ws={bool(_has_ws)}"
                        )

                        # Ciphertext diagnostics
                        try:
                            _ct_bytes = base64.b64decode(ct_b64, validate=True)
                            _ct_sha256 = _hash.sha256(_ct_bytes).hexdigest()
                            click.echo(
                                f"[preflight] ciphertext: b64_len={len(ct_b64)} sha256={_ct_sha256} head={ct_b64[:16]} tail={ct_b64[-16:]}"
                            )
                        except Exception:
                            click.echo("[preflight] ciphertext: invalid base64 (unexpected)")
                    except Exception:
                        # Best-effort logging only
                        pass
                    with httpx.Client(
                        timeout=15.0,
                        headers={
                            "Authorization": f"Bearer {synth_key}",
                            "Content-Type": "application/json",
                        },
                    ) as c:
                        click.echo("[preflight] upserting env key…")
                        up = c.post(f"{backend_base.rstrip('/')}/v1/env-keys", json=payload)
                        body_snip = ""
                        try:
                            body_snip = up.text[:400] if up.text else ""
                        except Exception:
                            body_snip = ""
                        click.echo(f"[preflight] upsert status={up.status_code}{(' body='+body_snip) if body_snip else ''}")

                        # If upload succeeded (2xx), consider it successful even if verification fails
                        # This handles cases where verification endpoint has issues
                        if 200 <= up.status_code < 300:
                            key_preview = (
                                _preview(env_api_key)
                            )
                            click.echo(
                                f"✅ ENVIRONMENT_API_KEY uploaded successfully ({key_preview})"
                            )

                            # Try verification, but don't fail if it doesn't work
                            click.echo("[preflight] verifying env key presence…")
                            try:
                                ver = c.get(f"{backend_base.rstrip('/')}/v1/env-keys/verify")
                                if ver.status_code == 200 and (ver.json() or {}).get("present"):
                                    click.echo("✅ Key verified in backend")
                                else:
                                    click.echo(
                                        f"⚠️  Verification returned {ver.status_code}, but upload succeeded - proceeding"
                                    )
                            except Exception as verify_err:
                                click.echo(
                                    f"⚠️  Verification check failed ({verify_err}), but upload succeeded - proceeding"
                                )
                        else:
                            error_msg = (
                                f"ENVIRONMENT_API_KEY upload failed with status {up.status_code}"
                                + (f" body={body_snip}" if body_snip else "")
                            )
                            if crash_on_failure:
                                raise click.ClickException(f"[CRITICAL] {error_msg}")
                            click.echo(f"[WARN] {error_msg}; proceeding anyway")
                except Exception as e:
                    error_msg = f"Failed to encrypt/upload ENVIRONMENT_API_KEY: {e}"
                    if crash_on_failure:
                        raise click.ClickException(f"[CRITICAL] {error_msg}") from e
                    click.echo(f"[WARN] {error_msg}; proceeding anyway")
    except Exception as e:
        error_msg = f"Backend preflight for ENVIRONMENT_API_KEY failed: {e}"
        if crash_on_failure:
            raise click.ClickException(f"[CRITICAL] {error_msg}") from e
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
    env_paths_list = [Path(p).resolve() for p in env_paths]
    dotenv_paths = [str(p) for p in env_paths_list]
    _load_env_files_into_process(dotenv_paths)
    fallback_dir = env_paths_list[0].parent if env_paths_list else Path.cwd()
    _ensure_env_values(env_paths_list, fallback_dir)
    _load_env_values(env_paths_list)
    _preflight_env_key(env_paths_list, crash_on_failure=True)

    inline_secret_values: dict[str, str] = {}
    env_key = os.environ.get("ENVIRONMENT_API_KEY", "").strip()
    if env_key:
        inline_secret_values["ENVIRONMENT_API_KEY"] = env_key
        inline_secret_values.setdefault("DEV_ENVIRONMENT_API_KEY", env_key)
    aliases = os.environ.get("ENVIRONMENT_API_KEY_ALIASES", "").strip()
    if aliases:
        inline_secret_values["ENVIRONMENT_API_KEY_ALIASES"] = aliases
    for vendor_key in ("GROQ_API_KEY", "OPENAI_API_KEY"):
        val = os.environ.get(vendor_key, "").strip()
        if val:
            inline_secret_values[vendor_key] = val

    if inline_secret_values:
        preview = inline_secret_values.get("ENVIRONMENT_API_KEY", "")
        shown = f"{preview[:6]}...{preview[-4:]}" if preview and len(preview) > 10 else preview
        click.echo(f"[deploy] inline ENVIRONMENT_API_KEY prepared ({shown})")
    else:
        click.echo("[deploy] no inline ENVIRONMENT_API_KEY found; relying on Modal secrets/dotenv")

    script_path = _write_modal_entrypoint(
        entry,
        modal_cfg,
        modal_name,
        dotenv_paths=dotenv_paths,
        original_path=original_path,
        inline_secret_values=inline_secret_values,
    )
    cmd = [*_modal_command_prefix(modal_cli), command, str(script_path)]

    if modal_name and command == "deploy":
        cmd.extend(["--name", modal_name])

    proc_env = os.environ.copy()
    pythonpath_entries: list[str] = [str(REPO_ROOT)]
    if original_path is not None:
        source_dir = Path(original_path).resolve().parent
        pythonpath_entries.insert(0, str(source_dir))
    existing_pp = proc_env.get("PYTHONPATH")
    if existing_pp:
        pythonpath_entries.append(existing_pp)
    proc_env["PYTHONPATH"] = os.pathsep.join(list(dict.fromkeys(pythonpath_entries)))

    if dry_run:
        click.echo("Dry run: " + " ".join(cmd))
        script_path.unlink(missing_ok=True)
        return

    try:
        # Stream output live for better diagnostics
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=proc_env,
        )
        task_app_url = None
        assert proc.stdout is not None
        for line in proc.stdout:
            # Echo lines as they arrive
            click.echo(line, nl=False)
            # Look for lines containing modal.run URLs
            if task_app_url is None and ("modal.run" in line and "=>" in line):
                parts = line.split("=>")
                if len(parts) >= 2:
                    task_app_url = parts[-1].strip()
                    # Save URL immediately for convenience
                    if task_app_url and env_paths_list:
                        env_file = env_paths_list[0]
                        _save_to_env_file(env_file, "TASK_APP_BASE_URL", task_app_url)
                        click.echo(f"\n✓ Task app URL: {task_app_url}\n")
        rc = proc.wait()
        if rc != 0:
            raise subprocess.CalledProcessError(rc, cmd)
    except subprocess.CalledProcessError as exc:
        raise click.ClickException(
            f"modal {command} failed with exit code {exc.returncode}"
        ) from exc
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
            if not line or line.lstrip().startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
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
            if not line or line.lstrip().startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            data[key.strip()] = value.strip()
    except FileNotFoundError:
        pass
    return data


def _interactive_fill_env(env_path: Path) -> Path | None:
    existing = _parse_env_file(env_path) if env_path.exists() else {}

    def _prompt(label: str, *, default: str = "", required: bool) -> str | None:
        while True:
            try:
                value = click.prompt(
                    label, default=default, show_default=bool(default) or not required
                ).strip()
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
        raise click.ClickException(
            f"Task app '{entry.app_id}' does not define Modal deployment settings"
        )

    env_paths = _determine_env_files(entry, env_file)
    click.echo("Using env file(s): " + ", ".join(str(p.resolve()) for p in env_paths))
    _run_modal_with_entry(
        entry,
        modal_cfg,
        modal_cli,
        modal_name,
        env_paths,
        command="deploy",
        dry_run=dry_run,
        original_path=original_path,
    )


def _modal_serve_entry(
    entry: TaskAppEntry,
    modal_name: str | None,
    modal_cli: str,
    env_file: Sequence[str],
    original_path: Path | None = None,
) -> None:
    modal_cfg = entry.modal
    if modal_cfg is None:
        raise click.ClickException(
            f"Task app '{entry.app_id}' does not define Modal deployment settings"
        )

    env_paths = _determine_env_files(entry, env_file)
    click.echo("Using env file(s): " + ", ".join(str(p.resolve()) for p in env_paths))
    _run_modal_with_entry(
        entry,
        modal_cfg,
        modal_cli,
        modal_name,
        env_paths,
        command="serve",
        original_path=original_path,
    )


@click.group(name="task-app", help="Utilities for serving and deploying Synth task apps.")
def task_app_group() -> None:
    pass


@task_app_group.command("list")
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
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            key = k.strip()
            val = v.strip().strip('"').strip("'")
            # Load into process, but allow overriding if the current value is empty
            if key:
                current = os.environ.get(key)
                if current is None or not str(current).strip():
                    os.environ[key] = val


@click.command("serve")
@click.argument("app_id", type=str, required=False)
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", default=None, type=int, help="Port to serve on (default: 8001)")
@click.option("--env-file", multiple=True, type=click.Path(), help="Extra .env files to load")
@click.option(
    "--reload/--no-reload", "reload_flag", default=False, help="Enable uvicorn auto-reload"
)
@click.option(
    "--force/--no-force",
    "force",
    default=False,
    help="Kill any process already bound to the selected port before starting",
)
@click.option(
    "--trace",
    "trace_dir",
    type=click.Path(),
    default=None,
    help="Enable tracing and write SFT JSONL files to this directory (default: traces/v3)",
)
@click.option(
    "--trace-db",
    "trace_db",
    type=click.Path(),
    default=None,
    help="Override local trace DB path (default: traces/v3/synth_ai.db)",
)
def serve_command(
    app_id: str | None,
    host: str,
    port: int | None,
    env_file: Sequence[str],
    reload_flag: bool,
    force: bool,
    trace_dir: str | None,
    trace_db: str | None,
) -> None:
    # Change to demo directory if stored (REQUIRED for demo isolation)
    from synth_ai.demos.demo_task_apps.core import load_demo_dir

    demo_dir = load_demo_dir()
    if demo_dir:
        demo_path = Path(demo_dir)
        if not demo_path.is_dir():
            raise click.ClickException(
                f"Demo directory not found: {demo_dir}\nRun 'synth-ai setup' to create a demo."
            )
        os.chdir(demo_dir)
        click.echo(f"Using demo directory: {demo_dir}\n")
        # Store demo directory for path resolution
        os.environ["SYNTH_DEMO_DIR"] = str(demo_path.resolve())

    # Prompt for port if not provided
    if port is None:
        port = click.prompt("Port to serve on", type=int, default=8001)

    # Prompt for trace directory if not provided
    if trace_dir is None:
        click.echo(
            "\nTracing captures rollout data (actions, rewards, model outputs) to a local SQLite DB."
        )
        click.echo("This data can be exported to JSONL for supervised fine-tuning (SFT).")
        enable_tracing = click.confirm("Enable tracing?", default=True)
        if enable_tracing:
            demo_base = Path(os.environ.get("SYNTH_DEMO_DIR") or Path.cwd())
            default_trace_dir = str((demo_base / "traces/v3").resolve())
            trace_dir = click.prompt(
                "Trace directory", type=str, default=default_trace_dir, show_default=True
            )
        else:
            trace_dir = None

    # Prompt for trace DB if not provided and tracing is enabled
    if trace_dir and trace_db is None:
        demo_base = Path(os.environ.get("SYNTH_DEMO_DIR") or Path.cwd())
        default_trace_db = str((demo_base / "traces/v3/synth_ai.db").resolve())
        trace_db = click.prompt(
            "Trace DB path", type=str, default=default_trace_db, show_default=True
        )

    choice = _select_app_choice(app_id, purpose="serve")
    entry = choice.ensure_entry()
    _serve_entry(
        entry, host, port, env_file, reload_flag, force, trace_dir=trace_dir, trace_db=trace_db
    )


@task_app_group.command("info")
@click.option(
    "--base",
    "base_url",
    default=None,
    help="Task app base URL (default: TASK_APP_BASE_URL or http://127.0.0.1:8001)",
)
@click.option(
    "--api-key",
    default=None,
    help="Environment API key (default: ENVIRONMENT_API_KEY or dev fallbacks)",
)
@click.option(
    "--seed",
    "seeds",
    multiple=True,
    type=int,
    help="Optional seed(s) to request specific instances (repeatable)",
)
def info_command(base_url: str | None, api_key: str | None, seeds: tuple[int, ...]) -> None:
    """Fetch Task App /task_info with authentication and print JSON."""
    import json as _json
    import os as _os

    import requests as _requests

    base = (base_url or _os.getenv("TASK_APP_BASE_URL") or "http://127.0.0.1:8001").rstrip("/")

    # Resolve API key, permitting dev fallbacks
    try:
        from synth_ai.task.auth import normalize_environment_api_key as _norm_key
    except Exception:
        _norm_key = lambda: _os.getenv("ENVIRONMENT_API_KEY")  # noqa: E731
    key = (api_key or _norm_key() or "").strip()
    if not key:
        raise click.ClickException("Missing API key. Provide --api-key or set ENVIRONMENT_API_KEY.")

    headers: dict[str, str] = {"X-API-Key": key, "Authorization": f"Bearer {key}"}
    aliases = (_os.getenv("ENVIRONMENT_API_KEY_ALIASES") or "").strip()
    keys_csv = (
        ",".join([key] + [p.strip() for p in aliases.split(",") if p.strip()]) if aliases else key
    )
    if keys_csv:
        headers["X-API-Keys"] = keys_csv

    params: list[tuple[str, str]] = []
    for s in seeds:
        params.append(("seed", str(int(s))))

    url = f"{base}/task_info"
    try:
        r = _requests.get(url, headers=headers, params=params or None, timeout=30)
    except Exception as exc:
        raise click.ClickException(f"Request failed: {exc}") from exc
    if not (200 <= r.status_code < 300):
        ct = r.headers.get("content-type", "")
        detail = r.text
        if ct.startswith("application/json"):
            with contextlib.suppress(Exception):
                detail = _json.dumps(r.json(), indent=2)
        raise click.ClickException(f"{url} returned {r.status_code}:\n{detail}")

    data = (
        r.json()
        if r.headers.get("content-type", "").startswith("application/json")
        else {"raw": r.text}
    )
    click.echo(_json.dumps(data, indent=2, sort_keys=True))


@task_app_group.command("serve")
@click.argument("app_id", type=str, required=False)
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", default=None, type=int, help="Port to serve on (default: 8001)")
@click.option("--env-file", multiple=True, type=click.Path(), help="Extra .env files to load")
@click.option(
    "--reload/--no-reload", "reload_flag", default=False, help="Enable uvicorn auto-reload"
)
@click.option(
    "--force/--no-force",
    "force",
    default=False,
    help="Kill any process already bound to the selected port before starting",
)
@click.option(
    "--trace",
    "trace_dir",
    type=click.Path(),
    default=None,
    help="Enable tracing and write SFT JSONL files to this directory (default: traces/v3)",
)
@click.option(
    "--trace-db",
    "trace_db",
    type=click.Path(),
    default=None,
    help="Override local trace DB path (default: traces/v3/synth_ai.db)",
)
def serve_task_group(
    app_id: str | None,
    host: str,
    port: int | None,
    env_file: Sequence[str],
    reload_flag: bool,
    force: bool,
    trace_dir: str | None,
    trace_db: str | None,
) -> None:
    # Change to demo directory if stored (REQUIRED for demo isolation)
    from synth_ai.demos.demo_task_apps.core import load_demo_dir

    demo_dir = load_demo_dir()
    if demo_dir:
        demo_path = Path(demo_dir)
        if not demo_path.is_dir():
            raise click.ClickException(
                f"Demo directory not found: {demo_dir}\nRun 'synth-ai setup' to create a demo."
            )
        os.chdir(demo_dir)
        click.echo(f"Using demo directory: {demo_dir}\n")
        # Store demo directory for path resolution
        os.environ["SYNTH_DEMO_DIR"] = str(demo_path.resolve())

    # Prompt for port if not provided
    if port is None:
        port = click.prompt("Port to serve on", type=int, default=8001)

    # Prompt for trace directory if not provided
    if trace_dir is None:
        click.echo(
            "\nTracing captures rollout data (actions, rewards, model outputs) to a local SQLite DB."
        )
        click.echo("This data can be exported to JSONL for supervised fine-tuning (SFT).")
        enable_tracing = click.confirm("Enable tracing?", default=True)
        if enable_tracing:
            demo_base = Path(os.environ.get("SYNTH_DEMO_DIR") or Path.cwd())
            default_trace_dir = str((demo_base / "traces/v3").resolve())
            trace_dir = click.prompt(
                "Trace directory", type=str, default=default_trace_dir, show_default=True
            )
        else:
            trace_dir = None

    # Prompt for trace DB if not provided and tracing is enabled
    if trace_dir and trace_db is None:
        demo_base = Path(os.environ.get("SYNTH_DEMO_DIR") or Path.cwd())
        default_trace_db = str((demo_base / "traces/v3/synth_ai.db").resolve())
        trace_db = click.prompt(
            "Trace DB path", type=str, default=default_trace_db, show_default=True
        )

    choice = _select_app_choice(app_id, purpose="serve")
    entry = choice.ensure_entry()
    _serve_entry(
        entry, host, port, env_file, reload_flag, force, trace_dir=trace_dir, trace_db=trace_db
    )


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
    cwd_env_files = sorted(cwd.glob("**/*.env"))
    env_candidates.extend(cwd_env_files)

    # Add repo root env files
    repo_env_files = sorted(REPO_ROOT.glob("**/*.env"))
    # Avoid duplicates
    for repo_file in repo_env_files:
        if repo_file not in env_candidates:
            env_candidates.append(repo_file)

    if not env_candidates:
        raise click.ClickException("No env file found. Pass --env-file explicitly.")

    click.echo("Select env file to load:")
    for idx, path in enumerate(env_candidates, start=1):
        click.echo(f"  {idx}) {path.resolve()}")
    choice = click.prompt("Enter choice", type=click.IntRange(1, len(env_candidates)), default=1)
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
        out = subprocess.run(
            ["lsof", "-ti", f"TCP:{port}"], capture_output=True, text=True, check=False
        )
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
            raise click.ClickException(f"Failed to terminate PID {pid}: {exc}") from exc

    time.sleep(0.5)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        still_in_use = s.connect_ex((host, port)) == 0

    if still_in_use:
        for pid in pids:
            try:
                os.kill(int(pid), signal.SIGKILL)
            except Exception as exc:
                raise click.ClickException(f"Failed to force terminate PID {pid}: {exc}") from exc
        time.sleep(0.5)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        in_use_after = s.connect_ex((host, port)) == 0
    if in_use_after:
        raise click.ClickException(
            f"Port {port} is still in use after attempting to terminate processes."
        )


def _save_to_env_file(env_path: Path, key: str, value: str) -> None:
    """Save or update a key-value pair in the .env file."""
    try:
        # Read existing .env
        existing_lines = []
        if env_path.exists():
            existing_lines = env_path.read_text().splitlines()
        else:
            env_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if key already exists and update it
        key_updated = False
        new_lines = []
        for line in existing_lines:
            if line.strip().startswith(f"{key}="):
                new_lines.append(f"{key}={value}")
                key_updated = True
            else:
                new_lines.append(line)

        if key_updated:
            # Write updated lines back
            env_path.write_text("\n".join(new_lines) + "\n")
            click.echo(f"Updated {key} in {env_path}")
        else:
            # Append to .env
            with open(env_path, "a") as f:
                if existing_lines and not existing_lines[-1].strip():
                    # File exists and last line is not empty
                    pass
                elif existing_lines:
                    # Add newline before appending
                    f.write("\n")
                f.write(f"{key}={value}\n")
        click.echo(f"Saved {key} to {env_path}")
    except Exception as e:
        click.echo(f"Warning: Could not save {key} to .env: {e}", err=True)


def _persist_env_api_key(env_api_key: str, env_paths: Sequence[Path] | None) -> None:
    """Persist ENVIRONMENT_API_KEY to provided env files (or default .env)."""
    targets: list[Path] = []
    seen: set[Path] = set()
    for path in env_paths or ():
        try:
            resolved = Path(path).resolve()
        except Exception:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        targets.append(resolved)

    if not targets:
        demo_dir = Path(os.environ.get("SYNTH_DEMO_DIR") or Path.cwd())
        targets.append((demo_dir / ".env").resolve())

    for target in targets:
        _save_to_env_file(target, "ENVIRONMENT_API_KEY", env_api_key)


def _validate_required_env_keys() -> None:
    """Validate required environment keys are set, prompting if missing."""
    # Use demo directory .env file if set, otherwise current directory
    demo_base = Path(os.environ.get("SYNTH_DEMO_DIR") or Path.cwd())
    env_file = demo_base / ".env"

    if env_file.exists():
        try:
            from dotenv import load_dotenv

            load_dotenv(env_file, override=False)
        except Exception:
            pass  # Best effort

    env_api_key = os.environ.get("ENVIRONMENT_API_KEY", "").strip()

    if not env_api_key:
        env_api_key = input("Please enter your RL Environment API key:\n> ").strip()
        if not env_api_key:
            raise click.ClickException("RL Environment API key is required to start the server")
        os.environ["ENVIRONMENT_API_KEY"] = env_api_key
        _save_to_env_file(env_file, "ENVIRONMENT_API_KEY", env_api_key)

    # Check for Groq API key
    groq_api_key = os.environ.get("GROQ_API_KEY", "").strip()

    if not groq_api_key:
        click.echo("\nInference API key configuration:")
        click.echo("This workflow requires a Groq API key.")
        groq_api_key = input("Groq API key (or press Enter to skip): ").strip()
        if groq_api_key:
            os.environ["GROQ_API_KEY"] = groq_api_key
            _save_to_env_file(env_file, "GROQ_API_KEY", groq_api_key)


def _print_demo_next_steps_if_applicable() -> None:
    """Print next steps if currently in a demo directory."""
    try:
        from synth_ai.demos.demo_task_apps.core import load_demo_dir

        cwd = Path.cwd().resolve()
        demo_dir = load_demo_dir()

        # Check if we're in the demo directory
        if (
            demo_dir
            and Path(demo_dir).resolve() == cwd
            and (cwd / "run_local_rollout_traced.py").exists()
        ):
            click.echo("\n" + "=" * 60)
            click.echo("Next step: Collect traced rollouts")
            click.echo("=" * 60)
            click.echo("\nIn another terminal, run:")
            click.echo(f"  cd {cwd}")
            click.echo("  uv run python run_local_rollout_traced.py")
            click.echo("\nRun this 5-10 times to collect diverse traces.")
            click.echo("=" * 60 + "\n")
    except Exception:
        # Silently fail - this is just a helpful printout
        pass


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
        os.environ["TASKAPP_TRACING_ENABLED"] = "1"

        # Ensure paths are absolute relative to demo directory
        demo_base = Path(os.environ.get("SYNTH_DEMO_DIR") or Path.cwd())

        if trace_dir is not None:
            dir_path = Path(trace_dir).expanduser()
            if not dir_path.is_absolute():
                dir_path = (demo_base / dir_path).resolve()
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                raise click.ClickException(
                    f"Failed to create trace directory {dir_path}: {exc}"
                ) from exc
            os.environ["TASKAPP_SFT_OUTPUT_DIR"] = str(dir_path)
            click.echo(f"Tracing enabled. SFT JSONL will be written to {dir_path}")
        if trace_db is not None:
            db_path = Path(trace_db).expanduser()
            if not db_path.is_absolute():
                db_path = (demo_base / db_path).resolve()
            # Construct the sqlite URL from the absolute path
            db_url = f"sqlite+aiosqlite:///{db_path}"
            os.environ["SQLD_DB_PATH"] = str(db_path)
            os.environ["TURSO_LOCAL_DB_URL"] = db_url
            click.echo(f"Tracing DB path set to {db_path}")
        from synth_ai.tracing_v3.config import CONFIG as TRACE_CONFIG

        # Use the explicitly set URL if available
        new_db_url = os.getenv("TURSO_LOCAL_DB_URL") or TRACE_CONFIG.db_url
        TRACE_CONFIG.db_url = new_db_url
        if new_db_url:
            click.echo(f"Tracing DB URL resolved to {new_db_url}")
    elif os.getenv("TASKAPP_TRACING_ENABLED"):
        click.echo("Tracing enabled via environment variables")

    _ensure_port_free(port, host, force=force)

    _validate_required_env_keys()
    env_path_objs = [Path(p) for p in env_files if p]
    _preflight_env_key(env_path_objs)

    # Print next steps if in demo context
    if trace_enabled:
        _print_demo_next_steps_if_applicable()

    run_task_app(
        entry.config_factory,
        host=host,
        port=port,
        reload=reload_flag,
        env_files=env_files,
    )


@task_app_group.command("deploy")
@click.argument("app_id", type=str, required=False)
@click.option("--name", "modal_name", default=None, help="Override Modal app name")
@click.option("--dry-run", is_flag=True, help="Print modal deploy command without executing")
@click.option("--modal-cli", default="modal", help="Path to modal CLI executable")
@click.option(
    "--env-file",
    multiple=True,
    type=click.Path(),
    help="Env file to load into the container (can be repeated)",
)
def deploy_app(
    app_id: str | None,
    modal_name: str | None,
    dry_run: bool,
    modal_cli: str,
    env_file: Sequence[str],
) -> None:
    """Deploy a task app to Modal."""

    # Change to demo directory if stored (for consistent discovery)
    from synth_ai.demos.demo_task_apps.core import load_demo_dir

    demo_dir = load_demo_dir()
    if demo_dir:
        demo_path = Path(demo_dir)
        if not demo_path.is_dir():
            raise click.ClickException(
                f"Demo directory not found: {demo_dir}\nRun 'synth-ai demo' to create a demo."
            )
        os.chdir(demo_dir)
        click.echo(f"Using demo directory: {demo_dir}\n")

    choice = _select_app_choice(app_id, purpose="deploy")

    if choice.modal_script:
        env_paths = _resolve_env_paths_for_script(choice.modal_script, env_file)
        click.echo("Using env file(s): " + ", ".join(str(p.resolve()) for p in env_paths))
        _run_modal_script(
            choice.modal_script,
            modal_cli,
            "deploy",
            env_paths,
            modal_name=modal_name,
            dry_run=dry_run,
        )
        return

    entry = choice.ensure_entry()
    _deploy_entry(entry, modal_name, dry_run, modal_cli, env_file, original_path=choice.path)


@task_app_group.command("modal-serve")
@click.argument("app_id", type=str, required=False)
@click.option("--modal-cli", default="modal", help="Path to modal CLI executable")
@click.option("--name", "modal_name", default=None, help="Override Modal app name (optional)")
@click.option(
    "--env-file",
    multiple=True,
    type=click.Path(),
    help="Env file to load into the container (can be repeated)",
)
def modal_serve_app(
    app_id: str | None, modal_cli: str, modal_name: str | None, env_file: Sequence[str]
) -> None:
    choice = _select_app_choice(app_id, purpose="modal-serve")

    if choice.modal_script:
        env_paths = _resolve_env_paths_for_script(choice.modal_script, env_file)
        click.echo("Using env file(s): " + ", ".join(str(p.resolve()) for p in env_paths))
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
    inline_secret_values: dict[str, str] | None = None,
) -> Path:
    modal_name = override_name or modal_cfg.app_name

    # For dynamically discovered apps, import the module by its package path
    # Compute the module name relative to the mounted repo root (/opt/synth_ai_repo)
    remote_file_str: str | None = None
    if original_path:
        try:
            # Build lookup of local->remote mounts
            mount_map: list[tuple[Path, Path]] = [
                (Path(local).resolve(), Path(remote))
                for (local, remote) in modal_cfg.extra_local_dirs
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
        guaranteed_file_str = str(
            (Path("/opt/synth_ai_repo/__local_task_app__") / Path(original_path).stem).with_suffix(
                ".py"
            )
        )

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
    inline_secret_values = {k: v for k, v in (inline_secret_values or {}).items() if v}

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
INLINE_SECRET_VALUES = {inline_secret_values!r}

image = Image.debian_slim(python_version={modal_cfg.python_version!r})

pip_packages = {pip_packages!r}
if pip_packages:
    image = image.pip_install(*pip_packages)

local_dirs = {local_dirs!r}

def _copy_tree_filtered(src_dir: str) -> str:
    src = _Path(src_dir)
    temp_dir = _Path(tempfile.mkdtemp(prefix='synth_mount_'))

    exclude_dirs = {".cache", ".git", "__pycache__"}
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

if INLINE_SECRET_VALUES:
    secret_objs.append(Secret.from_dict(INLINE_SECRET_VALUES))

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

    with tempfile.NamedTemporaryFile("w", suffix=f"_{entry.app_id}_modal.py", delete=False) as tmp:
        tmp.write(script)
        tmp.flush()
        name = tmp.name
    return Path(name)


def register(cli: click.Group) -> None:
    cli.add_command(serve_command)
    cli.add_command(task_app_group)
    cli.add_command(eval_command)


@click.command("eval")
@click.argument("app_id", type=str, required=False)
@click.option("--config", type=click.Path(), default=None, help="Path to eval TOML (short schema)")
@click.option(
    "--url",
    "task_app_url",
    type=str,
    default=None,
    help="Base URL of a running task app (skip in-process server)",
)
@click.option("--seeds", default="0,1,2,3,4", help="Comma-separated seeds/indices to evaluate")
@click.option("--split", default="train", show_default=True, help="Dataset split to use")
@click.option("--model", default=None, help="Model identifier (prompted if omitted)")
@click.option("--env-file", multiple=True, type=click.Path(), help="Env file(s) for keys")
def eval_command(
    app_id: str | None,
    config: str | None,
    task_app_url: str | None,
    seeds: str,
    split: str,
    model: str | None,
    env_file: Sequence[str],
) -> None:
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
            raise click.ClickException(
                "TOML parser not available; use Python 3.11+ or install tomli"
            )
        if not config_path.exists():
            raise click.ClickException(f"Eval config not found: {config_path}")
        try:
            data = config_path.read_bytes()
            parsed = _toml.loads(data.decode("utf-8"))
            if isinstance(parsed, dict):
                section = parsed.get("eval")
                cfg = dict(section) if isinstance(section, dict) else dict(parsed)
        except Exception as exc:
            raise click.ClickException(f"Failed to parse TOML '{config_path}': {exc}") from exc

    app_id = app_id or (cfg.get("app_id") if isinstance(cfg.get("app_id"), str) else None)  # type: ignore

    # Determine selection params (CLI takes precedence; TOML only fills unset model/seeds/env)
    if cfg.get("model") and not model:
        model = str(cfg["model"])  # type: ignore[index]
    if cfg.get("seeds") and seeds == "0,1,2,3,4":
        val = cfg["seeds"]
        if isinstance(val, list):
            with contextlib.suppress(Exception):
                seeds = ",".join(str(int(x)) for x in val)
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

    click.echo("Using env file(s): " + ", ".join(str(p) for p in env_paths))
    _load_env_files_into_process([str(Path(p)) for p in env_paths])

    if task_app_url is None:
        config = entry.config_factory()  # type: ignore[union-attr]
        # Help the type checker; runtime check also enforced in server.run_task_app
        if not isinstance(config, TaskAppConfig):
            raise click.ClickException(
                "Invalid task app: config_factory did not return TaskAppConfig"
            )
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
            raise click.ClickException(
                "No supported models; supply --model or add base_task_info.inference.models"
            )
        click.echo("Select model to evaluate:")
        for idx, m in enumerate(supported, start=1):
            click.echo(f"  {idx}) {m}")
        choice_idx = click.prompt("Enter choice", type=click.IntRange(1, len(supported)))
        selected_model = supported[choice_idx - 1]

    try:
        seed_values = [int(s.strip()) for s in seeds.split(",") if s.strip()]
    except Exception as exc:
        raise click.ClickException("Invalid --seeds; expected comma-separated integers") from exc

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
        client = httpx.Client(
            transport=cast(Any, transport),
            base_url="http://eval.local",
            timeout=60.0,
            headers=headers,
        )
    else:
        client = httpx.Client(base_url=task_app_url, timeout=60.0, headers=headers)
    try:
        with contextlib.suppress(Exception):
            client.get("/task_info")
        # Precompute optional policy overrides from TOML
        policy_overrides: dict[str, Any] = {}
        try:
            # Accept [eval.policy] table or top-level keys for convenience
            if isinstance(cfg.get("policy"), dict):
                policy_overrides.update(dict(cfg["policy"]))
            # Back-compat: allow temperature/max_tokens at top level
            for k in (
                "temperature",
                "max_tokens",
                "reasoning_effort",
                "system_hint",
                "tool_choice",
            ):
                if k in cfg and k not in policy_overrides:
                    policy_overrides[k] = cfg.get(k)
        except Exception:
            policy_overrides = {}

        for seed_val in seed_values:
            body = {
                "run_id": str(uuid.uuid4()),
                "env": {"config": {"split": split, "index": seed_val}, "seed": seed_val},
                "policy": {
                    "policy_name": selected_model,
                    "config": {"model": selected_model, **policy_overrides},
                },
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
                    trajs = (
                        data.get("trajectories")
                        if isinstance(data.get("trajectories"), list)
                        else None
                    )
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
                    with contextlib.suppress(Exception):
                        click.echo(json.dumps(data, indent=2))
                else:
                    click.echo(" ".join(summary))
            except Exception as exc:
                failures += 1
                click.echo(f"seed={seed_val} error={exc}")

    finally:
        try:
            client.close()
        except AttributeError:
            transport_obj = getattr(client, "_transport", None)
            if transport_obj and hasattr(transport_obj, "aclose"):
                try:
                    asyncio.run(transport_obj.aclose())
                except RuntimeError:
                    # Fallback when already inside a running loop (rare for CLI).
                    new_loop = asyncio.new_event_loop()
                    try:
                        new_loop.run_until_complete(transport_obj.aclose())
                    finally:
                        new_loop.close()
        except Exception:
            pass

    click.echo(
        f"Eval complete: {successes} ok, {failures} failed; model={selected_model}, split={split}"
    )
    # Print outcome summary if any successes
    if outcome_count > 0:
        mean_outcome = outcome_sum / float(outcome_count)
        frac_right = outcome_correct / float(outcome_count)
        click.echo(
            f"Outcome summary: correct={outcome_correct}/{outcome_count} ({frac_right:.2%}), mean_outcome={mean_outcome:.3f}"
        )


def register_eval(cli: click.Group) -> None:
    cli.add_command(eval_command)

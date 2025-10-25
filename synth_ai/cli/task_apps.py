from __future__ import annotations

import argparse
import ast
import asyncio
import contextlib
import functools
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
import shlex
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
import textwrap
import time
import types
import uuid
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, cast

try:  # Python 3.11+
    import tomllib as _toml
except Exception:  # pragma: no cover - fallback
    _toml = None  # type: ignore

import click
from click.exceptions import Abort

# Tracing imports - make conditional for optional dependencies
try:
    from synth_ai.tracing_v3 import (  # type: ignore[import-untyped]
        BaseEvent,
        EnvironmentEvent,
        RuntimeEvent,
        SessionEventMarkovBlanketMessage,
        SessionMessageContent,
        SessionTimeStep,
        SessionTracer,
        TimeRecord,
    )
    from synth_ai.tracing_v3 import (  # type: ignore[import-untyped]
        SessionTrace as V3SessionTrace,
    )
    _TRACING_AVAILABLE = True
except (ImportError, ModuleNotFoundError, TypeError):
    # Tracing system not available (missing optional dependencies)
    BaseEvent = EnvironmentEvent = RuntimeEvent = None  # type: ignore
    SessionEventMarkovBlanketMessage = SessionMessageContent = None  # type: ignore
    SessionTimeStep = SessionTracer = TimeRecord = None  # type: ignore
    V3SessionTrace = None  # type: ignore
    _TRACING_AVAILABLE = False

# ---------------------------------------------------------------------------
# Dynamic imports to avoid hard dependencies during type checking.
# ---------------------------------------------------------------------------
ModalDeploymentConfigType = TaskAppConfigType = TaskAppEntryType = Any

try:  # Resolve base URL defaults lazily
    _config_module = cast(
        Any, importlib.import_module("synth_ai.config.base_url")
    )
    PROD_BASE_URL_DEFAULT = cast(str, _config_module.PROD_BASE_URL_DEFAULT)
except Exception:  # pragma: no cover - fallback
    PROD_BASE_URL_DEFAULT = "https://agent-learning.onrender.com"

try:
    _task_apps_module = cast(Any, importlib.import_module("synth_ai.task.apps"))
    ModalDeploymentConfig = cast(
        type[ModalDeploymentConfigType], _task_apps_module.ModalDeploymentConfig
    )
    TaskAppConfig = cast(type[TaskAppConfigType], _task_apps_module.TaskAppConfig)
    TaskAppEntry = cast(type[TaskAppEntryType], _task_apps_module.TaskAppEntry)
    registry = _task_apps_module.registry
except Exception as exc:  # pragma: no cover - critical dependency
    raise RuntimeError("Unable to load task app registry") from exc

try:
    _task_server_module = cast(Any, importlib.import_module("synth_ai.task.server"))
    create_task_app = cast(Callable[..., Any], _task_server_module.create_task_app)
    run_task_app = cast(Callable[..., Any], _task_server_module.run_task_app)
except Exception as exc:  # pragma: no cover - critical dependency
    raise RuntimeError("Unable to load task app server utilities") from exc


def _load_demo_directory() -> Optional[Path]:
    """Return the demo task apps directory if available."""

    try:
        module = cast(
            Any, importlib.import_module("synth_ai.demos.demo_task_apps.core")
        )
        loader = cast(Callable[[], Optional[str | Path]], module.load_demo_dir)
        demo_dir = loader()
        if isinstance(demo_dir, str | Path):
            demo_path = Path(demo_dir)
            if demo_path.exists():
                return demo_path.resolve()
    except Exception:
        return None
    return None


def _maybe_import(name: str) -> Any:
    """Safely import a module by name and return it, or None on failure."""

    try:
        return importlib.import_module(name)
    except Exception:
        return None

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


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = 0.0
    denom_x = 0.0
    denom_y = 0.0
    for x, y in zip(xs, ys, strict=False):
        dx = x - mean_x
        dy = y - mean_y
        num += dx * dy
        denom_x += dx * dx
        denom_y += dy * dy
    if denom_x <= 0 or denom_y <= 0:
        return None
    return num / (denom_x ** 0.5 * denom_y ** 0.5)


@dataclass
class AppChoice:
    app_id: str
    label: str
    path: Path
    source: str
    description: Optional[str] = None
    aliases: tuple[str, ...] = ()
    entry: TaskAppEntryType | None = None
    entry_loader: Callable[[], TaskAppEntryType] | None = None
    modal_script: Path | None = None
    lineno: int | None = None

    def ensure_entry(self) -> TaskAppEntryType:
        if self.entry is not None:
            return self.entry
        if self.entry_loader is None:
            raise click.ClickException(f"Unable to load task app '{self.app_id}' from {self.path}")
        entry = self.entry_loader()
        self.entry = entry
        return entry


@dataclass
class JudgeSpec:
    name: str
    fn: Callable[..., Any]
    kwargs: dict[str, Any]


def _parse_datetime_for_trace(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        value = value.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(value)
        except ValueError:
            try:
                dt = datetime.fromtimestamp(float(value), tz=timezone.utc)
            except Exception:
                return None
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    if isinstance(value, int | float):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    return None


def _time_record_from_dict(payload: dict[str, Any] | None) -> TimeRecord:
    payload = payload or {}
    event_time = payload.get("event_time")
    if not isinstance(event_time, int | float):
        try:
            event_time = float(event_time)
        except Exception:
            event_time = float(time.time())
    message_time = payload.get("message_time")
    if message_time is not None:
        try:
            message_time = int(message_time)
        except Exception:
            message_time = None
    return TimeRecord(event_time=event_time, message_time=message_time)


def _event_from_dict(payload: dict[str, Any]) -> BaseEvent:
    base_kwargs = {
        "system_instance_id": payload.get("system_instance_id", ""),
        "time_record": _time_record_from_dict(payload.get("time_record")),
        "metadata": payload.get("metadata") or {},
        "event_metadata": payload.get("event_metadata"),
    }
    if "actions" in payload:
        return RuntimeEvent(actions=payload.get("actions") or [], **base_kwargs)
    if any(key in payload for key in ("reward", "terminated", "truncated")):
        return EnvironmentEvent(
            reward=float(payload.get("reward", 0.0) or 0.0),
            terminated=bool(payload.get("terminated", False)),
            truncated=bool(payload.get("truncated", False)),
            system_state_before=payload.get("system_state_before"),
            system_state_after=payload.get("system_state_after"),
            **base_kwargs,
        )
    return BaseEvent(**base_kwargs)


def _markov_message_from_dict(payload: dict[str, Any]) -> SessionEventMarkovBlanketMessage:
    content_payload = payload.get("content") or {}
    content = SessionMessageContent(
        text=content_payload.get("text"),
        json_payload=content_payload.get("json_payload"),
    )
    raw_type = (payload.get("message_type") or "").lower()
    if raw_type == "observation":
        normalized_type = "system"
    elif raw_type == "action":
        normalized_type = "assistant"
    elif raw_type in {"user", "assistant", "system", "tool_use", "tool_result"}:
        normalized_type = raw_type
    else:
        normalized_type = "system"

    return SessionEventMarkovBlanketMessage(
        content=content,
        message_type=normalized_type,
        time_record=_time_record_from_dict(payload.get("time_record")),
        metadata=payload.get("metadata") or {},
    )


def _step_from_dict(payload: dict[str, Any]) -> SessionTimeStep:
    events = [
        _event_from_dict(event)
        for event in payload.get("events", [])
        if isinstance(event, dict)
    ]
    messages = [
        _markov_message_from_dict(msg)
        for msg in payload.get("markov_blanket_messages", [])
        if isinstance(msg, dict)
    ]
    timestamp = _parse_datetime_for_trace(payload.get("timestamp")) or datetime.now(timezone.utc)
    completed_at = _parse_datetime_for_trace(payload.get("completed_at"))
    return SessionTimeStep(
        step_id=payload.get("step_id", ""),
        step_index=int(payload.get("step_index", 0) or 0),
        timestamp=timestamp,
        turn_number=payload.get("turn_number"),
        events=events,
        markov_blanket_messages=messages,
        step_metadata=payload.get("step_metadata") or {},
        completed_at=completed_at,
    )


def _session_trace_from_dict(payload: dict[str, Any]) -> Optional[V3SessionTrace]:
    if not isinstance(payload, dict):
        return None
    steps = [
        _step_from_dict(step)
        for step in payload.get("session_time_steps", [])
        if isinstance(step, dict)
    ]
    events = [
        _event_from_dict(event)
        for event in payload.get("event_history", [])
        if isinstance(event, dict)
    ]
    markov_history = [
        _markov_message_from_dict(msg)
        for msg in payload.get("markov_blanket_message_history", [])
        if isinstance(msg, dict)
    ]
    created_at = _parse_datetime_for_trace(payload.get("created_at")) or datetime.now(timezone.utc)
    metadata = payload.get("metadata") or {}
    session_metadata = payload.get("session_metadata")
    return V3SessionTrace(
        session_id=payload.get("session_id", ""),
        created_at=created_at,
        session_time_steps=steps,
        event_history=events,
        markov_blanket_message_history=markov_history,
        metadata=metadata,
        session_metadata=session_metadata,
    )


async def _store_trace(
    tracer: SessionTracer | None,
    trace_namespace: dict[str, Any] | None,
    extra_metadata: dict[str, Any] | None = None,
):
    import logging
    _logger = logging.getLogger(__name__)
    
    _logger.info(f"[STORE_TRACE_DEBUG] Called with tracer={tracer is not None}, trace_namespace={trace_namespace is not None}")
    
    if tracer is None or not isinstance(trace_namespace, dict):
        _logger.warning(f"[STORE_TRACE_DEBUG] Early return: tracer={tracer is not None}, trace_namespace type={type(trace_namespace)}")
        return
    
    _logger.info(f"[STORE_TRACE_DEBUG] trace_namespace keys: {list(trace_namespace.keys())}")
    
    session_payload = trace_namespace.get("session_trace")
    if not isinstance(session_payload, dict):
        _logger.warning(f"[STORE_TRACE_DEBUG] No session_trace found or wrong type: {type(session_payload)}")
        return
    
    _logger.info(f"[STORE_TRACE_DEBUG] session_payload keys: {list(session_payload.keys())}")
    msg_count = len(session_payload.get("markov_blanket_message_history", []))
    _logger.info(f"[STORE_TRACE_DEBUG] Found {msg_count} messages in session_payload")
    
    trace_obj = _session_trace_from_dict(session_payload)
    if trace_obj is None:
        _logger.warning(f"[STORE_TRACE_DEBUG] _session_trace_from_dict returned None")
        return
    
    _logger.info(f"[STORE_TRACE_DEBUG] Created SessionTrace object with {len(trace_obj.markov_blanket_message_history)} messages")
    
    if tracer.db is None:
        await tracer.initialize()
    meta = dict(trace_obj.metadata or {})
    if extra_metadata:
        meta.update(extra_metadata)
    trace_obj.metadata = meta
    
    _logger.info(f"[STORE_TRACE_DEBUG] Calling insert_session_trace for session_id={trace_obj.session_id}")
    await tracer.db.insert_session_trace(trace_obj)
    _logger.info(f"[STORE_TRACE_DEBUG] Successfully inserted trace")

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

    demo_path = _load_demo_directory()
    if demo_path is not None and demo_path.is_dir():
        roots.append(demo_path)

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
        _maybe_import("synth_ai.demos.demo_task_apps")
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
    demo_dir = _load_demo_directory()
    if demo_dir and choice.path.is_relative_to(demo_dir):
        demo_rank = 0

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


def _extract_modal_config_from_file(path: Path) -> ModalDeploymentConfigType | None:
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


def _build_modal_config_from_ast(modal_call: ast.Call) -> ModalDeploymentConfigType | None:
    """Build a ModalDeploymentConfig from an AST Call node."""
    try:
        # Extract keyword arguments
        kwargs = {}
        for kw in modal_call.keywords:
            if kw.arg and isinstance(kw.value, ast.Constant):
                kwargs[kw.arg] = kw.value.value
            elif kw.arg == "pip_packages" and isinstance(kw.value, (ast.List, ast.Tuple)):
                # Handle pip_packages list/tuple
                packages: list[str] = []
                value_node = kw.value
                if isinstance(value_node, (ast.List, ast.Tuple)):
                    for elt in value_node.elts:
                        if isinstance(elt, ast.Constant):
                            packages.append(elt.value)
                kwargs[kw.arg] = tuple(packages)
            elif kw.arg == "extra_local_dirs" and isinstance(kw.value, (ast.List, ast.Tuple)):
                # Handle extra_local_dirs list/tuple of tuples
                dirs = []
                value_node = kw.value
                if isinstance(value_node, (ast.List, ast.Tuple)):
                    for elt in value_node.elts:
                        if isinstance(elt, (ast.List, ast.Tuple)) and len(elt.elts) == 2:
                            src = elt.elts[0].value if isinstance(elt.elts[0], ast.Constant) else None
                            dst = elt.elts[1].value if isinstance(elt.elts[1], ast.Constant) else None
                            if src and dst:
                                dirs.append((src, dst))
                kwargs[kw.arg] = tuple(dirs)
            elif kw.arg == "secret_names" and isinstance(kw.value, (ast.List, ast.Tuple)):
                # Handle secret_names list/tuple
                secrets = []
                value_node = kw.value
                if isinstance(value_node, (ast.List, ast.Tuple)):
                    for elt in value_node.elts:
                        if isinstance(elt, ast.Constant):
                            secrets.append(elt.value)
                kwargs[kw.arg] = tuple(secrets)
            elif kw.arg == "volume_mounts" and isinstance(kw.value, (ast.List, ast.Tuple)):
                # Handle volume_mounts list/tuple of tuples
                mounts = []
                value_node = kw.value
                if isinstance(value_node, (ast.List, ast.Tuple)):
                    for elt in value_node.elts:
                        if isinstance(elt, (ast.List, ast.Tuple)) and len(elt.elts) == 2:
                            name = elt.elts[0].value if isinstance(elt.elts[0], ast.Constant) else None
                            mount = elt.elts[1].value if isinstance(elt.elts[1], ast.Constant) else None
                            if name and mount:
                                mounts.append((name, mount))
                kwargs[kw.arg] = tuple(mounts)

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
    except (Abort, EOFError, KeyboardInterrupt) as exc:
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
        exact_matches = [c for c in matches if c.app_id == app_id]
        if len(exact_matches) == 1:
            return exact_matches[0]
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


@contextlib.contextmanager
def _safe_import_context() -> Iterator[None]:
    """Guard module imports against argparse/uvicorn side effects."""

    original_argv = sys.argv[:]
    sys.argv = [original_argv[0]] if original_argv else ["python"]

    parser_cls = argparse.ArgumentParser
    old_parse_args = parser_cls.parse_args

    def _parse_noargs(self, args=None, namespace=None):  # type: ignore[override]
        if args is None:
            args = []
        if namespace is None:
            namespace = argparse.Namespace()
        try:
            return old_parse_args(self, args, namespace)
        except SystemExit:
            return namespace

    parser_cls.parse_args = _parse_noargs  # type: ignore[assignment]

    uvicorn_run = None
    run_task_app_orig = None
    try:
        import uvicorn  # type: ignore

        uvicorn_run = uvicorn.run
        uvicorn.run = lambda *args, **kwargs: None  # type: ignore[assignment]
    except Exception:
        uvicorn_run = None

    try:
        _task_server_patch = cast(
            Any, importlib.import_module("synth_ai.task.server")
        )
        run_task_app_orig = cast(Callable[..., Any], _task_server_patch.run_task_app)
        _task_server_patch.run_task_app = (  # type: ignore[assignment]
            lambda *args, **kwargs: None
        )
    except Exception:
        run_task_app_orig = None

    try:
        yield
    finally:
        sys.argv = original_argv
        parser_cls.parse_args = old_parse_args  # type: ignore[assignment]
        if uvicorn_run is not None:
            try:
                import uvicorn  # type: ignore

                uvicorn.run = uvicorn_run  # type: ignore[assignment]
            except Exception:
                pass
        if run_task_app_orig is not None:
            try:
                _task_server_patch = cast(
                    Any, importlib.import_module("synth_ai.task.server")
                )
                _task_server_patch.run_task_app = run_task_app_orig  # type: ignore[assignment]
            except Exception:
                pass


def _load_entry_from_path(
    path: Path, app_id: str, module_search_roots: Sequence[Path] | None = None
) -> TaskAppEntryType:
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
            with _safe_import_context():
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
            with _safe_import_context():
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

    config_obj: TaskAppConfigType | None = None
    factory_callable: Callable[[], TaskAppConfigType] | None = None

    for attr_name in dir(module):
        try:
            attr = getattr(module, attr_name)
        except Exception:
            continue
        if isinstance(attr, TaskAppConfig) and attr.app_id == app_id:
            config_obj = attr

            def _return_config(cfg: TaskAppConfigType = attr) -> TaskAppConfigType:
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
                with _safe_import_context():
                    result = attr()
            except SystemExit:
                continue
            except Exception:
                continue
            if isinstance(result, TaskAppConfig) and result.app_id == app_id:
                # Bind attr to a local and close over it without exposing parameters
                bound_func: Callable[[], TaskAppConfig] = cast(Callable[[], TaskAppConfig], attr)  # type: ignore[assignment]

                def _factory_noargs(
                    func: Callable[[], TaskAppConfigType] = bound_func,
                ) -> TaskAppConfigType:
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

    modal_cfg: ModalDeploymentConfigType | None = None
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


def _path_is_within(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


@functools.lru_cache(maxsize=16)
def _is_modal_shim(path_str: str) -> bool:
    """Return True if the candidate CLI path refers to the synth-ai shim."""

    path = Path(path_str)
    try:
        resolved = path.resolve(strict=True)
    except Exception:
        resolved = path

    if not resolved.exists() or resolved.is_dir():
        return False

    snippet = ""
    try:
        snippet = resolved.read_bytes()[:4096].decode("utf-8", errors="ignore")
    except Exception:
        snippet = ""

    shim_markers = (
        "synth_ai.cli._modal_wrapper",
        "from modal.__main__ import main",
        "import modal.__main__",
        "run_module('modal.__main__'",
    )
    if snippet and any(marker in snippet for marker in shim_markers):
        return True

    try:
        size = resolved.stat().st_size
    except Exception:
        size = None

    if (
        size is not None
        and size < 2048
        and "python" in (snippet.splitlines() or [""])[0]
        and (
            "modal.__main__" in snippet
            or "modal.__main__" in snippet.replace(" ", "")
        )
    ):
        return True

    virtual_env = os.environ.get("VIRTUAL_ENV")
    if virtual_env and _path_is_within(resolved, Path(virtual_env)):
        return True

    if _path_is_within(resolved, REPO_ROOT):
        return True

    uv_tools_dir = Path.home() / ".local" / "share" / "uv" / "tools"
    return uv_tools_dir.exists() and _path_is_within(resolved, uv_tools_dir)


def _find_modal_executable(modal_cli: str) -> tuple[str | None, str | None]:
    """Return the first non-shim executable and the first shim discovered on PATH."""

    if not modal_cli:
        modal_cli = "modal"

    candidate_path = Path(modal_cli).expanduser()
    if candidate_path.is_absolute() or len(candidate_path.parts) > 1:
        resolved_candidate = candidate_path
        if not resolved_candidate.is_absolute():
            resolved_candidate = (Path.cwd() / resolved_candidate).resolve()
        else:
            resolved_candidate = resolved_candidate.resolve()
        if not resolved_candidate.exists():
            raise click.ClickException(f"--modal-cli path does not exist: {resolved_candidate}")
        if not os.access(resolved_candidate, os.X_OK):
            raise click.ClickException(f"--modal-cli is not executable: {resolved_candidate}")
        return str(resolved_candidate), None

    path_env = os.environ.get("PATH", "")
    if not path_env:
        return None, None

    seen_dirs: set[str] = set()
    seen_candidates: set[str] = set()
    shim_path: str | None = None

    for raw_entry in path_env.split(os.pathsep):
        if not raw_entry:
            continue
        try:
            resolved_entry = str(Path(raw_entry).resolve())
        except Exception:
            resolved_entry = os.path.normpath(raw_entry)
        if resolved_entry in seen_dirs:
            continue
        seen_dirs.add(resolved_entry)

        candidate = shutil.which(modal_cli, path=raw_entry)
        if candidate is None:
            continue
        if candidate in seen_candidates:
            continue
        seen_candidates.add(candidate)

        if _is_modal_shim(candidate):
            if shim_path is None:
                shim_path = candidate
            continue
        return candidate, shim_path

    return None, shim_path


def _modal_command_prefix(modal_cli: str) -> list[str]:
    """Resolve a command prefix for invoking the Modal CLI within the active environment."""

    force_wrapper_env = os.environ.get("SYNTH_FORCE_MODAL_WRAPPER", "").strip().lower()
    if force_wrapper_env in {"1", "true", "yes"}:
        click.secho(
            "[modal-prefix] SYNTH_FORCE_MODAL_WRAPPER=1 -> using in-process wrapper",
            fg="yellow",
        )
        return [sys.executable, "-m", "synth_ai.cli._modal_wrapper"]

    lookup = modal_cli or "modal"
    spec = importlib.util.find_spec("modal") if lookup == "modal" else None

    preferred, shim_candidate = _find_modal_executable(lookup)
    if preferred is not None:
        detail = f"[modal-prefix] modal_cli={lookup} selected={preferred}"
        if lookup == "modal":
            detail += f" spec={'yes' if spec else 'no'}"
        click.secho(detail, fg="cyan")
        return [preferred]

    if lookup != "modal":
        raise click.ClickException(f"Modal CLI not found (looked for '{lookup}')")

    if spec is not None:
        warning = "[modal-prefix] Using synth-ai modal shim; pass --modal-cli /path/to/modal to override."
        if shim_candidate is not None:
            warning = (
                f"[modal-prefix] Using synth-ai modal shim at {shim_candidate}; "
                "pass --modal-cli /path/to/modal to override."
            )
        click.secho(warning, fg="yellow")
        click.secho(
            "[modal-prefix] modal_cli=modal selected=module-wrapper spec=yes",
            fg="yellow",
        )
        return [sys.executable, "-m", "synth_ai.cli._modal_wrapper"]

    if shim_candidate is not None:
        raise click.ClickException(
            "Modal CLI resolution found the synth-ai shim but the 'modal' package "
            "is not importable in this environment. Install the official Modal CLI "
            "or pass --modal-cli with its path."
        )

    raise click.ClickException(
        "Modal CLI not found. Install the 'modal' package in this environment or pass "
        "--modal-cli with an explicit path."
    )


def _build_modal_app_wrapper(original_script: Path) -> tuple[Path, Path]:
    source_dir = original_script.parent.resolve()
    repo_root = REPO_ROOT
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
        raise click.ClickException(str(_pf_err)) from _pf_err

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
        click.echo(
            "Dry run: " + " ".join(shlex.quote(component) for component in cmd),
            err=False,
        )
        return
    click.secho(
        "[modal-exec] " + " ".join(shlex.quote(component) for component in cmd),
        fg="cyan",
    )
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
            with contextlib.suppress(Exception):
                wrapper_path.unlink(missing_ok=True)
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
            secrets_module = _maybe_import("synth_ai.learning.rl.secrets")
            try:
                if secrets_module is None:
                    raise RuntimeError("secrets module unavailable")
                mint_env_key = secrets_module.mint_environment_api_key
                env_api_key = mint_env_key()
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
    entry: TaskAppEntryType,
    modal_cfg: ModalDeploymentConfigType,
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
        click.echo("Dry run: " + " ".join(shlex.quote(component) for component in cmd))
        script_path.unlink(missing_ok=True)
        return
    click.secho(
        "[modal-exec] " + " ".join(shlex.quote(component) for component in cmd),
        fg="cyan",
    )

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
    if not sys.stdin.isatty():
        raise click.ClickException(
            "ENVIRONMENT_API_KEY missing. Provide --env-file or run `synth-ai setup` in an interactive shell to create one."
        )
    existing = _parse_env_file(env_path) if env_path.exists() else {}

    def _prompt(label: str, *, default: str = "", required: bool) -> str | None:
        while True:
            try:
                value = click.prompt(
                    label, default=default, show_default=bool(default) or not required
                ).strip()
            except (Abort, EOFError, KeyboardInterrupt):
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
    click.echo(
        "⚠️  ENVIRONMENT_API_KEY not set. Run `uvx synth-ai setup`, "
        "or pass --env-file pointing at a .env with ENVIRONMENT_API_KEY."
    )
    result = _interactive_fill_env(target)
    if result is None:
        raise click.ClickException("ENVIRONMENT_API_KEY required to continue")
    # After generating .env, load it and override any previously-empty values
    _load_env_values([result])
    if not (os.environ.get("ENVIRONMENT_API_KEY") or "").strip():
        raise click.ClickException("Failed to load ENVIRONMENT_API_KEY from generated .env")


def _deploy_entry(
    entry: TaskAppEntryType,
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

    env_paths = _determine_env_files(entry, env_file, original_path=original_path)
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
    entry: TaskAppEntryType,
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

    env_paths = _determine_env_files(entry, env_file, original_path=original_path)
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


@task_app_group.command("validate")
@click.argument("app_id", type=str, required=True)
@click.option(
    "--url",
    type=str,
    default=None,
    help="Task app URL to validate (if not provided, starts a local server)",
)
@click.option(
    "--port",
    type=int,
    default=8765,
    help="Port to use for temporary server (default: 8765)",
)
@click.option(
    "--api-key",
    type=str,
    default=None,
    envvar="ENVIRONMENT_API_KEY",
    help="API key for authentication (default: $ENVIRONMENT_API_KEY)",
)
@click.option(
    "--min-instances",
    type=int,
    default=10,
    help="Minimum number of task instances required (default: 10)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Show detailed information about the task app",
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Output results as JSON",
)
def validate_task_app_cmd(
    app_id: str,
    url: str | None,
    port: int,
    api_key: str | None,
    min_instances: int,
    verbose: bool,
    output_json: bool,
) -> None:
    """Validate a task app deployment readiness.
    
    This command verifies that a task app is properly configured and ready to run
    by checking all required HTTP endpoints, authentication, and task availability.
    
    By default, it starts a temporary local server for validation. You can also
    validate a remote deployment by passing --url.
    
    \b
    What gets validated:
    • Root endpoint (/) responds correctly
    • Health endpoint (/health) is accessible with proper authentication
    • Info endpoint (/info) returns valid task metadata  
    • Task info endpoint (/task_info) provides task instances
    • Rollout endpoint (/rollout) is registered
    • At least N task instances are available (default: 10)
    
    \b
    Examples:
    
    \b
    Validate grpo-crafter (starts local server automatically):
        $ synth-ai task-app validate grpo-crafter
    
    \b
    Validate sokoban with verbose output:
        $ synth-ai task-app validate sokoban --verbose
    
    \b
    Validate with custom port:
        $ synth-ai task-app validate sokoban --port 9000
    
    \b
    Validate a remote deployment:
        $ synth-ai task-app validate grpo-crafter --url https://my-crafter.modal.run
    
    \b
    Require at least 20 task instances:
        $ synth-ai task-app validate grpo-crafter --min-instances 20
    
    \b
    Get JSON output for automation:
        $ synth-ai task-app validate sokoban --json
    
    \b
    Common use cases:
    • Pre-deployment verification: Check task app works before deploying to Modal
    • CI/CD integration: Use --json flag for automated validation in pipelines
    • Debug failing deployments: Use --verbose to see detailed endpoint responses
    • Test API key configuration: Verify authentication is set up correctly
    """
    import asyncio
    import socket
    import subprocess
    import tempfile
    import time
    
    # Import the validate_task_app function defined in this module
    from synth_ai.cli._validate_task_app import validate_task_app  # type: ignore[attr-defined]
    
    proc = None
    task_app_url = url
    
    try:
        # If no URL provided, start a temporary server
        if not task_app_url:
            # Find an available port
            def is_port_available(port: int) -> bool:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    try:
                        s.bind(("", port))
                        return True
                    except OSError:
                        return False
            
            while not is_port_available(port):
                port += 1
            
            task_app_url = f"http://localhost:{port}"
            
            if not output_json:
                click.echo(f"Starting temporary {app_id} server on port {port}...")
            
            # Start the server in background
            env = os.environ.copy()
            if api_key:
                env["ENVIRONMENT_API_KEY"] = api_key
            
            # Create a temporary trace DB and trace dir to avoid prompts
            import tempfile
            temp_dir = tempfile.mkdtemp()
            temp_trace_db = os.path.join(temp_dir, "validate_trace.db")
            temp_trace_dir = os.path.join(temp_dir, "traces")
            os.makedirs(temp_trace_dir, exist_ok=True)
            
            proc = subprocess.Popen(
                [
                    "uv",
                    "run",
                    "synth-ai",
                    "task-app",
                    "serve",
                    app_id,
                    "--port",
                    str(port),
                    "--no-reload",
                    "--trace",
                    temp_trace_dir,
                    "--trace-db",
                    temp_trace_db,
                ],
                env=env,
                stdin=subprocess.PIPE,  # Add stdin to handle any prompts
                stdout=subprocess.DEVNULL if output_json else subprocess.PIPE,
                stderr=subprocess.DEVNULL if output_json else subprocess.PIPE,
                text=True,
            )
            
            # Write empty input to stdin to skip any prompts
            if proc.stdin:
                try:
                    proc.stdin.write("\n")
                    proc.stdin.flush()
                    proc.stdin.close()
                except Exception:
                    pass
            
            # Wait for server to be ready
            if not output_json:
                click.echo("Waiting for server to start...")
            
            import httpx
            for _attempt in range(60):  # 30 seconds timeout
                try:
                    async def check_health():
                        async with httpx.AsyncClient(timeout=2.0) as client:
                            resp = await client.get(f"{task_app_url}/")
                            return resp.status_code == 200
                    
                    if asyncio.run(check_health()):
                        break
                except Exception:
                    pass
                
                # Check if process died
                if proc.poll() is not None:
                    stderr_output = ""
                    if proc.stderr and not output_json:
                        stderr_output = proc.stderr.read()
                    click.echo(click.style("✗ Server process exited unexpectedly", fg="red"), err=True)
                    if stderr_output and not output_json:
                        click.echo(f"Error output:\n{stderr_output}", err=True)
                    sys.exit(1)
                
                time.sleep(0.5)
            else:
                click.echo(click.style("✗ Server failed to start within 30 seconds", fg="red"), err=True)
                sys.exit(1)
            
            if not output_json:
                click.echo(click.style("✓ Server started", fg="green"))
                click.echo()
        
        # Ensure URL doesn't have trailing slash
        task_app_url = task_app_url.rstrip("/")
        
        async def _run() -> tuple[bool, dict[str, Any]]:
            return await validate_task_app(
                url=task_app_url,
                api_key=api_key,
                min_instances=min_instances,
                verbose=verbose,
            )
        
        success, results = asyncio.run(_run())
        
        if output_json:
            import json as _json
            click.echo(_json.dumps(results, indent=2))
        
        sys.exit(0 if success else 1)
    
    finally:
        # Cleanup: stop the temporary server
        if proc is not None:
            if not output_json:
                click.echo("\nStopping temporary server...")
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
        
        # Cleanup temp trace DB
        if not url and 'temp_dir' in locals():
            import contextlib
            import shutil
            with contextlib.suppress(Exception):
                shutil.rmtree(temp_dir, ignore_errors=True)


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
    demo_dir_path = _load_demo_directory()
    if demo_dir_path:
        if not demo_dir_path.is_dir():
            raise click.ClickException(
                f"Demo directory not found: {demo_dir_path}\nRun 'synth-ai setup' to create a demo."
            )
        os.chdir(demo_dir_path)
        click.echo(f"Using demo directory: {demo_dir_path}\n")
        os.environ["SYNTH_DEMO_DIR"] = str(demo_dir_path.resolve())

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
    auth_module = _maybe_import("synth_ai.task.auth")
    if auth_module is not None:
        _norm_key = getattr(auth_module, "normalize_environment_api_key", lambda: _os.getenv("ENVIRONMENT_API_KEY"))
    else:
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
    demo_dir_path = _load_demo_directory()
    if demo_dir_path:
        if not demo_dir_path.is_dir():
            raise click.ClickException(
                f"Demo directory not found: {demo_dir_path}\nRun 'synth-ai setup' to create a demo."
            )
        os.chdir(demo_dir_path)
        click.echo(f"Using demo directory: {demo_dir_path}\n")
        os.environ["SYNTH_DEMO_DIR"] = str(demo_dir_path.resolve())

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


def _determine_env_files(
    entry: TaskAppEntryType, user_env_files: Sequence[str], *, original_path: Path | None = None
) -> list[Path]:
    resolved: list[Path] = []
    for candidate in user_env_files:
        p = Path(candidate).expanduser()
        if not p.exists():
            raise click.ClickException(f"Env file not found: {p}")
        resolved.append(p)
    if resolved:
        return resolved

    declared: list[Path] = []
    for candidate in getattr(entry, "env_files", ()) or ():
        try:
            p = Path(candidate).expanduser()
        except Exception:
            continue
        if p.exists() and p.is_file():
            declared.append(p)
    if declared:
        return declared

    def _append_candidate(collection: list[Path], candidate: Path) -> None:
        if candidate.exists() and candidate.is_file() and candidate not in collection:
            collection.append(candidate)

    auto_candidates: list[Path] = []

    search_dirs: list[Path] = []
    if original_path is not None:
        search_dirs.append(original_path.parent.resolve())
        for parent in original_path.parent.resolve().parents:
            search_dirs.append(parent)
    cwd = Path.cwd().resolve()
    if cwd not in search_dirs:
        search_dirs.append(cwd)
    repo_root = REPO_ROOT.resolve()
    if repo_root not in search_dirs:
        search_dirs.append(repo_root)

    for directory in search_dirs:
        _append_candidate(auto_candidates, directory / ".env")
        for candidate in sorted(directory.glob("*.env")):
            _append_candidate(auto_candidates, candidate)

    if auto_candidates:
        return [auto_candidates[0]]

    raise click.ClickException(
        "No .env file discovered automatically. Pass --env-file /path/to/.env or generate one with `uvx synth-ai setup`."
    )


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
        cwd = Path.cwd().resolve()
        demo_dir = _load_demo_directory()

        if demo_dir and demo_dir == cwd and (cwd / "run_local_rollout_traced.py").exists():
            click.echo("\n" + "=" * 60)
            click.echo("Next step: Collect traced rollouts")
            click.echo("=" * 60)
            click.echo("\nIn another terminal, run:")
            click.echo(f"  cd {cwd}")
            click.echo("  uv run python run_local_rollout_traced.py")
            click.echo("\nRun this 5-10 times to collect diverse traces.")
            click.echo("=" * 60 + "\n")
    except Exception:
        pass


def _serve_entry(
    entry: TaskAppEntryType,
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
        tracing_config_module = _maybe_import("synth_ai.tracing_v3.config")
        if tracing_config_module is not None:
            trace_config = tracing_config_module.CONFIG
            new_db_url = os.getenv("TURSO_LOCAL_DB_URL") or trace_config.db_url
            trace_config.db_url = new_db_url
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

    demo_dir_path = _load_demo_directory()
    if demo_dir_path:
        if not demo_dir_path.is_dir():
            raise click.ClickException(
                f"Demo directory not found: {demo_dir_path}\nRun 'synth-ai demo' to create a demo."
            )
        os.chdir(demo_dir_path)
        click.echo(f"Using demo directory: {demo_dir_path}\n")

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
    click.echo(f"[modal-serve] requested app_id={app_id or '(auto)'} modal_cli={modal_cli}")
    try:
        choice = _select_app_choice(app_id, purpose="modal-serve")
    except SystemExit as exc:  # bubble up with context (legacy argparse would trigger this)
        raise click.ClickException(
            f"Legacy CLI intercepted modal-serve (exit {exc.code}). "
            "Make sure you're running the Click CLI (synth_ai.cli:cli)."
        ) from exc

    if choice.modal_script:
        env_paths = _resolve_env_paths_for_script(choice.modal_script, env_file)
        click.echo("Using env file(s): " + ", ".join(str(p.resolve()) for p in env_paths))
        _run_modal_script(choice.modal_script, modal_cli, "serve", env_paths, modal_name=modal_name)
        return

    entry = choice.ensure_entry()
    click.echo(f"[modal-serve] serving entry {entry.app_id} from {choice.path}")
    _modal_serve_entry(entry, modal_name, modal_cli, env_file, original_path=choice.path)


def _write_modal_entrypoint(
    entry: TaskAppEntryType,
    modal_cfg: ModalDeploymentConfigType,
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
    host_synth = _maybe_import("synth_ai")
    if host_synth is not None:
        host_ver = getattr(host_synth, "__version__", None)
        if host_ver:
            synth_pkg = f"synth-ai=={host_ver}"
    if not any(str(p).startswith("synth-ai") for p in pip_packages):
        pip_packages.insert(0, synth_pkg)

    apt_packages = list(modal_cfg.apt_packages)
    click.echo(f"[DEBUG] modal_cfg.apt_packages type: {type(modal_cfg.apt_packages)}")
    click.echo(f"[DEBUG] modal_cfg.apt_packages value: {modal_cfg.apt_packages}")
    click.echo(f"[DEBUG] apt_packages after list(): {apt_packages}")
    
    local_dirs = [(str(Path(src)), dst) for src, dst in modal_cfg.extra_local_dirs]
    # Also mount the host synth_ai source if available to ensure latest code is used
    if host_synth is not None:
        try:
            host_synth_dir = Path(host_synth.__file__).resolve().parent
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

# CRITICAL: Install iverilog for Verilog task app (hardcoded to prevent config issues)
if {entry.app_id!r} == "grpo-verilog":
    image = image.apt_install("iverilog")

# Install apt packages first (before pip)
apt_packages = {apt_packages!r}
if apt_packages:
    image = image.apt_install(*apt_packages)

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
    cli.add_command(filter_command)


@click.command(
    "eval",
    help="Run one-off rollouts against a task app and print judge/eval summaries.",
)
@click.argument("app_id", type=str, required=False)
@click.option(
    "--config",
    type=click.Path(),
    default=None,
    help="Path to eval TOML (short schema). Auto-discovers the first matching file when omitted.",
)
@click.option(
    "--url",
    "task_app_url",
    type=str,
    default=None,
    help="Base URL of a running task app instead of spawning locally (requires --env-file for secrets).",
)
@click.option(
    "--seeds",
    default="0,1,2,3,4",
    help="Comma-separated seeds/indices to evaluate. Use negative numbers to wrap around the dataset.",
)
@click.option("--split", default="train", show_default=True, help="Dataset split to use")
@click.option(
    "--model",
    default=None,
    help="Model identifier. When omitted the CLI will prompt based on task metadata.",
)
@click.option(
    "--env-file",
    multiple=True,
    type=click.Path(),
    help="Env file(s) to load (API keys, etc.). Required when using --url or remote judges.",
)
@click.option(
    "--trace-db",
    default="traces/v3/synth_ai.db",
    show_default=True,
    help="SQLite/Turso URL for storing rollout traces set to 'none' to disable persistence.",
)
@click.option(
    "--metadata",
    multiple=True,
    help="Filter tasks by key=value metadata (e.g., --metadata difficulty=easy)",
)
@click.option(
    "--metadata-sql",
    default=None,
    help="SQLite query that returns seeds to evaluate (e.g., SELECT seed FROM tasks WHERE difficulty='easy' LIMIT 5)",
)
def eval_command(
    app_id: str | None,
    config: str | None,
    task_app_url: str | None,
    seeds: str,
    split: str,
    model: str | None,
    env_file: Sequence[str],
    trace_db: str,
    metadata: Sequence[str],
    metadata_sql: str | None,
) -> None:
    """Run rollouts against a task app and report judge statistics.

    By default the command spins up the selected task app in-process, executes the
    requested seeds, and prints aggregate scores (official and custom judges). When
    pointing at a remote `--url`, supply matching `--env-file` values so the CLI can
    forward authentication headers to the running service.
    """
    # Parse and validate TOML config
    from synth_ai.task.config import EvalConfig
    
    cfg: dict[str, Any] = {}
    eval_cfg: EvalConfig | None = None
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
            
            # Validate config with dataclass
            try:
                eval_cfg = EvalConfig.from_dict(cfg)
                click.echo(f"✓ Config validated: {len(eval_cfg.seeds)} seeds, model={eval_cfg.model}")
            except (ValueError, TypeError) as validation_error:
                raise click.ClickException(f"Invalid eval config: {validation_error}") from validation_error
        except click.ClickException:
            raise
        except Exception as exc:
            raise click.ClickException(f"Failed to parse TOML '{config_path}': {exc}") from exc

    # CLI args override config
    if eval_cfg:
        app_id = app_id or eval_cfg.app_id
    else:
        app_id = app_id or (cfg.get("app_id") if isinstance(cfg.get("app_id"), str) else None)  # type: ignore

    metadata_filters: dict[str, str] = {}
    if eval_cfg:
        metadata_filters.update(eval_cfg.metadata)
    else:
        cfg_metadata = cfg.get("metadata")
        if isinstance(cfg_metadata, dict):
            for key, value in cfg_metadata.items():
                metadata_filters[str(key)] = str(value)
        elif isinstance(cfg_metadata, list):
            for item in cfg_metadata:
                if isinstance(item, str) and "=" in item:
                    key, value = item.split("=", 1)
                    metadata_filters[key.strip()] = value.strip()

    for item in metadata or ():
        if "=" not in item:
            raise click.ClickException(f"Metadata filters must be key=value (got: {item})")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise click.ClickException(f"Invalid metadata filter: {item}")
        metadata_filters[key] = value

    metadata_sql_query: str | None = None
    if eval_cfg and eval_cfg.metadata_sql:
        metadata_sql_query = eval_cfg.metadata_sql
    else:
        cfg_metadata_sql = cfg.get("metadata_sql")
        if isinstance(cfg_metadata_sql, dict):
            metadata_sql_query = cfg_metadata_sql.get("query") or cfg_metadata_sql.get("sql")
        elif isinstance(cfg_metadata_sql, str):
            metadata_sql_query = cfg_metadata_sql

    if metadata_sql:
        metadata_sql_query = metadata_sql
    if metadata_sql_query is not None:
        metadata_sql_query = str(metadata_sql_query)

    trace_db_url: str | None = None
    trace_db = (trace_db or "").strip()
    if trace_db and trace_db.lower() not in {"none", "off", "disable"}:
        if "://" in trace_db:
            trace_db_url = trace_db
        else:
            trace_path = Path(trace_db).expanduser()
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_db_url = f"sqlite+aiosqlite:///{trace_path}"
    trace_tracer: SessionTracer | None = SessionTracer(db_url=trace_db_url, auto_save=True) if trace_db_url else None

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

    choice_for_env: AppChoice | None = None
    entry: TaskAppEntryType | None = None
    if task_app_url is None:
        choice_for_env = _select_app_choice(app_id, purpose="eval")
        entry = choice_for_env.ensure_entry()

    env_paths: list[Path] = []
    if entry is not None:
        original_env_path = choice_for_env.path if choice_for_env is not None else None
        env_paths = _determine_env_files(entry, env_file, original_path=original_env_path)
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
    inference_meta: dict[str, Any] = {}
    supported: list[str] = []
    seen_models: set[str] = set()

    def _add_supported_model(candidate: Any) -> None:
        if not candidate:
            return
        text = str(candidate).strip()
        if not text or text in seen_models:
            return
        supported.append(text)
        seen_models.add(text)

    if task_app_url is None:
        try:
            if hasattr(config, "base_task_info") and config.base_task_info:
                inf_obj = getattr(config.base_task_info, "inference", None)
                if inf_obj is not None:
                    if hasattr(inf_obj, "model_dump"):
                        inference_meta = dict(inf_obj.model_dump(exclude_none=True))  # type: ignore[attr-defined]
                    elif isinstance(inf_obj, dict):
                        inference_meta = dict(inf_obj)
        except Exception:
            inference_meta = {}
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
                inference_meta = dict(inf)
        except Exception:
            inference_meta = {}

    default_model = inference_meta.get("model")
    if isinstance(default_model, str):
        _add_supported_model(default_model)

    models_field = inference_meta.get("models")
    if isinstance(models_field, list):
        for candidate in models_field:
            _add_supported_model(candidate)

    supported_models = inference_meta.get("supported_models")
    if isinstance(supported_models, list):
        for candidate in supported_models:
            _add_supported_model(candidate)

    providers = inference_meta.get("providers")
    if isinstance(providers, list):
        if "openai" in providers:
            _add_supported_model("gpt-5")
        if "groq" in providers:
            _add_supported_model("groq:llama-3.1-70b-versatile")

    _add_supported_model("synth:qwen-0.6b")

    selected_model = model
    if not selected_model:
        if not supported:
            raise click.ClickException(
                "No supported models; supply --model or add base_task_info.inference.model"
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
            "inference_url",
        ):
            if k in cfg and k not in policy_overrides:
                policy_overrides[k] = cfg.get(k)
    except Exception:
        policy_overrides = {}

    raw_concurrency = cfg.get("concurrency")
    try:
        concurrency_limit = int(raw_concurrency) if raw_concurrency is not None else 1
    except Exception:
        concurrency_limit = 1
    if concurrency_limit <= 0:
        concurrency_limit = 1
    concurrency_limit = min(concurrency_limit, max(1, len(seed_values)))

    judge_specs: list[JudgeSpec] = []

    def _register_judge(name_hint: str | None, judge_cfg: dict[str, Any]) -> None:
        if not judge_cfg:
            return
        judge_module = judge_cfg.get("module")
        judge_path = judge_cfg.get("path")
        judge_callable_name = judge_cfg.get("callable") or judge_cfg.get("function")
        if judge_module and judge_path:
            raise click.ClickException("Judge config cannot set both 'module' and 'path'")
        if not judge_module and not judge_path:
            raise click.ClickException("Judge config requires 'module' or 'path'")
        try:
            if judge_module:
                module = importlib.import_module(str(judge_module))
            else:
                path = Path(str(judge_path)).expanduser()
                if not path.exists():
                    raise click.ClickException(f"Judge module path not found: {path}")
                spec = importlib.util.spec_from_file_location(
                    f"_eval_judge_{path.stem}", path
                )
                if not spec or not spec.loader:
                    raise click.ClickException(f"Failed to load judge module from {path}")
                module = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = module
                spec.loader.exec_module(module)
        except click.ClickException:
            raise
        except Exception as exc:
            raise click.ClickException(f"Unable to load judge module: {exc}") from exc

        if judge_callable_name:
            try:
                judge_fn = getattr(module, str(judge_callable_name))
            except AttributeError as exc:
                raise click.ClickException(
                    f"Judge callable '{judge_callable_name}' not found in module"
                ) from exc
        else:
            if hasattr(module, "judge"):
                judge_fn = module.judge
            else:
                raise click.ClickException("Judge module must expose 'judge' callable")

        if not callable(judge_fn):
            raise click.ClickException("Judge callable is not callable")

        judge_kwargs = {
            k: v
            for k, v in judge_cfg.items()
            if k not in {"module", "path", "callable", "function", "name"}
        }
        display_name = str(
            judge_cfg.get("name")
            or name_hint
            or f"judge{len(judge_specs) + 1}"
        )
        judge_specs.append(JudgeSpec(display_name, judge_fn, judge_kwargs))

    raw_judge_cfg = cfg.get("judge")
    if isinstance(raw_judge_cfg, dict) and raw_judge_cfg:
        direct_keys = {"module", "path", "callable", "function", "name"}
        has_direct_keys = any(key in raw_judge_cfg for key in direct_keys)
        nested_candidates = [
            (key, value)
            for key, value in raw_judge_cfg.items()
            if isinstance(value, dict)
        ]
        if has_direct_keys and not nested_candidates:
            _register_judge(None, raw_judge_cfg)
        else:
            for sub_name, sub_cfg in nested_candidates:
                _register_judge(sub_name, sub_cfg)

    raw_judges_list = cfg.get("judges")
    if isinstance(raw_judges_list, list):
        for _index, entry in enumerate(raw_judges_list, start=1):
            if isinstance(entry, dict):
                _register_judge(entry.get("name") or f"judge{len(judge_specs) + 1}", entry)

    records: list[dict[str, Any]] = []

    successes = 0
    failures = 0
    # Aggregate outcome stats across successful seeds
    outcome_sum: float = 0.0
    outcome_count: int = 0
    outcome_correct: int = 0

    def _build_task_rows(taskset: Any) -> dict[int, dict[str, Any]]:
        rows: dict[int, dict[str, Any]] = {}
        if not isinstance(taskset, dict):
            return rows

        scenario_ids = taskset.get("scenario_ids") or []
        loop_ids = taskset.get("loop_ids") or []
        thread_ids = taskset.get("thread_ids") or []
        difficulty_map = taskset.get("difficulty_map") or {}

        max_len = max(len(scenario_ids), len(loop_ids), len(thread_ids))
        for seed in range(max_len):
            scenario_id = scenario_ids[seed] if seed < len(scenario_ids) else None
            loop_id = loop_ids[seed] if seed < len(loop_ids) else None
            thread_id = thread_ids[seed] if seed < len(thread_ids) else None
            difficulty = None
            if isinstance(difficulty_map, dict):
                if scenario_id and scenario_id in difficulty_map:
                    difficulty = difficulty_map.get(scenario_id)
                elif str(seed) in difficulty_map:
                    difficulty = difficulty_map.get(str(seed))

            rows[seed] = {
                "seed": seed,
                "scenario_id": scenario_id,
                "loop_id": loop_id,
                "thread_id": thread_id,
                "difficulty": difficulty,
            }
        return rows

    def _apply_metadata_filters(
        rows: dict[int, dict[str, Any]], seeds_list: list[int], filters: dict[str, str]
    ) -> list[int]:
        if not filters:
            return seeds_list
        filtered: list[int] = []
        for seed in seeds_list:
            row = rows.get(seed)
            if not row:
                continue
            include = True
            for key, expected in filters.items():
                actual = row.get(key)
                if actual is None:
                    include = False
                    break
                if str(actual).lower() != expected.lower():
                    include = False
                    break
            if include:
                filtered.append(seed)
        return filtered

    def _apply_metadata_sql(
        rows: dict[int, dict[str, Any]], seeds_list: list[int], query: str
    ) -> list[int]:
        """Return seeds that satisfy an arbitrary SQL query.

        The query is executed against an in-memory SQLite table named `tasks`
        with columns (seed INTEGER, scenario_id TEXT, loop_id TEXT, thread_id TEXT, difficulty TEXT).
        Any rows whose `seed` value (or first column if `seed` is absent) appear in the result set are retained.
        """
        if not query:
            return seeds_list
        conn = sqlite3.connect(":memory:")
        try:
            cur = conn.cursor()
            cur.execute(
                "CREATE TABLE tasks (seed INTEGER, scenario_id TEXT, loop_id TEXT, thread_id TEXT, difficulty TEXT)"
            )
            insert_stmt = (
                "INSERT INTO tasks (seed, scenario_id, loop_id, thread_id, difficulty) VALUES (?,?,?,?,?)"
            )
            for seed in seeds_list:
                row = rows.get(seed, {})
                cur.execute(
                    insert_stmt,
                    [
                        seed,
                        row.get("scenario_id"),
                        row.get("loop_id"),
                        row.get("thread_id"),
                        row.get("difficulty"),
                    ],
                )

            result = cur.execute(query)
            fetched = result.fetchall()
            if not fetched:
                return []
            description = result.description or []
            col_names = [col[0] for col in description]
            seeds_out: list[int] = []
            for entry in fetched:
                value = entry[col_names.index("seed")] if "seed" in col_names else entry[0]
                try:
                    seeds_out.append(int(value))
                except Exception as exc:
                    raise click.ClickException(
                        "metadata SQL query must return seed integers"
                    ) from exc
            seeds_set = set(seeds_out)
            return [seed for seed in seeds_list if seed in seeds_set]
        except sqlite3.Error as exc:
            raise click.ClickException(f"Failed to execute metadata SQL query: {exc}") from exc
        finally:
            conn.close()

    async def _run_eval() -> None:
        nonlocal successes, failures, outcome_sum, outcome_count, outcome_correct, records, seed_values

        if trace_tracer is not None and trace_tracer.db is None:
            await trace_tracer.initialize()

        if task_app_url is None:
            transport = httpx.ASGITransport(app=app)  # type: ignore[name-defined]
            async_client = httpx.AsyncClient(
                transport=cast(Any, transport),
                base_url="http://eval.local",
                timeout=300.0,
                follow_redirects=True,
                headers=headers,
            )
        else:
            async_client = httpx.AsyncClient(
                base_url=task_app_url,
                timeout=300.0,
                follow_redirects=True,
                headers=headers,
            )

        try:
            taskset_payload: dict[str, Any] | None = None
            try:
                task_info_response = await async_client.get("/task_info")
            except Exception:
                task_info_response = None
            if task_info_response is not None and task_info_response.status_code == 200:
                with contextlib.suppress(Exception):
                    payload_json = task_info_response.json()
                if isinstance(payload_json, dict) and "taskset" in payload_json:
                    taskset_payload = payload_json.get("taskset")
                    if not isinstance(taskset_payload, dict):
                        taskset_payload = None
                elif isinstance(payload_json, dict):
                    taskset_payload = payload_json

            available_seeds = list(seed_values)
            if metadata_sql_query or metadata_filters:
                if not taskset_payload:
                    raise click.ClickException(
                        "Task metadata filters require the task app to expose /task_info metadata"
                    )
                rows = _build_task_rows(taskset_payload)
                if metadata_sql_query:
                    available_seeds = _apply_metadata_sql(rows, available_seeds, metadata_sql_query)
                if metadata_filters:
                    available_seeds = _apply_metadata_filters(rows, available_seeds, metadata_filters)
                if not available_seeds:
                    raise click.ClickException("No seeds match the provided metadata filters")
                seed_values = available_seeds

            semaphore = asyncio.Semaphore(concurrency_limit)

            async def _run_seed(seed_val: int) -> None:
                nonlocal successes, failures, outcome_sum, outcome_count, outcome_correct, records
                # Read env_name and policy_name from config if available
                env_name = cfg.get("env_name") or (cfg.get("env", {}).get("env_name") if isinstance(cfg.get("env"), dict) else None)
                policy_name = cfg.get("policy_name") or (cfg.get("policy", {}).get("policy_name") if isinstance(cfg.get("policy"), dict) else None)
                env_config_overrides = cfg.get("env_config", {}) if isinstance(cfg.get("env_config"), dict) else {}
                policy_config_overrides = cfg.get("policy_config", {}) if isinstance(cfg.get("policy_config"), dict) else {}
                
                # Debug: print config parsing
                if seed_val == 0:
                    click.echo(f"[DEBUG] env_name from config: {env_name}")
                    click.echo(f"[DEBUG] policy_name from config: {policy_name}")
                
                # Generate default ops sequence if not provided
                max_llm_calls = policy_config_overrides.get("max_llm_calls", 10)
                ops_list = cfg.get("ops", [])
                if not ops_list:
                    # Generate default "agent, env" pairs for max_llm_calls
                    ops_list = ["agent", "env"] * int(max_llm_calls)
                
                body = {
                    "run_id": str(uuid.uuid4()),
                    "env": {"config": {"split": split, "index": seed_val, **env_config_overrides}, "seed": seed_val},
                    "policy": {
                        "policy_name": policy_name or selected_model,
                        "config": {"model": selected_model, **policy_overrides, **policy_config_overrides},
                    },
                    "ops": ops_list,
                    "record": {
                        "return_trace": cfg.get("return_trace", True),
                        "trace_format": cfg.get("trace_format", "structured"),
                    },
                    "mode": "eval",  # RolloutMode.EVAL: use inference URLs as-is, no transformations
                }
                if env_name:
                    body["env"]["env_name"] = env_name
                
                # Debug: print the body being sent
                if seed_val == 0:
                    click.echo(f"[DEBUG] rollout body env: {body['env']}")
                    click.echo(f"[DEBUG] rollout body policy: {body['policy']}")
                    click.echo(f"[DEBUG] rollout body mode: {body.get('mode', 'NOT SET')}")
                rollout_elapsed: float | None = None
                rollout_start = time.perf_counter()
                try:
                    import logging
                    _log = logging.getLogger(__name__)
                    _log.info(f"[EVAL_BODY_DEBUG] Sending body with mode={body.get('mode')}")
                    async with semaphore:
                        response = await async_client.post("/rollout", json=body)
                    rollout_elapsed = time.perf_counter() - rollout_start
                except Exception as exc:
                    failures += 1
                    click.echo(f"seed={seed_val} error={exc}")
                    return

                ok = 200 <= response.status_code < 300
                if ok:
                    successes += 1
                else:
                    failures += 1

                summary = [f"seed={seed_val}", f"status={response.status_code}"]
                data: Any
                try:
                    data = response.json()
                except Exception:
                    data = None
                
                # Debug: print validation errors
                if response.status_code == 422 and data:
                    click.echo(f"[DEBUG] 422 Validation Error: {data}")

                metrics: dict[str, Any] | None = None
                completion: str | None = None
                prompt_index: int | None = None
                prompt_text: str | None = None
                task_id: str | None = None
                task_split: str | None = None
                task_rubric_id: str | None = None

                trace_namespace: dict[str, Any] | None = None
                session_trace_dict: dict[str, Any] | None = None

                if isinstance(data, dict):
                    import logging
                    _logger = logging.getLogger(__name__)
                    _logger.info(f"[EVAL_DEBUG] Response data keys: {list(data.keys())}")
                    if "detail" in data:
                        _logger.error(f"[EVAL_DEBUG] Task app returned error: {data['detail']}")
                    trace_namespace = data.get("trace")
                    _logger.info(f"[EVAL_DEBUG] trace_namespace type: {type(trace_namespace)}, value: {trace_namespace if not isinstance(trace_namespace, dict) else 'dict with keys: ' + str(list(trace_namespace.keys()) if trace_namespace else 'None')}")
                    if not isinstance(trace_namespace, dict):
                        raise RuntimeError(
                            "The 'synth-ai eval' command requires trace payloads in rollout responses. "
                            "Ensure the rollout request includes 'trace_format': 'structured' and 'return_trace': true, "
                            "and that task app tracing is enabled (TASKAPP_TRACING_ENABLED=1). "
                            "Note: This is specific to the eval command - general rollout endpoints don't require traces."
                        )
                    # Handle both "compact" and "full" trace formats:
                    # - compact: trace_namespace contains {session_id, metadata, ...}
                    # - full: trace_namespace IS the full session_trace dict
                    session_trace_dict = trace_namespace.get("session_trace")
                    if not isinstance(session_trace_dict, dict):
                        # If no session_trace key, assume "full" format where trace itself is the session_trace
                        if "session_id" in trace_namespace:
                            session_trace_dict = trace_namespace
                        else:
                            raise RuntimeError(
                                "The 'synth-ai eval' command requires 'session_trace' in the trace payload or a valid full trace format. "
                                "Ensure the task app is using tracing_v3 and returning structured trace data."
                            )
                    metrics = data.get("metrics") if isinstance(data.get("metrics"), dict) else None
                    if metrics:
                        mean_return = metrics.get("mean_return") or metrics.get("total_reward")
                        outcome = metrics.get("outcome_score")
                        if mean_return is not None:
                            summary.append(f"mean_return={mean_return}")
                        if outcome is not None:
                            summary.append(f"outcome={outcome}")
                            try:
                                val = float(outcome)
                                outcome_sum += val
                                outcome_count += 1
                                if val >= 0.5:
                                    outcome_correct += 1
                            except Exception:
                                pass
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
                            obs = step0.get("obs") if isinstance(step0, dict) else None
                            if isinstance(obs, dict):
                                idx_val = obs.get("prompt_index")
                                if isinstance(idx_val, int):
                                    prompt_index = idx_val
                                prompt_raw = obs.get("prompt")
                                if isinstance(prompt_raw, str):
                                    prompt_text = prompt_raw
                                if task_id is None:
                                    candidate_id = obs.get("task_id")
                                    if isinstance(candidate_id, str) and candidate_id:
                                        task_id = candidate_id
                                if task_split is None:
                                    candidate_split = obs.get("task_split")
                                    if isinstance(candidate_split, str) and candidate_split:
                                        task_split = candidate_split
                                if task_rubric_id is None:
                                    candidate_rid = obs.get("task_rubric_id")
                                    if isinstance(candidate_rid, str) and candidate_rid:
                                        task_rubric_id = candidate_rid
                        final = first.get("final") if isinstance(first, dict) else None
                        if isinstance(final, dict):
                            final_obs = final.get("observation")
                            if isinstance(final_obs, dict):
                                comp_val = final_obs.get("completion")
                                if isinstance(comp_val, str):
                                    completion = comp_val
                                if task_id is None:
                                    candidate_id = final_obs.get("task_id")
                                    if isinstance(candidate_id, str) and candidate_id:
                                        task_id = candidate_id
                                if task_split is None:
                                    candidate_split = final_obs.get("task_split")
                                    if isinstance(candidate_split, str) and candidate_split:
                                        task_split = candidate_split
                                if task_rubric_id is None:
                                    candidate_rid = final_obs.get("task_rubric_id")
                                    if isinstance(candidate_rid, str) and candidate_rid:
                                        task_rubric_id = candidate_rid
                            final_info = final.get("info")
                            if isinstance(final_info, dict):
                                if task_id is None:
                                    candidate_id = final_info.get("task_id")
                                    if isinstance(candidate_id, str) and candidate_id:
                                        task_id = candidate_id
                                if task_split is None:
                                    candidate_split = final_info.get("task_split")
                                    if isinstance(candidate_split, str) and candidate_split:
                                        task_split = candidate_split
                                if task_rubric_id is None:
                                    candidate_rid = final_info.get("task_rubric_id")
                                    if isinstance(candidate_rid, str) and candidate_rid:
                                        task_rubric_id = candidate_rid
                    if task_id:
                        summary.append(f"task_id={task_id}")
                    click.echo(" ".join(summary))
                    with contextlib.suppress(Exception):
                        click.echo(json.dumps(data, indent=2))
                else:
                    click.echo(" ".join(summary))

                official_score = None
                if isinstance(metrics, dict):
                    for key in ("mean_return", "total_reward", "outcome_score"):
                        val = metrics.get(key)
                        if isinstance(val, int | float):
                            official_score = float(val)
                            break
                if official_score is None and isinstance(data, dict):
                    try:
                        reward_val = data["trajectories"][0]["steps"][0].get("reward")
                        if isinstance(reward_val, int | float):
                            official_score = float(reward_val)
                    except Exception:
                        pass

                if official_score is not None:
                    if official_score < 0.0:
                        official_score = 0.0
                    elif official_score > 1.0:
                        official_score = min(1.0, official_score)

                judge_scores: dict[str, float | None] = {}
                judges_timings: dict[str, float | None] = {}
                timings: dict[str, Any] = {
                    "rollout_s": rollout_elapsed,
                    "judges": judges_timings,
                }
                if judge_specs:
                    for spec in judge_specs:
                        score_value: float | None = None
                        judge_elapsed: float | None = None
                        # Run judges for all tasks (text-based and trajectory-based)
                        # Text-based tasks have completion, trajectory-based tasks use response
                        judge_payload = {
                            "seed": seed_val,
                            "prompt_index": prompt_index,
                            "prompt": prompt_text,
                            "completion": completion,
                            "metrics": metrics,
                            "response": data,
                            "trace": trace_namespace,
                        }
                        try:
                            judge_start = time.perf_counter()
                            result = spec.fn(judge_payload, **spec.kwargs)
                            judge_elapsed = time.perf_counter() - judge_start
                            if isinstance(result, int | float):
                                score_value = float(result)
                        except Exception as exc:
                            if judge_elapsed is None:
                                judge_elapsed = time.perf_counter() - judge_start
                            click.echo(f"seed={seed_val} judge[{spec.name}]_error={exc}")
                        judges_timings[spec.name] = judge_elapsed
                        judge_scores[spec.name] = score_value

                if trace_tracer is not None and trace_namespace:
                    storage_metadata = {
                        "eval_seed": seed_val,
                        "prompt_index": prompt_index,
                        "task_id": task_id,
                        "task_split": task_split,
                        "task_rubric_id": task_rubric_id,
                        "official_score": official_score,
                        "judge_scores": judge_scores,
                        "model": selected_model,
                        "prompt": prompt_text,
                        "completion": completion,
                    }
                    await _store_trace(trace_tracer, trace_namespace, storage_metadata)

                records.append(
                    {
                        "seed": seed_val,
                        "prompt_index": prompt_index,
                        "task_id": task_id,
                        "task_split": task_split,
                        "task_rubric_id": task_rubric_id,
                        "official_score": official_score,
                        "judge_scores": judge_scores,
                        "timings": timings,
                    }
                )

            await asyncio.gather(*[_run_seed(seed_val) for seed_val in seed_values])
        finally:
            await async_client.aclose()

    try:
        asyncio.run(_run_eval())
    finally:
        if trace_tracer is not None and trace_tracer.db is not None:
            asyncio.run(trace_tracer.db.close())

    click.echo(
        f"Eval complete: {successes} ok, {failures} failed; model={selected_model}, split={split}"
    )

    if outcome_count > 0:
        mean_outcome = outcome_sum / float(outcome_count)
        frac_right = outcome_correct / float(outcome_count)
        click.echo(
            f"Outcome summary: correct={outcome_correct}/{outcome_count} ({frac_right:.2%}), mean_outcome={mean_outcome:.3f}"
        )

    if records:
        judge_specs = judge_specs or []  # ensure iterable
        official_scores = [
            r["official_score"] for r in records if r["official_score"] is not None
        ]
        if official_scores:
            click.echo(f"  Official mean: {sum(official_scores) / len(official_scores):.3f}")
        else:
            click.echo("  Official mean: n/a")

        for spec in judge_specs:
            spec_scores = [
                record["judge_scores"].get(spec.name)
                for record in records
                if record["judge_scores"].get(spec.name) is not None
            ]
            if spec_scores:
                mean_spec = sum(spec_scores) / len(spec_scores)
                click.echo(f"  [{spec.name}] mean: {mean_spec:.3f}")
            else:
                click.echo(f"  [{spec.name}] mean: n/a")

            paired = [
                (
                    record["official_score"],
                    record["judge_scores"].get(spec.name),
                )
                for record in records
                if record["official_score"] is not None
                and record["judge_scores"].get(spec.name) is not None
            ]
            if len(paired) >= 2:
                corr = _pearson(
                    [p[0] for p in paired if p[0] is not None],
                    [p[1] for p in paired if p[1] is not None],
                )
                if corr is not None:
                    click.echo(f"    Pearson r: {corr:.3f}")
                else:
                    click.echo("    Pearson r: undefined (zero variance)")
            else:
                click.echo("    Pearson r: n/a (need ≥2 paired scores)")

        header = ["Seed", "Prompt", "Official"]
        header.extend(spec.name for spec in judge_specs)
        rows: list[list[str]] = []
        for record in sorted(records, key=lambda r: (r["seed"], r.get("prompt_index") or -1)):
            seed_val = str(record["seed"])
            prompt_idx = (
                str(record["prompt_index"])
                if record["prompt_index"] is not None
                else "-"
            )
            official_val = (
                f"{record['official_score']:.3f}"
                if record["official_score"] is not None
                else "-"
            )
            row = [seed_val, prompt_idx, official_val]
            for spec in judge_specs:
                score_val = record["judge_scores"].get(spec.name)
                row.append(f"{score_val:.3f}" if isinstance(score_val, int | float) else "-")
            rows.append(row)

        widths = [len(col) for col in header]
        for row in rows:
            for idx, cell in enumerate(row):
                widths[idx] = max(widths[idx], len(cell))

        click.echo("")
        click.echo("  ".join(h.ljust(widths[idx]) for idx, h in enumerate(header)))
        click.echo("  ".join("-" * widths[idx] for idx in range(len(header))))
        for row in rows:
            click.echo("  ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row)))



@click.command(
    "filter",
    help="Export filtered tracing sessions to SFT-ready JSONL based on a TOML config.",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(),
    required=True,
    help="Path to TOML config describing the input trace DB, score thresholds, and output JSONL.",
)
def filter_command(config_path: str) -> None:
    """Render tracing sessions that match filter rules into SFT JSONL.

    The TOML file should contain a `[filter]` table with at least:

        db = \"path/to/traces.db\"      # sqlite path or URL (sqlite+aiosqlite://...)
        output = \"ft_data/out.jsonl\"  # destination JSONL

    Optional keys such as `splits`, `task_ids`, `models`, `min_official_score`, or
    `min_judge_scores.my_judge = 0.7` allow you to narrow the dataset down to
    high-quality traces. See `customers/agora_single_file/configs/filter_local.toml`
    for a working example.
    """
    # Parse and validate TOML config
    from synth_ai.task.config import FilterConfig
    
    if _toml is None:
        raise click.ClickException("TOML parser not available; install tomli or use Python 3.11+")

    cfg_path = Path(config_path)
    if not cfg_path.exists():
        raise click.ClickException(f"Filter config not found: {cfg_path}")

    try:
        config_data = _toml.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise click.ClickException(f"Failed to parse TOML '{cfg_path}': {exc}") from exc

    filter_cfg_dict = config_data.get("filter") if isinstance(config_data, dict) else None
    if not isinstance(filter_cfg_dict, dict):
        raise click.ClickException("Config must contain a [filter] table")

    # Validate config with dataclass
    try:
        filter_cfg = FilterConfig.from_dict(filter_cfg_dict)
        click.echo(f"✓ Config validated: db={filter_cfg.db}, output={filter_cfg.output}")
        if filter_cfg.min_official_score is not None:
            click.echo(f"  → Filtering for official score >= {filter_cfg.min_official_score}")
        if filter_cfg.limit:
            click.echo(f"  → Limiting to {filter_cfg.limit} examples")
    except (ValueError, TypeError) as validation_error:
        raise click.ClickException(f"Invalid filter config: {validation_error}") from validation_error

    # Use validated config
    db_url = filter_cfg.get_db_url()
    output_path = filter_cfg.get_output_path()

    # Extract validated fields from dataclass
    splits = set(filter_cfg.splits)
    task_ids = set(filter_cfg.task_ids)
    models = set(filter_cfg.models)
    min_official = filter_cfg.min_official_score
    max_official = filter_cfg.max_official_score
    min_judge_scores = filter_cfg.min_judge_scores
    max_judge_scores = filter_cfg.max_judge_scores
    # Note: min_created_at and max_created_at not yet in FilterConfig dataclass
    min_created = _parse_datetime_for_trace(filter_cfg_dict.get("min_created_at"))
    max_created = _parse_datetime_for_trace(filter_cfg_dict.get("max_created_at"))
    limit = filter_cfg.limit

    def _score_ok(value: Any, min_val: Any, max_val: Any) -> bool:
        try:
            if value is None:
                return min_val is None
            value = float(value)
        except Exception:
            return False
        if min_val is not None and value < float(min_val):
            return False
        return not (max_val is not None and value > float(max_val))

    async def _run_filter() -> None:
        tracer = SessionTracer(db_url=db_url, auto_save=False)
        await tracer.initialize()

        df = await tracer.db.query_traces(
            "SELECT session_id, created_at, metadata FROM session_traces ORDER BY created_at"
        )
        if getattr(df, "empty", True):
            raise click.ClickException("No traces found in database")

        sessions = df.to_dict("records")
        accepted: list[dict[str, Any]] = []

        for row in sessions:
            metadata_raw = row.get("metadata")
            if isinstance(metadata_raw, str):
                try:
                    metadata = json.loads(metadata_raw)
                except Exception:
                    metadata = {}
            elif isinstance(metadata_raw, dict):
                metadata = dict(metadata_raw)
            else:
                metadata = {}

            created_at_raw = row.get("created_at")
            created_at_dt = _parse_datetime_for_trace(created_at_raw)

            session_id = row.get("session_id")

            if splits and metadata.get("task_split") not in splits:
                continue
            if task_ids and metadata.get("task_id") not in task_ids:
                continue
            if models and metadata.get("model") not in models:
                continue

            if min_created and (created_at_dt is None or created_at_dt < min_created):
                continue
            if max_created and (created_at_dt is None or created_at_dt > max_created):
                continue

            # Check against outcome_rewards if score filter is set
            total_reward = None
            achievements_count = None
            if min_official is not None or max_official is not None:
                reward_query = "SELECT total_reward, achievements_count FROM outcome_rewards WHERE session_id = :session_id"
                reward_rows = await tracer.db.query_traces(reward_query, {"session_id": session_id})
                reward_records = reward_rows.to_dict("records") if hasattr(reward_rows, "to_dict") else []
                if reward_records:
                    total_reward = reward_records[0].get("total_reward")
                    achievements_count = reward_records[0].get("achievements_count")
                    if not _score_ok(total_reward, min_official, max_official):
                        continue
                elif min_official is not None:
                    # No reward found, but score filter requires it
                    continue

            judge_scores = metadata.get("judge_scores") or {}
            include = True
            for judge_name, threshold in (min_judge_scores or {}).items():
                if not _score_ok(judge_scores.get(judge_name), threshold, None):
                    include = False
                    break
            if not include:
                continue
            for judge_name, threshold in (max_judge_scores or {}).items():
                if not _score_ok(judge_scores.get(judge_name), None, threshold):
                    include = False
                    break
            if not include:
                continue

            # Query messages for this session
            messages_query = """
                SELECT message_type, content, timestamp 
                FROM messages 
                WHERE session_id = :session_id
                ORDER BY timestamp ASC, id ASC
            """
            msg_df = await tracer.db.query_traces(messages_query, {"session_id": session_id})
            message_rows = msg_df.to_dict("records") if hasattr(msg_df, "to_dict") else []
            
            if not message_rows:
                # Fallback: check if prompt/completion in metadata (old format)
                prompt = metadata.get("prompt") or ""
                completion = metadata.get("completion") or ""
                if prompt and completion:
                    record = {
                        "messages": [
                            {"role": "user", "content": str(prompt)},
                            {"role": "assistant", "content": str(completion)},
                        ],
                        "metadata": {
                            "session_id": session_id,
                            "env_name": metadata.get("env_name"),
                            "policy_name": metadata.get("policy_name"),
                            "seed": metadata.get("seed"),
                            "total_reward": total_reward,
                            "achievements_count": achievements_count,
                            "model": metadata.get("model"),
                            "created_at": created_at_dt.isoformat() if created_at_dt else created_at_raw,
                        },
                    }
                    accepted.append(record)
                continue

            # Extract user/assistant pairs from messages
            for i, msg_row in enumerate(message_rows):
                msg_type = msg_row.get("message_type")
                content_raw = msg_row.get("content")
                
                # Look for user message
                if msg_type in ("user", "policy_user_prompt"):
                    # Find next policy_system_prompt or assistant
                    assistant_msg = None
                    for j in range(i + 1, len(message_rows)):
                        next_type = message_rows[j].get("message_type")
                        if next_type in ("assistant", "policy_system_prompt"):
                            if next_type == "assistant":
                                assistant_msg = message_rows[j]
                            break
                    
                    # Parse content
                    try:
                        user_content = json.loads(content_raw) if isinstance(content_raw, str) else content_raw
                    except Exception:
                        user_content = content_raw
                    
                    # Extract text from structured content
                    def extract_text(content: Any) -> str:
                        if isinstance(content, str):
                            return content
                        if isinstance(content, dict):
                            # Try payload.content for user prompts
                            if "payload" in content and isinstance(content["payload"], dict):
                                payload = content["payload"]
                                if "content" in payload:
                                    return extract_text(payload["content"])
                            # Try common keys
                            for key in ["text", "content", "content_text"]:
                                if key in content:
                                    val = content[key]
                                    if isinstance(val, str):
                                        return val
                            return json.dumps(content)
                        if isinstance(content, list):
                            # Multimodal content - concatenate text parts
                            parts = []
                            for item in content:
                                if isinstance(item, dict) and item.get("type") == "text":
                                    parts.append(item.get("text", ""))
                            return " ".join(parts) if parts else str(content)
                        return str(content)
                    
                    user_text = extract_text(user_content)
                    
                    # For assistant, we might not have it recorded, so use tool calls as completion
                    assistant_text = ""
                    if assistant_msg:
                        assistant_content_raw = assistant_msg.get("content")
                        try:
                            assistant_content = json.loads(assistant_content_raw) if isinstance(assistant_content_raw, str) else assistant_content_raw
                        except Exception:
                            assistant_content = assistant_content_raw
                        assistant_text = extract_text(assistant_content)
                    
                    if not user_text:
                        continue

                    record = {
                        "messages": [
                            {"role": "user", "content": user_text},
                            {"role": "assistant", "content": assistant_text if assistant_text else "[no response recorded]"},
                        ],
                        "metadata": {
                            "session_id": session_id,
                            "env_name": metadata.get("env_name"),
                            "policy_name": metadata.get("policy_name"),
                            "seed": metadata.get("seed"),
                            "total_reward": total_reward,
                            "achievements_count": achievements_count,
                            "model": metadata.get("model"),
                            "created_at": created_at_dt.isoformat() if created_at_dt else created_at_raw,
                        },
                    }
                    accepted.append(record)

        if not accepted:
            raise click.ClickException("No sessions matched the provided filters")

        if limit is not None and limit > 0:
            accepted = accepted[:limit]

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for item in accepted:
                handle.write(json.dumps(item, ensure_ascii=False))
                handle.write("\n")

        click.echo(f"Wrote {len(accepted)} examples -> {output_path}")
        await tracer.db.close()

    asyncio.run(_run_filter())


def register_eval(cli: click.Group) -> None:
    cli.add_command(eval_command)

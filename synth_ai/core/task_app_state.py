import json
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_TASK_APP_SECRET_NAME = "synth-demo-task-app-secret"

__all__ = [
    "DEFAULT_TASK_APP_SECRET_NAME",
    "current_task_app_id",
    "load_demo_dir",
    "load_template_id",
    "now_iso",
    "persist_api_key",
    "persist_demo_dir",
    "persist_env_api_key",
    "persist_task_url",
    "persist_template_id",
    "read_task_app_config",
    "record_task_app",
    "resolve_task_app_entry",
    "task_app_config_path",
    "task_app_id_from_path",
    "update_task_app_entry",
    "write_task_app_config",
]


def task_app_config_path() -> str:
    return os.path.expanduser("~/.synth-ai/task_app_config.json")


def read_task_app_config() -> dict[str, Any]:
    path = task_app_config_path()
    try:
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as handle:
                loaded = json.load(handle) or {}
                if isinstance(loaded, dict):
                    apps = loaded.get("apps")
                    if isinstance(apps, dict):
                        return apps
    except Exception:
        pass
    return {}


def write_task_app_config(apps: dict[str, Any]) -> None:
    payload = {"apps": apps}
    try:
        path = task_app_config_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
    except Exception:
        pass


def now_iso() -> str:
    return (
        datetime.now(UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def task_app_id_from_path(path: str | Path | None) -> str | None:
    if not path:
        return None
    try:
        return str(Path(path).expanduser().resolve())
    except Exception:
        return None


def current_task_app_id() -> str | None:
    try:
        return str(Path.cwd().resolve())
    except Exception:
        return None


def update_task_app_entry(
    path: str | Path | None,
    *,
    template_id: str | None = None,
    mutate: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    task_id = task_app_id_from_path(path)
    if task_id is None:
        task_id = current_task_app_id()
    if task_id is None:
        return {}

    apps = read_task_app_config()
    now = now_iso()
    entry = apps.get(task_id)
    if entry is None:
        entry = {
            "task_app_path": task_id,
            "template_id": template_id,
            "created_at": now,
            "last_used": now,
            "modal": {
                "app_name": None,
                "base_url": None,
                "secret_name": DEFAULT_TASK_APP_SECRET_NAME,
                "created_at": None,
                "last_used": None,
            },
        }
        apps[task_id] = entry
    else:
        entry.setdefault(
            "modal",
            {
                "app_name": None,
                "base_url": None,
                "secret_name": DEFAULT_TASK_APP_SECRET_NAME,
                "created_at": None,
                "last_used": None,
            },
        )
        if template_id is not None:
            entry["template_id"] = template_id

    if mutate is not None:
        mutate(entry)

    entry["last_used"] = now
    write_task_app_config(apps)
    return entry


def record_task_app(
    path: str,
    *,
    template_id: str | None = None,
    secret_name: str | None = None,
) -> None:

    def _mutate(entry: dict[str, Any]) -> None:
        if secret_name:
            modal_block = entry.setdefault(
                "modal",
                {
                    "app_name": None,
                    "base_url": None,
                    "secret_name": DEFAULT_TASK_APP_SECRET_NAME,
                    "created_at": None,
                    "last_used": None,
                },
            )
            modal_block["secret_name"] = secret_name

    update_task_app_entry(path, template_id=template_id, mutate=_mutate if secret_name else None)


def _select_entry(
    *,
    preferred_path: str | None = None,
    predicate: Callable[[dict[str, Any]], bool] | None = None,
) -> tuple[str | None, dict[str, Any]]:
    entries = read_task_app_config()
    if not entries:
        return None, {}

    def _matches(entry: dict[str, Any]) -> bool:
        return predicate(entry) if predicate is not None else True

    if preferred_path:
        entry = entries.get(preferred_path)
        if isinstance(entry, dict) and _matches(entry):
            return preferred_path, entry

    try:
        cwd = str(Path.cwd().resolve())
        entry = entries.get(cwd)
        if isinstance(entry, dict) and _matches(entry):
            return cwd, entry
    except Exception:
        pass

    best_path: str | None = None
    best_entry: dict[str, Any] = {}
    best_ts = ""
    for path, entry in entries.items():
        if not isinstance(entry, dict) or not _matches(entry):
            continue
        ts = str(entry.get("last_used") or "")
        if ts > best_ts:
            best_ts = ts
            best_path = path
            best_entry = entry
    return best_path, best_entry


def resolve_task_app_entry(
    preferred_path: str | None = None,
    *,
    predicate: Callable[[dict[str, Any]], bool] | None = None,
) -> tuple[str | None, dict[str, Any]]:
    path, entry = _select_entry(preferred_path=preferred_path, predicate=predicate)
    return path, entry if isinstance(entry, dict) else {}


def persist_demo_dir(demo_dir: str) -> None:
    def _mutate(entry: dict[str, Any]) -> None:
        entry["is_demo"] = True

    update_task_app_entry(demo_dir, mutate=_mutate)


def load_demo_dir() -> str | None:
    path, _ = _select_entry(predicate=lambda entry: entry.get("is_demo") or entry.get("template_id"))
    return path


def persist_template_id(template_id: str | None) -> None:
    demo_dir = load_demo_dir() or current_task_app_id()
    if demo_dir is None:
        return
    update_task_app_entry(demo_dir, template_id=template_id)


def load_template_id() -> str | None:
    _, entry = _select_entry(predicate=lambda item: item.get("template_id"))
    value = entry.get("template_id") if isinstance(entry, dict) else None
    return str(value) if isinstance(value, str) else None


def persist_api_key(key: str) -> None:
    target = load_demo_dir() or current_task_app_id()

    def _mutate(entry: dict[str, Any]) -> None:
        secrets = entry.setdefault("secrets", {})
        secrets["synth_api_key"] = key

    update_task_app_entry(target, mutate=_mutate)


def persist_env_api_key(key: str, path: str | Path | None = None) -> None:
    target = path or load_demo_dir()

    def _mutate(entry: dict[str, Any]) -> None:
        secrets = entry.setdefault("secrets", {})
        secrets["environment_api_key"] = key

    update_task_app_entry(target, mutate=_mutate)


def _derive_modal_app_name(url: str | None) -> str | None:
    if not url:
        return None
    try:
        from urllib.parse import urlparse

        host = urlparse(url).hostname or ""
        if "--" not in host:
            return None
        suffix = host.split("--", 1)[1]
        core = suffix.split(".modal", 1)[0]
        if core.endswith("-fastapi-app"):
            core = core[: -len("-fastapi-app")]
        return core.strip() or None
    except Exception:
        return None


def persist_task_url(url: str, *, name: str | None = None, path: str | None = None) -> None:
    normalized_url = (url or "").rstrip("/")
    task_id = task_app_id_from_path(path) or current_task_app_id()
    if task_id is None:
        return

    existing = read_task_app_config().get(task_id, {})
    previous_modal = dict(existing.get("modal", {})) if isinstance(existing, dict) else {}

    derived_name = name or _derive_modal_app_name(normalized_url)

    def _mutate(entry: dict[str, Any]) -> None:
        modal_block = entry.setdefault(
            "modal",
            {
                "app_name": None,
                "base_url": None,
                "secret_name": DEFAULT_TASK_APP_SECRET_NAME,
                "created_at": None,
                "last_used": None,
            },
        )
        now = now_iso()
        if modal_block.get("created_at") is None:
            modal_block["created_at"] = now
        modal_block["last_used"] = now
        modal_block["base_url"] = normalized_url
        if derived_name:
            modal_block["app_name"] = derived_name
        modal_block["secret_name"] = DEFAULT_TASK_APP_SECRET_NAME

    entry = update_task_app_entry(path or task_id, mutate=_mutate)

    modal_after = entry.get("modal", {}) if isinstance(entry, dict) else {}
    changed: list[str] = []
    if previous_modal.get("base_url") != modal_after.get("base_url"):
        changed.append("TASK_APP_BASE_URL")
    if derived_name and previous_modal.get("app_name") != modal_after.get("app_name"):
        changed.append("TASK_APP_NAME")
    if previous_modal.get("secret_name") != modal_after.get("secret_name"):
        changed.append("TASK_APP_SECRET_NAME")

    if changed:
        print(f"Saved {', '.join(changed)} to {task_app_config_path()}")
        if "TASK_APP_SECRET_NAME" in changed:
            print(f"TASK_APP_SECRET_NAME={DEFAULT_TASK_APP_SECRET_NAME}")

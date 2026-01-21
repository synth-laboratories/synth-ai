import json
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from synth_ai.core.utils.paths import SYNTH_LOCALAPI_CONFIG_PATH
from synth_ai.core.utils.secure_files import write_private_json

DEFAULT_LOCALAPI_SECRET_NAME = "synth-demo-task-app-secret"

__all__ = [
    "DEFAULT_LOCALAPI_SECRET_NAME",
    "current_localapi_id",
    "load_template_id",
    "now_iso",
    "persist_api_key",
    "persist_env_api_key",
    "persist_localapi_url",
    "persist_template_id",
    "read_localapi_config",
    "record_localapi",
    "resolve_localapi_entry",
    "localapi_config_path",
    "localapi_id_from_path",
    "update_localapi_entry",
    "write_localapi_config",
]


def localapi_config_path() -> str:
    return str(SYNTH_LOCALAPI_CONFIG_PATH)


def read_localapi_config() -> dict[str, Any]:
    path = str(SYNTH_LOCALAPI_CONFIG_PATH)
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


def write_localapi_config(apps: dict[str, Any]) -> None:
    payload = {"apps": apps}
    try:
        path = SYNTH_LOCALAPI_CONFIG_PATH
        write_private_json(path, payload, indent=2, sort_keys=True)
    except Exception:
        pass


def now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def localapi_id_from_path(path: str | Path | None) -> str | None:
    if not path:
        return None
    try:
        return str(Path(path).expanduser().resolve())
    except Exception:
        return None


def current_localapi_id() -> str | None:
    try:
        return str(Path.cwd().resolve())
    except Exception:
        return None


def update_localapi_entry(
    path: str | Path | None,
    *,
    template_id: str | None = None,
    mutate: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    localapi_id = localapi_id_from_path(path)
    if localapi_id is None:
        localapi_id = current_localapi_id()
    if localapi_id is None:
        return {}

    apps = read_localapi_config()
    now = now_iso()
    entry = apps.get(localapi_id)
    if entry is None:
        entry = {
            "localapi_path": localapi_id,
            "template_id": template_id,
            "created_at": now,
            "last_used": now,
            "modal": {
                "app_name": None,
                "base_url": None,
                "secret_name": DEFAULT_LOCALAPI_SECRET_NAME,
                "created_at": None,
                "last_used": None,
            },
        }
        apps[localapi_id] = entry
    else:
        entry.setdefault(
            "modal",
            {
                "app_name": None,
                "base_url": None,
                "secret_name": DEFAULT_LOCALAPI_SECRET_NAME,
                "created_at": None,
                "last_used": None,
            },
        )
        if template_id is not None:
            entry["template_id"] = template_id

    if mutate is not None:
        mutate(entry)

    entry["last_used"] = now
    write_localapi_config(apps)
    return entry


def record_localapi(
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
                    "secret_name": DEFAULT_LOCALAPI_SECRET_NAME,
                    "created_at": None,
                    "last_used": None,
                },
            )
            modal_block["secret_name"] = secret_name

    update_localapi_entry(path, template_id=template_id, mutate=_mutate if secret_name else None)


def _select_entry(
    *,
    preferred_path: str | None = None,
    predicate: Callable[[dict[str, Any]], bool] | None = None,
) -> tuple[str | None, dict[str, Any]]:
    entries = read_localapi_config()
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


def resolve_localapi_entry(
    preferred_path: str | None = None,
    *,
    predicate: Callable[[dict[str, Any]], bool] | None = None,
) -> tuple[str | None, dict[str, Any]]:
    path, entry = _select_entry(preferred_path=preferred_path, predicate=predicate)
    return path, entry if isinstance(entry, dict) else {}


def persist_template_id(template_id: str | None) -> None:
    target = current_localapi_id()
    if target is None:
        return
    update_localapi_entry(target, template_id=template_id)


def load_template_id() -> str | None:
    _, entry = _select_entry(predicate=lambda item: item.get("template_id"))
    value = entry.get("template_id") if isinstance(entry, dict) else None
    return str(value) if isinstance(value, str) else None


def persist_api_key(key: str) -> None:
    target = current_localapi_id()

    def _mutate(entry: dict[str, Any]) -> None:
        secrets = entry.setdefault("secrets", {})
        secrets["synth_api_key"] = key

    update_localapi_entry(target, mutate=_mutate)


def persist_env_api_key(key: str, path: str | Path | None = None) -> None:
    target = path or current_localapi_id()

    def _mutate(entry: dict[str, Any]) -> None:
        secrets = entry.setdefault("secrets", {})
        secrets["environment_api_key"] = key

    update_localapi_entry(target, mutate=_mutate)


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


def persist_localapi_url(url: str, *, name: str | None = None, path: str | None = None) -> None:
    normalized_url = (url or "").rstrip("/")
    localapi_id = localapi_id_from_path(path) or current_localapi_id()
    if localapi_id is None:
        return

    existing = read_localapi_config().get(localapi_id, {})
    previous_modal = dict(existing.get("modal", {})) if isinstance(existing, dict) else {}

    derived_name = name or _derive_modal_app_name(normalized_url)

    def _mutate(entry: dict[str, Any]) -> None:
        modal_block = entry.setdefault(
            "modal",
            {
                "app_name": None,
                "base_url": None,
                "secret_name": DEFAULT_LOCALAPI_SECRET_NAME,
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
        modal_block["secret_name"] = DEFAULT_LOCALAPI_SECRET_NAME

    entry = update_localapi_entry(path or localapi_id, mutate=_mutate)

    modal_after = entry.get("modal", {}) if isinstance(entry, dict) else {}
    changed: list[str] = []
    if previous_modal.get("base_url") != modal_after.get("base_url"):
        changed.append("TASK_APP_BASE_URL")
    if derived_name and previous_modal.get("app_name") != modal_after.get("app_name"):
        changed.append("TASK_APP_NAME")
    if previous_modal.get("secret_name") != modal_after.get("secret_name"):
        changed.append("TASK_APP_SECRET_NAME")

    if changed:
        print(f"Saved {', '.join(changed)} to {localapi_config_path()}")
        if "TASK_APP_SECRET_NAME" in changed:
            print(f"TASK_APP_SECRET_NAME={DEFAULT_LOCALAPI_SECRET_NAME}")

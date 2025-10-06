from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable

import click
import requests

from .utils import CLIResult, http_get, run_cli


@dataclass(slots=True)
class TaskAppHealth:
    ok: bool
    health_status: int | None
    task_info_status: int | None
    detail: str | None = None


def check_task_app_health(base_url: str, api_key: str, *, timeout: float = 10.0) -> TaskAppHealth:
    headers = {"X-API-Key": api_key}
    base = base_url.rstrip("/")
    health_resp = None
    detail_parts: list[str] = []
    try:
        health_resp = http_get(f"{base}/health", headers=headers, timeout=timeout)
        detail_parts.append(f"/health={health_resp.status_code}")
    except requests.RequestException as exc:
        detail_parts.append(f"/health_error={exc}")
    task_resp = None
    try:
        task_resp = http_get(f"{base}/task_info", headers=headers, timeout=timeout)
        detail_parts.append(f"/task_info={task_resp.status_code}")
    except requests.RequestException as exc:
        detail_parts.append(f"/task_info_error={exc}")
    ok = bool(health_resp and health_resp.status_code == 200 and task_resp and task_resp.status_code == 200)
    detail = ", ".join(detail_parts)
    return TaskAppHealth(
        ok=ok,
        health_status=None if health_resp is None else health_resp.status_code,
        task_info_status=None if task_resp is None else task_resp.status_code,
        detail=detail,
    )


@dataclass(slots=True)
class ModalSecret:
    name: str
    value: str


@dataclass(slots=True)
class ModalApp:
    app_id: str
    label: str
    url: str


def _run_modal(args: Iterable[str]) -> CLIResult:
    return run_cli(["modal", *args], timeout=30.0)


def list_modal_secrets(pattern: str | None = None) -> list[str]:
    result = _run_modal(["secret", "list"])
    if result.code != 0:
        raise click.ClickException(f"modal secret list failed: {result.stderr or result.stdout}")
    names: list[str] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("NAME"):
            continue
        parts = line.split()
        name = parts[0]
        if pattern and pattern.lower() not in name.lower():
            continue
        names.append(name)
    return names


def get_modal_secret_value(name: str) -> str:
    result = _run_modal(["secret", "get", name])
    if result.code != 0:
        raise click.ClickException(f"modal secret get {name} failed: {result.stderr or result.stdout}")
    value = result.stdout.strip()
    if not value:
        raise click.ClickException(f"Secret {name} is empty")
    return value


def list_modal_apps(pattern: str | None = None) -> list[ModalApp]:
    result = _run_modal(["app", "list"])
    if result.code != 0:
        raise click.ClickException(f"modal app list failed: {result.stderr or result.stdout}")
    apps: list[ModalApp] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("APP"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        app_id, label, url = parts[0], parts[1], parts[-1]
        if pattern and pattern.lower() not in (label.lower() + url.lower() + app_id.lower()):
            continue
        apps.append(ModalApp(app_id=app_id, label=label, url=url))
    return apps


def format_modal_apps(apps: list[ModalApp]) -> str:
    rows = [f"{idx}) {app.label} {app.url}" for idx, app in enumerate(apps, start=1)]
    return "\n".join(rows)


def format_modal_secrets(names: list[str]) -> str:
    return "\n".join(f"{idx}) {name}" for idx, name in enumerate(names, start=1))


__all__ = [
    "ModalApp",
    "ModalSecret",
    "check_task_app_health",
    "format_modal_apps",
    "format_modal_secrets",
    "get_modal_secret_value",
    "list_modal_apps",
    "list_modal_secrets",
]

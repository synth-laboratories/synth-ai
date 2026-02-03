"""Environment pools SDK — rollouts, pools, queue status."""

from __future__ import annotations

import json
import os
from typing import Any, Iterator

from synth_ai.core.utils.env import get_api_key
from synth_ai.core.utils.urls import BACKEND_URL_BASE, join_url, normalize_backend_base

__all__ = [
    "create_rollout",
    "get_rollout",
    "replay_rollout",
    "stream_rollout_events",
    "list_rollout_artifacts",
    "fetch_artifact",
    "list_pools",
    "get_pool",
    "create_pool",
    "update_pool",
    "delete_pool",
    "get_pool_metrics",
    "get_queue_status",
]

_API_PREFIX = "/api/v1/environment-pools"


def _resolve_base_url(base_url: str | None, *, default: str) -> str:
    if base_url and base_url.strip():
        return normalize_backend_base(base_url)
    return normalize_backend_base(default)


def _resolve_api_key(api_key: str | None) -> str:
    if api_key and api_key.strip():
        return api_key
    try:
        resolved = get_api_key("SYNTH_API_KEY", required=True)
    except Exception:
        resolved = os.environ.get("SYNTH_API_KEY", "").strip()
    if not resolved:
        raise ValueError("api_key is required (provide or set SYNTH_API_KEY)")
    return resolved


def _auth_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


def _url(base: str, path: str) -> str:
    return join_url(base, f"{_API_PREFIX}/{path.lstrip('/')}")


# --- Rollouts ---


def create_rollout(
    *,
    backend_base: str | None = None,
    api_key: str | None = None,
    request: dict[str, Any],
    timeout: float = 120.0,
) -> dict[str, Any]:
    """Create a new rollout."""
    import httpx

    base = _resolve_base_url(backend_base, default=BACKEND_URL_BASE)
    api_key = _resolve_api_key(api_key)
    url = _url(base, "rollouts")
    resp = httpx.post(url, headers=_auth_headers(api_key), json=request, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data if isinstance(data, dict) else {}


def get_rollout(
    rollout_id: str,
    *,
    backend_base: str | None = None,
    api_key: str | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Get rollout status."""
    import httpx

    base = _resolve_base_url(backend_base, default=BACKEND_URL_BASE)
    api_key = _resolve_api_key(api_key)
    url = _url(base, f"rollouts/{rollout_id}")
    resp = httpx.get(url, headers=_auth_headers(api_key), timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data if isinstance(data, dict) else {}


def replay_rollout(
    rollout_id: str,
    *,
    backend_base: str | None = None,
    api_key: str | None = None,
    overrides: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """Replay a rollout with optional overrides."""
    import httpx

    base = _resolve_base_url(backend_base, default=BACKEND_URL_BASE)
    api_key = _resolve_api_key(api_key)
    url = _url(base, f"rollouts/{rollout_id}/replay")
    payload: dict[str, Any] = {}
    if overrides is not None:
        payload["overrides"] = overrides
    if metadata is not None:
        payload["metadata"] = metadata
    resp = httpx.post(url, headers=_auth_headers(api_key), json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data if isinstance(data, dict) else {}


def stream_rollout_events(
    rollout_id: str,
    *,
    backend_base: str | None = None,
    api_key: str | None = None,
    since: str | None = None,
    cursor: str | None = None,
    limit: int | None = None,
    timeout: float | None = None,
) -> Iterator[dict[str, Any]]:
    """Stream rollout events via SSE. Yields parsed event dicts."""
    import httpx

    base = _resolve_base_url(backend_base, default=BACKEND_URL_BASE)
    api_key = _resolve_api_key(api_key)
    url = _url(base, f"rollouts/{rollout_id}/events")
    params: dict[str, str] = {}
    if since is not None:
        params["since"] = since
    if cursor is not None:
        params["cursor"] = cursor
    if limit is not None:
        params["limit"] = str(limit)

    headers = _auth_headers(api_key)
    headers["Accept"] = "text/event-stream"
    headers["Cache-Control"] = "no-cache"

    with httpx.stream(
        "GET",
        url,
        headers=headers,
        params=params,
        timeout=httpx.Timeout(30.0, read=timeout),
    ) as response:
        response.raise_for_status()
        event_data = ""
        event_id = ""
        event_type = ""
        for line in response.iter_lines():
            if line.startswith("id:"):
                event_id = line[3:].strip()
            elif line.startswith("event:"):
                event_type = line[6:].strip()
            elif line.startswith("data:"):
                event_data = line[5:].strip()
            elif line == "":
                if event_data:
                    try:
                        parsed = json.loads(event_data)
                    except json.JSONDecodeError:
                        parsed = event_data
                    evt: dict[str, Any] = {"data": parsed}
                    if event_id:
                        evt["id"] = event_id
                    if event_type:
                        evt["event"] = event_type
                    yield evt
                event_data = ""
                event_id = ""
                event_type = ""


def list_rollout_artifacts(
    rollout_id: str,
    *,
    backend_base: str | None = None,
    api_key: str | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """List artifacts for a rollout."""
    import httpx

    base = _resolve_base_url(backend_base, default=BACKEND_URL_BASE)
    api_key = _resolve_api_key(api_key)
    url = _url(base, f"rollouts/{rollout_id}/artifacts")
    resp = httpx.get(url, headers=_auth_headers(api_key), timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data if isinstance(data, dict) else {}


def fetch_artifact(
    rollout_id: str,
    path: str,
    *,
    backend_base: str | None = None,
    api_key: str | None = None,
    timeout: float = 60.0,
) -> bytes:
    """Fetch a specific artifact by path."""
    import httpx

    base = _resolve_base_url(backend_base, default=BACKEND_URL_BASE)
    api_key = _resolve_api_key(api_key)
    url = _url(base, f"rollouts/{rollout_id}/artifacts/{path.lstrip('/')}")
    resp = httpx.get(url, headers=_auth_headers(api_key), timeout=timeout)
    resp.raise_for_status()
    return resp.content


# --- Pools ---


def list_pools(
    *,
    backend_base: str | None = None,
    api_key: str | None = None,
    timeout: float = 30.0,
) -> list[dict[str, Any]]:
    """List all pools."""
    import httpx

    base = _resolve_base_url(backend_base, default=BACKEND_URL_BASE)
    api_key = _resolve_api_key(api_key)
    url = _url(base, "pools")
    resp = httpx.get(url, headers=_auth_headers(api_key), timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data if isinstance(data, list) else []


def get_pool(
    pool_id: str,
    *,
    backend_base: str | None = None,
    api_key: str | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Get pool details."""
    import httpx

    base = _resolve_base_url(backend_base, default=BACKEND_URL_BASE)
    api_key = _resolve_api_key(api_key)
    url = _url(base, f"pools/{pool_id}")
    resp = httpx.get(url, headers=_auth_headers(api_key), timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data if isinstance(data, dict) else {}


def create_pool(
    *,
    backend_base: str | None = None,
    api_key: str | None = None,
    request: dict[str, Any],
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Create a new pool.

    The ``request`` dict should include at minimum ``pool_type`` (one of
    ``"sandbox"``, ``"browser"``, ``"openenv"``, ``"archipelago"``) and
    optionally ``tasks`` — a list of task instance definitions that declare
    which backend resources and configurations each instance uses (datasets,
    docker images, openenv configs, archipelago setups, etc.).
    """
    import httpx

    base = _resolve_base_url(backend_base, default=BACKEND_URL_BASE)
    api_key = _resolve_api_key(api_key)
    url = _url(base, "pools")
    resp = httpx.post(url, headers=_auth_headers(api_key), json=request, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data if isinstance(data, dict) else {}


def update_pool(
    pool_id: str,
    *,
    backend_base: str | None = None,
    api_key: str | None = None,
    request: dict[str, Any],
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Update pool configuration.

    The ``request`` dict may include any mutable pool fields such as
    ``capacity``, ``concurrency``, ``tasks``, etc.
    """
    import httpx

    base = _resolve_base_url(backend_base, default=BACKEND_URL_BASE)
    api_key = _resolve_api_key(api_key)
    url = _url(base, f"pools/{pool_id}")
    resp = httpx.put(url, headers=_auth_headers(api_key), json=request, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data if isinstance(data, dict) else {}


def delete_pool(
    pool_id: str,
    *,
    backend_base: str | None = None,
    api_key: str | None = None,
    timeout: float = 30.0,
) -> None:
    """Delete a pool."""
    import httpx

    base = _resolve_base_url(backend_base, default=BACKEND_URL_BASE)
    api_key = _resolve_api_key(api_key)
    url = _url(base, f"pools/{pool_id}")
    resp = httpx.delete(url, headers=_auth_headers(api_key), timeout=timeout)
    resp.raise_for_status()


def get_pool_metrics(
    pool_id: str,
    *,
    backend_base: str | None = None,
    api_key: str | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Get pool metrics including queue depth and running count."""
    import httpx

    base = _resolve_base_url(backend_base, default=BACKEND_URL_BASE)
    api_key = _resolve_api_key(api_key)
    url = _url(base, f"pools/{pool_id}/metrics")
    resp = httpx.get(url, headers=_auth_headers(api_key), timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data if isinstance(data, dict) else {}


# --- Queue ---


def get_queue_status(
    *,
    backend_base: str | None = None,
    api_key: str | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Get queue status."""
    import httpx

    base = _resolve_base_url(backend_base, default=BACKEND_URL_BASE)
    api_key = _resolve_api_key(api_key)
    url = _url(base, "queue/status")
    resp = httpx.get(url, headers=_auth_headers(api_key), timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data if isinstance(data, dict) else {}

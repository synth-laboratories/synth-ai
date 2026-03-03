"""SynthTunnel relay-based tunnel client and agent (SDK side).

SynthTunnel exposes a local container to Synth's training infrastructure
without requiring ``cloudflared`` or any external binary. Traffic flows
through Synth's relay servers via a WebSocket agent that runs locally.

Architecture::

    Synth backend ──HTTP──▶ relay (st.usesynth.ai) ──WS──▶ local agent ──HTTP──▶ localhost:PORT

The agent connects outbound (no inbound ports needed), receives requests
over the WebSocket, forwards them to the local container, and streams
responses back. Up to 128 concurrent in-flight requests are supported,
with a dynamic memory budget that automatically reduces concurrency when
request/response payloads are large.

For most users, use ``TunneledContainer.create()`` instead of this module
directly. This module is the low-level implementation.
"""

from __future__ import annotations

import logging
import os
import secrets
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import httpx

from synth_ai.core.tunnels.errors import (
    TunnelErrorCode,
    TunnelProviderError,
    map_problem_to_tunnel_error_code,
)
from synth_ai.core.utils.urls import normalize_backend_base, resolve_synth_backend_url

logger = logging.getLogger(__name__)


def _client_instance_id_path() -> Path:
    home = Path(os.environ.get("HOME", ""))
    return home / ".synth" / "client_instance_id"


def get_client_instance_id() -> str:
    try:
        path = _client_instance_id_path()
        if path.exists():
            value = path.read_text().strip()
            if len(value) >= 8:
                return value
        path.parent.mkdir(parents=True, exist_ok=True)
        new_id = f"client-{secrets.token_hex(8)}"
        path.write_text(new_id)
        return new_id
    except Exception:
        strict_id = f"session-{secrets.token_hex(8)}"
        logger.warning("[SynthTunnel] Failed to persist client_instance_id; using %s", strict_id)
        return strict_id


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _raise_for_problem_response(resp: httpx.Response, *, provider: str) -> None:
    if resp.status_code < 400:
        return
    payload: dict[str, Any] = {}
    try:
        parsed = resp.json()
        if isinstance(parsed, dict):
            payload = parsed
    except Exception:
        payload = {}
    detail = str(payload.get("detail") or payload.get("message") or "").strip()
    backend_code = str(payload.get("code") or "").strip() or None
    request_id = str(payload.get("request_id") or "").strip() or None
    mapped_code = map_problem_to_tunnel_error_code(backend_code, detail)
    # Refine common SynthTunnel lifecycle cases.
    detail_lc = detail.lower()
    if "lease_pending" in detail_lc:
        mapped_code = TunnelErrorCode.LEASE_PENDING
    elif "lease_expired" in detail_lc:
        mapped_code = TunnelErrorCode.LEASE_EXPIRED
    elif "agent_offline" in detail_lc:
        mapped_code = TunnelErrorCode.AGENT_OFFLINE

    raise TunnelProviderError(
        f"{provider} backend request failed (HTTP {resp.status_code}).",
        code=mapped_code,
        status=resp.status_code,
        request_id=request_id,
        backend_code=backend_code,
        provider=provider,
        hint=detail or resp.text[:200],
    )


def _required_response_str(payload: Dict[str, Any], field: str) -> str:
    raw = payload.get(field)
    if raw is None:
        raise RuntimeError(f"SynthTunnel lease response missing required field: {field}")
    value = str(raw).strip()
    if not value:
        raise RuntimeError(f"SynthTunnel lease response has empty required field: {field}")
    return value


def _parse_lease_response(data: Dict[str, Any]) -> SynthTunnelLease:
    if not isinstance(data, dict):
        raise RuntimeError("SynthTunnel lease response must be a JSON object")

    agent_connect = data.get("agent_connect")
    if not isinstance(agent_connect, dict):
        raise RuntimeError("SynthTunnel lease response missing agent connection details")

    expires_raw = _required_response_str(data, "expires_at")
    try:
        expires_at = _parse_datetime(expires_raw)
    except Exception as exc:
        raise RuntimeError(
            f"SynthTunnel lease response has invalid expires_at: {expires_raw}"
        ) from exc

    limits = data.get("limits", {}) or {}
    heartbeat = data.get("heartbeat", {}) or {}
    if not isinstance(limits, dict):
        limits = {}
    if not isinstance(heartbeat, dict):
        heartbeat = {}

    return SynthTunnelLease(
        lease_id=_required_response_str(data, "lease_id"),
        route_token=_required_response_str(data, "route_token"),
        public_base_url=_required_response_str(data, "public_base_url"),
        public_url=_required_response_str(data, "public_url"),
        agent_url=_required_response_str(agent_connect, "url"),
        agent_token=_required_response_str(agent_connect, "agent_token"),
        worker_token=_required_response_str(data, "worker_token"),
        expires_at=expires_at,
        limits=limits,
        heartbeat=heartbeat,
    )


def _collect_container_keys(primary: Optional[str]) -> list[str]:
    # Legacy compatibility shim: relay path uses worker-token gated auth.
    return [primary.strip()] if primary and primary.strip() else []


@dataclass
class SynthTunnelLease:
    lease_id: str
    route_token: str
    public_base_url: str
    public_url: str
    agent_url: str
    agent_token: str
    worker_token: str
    expires_at: datetime
    limits: Dict[str, Any]
    heartbeat: Dict[str, Any]


class SynthTunnelClient:
    def __init__(self, api_key: str, backend_url: Optional[str] = None) -> None:
        if not api_key or not str(api_key).strip():
            raise ValueError("api_key is required to create SynthTunnel leases")
        self.api_key = api_key
        self.backend_url = normalize_backend_base(resolve_synth_backend_url(backend_url)).rstrip(
            "/"
        )

    async def create_lease(
        self,
        *,
        client_instance_id: str,
        local_host: str,
        local_port: int,
        requested_ttl_seconds: int = 3600,
        metadata: Optional[Dict[str, Any]] = None,
        capabilities: Optional[Dict[str, Any]] = None,
    ) -> SynthTunnelLease:
        url = f"{self.backend_url}/api/v1/synthtunnel/leases"
        payload = {
            "client_instance_id": client_instance_id,
            "local_target": {"host": local_host, "port": local_port},
            "requested_ttl_seconds": requested_ttl_seconds,
            "metadata": metadata or {},
            "capabilities": capabilities or {},
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        timeout_sec = float(os.environ.get("SYNTH_TUNNEL_TIMEOUT_SEC", "30"))
        async with httpx.AsyncClient(timeout=timeout_sec) as client:
            resp = await client.post(url, json=payload, headers=headers)
            _raise_for_problem_response(resp, provider="SynthTunnel")
            data = resp.json()
        return _parse_lease_response(data)

    async def close_lease(self, lease_id: str) -> None:
        url = f"{self.backend_url}/api/v1/synthtunnel/leases/{lease_id}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        timeout_sec = float(os.environ.get("SYNTH_TUNNEL_TIMEOUT_SEC", "30"))
        async with httpx.AsyncClient(timeout=timeout_sec) as client:
            resp = await client.delete(url, headers=headers)
            _raise_for_problem_response(resp, provider="SynthTunnel")


def hostname_from_url(url: str) -> str:
    try:
        return urlparse(url).hostname or ""
    except Exception:
        return ""

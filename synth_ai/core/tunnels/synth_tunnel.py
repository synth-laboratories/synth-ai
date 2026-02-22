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

import asyncio
import base64
import binascii
import json
import logging
import os
import secrets
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import aiohttp
import httpx

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
        fallback_id = f"session-{secrets.token_hex(8)}"
        logger.warning("[SynthTunnel] Failed to persist client_instance_id; using %s", fallback_id)
        return fallback_id


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _encode_bytes(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _decode_bytes(data: str | None) -> bytes:
    if not data:
        return b""
    return base64.b64decode(data.encode("ascii"))


def _normalize_header_pairs(raw_headers: Any) -> list[tuple[str, str]]:
    normalized: list[tuple[str, str]] = []
    if isinstance(raw_headers, dict):
        source = raw_headers.items()
    else:
        source = raw_headers or []
    for item in source:
        if isinstance(item, tuple) and len(item) == 2:
            key, value = item
        elif isinstance(item, list) and len(item) == 2:
            key, value = item
        else:
            continue
        if key is None or value is None:
            continue
        normalized.append((str(key), str(value)))
    return normalized


def _strip_hop_by_hop(headers: Any) -> list[tuple[str, str]]:
    normalized = _normalize_header_pairs(headers)
    blocked = {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "authorization",
        "host",
        "x-synthtunnel-worker-token",
        "x-forwarded-host",
        "x-forwarded-proto",
        "x-forwarded-port",
        "x-forwarded-for",
        "forwarded",
        "content-length",
    }
    return [(k, v) for k, v in normalized if k.lower() not in blocked]


def _has_header(headers: list[tuple[str, str]], name: str) -> bool:
    lowered = name.lower()
    return any(key.lower() == lowered for key, _ in headers)


def _encode_response_headers(headers: httpx.Headers) -> list[list[str]]:
    return [[str(k), str(v)] for k, v in headers.multi_items()]


def _request_timeout_from_deadline_ms(deadline_ms: int) -> httpx.Timeout:
    # Respect relay-provided request deadline while keeping connect/write bounded.
    total_seconds = max(float(deadline_ms) / 1000.0, 0.05)
    connect_seconds = min(10.0, total_seconds)
    write_seconds = min(10.0, total_seconds)
    return httpx.Timeout(total_seconds, connect=connect_seconds, read=total_seconds, write=write_seconds)


def _required_response_str(payload: Dict[str, Any], field: str) -> str:
    raw = payload.get(field)
    if raw is None:
        raise RuntimeError(f"SynthTunnel lease response missing required field: {field}")
    value = str(raw).strip()
    if not value:
        raise RuntimeError(f"SynthTunnel lease response has empty required field: {field}")
    return value


def _parse_lease_response(data: Dict[str, Any]) -> "SynthTunnelLease":
    if not isinstance(data, dict):
        raise RuntimeError("SynthTunnel lease response must be a JSON object")

    agent_connect = data.get("agent_connect")
    if not isinstance(agent_connect, dict):
        raise RuntimeError("SynthTunnel lease response missing agent connection details")

    expires_raw = _required_response_str(data, "expires_at")
    try:
        expires_at = _parse_datetime(expires_raw)
    except Exception as exc:
        raise RuntimeError(f"SynthTunnel lease response has invalid expires_at: {expires_raw}") from exc

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
    keys: list[str] = []
    if primary and primary.strip():
        keys.append(primary.strip())
    env_primary = (os.environ.get("ENVIRONMENT_API_KEY") or "").strip()
    if env_primary and env_primary not in keys:
        keys.append(env_primary)
    dev_primary = (os.environ.get("DEV_ENVIRONMENT_API_KEY") or "").strip()
    if dev_primary and dev_primary not in keys:
        keys.append(dev_primary)
    aliases_raw = (os.environ.get("ENVIRONMENT_API_KEY_ALIASES") or "").strip()
    if aliases_raw:
        for part in aliases_raw.split(","):
            candidate = part.strip()
            if candidate and candidate not in keys:
                keys.append(candidate)
    return keys


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
            resp.raise_for_status()
            data = resp.json()
        return _parse_lease_response(data)

    async def close_lease(self, lease_id: str) -> None:
        url = f"{self.backend_url}/api/v1/synthtunnel/leases/{lease_id}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        timeout_sec = float(os.environ.get("SYNTH_TUNNEL_TIMEOUT_SEC", "30"))
        async with httpx.AsyncClient(timeout=timeout_sec) as client:
            resp = await client.delete(url, headers=headers)
            resp.raise_for_status()


@dataclass
class RequestContext:
    lease_id: str
    rid: str
    method: str
    path: str
    query: str
    headers: list[tuple[str, str]]
    deadline_ms: int
    body: bytearray = field(default_factory=bytearray)


class SynthTunnelAgent:
    def __init__(
        self,
        *,
        lease: SynthTunnelLease,
        local_host: str,
        local_port: int,
        stop_event: asyncio.Event,
        container_key: Optional[str] = None,
    ) -> None:
        self.lease = lease
        self.local_host = local_host
        self.local_port = local_port
        self.stop_event = stop_event
        self._contexts: Dict[str, RequestContext] = {}
        self._tasks: Dict[str, asyncio.Task] = {}
        self._send_lock = asyncio.Lock()
        self._container_keys = _collect_container_keys(container_key)
        if not self._container_keys:
            logger.warning(
                "[SynthTunnel] ENVIRONMENT_API_KEY not set; forwarding without local auth headers."
            )

    def _attach_local_auth(self, headers: list[tuple[str, str]]) -> None:
        if not self._container_keys:
            return
        if not _has_header(headers, "x-api-key"):
            headers.append(("X-API-Key", self._container_keys[0]))
        if len(self._container_keys) > 1 and not _has_header(headers, "x-api-keys"):
            headers.append(("X-API-Keys", ",".join(self._container_keys)))
        if not _has_header(headers, "authorization"):
            headers.append(("Authorization", f"Bearer {self._container_keys[0]}"))

    async def _send(self, ws: aiohttp.ClientWebSocketResponse, payload: dict[str, Any]) -> None:
        async with self._send_lock:
            await ws.send_str(json.dumps(payload))

    @staticmethod
    def _try_parse_ws_payload(raw_text: str) -> dict[str, Any] | None:
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

    async def _handle_req_body_frame(
        self, ws: aiohttp.ClientWebSocketResponse, payload: dict[str, Any]
    ) -> None:
        rid = str(payload.get("rid"))
        ctx = self._contexts.get(rid)
        if not ctx:
            return
        try:
            chunk = _decode_bytes(payload.get("chunk_b64"))
        except (binascii.Error, ValueError):
            logger.warning(
                "[SynthTunnel] Ignoring malformed REQ_BODY chunk rid=%s",
                rid,
            )
            self._contexts.pop(rid, None)
            await self._send(
                ws,
                {
                    "type": "RESP_ERROR",
                    "lease_id": ctx.lease_id,
                    "rid": rid,
                    "code": "BAD_REQUEST_FRAME",
                    "message": "Malformed REQ_BODY chunk",
                },
            )
            return
        ctx.body.extend(chunk)

    async def _drain_reconnect_state(self) -> None:
        if self._contexts:
            self._contexts.clear()
        if not self._tasks:
            return
        tasks = list(self._tasks.values())
        self._tasks.clear()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _handle_request(
        self, ws: aiohttp.ClientWebSocketResponse, ctx: RequestContext
    ) -> None:
        base_url = f"http://{self.local_host}:{self.local_port}"
        url = f"{base_url}{ctx.path}"
        if ctx.query:
            url = f"{url}?{ctx.query}"

        headers = _strip_hop_by_hop(ctx.headers)
        self._attach_local_auth(headers)
        if not _has_header(headers, "x-synthtunnel-lease-id"):
            headers.append(("x-synthtunnel-lease-id", ctx.lease_id))
        if not _has_header(headers, "x-synthtunnel-request-id"):
            headers.append(("x-synthtunnel-request-id", ctx.rid))
        if not _has_header(headers, "x-forwarded-proto"):
            headers.append(("x-forwarded-proto", "https"))

        timeout = _request_timeout_from_deadline_ms(ctx.deadline_ms)
        max_response_bytes = int(self.lease.limits.get("max_response_bytes", 200_000_000))
        sent_bytes = 0

        print(
            f"[SynthTunnel] _handle_request rid={ctx.rid} method={ctx.method} url={url} body_len={len(ctx.body)}",
            flush=True,
        )
        try:
            async with (
                httpx.AsyncClient(timeout=timeout) as client,
                client.stream(
                    ctx.method,
                    url,
                    headers=headers,
                    content=bytes(ctx.body),
                ) as resp,
            ):
                print(
                    f"[SynthTunnel] local API responded rid={ctx.rid} status={resp.status_code}",
                    flush=True,
                )
                await self._send(
                    ws,
                    {
                        "type": "RESP_HEADERS",
                        "lease_id": ctx.lease_id,
                        "rid": ctx.rid,
                        "status": resp.status_code,
                        "headers": _encode_response_headers(resp.headers),
                    },
                )
                async for chunk in resp.aiter_bytes():
                    if not chunk:
                        continue
                    sent_bytes += len(chunk)
                    if sent_bytes > max_response_bytes:
                        await self._send(
                            ws,
                            {
                                "type": "RESP_ERROR",
                                "lease_id": ctx.lease_id,
                                "rid": ctx.rid,
                                "code": "LOCAL_BAD_RESPONSE",
                                "message": "Response too large",
                            },
                        )
                        return
                    await self._send(
                        ws,
                        {
                            "type": "RESP_BODY",
                            "lease_id": ctx.lease_id,
                            "rid": ctx.rid,
                            "chunk_b64": _encode_bytes(chunk),
                            "eof": False,
                        },
                    )
            await self._send(ws, {"type": "RESP_END", "lease_id": ctx.lease_id, "rid": ctx.rid})
        except Exception as exc:
            print(f"[SynthTunnel] _handle_request FAILED rid={ctx.rid} error={exc}", flush=True)
            await self._send(
                ws,
                {
                    "type": "RESP_ERROR",
                    "lease_id": ctx.lease_id,
                    "rid": ctx.rid,
                    "code": "LOCAL_CONNECT_FAILED",
                    "message": str(exc),
                },
            )
        finally:
            self._tasks.pop(ctx.rid, None)

    async def run(self) -> None:
        headers = {"Authorization": f"Bearer {self.lease.agent_token}"}
        async with aiohttp.ClientSession() as session:
            while not self.stop_event.is_set():
                try:
                    async with session.ws_connect(
                        self.lease.agent_url, headers=headers, heartbeat=20
                    ) as ws:
                        await self._send(
                            ws,
                            {
                                "type": "ATTACH",
                                "agent_id": get_client_instance_id(),
                                "leases": [
                                    {
                                        "lease_id": self.lease.lease_id,
                                        "local_target": {
                                            "host": self.local_host,
                                            "port": self.local_port,
                                        },
                                    }
                                ],
                                "capabilities": {
                                    "streaming": True,
                                    "max_inflight": self.lease.limits.get("max_inflight", 16),
                                    "max_body_chunk_bytes": 65536,
                                    "supports_cancel": True,
                                },
                            },
                        )
                        async for msg in ws:
                            if self.stop_event.is_set():
                                break
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                payload = self._try_parse_ws_payload(msg.data)
                                if payload is None:
                                    logger.warning(
                                        "[SynthTunnel] Ignoring malformed WS text frame (invalid JSON object), len=%d",
                                        len(msg.data) if isinstance(msg.data, str) else 0,
                                    )
                                    continue
                                msg_type = payload.get("type")
                                rid_log = payload.get("rid", "?")
                                print(
                                    f"[SynthTunnel] recv frame type={msg_type} rid={rid_log} len={len(msg.data)}",
                                    flush=True,
                                )
                                if msg_type == "REQ_HEADERS":
                                    ctx = RequestContext(
                                        lease_id=str(payload.get("lease_id")),
                                        rid=str(payload.get("rid")),
                                        method=str(payload.get("method", "GET")),
                                        path=str(payload.get("path", "/")),
                                        query=str(payload.get("query", "")),
                                        headers=_normalize_header_pairs(payload.get("headers") or []),
                                        deadline_ms=int(payload.get("deadline_ms") or 600000),
                                    )
                                    self._contexts[ctx.rid] = ctx
                                elif msg_type == "REQ_BODY":
                                    await self._handle_req_body_frame(ws, payload)
                                elif msg_type == "REQ_END":
                                    rid = str(payload.get("rid"))
                                    ctx = self._contexts.pop(rid, None)
                                    if ctx:
                                        self._tasks[rid] = asyncio.create_task(
                                            self._handle_request(ws, ctx)
                                        )
                                elif msg_type == "CANCEL":
                                    rid = str(payload.get("rid"))
                                    self._contexts.pop(rid, None)
                                    task = self._tasks.pop(rid, None)
                                    if task:
                                        task.cancel()
                            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                                break
                        if self.stop_event.is_set():
                            break
                except Exception as exc:
                    logger.warning("[SynthTunnel] Agent reconnecting after error: %s", exc)
                    await self._drain_reconnect_state()
                    await asyncio.sleep(1.0)
                    continue
                await self._drain_reconnect_state()


@dataclass
class SynthTunnelSession:
    lease: SynthTunnelLease
    agent: SynthTunnelAgent
    task: asyncio.Task
    client: SynthTunnelClient
    stop_event: asyncio.Event

    async def close_async(self) -> None:
        self.stop_event.set()
        if not self.task.done():
            self.task.cancel()
        try:
            await self.client.close_lease(self.lease.lease_id)
        except Exception as exc:
            logger.warning("[SynthTunnel] Failed to close lease %s: %s", self.lease.lease_id, exc)

    def close(self) -> None:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self.close_async())
            return
        raise RuntimeError("close() cannot be called from an async context; await close_async().")


async def open_synth_tunnel(
    *,
    local_host: str,
    local_port: int,
    api_key: str,
    backend_url: Optional[str] = None,
    container_key: Optional[str] = None,
) -> SynthTunnelSession:
    if not api_key or not str(api_key).strip():
        raise ValueError("api_key is required to open SynthTunnel")
    client = SynthTunnelClient(api_key, backend_url=backend_url)
    lease = await client.create_lease(
        client_instance_id=get_client_instance_id(),
        local_host=local_host,
        local_port=local_port,
    )
    stop_event = asyncio.Event()
    agent = SynthTunnelAgent(
        lease=lease,
        local_host=local_host,
        local_port=local_port,
        stop_event=stop_event,
        container_key=container_key,
    )
    task = asyncio.create_task(agent.run())
    return SynthTunnelSession(
        lease=lease,
        agent=agent,
        task=task,
        client=client,
        stop_event=stop_event,
    )


def hostname_from_url(url: str) -> str:
    try:
        return urlparse(url).hostname or ""
    except Exception:
        return ""

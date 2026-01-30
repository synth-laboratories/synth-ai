"""SynthTunnel relay-based tunnel client and agent (SDK side).

SynthTunnel exposes a local task app to Synth's training infrastructure
without requiring ``cloudflared`` or any external binary. Traffic flows
through Synth's relay servers via a WebSocket agent that runs locally.

Architecture::

    Synth backend ──HTTP──▶ relay (st.usesynth.ai) ──WS──▶ local agent ──HTTP──▶ localhost:PORT

The agent connects outbound (no inbound ports needed), receives requests
over the WebSocket, forwards them to the local task app, and streams
responses back. Up to 128 concurrent in-flight requests are supported,
with a dynamic memory budget that automatically reduces concurrency when
request/response payloads are large.

For most users, use ``TunneledLocalAPI.create()`` instead of this module
directly. This module is the low-level implementation.
"""

from __future__ import annotations

import asyncio
import base64
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

logger = logging.getLogger(__name__)

DEFAULT_BACKEND_URL = "https://api.usesynth.ai"


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


def _strip_hop_by_hop(headers: Dict[str, str]) -> Dict[str, str]:
    hop_by_hop = {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
    }
    return {k: v for k, v in headers.items() if k.lower() not in hop_by_hop}


def _collect_local_api_keys(primary: Optional[str]) -> list[str]:
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
        self.backend_url = (
            backend_url or os.getenv("SYNTH_BACKEND_URL") or DEFAULT_BACKEND_URL
        ).rstrip("/")

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
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        agent_connect = data.get("agent_connect", {}) or {}
        if not agent_connect.get("url") or not agent_connect.get("agent_token"):
            raise RuntimeError("SynthTunnel lease response missing agent connection details")
        return SynthTunnelLease(
            lease_id=str(data.get("lease_id")),
            route_token=str(data.get("route_token")),
            public_base_url=str(data.get("public_base_url")),
            public_url=str(data.get("public_url")),
            agent_url=str(agent_connect.get("url")),
            agent_token=str(agent_connect.get("agent_token")),
            worker_token=str(data.get("worker_token")),
            expires_at=_parse_datetime(str(data.get("expires_at"))),
            limits=data.get("limits", {}) or {},
            heartbeat=data.get("heartbeat", {}) or {},
        )

    async def close_lease(self, lease_id: str) -> None:
        url = f"{self.backend_url}/api/v1/synthtunnel/leases/{lease_id}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.delete(url, headers=headers)
            resp.raise_for_status()


@dataclass
class RequestContext:
    lease_id: str
    rid: str
    method: str
    path: str
    query: str
    headers: Dict[str, str]
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
        local_api_key: Optional[str] = None,
    ) -> None:
        self.lease = lease
        self.local_host = local_host
        self.local_port = local_port
        self.stop_event = stop_event
        self._contexts: Dict[str, RequestContext] = {}
        self._tasks: Dict[str, asyncio.Task] = {}
        self._send_lock = asyncio.Lock()
        self._local_api_keys = _collect_local_api_keys(local_api_key)
        if not self._local_api_keys:
            logger.warning(
                "[SynthTunnel] ENVIRONMENT_API_KEY not set; forwarding without local auth headers."
            )

    def _attach_local_auth(self, headers: Dict[str, str]) -> None:
        if not self._local_api_keys:
            return
        headers.setdefault("X-API-Key", self._local_api_keys[0])
        if len(self._local_api_keys) > 1 and "X-API-Keys" not in headers:
            headers["X-API-Keys"] = ",".join(self._local_api_keys)
        headers.setdefault("Authorization", f"Bearer {self._local_api_keys[0]}")

    async def _send(self, ws: aiohttp.ClientWebSocketResponse, payload: dict[str, Any]) -> None:
        async with self._send_lock:
            await ws.send_str(json.dumps(payload))

    async def _handle_request(
        self, ws: aiohttp.ClientWebSocketResponse, ctx: RequestContext
    ) -> None:
        base_url = f"http://{self.local_host}:{self.local_port}"
        url = f"{base_url}{ctx.path}"
        if ctx.query:
            url = f"{url}?{ctx.query}"

        headers = _strip_hop_by_hop(ctx.headers)
        self._attach_local_auth(headers)
        headers.setdefault("x-synthtunnel-lease-id", ctx.lease_id)
        headers.setdefault("x-synthtunnel-request-id", ctx.rid)
        headers.setdefault("x-forwarded-proto", "https")

        timeout = httpx.Timeout(60.0, connect=10.0, read=60.0, write=10.0)
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
                        "headers": [[k, v] for k, v in resp.headers.items()],
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
                                payload = json.loads(msg.data)
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
                                        headers={
                                            str(k): str(v)
                                            for k, v in (payload.get("headers") or [])
                                            if k
                                        },
                                        deadline_ms=int(payload.get("deadline_ms") or 600000),
                                    )
                                    self._contexts[ctx.rid] = ctx
                                elif msg_type == "REQ_BODY":
                                    rid = str(payload.get("rid"))
                                    ctx = self._contexts.get(rid)
                                    if ctx:
                                        ctx.body.extend(_decode_bytes(payload.get("chunk_b64")))
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
                    await asyncio.sleep(1.0)


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
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self.close_async())
            return
        loop.create_task(self.close_async())


async def open_synth_tunnel(
    *,
    local_host: str,
    local_port: int,
    api_key: str,
    backend_url: Optional[str] = None,
    local_api_key: Optional[str] = None,
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
        local_api_key=local_api_key,
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

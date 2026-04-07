"""Managed tunnels SDK."""

from __future__ import annotations

import asyncio
import builtins
import os
from enum import Enum
from typing import Any

from synth_ai.core.utils.urls import BACKEND_URL_BASE, join_url, normalize_backend_base

__all__ = [
    "AsyncTunnelsClient",
    "TunnelProvider",
    "TunnelsClient",
]


class TunnelProvider(str, Enum):
    CLOUDFLARED = "cloudflared"
    NGROK = "ngrok"


class _AsyncThreadProxy:
    def __init__(self, sync_obj: Any) -> None:
        self._sync_obj = sync_obj
        self._proxy_cache: dict[str, Any] = {}

    def __getattr__(self, name: str) -> Any:
        cached = self._proxy_cache.get(name)
        if cached is not None:
            return cached
        attr = getattr(self._sync_obj, name)
        if callable(attr):

            async def _wrapped(*args: Any, **kwargs: Any) -> Any:
                return await asyncio.to_thread(attr, *args, **kwargs)

            self._proxy_cache[name] = _wrapped
            return _wrapped
        return attr


class TunnelsClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        backend_base: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._api_key = (api_key or os.getenv("SYNTH_API_KEY") or "").strip()
        if not self._api_key:
            raise ValueError("api_key is required (provide explicitly or set SYNTH_API_KEY)")
        resolved_base = backend_base or base_url or BACKEND_URL_BASE
        self._backend_base = normalize_backend_base(resolved_base)
        self._timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        import httpx

        resp = httpx.request(
            method,
            join_url(self._backend_base, path),
            headers=self._headers(),
            json=json_body,
            params=params,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        if not resp.content:
            return {}
        return resp.json()

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/v1/tunnels/health")

    def list(
        self,
        *,
        status_filter: str | None = None,
        include_deleted: bool = False,
    ) -> builtins.list[dict[str, Any]]:
        params = {
            k: v
            for k, v in {
                "status_filter": status_filter,
                "include_deleted": include_deleted if include_deleted else None,
            }.items()
            if v is not None
        }
        return self._request("GET", "/v1/tunnels/", params=params)

    def create(
        self,
        *,
        subdomain: str,
        local_port: int,
        local_host: str = "127.0.0.1",
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/v1/tunnels/",
            json_body={
                "subdomain": subdomain,
                "local_port": local_port,
                "local_host": local_host,
            },
        )

    def delete(self, tunnel_id: str) -> dict[str, Any]:
        return self._request("DELETE", f"/v1/tunnels/{tunnel_id}")

    def rotate(
        self,
        *,
        local_port: int = 8000,
        local_host: str = "127.0.0.1",
        reason: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"local_port": local_port, "local_host": local_host}
        if reason:
            body["reason"] = reason
        return self._request("POST", "/v1/tunnels/rotate", json_body=body)

    def create_lease(
        self,
        *,
        client_instance_id: str,
        local_host: str,
        local_port: int,
        app_name: str | None = None,
        provider_preference: TunnelProvider | str = TunnelProvider.NGROK,
        requested_ttl_seconds: int = 3600,
        reuse_connector: bool = True,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "client_instance_id": client_instance_id,
            "local_host": local_host,
            "local_port": local_port,
            "provider_preference": getattr(provider_preference, "value", provider_preference),
            "requested_ttl_seconds": requested_ttl_seconds,
            "reuse_connector": reuse_connector,
        }
        if app_name:
            body["app_name"] = app_name
        if idempotency_key:
            body["idempotency_key"] = idempotency_key
        return self._request("POST", "/v1/tunnels/lease", json_body=body)

    def heartbeat(
        self,
        lease_id: str,
        *,
        connected_to_edge: bool = False,
        gateway_ready: bool = False,
        local_ready: bool = False,
        last_error: str | None = None,
    ) -> dict[str, Any]:
        body = {
            "connected_to_edge": connected_to_edge,
            "gateway_ready": gateway_ready,
            "local_ready": local_ready,
            "last_error": last_error,
        }
        return self._request("POST", f"/v1/tunnels/lease/{lease_id}/heartbeat", json_body=body)

    def release_lease(self, lease_id: str) -> dict[str, Any]:
        return self._request("POST", f"/v1/tunnels/lease/{lease_id}/release")

    def refresh_lease(
        self,
        lease_id: str,
        *,
        requested_ttl_seconds: int = 3600,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/v1/tunnels/lease/{lease_id}/refresh",
            json_body={"requested_ttl_seconds": requested_ttl_seconds},
        )

    def delete_lease(self, lease_id: str) -> dict[str, Any]:
        return self._request("DELETE", f"/v1/tunnels/lease/{lease_id}")

    def list_leases(
        self,
        *,
        client_instance_id: str | None = None,
        include_expired: bool = False,
    ) -> builtins.list[dict[str, Any]]:
        params = {
            k: v
            for k, v in {
                "client_instance_id": client_instance_id,
                "include_expired": include_expired if include_expired else None,
            }.items()
            if v is not None
        }
        return self._request("GET", "/v1/tunnels/lease", params=params)

    def create_synth_lease(
        self,
        *,
        client_instance_id: str,
        local_host: str,
        local_port: int,
        requested_ttl_seconds: int = 3600,
        metadata: dict[str, Any] | None = None,
        capabilities: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/api/v1/synthtunnel/leases",
            json_body={
                "client_instance_id": client_instance_id,
                "local_target": {"host": local_host, "port": local_port},
                "requested_ttl_seconds": requested_ttl_seconds,
                "metadata": metadata or {},
                "capabilities": capabilities or {},
            },
        )

    def get_synth_lease(self, lease_id: str) -> dict[str, Any]:
        return self._request("GET", f"/api/v1/synthtunnel/leases/{lease_id}")

    def close_synth_lease(self, lease_id: str) -> dict[str, Any]:
        return self._request("DELETE", f"/api/v1/synthtunnel/leases/{lease_id}")

    def refresh_synth_worker_token(self, lease_id: str) -> dict[str, Any]:
        return self._request("POST", f"/api/v1/synthtunnel/leases/{lease_id}/token:refresh")


class AsyncTunnelsClient(_AsyncThreadProxy):
    """Async adapter over ``TunnelsClient``."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        backend_base: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        super().__init__(
            TunnelsClient(
                api_key=api_key,
                backend_base=backend_base,
                base_url=base_url,
                timeout=timeout,
            )
        )

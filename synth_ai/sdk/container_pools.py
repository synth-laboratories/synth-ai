"""Canonical container pools client.

Targets consolidated `/v1/pools/*` endpoints.
"""

from __future__ import annotations

import json
import os
from typing import Any, Iterator, Literal, Optional

from synth_ai.core.utils.urls import BACKEND_URL_BASE, join_url, normalize_backend_base


class ContainerPoolsClient:
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        backend_base: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        self._api_key = api_key or os.getenv("SYNTH_API_KEY", "").strip()
        if not self._api_key:
            raise ValueError("api_key is required (provide explicitly or set SYNTH_API_KEY)")
        self._backend_base = normalize_backend_base(backend_base or BACKEND_URL_BASE)
        self._timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Optional[dict[str, Any]] = None,
        params: Optional[dict[str, Any]] = None,
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
        data = resp.json()
        return data

    def _request_with_fallback(
        self,
        primary_method: str,
        fallback_method: str,
        path: str,
        *,
        json_body: Optional[dict[str, Any]] = None,
        params: Optional[dict[str, Any]] = None,
    ) -> Any:
        import httpx

        url = join_url(self._backend_base, path)
        headers = self._headers()

        resp = httpx.request(
            primary_method,
            url,
            headers=headers,
            json=json_body,
            params=params,
            timeout=self._timeout,
        )
        if resp.status_code in (404, 405):
            resp = httpx.request(
                fallback_method,
                url,
                headers=headers,
                json=json_body,
                params=params,
                timeout=self._timeout,
            )
        resp.raise_for_status()
        if not resp.content:
            return {}
        return resp.json()

    # Uploads
    def create_upload(
        self,
        *,
        filename: Optional[str] = None,
        content_type: Optional[str] = None,
        expires_in_seconds: Optional[int] = None,
        auto_create_data_source: bool = False,
        auto_create_timeout_sec: Optional[int] = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "auto_create_data_source": auto_create_data_source,
        }
        if filename is not None:
            payload["filename"] = filename
        if content_type is not None:
            payload["content_type"] = content_type
        if expires_in_seconds is not None:
            payload["expires_in_seconds"] = expires_in_seconds
        if auto_create_timeout_sec is not None:
            payload["auto_create_timeout_sec"] = auto_create_timeout_sec
        return self._request("POST", "/v1/pools/uploads", json_body=payload)

    def get_upload(self, upload_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/pools/uploads/{upload_id}")

    # Data sources
    def create_data_source(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/v1/pools/data-sources", json_body=request)

    def list_data_sources(
        self,
        *,
        state: Optional[str] = None,
        limit: int = 100,
        cursor: Optional[str] = None,
    ) -> dict[str, Any]:
        params = {"state": state, "limit": limit, "cursor": cursor}
        params = {k: v for k, v in params.items() if v is not None}
        return self._request("GET", "/v1/pools/data-sources", params=params)

    def get_data_source(self, data_source_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/pools/data-sources/{data_source_id}")

    def update_data_source(self, data_source_id: str, request: dict[str, Any]) -> dict[str, Any]:
        return self._request_with_fallback(
            "PATCH",
            "PUT",
            f"/v1/pools/data-sources/{data_source_id}",
            json_body=request,
        )

    def refresh_data_source(self, data_source_id: str) -> dict[str, Any]:
        return self._request("POST", f"/v1/pools/data-sources/{data_source_id}/refresh", json_body={})

    # Assemblies
    def create_assembly(
        self,
        *,
        data_source_id: str,
        exclusion_patterns: Optional[list[str]] = None,
        runtime_type: Literal["custom_container", "managed_template"] = "custom_container",
        template_name: Optional[str] = None,
        agent_model: Optional[str] = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "data_source_id": data_source_id,
            "runtime_type": runtime_type,
        }
        if exclusion_patterns is not None:
            payload["exclusion_patterns"] = exclusion_patterns
        if agent_model is not None:
            payload["agent_model"] = agent_model
        if runtime_type == "managed_template" and template_name:
            payload["template_name"] = template_name
            # Keep alias during migration while rust/backend still accept target_hint.
            payload["target_hint"] = template_name
        return self._request("POST", "/v1/pools/assemblies", json_body=payload)

    def list_assemblies(
        self,
        *,
        state: Optional[str] = None,
        limit: int = 100,
        cursor: Optional[str] = None,
    ) -> dict[str, Any]:
        params = {"state": state, "limit": limit, "cursor": cursor}
        params = {k: v for k, v in params.items() if v is not None}
        return self._request("GET", "/v1/pools/assemblies", params=params)

    def get_assembly(self, assembly_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/pools/assemblies/{assembly_id}")

    def stream_assembly_events(self, assembly_id: str, *, cursor: Optional[str] = None) -> Iterator[dict[str, Any]]:
        import httpx

        params = {"cursor": cursor} if cursor is not None else None
        with httpx.stream(
            "GET",
            join_url(self._backend_base, f"/v1/pools/assemblies/{assembly_id}/events"),
            headers=self._headers(),
            params=params,
            timeout=self._timeout,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                text = line.decode("utf-8") if isinstance(line, (bytes, bytearray)) else str(line)
                if text.startswith("data:"):
                    payload = text[5:].strip()
                    if payload:
                        yield json.loads(payload)

    # Pools
    def create_pool(self, request: dict[str, Any]) -> dict[str, Any]:
        assembly_id = request.get("assembly_id")
        if not isinstance(assembly_id, str) or not assembly_id.strip():
            raise ValueError("create_pool requires request['assembly_id']")
        return self._request("POST", "/v1/pools", json_body=request)

    def list_pools(self, *, state: Optional[str] = None, limit: int = 100, cursor: Optional[str] = None) -> dict[str, Any]:
        params = {"state": state, "limit": limit, "cursor": cursor}
        params = {k: v for k, v in params.items() if v is not None}
        return self._request("GET", "/v1/pools", params=params)

    def get_pool(self, pool_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/pools/{pool_id}")

    def update_pool(self, pool_id: str, request: dict[str, Any]) -> dict[str, Any]:
        return self._request_with_fallback(
            "PATCH",
            "PUT",
            f"/v1/pools/{pool_id}",
            json_body=request,
        )

    def delete_pool(self, pool_id: str) -> dict[str, Any]:
        return self._request("DELETE", f"/v1/pools/{pool_id}")

    def reassemble_pool(
        self,
        pool_id: str,
        *,
        exclusion_patterns: Optional[list[str]] = None,
        runtime_type: Literal["custom_container", "managed_template"] = "custom_container",
        template_name: Optional[str] = None,
        agent_model: Optional[str] = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"runtime_type": runtime_type}
        if exclusion_patterns is not None:
            payload["exclusion_patterns"] = exclusion_patterns
        if agent_model is not None:
            payload["agent_model"] = agent_model
        if runtime_type == "managed_template" and template_name:
            payload["template_name"] = template_name
            payload["target_hint"] = template_name
        return self._request("POST", f"/v1/pools/{pool_id}/assemblies", json_body=payload)

    # Rollouts
    def create_rollout(self, pool_id: str, request: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", f"/v1/pools/{pool_id}/rollouts", json_body=request)

    def get_rollout(self, pool_id: str, rollout_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/pools/{pool_id}/rollouts/{rollout_id}")

    def list_rollouts(self, pool_id: str, *, state: Optional[str] = None, limit: int = 100, cursor: Optional[str] = None) -> dict[str, Any]:
        params = {"state": state, "limit": limit, "cursor": cursor}
        params = {k: v for k, v in params.items() if v is not None}
        return self._request("GET", f"/v1/pools/{pool_id}/rollouts", params=params)

    def cancel_rollout(self, pool_id: str, rollout_id: str) -> dict[str, Any]:
        return self._request("POST", f"/v1/pools/{pool_id}/rollouts/{rollout_id}/cancel", json_body={})

    def get_rollout_artifacts(self, pool_id: str, rollout_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/pools/{pool_id}/rollouts/{rollout_id}/artifacts")

    def get_rollout_usage(self, pool_id: str, rollout_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/pools/{pool_id}/rollouts/{rollout_id}/usage")

    def stream_rollout_events(self, pool_id: str, rollout_id: str, *, cursor: Optional[str] = None) -> Iterator[dict[str, Any]]:
        import httpx

        params = {"cursor": cursor} if cursor is not None else None
        with httpx.stream(
            "GET",
            join_url(self._backend_base, f"/v1/pools/{pool_id}/rollouts/{rollout_id}/events"),
            headers=self._headers(),
            params=params,
            timeout=self._timeout,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                text = line.decode("utf-8") if isinstance(line, (bytes, bytearray)) else str(line)
                if text.startswith("data:"):
                    payload = text[5:].strip()
                    if payload:
                        yield json.loads(payload)


__all__ = ["ContainerPoolsClient"]

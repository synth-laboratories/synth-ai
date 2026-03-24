"""Managed Pools compatibility helpers.

Deprecated wrappers retained through the published 2026-10-01 removal date.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any, Mapping

from synth_ai.core.errors import PlanGatingError
from synth_ai.core.utils.env import get_api_key
from synth_ai.core.utils.urls import BACKEND_URL_BASE, join_url, normalize_backend_base
from synth_ai.sdk.container.auth import encrypt_for_backend

__all__ = [
    "create_managed_pool_upload_url",
    "upload_managed_pool_bytes",
    "upload_managed_pool_file",
    "create_managed_pool_upload_data_source",
    "create_managed_pool_s3_data_source",
]


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


def _ensure_prefix(prefix: str) -> str:
    return prefix.lstrip("/")


_UPGRADE_URL = "https://usesynth.ai/pricing"


def _warn_managed_pools_deprecated() -> None:
    warnings.warn(
        "managed_pools helper functions are deprecated and will be removed on 2026-10-01. "
        "Use ContainerPoolsClient upload/data_source methods.",
        DeprecationWarning,
        stacklevel=2,
    )


def _raise_for_status_with_plan_check(response: Any) -> None:
    """Convert HTTP 403 plan-gating responses into :class:`PlanGatingError`."""
    status_code = getattr(response, "status_code", None)
    if status_code == 403:
        try:
            data = response.json()
        except Exception:
            data = {}
        error = data.get("error", data)
        code = error.get("code", "")
        if code in ("plan_required", "feature_not_available", "upgrade_required") or (
            "plan" in str(error.get("message", "")).lower()
            or "upgrade" in str(error.get("message", "")).lower()
        ):
            plan = error.get("current_plan", error.get("tier", "free"))
            raise PlanGatingError(
                feature="managed_pools",
                current_plan=str(plan),
                required_plans=("pro", "team"),
                upgrade_url=error.get("upgrade_url", _UPGRADE_URL),
            )
    response.raise_for_status()


def _post_canonical(
    *,
    base: str,
    api_key: str,
    path: str,
    payload: dict[str, Any],
    timeout: float,
) -> Any:
    import httpx

    return httpx.post(
        join_url(base, path),
        headers=_auth_headers(api_key),
        json=payload,
        timeout=timeout,
    )


def create_managed_pool_upload_url(
    *,
    backend_base: str | None = None,
    api_key: str | None = None,
    filename: str | None = None,
    content_type: str | None = None,
    expires_in_seconds: int | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Request a presigned upload URL for managed pool data."""
    _warn_managed_pools_deprecated()

    base = _resolve_base_url(backend_base, default=BACKEND_URL_BASE)
    resolved_api_key = _resolve_api_key(api_key)
    payload: dict[str, Any] = {}
    if filename:
        payload["filename"] = filename
    if content_type:
        payload["content_type"] = content_type
    if expires_in_seconds is not None:
        payload["expires_in_seconds"] = expires_in_seconds

    response = _post_canonical(
        base=base,
        api_key=resolved_api_key,
        path="/v1/pools/uploads",
        payload=payload,
        timeout=timeout,
    )
    _raise_for_status_with_plan_check(response)
    data = response.json()
    return data if isinstance(data, dict) else {}


def upload_managed_pool_bytes(
    upload_url: str,
    data: bytes,
    *,
    content_type: str | None = None,
    timeout: float = 60.0,
) -> None:
    """Upload raw bytes to a managed pool presigned URL."""
    _warn_managed_pools_deprecated()
    import httpx

    headers = {}
    if content_type:
        headers["Content-Type"] = content_type
    response = httpx.put(upload_url, headers=headers, content=data, timeout=timeout)
    response.raise_for_status()


def upload_managed_pool_file(
    upload_url: str,
    file_path: str | Path,
    *,
    content_type: str | None = None,
    timeout: float = 60.0,
) -> None:
    """Upload a local file to a managed pool presigned URL."""
    _warn_managed_pools_deprecated()
    path = Path(file_path)
    upload_managed_pool_bytes(
        upload_url,
        path.read_bytes(),
        content_type=content_type,
        timeout=timeout,
    )


def create_managed_pool_upload_data_source(
    *,
    backend_base: str | None = None,
    api_key: str | None = None,
    upload_id: str,
    upload_key: str,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Create a managed pool data source from a presigned upload."""
    _warn_managed_pools_deprecated()

    base = _resolve_base_url(backend_base, default=BACKEND_URL_BASE)
    resolved_api_key = _resolve_api_key(api_key)
    response = _post_canonical(
        base=base,
        api_key=resolved_api_key,
        path="/v1/pools/data-sources",
        payload={
            "type": "upload",
            "upload_id": upload_id,
            "upload_key": upload_key,
        },
        timeout=timeout,
    )
    _raise_for_status_with_plan_check(response)
    data = response.json()
    return data if isinstance(data, dict) else {}


def create_managed_pool_s3_data_source(
    *,
    backend_base: str | None = None,
    crypto_base: str | None = None,
    api_key: str | None = None,
    bucket: str,
    prefix: str,
    credentials: Mapping[str, str | None],
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Create a managed pool S3 data source with encrypted credentials."""
    _warn_managed_pools_deprecated()
    import httpx

    resolved_backend_base = _resolve_base_url(backend_base, default=BACKEND_URL_BASE)
    resolved_crypto_base = _resolve_base_url(crypto_base, default=BACKEND_URL_BASE)
    resolved_api_key = _resolve_api_key(api_key)

    public_key_response = httpx.get(
        join_url(resolved_crypto_base, "/api/v1/crypto/public-key"),
        headers=_auth_headers(resolved_api_key),
        timeout=timeout,
    )
    _raise_for_status_with_plan_check(public_key_response)
    public_key_payload = public_key_response.json()
    if not isinstance(public_key_payload, dict):
        raise RuntimeError("backend response missing public_key")
    public_key = public_key_payload.get("public_key")
    if not isinstance(public_key, str) or not public_key:
        raise RuntimeError("backend response missing public_key")

    ciphertext_b64 = encrypt_for_backend(
        public_key,
        json.dumps(
            {
                "access_key_id": credentials.get("access_key_id"),
                "secret_access_key": credentials.get("secret_access_key"),
                "session_token": credentials.get("session_token"),
            }
        ),
    )

    response = _post_canonical(
        base=resolved_backend_base,
        api_key=resolved_api_key,
        path="/v1/pools/data-sources",
        payload={
            "type": "s3",
            "bucket": bucket,
            "prefix": _ensure_prefix(prefix),
            "credentials": {
                "ciphertext_b64": ciphertext_b64,
                "alg": public_key_payload.get("alg"),
            },
        },
        timeout=timeout,
    )
    _raise_for_status_with_plan_check(response)
    data = response.json()
    return data if isinstance(data, dict) else {}

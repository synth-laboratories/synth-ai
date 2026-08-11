"""Workload-side client for an injected managed-inference capability."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any
from uuid import uuid4

import httpx


class ManagedInferenceClient:
    """A narrow OpenAI-compatible client that never mints authority."""

    def __init__(self, *, base_url: str, capability: str, timeout_seconds: float = 180.0) -> None:
        normalized_url = base_url.strip().rstrip("/")
        if not normalized_url or not capability.strip():
            raise ValueError("managed inference base_url and capability are required")
        self._base_url = normalized_url
        self._capability = capability.strip()
        self._timeout_seconds = timeout_seconds

    @classmethod
    def from_environment(cls, *, timeout_seconds: float = 180.0) -> ManagedInferenceClient:
        base_url = os.environ.get("SMR_METERED_INFERENCE_BASE_URL") or os.environ.get(
            "OPENAI_BASE_URL", ""
        )
        capability = os.environ.get("SMR_METERED_INFERENCE_API_KEY") or os.environ.get(
            "SMR_INFERENCE_CAPABILITY", ""
        )
        return cls(base_url=base_url, capability=capability, timeout_seconds=timeout_seconds)

    def _headers(self, idempotency_key: str | None = None) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._capability}",
            "Idempotency-Key": idempotency_key or uuid4().hex,
        }

    def models(self) -> dict[str, Any]:
        response = httpx.get(
            f"{self._base_url}/models",
            headers=self._headers(),
            timeout=self._timeout_seconds,
        )
        response.raise_for_status()
        return response.json()

    def limits(self) -> dict[str, Any]:
        response = httpx.get(
            f"{self._base_url}/limits",
            headers=self._headers(),
            timeout=self._timeout_seconds,
        )
        response.raise_for_status()
        return response.json()

    def chat_completions(
        self, payload: Mapping[str, Any], *, idempotency_key: str | None = None
    ) -> dict[str, Any]:
        return self._post("chat/completions", payload, idempotency_key=idempotency_key)

    def responses(
        self, payload: Mapping[str, Any], *, idempotency_key: str | None = None
    ) -> dict[str, Any]:
        return self._post("responses", payload, idempotency_key=idempotency_key)

    def _post(
        self,
        path: str,
        payload: Mapping[str, Any],
        *,
        idempotency_key: str | None,
    ) -> dict[str, Any]:
        if payload.get("stream") is True:
            raise ValueError("use the OpenAI/Codex client for managed SSE streaming")
        response = httpx.post(
            f"{self._base_url}/{path}",
            headers=self._headers(idempotency_key),
            json=dict(payload),
            timeout=self._timeout_seconds,
        )
        response.raise_for_status()
        result = response.json()
        if not isinstance(result, dict):
            raise RuntimeError("managed inference response must be an object")
        return result


__all__ = ["ManagedInferenceClient"]

"""Synth-wrapped AsyncAnthropic client with shared HTTP pool."""

from __future__ import annotations

import uuid
from typing import Any

try:
    from anthropic import AsyncAnthropic as _AsyncAnthropic
except Exception as exc:  # pragma: no cover - optional dependency
    _AsyncAnthropic = object  # type: ignore[assignment]
    _ANTHROPIC_IMPORT_ERROR: Exception | None = exc
else:  # pragma: no cover
    _ANTHROPIC_IMPORT_ERROR = None

from synth_ai.core.utils.urls import resolve_synth_interceptor_base_url
from synth_ai.sdk.localapi._impl.http_pool import get_shared_http_client

DEFAULT_INTERCEPTOR_BASE_URL = resolve_synth_interceptor_base_url()


class AsyncAnthropic(_AsyncAnthropic):
    """Anthropic client that auto-routes through the Synth interceptor."""

    def __init__(
        self,
        *,
        trial_id: str,
        correlation_id: str | None = None,
        interceptor_base_url: str = DEFAULT_INTERCEPTOR_BASE_URL,
        api_key: str | None = None,
        synth_api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        if _ANTHROPIC_IMPORT_ERROR is not None:
            raise ImportError(
                "Optional dependency 'anthropic' is required to use synth_ai.sdk.clients.anthropic."
            ) from _ANTHROPIC_IMPORT_ERROR

        if correlation_id is None:
            correlation_id = f"corr_{uuid.uuid4().hex[:12]}"

        base_url = kwargs.pop("base_url", None)
        if base_url is None:
            base_url = f"{interceptor_base_url}/{trial_id}/{correlation_id}"

        http_client = kwargs.pop("http_client", None) or get_shared_http_client()

        super().__init__(
            base_url=base_url,
            api_key=api_key or "synth-interceptor",
            http_client=http_client,
            **kwargs,
        )

        self._trial_id = trial_id
        self._correlation_id = correlation_id
        self._synth_api_key = synth_api_key

    @property
    def trial_id(self) -> str:
        return self._trial_id

    @property
    def correlation_id(self) -> str:
        return self._correlation_id

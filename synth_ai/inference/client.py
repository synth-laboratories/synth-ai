from __future__ import annotations

from typing import Any

from synth_ai.api.models.supported import (
    UnsupportedModelError,
    normalize_model_identifier,
)

from ..http import AsyncHttpClient


class InferenceClient:
    def __init__(self, base_url: str, api_key: str, *, timeout: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout

    async def create_chat_completion(
        self, *, model: str, messages: list[dict], **kwargs: Any
    ) -> dict[str, Any]:
        try:
            normalized_model = normalize_model_identifier(model)
        except UnsupportedModelError as exc:
            raise ValueError(str(exc)) from exc

        body: dict[str, Any] = {"model": normalized_model, "messages": messages}
        body.update(kwargs)
        # Backend now expects an explicit thinking_budget; provide a sensible default if omitted
        if "thinking_budget" not in body:
            body["thinking_budget"] = 256
        async with AsyncHttpClient(self._base_url, self._api_key, timeout=self._timeout) as http:
            # Route through backend inference proxy to Modal
            return await http.post_json("/api/inference/v1/chat/completions", json=body)

from __future__ import annotations

from typing import Any, Dict

from ..http import AsyncHttpClient


class InferenceClient:
    def __init__(self, base_url: str, api_key: str, *, timeout: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout

    async def create_chat_completion(
        self, *, model: str, messages: list[dict], **kwargs: Any
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {"model": model, "messages": messages}
        body.update(kwargs)
        # Backend now expects an explicit thinking_budget; provide a sensible default if omitted
        if "thinking_budget" not in body:
            body["thinking_budget"] = 256
        async with AsyncHttpClient(self._base_url, self._api_key, timeout=self._timeout) as http:
            # Public learning-v2 inference path mounted under /api/v1
            return await http.post_json("/api/v1/chat/completions", json=body)

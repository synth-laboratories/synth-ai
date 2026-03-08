"""Inference client for model inference via Synth AI."""

from __future__ import annotations

import asyncio
from typing import Any

from synth_ai.sdk.shared.models import UnsupportedModelError, normalize_model_identifier

try:
    import synth_ai_py
except Exception as exc:  # pragma: no cover
    raise RuntimeError("synth_ai_py is required for sdk.inference.") from exc


def _require_rust() -> Any:
    if synth_ai_py is None or not hasattr(synth_ai_py, "SynthClient"):
        raise RuntimeError("Rust core inference client required; synth_ai_py is unavailable.")
    return synth_ai_py


class InferenceClient:
    """Client for making inference requests through Synth AI's inference proxy."""

    def __init__(self, base_url: str, api_key: str, *, timeout: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout

    async def create_chat_completion(
        self, *, model: str, messages: list[dict[str, Any]], **kwargs: Any
    ) -> dict[str, Any]:
        try:
            normalized_model = normalize_model_identifier(model)
        except UnsupportedModelError as exc:
            raise ValueError(str(exc)) from exc

        body: dict[str, Any] = {"model": normalized_model, "messages": messages}
        body.update(kwargs)
        if "thinking_budget" not in body:
            body["thinking_budget"] = 256
        rust = _require_rust()
        client = rust.SynthClient(self._api_key, self._base_url)
        return await asyncio.to_thread(client.inference_chat_completion, body)

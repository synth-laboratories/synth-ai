from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence


class InferenceAPIClient:
    """Async client that normalizes chat completion calls across providers."""

    def __init__(self, *, provider: str, inference_url: Optional[str] = None, timeout: float = 30.0) -> None:
        self.provider = (provider or "").lower()
        self.inference_url = inference_url.rstrip("/") if inference_url else None
        self.timeout = timeout

        if self.provider == "openai":
            try:
                from openai import AsyncOpenAI
            except Exception as exc:  # pragma: no cover - import guard
                raise RuntimeError("openai package not installed") from exc

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY must be set for meta-model proposals")

            kwargs: Dict[str, Any] = {"api_key": api_key}
            if self.inference_url:
                kwargs["base_url"] = self.inference_url
            self._client = AsyncOpenAI(**kwargs)

        elif self.provider == "groq":
            try:
                from groq import AsyncGroq
            except Exception as exc:  # pragma: no cover - import guard
                raise RuntimeError("groq package not installed") from exc

            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise RuntimeError("GROQ_API_KEY must be set for meta-model proposals")

            kwargs: Dict[str, Any] = {"api_key": api_key}
            if self.inference_url:
                kwargs["base_url"] = self.inference_url
            self._client = AsyncGroq(**kwargs)

        else:
            raise RuntimeError(f"Unsupported meta-model provider: {provider}")

    async def chat_completion(
        self,
        *,
        messages: Sequence[Dict[str, Any]],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> Dict[str, Any]:
        if self.provider == "openai":
            response = await self._client.chat.completions.create(  # type: ignore[attr-defined]
                model=model,
                messages=self._normalize_messages(messages),
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=self.timeout,
            )
            return self._model_dump(response)

        if self.provider == "groq":
            response = await self._client.chat.completions.create(  # type: ignore[attr-defined]
                model=model,
                messages=self._normalize_messages(messages),
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=self.timeout,
            )
            return self._model_dump(response)

        raise RuntimeError(f"Unsupported meta-model provider: {self.provider}")

    @staticmethod
    def _normalize_messages(messages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [dict(message) for message in messages]

    @staticmethod
    def _model_dump(obj: Any) -> Dict[str, Any]:
        if hasattr(obj, "model_dump"):
            data = obj.model_dump()
        elif hasattr(obj, "dict"):
            data = obj.dict()
        else:
            data = dict(obj)  # type: ignore[arg-type]

        usage = getattr(obj, "usage", None)
        if usage and "usage" not in data:
            if hasattr(usage, "model_dump"):
                data["usage"] = usage.model_dump()
            elif isinstance(usage, dict):
                data["usage"] = usage
        return data


__all__ = ["InferenceAPIClient"]

from __future__ import annotations

import json
import os
import time
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

        elif self.provider == "google":
            try:
                import google.genai as genai
            except Exception as exc:  # pragma: no cover - import guard
                raise RuntimeError("google-genai package not installed") from exc

            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY must be set for Google/Gemini models")

            # Set API key for SDK
            os.environ["GEMINI_API_KEY"] = api_key
            # Google GenAI SDK doesn't use base_url like OpenAI/Groq
            self._client = genai.Client()

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

        if self.provider == "google":
            return await self._google_chat_completion(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        raise RuntimeError(f"Unsupported meta-model provider: {self.provider}")

    async def _google_chat_completion(
        self,
        *,
        messages: Sequence[Dict[str, Any]],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> Dict[str, Any]:
        """Handle Google Gemini chat completion."""
        from google.genai import types

        client = self._client

        # Convert messages format (OpenAI -> Gemini)
        contents = []
        system_instruction = None
        for msg in messages:
            role = msg.get("role", "").lower()
            content = msg.get("content", "")
            if role == "system":
                system_instruction = content
            elif role in ["user", "assistant"]:
                # Gemini uses "user" and "model" roles (not "assistant")
                gemini_role = "user" if role == "user" else "model"
                contents.append(types.Content(role=gemini_role, parts=[types.Part.from_text(text=str(content))]))

        # Build generation config
        cfg_kwargs: Dict[str, Any] = {"temperature": temperature}
        if system_instruction:
            cfg_kwargs["system_instruction"] = system_instruction
        if max_tokens:
            cfg_kwargs["max_output_tokens"] = max_tokens

        generation_config = types.GenerateContentConfig(**cfg_kwargs)

        # Make the API call
        resp = await client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=generation_config,
        )

        # Convert response format (Gemini -> OpenAI)
        # Extract text
        try:
            text = resp.text
            if text is None:
                text = ""
        except (ValueError, AttributeError):
            text = ""

        # Extract tool calls (if any)
        tool_calls = []
        if resp.candidates and resp.candidates[0].content:
            for part in resp.candidates[0].content.parts:
                if part.function_call:
                    tool_calls.append({
                        "id": f"call_{len(tool_calls) + 1}",
                        "type": "function",
                        "function": {
                            "name": part.function_call.name,
                            "arguments": json.dumps(dict(part.function_call.args)),
                        },
                    })

        # Build OpenAI-compatible response
        message = {"role": "assistant", "content": text or ""}
        if tool_calls:
            message["tool_calls"] = tool_calls

        # Estimate token usage (Gemini doesn't always provide this in the same format)
        # Simple approximation: 1 token per 4 characters
        prompt_text = "\n".join(str(m.get("content", "")) for m in messages)
        prompt_tokens = max(1, len(prompt_text) // 4)
        completion_tokens = max(1, len(text or "") // 4)

        response_data = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

        return response_data

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


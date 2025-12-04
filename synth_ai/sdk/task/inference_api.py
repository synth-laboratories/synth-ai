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
        tools: Sequence[Dict[str, Any]] | None = None,
        tool_choice: str | Dict[str, Any] | None = None,
        response_format: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        if self.provider == "openai":
            kwargs: Dict[str, Any] = {
                "model": model,
                "messages": self._normalize_messages(messages),
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timeout": self.timeout,
            }
            if tools:
                kwargs["tools"] = list(tools)
            if tool_choice is not None:
                kwargs["tool_choice"] = tool_choice
            if response_format:
                kwargs["response_format"] = response_format
            response = await self._client.chat.completions.create(**kwargs)  # type: ignore[attr-defined]
            return self._model_dump(response)

        if self.provider == "groq":
            groq_kwargs: Dict[str, Any] = {
                "model": model,
                "messages": self._normalize_messages(messages),
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timeout": self.timeout,
            }
            if tools:
                groq_kwargs["tools"] = list(tools)
            if tool_choice is not None:
                groq_kwargs["tool_choice"] = tool_choice
            if response_format:
                # Groq response_format normalization (json_schema -> json_object for older models)
                from .proxy import normalize_response_format_for_groq
                normalized_format = dict(response_format)
                normalize_response_format_for_groq(model, {"response_format": normalized_format})
                groq_kwargs["response_format"] = normalized_format
            response = await self._client.chat.completions.create(**groq_kwargs)  # type: ignore[attr-defined]
            return self._model_dump(response)

        if self.provider == "google":
            return await self._google_chat_completion(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                tool_choice=tool_choice,
                response_format=response_format,
            )

        raise RuntimeError(f"Unsupported meta-model provider: {self.provider}")

    async def _google_chat_completion(
        self,
        *,
        messages: Sequence[Dict[str, Any]],
        model: str,
        temperature: float,
        max_tokens: int,
        tools: Sequence[Dict[str, Any]] | None = None,
        tool_choice: str | Dict[str, Any] | None = None,
        response_format: Dict[str, Any] | None = None,
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

        # Convert tools format (OpenAI -> Gemini)
        gemini_tools = None
        tool_config = None
        if tools:
            function_declarations = []
            for tool in tools:
                if isinstance(tool, dict) and tool.get("type") == "function":
                    func = tool.get("function", {})
                    func_decl = types.FunctionDeclaration(
                        name=func.get("name", ""),
                        description=func.get("description", ""),
                        parameters=func.get("parameters", {}),
                    )
                    function_declarations.append(func_decl)

            if function_declarations:
                gemini_tools = [types.Tool(function_declarations=function_declarations)]

                # Handle tool_choice - Gemini uses tool_config
                if tool_choice and isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
                    func_name = tool_choice.get("function", {}).get("name")
                    if func_name:
                        mode_enum = types.FunctionCallingConfigMode.ANY
                        tool_config = types.ToolConfig(
                            function_calling_config=types.FunctionCallingConfig(
                                mode=mode_enum,
                                allowed_function_names=[func_name],
                            )
                        )

        # Build generation config
        cfg_kwargs: Dict[str, Any] = {"temperature": temperature}
        if system_instruction:
            cfg_kwargs["system_instruction"] = system_instruction
        if max_tokens:
            cfg_kwargs["max_output_tokens"] = max_tokens

        # Handle response_format -> Gemini responseSchema conversion
        # OpenAI: {"type": "json_schema", "json_schema": {"name": "...", "schema": {...}}}
        # Gemini: response_mime_type="application/json", response_schema={...}
        if response_format and isinstance(response_format, dict):
            format_type = response_format.get("type")
            if format_type == "json_schema":
                json_schema_obj = response_format.get("json_schema", {})
                schema = json_schema_obj.get("schema", {})
                if schema:
                    cfg_kwargs["response_mime_type"] = "application/json"
                    cfg_kwargs["response_schema"] = schema
            elif format_type == "json_object":
                # json_object mode - just request JSON without schema
                cfg_kwargs["response_mime_type"] = "application/json"

        # Add tool config if present
        if tool_config:
            cfg_kwargs["tool_config"] = tool_config
        if gemini_tools:
            cfg_kwargs["tools"] = gemini_tools

        generation_config = types.GenerateContentConfig(**cfg_kwargs)

        # Make the API call
        resp = await client.aio.models.generate_content(  # type: ignore[attr-defined]
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


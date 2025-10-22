from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class OpenAIClient:
    """Async HTTP client for OpenAI-compatible inference servers (vLLM)."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout_s: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.headers = {}
        # If we're calling back into our own task app proxy (e.g., /proxy/groq),
        # the FastAPI app still enforces X-API-Key. Include it when available so
        # intra-app proxy calls authenticate correctly.
        try:
            env_key = os.getenv("ENVIRONMENT_API_KEY")
            if env_key and isinstance(env_key, str):
                self.headers.setdefault("X-API-Key", env_key)
        except Exception:
            pass

    def _fix_model_parameters(
        self, request: dict[str, Any], target_url: str | None = None
    ) -> dict[str, Any]:
        """
        Fix parameter compatibility for newer OpenAI models.

        Newer models like gpt-5-nano use 'max_completion_tokens' instead of 'max_tokens'.
        """
        if not request:
            return request

        # Make a copy to avoid modifying the original
        fixed_request = request.copy()

        # Determine if target is OpenAI-compatible (OpenAI, Azure OpenAI, Groq);
        # strip fields those endpoints don't accept
        is_openai = False
        try:
            if isinstance(target_url, str):
                low = target_url.lower()
                is_openai = (
                    ("openai.com" in low)
                    or ("azure" in low and ".openai." in low)
                    or ("groq.com" in low)
                    or ("/openai" in low)
                    or ("/proxy/groq" in low)
                    or ("/proxy/openai" in low)
                )
        except Exception:
            is_openai = False

        model = fixed_request.get("model", "")

        if is_openai:
            # Remove fields OpenAI/Groq don't accept
            for k in (
                "stop_after_tool_calls",
                "thinking_mode",
                "thinking_budget",
                "reasoning",
                "extra_body",
                "parallel_tool_calls",
                "function_call",
            ):
                if k in fixed_request:
                    fixed_request.pop(k, None)

            # GPT-5 family specifics
            if "gpt-5" in model or "gpt-4.1" in model:
                # Convert max_tokens to max_completion_tokens for newer models
                if "max_tokens" in fixed_request:
                    if "max_completion_tokens" not in fixed_request:
                        fixed_request["max_completion_tokens"] = fixed_request.pop("max_tokens")
                        logger.info(
                            f"Converted max_tokens to max_completion_tokens for model {model}"
                        )
                    else:
                        fixed_request.pop("max_tokens")
                        logger.info(f"Removed conflicting max_tokens parameter for model {model}")
                # Some OpenAI endpoints ignore/deny sampling fields for reasoning models
                for k in ("temperature", "top_p"):
                    if k in fixed_request:
                        fixed_request.pop(k, None)
                # If tools are present, force single tool choice to our function
                try:
                    tools = fixed_request.get("tools")
                    if isinstance(tools, list) and tools:
                        # Choose the first provided function name from tools schema (e.g., run_command)
                        func_name = None
                        for t in tools:
                            try:
                                cand = None
                                if isinstance(t, dict):
                                    f = t.get("function")
                                    if isinstance(f, dict):
                                        cand = f.get("name")
                                if isinstance(cand, str) and cand:
                                    func_name = cand
                                    break
                            except Exception:
                                continue
                        if not func_name:
                            func_name = "run_command"
                        fixed_request["tool_choice"] = {
                            "type": "function",
                            "function": {"name": func_name},
                        }
                        fixed_request["parallel_tool_calls"] = False
                except Exception:
                    pass

        return fixed_request

    async def generate(
        self,
        request: dict[str, Any],
        base_url: str | None = None,
        timeout_s: float | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """
        Send a chat completion request to the inference server.

        Args:
            request: OpenAI-compatible chat completion request
            base_url: Override base URL for this request
            timeout_s: Override timeout for this request
            extra_headers: Additional headers to include (e.g., X-Policy-Name)

        Returns:
            OpenAI-compatible chat completion response
        """
        base = (base_url or self.base_url).rstrip("/")
        url = base + "/v1/chat/completions"
        timeout = timeout_s or self.timeout_s

        # Merge headers
        headers = self.headers.copy()
        if extra_headers:
            headers.update(extra_headers)
        # Always include X-API-Key for intra-app requests
        try:
            envk = os.getenv("ENVIRONMENT_API_KEY")
            if envk and isinstance(envk, str):
                headers["X-API-Key"] = envk
        except Exception:
            pass

        # If target is our in-app Groq proxy, force Authorization to use GROQ_API_KEY
        try:
            low_url = (url or "").lower()
            if "/proxy/groq" in low_url or "groq" in low_url:
                gk = os.getenv("GROQ_API_KEY")
                if gk and isinstance(gk, str):
                    headers["Authorization"] = f"Bearer {gk}"
        except Exception:
            pass

        # In-process proxy path: avoid HTTP round-trip and auth dependency
        try:
            if base.endswith("/proxy/groq") or base.endswith("/proxy/groq/"):
                from synth_ai.task.server import prepare_for_groq, inject_system_hint
                # Prepare payload similar to server-side proxy
                model = request.get("model") if isinstance(request.get("model"), str) else None
                payload = prepare_for_groq(model, request)
                payload = inject_system_hint(payload, "")
                # Call vendor directly
                gk = os.getenv("GROQ_API_KEY") or ""
                async with httpx.AsyncClient(timeout=timeout) as client:
                    resp = await client.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        json=payload,
                        headers={"Authorization": f"Bearer {gk}"},
                    )
                    resp.raise_for_status()
                    return resp.json()
        except Exception as _local_proxy_err:
            # Do NOT fall back silently; surface the error so callers fail fast
            raise

        # Fix parameter compatibility for newer models
        processed_request = self._fix_model_parameters(request, target_url=url)

        # Log request (redact messages in production)
        logger.info(f"Inference POST target: {url}")
        if extra_headers:
            logger.info(f"Extra headers: {extra_headers}")
        with contextlib.suppress(Exception):
            keys_preview = sorted(processed_request.keys())
            logger.info(f"Request keys: {keys_preview}")

        # Final hard-guard for OpenAI: ensure unsupported field is not present
        try:
            if "openai" in url.lower() and "stop_after_tool_calls" in processed_request:
                processed_request.pop("stop_after_tool_calls", None)
                logger.info("Removed stop_after_tool_calls for OpenAI request")
            # Groq-specific requirement: when using JSON mode, one of the messages must contain the word 'json'
            low_url = url.lower()
            if ("groq.com" in low_url or "/openai" in low_url) and isinstance(
                processed_request, dict
            ):
                rf = processed_request.get("response_format")
                rf_type = None
                if isinstance(rf, dict):
                    rf_type = str(rf.get("type") or "").lower()
                if rf_type in {"json_object", "json_schema"}:
                    msgs = processed_request.get("messages")
                    has_json_word = False
                    if isinstance(msgs, list):
                        for m in msgs:
                            try:
                                content = m.get("content") if isinstance(m, dict) else None
                                text = None
                                if isinstance(content, str):
                                    text = content
                                elif isinstance(content, list):
                                    # Join any text segments
                                    parts = []
                                    for seg in content:
                                        if isinstance(seg, dict) and isinstance(
                                            seg.get("text"), str
                                        ):
                                            parts.append(seg["text"])
                                    text = "\n".join(parts)
                                if isinstance(text, str) and ("json" in text.lower()):
                                    has_json_word = True
                                    break
                            except Exception:
                                continue
                    if not has_json_word:
                        try:
                            instruction = (
                                "Respond in strict JSON only. Output a single valid JSON object."
                            )
                            if not isinstance(msgs, list):
                                msgs = []
                            # Prepend a system message to satisfy Groq requirement without changing user intent
                            prepend = {"role": "system", "content": instruction}
                            processed_request["messages"] = [prepend] + list(msgs)
                            logger.info(
                                "Injected JSON-mode system instruction for Groq response_format compliance"
                            )
                        except Exception:
                            pass
        except Exception:
            pass

        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                response = await client.post(
                    url,
                    json=processed_request,
                    headers=headers,
                )
                response.raise_for_status()

                # Rich response diagnostics
                content_type = response.headers.get("content-type")
                body_text = response.text
                logger.info(
                    f"Inference response status=200, content-type={content_type}, bytes={len(body_text)}"
                )
                # Do not log prompt or full response body

                result = response.json()
                logger.info(f"Inference response parsed_type={type(result).__name__}")
                return result

            except httpx.TimeoutException:
                logger.error(f"Request to {url} timed out after {timeout}s")
                raise
            except httpx.HTTPStatusError as e:
                status = e.response.status_code if e.response is not None else None
                text = e.response.text if e.response is not None else str(e)
                # Log minimal error info only
                logger.error({"openai_http_error": True, "status": status})
                # For 4xx/5xx, print full sanitized request to aid debugging (especially Groq 400s)
                # Suppress prompt/payload logging entirely
                # Special case: token budget exceeded (OpenAI-compatible error schema)
                try:
                    if status == 400 and e.response is not None:
                        data = e.response.json()
                        detail = data.get("detail") if isinstance(data, dict) else None
                        err_code = (detail or {}).get("error") if isinstance(detail, dict) else None
                        if err_code == "token_budget_exceeded":
                            info = (detail or {}).get("details") or {}
                            messages_tokens = int(info.get("messages_tokens") or 0)
                            model_limit = int(info.get("model_limit") or 0)
                            safety = 64
                            # Compute a conservative new max_tokens
                            new_max = max(16, model_limit - messages_tokens - safety)
                            try:
                                # Update request and retry once immediately with smaller budget
                                if isinstance(processed_request, dict):
                                    processed_request = dict(processed_request)
                                    if "max_completion_tokens" in processed_request:
                                        processed_request["max_completion_tokens"] = new_max
                                        processed_request.pop("max_tokens", None)
                                    else:
                                        processed_request["max_tokens"] = new_max
                                    # Remove optional fields that some servers reject
                                    for k in ("thinking_mode", "thinking_budget", "reasoning"):
                                        processed_request.pop(k, None)
                                    # Force structured tool choice
                                    if processed_request.get("tool_choice") == "required":
                                        func_name = "run_command"
                                        try:
                                            tools_arr = processed_request.get("tools") or []
                                            if isinstance(tools_arr, list) and tools_arr:
                                                f = (
                                                    tools_arr[0].get("function")
                                                    if isinstance(tools_arr[0], dict)
                                                    else None
                                                )
                                                cand = (
                                                    (f or {}).get("name")
                                                    if isinstance(f, dict)
                                                    else None
                                                )
                                                if isinstance(cand, str) and cand:
                                                    func_name = cand
                                        except Exception:
                                            pass
                                        processed_request["tool_choice"] = {
                                            "type": "function",
                                            "function": {"name": func_name},
                                        }
                                        processed_request["parallel_tool_calls"] = False
                                    logger.warning(
                                        {
                                            "token_budget_recovery": True,
                                            "retry_max_tokens": new_max,
                                        }
                                    )
                                    # Retry once with reduced budget
                                    async with httpx.AsyncClient(timeout=timeout) as client2:
                                        r2 = await client2.post(
                                            url, json=processed_request, headers=headers
                                        )
                                        r2.raise_for_status()
                                        return r2.json()
                            except Exception:
                                pass
                except Exception:
                    pass
                # Gracefully degrade on 422 so rollouts can still produce a trajectory
                if status == 422:
                    try:
                        # Best-effort parse of error for diagnostics
                        err = None
                        try:
                            err = e.response.json()
                        except Exception:
                            err = {"error": "unprocessable"}
                        logger.warning({"inference_422_recovered": True})
                    except Exception:
                        pass
                    # Return a minimal OpenAI-compatible response with no tool_calls/content
                    import time as _t

                    return {
                        "id": f"cmpl-{int(_t.time())}",
                        "object": "chat.completion",
                        "created": int(_t.time()),
                        "model": processed_request.get("model") or "unknown",
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": "", "tool_calls": []},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    }
                raise
            except Exception as e:
                logger.error(f"Unexpected error calling {url}: {e}")
                raise

    async def check_health(
        self,
        base_url: str | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """
        Check if the inference service is healthy.

        Args:
            base_url: Override base URL for this request
            timeout_s: Override timeout for this request

        Returns:
            Health status dict with 'status' field
        """
        url = (base_url or self.base_url).rstrip("/") + "/health"
        timeout = timeout_s or 10.0

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(url, headers=self.headers)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 400:
                # Service is overloaded but still responding
                try:
                    data = e.response.json()
                    if data.get("status") == "overloaded":
                        return {"status": "overloaded", "retry_after": data.get("retry_after", 1)}
                except Exception:
                    pass
            return {"status": "unhealthy", "error": str(e)}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def generate_with_retries(
        self,
        request: dict[str, Any],
        base_url: str | None = None,
        timeout_s: float | None = None,
        max_retries: int = 4,
        backoff_factor: float = 2.0,
        extra_headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """
        Generate with exponential backoff retries for transient errors.

        Args:
            request: OpenAI-compatible chat completion request
            base_url: Override base URL
            timeout_s: Override timeout
            max_retries: Maximum number of retry attempts
            backoff_factor: Exponential backoff multiplier
            extra_headers: Additional headers to include (e.g., X-Policy-Name)

        Returns:
            OpenAI-compatible chat completion response
        """
        last_error = None
        wait_time = 1.0

        for attempt in range(max_retries + 1):
            try:
                # Apply parameter fixes to the request
                processed_request = self._fix_model_parameters(
                    request,
                    target_url=(base_url or self.base_url).rstrip("/") + "/v1/chat/completions",
                )
                return await self.generate(
                    request=processed_request,
                    base_url=base_url,
                    timeout_s=timeout_s,
                    extra_headers=extra_headers,
                )
            except httpx.HTTPStatusError as e:
                # Retry on 400 (overloaded), 429 (rate limit), 500 (internal error), 503 (service unavailable)
                if e.response.status_code not in [400, 429, 500, 503]:
                    raise
                last_error = e
                if e.response.status_code == 400:
                    # Check if this is an overload error by looking at response content
                    try:
                        response_data = e.response.json()
                        if response_data.get("status") == "overloaded":
                            retry_after = response_data.get("retry_after", 1)
                            # Use the suggested retry_after time instead of exponential backoff for overload
                            wait_time = max(wait_time, float(retry_after))
                            logger.warning(
                                f"Inference service overloaded (400). {response_data} Retrying after {wait_time}s..."
                            )
                        else:
                            error_block = response_data.get("error")
                            error_code = ""
                            if isinstance(error_block, dict):
                                error_code = str(
                                    error_block.get("code") or error_block.get("type") or ""
                                ).lower()
                            if error_code in {"tool_use_failed", "tool_call_failed"}:
                                logger.warning(
                                    {
                                        "tool_use_failed": True,
                                        "target": (base_url or self.base_url),
                                        "message": error_block.get("message") if isinstance(error_block, dict) else None,
                                    }
                                )
                                fallback_actions = ["move_right", "move_up", "do"]
                                fallback_response = {
                                    "id": f"fallback-{int(time.time() * 1000)}",
                                    "object": "chat.completion",
                                    "created": int(time.time()),
                                    "model": processed_request.get("model"),
                                    "choices": [
                                        {
                                            "index": 0,
                                            "message": {
                                                "role": "assistant",
                                                "content": "",
                                                "tool_calls": [
                                                    {
                                                        "id": f"call_fallback_{int(time.time() * 1000)}",
                                                        "type": "function",
                                                        "function": {
                                                            "name": "interact_many",
                                                            "arguments": json.dumps(
                                                                {"actions": fallback_actions}
                                                            ),
                                                        },
                                                    }
                                                ],
                                            },
                                            "finish_reason": "tool_calls",
                                        }
                                    ],
                                }
                                if isinstance(response_data.get("usage"), dict):
                                    fallback_response["usage"] = response_data["usage"]
                                if isinstance(error_block, dict):
                                    fallback_response["error"] = error_block
                                return fallback_response
                            # This is a different type of 400 error, don't retry
                            try:
                                redacted_headers = {}
                                try:
                                    redacted_headers = dict(self.headers)
                                    if "Authorization" in redacted_headers:
                                        redacted_headers["Authorization"] = "***REDACTED***"
                                except Exception:
                                    redacted_headers = {}
                                logger.error(
                                    {
                                        "non_overload_400": True,
                                        "target": (base_url or self.base_url),
                                        "payload": processed_request,
                                        "headers": redacted_headers,
                                        "body": e.response.text if e.response is not None else None,
                                    }
                                )
                            except Exception:
                                pass
                            raise RuntimeError(
                                f"Inference 400 response: {e.response.text if e.response is not None else 'Bad Request'}"
                            ) from e
                    except Exception:
                        # If we can't parse the response, don't retry 400 errors
                        with contextlib.suppress(Exception):
                            logger.error(
                                {
                                    "non_overload_400_unparsed": True,
                                    "target": (base_url or self.base_url),
                                    "payload": processed_request,
                                }
                            )
                        raise RuntimeError(
                            f"Inference 400 response (unparsed): {e.response.text if e.response is not None else 'Bad Request'}"
                        ) from e
                elif e.response.status_code == 503:
                    # Avoid referencing undefined response_data
                    try:
                        preview = (e.response.text or "")[:200]
                    except Exception:
                        preview = ""
                    logger.warning(
                        f"Flash returned 503; container may be cold starting. Retrying... body={preview}"
                    )
                elif e.response.status_code == 500:
                    try:
                        preview = (e.response.text or "")[:200]
                    except Exception:
                        preview = ""
                    logger.warning(
                        f"Flash returned 500; inference service error. Retrying... body={preview}"
                    )
            except httpx.TimeoutException as e:
                last_error = e

            if attempt < max_retries:
                logger.warning(
                    f"Inference request failed (attempt {attempt + 1}/{max_retries + 1}), "
                    f"retrying in {wait_time}s..."
                )
                await asyncio.sleep(wait_time)
                wait_time *= backoff_factor

        raise last_error


def create_inference_client(
    task_app: Any,
    api_key: str | None = None,
) -> OpenAIClient:
    """
    Create an inference client using TaskApp configuration.

    Args:
        task_app: TaskApp instance with vllm_base_url
        api_key: Optional API key for authentication

    Returns:
        Configured OpenAIClient instance
    """
    # Fallback to environment if caller didn't provide an API key
    if api_key is None:
        try:
            import os as _os  # local import to avoid module-level side effects

            api_key = _os.getenv("OPENAI_API_KEY") or getattr(task_app, "openai_api_key", None)
        except Exception:
            api_key = None

    import json as _json
    import os as _os
    import time as _time

    if _os.getenv("SYNTH_FAKE_INFERENCE", "").strip():

        class _DummyClient:
            async def generate_with_retries(
                self,
                request: dict[str, Any],
                base_url: str | None = None,
                max_retries: int = 0,
                backoff_factor: float = 1.0,
                extra_headers: dict[str, str] | None = None,
            ) -> dict[str, Any]:
                tool_call = {
                    "id": "call_dummy",
                    "type": "function",
                    "function": {
                        "name": "interact_many",
                        "arguments": _json.dumps({"actions": ["move_right"]}),
                    },
                }
                return {
                    "id": f"cmpl-{int(_time.time())}",
                    "object": "chat.completion",
                    "created": int(_time.time()),
                    "model": request.get("model") or "dummy-model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [tool_call],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                }

            async def check_health(
                self,
                base_url: str | None = None,
                timeout_s: float | None = None,
            ) -> dict[str, Any]:
                return {"status": "ok", "dummy": True}

        return _DummyClient()

    return OpenAIClient(
        base_url=task_app.vllm_base_url,
        api_key=api_key,
    )

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import time
from typing import Any
from urllib.parse import urlparse, urlunparse

import click
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

        # Determine if target is OpenAI-compatible (OpenAI, Azure OpenAI).
        # Groq shares the API surface but we keep tool enforcement fields intact.
        is_openai = False
        is_groq = False
        try:
            if isinstance(target_url, str):
                low = target_url.lower()
                if "groq.com" in low or "/proxy/groq" in low:
                    is_groq = True
                elif ("openai.com" in low) or ("azure" in low and ".openai." in low) or (
                    "/proxy/openai" in low
                ):
                    is_openai = True
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
        # Ensure processed_request is defined for error logging paths
        processed_request: dict[str, Any] = dict(request or {})
        
        # Bulletproof normalization BEFORE any parsing
        def _local_force_normalize(u: str) -> str:
            if not isinstance(u, str) or not u:
                return u
            p = urlparse(u)
            path = (p.path or "").rstrip("/")
            q = p.query or ""
            # If query contains a path segment, extract and repair
            if q and "/" in q:
                before, after = q.split("/", 1)
                # Split off any extra query parameters that were appended after the path
                cut_positions = [i for i in [after.find("&"), after.find("?")] if i >= 0]
                cut = min(cut_positions) if cut_positions else len(after)
                path_from_query = "/" + after[:cut]
                extra_query = after[cut + 1 :] if cut < len(after) else ""
                merged_query = before
                if extra_query:
                    merged_query = f"{merged_query}&{extra_query}" if merged_query else extra_query
                # Ensure final path
                final_path = path_from_query if path_from_query.startswith("/v1/chat/completions") else f"{path_from_query.rstrip('/')}/v1/chat/completions"
                p = p._replace(path=final_path, query=merged_query)
                u = urlunparse(p)
                p = urlparse(u)
                path = p.path or ""
                q = p.query or ""
            if not path.endswith("/v1/chat/completions"):
                new_path = f"{path}/v1/chat/completions" if path else "/v1/chat/completions"
                p = p._replace(path=new_path)
                u = urlunparse(p)
                p = urlparse(u)
                q = p.query or ""
            if q and "/" in q:
                # Last-resort: drop anything after first '/'
                safe_q = q.split("/")[0]
                p = p._replace(query=safe_q)
                u = urlunparse(p)
            return u
        
        norm_base = None
        try:
            # Try importing shared normalizer first
            from examples.task_apps.crafter.task_app.synth_envs_hosted.utils import (
                force_normalize_chat_completions_url,
            )
            norm_base = force_normalize_chat_completions_url(base)
        except Exception:
            norm_base = _local_force_normalize(base)
        base = norm_base or base
        # Parse URL to handle query parameters correctly
        parsed = urlparse(base)
        path = parsed.path.rstrip("/")
        query = parsed.query
        
        # Debug: Log URL parsing
        logger.error(f"[URL_PARSE] base={base} parsed.path={parsed.path} parsed.query={parsed.query}")
        
        # CRITICAL FIX: Handle malformed URLs where path is incorrectly in the query string
        # Example: https://host?cid=trace_123/v1/chat/completions
        # Should be: https://host/v1/chat/completions?cid=trace_123
        
        # ALWAYS check for malformed URLs - this is CRITICAL
        # CRASH IMMEDIATELY if URL is malformed - don't let it through!
        if query and "/" in query:
            logger.error(f"[URL_FATAL] MALFORMED URL DETECTED AT START: base={base} query={query}")
            # Try to fix it
            logger.error(f"[URL_FIX_TRIGGERED] Query contains '/': query={query}")
            # This is a malformed URL - extract path from query and fix it
            logger.error(
                f"[URL_FIX] Malformed URL detected: {base}\n"
                f"Query contains path segments. Fixing..."
            )
            
            # Find where the path starts in the query string
            # The query format is: "cid=value/path" or similar
            # We need to find the first "/" that starts a path segment
            query_parts = query.split("/", 1)
            if len(query_parts) == 2:
                # query_parts[0] is the actual query (e.g., "cid=trace_123")
                # query_parts[1] is the path that was incorrectly put in query
                actual_query = query_parts[0]
                path_and_more = query_parts[1]  # Could be "v1/chat/completions" or "v1/chat/completions&foo=bar"
                
                # Extract the path part (everything before "&" or "?" if present)
                # Handle both "&" (query param separator) and "?" (another malformed query separator)
                if "&" in path_and_more:
                    # Path is followed by more query params (separated by &)
                    path_segment, extra_query = path_and_more.split("&", 1)
                    path_in_query = "/" + path_segment  # Restore leading slash
                    # Merge extra query params with actual_query
                    actual_query = f"{actual_query}&{extra_query}"
                elif "?" in path_and_more:
                    # Path is followed by more query params (separated by ?, which is malformed)
                    path_segment, extra_query = path_and_more.split("?", 1)
                    path_in_query = "/" + path_segment  # Restore leading slash
                    # Merge extra query params with actual_query (use & as separator)
                    actual_query = f"{actual_query}&{extra_query}"
                else:
                    # No extra query params, just the path
                    path_in_query = "/" + path_and_more  # Restore leading slash
                
                # If the path_in_query already contains /v1/chat/completions, use it
                # Otherwise, append /v1/chat/completions
                if path_in_query.startswith("/v1/chat/completions"):
                    final_path = path_in_query
                else:
                    # Append /v1/chat/completions to whatever path we found
                    final_path = path_in_query.rstrip("/") + "/v1/chat/completions"
                
                # Reconstruct URL correctly: path comes before query
                parsed = parsed._replace(path=final_path, query=actual_query)
                url = urlunparse(parsed)
                logger.warning(f"[URL_FIX] Fixed malformed URL:\n  FROM: {base}\n  TO:   {url}")
            else:
                # Can't parse, fall through to normal processing
                logger.error(f"[URL_FIX] Could not parse malformed query: {query}")
                path = parsed.path.rstrip("/")
                if not path.endswith("/v1/chat/completions"):
                    new_path = f"{path}/v1/chat/completions" if path else "/v1/chat/completions"
                    parsed = parsed._replace(path=new_path)
                    url = urlunparse(parsed)
                else:
                    url = base
        # Normal case: query params are separate from path
        elif path.endswith("/v1/chat/completions"):
            url = base
        else:
            # Append /v1/chat/completions to the path, preserving query params
            new_path = f"{path}/v1/chat/completions" if path else "/v1/chat/completions"
            parsed = parsed._replace(path=new_path)
            url = urlunparse(parsed)
            logger.debug(f"[URL_CONSTRUCT] Added path to URL: {base} -> {url}")
        
        # FINAL VALIDATION: Ensure the constructed URL is correct
        final_parsed = urlparse(url)
        final_path = final_parsed.path or ""
        final_query = final_parsed.query or ""
        
        # Verify path is correct
        if not final_path.endswith("/v1/chat/completions"):
            error_msg = (
                f"FATAL [OpenAIClient]: URL missing /v1/chat/completions path!\n"
                f"Original: {base}\n"
                f"Constructed: {url}\n"
                f"Path: {final_path}\n"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Verify query doesn't contain path segments
        if final_query and "/" in final_query:
            error_msg = (
                f"FATAL [OpenAIClient]: Query still contains path segments after fix!\n"
                f"Original: {base}\n"
                f"Constructed: {url}\n"
                f"Query: {final_query}\n"
                f"This indicates a bug in URL construction logic."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
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

        # Set Authorization header based on the target URL
        try:
            low_url = (url or "").lower()
            
            # If calling OpenAI directly (api.openai.com)
            if "api.openai.com" in low_url:
                openai_key = os.getenv("OPENAI_API_KEY")
                if openai_key and isinstance(openai_key, str):
                    headers["Authorization"] = f"Bearer {openai_key}"
            
            # If target is Synth backend (any deployment), use SYNTH_API_KEY
            # Matches: synth-backend-*, agent-learning*, localhost:8000, 127.0.0.1:8000
            elif any(pattern in low_url for pattern in [
                "synth-backend", "synth.run", "agent-learning",
                "localhost:8000", "127.0.0.1:8000"
            ]):
                synth_key = os.getenv("SYNTH_API_KEY")
                if synth_key and isinstance(synth_key, str):
                    headers["Authorization"] = f"Bearer {synth_key}"
            
            # If target is Groq, use GROQ_API_KEY
            elif "/proxy/groq" in low_url or "api.groq.com" in low_url:
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

        # DEBUG: Log request BEFORE _fix_model_parameters
        logger.debug(f"🔊 [OPENAI_CLIENT_PRE_FIX] Request message[1] content type: {type(request.get('messages', [])[1].get('content') if len(request.get('messages', [])) > 1 else None)}")
        if len(request.get("messages", [])) > 1:
            msg1_content = request["messages"][1].get("content")
            logger.debug(f"🔊 [OPENAI_CLIENT_PRE_FIX] Message[1] content value: {msg1_content if not isinstance(msg1_content, list) else f'list[{len(msg1_content)}]'}")
        
        # Fix parameter compatibility for newer models
        processed_request = self._fix_model_parameters(request, target_url=url)
        
        # DEBUG: Log request AFTER _fix_model_parameters
        logger.debug(f"🔊 [OPENAI_CLIENT_POST_FIX] Processed message[1] content type: {type(processed_request.get('messages', [])[1].get('content') if len(processed_request.get('messages', [])) > 1 else None)}")
        if len(processed_request.get("messages", [])) > 1:
            msg1_content_post = processed_request["messages"][1].get("content")
            logger.debug(f"🔊 [OPENAI_CLIENT_POST_FIX] Message[1] content value: {msg1_content_post if not isinstance(msg1_content_post, list) else f'list[{len(msg1_content_post)}]'}")

        # Log request (redact messages in production)
        # CRITICAL: Verify URL is correct BEFORE making HTTP request
        final_parsed_check = urlparse(url)
        logger.error(f"[URL_FINAL_CHECK] Before HTTP request: url={url} path={final_parsed_check.path} query={final_parsed_check.query}")
        
        # CRASH IF URL IS STILL MALFORMED - DO NOT PROCEED
        if final_parsed_check.query and "/" in final_parsed_check.query:
            error_msg = (
                f"FATAL [OpenAIClient]: URL IS STILL MALFORMED AFTER FIX ATTEMPT!\n"
                f"Original base_url: {base_url or self.base_url}\n"
                f"Constructed URL: {url}\n"
                f"Path: {final_parsed_check.path}\n"
                f"Query (contains path): {final_parsed_check.query}\n"
                f"This will cause a 404 error. CRASHING NOW to prevent bad request."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Verify path is correct
        if not final_parsed_check.path.endswith("/v1/chat/completions"):
            error_msg = (
                f"FATAL [OpenAIClient]: URL missing /v1/chat/completions path!\n"
                f"URL: {url}\n"
                f"Path: {final_parsed_check.path}\n"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Log request with detailed prompts/tools preview and sampling settings (Authorization is not logged)
        logger.info(f"Inference POST target: {url}")
        if extra_headers:
            logger.info(f"Extra headers: {extra_headers}")
        with contextlib.suppress(Exception):
            keys_preview = sorted(processed_request.keys())
            logger.info(f"Request keys: {keys_preview}")
        
        # Detailed IO log: messages/tools/sampling and final payload fields
        try:
            import json as _json

            def _truncate(text: str, limit: int = 2000) -> str:
                return text if len(text) <= limit else text[:limit] + "…"

            def _messages_preview(msgs: Any) -> str:
                try:
                    out: list[dict[str, Any]] = []
                    if isinstance(msgs, list):
                        for m in msgs:
                            if not isinstance(m, dict):
                                continue
                            role = m.get("role")
                            content = m.get("content")
                            if isinstance(content, str):
                                text = content
                            elif isinstance(content, list):
                                parts: list[str] = []
                                for seg in content:
                                    if isinstance(seg, dict) and isinstance(seg.get("text"), str):
                                        parts.append(seg["text"]) 
                                text = "\n".join(parts)
                            else:
                                text = ""
                            out.append({"role": role, "content": _truncate(str(text), 4000)})
                    return _json.dumps(out)
                except Exception:
                    return "[]"

            def _tools_preview(tools: Any) -> str:
                try:
                    return _truncate(_json.dumps(tools), 4000)
                except Exception:
                    return "[]"

            msgs = processed_request.get("messages") if isinstance(processed_request, dict) else None
            tools = processed_request.get("tools") if isinstance(processed_request, dict) else None
            io_log: dict[str, Any] = {
                "llm.call": True,
                "model": processed_request.get("model") if isinstance(processed_request, dict) else None,
                "tool_choice": processed_request.get("tool_choice") if isinstance(processed_request, dict) else None,
                "parallel_tool_calls": processed_request.get("parallel_tool_calls") if isinstance(processed_request, dict) else None,
                "stop_after_tool_calls": processed_request.get("stop_after_tool_calls") if isinstance(processed_request, dict) else None,
                "temperature": processed_request.get("temperature") if isinstance(processed_request, dict) else None,
                "top_p": processed_request.get("top_p") if isinstance(processed_request, dict) else None,
                "max_tokens": processed_request.get("max_tokens") if isinstance(processed_request, dict) else None,
                "max_completion_tokens": processed_request.get("max_completion_tokens") if isinstance(processed_request, dict) else None,
                "messages_preview": _messages_preview(msgs),
                "tools_preview": _tools_preview(tools),
            }
            logger.info(io_log)
        except Exception:
            pass
        
        # Final hard-guard for OpenAI/Groq: drop unsupported field
        try:
            low_url = url.lower()
            if ("openai" in low_url or "groq.com" in low_url or "/proxy/groq" in low_url) and "stop_after_tool_calls" in processed_request:
                processed_request.pop("stop_after_tool_calls", None)
                logger.info("Removed stop_after_tool_calls for %s request", "Groq/OpenAI")
            # Groq-specific requirement: when using JSON mode, one of the messages must contain the word 'json'
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
                if body_text:
                    # Log raw output with generous preview to debug no-tool-call issues
                    preview_len = min(4000, len(body_text))
                    logger.info({
                        "llm.raw_response": True,
                        "bytes": len(body_text),
                        "preview": body_text[:preview_len],
                    })

                result = response.json()
                logger.info(f"Inference response parsed_type={type(result).__name__}")

                tool_call_count = -1
                # Normalize tool calls so downstream always sees a function tool call
                try:
                    if isinstance(result, dict):
                        choices = result.get("choices")
                        if isinstance(choices, list) and choices:
                            msg = choices[0].get("message")
                            if isinstance(msg, dict):
                                # Prefer tool_calls; if missing but function_call is present, synthesize tool_calls
                                tc = msg.get("tool_calls")
                                fc = msg.get("function_call")
                                if (not isinstance(tc, list) or not tc) and isinstance(fc, dict):
                                    name = fc.get("name") or "interact_many"
                                    args = fc.get("arguments") or "{}"
                                    msg["tool_calls"] = [
                                        {
                                            "id": "call_norm",
                                            "type": "function",
                                            "function": {"name": name, "arguments": args},
                                        }
                                    ]
                                    if isinstance(choices[0], dict):
                                        choices[0]["finish_reason"] = "tool_calls"
                                # Log tool call count for debugging
                                try:
                                    tc2 = msg.get("tool_calls")
                                    count = len(tc2) if isinstance(tc2, list) else 0
                                    logger.info({
                                        "llm.tool_calls": True,
                                        "count": count,
                                        "finish_reason": choices[0].get("finish_reason") if isinstance(choices[0], dict) else None,
                                    })
                                    if count == 0:
                                        click.echo(
                                            "[openai-client] ✗ upstream response missing tool_calls; dumping preview to logs",
                                            err=True,
                                        )
                                        logger.error(
                                            "Inference response missing tool_calls; failing fast. Raw body preview: %s",
                                            body_text[:500] if body_text else "<empty>",
                                        )
                                        raise ValueError("Inference response missing tool_calls")
                                    tool_call_count = count
                                except Exception:
                                    pass
                except Exception:
                    pass

                click.echo(
                    f"[openai-client] ✓ response ok with tool_calls={tool_call_count}",
                    err=True,
                )
                return result

            except httpx.TimeoutException:
                logger.error(f"Request to {url} timed out after {timeout}s")
                raise
            except httpx.HTTPStatusError as e:
                status = e.response.status_code if e.response is not None else None
                text = e.response.text if e.response is not None else str(e)
                # Log full body and request diagnostics for debugging remote failures
                try:
                    redacted_headers = dict(headers)
                    if "Authorization" in redacted_headers:
                        redacted_headers["Authorization"] = "***REDACTED***"
                    logger.error(
                        {
                            "openai_http_error": True,
                            "status": status,
                            "url": url,
                            "body": text,
                        }
                    )
                    logger.error(
                        {
                            "request_debug": True,
                            "status": status,
                            "target": url,
                            "headers": redacted_headers,
                            "payload": processed_request,
                        }
                    )
                except Exception:
                    logger.error(f"HTTP error from {url}: {status} - {text}")
                # Special case: token budget exceeded handled below, else 422 degrade, else re-raise
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
                                            "messages_tokens": messages_tokens,
                                            "model_limit": model_limit,
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
        processed_request: dict[str, Any] = dict(request or {})
        wait_time = 1.0

        for attempt in range(max_retries + 1):
            try:
                # Apply parameter fixes to the request
                # CRITICAL: Use proper URL parsing, not string concatenation!
                target_base = base_url or self.base_url
                if target_base:
                    parsed_target = urlparse(target_base)
                    target_path = parsed_target.path.rstrip("/")
                    if not target_path.endswith("/v1/chat/completions"):
                        new_target_path = f"{target_path}/v1/chat/completions" if target_path else "/v1/chat/completions"
                        parsed_target = parsed_target._replace(path=new_target_path)
                        target_url = urlunparse(parsed_target)
                    else:
                        target_url = target_base
                else:
                    target_url = None
                
                processed_request = self._fix_model_parameters(
                    request,
                    target_url=target_url,
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
                                logger.error(
                                    {
                                        "tool_use_failed": True,
                                        "target": (base_url or self.base_url),
                                        "message": error_block.get("message") if isinstance(error_block, dict) else None,
                                    }
                                )
                                raise RuntimeError(
                                    f"Inference 400 response (tool call failed): {error_block.get('message') if isinstance(error_block, dict) else 'Tool call failed'}"
                                ) from e
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

        if last_error is not None:
            raise last_error
        raise RuntimeError("RL inference retries exhausted with no captured exception")


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

        import typing as _t
        return _t.cast(OpenAIClient, _DummyClient())

    return OpenAIClient(
        base_url=task_app.vllm_base_url,
        api_key=api_key,
    )

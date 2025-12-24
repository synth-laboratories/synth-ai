"""Shared helper utilities for LocalAPI task apps."""

from __future__ import annotations

import contextlib
import inspect
import os
import socket
from collections.abc import Callable, Sequence
from typing import Any
from urllib.parse import urlparse, urlunparse

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse


def normalize_chat_completion_url(url: str) -> str:
    """Normalize inference URL to include /chat/completions path."""
    u = (url or "").rstrip("/")
    if not u:
        return "/chat/completions"

    parsed = urlparse(u)
    path = parsed.path.rstrip("/")
    query = parsed.query
    fragment = parsed.fragment

    if path.endswith("/v1/chat/completions") or path.endswith("/chat/completions"):
        return u

    if "/v1/" in path and not path.endswith("/v1"):
        new_path = f"{path}/chat/completions"
        return urlunparse((parsed.scheme, parsed.netloc, new_path, parsed.params, query, fragment))

    if path.endswith("/v1"):
        new_path = f"{path}/chat/completions"
    elif path.endswith("/completions"):
        new_path = path.rsplit("/", 1)[0] + "/chat/completions"
    else:
        new_path = f"{path}/v1/chat/completions" if path else "/v1/chat/completions"

    return urlunparse((parsed.scheme, parsed.netloc, new_path, parsed.params, query, fragment))


def get_default_max_completion_tokens(model_name: str) -> int:
    """Get default max_completion_tokens based on model name."""
    model_lower = model_name.lower()
    if "gpt-5" in model_lower or "gpt5" in model_lower:
        return 2048
    if "gpt-4" in model_lower or "gpt4" in model_lower:
        return 4096
    if "o1" in model_lower or "o3" in model_lower:
        return 16384
    if "claude" in model_lower:
        return 4096
    return 512


def get_current_module_source() -> str | None:
    """Extract source code for the caller's module using inspect."""
    frame = inspect.currentframe()
    try:
        if frame is None:
            return None
        caller_frame = frame.f_back
        if caller_frame is None:
            return None
        module = inspect.getmodule(caller_frame)
        if module is None:
            return None
        try:
            return inspect.getsource(module)
        except (OSError, TypeError, IOError):
            return None
    finally:
        del frame


def preload_dataset_splits(dataset: Any, splits: Sequence[str], app_name: str) -> None:
    """Preload dataset splits with standardized logging."""
    print(f"[{app_name}] Preloading dataset splits...", flush=True)
    try:
        dataset.ensure_ready(splits)
        sizes = []
        with contextlib.suppress(Exception):
            sizes = [dataset.size(split) for split in splits]
        if sizes:
            print(f"[{app_name}] Dataset preloaded successfully: {sizes} examples", flush=True)
        else:
            print(f"[{app_name}] Dataset preloaded successfully", flush=True)
    except Exception as exc:
        print(f"[{app_name}] WARNING: Dataset preload failed: {exc}", flush=True)
        import traceback

        traceback.print_exc()


def create_http_client_hooks(
    timeout: float = 30.0,
    *,
    log_prefix: str | None = None,
    aiohttp_connector_kwargs: dict[str, Any] | None = None,
    httpx_limits: Any | None = None,
) -> tuple[Callable[[Any], Any], Callable[[Any], Any]]:
    """Return (startup_hook, shutdown_hook) for HTTP client lifecycle."""
    connector_kwargs = {
        "limit": 10,
        "limit_per_host": 5,
        "ttl_dns_cache": 300,
        "use_dns_cache": True,
    }
    if aiohttp_connector_kwargs:
        connector_kwargs.update(aiohttp_connector_kwargs)

    def _log(message: str) -> None:
        if log_prefix:
            print(f"[{log_prefix}] {message}", flush=True)

    async def startup_http_client(app: Any) -> None:
        try:
            import aiohttp

            timeout_cfg = aiohttp.ClientTimeout(total=timeout)
            connector = aiohttp.TCPConnector(**connector_kwargs)
            app.state.http_client = aiohttp.ClientSession(timeout=timeout_cfg, connector=connector)
            _log("Created app-level aiohttp client session singleton")
        except ImportError:
            try:
                import httpx

                limits = httpx_limits or httpx.Limits(max_keepalive_connections=5, max_connections=10)
                app.state.http_client = httpx.AsyncClient(timeout=timeout, limits=limits)
                _log("Created app-level httpx client singleton (fallback)")
            except Exception as exc:
                _log(f"WARNING: Failed to create http client: {exc}")
                app.state.http_client = None
        except Exception as exc:
            _log(f"WARNING: Failed to create aiohttp client: {exc}")
            app.state.http_client = None

    async def shutdown_http_client(app: Any) -> None:
        http_client = getattr(app.state, "http_client", None)
        if http_client is not None:
            try:
                if hasattr(http_client, "close"):
                    await http_client.close()
                elif hasattr(http_client, "aclose"):
                    await http_client.aclose()
                _log("Closed app-level http client")
            except Exception as exc:
                _log(f"WARNING: Error closing http client: {exc}")

    return startup_http_client, shutdown_http_client


def extract_api_key(
    request: Request,
    policy_config: dict[str, Any],
    default_env_keys: dict[str, str] | None = None,
) -> str | None:
    """Extract API key from request headers or environment based on inference URL."""
    default_env_keys = default_env_keys or {
        "api.groq.com": "GROQ_API_KEY",
        "api.openai.com": "OPENAI_API_KEY",
    }

    inference_url_raw = policy_config.get("inference_url")
    api_base_raw = policy_config.get("api_base")
    base_url_raw = policy_config.get("base_url")
    route_base = (
        (str(inference_url_raw).strip() if inference_url_raw else "")
        or (str(api_base_raw).strip() if api_base_raw else "")
        or (str(base_url_raw).strip() if base_url_raw else "")
    )
    lowered = route_base.lower()
    for host, env_var in default_env_keys.items():
        if host in lowered:
            return os.getenv(env_var)

    api_key = request.headers.get("X-API-Key") or request.headers.get("x-api-key")
    if api_key:
        return api_key
    auth_header = request.headers.get("Authorization") or request.headers.get("authorization")
    if auth_header:
        return auth_header.replace("Bearer ", "").strip()
    return None


def parse_tool_calls_from_response(
    response_json: dict[str, Any],
    expected_tool_name: str | None = None,
) -> list[dict[str, Any]]:
    """Parse tool calls from chat completion response."""
    if not isinstance(response_json, dict):
        return []
    choices = response_json.get("choices") or []
    if not choices:
        return []
    message = (choices[0] or {}).get("message", {}) if choices else {}
    tool_calls_raw = message.get("tool_calls", []) or []
    tool_calls: list[dict[str, Any]] = []
    for call in tool_calls_raw:
        function_block = (call or {}).get("function", {}) or {}
        name = function_block.get("name", "")
        if expected_tool_name and name and name != expected_tool_name:
            raise ValueError(f"Unexpected tool name: {name}")
        tool_calls.append(
            {
                "id": (call or {}).get("id", ""),
                "type": (call or {}).get("type", "function"),
                "function": {
                    "name": name,
                    "arguments": function_block.get("arguments", "{}"),
                },
            }
        )
    return tool_calls


async def call_chat_completion_api(
    policy_config: dict[str, Any],
    messages: list[dict[str, str]],
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | None = None,
    api_key: str | None = None,
    http_client: Any | None = None,
    enable_dns_preresolution: bool = True,
    validate_response: bool = True,
    expected_tool_name: str | None = None,
    *,
    default_temperature: float = 0.7,
    log_prefix: str | None = None,
) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    """Unified chat completion API caller with common LocalAPI logic."""
    missing_fields: list[str] = []
    model_val = policy_config.get("model")
    if not isinstance(model_val, str) or not model_val.strip():
        missing_fields.append("model")

    inference_url_raw = policy_config.get("inference_url")
    api_base_raw = policy_config.get("api_base")
    base_url_raw = policy_config.get("base_url")

    if inference_url_raw:
        route_base = str(inference_url_raw).strip()
        if (api_base_raw or base_url_raw) and log_prefix:
            print(
                f"{log_prefix} inference_url is set ({route_base}), ignoring api_base/base_url",
                flush=True,
            )
    else:
        route_base = ((api_base_raw or "").strip()) or ((base_url_raw or "").strip())

    if not route_base:
        missing_fields.append("inference_url")
    if missing_fields:
        raise HTTPException(
            status_code=400,
            detail="Missing policy fields in TOML [prompt_learning.policy]: "
            + ", ".join(missing_fields),
        )

    model = policy_config["model"].strip()
    inference_url = normalize_chat_completion_url(str(route_base))
    temperature = policy_config.get("temperature", default_temperature)

    if "max_completion_tokens" in policy_config:
        max_tokens = policy_config.get("max_completion_tokens")
    elif "max_tokens" in policy_config:
        max_tokens = policy_config.get("max_tokens")
    else:
        max_tokens = get_default_max_completion_tokens(model)

    headers: dict[str, str] = {"Content-Type": "application/json"}
    lowered = route_base.lower()
    is_provider_host = ("api.openai.com" in lowered) or ("api.groq.com" in lowered)

    if api_key:
        if is_provider_host:
            headers["Authorization"] = f"Bearer {api_key}"
        else:
            headers["X-API-Key"] = api_key

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_tokens,
    }
    if tools is not None:
        payload["tools"] = tools
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
    if temperature != 0.0:
        payload["temperature"] = temperature

    if log_prefix:
        with contextlib.suppress(Exception):
            print(f"{log_prefix} POLICY ROUTE -> {inference_url}", flush=True)

    if enable_dns_preresolution and not is_provider_host:
        parsed = urlparse(inference_url)
        host = parsed.hostname or ""
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        with contextlib.suppress(Exception):
            addrinfo = socket.getaddrinfo(host, None, socket.AF_INET)
            ips = sorted({ai[4][0] for ai in addrinfo})
            resolved_ip = ips[0] if ips else None
            if log_prefix:
                print(
                    f"{log_prefix} PROXY_DNS resolved {host} -> {resolved_ip} (from {ips})",
                    flush=True,
                )
            if resolved_ip and parsed.scheme == "https":
                netloc = f"{resolved_ip}:{port}" if port else resolved_ip
                inference_url = f"{parsed.scheme}://{netloc}{parsed.path}"
                if parsed.query:
                    inference_url += f"?{parsed.query}"
                headers["_original_host"] = host
                headers["_use_ip"] = "1"
                headers["Host"] = host

    if http_client is None:
        raise HTTPException(status_code=500, detail="HTTP client not initialized (should be created at startup)")

    response_json: dict[str, Any] | None = None
    try:
        is_aiohttp = False
        with contextlib.suppress(Exception):
            import aiohttp

            is_aiohttp = isinstance(http_client, aiohttp.ClientSession)

        if is_aiohttp:
            use_ip = headers.pop("_use_ip", None) is not None
            original_host = headers.pop("_original_host", None)
            request_headers = {k: v for k, v in headers.items() if not k.startswith("_")}

            ssl_setting: Any = None
            if use_ip and original_host:
                import ssl

                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
                ssl_setting = ssl_context

            async with http_client.post(
                inference_url,
                json=payload,
                headers=request_headers,
                ssl=ssl_setting,
                server_hostname=original_host if (use_ip and original_host) else None,
            ) as response:
                status_code = response.status
                if status_code != 200:
                    try:
                        error_json = await response.json()
                        error_message = _extract_error_message(error_json)
                        raise HTTPException(status_code=status_code, detail=f"Interceptor/provider error: {error_message}")
                    except HTTPException:
                        raise
                    except Exception:
                        error_text = (await response.text())[:500]
                        raise HTTPException(
                            status_code=status_code,
                            detail=f"Interceptor/provider returned error: {error_text}",
                        )

                try:
                    response_json = await response.json()
                except Exception:
                    response_text = await response.text()
                    if status_code >= 400:
                        raise HTTPException(status_code=status_code, detail=f"HTTP error: {response_text[:200]}")
                    response_json = {}
        else:
            response = await http_client.post(inference_url, json=payload, headers=headers)
            status_code = response.status_code
            if status_code != 200:
                try:
                    error_json = response.json()
                    error_message = _extract_error_message(error_json)
                    raise HTTPException(status_code=status_code, detail=f"Interceptor/provider error: {error_message}")
                except HTTPException:
                    raise
                except Exception:
                    error_text = response.text[:500] if hasattr(response, "text") else "Unknown error"
                    raise HTTPException(
                        status_code=status_code,
                        detail=f"Interceptor/provider returned error: {error_text}",
                    )

            try:
                response_json = response.json()
            except Exception:
                response_text = response.text
                if status_code >= 400:
                    raise HTTPException(status_code=status_code, detail=f"HTTP error: {response_text[:200]}")
                response_json = {}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Proxy POST failed: {exc}")

    if response_json is None:
        raise HTTPException(status_code=502, detail="No response data received")

    response_text = ""
    tool_calls: list[dict[str, Any]] = []
    if isinstance(response_json, dict):
        choices = response_json.get("choices") or []
        if choices:
            message = (choices[0] or {}).get("message", {}) if choices else {}
            response_text = str(message.get("content", "") or "")
            try:
                tool_calls = parse_tool_calls_from_response(
                    response_json,
                    expected_tool_name=expected_tool_name,
                )
            except ValueError as exc:
                raise HTTPException(status_code=502, detail=str(exc)) from exc

    if validate_response:
        if not isinstance(response_json, dict) or not response_json:
            raise HTTPException(status_code=502, detail="Proxy returned missing/empty JSON")
        choices = response_json.get("choices") or []
        if not isinstance(choices, list) or len(choices) == 0:
            raise HTTPException(status_code=502, detail="Proxy JSON missing choices")
        first_msg = (choices[0] or {}).get("message", {}) if choices else {}
        if not isinstance(first_msg, dict):
            raise HTTPException(status_code=502, detail="Proxy JSON message malformed")
        content_text = str(first_msg.get("content", ""))
        if not tool_calls and not content_text.strip():
            raise HTTPException(status_code=502, detail="Empty model output: no tool_calls and no content")

    return response_text, response_json, tool_calls


def add_health_endpoints(app: FastAPI) -> None:
    """Add standard /health and /health/rollout endpoints."""
    from synth_ai.sdk.task.auth import is_api_key_header_authorized, normalize_environment_api_key

    def _log_env_key_prefix(source: str, env_key: str | None) -> str | None:
        if not env_key:
            return None
        prefix = env_key[: max(1, len(env_key) // 2)]
        print(f"[{source}] expected ENVIRONMENT_API_KEY prefix: {prefix}")
        return prefix

    @app.get("/health")
    async def health(request: Request):
        env_key = normalize_environment_api_key()
        if not env_key:
            return JSONResponse(
                status_code=503,
                content={"status": "unhealthy", "detail": "Missing ENVIRONMENT_API_KEY"},
            )
        if not is_api_key_header_authorized(request):
            prefix = _log_env_key_prefix("health", env_key)
            content = {"status": "healthy", "authorized": False}
            if prefix:
                content["expected_api_key_prefix"] = prefix
            return JSONResponse(status_code=200, content=content)
        return {"status": "healthy", "authorized": True}

    @app.get("/health/rollout")
    async def health_rollout(request: Request):
        env_key = normalize_environment_api_key()
        if not env_key:
            return JSONResponse(
                status_code=503,
                content={"status": "unhealthy", "detail": "Missing ENVIRONMENT_API_KEY"},
            )
        if not is_api_key_header_authorized(request):
            prefix = _log_env_key_prefix("health/rollout", env_key)
            content = {"status": "healthy", "authorized": False}
            if prefix:
                content["expected_api_key_prefix"] = prefix
            return JSONResponse(status_code=200, content=content)
        return {"ok": True, "authorized": True}


def add_metadata_endpoint(app: FastAPI) -> None:
    """Add standard /metadata endpoint."""

    @app.get("/metadata")
    async def get_metadata(request: Request):
        program_code = get_current_module_source()

        frame = inspect.currentframe()
        try:
            if frame is None:
                module_path = None
            else:
                caller_frame = frame.f_back
                if caller_frame is None:
                    module_path = None
                else:
                    module = inspect.getmodule(caller_frame)
                    module_path = module.__name__ if module else None
        finally:
            del frame

        return {
            "program_code": program_code,
            "module_path": module_path,
            "extraction_method": "inspect",
        }


def _extract_error_message(error_json: Any) -> str:
    if isinstance(error_json, dict):
        error_obj = error_json.get("error")
        if isinstance(error_obj, dict):
            return error_obj.get("message") or error_obj.get("detail") or str(error_obj)
        if isinstance(error_obj, str):
            return error_obj
        return error_json.get("detail") or str(error_json.get("error", "Unknown error"))
    return str(error_json)

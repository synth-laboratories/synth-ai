"""Financial NER local API for Synth prompt optimization benchmarks."""

from __future__ import annotations

import contextlib
import inspect
import json
import os
import socket
import uuid
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Mapping, cast
from urllib.parse import urlparse, urlunparse

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.requests import Request as StarletteRequest

# Synth-AI SDK imports
from synth_ai.sdk.task.apps import LocalAPIEntry, ModalDeploymentConfig, register_local_api
from synth_ai.sdk.task.auth import is_api_key_header_authorized, normalize_environment_api_key
from synth_ai.sdk.task.contracts import (
    RolloutMetrics,
    RolloutMode,
    RolloutRequest,
    RolloutResponse,
    TaskInfo,
)
from synth_ai.sdk.task.datasets import TaskDatasetRegistry, TaskDatasetSpec
from synth_ai.sdk.task.rubrics import Rubric, load_rubric
from synth_ai.sdk.localapi.server import LocalAPIConfig, ProxyConfig, RubricBundle, create_local_api, run_local_api
from synth_ai.sdk.task.trace_correlation_helpers import extract_trace_correlation_id
from synth_ai.sdk.task.vendors import normalize_vendor_keys

# Business logic imports (no synth-ai dependencies)
from financial_ner_business_logic import (
    AVAILABLE_SPLITS,
    DATASET_NAME,
    DEFAULT_SPLIT,
    ENTITY_TYPES,
    REPO_ROOT,
    TOOL_NAME,
    ExtractRequest,
    ExtractResponse,
    FinancialNERDataset,
    FinancialNERScorer,
    get_default_messages_templates,
    get_extract_tool_schema,
    parse_entities_from_tool_call,
)


def get_current_module_code():
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


# Log environment at module load time for debugging
print(
    f"[financial_ner_task_app] Module loaded: DATASET_NAME={DATASET_NAME}, "
    f"HF_HOME={os.getenv('HF_HOME')}, "
    f"HF_DATASETS_CACHE={os.getenv('HF_DATASETS_CACHE')}, "
    f"HF_HUB_CACHE={os.getenv('HF_HUB_CACHE')}",
    flush=True,
)


# Dataset spec for registry
FINANCIAL_NER_DATASET_SPEC = TaskDatasetSpec(
    id="financial_ner",
    name="Financial NER Task",
    version="1.0.0",
    splits=list(AVAILABLE_SPLITS),
    default_split=DEFAULT_SPLIT,
    description="Financial named entity recognition task extracting 7 entity types from financial news.",
)


# Router for additional endpoints
financial_ner_router = APIRouter()


@financial_ner_router.post("/extract", response_model=ExtractResponse)
async def extract_endpoint(req: ExtractRequest, request: Request):
    dataset: FinancialNERDataset = request.app.state.financial_ner_dataset
    return ExtractResponse(entities={etype: [] for etype in ENTITY_TYPES})


def _normalize_chat_url(url: str) -> str:
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
        result = urlunparse((parsed.scheme, parsed.netloc, new_path, parsed.params, query, fragment))
        return result

    if path.endswith("/v1"):
        new_path = f"{path}/chat/completions"
    elif path.endswith("/completions"):
        new_path = path.rsplit("/", 1)[0] + "/chat/completions"
    else:
        new_path = f"{path}/v1/chat/completions" if path else "/v1/chat/completions"

    result = urlunparse((parsed.scheme, parsed.netloc, new_path, parsed.params, query, fragment))
    return result


def _get_default_max_completion_tokens(model_name: str) -> int:
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


async def call_chat_completion(
    policy_config: dict[str, Any],
    placeholders: dict[str, Any],
    default_messages: list[dict[str, str]],
    api_key: str | None = None,
    http_client: Any | None = None,
) -> tuple[str, dict[str, Any] | None, list[dict[str, Any]]]:
    """Call the chat completion API through proxy or directly."""
    # Validate required policy fields
    missing_fields: list[str] = []
    model_val = policy_config.get("model")
    if not isinstance(model_val, str) or not model_val.strip():
        missing_fields.append("model")

    # Resolve routing base - prioritize inference_url
    inference_url_raw = policy_config.get("inference_url")
    api_base_raw = policy_config.get("api_base")
    base_url_raw = policy_config.get("base_url")

    if inference_url_raw:
        route_base = str(inference_url_raw).strip()
        if api_base_raw or base_url_raw:
            with contextlib.suppress(Exception):
                print(
                    f"[TASK_APP] ⚠️  inference_url is set ({route_base}), ignoring api_base/base_url",
                    flush=True,
                )
    else:
        route_base = (
            (api_base_raw or "").strip()
            or (base_url_raw or "").strip()
        )
    if not route_base:
        missing_fields.append("inference_url")
    if missing_fields:
        raise HTTPException(
            status_code=400,
            detail=(
                "Missing policy fields in TOML [prompt_learning.policy]: " + ", ".join(missing_fields)
            ),
        )

    model = policy_config["model"].strip()
    lowered = route_base.lower()
    is_provider_host = ("api.openai.com" in lowered) or ("api.groq.com" in lowered)

    inference_url = _normalize_chat_url(str(route_base))
    temperature = policy_config.get("temperature", 0.7)

    # Determine max_completion_tokens
    if "max_completion_tokens" in policy_config:
        max_tokens = policy_config.get("max_completion_tokens")
        with contextlib.suppress(Exception):
            print(f"[TASK_APP] Using explicit max_completion_tokens: {max_tokens}", flush=True)
    elif "max_tokens" in policy_config:
        max_tokens = policy_config.get("max_tokens")
        with contextlib.suppress(Exception):
            print(f"[TASK_APP] Using legacy max_tokens: {max_tokens}", flush=True)
    else:
        max_tokens = _get_default_max_completion_tokens(model)
        with contextlib.suppress(Exception):
            print(f"[TASK_APP] Using model-based default max_completion_tokens for {model}: {max_tokens}", flush=True)

    with contextlib.suppress(Exception):
        print(f"[TASK_APP] POLICY ROUTE → {inference_url}", flush=True)

    # Render messages
    messages = []
    for msg_template in default_messages:
        role = msg_template.get("role", "user")
        pattern = msg_template.get("pattern", "")
        content = pattern.format(**placeholders)
        messages.append({"role": role, "content": content})

    preview = [
        {"role": m.get("role"), "len": len(m.get("content", "")), "head": (m.get("content", "")[:160])}
        for m in messages
    ]
    print(f"[TASK_APP] MESSAGES: {preview}", flush=True)

    # Set auth headers
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        if is_provider_host:
            headers["Authorization"] = f"Bearer {api_key}"
            with contextlib.suppress(Exception):
                print(f"[TASK_APP] 🔐 DIRECT PROVIDER CALL with API key: {api_key[:12]}...{api_key[-4:]} (len={len(api_key)})", flush=True)
        else:
            headers["X-API-Key"] = api_key
            with contextlib.suppress(Exception):
                print(f"[TASK_APP] 🔐 PROXY ROUTING with API key: {api_key[:12]}...{api_key[-4:]} (len={len(api_key)})", flush=True)
    else:
        with contextlib.suppress(Exception):
            print("[TASK_APP] ⚠️  NO API KEY PROVIDED!", flush=True)

    # Get tool schema
    extract_tool = get_extract_tool_schema()

    # Build payload
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_tokens,
        "tools": [extract_tool],
        "tool_choice": "required" if extract_tool else None,
    }
    if temperature != 0.0:
        payload["temperature"] = temperature

    print(
        f"[TASK_APP] OUTBOUND: model={model} temp={temperature} max={max_tokens} tools=1 choice={TOOL_NAME}",
        flush=True,
    )

    # HTTP client setup
    try:
        import aiohttp
    except ImportError:
        try:
            import httpx
        except Exception as _exc:
            raise HTTPException(status_code=500, detail=f"Neither aiohttp nor httpx available: {_exc}")

    # DNS pre-resolution for proxy URLs
    parsed = urlparse(inference_url)
    host = parsed.hostname or ""
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    print(f"[TASK_APP] PROXY_TARGET: scheme={parsed.scheme} host={host} port={port} path={parsed.path}", flush=True)

    skip_dns_preresolution = is_provider_host

    if skip_dns_preresolution:
        print(f"[TASK_APP] PROXY_DNS: Skipping DNS pre-resolution for direct provider host: {host}", flush=True)
    else:
        try:
            addrinfo = socket.getaddrinfo(host, None, socket.AF_INET)
            ips = sorted({ai[4][0] for ai in addrinfo})
            resolved_ip = ips[0] if ips else None
            print(f"[TASK_APP] PROXY_DNS: resolved {host} -> {resolved_ip} (from {ips})", flush=True)

            if resolved_ip and parsed.scheme == "https":
                netloc = f"{resolved_ip}:{port}" if port else resolved_ip
                inference_url = f"{parsed.scheme}://{netloc}{parsed.path}"
                if parsed.query:
                    inference_url += f"?{parsed.query}"
                headers["_original_host"] = host
                headers["_use_ip"] = "1"
                headers["Host"] = host
                print(f"[TASK_APP] PROXY_URL_REWRITTEN: {inference_url} (will use SNI with host={host}, Host header set)", flush=True)
        except Exception as e:
            print(f"[TASK_APP] PROXY_DNS_ERROR: {e}, continuing with original URL", flush=True)

    if http_client is None:
        raise HTTPException(status_code=500, detail="HTTP client not initialized (should be created at startup)")

    with contextlib.suppress(Exception):
        headers_log = {k: (f"{v[:15]}..." if k == "X-API-Key" and len(v) > 15 else v) for k, v in headers.items()}
        print(f"[TASK_APP] 📤 Sending POST to: {inference_url}", flush=True)
        print(f"[TASK_APP] 📤 With headers: {headers_log}", flush=True)

    # Make HTTP request
    response_json: dict[str, Any] | None = None
    try:
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
                print(f"[TASK_APP] RESPONSE_STATUS: {status_code}", flush=True)

                if status_code != 200:
                    try:
                        error_json = await response.json()
                        error_obj = error_json.get("error")
                        if isinstance(error_obj, dict):
                            error_msg = error_obj.get("message") or error_obj.get("detail") or str(error_obj)
                        elif isinstance(error_obj, str):
                            error_msg = error_obj
                        else:
                            error_msg = error_json.get("detail") or str(error_json.get("error", "Unknown error"))
                        raise HTTPException(status_code=status_code, detail=f"Interceptor/provider error: {error_msg}")
                    except HTTPException:
                        raise
                    except Exception:
                        error_text = (await response.text())[:500]
                        raise HTTPException(status_code=status_code, detail=f"Interceptor/provider returned error: {error_text}")

                try:
                    response_json = await response.json()
                    raw = json.dumps(response_json, ensure_ascii=False)
                    print(f"[TASK_APP] RESPONSE_JSON ({len(raw)} bytes): {raw}", flush=True)
                except Exception:
                    response_text = await response.text()
                    print(f"[TASK_APP] RESPONSE_TEXT ({len(response_text)} bytes): {response_text}", flush=True)
                    if status_code >= 400:
                        raise HTTPException(status_code=status_code, detail=f"HTTP error: {response_text[:200]}")
                    response_json = {}
        else:
            # httpx fallback
            import httpx
            response = await http_client.post(inference_url, json=payload, headers=headers)
            status_code = response.status_code

            if status_code != 200:
                try:
                    error_json = response.json()
                    error_obj = error_json.get("error")
                    if isinstance(error_obj, dict):
                        error_msg = error_obj.get("message") or error_obj.get("detail") or str(error_obj)
                    elif isinstance(error_obj, str):
                        error_msg = error_obj
                    else:
                        error_msg = error_json.get("detail") or str(error_json.get("error", "Unknown error"))
                    raise HTTPException(status_code=status_code, detail=f"Interceptor/provider error: {error_msg}")
                except HTTPException:
                    raise
                except Exception as e:
                    error_text = response.text[:500] if hasattr(response, 'text') else str(e)
                    raise HTTPException(status_code=status_code, detail=f"Interceptor/provider returned error: {error_text}")

            try:
                response_json = response.json()
            except Exception:
                response_text = response.text
                if status_code >= 400:
                    raise HTTPException(status_code=status_code, detail=f"HTTP error: {response_text[:200]}")
                response_json = {}
    except HTTPException:
        raise
    except Exception as e:
        print(f"[TASK_APP] POST_EXCEPTION: {type(e).__name__}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=502, detail=f"Proxy POST failed: {e}")

    if response_json is None:
        raise HTTPException(status_code=502, detail="No response data received")

    # Validate response
    try:
        choices = response_json.get("choices") or []
        first_msg = (choices[0] or {}).get("message", {}) if choices else {}
        tool_calls_raw = first_msg.get("tool_calls", []) or []
        content_text = str(first_msg.get("content", ""))
        if not tool_calls_raw and not content_text.strip():
            raise HTTPException(status_code=502, detail="Empty model output: no tool_calls and no content")
        if tool_calls_raw:
            for call in tool_calls_raw:
                fn = (call or {}).get("function", {}) or {}
                if fn.get("name") != TOOL_NAME:
                    raise HTTPException(status_code=502, detail=f"Unexpected tool name: {fn.get('name')}")
                args_raw = fn.get("arguments", "{}")
                try:
                    args = json.loads(args_raw)
                except Exception:
                    raise HTTPException(status_code=502, detail="Tool call arguments not valid JSON")
                if "entities" not in args or not isinstance(args["entities"], dict):
                    raise HTTPException(status_code=502, detail="Tool call missing valid 'entities' dict")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Response validation failed: {exc}")

    # Parse response
    response_text = ""
    tool_calls = []

    if "choices" in response_json and len(response_json["choices"]) > 0:
        choice = response_json["choices"][0]
        message = choice.get("message", {})
        response_text = message.get("content", "")

        if "tool_calls" in message and message["tool_calls"]:
            for tc in message["tool_calls"]:
                tool_calls.append({
                    "id": tc.get("id", ""),
                    "type": tc.get("type", "function"),
                    "function": {
                        "name": tc.get("function", {}).get("name", ""),
                        "arguments": tc.get("function", {}).get("arguments", "{}"),
                    }
                })

    return response_text, response_json, tool_calls


async def rollout_executor(request: RolloutRequest, fastapi_request: Request) -> RolloutResponse:
    """Execute a rollout for the Financial NER task."""
    dataset: FinancialNERDataset = fastapi_request.app.state.financial_ner_dataset

    with contextlib.suppress(Exception):
        cfg = (request.policy.config or {})
        print(
            f"[TASK_APP] INBOUND_ROLLOUT: run_id={request.run_id} seed={request.env.seed} env={request.env.env_name} "
            f"policy.model={cfg.get('model')} provider={cfg.get('provider')} api_base={cfg.get('inference_url') or cfg.get('api_base') or cfg.get('base_url')}",
            flush=True,
        )

    split = str(((request.env.config or {}).get("split")) or DEFAULT_SPLIT)
    seed = request.env.seed or 0

    sample = dataset.sample(split=split, index=seed)

    entity_types_list = ", ".join(ENTITY_TYPES)
    placeholders = {
        "text": sample["text"],
        "entity_types": entity_types_list,
    }

    default_messages = get_default_messages_templates()

    # Render baseline messages
    rendered_messages: list[dict[str, str]] = []
    for msg_template in default_messages:
        role = msg_template.get("role", "user")
        pattern = msg_template.get("pattern", "")
        content = pattern.format(**placeholders)
        rendered_messages.append({"role": role, "content": content})

    # Extract API key
    inference_url_check = (request.policy.config or {}).get("inference_url") or ""
    is_direct_provider = "api.groq.com" in inference_url_check.lower() or "api.openai.com" in inference_url_check.lower()

    if is_direct_provider:
        if "api.groq.com" in inference_url_check.lower():
            api_key = os.getenv("GROQ_API_KEY")
        elif "api.openai.com" in inference_url_check.lower():
            api_key = os.getenv("OPENAI_API_KEY")
        else:
            api_key = None
    else:
        api_key = (
            fastapi_request.headers.get("X-API-Key")
            or fastapi_request.headers.get("x-api-key")
            or (fastapi_request.headers.get("Authorization", "").replace("Bearer ", "").strip() if fastapi_request.headers.get("Authorization") else None)
            or None
        )

    http_client = getattr(fastapi_request.app.state, "http_client", None)

    response_text, response_json, tool_calls = await call_chat_completion(
        request.policy.config or {},
        placeholders,
        default_messages,
        api_key=api_key,
        http_client=http_client,
    )

    # Validate response
    try:
        raw_upstream = json.dumps(response_json, ensure_ascii=False)
    except Exception:
        raw_upstream = str(response_json)
    print(f"[TASK_APP] UPSTREAM_RESPONSE_JSON ({len(raw_upstream)} bytes): {raw_upstream}", flush=True)

    if not isinstance(response_json, dict) or not response_json:
        raise RuntimeError("Proxy returned missing/empty JSON")
    choices = response_json.get("choices") or []
    if not isinstance(choices, list) or len(choices) == 0:
        raise RuntimeError("Proxy JSON missing choices")
    first_msg = (choices[0] or {}).get("message", {}) if choices else {}
    if not isinstance(first_msg, dict):
        raise RuntimeError("Proxy JSON message malformed")
    tc_list = first_msg.get("tool_calls") or []
    content_text = str(first_msg.get("content", ""))
    if not tc_list and not content_text.strip():
        raise RuntimeError("Proxy JSON has neither tool_calls nor content")

    # Extract predicted entities using business logic helper
    predicted_entities: dict[str, list[str]] = {etype: [] for etype in ENTITY_TYPES}
    if tool_calls:
        for tc in tool_calls:
            if tc.get("function", {}).get("name") == TOOL_NAME:
                args_str = tc.get("function", {}).get("arguments", "{}")
                predicted_entities = parse_entities_from_tool_call(args_str)
                print(f"[TASK_APP] PARSED_TOOL_ENTITIES: {predicted_entities}", flush=True)
    elif response_text:
        try:
            parsed = json.loads(response_text)
            if isinstance(parsed, dict) and "entities" in parsed:
                predicted_entities = parsed["entities"]
        except Exception:
            print(f"[TASK_APP] CONTENT_FALLBACK_PARSE_ERROR: {response_text[:200]}", flush=True)

    if not any(predicted_entities.values()):
        print(f"[TASK_APP] WARNING: No entities extracted from proxy response, returning 0 score", flush=True)

    expected_entities = sample["entities"]

    # Score using business logic
    correct_types, total_types, reward = FinancialNERScorer.score_entities(predicted_entities, expected_entities)
    is_correct = reward == 1.0

    print(
        f"[TASK_APP] PREDICTION: expected_entities={expected_entities} predicted_entities={predicted_entities} "
        f"correct_types={correct_types}/{total_types} reward={reward:.2f}",
        flush=True,
    )

    with contextlib.suppress(Exception):
        print(
            f"[FINANCIAL_NER_ROLLOUT] run_id={request.run_id} split={sample['split']} "
            f"index={sample['index']} correct_types={correct_types}/{total_types} reward={reward:.2f}",
            flush=True,
        )

    inference_url = (request.policy.config or {}).get("inference_url")
    trace_correlation_id = extract_trace_correlation_id(
        policy_config=request.policy.config or {},
        inference_url=str(inference_url or ""),
        mode=request.mode,
    )

    # Build V3 trace for verifier evaluation
    trace_id = str(uuid.uuid4())
    v3_event_history = [
        {
            "type": "llm_request",
            "step_index": 0,
            "llm_request": {
                "messages": rendered_messages,
                "model": (request.policy.config or {}).get("model", "unknown"),
            },
            "llm_response": {
                "message": {
                    "content": response_text,
                    "tool_calls": tool_calls,
                },
                "model": (request.policy.config or {}).get("model", "unknown"),
            },
        },
        {
            "type": "environment_step",
            "step_index": 1,
            "observation": {
                "text": sample["text"],
                "predicted_entities": predicted_entities,
                "expected_entities": expected_entities,
            },
            "reward": reward,
            "terminated": True,
            "info": {
                "correct": is_correct,
                "correct_types": correct_types,
                "total_types": total_types,
            },
        },
    ]
    trajectory_trace = {
        "schema_version": "3.0",
        "event_history": v3_event_history,
        "markov_blanket_message_history": [],
        "metadata": {
            "trace_id": trace_id,
            "session_id": trace_id,
            "environment": "financial_ner",
            "split": sample["split"],
            "index": sample["index"],
            "correct": is_correct,
            "correct_types": correct_types,
            "expected_entities": expected_entities,
            "correlation_ids": {
                "run_id": request.run_id,
                "seed": seed,
            },
        },
    }
    if trace_correlation_id:
        metadata_block = trajectory_trace.get("metadata")
        if isinstance(metadata_block, dict):
            metadata_block["trace_correlation_id"] = trace_correlation_id
            corr_ids = metadata_block.get("correlation_ids")
            corr_map = dict(corr_ids) if isinstance(corr_ids, dict) else {}
            corr_map.setdefault("trace_correlation_id", trace_correlation_id)
            metadata_block["correlation_ids"] = corr_map

    metrics = RolloutMetrics(
        outcome_reward=reward,
        details={"correct": is_correct, "correct_types": correct_types},
    )

    trace_payload = trajectory_trace

    return RolloutResponse(
        run_id=request.run_id,
        branches={},
        metrics=metrics,
        aborted=False,
        trace_correlation_id=trace_correlation_id,
        trace=trace_payload,
        pipeline_metadata={
            "inference_url": str(inference_url or ""),
            "trace_correlation_id": trace_correlation_id,
        },
    )


def build_dataset() -> tuple[TaskDatasetRegistry, FinancialNERDataset]:
    """Build the dataset registry and dataset instance."""
    registry = TaskDatasetRegistry()
    dataset = FinancialNERDataset()
    registry.register(FINANCIAL_NER_DATASET_SPEC, lambda _spec: dataset, cache=True)
    return registry, dataset


def _base_task_info() -> TaskInfo:
    """Get the base task info for Financial NER."""
    return TaskInfo(
        task={
            "id": "financial_ner",
            "name": "Financial NER Task",
            "version": "1.0.0",
            "action_space": {
                "type": "tool_call",
                "tool_name": TOOL_NAME,
                "description": "Extract named entities from financial text.",
            },
        },
        environment="financial_ner",
        dataset={
            **FINANCIAL_NER_DATASET_SPEC.model_dump(),
        },
        rubric={
            "version": "1",
            "criteria_count": 7,
            "source": "inline",
        },
        inference={
            "supports_proxy": True,
            "tool": TOOL_NAME,
        },
        limits={"max_turns": 1},
        task_metadata={"format": "tool_call"},
    )


def describe_taskset(dataset: FinancialNERDataset) -> Mapping[str, Any]:
    """Describe the taskset for the API."""
    return {
        **FINANCIAL_NER_DATASET_SPEC.model_dump(),
        "num_entity_types": len(ENTITY_TYPES),
        "entity_types": ENTITY_TYPES,
        "sizes": {split: dataset.size(split) for split in AVAILABLE_SPLITS},
    }


def provide_task_instances(dataset: FinancialNERDataset, seeds: Sequence[int]) -> Iterable[TaskInfo]:
    """Provide task instances for the given seeds."""
    base_info = _base_task_info()
    for seed in seeds:
        sample = dataset.sample(split=DEFAULT_SPLIT, index=seed)
        expected_entities = sample["entities"]

        instance_rubric = {
            "outcome": {
                "name": "Entity Extraction Accuracy",
                "criteria": [
                    {
                        "id": f"{etype}_accuracy",
                        "description": f"Correctly extract {etype} entities. Expected: {expected_entities.get(etype, [])}",
                        "weight": 1.0 / len(ENTITY_TYPES),
                        "expected_answer": expected_entities.get(etype, []),
                    }
                    for etype in ENTITY_TYPES
                ]
            },
        }

        dataset_dict = base_info.dataset
        if hasattr(dataset_dict, "model_dump"):
            dataset_dict = dataset_dict.model_dump()
        elif not isinstance(dataset_dict, dict):
            if hasattr(dataset_dict, "__dict__"):
                dataset_dict = dict(dataset_dict.__dict__)
            else:
                dataset_dict = {}

        dataset_dict = {
            **dataset_dict,
            "split": sample["split"],
            "index": sample["index"],
        }

        task_metadata_dict = base_info.task_metadata
        if hasattr(task_metadata_dict, "model_dump"):
            task_metadata_dict = task_metadata_dict.model_dump()
        elif not isinstance(task_metadata_dict, dict):
            if hasattr(task_metadata_dict, "__dict__"):
                task_metadata_dict = dict(task_metadata_dict.__dict__)
            else:
                task_metadata_dict = {}

        task_metadata_dict = {
            **task_metadata_dict,
            "text": sample["text"],
            "expected_entities": expected_entities,
        }

        yield TaskInfo(
            task=base_info.task,
            environment=base_info.environment,
            dataset=dataset_dict,
            rubric=instance_rubric,
            inference=base_info.inference,
            limits=base_info.limits,
            task_metadata=task_metadata_dict,
        )


OUTCOME_RUBRIC: Rubric = cast(
    Rubric,
    load_rubric(
        {
            "version": "1",
            "goal_text": "Extract all named entities of specified types from financial text.",
            "aggregation": "weighted_sum",
            "criteria": [
                {
                    "id": f"{etype}_extraction",
                    "description": f"Correctly extract all {etype} entities from the text.",
                    "weight": 1.0 / len(ENTITY_TYPES),
                }
                for etype in ENTITY_TYPES
            ],
        }
    ),
)

EVENTS_RUBRIC: Rubric = cast(
    Rubric,
    load_rubric(
        {
            "version": "1",
            "goal_text": "Use the extract_entities tool correctly.",
            "aggregation": "weighted_sum",
            "criteria": [
                {
                    "id": "tool_usage",
                    "description": "Properly invoke the extract_entities tool with correct JSON format.",
                    "weight": 1.0,
                }
            ],
        }
    ),
)


class RolloutRequestWrapper(BaseModel):
    """Wrapper for RolloutRequest that makes mode optional with defaults."""
    run_id: str
    env: dict[str, Any] = {}
    policy: dict[str, Any] = {}
    mode: str | None = None
    record: dict[str, Any] | None = None
    on_done: str = "reset"
    safety: dict[str, Any] | None = None
    training_session_id: str | None = None
    synth_base_url: str | None = None

    def to_rollout_request(self) -> RolloutRequest:
        """Convert wrapper to proper RolloutRequest with defaults."""
        from synth_ai.sdk.task.contracts import RolloutEnvSpec, RolloutPolicySpec, RolloutRecordConfig, RolloutSafetyConfig

        return RolloutRequest(
            run_id=self.run_id,
            env=RolloutEnvSpec(**self.env) if isinstance(self.env, dict) else self.env,
            policy=RolloutPolicySpec(**self.policy) if isinstance(self.policy, dict) else self.policy,
            mode=RolloutMode(self.mode) if self.mode else RolloutMode.EVAL,
            record=RolloutRecordConfig(**self.record) if self.record else RolloutRecordConfig(),
            on_done=self.on_done,
            safety=RolloutSafetyConfig(**self.safety) if self.safety else RolloutSafetyConfig(),
            training_session_id=self.training_session_id,
            synth_base_url=self.synth_base_url,
        )


def build_config() -> LocalAPIConfig:
    """Build the LocalAPIConfig for the local API."""
    registry, dataset = build_dataset()
    base_info = _base_task_info()

    print("[financial_ner_task_app] Preloading dataset splits...", flush=True)
    try:
        dataset.ensure_ready(AVAILABLE_SPLITS)
        print(f"[financial_ner_task_app] Dataset preloaded successfully: {[dataset.size(s) for s in AVAILABLE_SPLITS]} examples", flush=True)
    except Exception as exc:
        print(f"[financial_ner_task_app] WARNING: Dataset preload failed: {exc}", flush=True)
        import traceback
        traceback.print_exc()

    proxy_keys = normalize_vendor_keys()
    proxy_config = ProxyConfig(
        enable_openai=proxy_keys.get("OPENAI_API_KEY") is not None,
        enable_groq=proxy_keys.get("GROQ_API_KEY") is not None,
        system_hint="Use the extract_entities tool to extract named entities from financial text.",
    )

    async def startup_http_client(app: Any) -> None:
        try:
            import aiohttp
            timeout = aiohttp.ClientTimeout(total=30.0)
            connector = aiohttp.TCPConnector(
                limit=10,
                limit_per_host=5,
                ttl_dns_cache=300,
                use_dns_cache=True,
            )
            app.state.http_client = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
            )
            print("[financial_ner_task_app] Created app-level aiohttp client session singleton", flush=True)
        except ImportError:
            try:
                import httpx
                app.state.http_client = httpx.AsyncClient(
                    timeout=30.0,
                    limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
                )
                print("[financial_ner_task_app] Created app-level httpx client singleton (fallback)", flush=True)
            except Exception as exc:
                print(f"[financial_ner_task_app] WARNING: Failed to create http client: {exc}", flush=True)
                app.state.http_client = None
        except Exception as exc:
            print(f"[financial_ner_task_app] WARNING: Failed to create aiohttp client: {exc}", flush=True)
            app.state.http_client = None

    async def shutdown_http_client(app: Any) -> None:
        http_client = getattr(app.state, "http_client", None)
        if http_client is not None:
            try:
                if hasattr(http_client, 'close'):
                    await http_client.close()
                elif hasattr(http_client, 'aclose'):
                    await http_client.aclose()
                print("[financial_ner_task_app] Closed app-level http client", flush=True)
            except Exception as exc:
                print(f"[financial_ner_task_app] WARNING: Error closing http client: {exc}", flush=True)

    config = LocalAPIConfig(
        app_id="financial_ner",
        name="Financial NER Task",
        description="Financial named entity recognition task for extracting entities from financial news.",
        base_task_info=base_info,
        describe_taskset=lambda: describe_taskset(dataset),
        provide_task_instances=lambda seeds: provide_task_instances(dataset, seeds),
        rollout=rollout_executor,
        dataset_registry=registry,
        rubrics=RubricBundle(outcome=OUTCOME_RUBRIC, events=EVENTS_RUBRIC),
        proxy=proxy_config,
        routers=(financial_ner_router,),
        app_state={"financial_ner_dataset": dataset},
        cors_origins=["*"],
        startup_hooks=[startup_http_client],
        shutdown_hooks=[shutdown_http_client],
    )
    return config


register_local_api(
    entry=LocalAPIEntry(
        api_id="financial_ner",
        description="Financial NER local API for entity extraction from financial text.",
        config_factory=build_config,
        aliases=("financial-entities",),
        modal=ModalDeploymentConfig(
            app_name="synth-financial-ner",
            pip_packages=(
                "fastapi>=0.115.0",
                "pydantic>=2.0.0",
                "httpx>=0.26.0",
            ),
            extra_local_dirs=((str(REPO_ROOT / "synth_ai"), "/opt/synth_ai_repo/synth_ai"),),
        ),
    )
)

# Modal deployment
try:
    import modal

    app = modal.App("synth-financial-ner")

    _image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "synth-ai",
            "fastapi>=0.115.0",
            "pydantic>=2.0.0",
            "httpx>=0.26.0",
            "python-dotenv>=1.0.0",
        )
        .env({"PYTHONPATH": "/opt/synth_ai_repo"})
        .add_local_dir(str(REPO_ROOT / "synth_ai"), "/opt/synth_ai_repo/synth_ai", copy=True)
    )
    _env_file = REPO_ROOT / ".env"
    if _env_file.exists():
        _image = _image.add_local_file(str(_env_file), "/opt/synth_ai_repo/.env")

    @app.function(
        image=_image,
        timeout=600,
    )
    @modal.asgi_app()
    def web():
        return fastapi_app()

except ImportError:
    pass


def fastapi_app():
    """Return the FastAPI application for Modal or other ASGI hosts."""
    with contextlib.suppress(Exception):
        load_dotenv(str(REPO_ROOT / ".env"), override=False)

    print(f"[financial_ner] Creating local API...", flush=True)
    app = create_local_api(build_config())
    print(f"[financial_ner] Local API created, attempting to remove SDK /rollout route...", flush=True)

    try:
        all_routes = list(app.routes) + list(getattr(app, 'router', type('', (), {'routes': []})()).routes)
        print(f"[financial_ner] All routes found: {[getattr(r, 'path', 'unknown') for r in all_routes]}", flush=True)

        if hasattr(app, "router"):
            routes_to_remove = []
            for route in list(app.router.routes):
                if hasattr(route, "path") and route.path == "/rollout":
                    if hasattr(route, "methods") and "POST" in (route.methods or set()):
                        routes_to_remove.append(route)
                        print(f"[financial_ner] Found /rollout POST in router to remove", flush=True)

            for route in routes_to_remove:
                app.router.routes.remove(route)
                print(f"[financial_ner] Removed /rollout POST from router", flush=True)
    except Exception as e:
        print(f"[financial_ner] Could not remove routes: {e}", flush=True)

    config_instance = build_config()

    @app.post("/rollout")
    async def custom_rollout(request: StarletteRequest):
        """Custom rollout endpoint that handles missing ops/mode fields."""
        try:
            body = await request.json()

            if "ops" not in body:
                body["ops"] = []
            if "mode" not in body:
                body["mode"] = "eval"

            from synth_ai.sdk.task.contracts import RolloutEnvSpec, RolloutPolicySpec

            rollout_request = RolloutRequest(
                run_id=body["run_id"],
                env=RolloutEnvSpec(**body["env"]) if isinstance(body["env"], dict) else body["env"],
                policy=RolloutPolicySpec(**body["policy"]) if isinstance(body["policy"], dict) else body["policy"],
                mode=RolloutMode(body["mode"]),
                record=body.get("record"),
                on_done=body.get("on_done", "reset"),
                safety=body.get("safety"),
                training_session_id=body.get("training_session_id"),
                synth_base_url=body.get("synth_base_url"),
            )

            response = await config_instance.rollout(rollout_request, request)
            return response.model_dump()

        except Exception as e:
            print(f"[financial_ner] Error in custom rollout handler: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

    routes_to_remove = []
    for route in list(app.router.routes):
        if hasattr(route, "path") and hasattr(route, "methods"):
            path = getattr(route, "path", None)
            methods = getattr(route, "methods", set()) or set()
            if path in {"/health", "/health/rollout"} and "GET" in methods:
                routes_to_remove.append(route)

    for route in routes_to_remove:
        app.router.routes.remove(route)
        print(f"[financial_ner] Removed default route: {getattr(route, 'path', 'unknown')}", flush=True)

    def _log_env_key_prefix(source: str, env_key: str | None) -> str | None:
        if not env_key:
            return None
        prefix = env_key[: max(1, len(env_key) // 2)]
        print(f"[{source}] expected ENVIRONMENT_API_KEY prefix: {prefix}")
        return prefix

    environment_api_key = os.getenv("ENVIRONMENT_API_KEY", "")
    _log_env_key_prefix("startup", environment_api_key)

    @app.get("/health")
    async def get_health(request: StarletteRequest):
        x_api_key = request.headers.get("x-api-key") or request.headers.get("X-API-Key")
        if environment_api_key and x_api_key != environment_api_key:
            raise HTTPException(status_code=401, detail="Unauthorized")
        return {"status": "ok"}

    @app.get("/health/rollout")
    async def get_health_rollout(request: StarletteRequest):
        x_api_key = request.headers.get("x-api-key") or request.headers.get("X-API-Key")
        if environment_api_key and x_api_key != environment_api_key:
            raise HTTPException(status_code=401, detail="Unauthorized")
        return {"status": "ok"}

    @app.get("/metadata")
    async def get_metadata(request: StarletteRequest):
        """Return program code and metadata for proposer use."""
        program_code = get_current_module_code()

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

    @app.exception_handler(RequestValidationError)
    async def _on_validation_error(request: StarletteRequest, exc: RequestValidationError):
        try:
            hdr = request.headers
            snapshot = {
                "path": str(request.url.path),
                "have_x_api_key": bool(hdr.get("x-api-key")),
                "have_x_api_keys": bool(hdr.get("x-api-keys")),
                "have_authorization": bool(hdr.get("authorization")),
                "errors": exc.errors()[:5],
            }
            print("[422] validation", snapshot, flush=True)
        except Exception:
            pass
        return JSONResponse(
            status_code=422,
            content={"status": "invalid", "detail": exc.errors()[:5]},
        )

    return app


def wrapped_config_factory():
    """Wrapper factory that creates a config and patches the Pydantic validation for /rollout."""
    config = build_config()

    original_executor = config.rollout

    async def patched_rollout_executor(request: RolloutRequest, fastapi_request: Request) -> RolloutResponse:
        """Execute rollout, handling cases where mode might be None/missing."""
        if not request.mode:
            request.mode = RolloutMode.EVAL
        return await original_executor(request, fastapi_request)

    config.rollout = patched_rollout_executor
    return config


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the Financial NER local API")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn autoreload")
    parser.add_argument(
        "--env-file",
        action="append",
        default=[],
        help="Additional .env files to load before startup",
    )
    args = parser.parse_args()

    default_env = Path(__file__).resolve().parents[3] / ".env"
    env_files = [str(default_env)] if default_env.exists() else []
    env_files.extend(args.env_file or [])

    run_local_api(
        wrapped_config_factory,
        host=args.host,
        port=args.port,
        reload=args.reload,
        env_files=env_files,
    )

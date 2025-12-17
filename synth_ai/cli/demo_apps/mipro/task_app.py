"""Banking77 intent classification task app for Synth prompt optimization benchmarks."""

import contextlib
import inspect
import json
import os
import socket
import uuid
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Mapping, cast
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.requests import Request as StarletteRequest
from synth_ai.sdk.task.auth import is_api_key_header_authorized, normalize_environment_api_key
from synth_ai.sdk.task.contracts import (
    RolloutMetrics,
    RolloutRequest,
    RolloutResponse,
    RolloutStep,
    RolloutTrajectory,
    TaskInfo,
)
from synth_ai.sdk.task.datasets import TaskDatasetRegistry, TaskDatasetSpec
from synth_ai.sdk.task.rubrics import Rubric, load_rubric
from synth_ai.sdk.task.server import (
    ProxyConfig,
    RubricBundle,
    TaskAppConfig,
    create_task_app,
    run_task_app,
)
from synth_ai.sdk.task.vendors import normalize_vendor_keys

# Dataset configuration
DATASET_NAME = os.getenv("BANKING77_DATASET_NAME", "banking77")
DEFAULT_SPLIT = "train"
AVAILABLE_SPLITS: tuple[str, ...] = ("train", "test")
TOOL_NAME = "banking77_classify"


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
        except (OSError, TypeError):
            return None
    finally:
        del frame


class Banking77Dataset:
    """Lazy Hugging Face dataset loader for Banking77."""

    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}
        self._label_names: list[str] | None = None

    def _load_split(self, split: str):
        if split not in AVAILABLE_SPLITS:
            raise ValueError(f"Unknown split: {split}. Available: {AVAILABLE_SPLITS}")
        if split not in self._cache:
            try:
                from datasets import load_dataset as _load_dataset  # lazy import
                ds = _load_dataset(DATASET_NAME, split=split, trust_remote_code=False)
                self._cache[split] = ds
                label_feature = ds.features.get("label")  # type: ignore[attr-defined]
                if self._label_names is None and label_feature is not None and hasattr(label_feature, "names"):
                    self._label_names = label_feature.names
            except Exception as exc:
                raise RuntimeError(
                    f"Dataset preparation failed: {split}: Failed to download Banking77 dataset from Hugging Face. "
                    f"Dataset: {DATASET_NAME} | Split: {split}"
                ) from exc
        return self._cache[split]

    def ensure_ready(self, splits: Sequence[str]) -> None:
        for split in splits:
            self._load_split(split)

    def size(self, split: str) -> int:
        dataset = self._load_split(split)
        return len(dataset)

    def sample(self, *, split: str, index: int) -> dict[str, Any]:
        dataset = self._load_split(split)
        size = len(dataset)
        if size == 0:
            raise RuntimeError(f"Banking77 split '{split}' is empty")
        idx = int(index) % size
        row = dataset[int(idx)]

        label_idx = int(row.get("label", 0))
        label_text = self.get_label_name(label_idx)

        return {
            "index": idx,
            "split": split,
            "text": str(row.get("text", "")),
            "label": label_text,
            "label_idx": label_idx,
        }

    def get_label_name(self, label_idx: int) -> str:
        if self._label_names is None:
            self._load_split(DEFAULT_SPLIT)
        if self._label_names and 0 <= label_idx < len(self._label_names):
            return self._label_names[label_idx]
        return f"label_{label_idx}"

    @property
    def label_names(self) -> list[str]:
        if self._label_names is None:
            self._load_split(DEFAULT_SPLIT)
        return self._label_names or []


banking77_router = APIRouter()


BANKING77_DATASET_SPEC = TaskDatasetSpec(
    id="banking77",
    name="Banking77 Intent Classification",
    version="1.0.0",
    splits=list(AVAILABLE_SPLITS),
    default_split=DEFAULT_SPLIT,
    description="Banking customer query intent classification with 77 intent categories.",
)


class ClassifyReq(BaseModel):
    query: str


class ClassifyRes(BaseModel):
    intent: str
    confidence: float | None = None


@banking77_router.post("/classify", response_model=ClassifyRes)
async def classify_endpoint(req: ClassifyReq, request: Request):
    _ = request.app.state.banking77_dataset  # Dataset loaded but not used in this stub endpoint
    return ClassifyRes(intent="activate_my_card", confidence=None)


async def call_chat_completion(
    policy_config: dict[str, Any],
    placeholders: dict[str, Any],
    default_messages: list[dict[str, str]],
    api_key: str | None = None,
) -> tuple[str, dict[str, Any] | None, list[dict[str, Any]]]:
    # STRICT: require all policy fields to come from TOML (no defaults)
    missing_fields: list[str] = []
    # Always require model; provider optional when routing via proxy
    model_val = policy_config.get("model")
    if not isinstance(model_val, str) or not model_val.strip():
        missing_fields.append("model")
    # Resolve routing base - ALWAYS prioritize inference_url if provided (trainer-provided interceptor URL)
    # If inference_url is set, use it exclusively and ignore api_base/base_url
    inference_url_raw = policy_config.get("inference_url")
    api_base_raw = policy_config.get("api_base")
    base_url_raw = policy_config.get("base_url")
    
    if inference_url_raw:
        # Trainer provided inference_url (interceptor URL) - use it exclusively
        route_base = str(inference_url_raw).strip()
        if api_base_raw or base_url_raw:
            # Log warning if api_base/base_url are also present (they'll be ignored)
            with contextlib.suppress(Exception):
                print(
                    f"[TASK_APP] ⚠️  inference_url is set ({route_base}), ignoring api_base/base_url",
                    flush=True,
                )
    else:
        # Fallback: use api_base or base_url if inference_url not provided
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
    # Normalize inference URL: allow bases like .../v1 and auto-append /chat/completions
    # Properly handles query strings and interceptor URLs with trial IDs
    # Matches the pattern used in gepa_benchmarks/common.py for consistency
    def _normalize_chat_url(url: str) -> str:
        from urllib.parse import urlparse, urlunparse
        
        u = (url or "").rstrip("/")
        if not u:
            return "/chat/completions"
        
        # Parse URL to separate path from query parameters
        parsed = urlparse(u)
        path = parsed.path.rstrip("/")
        query = parsed.query
        fragment = parsed.fragment
        
        # Already complete
        if path.endswith("/v1/chat/completions") or path.endswith("/chat/completions"):
            return u
        
        # Check if this looks like an interceptor URL with trial_id
        # Interceptor URLs have /v1/ followed by an identifier (e.g., /v1/cli-mipro-..., /v1/gepa-...)
        # These URLs already have /v1/{trial_id} in them, so we should append /chat/completions
        if "/v1/" in path and not path.endswith("/v1"):
            # This is likely an interceptor URL with trial_id - append /chat/completions to path
            new_path = f"{path}/chat/completions"
            # Reconstruct URL with query parameters preserved
            result = urlunparse((parsed.scheme, parsed.netloc, new_path, parsed.params, query, fragment))
            return result
        
        # Standard case: append /v1/chat/completions
        if path.endswith("/v1"):
            new_path = f"{path}/chat/completions"
        elif path.endswith("/completions"):
            new_path = path.rsplit("/", 1)[0] + "/chat/completions"
        else:
            new_path = f"{path}/v1/chat/completions" if path else "/v1/chat/completions"
        
        # Reconstruct URL with query parameters preserved
        result = urlunparse((parsed.scheme, parsed.netloc, new_path, parsed.params, query, fragment))
        return result
    inference_url = _normalize_chat_url(str(route_base))
    temperature = policy_config.get("temperature", 0.7)
    max_tokens = policy_config.get("max_completion_tokens", 100)

    # Loud route log
    with contextlib.suppress(Exception):
        print(f"[TASK_APP] POLICY ROUTE → {inference_url}", flush=True)

    messages = []
    for msg_template in default_messages:
        role = msg_template.get("role", "user")
        pattern = msg_template.get("pattern", "")
        content = pattern.format(**placeholders)
        messages.append({"role": role, "content": content})

    # Loud logging of rendered messages (trim for safety)
    preview = [
        {"role": m.get("role"), "len": len(m.get("content", "")), "head": (m.get("content", "")[:160])}
        for m in messages
    ]
    print(f"[TASK_APP] MESSAGES: {preview}", flush=True)

    # Assert we are NOT hitting a provider host directly for policy
    if is_provider_host:
        # Print full policy config for forensics
        with contextlib.suppress(Exception):
            print(
                f"[TASK_APP] POLICY_CONFIG: {json.dumps(policy_config, ensure_ascii=False)}",
                flush=True,
            )
        raise HTTPException(status_code=502, detail=f"Direct provider URL not allowed for policy: {route_base}")

    # If routing to proxy/interceptor, include task app API key if provided
    headers: dict[str, str]
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
        with contextlib.suppress(Exception):
            print(f"[TASK_APP] 🔐 PROXY ROUTING with API key: {api_key[:12]}...{api_key[-4:]} (len={len(api_key)})", flush=True)
            print(f"[TASK_APP] 🔐 Headers being sent to proxy: {list(headers.keys())}", flush=True)
            # Verify the key is actually in the headers
            assert "X-API-Key" in headers, "X-API-Key missing from headers!"
            assert headers["X-API-Key"] == api_key, "X-API-Key value mismatch!"
            print("[TASK_APP] ✅ Header validation passed: X-API-Key present", flush=True)
    else:
        with contextlib.suppress(Exception):
            print("[TASK_APP] ⚠️  PROXY ROUTING (NO API KEY PROVIDED!)", flush=True)
            print("[TASK_APP] ⚠️  This will likely fail auth at the proxy endpoint", flush=True)

    # Define tool schema for banking77 classification (no enum to keep payload small)
    classify_tool = {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": "Return the predicted banking77 intent label in the `intent` field.",
            "parameters": {
                "type": "object",
                "properties": {"intent": {"type": "string"}},
                "required": ["intent"],
            },
        },
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "tools": [classify_tool],
        "tool_choice": "required" if classify_tool else None,
    }

    print(
        f"[TASK_APP] OUTBOUND: model={model} temp={temperature} max={max_tokens} tools=1 choice={TOOL_NAME}",
        flush=True,
    )

    # Lazy import httpx to avoid top-level import during modal code gen
    try:
        import httpx  # type: ignore
    except Exception as _exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"httpx unavailable: {_exc}") from _exc

    # Proxy target diagnostics (no preflight health; we go straight to POST)
    try:
        parsed = urlparse(inference_url)
        host = parsed.hostname or ""
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        print(f"[TASK_APP] PROXY_TARGET: scheme={parsed.scheme} host={host} port={port} path={parsed.path}", flush=True)
        addrinfo = socket.getaddrinfo(host, None)
        ips = sorted({ai[4][0] for ai in addrinfo})
        print(f"[TASK_APP] PROXY_DNS: ips={ips}", flush=True)
    except Exception as e:
        print(f"[TASK_APP] PROXY_DNS_ERROR: {e}", flush=True)

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Log the actual request about to be sent
        with contextlib.suppress(Exception):
            headers_log = {k: (f"{v[:15]}..." if k == "X-API-Key" and len(v) > 15 else v) for k, v in headers.items()}
            print(f"[TASK_APP] 📤 Sending POST to: {inference_url}", flush=True)
            print(f"[TASK_APP] 📤 With headers: {headers_log}", flush=True)
            print(f"[TASK_APP] 📤 Payload keys: {list(payload.keys())}", flush=True)
            # Final assertion before sending
            if "X-API-Key" in headers:
                print(f"[TASK_APP] ✅ X-API-Key IS in headers (len={len(headers['X-API-Key'])})", flush=True)
            else:
                print("[TASK_APP] ❌ X-API-Key NOT in headers!", flush=True)
        
        try:
            response = await client.post(inference_url, json=payload, headers=headers)
        except Exception as e:
            print(f"[TASK_APP] POST_EXCEPTION: {type(e).__name__}: {e}", flush=True)
            raise HTTPException(status_code=502, detail=f"Proxy POST failed: {e}") from e
        
        # Always print status/headers/body BEFORE any error is raised
        print(f"[TASK_APP] RESPONSE_STATUS: {response.status_code}", flush=True)
        print(f"[TASK_APP] RESPONSE_HEADERS: {dict(response.headers)}", flush=True)
        
        # Handle error responses from interceptor/provider
        if response.status_code != 200:
            try:
                error_json = response.json()
                error_msg = str(error_json.get("error", {}).get("message", error_json.get("error", "Unknown error")))  # type: ignore[misc]
                print(f"[TASK_APP] ❌ Error response from interceptor: {error_msg}", flush=True)
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Interceptor/provider error: {error_msg}"
                )
            except HTTPException:
                raise
            except Exception as e:
                error_text = response.text[:500]
                print(f"[TASK_APP] ❌ Non-JSON error response: {error_text}", flush=True)
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Interceptor/provider returned error: {error_text}"
                ) from e
        
        # Try JSON, fallback to text
        try:
            response_json = response.json()
            raw = json.dumps(response_json, ensure_ascii=False)
            print(f"[TASK_APP] RESPONSE_JSON ({len(raw)} bytes): {raw}", flush=True)
        except Exception:
            response_text = response.text
            print(f"[TASK_APP] RESPONSE_TEXT ({len(response_text)} bytes): {response_text}", flush=True)
            response.raise_for_status()
            # If we got here, raise_for_status didn't throw; keep an empty JSON
            response_json = {}
        # After logging, surface HTTP errors (shouldn't reach here if status != 200)
        response.raise_for_status()

    with contextlib.suppress(Exception):
        usage = response_json.get("usage", {}) if isinstance(response_json, dict) else {}  # type: ignore[misc]
        ch = (response_json.get("choices") or [{}])[0]  # type: ignore[misc]
        txt = (ch.get("message", {}) or {}).get("content", "")  # type: ignore[misc]
        tc = (ch.get("message", {}) or {}).get("tool_calls", [])  # type: ignore[misc]
        print(
            f"[TASK_APP] RESPONSE: usage={usage} choices={len(response_json.get('choices', []))} first_len={len(txt)} tool_calls={len(tc)}",
            flush=True,
        )

    # Hard assertions: require either tool_calls or non-empty content
    try:
        choices = response_json.get("choices") or []  # type: ignore[misc]
        first_msg = (choices[0] or {}).get("message", {}) if choices else {}  # type: ignore[misc]
        tool_calls = first_msg.get("tool_calls", []) or []  # type: ignore[misc]
        content_text = str(first_msg.get("content", ""))  # type: ignore[misc]
        if not tool_calls and not content_text.strip():
            raise HTTPException(status_code=502, detail="Empty model output: no tool_calls and no content")
        # If tool_calls present, validate schema
        if tool_calls:
            for call in tool_calls:
                fn = (call or {}).get("function", {}) or {}  # type: ignore[misc]
                if fn.get("name") != TOOL_NAME:  # type: ignore[misc]
                    raise HTTPException(status_code=502, detail=f"Unexpected tool name: {fn.get('name')}")  # type: ignore[misc]
                args_raw = fn.get("arguments", "{}")  # type: ignore[misc]
                try:
                    args = json.loads(args_raw)
                except Exception as e:
                    raise HTTPException(status_code=502, detail="Tool call arguments not valid JSON") from e
                if not str(args.get("intent", "")).strip():  # type: ignore[misc]
                    raise HTTPException(status_code=502, detail="Tool call missing 'intent'")
    except HTTPException:
        raise
    except Exception as exc:
        # Convert unexpected errors to HTTP for visibility
        raise HTTPException(status_code=500, detail=f"Response validation failed: {exc}") from exc

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
    dataset: Banking77Dataset = fastapi_request.app.state.banking77_dataset
    # Inbound snapshot from GEPA
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
    observation = {
        "query": sample["text"],
        "index": sample["index"],
        "split": sample["split"],
        "available_intents": dataset.label_names,
    }

    # Format available intents as a numbered list for the prompt
    intents_list = "\n".join(f"{i+1}. {label}" for i, label in enumerate(dataset.label_names))
    placeholders = {
        "query": sample["text"],
        "available_intents": intents_list,
    }

    default_messages = [
        {
            "role": "system",
            "pattern": (
                "You are an expert banking assistant that classifies customer queries into banking intents. "
                "Given a customer message, respond with exactly one intent label from the provided list using the `banking77_classify` tool."
            ),
        },
        {
            "role": "user",
            "pattern": "Customer Query: {query}\n\nAvailable Intents:\n{available_intents}\n\nClassify this query into one of the above banking intents using the tool call.",
        },
    ]

    response_json: dict[str, Any] | None = None
    response_text = ""
    tool_calls = []
    # Render baseline messages for validation/introspection
    rendered_messages: list[dict[str, str]] = []
    for msg_template in default_messages:
        role = msg_template.get("role", "user")
        pattern = msg_template.get("pattern", "")
        content = pattern.format(**placeholders)
        rendered_messages.append({"role": role, "content": content})
    error_info: dict[str, Any] = {}

    # Extract API key from request headers for forwarding to proxy
    api_key = (
        fastapi_request.headers.get("X-API-Key")
        or fastapi_request.headers.get("x-api-key")
        or (fastapi_request.headers.get("Authorization", "").replace("Bearer ", "").strip() if fastapi_request.headers.get("Authorization") else None)
        or None
    )
    
    # Call proxy - HARD FAILS on any invalid/empty responses. No soft handling.
    response_text, response_json, tool_calls = await call_chat_completion(
        request.policy.config or {},
        placeholders,
        default_messages,
        api_key=api_key,
    )
    # Full upstream JSON must be present and non-empty
    try:
        raw_upstream = json.dumps(response_json, ensure_ascii=False)
    except Exception:
        raw_upstream = str(response_json)
    print(f"[TASK_APP] UPSTREAM_RESPONSE_JSON ({len(raw_upstream)} bytes): {raw_upstream}", flush=True)
    if not isinstance(response_json, dict) or not response_json:
        raise RuntimeError("Proxy returned missing/empty JSON")
    # Must have choices
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
    print(f"[TASK_APP] RAW_TOOL_CALLS: {tool_calls}", flush=True)

    predicted_intent = ""
    if tool_calls:
        for tc in tool_calls:
            if tc.get("function", {}).get("name") == TOOL_NAME:
                args_str = tc.get("function", {}).get("arguments", "{}")
                try:
                    args = json.loads(args_str)
                    predicted_intent = args.get("intent", "")
                    print(f"[TASK_APP] PARSED_TOOL_INTENT: {predicted_intent}", flush=True)
                except Exception:
                    print(f"[TASK_APP] TOOL_PARSE_ERROR: {args_str}", flush=True)
    elif response_text:
        predicted_intent = response_text.strip().split()[0] if response_text.strip() else ""
        print(f"[TASK_APP] CONTENT_FALLBACK_INTENT: {predicted_intent} text_len={len(response_text or '')}", flush=True)

    # Hard-crash if no prediction produced at this point
    if not str(predicted_intent or "").strip():
        raise RuntimeError("No prediction produced from proxy response")

    expected_intent = sample["label"]
    is_correct = (predicted_intent.lower().replace("_", " ") == expected_intent.lower().replace("_", " "))
    reward = 1.0 if is_correct else 0.0

    print(
        f"[TASK_APP] PREDICTION: expected={expected_intent} predicted={predicted_intent} correct={is_correct}",
        flush=True,
    )

    info_payload = {
        "expected_intent": expected_intent,
        "predicted_intent": predicted_intent,
        "response_json": response_json,
        "tool_calls": tool_calls,
        "correct": is_correct,
        # Provide messages so pattern validation can extract them reliably
        "messages": rendered_messages,
        **error_info,
    }

    with contextlib.suppress(Exception):
        print(
            f"[BANKING77_ROLLOUT] run_id={request.run_id} split={sample['split']} "
            f"index={sample['index']} expected={expected_intent} predicted={predicted_intent} "
            f"reward={reward}",
            flush=True,
        )

    step = RolloutStep(
        obs=observation,
        tool_calls=tool_calls,
        reward=reward,
        done=True,
        info=info_payload,
    )

    inference_url = (request.policy.config or {}).get("inference_url")
    trajectory = RolloutTrajectory(  # type: ignore[call-overload]
        env_id=f"banking77::{sample['split']}::{sample['index']}",
        policy_id=request.policy.policy_id or request.policy.policy_name or "policy",
        steps=[step],
        final={"observation": observation, "reward": reward},  # type: ignore[arg-type]
        length=1,
        inference_url=str(inference_url or ""),
    )

    metrics = RolloutMetrics(
        episode_returns=[reward],
        mean_return=reward,
        num_steps=1,
        num_episodes=1,
        outcome_score=reward,
        events_score=reward,
        details={"correct": is_correct},
    )

    trace_payload = None
    include_trace = bool(
        (request.record and getattr(request.record, "return_trace", False))
        or os.getenv("TASKAPP_TRACING_ENABLED")
    )
    if include_trace:
        trace_payload = {
            "session_id": str(uuid.uuid4()),
            "events_count": 1,
            "decision_rewards": [reward],
            "metadata": {
                "env": "banking77",
                "split": sample["split"],
                "index": sample["index"],
                "correct": is_correct,
            },
        }

    return RolloutResponse(
        run_id=request.run_id,
        trajectories=[trajectory],
        branches={},
        metrics=metrics,
        aborted=False,
        ops_executed=2,
        trace=trace_payload,
    )


def build_dataset() -> tuple[TaskDatasetRegistry, Banking77Dataset]:
    registry = TaskDatasetRegistry()
    dataset = Banking77Dataset()
    # Lazy load dataset on first use to avoid cold-start latency/timeouts
    registry.register(BANKING77_DATASET_SPEC, lambda _spec: dataset, cache=True)
    return registry, dataset


def _base_task_info() -> TaskInfo:
    return TaskInfo(  # type: ignore[call-overload]
        task={  # type: ignore[arg-type]
            "id": "banking77",
            "name": "Banking77 Intent Classification",
            "version": "1.0.0",
            "action_space": {
                "type": "tool_call",
                "tool_name": TOOL_NAME,
                "description": "Classify banking queries into one of 77 intent categories.",
            },
        },
        environment="banking77",
        dataset={  # type: ignore[arg-type]
            **BANKING77_DATASET_SPEC.model_dump(),
            "hf_dataset": DATASET_NAME,
        },
        rubric={  # type: ignore[arg-type]
            "version": "1",
            "criteria_count": 1,
            "source": "inline",
        },
        inference={  # type: ignore[arg-type]
            "supports_proxy": True,
            "tool": TOOL_NAME,
        },
        limits={"max_turns": 1},  # type: ignore[arg-type]
        task_metadata={"format": "tool_call"},  # type: ignore[arg-type]
    )


def describe_taskset(dataset: Banking77Dataset) -> Mapping[str, Any]:
    return {
        **BANKING77_DATASET_SPEC.model_dump(),
        "hf_dataset": DATASET_NAME,
        "num_labels": len(dataset.label_names),
        "sizes": {split: dataset.size(split) for split in AVAILABLE_SPLITS},
    }


def provide_task_instances(dataset: Banking77Dataset, seeds: Sequence[int]) -> Iterable[TaskInfo]:
    base_info = _base_task_info()
    # Convert pydantic models to dicts for spreading
    base_dataset = base_info.dataset.model_dump() if hasattr(base_info.dataset, 'model_dump') else dict(base_info.dataset)
    base_metadata = base_info.task_metadata.model_dump() if hasattr(base_info.task_metadata, 'model_dump') else dict(base_info.task_metadata)
    for seed in seeds:
        sample = dataset.sample(split=DEFAULT_SPLIT, index=seed)
        yield TaskInfo(  # type: ignore[call-overload]
            task=base_info.task,
            environment=base_info.environment,
            dataset={  # type: ignore[arg-type]
                **base_dataset,
                "split": sample["split"],
                "index": sample["index"],
            },
            rubric=base_info.rubric,
            inference=base_info.inference,
            limits=base_info.limits,
            task_metadata={
                **base_metadata,
                "query": sample["text"],
            },
        )


OUTCOME_RUBRIC: Rubric = cast(
    Rubric,
    load_rubric(
        {
            "version": "1",
            "goal_text": "Classify banking customer queries into the correct intent category.",
            "aggregation": "weighted_sum",
            "criteria": [
                {
                    "id": "intent_accuracy",
                    "description": "Correctly classify the customer query into the appropriate banking intent.",
                    "weight": 1.0,
                }
            ],
        }
    ),
)

EVENTS_RUBRIC: Rubric = cast(
    Rubric,
    load_rubric(
        {
            "version": "1",
            "goal_text": "Use the banking77_classify tool correctly.",
            "aggregation": "weighted_sum",
            "criteria": [
                {
                    "id": "tool_usage",
                    "description": "Properly invoke the banking77_classify tool with the correct format.",
                    "weight": 1.0,
                }
            ],
        }
    ),
)


def build_config() -> TaskAppConfig:
    registry, dataset = build_dataset()
    base_info = _base_task_info()

    proxy_keys = normalize_vendor_keys()
    proxy_config = ProxyConfig(
        enable_openai=proxy_keys.get("OPENAI_API_KEY") is not None,
        enable_groq=proxy_keys.get("GROQ_API_KEY") is not None,
        system_hint="Use the banking77_classify tool to classify the customer query.",
    )

    config = TaskAppConfig(
        app_id="banking77",
        name="Banking77 Intent Classification Task",
        description="Banking77 dataset task app for classifying customer queries into banking intents.",
        base_task_info=base_info,
        describe_taskset=lambda: describe_taskset(dataset),
        provide_task_instances=lambda seeds: provide_task_instances(dataset, seeds),
        rollout=rollout_executor,
        dataset_registry=registry,
        rubrics=RubricBundle(outcome=OUTCOME_RUBRIC, events=EVENTS_RUBRIC),
        proxy=proxy_config,
        routers=(banking77_router,),
        app_state={"banking77_dataset": dataset},
        cors_origins=["*"],
    )
    return config


def fastapi_app():
    """Return the FastAPI application for Modal or other ASGI hosts."""
        
    app = create_task_app(build_config())
    
    # Replace default health endpoints with auth-tolerant handlers
    # FastAPI matches routes in order, so we need to remove old routes and add new ones
    # Access the router's route registry directly
    routes_to_remove = []
    for route in list(app.router.routes):
        # Check if this is a route (not middleware or other components)
        if hasattr(route, "path") and hasattr(route, "methods"):
            path = getattr(route, "path", None)
            methods = getattr(route, "methods", set()) or set()
            if path in {"/health", "/health/rollout"} and "GET" in methods:
                routes_to_remove.append(route)
    
    # Remove routes from router
    for route in routes_to_remove:
        app.router.routes.remove(route)
        print(f"[banking77] Removed default route: {getattr(route, 'path', 'unknown')}", flush=True)
    
    def _log_env_key_prefix(source: str, env_key: str | None) -> str | None:
        if not env_key:
            return None
        prefix = env_key[: max(1, len(env_key) // 2)]
        print(f"[{source}] expected ENVIRONMENT_API_KEY prefix: {prefix}")
        return prefix
    
    @app.get("/health")
    async def health(request: StarletteRequest):
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
    async def health_rollout(request: StarletteRequest):
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
    
    @app.get("/metadata")
    async def get_metadata(request: StarletteRequest):
        """Return program code and metadata for proposer use.
        
        This endpoint allows task apps to self-extract their own code using inspect,
        keeping the architecture self-contained.
        """
        # Extract code using inspect
        program_code = get_current_module_code()
        
        # Get module path
        import inspect
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
            "program_code": program_code,  # Full source code of task app
            "module_path": module_path,    # Module path (e.g., "examples.task_apps.banking77.banking77_task_app")
            "extraction_method": "inspect", # How code was extracted
        }
    
    @app.exception_handler(RequestValidationError)
    async def _on_validation_error(request: StarletteRequest, exc: RequestValidationError):
        try:
            hdr = request.headers
            snapshot = {
                "path": str(request.url.path),
                "have_x_api_key": bool(hdr.get("x-api-key")),  # type: ignore[misc]
                "have_x_api_keys": bool(hdr.get("x-api-keys")),  # type: ignore[misc]
                "have_authorization": bool(hdr.get("authorization")),  # type: ignore[misc]
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the Banking77 task app locally")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8102)
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn autoreload")
    parser.add_argument(
        "--env-file",
        action="append",
        default=[],
        help="Additional .env files to load before startup",
    )
    args = parser.parse_args()

    # Look for .env at repo root (3 levels up: banking77/ -> task_apps/ -> examples/ -> repo_root/)
    default_env = Path(__file__).resolve().parents[3] / ".env"
    env_files = [str(default_env)] if default_env.exists() else []
    env_files.extend(args.env_file or [])

    run_task_app(
        build_config,
        host=args.host,
        port=args.port,
        reload=args.reload,
        env_files=env_files,
    )

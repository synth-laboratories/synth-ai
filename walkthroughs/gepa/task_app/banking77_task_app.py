"""Banking77 intent classification task app for Synth prompt optimization benchmarks."""

from __future__ import annotations

import contextlib
import json
import os
import uuid
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Mapping, cast
import socket
from urllib.parse import urlparse

# removed top-level httpx and datasets import to allow modal deploy without local deps
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from dotenv import load_dotenv

from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.requests import Request as StarletteRequest

from synth_ai.task.apps import ModalDeploymentConfig, TaskAppEntry, register_task_app
from synth_ai.task.auth import is_api_key_header_authorized, normalize_environment_api_key
from synth_ai.task.contracts import (
    RolloutMetrics,
    RolloutRequest,
    RolloutResponse,
    RolloutStep,
    RolloutTrajectory,
    TaskInfo,
)
from synth_ai.task.datasets import TaskDatasetRegistry, TaskDatasetSpec
from synth_ai.task.rubrics import Rubric, load_rubric
from synth_ai.task.server import ProxyConfig, RubricBundle, TaskAppConfig, create_task_app, run_task_app
from synth_ai.task.vendors import normalize_vendor_keys

# Import task app code extraction utility (from monorepo, but we'll use a local version)
try:
    from app.routes.prompt_learning.utils.task_app_code_extraction import get_current_module_code
except ImportError:
    # Fallback for synth-ai repo (not in monorepo)
    import inspect
    import sys
    
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

def _compute_repo_root() -> Path:
    p = Path(__file__).resolve()
    parents = list(p.parents)
    if len(parents) >= 4:
        # parents[3] exists when file is within repo (e.g., examples/task_apps/…)
        return parents[3]
    # Modal inline deploy: code may be at /root/*.py, but we mount synth_ai at /opt/synth_ai_repo/synth_ai
    if "/opt/synth_ai_repo" in os.getenv("PYTHONPATH", "") or Path("/opt/synth_ai_repo/synth_ai").exists():
        return Path("/opt/synth_ai_repo")
    # Fallback to current working directory
    return Path.cwd()

REPO_ROOT = _compute_repo_root()

# Dataset configuration
DATASET_NAME = os.getenv("BANKING77_DATASET_NAME", "banking77")
DEFAULT_SPLIT = "train"
AVAILABLE_SPLITS: tuple[str, ...] = ("train", "test")
TOOL_NAME = "banking77_classify"

# Log environment at module load time for debugging
print(
    f"[banking77_task_app] Module loaded: DATASET_NAME={DATASET_NAME}, "
    f"HF_HOME={os.getenv('HF_HOME')}, "
    f"HF_DATASETS_CACHE={os.getenv('HF_DATASETS_CACHE')}, "
    f"HF_HUB_CACHE={os.getenv('HF_HUB_CACHE')}",
    flush=True,
)


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
                
                # Normalize dataset name: use "banking77" (canonical name) instead of "PolyAI/banking77"
                # The canonical name works better with cached datasets and avoids script conflicts
                dataset_name = DATASET_NAME
                if dataset_name == "PolyAI/banking77":
                    dataset_name = "banking77"
                
                # Log environment for debugging
                import os
                hf_home = os.getenv("HF_HOME")
                hf_datasets_cache = os.getenv("HF_DATASETS_CACHE")
                hf_hub_cache = os.getenv("HF_HUB_CACHE")
                print(
                    f"[Banking77Dataset] Loading dataset '{dataset_name}' split '{split}' "
                    f"(HF_HOME={hf_home}, HF_DATASETS_CACHE={hf_datasets_cache}, HF_HUB_CACHE={hf_hub_cache})",
                    flush=True,
                )
                
                # Try to load with offline mode if cache is available
                # This prevents network failures when dataset is already cached
                # Use num_proc=0 to disable multiprocessing and avoid threading issues
                try:
                    ds = _load_dataset(
                        dataset_name,
                        split=split,
                        trust_remote_code=False,
                        download_mode="reuse_cache_if_exists",  # Use cached version if available
                        num_proc=0,  # Disable multiprocessing to avoid threading issues
                    )
                except Exception as cache_exc:
                    # If cache load fails, try without download_mode (will attempt download)
                    print(
                        f"[Banking77Dataset] Cache load failed, trying download: {cache_exc}",
                        flush=True,
                    )
                    ds = _load_dataset(
                        dataset_name,
                        split=split,
                        trust_remote_code=False,
                        num_proc=0,  # Disable multiprocessing to avoid threading issues
                    )
                
                self._cache[split] = ds
                if self._label_names is None and hasattr(ds.features.get("label"), "names"):
                    self._label_names = ds.features["label"].names
                print(
                    f"[Banking77Dataset] Successfully loaded {len(ds)} examples from '{dataset_name}' split '{split}'",
                    flush=True,
                )
            except Exception as exc:
                # Preserve original exception details for debugging
                import traceback
                error_details = traceback.format_exc()
                print(
                    f"[Banking77Dataset] Dataset load failed: {exc}\n{error_details}",
                    flush=True,
                )
                raise RuntimeError(
                    f"Dataset preparation failed: {split}: Failed to load Banking77 dataset. "
                    f"Dataset: {DATASET_NAME} | Split: {split} | Error: {exc}"
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


class ClassifyRequest(BaseModel):
    query: str


class ClassifyResponse(BaseModel):
    intent: str
    confidence: float | None = None


@banking77_router.post("/classify", response_model=ClassifyResponse)
async def classify_endpoint(req: ClassifyRequest, request: Request):
    dataset: Banking77Dataset = request.app.state.banking77_dataset
    return ClassifyResponse(intent="activate_my_card", confidence=None)


async def call_chat_completion(
    policy_config: dict[str, Any],
    placeholders: dict[str, Any],
    default_messages: list[dict[str, str]],
    api_key: str | None = None,
    http_client: Any | None = None,
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
    provider = str(policy_config.get("provider", "")).strip() or "groq"
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
    
    # Determine max_completion_tokens based on model name if not explicitly provided
    # Different models have different token requirements:
    # - gpt-5 models use reasoning tokens, need more tokens for reasoning + response
    # - Other models may have different limits
    def get_default_max_completion_tokens(model_name: str) -> int:
        """Get default max_completion_tokens based on model name."""
        model_lower = model_name.lower()
        
        # GPT-5 models use reasoning tokens, need more headroom
        if "gpt-5" in model_lower or "gpt5" in model_lower:
            return 2048  # Allow for reasoning tokens + tool call
        
        # GPT-4 models
        if "gpt-4" in model_lower or "gpt4" in model_lower:
            return 4096  # GPT-4 has larger context, but we don't need that much for classification
        
        # o1 models (reasoning models)
        if "o1" in model_lower or "o3" in model_lower:
            return 16384  # o1/o3 models have very large reasoning token budgets
        
        # Claude models
        if "claude" in model_lower:
            return 4096
        
        # Smaller models (llama, etc.) - default to reasonable limit
        return 512
    
    # Use explicit value if provided, otherwise use model-based default
    if "max_completion_tokens" in policy_config:
        max_tokens = policy_config.get("max_completion_tokens")
        with contextlib.suppress(Exception):
            print(f"[TASK_APP] Using explicit max_completion_tokens: {max_tokens}", flush=True)
    elif "max_tokens" in policy_config:
        # Legacy support for max_tokens
        max_tokens = policy_config.get("max_tokens")
        with contextlib.suppress(Exception):
            print(f"[TASK_APP] Using legacy max_tokens: {max_tokens}", flush=True)
    else:
        max_tokens = get_default_max_completion_tokens(model)
        with contextlib.suppress(Exception):
            print(f"[TASK_APP] Using model-based default max_completion_tokens for {model}: {max_tokens}", flush=True)

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
    # TEMPORARILY DISABLED FOR BASELINE TESTING
    # if is_provider_host:
    #     # Print full policy config for forensics
    #     with contextlib.suppress(Exception):
    #         print(
    #             f"[TASK_APP] POLICY_CONFIG: {json.dumps(policy_config, ensure_ascii=False)}",
    #             flush=True,
    #         )
    #     raise HTTPException(status_code=502, detail=f"Direct provider URL not allowed for policy: {route_base}")

    # Set appropriate auth headers based on whether we're calling provider directly or through proxy
    headers: dict[str, str]
    headers = {"Content-Type": "application/json"}
    if api_key:
        # For direct provider calls, use Authorization: Bearer
        # For proxy/interceptor calls, use X-API-Key
        if is_provider_host:
            headers["Authorization"] = f"Bearer {api_key}"
            with contextlib.suppress(Exception):
                print(f"[TASK_APP] 🔐 DIRECT PROVIDER CALL with API key: {api_key[:12]}...{api_key[-4:]} (len={len(api_key)})", flush=True)
                print(f"[TASK_APP] 🔐 Using Authorization: Bearer header", flush=True)
        else:
            headers["X-API-Key"] = api_key
            with contextlib.suppress(Exception):
                print(f"[TASK_APP] 🔐 PROXY ROUTING with API key: {api_key[:12]}...{api_key[-4:]} (len={len(api_key)})", flush=True)
                print(f"[TASK_APP] 🔐 Headers being sent to proxy: {list(headers.keys())}", flush=True)
    else:
        with contextlib.suppress(Exception):
            print("[TASK_APP] ⚠️  NO API KEY PROVIDED!", flush=True)
            print(f"[TASK_APP] ⚠️  This will likely fail auth", flush=True)

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

    # Build payload - omit temperature if 0.0 (some models only support default value of 1.0)
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_tokens,  # Use max_completion_tokens instead of max_tokens for newer models
        "tools": [classify_tool],
        "tool_choice": "required" if classify_tool else None,
    }
    # Only include temperature if it's not 0.0 (some models don't support 0.0, only default 1.0)
    if temperature != 0.0:
        payload["temperature"] = temperature

    print(
        f"[TASK_APP] OUTBOUND: model={model} temp={temperature} max={max_tokens} tools=1 choice={TOOL_NAME}",
        flush=True,
    )

    # Use aiohttp instead of httpx to avoid threading issues
    # httpx uses ThreadPoolExecutor internally which causes "cannot schedule new futures" errors
    # aiohttp uses async DNS resolution and doesn't have threading dependencies
    try:
        import aiohttp  # type: ignore
    except ImportError:
        # Fallback to httpx if aiohttp not available
        try:
            import httpx  # type: ignore
            use_aiohttp = False
        except Exception as _exc:  # pragma: no cover
            raise HTTPException(status_code=500, detail=f"Neither aiohttp nor httpx available: {_exc}")
    else:
        use_aiohttp = True

    # Proxy target diagnostics and DNS pre-resolution
    # Pre-resolve DNS synchronously and use IP address directly to avoid threading issues
    # Both httpx and aiohttp use ThreadPoolExecutor for DNS, which fails if executor is shut down
    # By pre-resolving DNS synchronously and using IP directly, we bypass DNS resolution during request
    # We'll configure SSL to send SNI (Server Name Indication) even when using IP address
    #
    # IMPORTANT: Only do DNS pre-resolution for proxy/interceptor URLs, NOT for direct provider URLs
    # Direct provider URLs (api.openai.com, api.groq.com) require proper SSL/TLS with valid certificates
    # DNS pre-resolution breaks SSL handshake with Cloudflare-backed APIs like Groq
    parsed = urlparse(inference_url)
    host = parsed.hostname or ""
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    print(f"[TASK_APP] PROXY_TARGET: scheme={parsed.scheme} host={host} port={port} path={parsed.path}", flush=True)

    # Pre-resolve DNS synchronously (before any async operations that might trigger executor shutdown)
    # Then use IP address directly in the URL to bypass DNS resolution during request
    # Skip this for direct provider URLs to avoid SSL handshake failures
    resolved_ip = None
    skip_dns_preresolution = is_provider_host  # Skip for api.openai.com, api.groq.com, etc.

    if skip_dns_preresolution:
        print(f"[TASK_APP] PROXY_DNS: Skipping DNS pre-resolution for direct provider host: {host}", flush=True)
    else:
        try:
            # Use synchronous DNS resolution (blocking, but avoids executor issues)
            # This happens before the HTTP request, so executor shutdown won't affect it
            addrinfo = socket.getaddrinfo(host, None, socket.AF_INET)
            ips = sorted({ai[4][0] for ai in addrinfo})
            resolved_ip = ips[0] if ips else None
            print(f"[TASK_APP] PROXY_DNS: resolved {host} -> {resolved_ip} (from {ips})", flush=True)

            # Replace hostname with IP in URL to bypass DNS resolution during request
            if resolved_ip and parsed.scheme == "https":
                # Reconstruct URL with IP address
                netloc = f"{resolved_ip}:{port}" if port else resolved_ip
                inference_url = f"{parsed.scheme}://{netloc}{parsed.path}"
                if parsed.query:
                    inference_url += f"?{parsed.query}"
                # Store original hostname for SNI and Host header
                headers["_original_host"] = host
                headers["_use_ip"] = "1"
                # Set Host header so server knows the original hostname (required for Cloudflare)
                headers["Host"] = host
                print(f"[TASK_APP] PROXY_URL_REWRITTEN: {inference_url} (will use SNI with host={host}, Host header set)", flush=True)
        except Exception as e:
            print(f"[TASK_APP] PROXY_DNS_ERROR: {e}, continuing with original URL", flush=True)
            # Continue with original URL - HTTP client will handle DNS (may fail if executor shut down)

    # Use app-level http client singleton to avoid threading issues
    # aiohttp doesn't use threading, so it's safer than httpx
    if http_client is None:
        raise HTTPException(status_code=500, detail="HTTP client not initialized (should be created at startup)")
    
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
            print(f"[TASK_APP] ❌ X-API-Key NOT in headers!", flush=True)
    
    # Use aiohttp session (doesn't use threading, avoids "cannot schedule new futures" errors)
    # aiohttp.ClientSession.post() returns a ClientResponse that must be used as async context manager
    response_json: dict[str, Any] | None = None
    try:
        # Check if it's aiohttp.ClientSession or httpx.AsyncClient
        is_aiohttp = hasattr(http_client, 'post') and not hasattr(http_client.post, '__call__')
        if not is_aiohttp:
            # Try to detect aiohttp by checking for ClientSession type
            import aiohttp
            is_aiohttp = isinstance(http_client, aiohttp.ClientSession)
        
        if is_aiohttp:
            # aiohttp: post() returns a coroutine that yields ClientResponse (use as async context manager)
            # If using IP address, configure SSL to send SNI (Server Name Indication)
            use_ip = headers.pop("_use_ip", None) is not None
            original_host = headers.pop("_original_host", None)
            request_headers = {k: v for k, v in headers.items() if not k.startswith("_")}
            
            # Prepare SSL settings if using IP address
            ssl_setting = None
            if use_ip and original_host:
                import ssl
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False  # Disable hostname check since we're using IP
                ssl_context.verify_mode = ssl.CERT_NONE  # Disable certificate verification
                ssl_setting = ssl_context
            
            # Make the request - aiohttp will send SNI based on server_hostname parameter
            async with http_client.post(
                inference_url,
                json=payload,
                headers=request_headers,
                ssl=ssl_setting,
                server_hostname=original_host if (use_ip and original_host) else None,
            ) as response:
                status_code = response.status
                
                # Always print status/headers/body BEFORE any error is raised
                print(f"[TASK_APP] RESPONSE_STATUS: {status_code}", flush=True)
                print(f"[TASK_APP] RESPONSE_HEADERS: {dict(response.headers)}", flush=True)
                
                # Handle error responses from interceptor/provider
                if status_code != 200:
                    try:
                        error_json = await response.json()
                        # Extract error message properly - handle both dict and string formats
                        error_obj = error_json.get("error")
                        if isinstance(error_obj, dict):
                            error_msg = error_obj.get("message") or error_obj.get("detail") or str(error_obj)
                        elif isinstance(error_obj, str):
                            error_msg = error_obj
                        else:
                            # Try detail field as fallback
                            error_msg = error_json.get("detail") or str(error_json.get("error", "Unknown error"))
                        
                        print(f"[TASK_APP] ❌ Error response from interceptor: {error_msg}", flush=True)
                        print(f"[TASK_APP] ❌ Full error JSON: {error_json}", flush=True)
                        raise HTTPException(
                            status_code=status_code,
                            detail=f"Interceptor/provider error: {error_msg}"
                        )
                    except HTTPException:
                        raise
                    except Exception as e:
                        error_text = (await response.text())[:500]
                        print(f"[TASK_APP] ❌ Non-JSON error response: {error_text}", flush=True)
                        raise HTTPException(
                            status_code=status_code,
                            detail=f"Interceptor/provider returned error: {error_text}"
                        )
                
                # Try JSON, fallback to text
                try:
                    response_json = await response.json()
                    raw = json.dumps(response_json, ensure_ascii=False)
                    print(f"[TASK_APP] RESPONSE_JSON ({len(raw)} bytes): {raw}", flush=True)
                except Exception:
                    response_text = await response.text()
                    print(f"[TASK_APP] RESPONSE_TEXT ({len(response_text)} bytes): {response_text}", flush=True)
                    if status_code >= 400:
                        raise HTTPException(status_code=status_code, detail=f"HTTP error: {response_text[:200]}")
                    # If we got here, keep an empty JSON
                    response_json = {}
                
                # After logging, surface HTTP errors
                if status_code >= 400:
                    raise HTTPException(status_code=status_code, detail=f"HTTP error: {status_code}")
        else:
            # httpx fallback (shouldn't happen if aiohttp available)
            import httpx
            response = await http_client.post(inference_url, json=payload, headers=headers)
            status_code = response.status_code
            
            print(f"[TASK_APP] RESPONSE_STATUS: {status_code}", flush=True)
            print(f"[TASK_APP] RESPONSE_HEADERS: {dict(response.headers)}", flush=True)
            
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
                    print(f"[TASK_APP] ❌ Error response from interceptor: {error_msg}", flush=True)
                    raise HTTPException(status_code=status_code, detail=f"Interceptor/provider error: {error_msg}")
                except HTTPException:
                    raise
                except Exception as e:
                    error_text = response.text[:500] if hasattr(response, 'text') else str(e)
                    print(f"[TASK_APP] ❌ Non-JSON error response: {error_text}", flush=True)
                    raise HTTPException(status_code=status_code, detail=f"Interceptor/provider returned error: {error_text}")
            
            try:
                response_json = response.json()
                raw = json.dumps(response_json, ensure_ascii=False)
                print(f"[TASK_APP] RESPONSE_JSON ({len(raw)} bytes): {raw}", flush=True)
            except Exception:
                response_text = response.text
                print(f"[TASK_APP] RESPONSE_TEXT ({len(response_text)} bytes): {response_text}", flush=True)
                if status_code >= 400:
                    raise HTTPException(status_code=status_code, detail=f"HTTP error: {response_text[:200]}")
                response_json = {}
            
            if status_code >= 400:
                raise HTTPException(status_code=status_code, detail=f"HTTP error: {status_code}")
    except HTTPException:
        raise
    except Exception as e:
        print(f"[TASK_APP] POST_EXCEPTION: {type(e).__name__}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=502, detail=f"Proxy POST failed: {e}")
    
    if response_json is None:
        raise HTTPException(status_code=502, detail="No response data received")

    with contextlib.suppress(Exception):
        usage = response_json.get("usage", {}) if isinstance(response_json, dict) else {}
        ch = (response_json.get("choices") or [{}])[0]
        txt = (ch.get("message", {}) or {}).get("content", "")
        tc = (ch.get("message", {}) or {}).get("tool_calls", [])
        print(
            f"[TASK_APP] RESPONSE: usage={usage} choices={len(response_json.get('choices', []))} first_len={len(txt)} tool_calls={len(tc)}",
            flush=True,
        )

    # Hard assertions: require either tool_calls or non-empty content
    try:
        choices = response_json.get("choices") or []
        first_msg = (choices[0] or {}).get("message", {}) if choices else {}
        tool_calls = first_msg.get("tool_calls", []) or []
        content_text = str(first_msg.get("content", ""))
        if not tool_calls and not content_text.strip():
            raise HTTPException(status_code=502, detail="Empty model output: no tool_calls and no content")
        # If tool_calls present, validate schema
        if tool_calls:
            for call in tool_calls:
                fn = (call or {}).get("function", {}) or {}
                if fn.get("name") != TOOL_NAME:
                    raise HTTPException(status_code=502, detail=f"Unexpected tool name: {fn.get('name')}")
                args_raw = fn.get("arguments", "{}")
                try:
                    args = json.loads(args_raw)
                except Exception:
                    raise HTTPException(status_code=502, detail="Tool call arguments not valid JSON")
                if not str(args.get("intent", "")).strip():
                    raise HTTPException(status_code=502, detail="Tool call missing 'intent'")
    except HTTPException:
        raise
    except Exception as exc:
        # Convert unexpected errors to HTTP for visibility
        raise HTTPException(status_code=500, detail=f"Response validation failed: {exc}")

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
    
    # Get app-level httpx client singleton (created at startup, reused across requests)
    http_client = getattr(fastapi_request.app.state, "http_client", None)
    
    # Call proxy - HARD FAILS on any invalid/empty responses. No soft handling.
    response_text, response_json, tool_calls = await call_chat_completion(
        request.policy.config or {},
        placeholders,
        default_messages,
        api_key=api_key,
        http_client=http_client,
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
    trajectory = RolloutTrajectory(
        env_id=f"banking77::{sample['split']}::{sample['index']}",
        policy_id=request.policy.policy_id or request.policy.policy_name or "policy",
        steps=[step],
        final={"observation": observation, "reward": reward},
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
    return TaskInfo(
        task={
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
        dataset={
            **BANKING77_DATASET_SPEC.model_dump(),
            "hf_dataset": DATASET_NAME,
        },
        rubric={
            "version": "1",
            "criteria_count": 1,
            "source": "inline",
        },
        inference={
            "supports_proxy": True,
            "tool": TOOL_NAME,
        },
        limits={"max_turns": 1},
        task_metadata={"format": "tool_call"},
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
    for seed in seeds:
        sample = dataset.sample(split=DEFAULT_SPLIT, index=seed)
        expected_intent = sample["label"]
        
        # Create instance-specific rubric with expected answer interpolated
        # RubricInfo structure: outcome requires name field (RubricSection)
        instance_rubric = {
            "outcome": {
                "name": "Intent Classification Accuracy",
                "criteria": [
                    {
                        "id": "intent_accuracy",
                        "description": f"Did it provide the correct intent: {expected_intent}?",
                        "weight": 1.0,
                        "expected_answer": expected_intent,
                    }
                ]
            },
        }
        
        # Convert dataset to dict if it's a Pydantic model
        dataset_dict = base_info.dataset
        if hasattr(dataset_dict, "model_dump"):
            dataset_dict = dataset_dict.model_dump()
        elif not isinstance(dataset_dict, dict):
            # Try to convert DatasetInfo or other objects to dict
            if hasattr(dataset_dict, "__dict__"):
                dataset_dict = dict(dataset_dict.__dict__)
            else:
                dataset_dict = {}
        
        # Merge in instance-specific fields
        dataset_dict = {
            **dataset_dict,
            "split": sample["split"],
            "index": sample["index"],
        }
        
        # Convert task_metadata to dict if needed
        task_metadata_dict = base_info.task_metadata
        if hasattr(task_metadata_dict, "model_dump"):
            task_metadata_dict = task_metadata_dict.model_dump()
        elif not isinstance(task_metadata_dict, dict):
            # Try to convert to dict
            if hasattr(task_metadata_dict, "__dict__"):
                task_metadata_dict = dict(task_metadata_dict.__dict__)
            else:
                task_metadata_dict = {}
        
        # Merge in instance-specific fields
        task_metadata_dict = {
            **task_metadata_dict,
            "query": sample["text"],
            "expected_intent": expected_intent,  # CRITICAL: Include expected_intent for judge validation
        }
        
        yield TaskInfo(
            task=base_info.task,
            environment=base_info.environment,
            dataset=dataset_dict,
            rubric=instance_rubric,  # Use instance-specific rubric with expected answer
            inference=base_info.inference,
            limits=base_info.limits,
            task_metadata=task_metadata_dict,
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

    # Preload dataset at startup to avoid threading issues during request handling
    # This ensures the dataset is loaded synchronously before any async requests
    print("[banking77_task_app] Preloading dataset splits...", flush=True)
    try:
        dataset.ensure_ready(AVAILABLE_SPLITS)
        print(f"[banking77_task_app] Dataset preloaded successfully: {[dataset.size(s) for s in AVAILABLE_SPLITS]} examples", flush=True)
    except Exception as exc:
        print(f"[banking77_task_app] WARNING: Dataset preload failed: {exc}", flush=True)
        # Continue anyway - will load lazily on first use
        import traceback
        traceback.print_exc()

    proxy_keys = normalize_vendor_keys()
    proxy_config = ProxyConfig(
        enable_openai=proxy_keys.get("OPENAI_API_KEY") is not None,
        enable_groq=proxy_keys.get("GROQ_API_KEY") is not None,
        system_hint="Use the banking77_classify tool to classify the customer query.",
    )

    # Startup hook: Create aiohttp session singleton
    # Note: aiohttp still uses ThreadPoolExecutor for DNS resolution via run_in_executor
    # This can cause "cannot schedule new futures" errors if the executor is shut down
    # We try to minimize this by using a persistent connector and pre-resolving DNS
    async def startup_http_client(app: Any) -> None:
        try:
            import aiohttp
            timeout = aiohttp.ClientTimeout(total=30.0)
            # Use a persistent connector with DNS caching to minimize DNS lookups
            # Note: This still uses threading internally, but reduces the number of DNS calls
            connector = aiohttp.TCPConnector(
                limit=10,
                limit_per_host=5,
                ttl_dns_cache=300,  # Cache DNS for 5 minutes
                use_dns_cache=True,
            )
            app.state.http_client = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
            )
            print("[banking77_task_app] Created app-level aiohttp client session singleton", flush=True)
        except ImportError:
            # Fallback to httpx if aiohttp not available
            try:
                import httpx
                app.state.http_client = httpx.AsyncClient(
                    timeout=30.0,
                    limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
                )
                print("[banking77_task_app] Created app-level httpx client singleton (fallback)", flush=True)
            except Exception as exc:
                print(f"[banking77_task_app] WARNING: Failed to create http client: {exc}", flush=True)
                app.state.http_client = None
        except Exception as exc:
            print(f"[banking77_task_app] WARNING: Failed to create aiohttp client: {exc}", flush=True)
            app.state.http_client = None
    
    # Shutdown hook: Clean up http client
    async def shutdown_http_client(app: Any) -> None:
        http_client = getattr(app.state, "http_client", None)
        if http_client is not None:
            try:
                # Handle both aiohttp.ClientSession and httpx.AsyncClient
                if hasattr(http_client, 'close'):
                    await http_client.close()
                elif hasattr(http_client, 'aclose'):
                    await http_client.aclose()
                print("[banking77_task_app] Closed app-level http client", flush=True)
            except Exception as exc:
                print(f"[banking77_task_app] WARNING: Error closing http client: {exc}", flush=True)

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
        startup_hooks=[startup_http_client],
        shutdown_hooks=[shutdown_http_client],
    )
    return config


register_task_app(
    entry=TaskAppEntry(
        app_id="banking77",
        description="Banking77 intent classification task app using the banking77 dataset.",
        config_factory=build_config,
        aliases=("banking-intents",),
        modal=ModalDeploymentConfig(
            app_name="synth-banking77",
            pip_packages=(
                "datasets>=2.14.0",
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
    
    # For direct Modal deployment (modal deploy banking77_task_app.py)
    app = modal.App("synth-banking77")
    
    _image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "synth-ai",
            "datasets>=2.14.0",
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
    
    # Load environment from .env if present (works in Modal via added local file)
    with contextlib.suppress(Exception):
        load_dotenv(str(REPO_ROOT / ".env"), override=False)
    
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

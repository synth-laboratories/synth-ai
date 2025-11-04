"""Banking77 intent classification task app for Synth prompt optimization benchmarks."""

from __future__ import annotations

import contextlib
import json
import os
import uuid
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Mapping, cast

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
                if self._label_names is None and hasattr(ds.features.get("label"), "names"):
                    self._label_names = ds.features["label"].names
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
) -> tuple[str, dict[str, Any] | None, list[dict[str, Any]]]:
    # STRICT: require all policy fields to come from TOML (no defaults)
    missing_fields: list[str] = []
    for key in ("model", "provider"):
        value = policy_config.get(key, None)
        if not isinstance(value, str) or not value.strip():
            missing_fields.append(key)
    # Resolve routing base from any of the accepted keys
    route_base = (
        (policy_config.get("inference_url") or "").strip()
        or (policy_config.get("api_base") or "").strip()
        or (policy_config.get("base_url") or "").strip()
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
    provider = policy_config["provider"].strip()
    # Hard-block direct provider hosts: must be proxy/interceptor URL
    lowered = route_base.lower()
    if ("api.openai.com" in lowered) or ("api.groq.com" in lowered):
        raise HTTPException(status_code=400, detail=f"Direct provider URL not allowed: {route_base}")
    # Normalize inference URL: allow bases like .../v1 and auto-append /chat/completions
    def _normalize_chat_url(url: str) -> str:
        u = (url or "").rstrip("/")
        if u.endswith("/chat/completions"):
            return u
        if u.endswith("/v1"):
            return u + "/chat/completions"
        if u.endswith("/completions"):
            return u.rsplit("/", 1)[0] + "/chat/completions"
        return u + "/chat/completions"
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

    api_key = None
    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise HTTPException(status_code=400, detail=f"Missing API key for provider: {provider}")

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    # Lazy import httpx to avoid top-level import during modal code gen
    try:
        import httpx  # type: ignore
    except Exception as _exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"httpx unavailable: {_exc}")

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(inference_url, json=payload, headers=headers)
        response.raise_for_status()
        response_json = response.json()

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

    split = str(((request.env.config or {}).get("split")) or DEFAULT_SPLIT)
    seed = request.env.seed or 0

    sample = dataset.sample(split=split, index=seed)
    observation = {
        "query": sample["text"],
        "index": sample["index"],
        "split": sample["split"],
        "available_intents": dataset.label_names,
    }

    placeholders = {"query": sample["text"]}

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
            "pattern": "Customer Query: {query}\n\nClassify this query into one of the banking intents using the tool call.",
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

    try:
        response_text, response_json, tool_calls = await call_chat_completion(
            request.policy.config or {},
            placeholders,
            default_messages,
        )
    except HTTPException as http_err:
        error_info = {"error": str(http_err.detail), "code": http_err.status_code}
    except Exception as exc:
        error_info = {"error": str(exc)}

    predicted_intent = ""
    if tool_calls:
        for tc in tool_calls:
            if tc.get("function", {}).get("name") == TOOL_NAME:
                args_str = tc.get("function", {}).get("arguments", "{}")
                try:
                    args = json.loads(args_str)
                    predicted_intent = args.get("intent", "")
                except Exception:
                    pass
    elif response_text:
        predicted_intent = response_text.strip().split()[0] if response_text.strip() else ""

    expected_intent = sample["label"]
    is_correct = (predicted_intent.lower().replace("_", " ") == expected_intent.lower().replace("_", " "))
    reward = 1.0 if is_correct else 0.0

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
        yield TaskInfo(
            task=base_info.task,
            environment=base_info.environment,
            dataset={
                **base_info.dataset,
                "split": sample["split"],
                "index": sample["index"],
            },
            rubric=base_info.rubric,
            inference=base_info.inference,
            limits=base_info.limits,
            task_metadata={
                **base_info.task_metadata,
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
    filtered_routes = []
    for route in app.router.routes:
        path = getattr(route, "path", None)
        methods = getattr(route, "methods", set()) or set()
        if path in {"/health", "/health/rollout"} and "GET" in methods:
            continue
        filtered_routes.append(route)
    app.router.routes = filtered_routes
    
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

    default_env = Path(__file__).resolve().parents[2] / ".env"
    env_files = [str(default_env)] if default_env.exists() else []
    env_files.extend(args.env_file or [])

    run_task_app(
        build_config,
        host=args.host,
        port=args.port,
        reload=args.reload,
        env_files=env_files,
    )

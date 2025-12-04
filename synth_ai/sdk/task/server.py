"""FastAPI scaffolding for Task Apps (local dev + deployment)."""

from __future__ import annotations

import asyncio
import inspect
import os
from collections.abc import Awaitable, Callable, Iterable, Mapping, MutableMapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, Depends, FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware import Middleware

from synth_ai.core.telemetry import log_info

from .auth import normalize_environment_api_key, require_api_key_dependency
from .contracts import RolloutRequest, RolloutResponse, TaskInfo
from .datasets import TaskDatasetRegistry
from .errors import http_exception
from .json import to_jsonable
from .proxy import (
    inject_system_hint,
    prepare_for_groq,
    prepare_for_openai,
    synthesize_tool_call_if_missing,
)
from .rubrics import Rubric
from .vendors import get_groq_key_or_503, get_openai_key_or_503, normalize_vendor_keys

TasksetDescriptor = Callable[[], Mapping[str, Any] | Awaitable[Mapping[str, Any]]]
InstanceProvider = Callable[[Sequence[int]], Iterable[TaskInfo] | Awaitable[Iterable[TaskInfo]]]
RolloutExecutor = Callable[[RolloutRequest, Request], Any | Awaitable[Any]]


def _default_app_state() -> dict[str, Any]:
    return {}


@dataclass(slots=True)
class RubricBundle:
    """Optional rubrics advertised by the task app."""

    outcome: Rubric | None = None
    events: Rubric | None = None


@dataclass(slots=True)
class ProxyConfig:
    """Configuration for optional vendor proxy endpoints."""

    enable_openai: bool = False
    enable_groq: bool = False
    system_hint: str | None = None
    openai_url: str = "https://api.openai.com/v1/chat/completions"
    groq_url: str = "https://api.groq.com/openai/v1/chat/completions"


@dataclass(slots=True)
class TaskAppConfig:
    """Declarative configuration describing a Task App.
    
    A TaskAppConfig defines all the components needed to create a task app:
    - Task information (base_task_info)
    - Task set description (describe_taskset)
    - Task instance provider (provide_task_instances)
    - Rollout executor (rollout)
    - Optional rubrics, datasets, proxy config, etc.
    
    This config is used by `create_task_app()` to build a FastAPI application
    that implements the Synth AI task app contract.
    
    Example:
        >>> from synth_ai.sdk.task.server import TaskAppConfig, create_task_app
        >>> from synth_ai.sdk.task.contracts import TaskInfo
        >>> 
        >>> def build_config() -> TaskAppConfig:
        ...     return TaskAppConfig(
        ...         app_id="my_task",
        ...         name="My Task App",
        ...         description="A simple task app",
        ...         base_task_info=TaskInfo(...),
        ...         describe_taskset=lambda: {"splits": ["train", "val"]},
        ...         provide_task_instances=lambda seeds: [...],
        ...         rollout=lambda req, r: {...},
        ...     )
        >>> 
        >>> app = create_task_app(build_config())
        >>> # app is a FastAPI instance ready to run
    
    Attributes:
        app_id: Unique identifier for this task app
        name: Human-readable name
        description: Description of what this task app does
        base_task_info: Base TaskInfo that all instances inherit from
        describe_taskset: Function that returns taskset metadata
        provide_task_instances: Function that yields TaskInfo instances for given seeds
        rollout: Function that executes a rollout request and returns response
        dataset_registry: Optional registry for task datasets
        rubrics: Optional rubrics for evaluation
        proxy: Optional proxy config for vendor endpoints
        routers: Additional FastAPI routers to include
        middleware: Additional middleware to apply
        app_state: Mutable state dict accessible to routes
        require_api_key: Whether to require API key authentication
        expose_debug_env: Whether to expose debug environment endpoint
        cors_origins: CORS allowed origins
        startup_hooks: Functions to call on app startup
        shutdown_hooks: Functions to call on app shutdown
    """

    app_id: str
    name: str
    description: str
    base_task_info: TaskInfo
    describe_taskset: TasksetDescriptor
    provide_task_instances: InstanceProvider
    rollout: RolloutExecutor
    dataset_registry: TaskDatasetRegistry | None = None
    rubrics: RubricBundle | None = field(default_factory=RubricBundle)
    proxy: ProxyConfig | None = None
    routers: Sequence[APIRouter] = field(default_factory=tuple)
    middleware: Sequence[Middleware] = field(default_factory=tuple)
    app_state: MutableMapping[str, Any] = field(default_factory=_default_app_state)
    require_api_key: bool = True
    expose_debug_env: bool = True
    cors_origins: Sequence[str] | None = None
    startup_hooks: Sequence[Callable[[], None | Awaitable[None]]] = field(default_factory=tuple)
    shutdown_hooks: Sequence[Callable[[], None | Awaitable[None]]] = field(default_factory=tuple)

    def clone(self) -> TaskAppConfig:
        """Return a shallow copy safe to mutate when wiring the app."""

        return TaskAppConfig(
            app_id=self.app_id,
            name=self.name,
            description=self.description,
            base_task_info=self.base_task_info,
            describe_taskset=self.describe_taskset,
            provide_task_instances=self.provide_task_instances,
            rollout=self.rollout,
            dataset_registry=self.dataset_registry,
            rubrics=self.rubrics or RubricBundle(),
            proxy=self.proxy,
            routers=tuple(self.routers),
            middleware=tuple(self.middleware),
            app_state=dict(self.app_state),
            require_api_key=self.require_api_key,
            expose_debug_env=self.expose_debug_env,
            cors_origins=tuple(self.cors_origins or ()),
            startup_hooks=tuple(self.startup_hooks),
            shutdown_hooks=tuple(self.shutdown_hooks),
        )


def _maybe_await(result: Any) -> Awaitable[Any]:
    if inspect.isawaitable(result):
        return asyncio.ensure_future(result)
    loop = asyncio.get_event_loop()
    future: asyncio.Future[Any] = loop.create_future()
    future.set_result(result)
    return future


def _ensure_task_info(obj: Any) -> TaskInfo:
    if isinstance(obj, TaskInfo):
        return obj
    if isinstance(obj, MutableMapping):
        return TaskInfo.model_validate(obj)
    raise TypeError(
        f"Task instance provider must yield TaskInfo-compatible objects (got {type(obj)!r})"
    )


def _normalise_seeds(values: Sequence[int]) -> list[int]:
    seeds: list[int] = []
    for value in values:
        try:
            seeds.append(int(value))
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(f"Seed values must be convertible to int (got {value!r})") from exc
    return seeds


def _build_proxy_routes(
    app: FastAPI, config: TaskAppConfig, auth_dependency: Callable[[Request], None]
) -> None:
    proxy = config.proxy
    if not proxy:
        return

    async def _call_vendor(
        url: str, payload: dict[str, Any], headers: dict[str, str]
    ) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=httpx.Timeout(600.0), follow_redirects=True) as client:
            response = await client.post(url, json=payload, headers=headers)
        data = (
            response.json()
            if response.headers.get("content-type", "").startswith("application/json")
            else {"raw": response.text}
        )
        if response.status_code >= 400:
            code = "vendor_error"
            if url.startswith("https://api.openai.com"):
                code = "openai_error"
            elif "groq" in url:
                code = "groq_error"
            raise http_exception(
                response.status_code,
                code,
                "Vendor proxy error",
                extra={"status": response.status_code, "body": data},
            )
        return data

    def _log_proxy(route: str, payload: dict[str, Any]) -> None:
        try:
            messages = payload.get("messages") if isinstance(payload, dict) else None
            msg_count = len(messages) if isinstance(messages, list) else 0
            tool_count = len(payload.get("tools") or []) if isinstance(payload, dict) else 0
            model = payload.get("model") if isinstance(payload, dict) else None
            print(
                f"[task:proxy:{route}] model={model} messages={msg_count} tools={tool_count}",
                flush=True,
            )
        except Exception:  # pragma: no cover - best effort logging
            pass

    system_hint = proxy.system_hint

    if proxy.enable_openai:

        @app.post("/proxy/v1/chat/completions", dependencies=[Depends(auth_dependency)])
        async def proxy_openai(body: dict[str, Any], request: Request) -> Any:  # type: ignore[no-redef]
            key = get_openai_key_or_503()
            model = body.get("model") if isinstance(body.get("model"), str) else None
            payload = prepare_for_openai(model, body)
            payload = inject_system_hint(payload, system_hint or "")
            _log_proxy("openai", payload)
            data = await _call_vendor(proxy.openai_url, payload, {"Authorization": f"Bearer {key}"})
            sanitized = synthesize_tool_call_if_missing(data)
            return to_jsonable(sanitized)

    if proxy.enable_groq:

        @app.post("/proxy/groq/v1/chat/completions", dependencies=[Depends(auth_dependency)])
        async def proxy_groq(body: dict[str, Any], request: Request) -> Any:  # type: ignore[no-redef]
            key = get_groq_key_or_503()
            model = body.get("model") if isinstance(body.get("model"), str) else None
            payload = prepare_for_groq(model, body)
            payload = inject_system_hint(payload, system_hint or "")
            _log_proxy("groq", payload)
            data = await _call_vendor(
                proxy.groq_url.rstrip("/"), payload, {"Authorization": f"Bearer {key}"}
            )
            sanitized = synthesize_tool_call_if_missing(data)
            return to_jsonable(sanitized)


def _auth_dependency_factory(config: TaskAppConfig) -> Callable[[Request], None]:
    def _dependency(request: Request) -> None:
        if not config.require_api_key:
            return
        require_api_key_dependency(request)

    return _dependency


def create_task_app(config: TaskAppConfig) -> FastAPI:
    """Create a FastAPI application from a TaskAppConfig.

    This function builds a complete FastAPI application that implements the Synth AI
    task app contract. The resulting app includes:
    - Health check endpoints (/health, /)
    - Task info endpoint (/task_info)
    - Rollout endpoint (/rollout)
    - Optional proxy endpoints for OpenAI/Groq
    - Optional dataset registry endpoints
    - CORS middleware (if configured)
    - Authentication middleware (if require_api_key=True)

    The app is ready to run with uvicorn or deploy to any ASGI-compatible server.

    Args:
        config: TaskAppConfig describing the task app

    Returns:
        FastAPI application instance ready to run

    Example:
        >>> from synth_ai.sdk.task.server import TaskAppConfig, create_task_app
        >>>
        >>> def build_config() -> TaskAppConfig:
        ...     return TaskAppConfig(
        ...         app_id="my_task",
        ...         name="My Task",
        ...         description="A task app",
        ...         base_task_info=TaskInfo(...),
        ...         describe_taskset=lambda: {"splits": ["train"]},
        ...         provide_task_instances=lambda seeds: [...],
        ...         rollout=lambda req, r: {...},
        ...     )
        >>>
        >>> app = create_task_app(build_config())
        >>> # Run with: uvicorn.run(app, host="0.0.0.0", port=8001)
    """
    ctx: dict[str, Any] = {"app_id": config.app_id, "name": config.name}
    log_info("create_task_app invoked", ctx=ctx)
    cfg = config.clone()
    cfg.rubrics = cfg.rubrics or RubricBundle()
    app = FastAPI(title=cfg.name, description=cfg.description)

    for key, value in cfg.app_state.items():
        setattr(app.state, key, value)

    if cfg.cors_origins is not None:
        app.add_middleware(
            CORSMiddleware,  # type: ignore[arg-type]
            allow_origins=list(cfg.cors_origins) or ["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Note: additional middleware from cfg.middleware is currently disabled to avoid typing ambiguity.
    # for middleware in cfg.middleware:
    #     try:
    #         opts = getattr(middleware, "options", {})
    #     except Exception:
    #         opts = {}
    #     app.add_middleware(middleware.cls, **(opts if isinstance(opts, dict) else {}))

    for router in cfg.routers:
        try:
            app.include_router(router)
        except Exception:
            try:
                inner = getattr(router, "router", None)
                if inner is not None:
                    app.include_router(inner)
            except Exception:
                raise

    auth_dependency = _auth_dependency_factory(cfg)

    def _call_hook(hook: Callable[..., Any]) -> Awaitable[Any]:
        try:
            params = inspect.signature(hook).parameters  # type: ignore[arg-type]
        except (TypeError, ValueError):
            params = {}
        if params:
            return _maybe_await(hook(app))  # type: ignore[misc]
        return _maybe_await(hook())

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        normalize_environment_api_key()
        normalize_vendor_keys()
        for hook in cfg.startup_hooks:
            await _call_hook(hook)
        try:
            yield
        finally:
            for hook in cfg.shutdown_hooks:
                await _call_hook(hook)

    app.router.lifespan_context = lifespan

    @app.get("/")
    async def root() -> Mapping[str, Any]:
        return to_jsonable({"status": "ok", "service": cfg.app_id})

    @app.head("/")
    async def root_head() -> Mapping[str, Any]:
        return to_jsonable({"status": "ok"})

    @app.get("/health", dependencies=[Depends(auth_dependency)])
    async def health(request: Request) -> Mapping[str, Any]:
        # If we got here, auth_dependency already verified the key exactly matches
        expected = normalize_environment_api_key()
        return to_jsonable(
            {
                "healthy": True,
                "auth": {
                    "required": True,
                    "expected_prefix": (expected[:6] + "...") if expected else "<unset>",
                },
            }
        )

    @app.post("/done", dependencies=[Depends(auth_dependency)])
    async def done() -> Mapping[str, Any]:
        # Coordination endpoint for tests and automation; indicates app is reachable
        return to_jsonable({"ok": True, "service": cfg.app_id})

    @app.get("/info", dependencies=[Depends(auth_dependency)])
    async def info() -> Mapping[str, Any]:
        dataset_meta = cfg.base_task_info.dataset
        rubrics: dict[str, Any] | None = None
        rubric_bundle = cfg.rubrics
        if rubric_bundle and (rubric_bundle.outcome or rubric_bundle.events):
            rubrics = {
                "outcome": rubric_bundle.outcome.model_dump() if rubric_bundle.outcome else None,
                "events": rubric_bundle.events.model_dump() if rubric_bundle.events else None,
            }
        payload = {
            "service": {
                "task": cfg.base_task_info.task,
                "version": cfg.base_task_info.task.version,
            },
            "dataset": dataset_meta,
            "rubrics": rubrics,
            "inference": cfg.base_task_info.inference,
            "limits": cfg.base_task_info.limits,
        }
        return to_jsonable(payload)

    @app.get("/task_info", dependencies=[Depends(auth_dependency)])
    async def task_info(
        seed: Sequence[int] | None = Query(default=None),
        seeds: Sequence[int] | None = Query(default=None),
    ) -> Any:
        all_seeds: list[int] = []
        if seed:
            all_seeds.extend(_normalise_seeds(seed))
        if seeds:
            all_seeds.extend(_normalise_seeds(seeds))

        if not all_seeds:
            descriptor_result = await _maybe_await(cfg.describe_taskset())
            return to_jsonable({"taskset": descriptor_result})

        instances = await _maybe_await(cfg.provide_task_instances(all_seeds))
        payload = [to_jsonable(_ensure_task_info(instance).model_dump()) for instance in instances]
        return payload[0] if len(payload) == 1 else payload

    @app.post("/rollout", dependencies=[Depends(auth_dependency)])
    async def rollout_endpoint(rollout_request: RolloutRequest, request: Request) -> Any:
        result = await _maybe_await(cfg.rollout(rollout_request, request))
        if isinstance(result, RolloutResponse):
            return to_jsonable(result.model_dump())
        if isinstance(result, Mapping):
            try:
                validated = RolloutResponse.model_validate(result)
            except Exception:
                return to_jsonable(result)
            return to_jsonable(validated.model_dump())
        raise TypeError("Rollout executor must return RolloutResponse or mapping")

    if cfg.expose_debug_env:

        @app.get("/debug/env", dependencies=[Depends(auth_dependency)])
        async def debug_env() -> Mapping[str, Any]:
            def _mask(value: str | None) -> str:
                if not value:
                    return ""
                return f"{value[:6]}…" if len(value) > 6 else value

            return to_jsonable(
                {
                    "has_ENVIRONMENT_API_KEY": bool(os.getenv("ENVIRONMENT_API_KEY")),
                    "OPENAI_API_KEY_prefix": _mask(os.getenv("OPENAI_API_KEY")),
                    "GROQ_API_KEY_prefix": _mask(os.getenv("GROQ_API_KEY")),
                }
            )

    _build_proxy_routes(app, cfg, auth_dependency)

    return app


def _load_env_files(env_files: Sequence[str]) -> list[str]:
    loaded: list[str] = []
    if not env_files:
        return loaded
    try:
        import dotenv
    except Exception:  # pragma: no cover - optional dep
        return loaded
    for path_str in env_files:
        path = Path(path_str)
        if not path.is_file():
            continue
        dotenv.load_dotenv(path, override=False)
        loaded.append(str(path))
    return loaded


def run_task_app(
    config_factory: Callable[[], TaskAppConfig],
    *,
    host: str = "0.0.0.0",
    port: int = 8001,
    reload: bool = False,
    env_files: Sequence[str] = (),
) -> None:
    """Run a task app locally with uvicorn.

    This is a convenience function that creates a task app from a config factory
    and runs it with uvicorn. It handles environment file loading and service
    registration for tunnel management.

    Args:
        config_factory: Function that returns a TaskAppConfig
        host: Host to bind to (default: "0.0.0.0")
        port: Port to listen on (default: 8001)
        reload: Enable auto-reload on file changes (default: False)
        env_files: Paths to .env files to load (default: empty)

    Example:
        >>> from synth_ai.sdk.task.server import run_task_app
        >>> from my_task_app import build_config
        >>>
        >>> run_task_app(
        ...     build_config,
        ...     host="127.0.0.1",
        ...     port=8114,
        ...     reload=True,
        ...     env_files=[".env.local"],
        ... )

    Raises:
        RuntimeError: If uvicorn is not installed
        TypeError: If config_factory doesn't return TaskAppConfig
    """
    ctx: dict[str, Any] = {"host": host, "port": port, "reload": reload}
    log_info("run_task_app invoked", ctx=ctx)

    loaded_files = _load_env_files(env_files)
    if loaded_files:
        print(f"[task:server] Loaded environment from: {', '.join(loaded_files)}", flush=True)

    config = config_factory()
    # Defensive: ensure the factory produced a valid TaskAppConfig to avoid
    # confusing attribute errors later in the boot sequence.
    if not isinstance(config, TaskAppConfig):  # type: ignore[arg-type]
        raise TypeError(
            f"Task app config_factory must return TaskAppConfig, got {type(config).__name__}"
        )
    app = create_task_app(config)

    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - uvicorn optional
        raise RuntimeError("uvicorn must be installed to run the task app locally") from exc

    print(f"[task:server] Starting '{config.app_id}' on {host}:{port}", flush=True)
    
    # Record local service before starting
    try:
        import os

        from synth_ai.cli.lib.tunnel_records import record_service
        local_url = f"http://{host if host not in ('0.0.0.0', '::') else '127.0.0.1'}:{port}"
        # Try to get current process PID
        pid: int | None = os.getpid()
        record_service(
            url=local_url,
            port=port,
            service_type="local",
            local_host=host if host not in ('0.0.0.0', '::') else '127.0.0.1',
            app_id=config.app_id,
            pid=pid,
        )
    except Exception:
        pass  # Fail silently - records are optional
    
    try:
        uvicorn.run(app, host=host, port=port, reload=reload)
    finally:
        # Clean up record when server exits
        try:
            from synth_ai.cli.lib.tunnel_records import remove_service_record
            remove_service_record(port)
        except Exception:
            pass

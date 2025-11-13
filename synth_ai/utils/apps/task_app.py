import contextlib
import inspect
import io
import os
import secrets
from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any, Callable, Set, cast

from fastapi.routing import APIRoute, APIRouter
from fastapi.testclient import TestClient
from starlette.middleware import Middleware
from starlette.types import ASGIApp
from synth_ai.task.contracts import TaskInfo
from synth_ai.task.datasets import TaskDatasetRegistry
from synth_ai.task.server import ProxyConfig, RubricBundle, TaskAppConfig
from synth_ai.utils.cli import ctx_print
from synth_ai.utils.paths import is_hidden_path, validate_file_type

from .common import (
    build_fastapi_route_index,
    extract_routes_from_app,
    get_asgi_app,
    load_module,
    validate_py_file_compiles,
)


def validate_required_routes_exist(app: ASGIApp) -> None:
    routes = set(extract_routes_from_app(app))
    required_endpoints: Set[str] = {
        '/',
        "/health",
        "/info",
        "/task_info",
        "/rollout"
    }
    missing = required_endpoints - routes
    if missing:
        raise ValueError(f"Missing required FastAPI endpoints: {', '.join(sorted(missing))}")
    return None


RouteContract = MutableMapping[str, Any]
ROUTE_CONTRACTS: dict[str, RouteContract] = {
    "/": {
        "method": "GET",
        "require_auth": False,
        "required_params": set(),
        "response_validator": lambda payload, _: _validate_root_payload(payload),
    },
    "/health": {
        "method": "GET",
        "require_auth": True,
        "required_params": {"request"},
        "response_validator": lambda payload, _: _validate_health_payload(payload),
    },
    "/info": {
        "method": "GET",
        "require_auth": True,
        "required_params": set(),
        "response_validator": lambda payload, _: _validate_info_payload(payload),
    },
    "/task_info": {
        "method": "GET",
        "require_auth": True,
        "required_params": {"seed", "seeds"},
        "response_validator": lambda payload, _: _validate_task_info_payload(payload),
    },
    "/rollout": {
        "method": "POST",
        "require_auth": True,
        "required_params": {"rollout_request"},
        "response_validator": None,
    },
}

def _ensure_mapping(payload: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must return a JSON object")
    return payload

def _validate_root_payload(payload: Any) -> None:
    data = _ensure_mapping(payload, "/")
    status = data.get("status")
    service = data.get("service")
    if not isinstance(status, str) or not status:
        raise ValueError("`/` must return a status string")
    if not isinstance(service, str) or not service:
        raise ValueError("`/` must return a service string")

def _validate_health_payload(payload: Any) -> None:
    data = _ensure_mapping(payload, "/health")
    healthy = data.get("healthy")
    auth = data.get("auth")
    if not isinstance(healthy, bool):
        raise ValueError("`/health` must return a boolean `healthy` field")
    if not isinstance(auth, Mapping):
        raise ValueError("`/health` must include an `auth` object")
    required_keys = {"required", "expected_prefix"}
    missing = required_keys - set(auth)
    if missing:
        raise ValueError(f"`/health` auth payload missing keys: {', '.join(sorted(missing))}")

def _validate_info_payload(payload: Any) -> None:
    data = _ensure_mapping(payload, "/info")
    service = data.get("service")
    dataset = data.get("dataset")
    if not isinstance(service, Mapping):
        raise ValueError("`/info` must include a service object")
    if "task" not in service:
        raise ValueError("`/info` service object must include a `task` field")
    if dataset is None:
        raise ValueError("`/info` must include dataset metadata")

def _validate_task_info_payload(payload: Any) -> None:
    data = _ensure_mapping(payload, "/task_info")
    if "taskset" not in data:
        raise ValueError("`/task_info` without seeds must include a `taskset` field")


def validate_route_contracts(app: ASGIApp) -> None:
    route_index = build_fastapi_route_index(app)

    def _get_route(path: str, method: str) -> APIRoute:
        candidates = route_index.get(path, [])
        if not candidates:
            raise ValueError(f"No route registered for {path}")
        method_upper = method.upper()
        for route in candidates:
            methods = {m.upper() for m in (route.methods or [])}
            if method_upper in methods:
                return route
        raise ValueError(f"Route {path} missing required method {method_upper}")

    def _ensure_params(route: APIRoute, expected: Set[str], path: str) -> None:
        if not expected:
            return
        signature = inspect.signature(route.endpoint)
        present = set(signature.parameters)
        missing = expected - present
        if missing:
            raise ValueError(
                f"{path} endpoint missing required parameters: {', '.join(sorted(missing))}"
            )

    for path, spec in ROUTE_CONTRACTS.items():
        route = _get_route(path, spec["method"])
        _ensure_params(route, set(spec.get("required_params", set())), path)


def test_route_contracts(app: ASGIApp) -> None:
    route_index = build_fastapi_route_index(app)

    def _get_route(path: str, method: str) -> APIRoute:
        candidates = route_index.get(path, [])
        if not candidates:
            raise ValueError(f"No route registered for {path}")
        method_upper = method.upper()
        for route in candidates:
            methods = {m.upper() for m in (route.methods or [])}
            if method_upper in methods:
                return route
        raise ValueError(f"Route {path} missing required method {method_upper}")

    for path, spec in ROUTE_CONTRACTS.items():
        _get_route(path, spec["method"])

    # Ensure ENVIRONMENT_API_KEY present for auth-required routes
    original_env_key = os.environ.get("ENVIRONMENT_API_KEY")
    auth_key = original_env_key or secrets.token_hex(16)
    if original_env_key is None:
        os.environ["ENVIRONMENT_API_KEY"] = auth_key
    auth_headers = {"X-API-Key": auth_key}

    try:
        with TestClient(cast(ASGIApp, app)) as client:
            for path, spec in ROUTE_CONTRACTS.items():
                headers = auth_headers if spec.get("require_auth") else {}
                params = None
                json_payload = None
                if path == "/task_info":
                    params = {"seed": 0}
                if path == "/rollout":
                    json_payload = {
                        "run_id": "validate",
                        "env": {"env_id": "dummy", "config": {}, "seed": 0},
                        "policy": {"policy_id": "dummy", "config": {}},
                        "ops": [],
                        "record": {
                            "trajectories": True,
                            "logprobs": False,
                            "value": False,
                            "return_trace": False,
                            "trace_format": "compact",
                        },
                        "on_done": "reset",
                        "safety": {"max_ops": 1, "max_time_s": 1.0},
                        "mode": "eval",
                    }
                validator = spec.get("response_validator")
                response = client.request(
                    spec["method"],
                    path,
                    headers=headers,
                    params=params,
                    json=json_payload,
                )
                if response.status_code >= 400:
                    raise ValueError(
                        f"{path} responded with HTTP {response.status_code} during validation"
                    )
                if validator is None:
                    continue
                try:
                    payload = response.json()
                except ValueError as exc:
                    raise ValueError(f"{path} did not return JSON during validation") from exc
                validator(payload, path)

            # Ensure auth dependency rejects missing key for protected endpoints
            protected_paths = [p for p, spec in ROUTE_CONTRACTS.items() if spec["require_auth"]]
            if protected_paths:
                resp = client.get(protected_paths[0])
                if resp.status_code == 200:
                    raise ValueError(
                        f"{protected_paths[0]} accepted requests without ENVIRONMENT_API_KEY"
                    )
    finally:
        if original_env_key is None:
            os.environ.pop("ENVIRONMENT_API_KEY", None)


def validate_config_structure(cfg: TaskAppConfig) -> TaskAppConfig:
    def callable_signature(name: str, fn: Callable[..., Any]) -> inspect.Signature:
        try:
            return inspect.signature(fn)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a callable with an inspectable signature") from exc
    
    def required_parameters(sig: inspect.Signature) -> list[inspect.Parameter]:
        positional_kinds = {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }
        return [
            param
            for param in sig.parameters.values()
            if param.kind in positional_kinds and param.default is inspect._empty
        ]
    
    for field_name in ("app_id", "name", "description"):
        value = getattr(cfg, field_name, None)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"TaskAppConfig.{field_name} must be a non-empty string")

    try:
        TaskInfo.model_validate(cfg.base_task_info)
    except Exception as exc:
        raise ValueError("TaskAppConfig.base_task_info must be a valid TaskInfo") from exc

    if not callable(cfg.describe_taskset):
        raise ValueError("TaskAppConfig.describe_taskset must be callable")
    describe_sig = callable_signature("describe_taskset", cfg.describe_taskset)
    describe_required = required_parameters(describe_sig)
    if describe_required:
        required_names = ", ".join(param.name for param in describe_required)
        raise ValueError(
            f"describe_taskset must not require positional arguments (found: {required_names})"
        )

    if not callable(cfg.provide_task_instances):
        raise ValueError("TaskAppConfig.provide_task_instances must be callable")
    provide_sig = callable_signature("provide_task_instances", cfg.provide_task_instances)
    provide_required = required_parameters(provide_sig)
    if not provide_required:
        raise ValueError("provide_task_instances must accept at least one positional argument")

    if not callable(cfg.rollout):
        raise ValueError("TaskAppConfig.rollout must be callable")
    rollout_sig = callable_signature("rollout", cfg.rollout)
    rollout_required = required_parameters(rollout_sig)
    if len(rollout_required) < 2:
        raise ValueError(
            "rollout must accept at least two positional parameters (rollout_request, request)"
        )

    if cfg.dataset_registry is not None and not isinstance(
        cfg.dataset_registry, TaskDatasetRegistry
    ):
        raise TypeError("dataset_registry must be a TaskDatasetRegistry or None")

    if cfg.rubrics is not None and not isinstance(cfg.rubrics, RubricBundle):
        raise TypeError("rubrics must be a RubricBundle or None")

    if cfg.proxy is not None and not isinstance(cfg.proxy, ProxyConfig):
        raise TypeError("proxy must be a ProxyConfig or None")

    if not isinstance(cfg.routers, Sequence):
        raise TypeError("routers must be a sequence of fastapi.APIRouter instances")
    for router in cfg.routers:
        if not isinstance(router, APIRouter):
            raise TypeError("routers must contain only fastapi.APIRouter instances")

    if not isinstance(cfg.middleware, Sequence):
        raise TypeError("middleware must be a sequence of starlette.middleware.Middleware")
    for middleware in cfg.middleware:
        if not isinstance(middleware, Middleware):
            raise TypeError("middleware entries must be starlette.middleware.Middleware instances")

    if not isinstance(cfg.app_state, MutableMapping):
        raise TypeError("app_state must be a mutable mapping")

    if not isinstance(cfg.require_api_key, bool):
        raise TypeError("require_api_key must be a boolean")
    if not isinstance(cfg.expose_debug_env, bool):
        raise TypeError("expose_debug_env must be a boolean")

    if cfg.cors_origins is not None:
        if not isinstance(cfg.cors_origins, Sequence):
            raise TypeError("cors_origins must be a sequence of strings")
        for origin in cfg.cors_origins:
            if not isinstance(origin, str):
                raise TypeError("cors_origins must contain only strings")

    for hook_name, hooks in (
        ("startup_hooks", cfg.startup_hooks),
        ("shutdown_hooks", cfg.shutdown_hooks),
    ):
        if not isinstance(hooks, Sequence):
            raise TypeError(f"{hook_name} must be a sequence of callables")
        for hook in hooks:
            if not callable(hook):
                raise TypeError(f"Each entry in {hook_name} must be callable")

    return cfg


def validate_task_app(
    path: Path,
    discovery: bool = False
) -> Path:

    def print_pass():
        ctx_print("Check passed", not discovery)

    ctx_print("\nChecking if .py file", not discovery)
    validate_file_type(path, ".py")
    print_pass()
    
    ctx_print("\nChecking if compiles", not discovery)
    validate_py_file_compiles(path)
    print_pass()
    
    ctx_print("\nChecking if loads to module", not discovery)
    with contextlib.redirect_stdout(io.StringIO()):
        module = load_module(path)
    print_pass()
    
    ctx_print("\nChecking if is ASGI app", not discovery)
    with contextlib.redirect_stdout(io.StringIO()):
        app = get_asgi_app(module)
    print_pass()

    ctx_print("\nChecking if config structure is valid", not discovery)
    with contextlib.redirect_stdout(io.StringIO()):
        config_factory = getattr(module, "build_config", None)
        if callable(config_factory):
            config = config_factory()
            if not isinstance(config, TaskAppConfig):
                raise TypeError("build_config must return a TaskAppConfig instance")
            validate_config_structure(config)
    print_pass()
    
    ctx_print("\nChecking if required routes exist", not discovery)
    validate_required_routes_exist(app)
    print_pass()
    
    ctx_print("\nChecking if required route contracts exist", not discovery)
    validate_route_contracts(app)
    print_pass()
    
    if discovery:
        return path
    
    ctx_print("Testing route contracts", not discovery)
    test_route_contracts(app)
    print_pass()
    print('\n')
    return path


def is_valid_task_app(
    path: Path,
    discovery: bool = False
) -> bool:
    try:
        validate_task_app(path, discovery)
    except Exception:
        return False
    return True


def find_task_apps_in_cwd() -> list[tuple[Path, str]]:
    from datetime import datetime

    cwd = Path.cwd().resolve()
    entries: list[tuple[Path, str, float]] = []
    for path in cwd.rglob("*.py"):
        if is_hidden_path(path, cwd):
            continue
        if not path.is_file():
            continue
        try:
            validate_task_app(path, True)
        except Exception:
            continue
        try:
            rel_path = path.relative_to(cwd)
        except ValueError:
            rel_path = path
        try:
            mtime = path.stat().st_mtime
            mtime_str = datetime.fromtimestamp(mtime).isoformat(sep=" ", timespec="seconds")
        except OSError:
            mtime = 0.0
            mtime_str = ""
        entries.append((rel_path, mtime_str, mtime))
    entries.sort(key=lambda item: item[2], reverse=True)
    return [(rel_path, mtime_str) for rel_path, mtime_str, _ in entries]

import contextlib
import inspect
import io
import os
import secrets
from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any, Callable, Set, cast
from urllib.parse import parse_qs, urlparse

from fastapi.routing import APIRoute, APIRouter
from fastapi.testclient import TestClient
from starlette.middleware import Middleware
from starlette.types import ASGIApp
from synth_ai.cli.lib.prompts import ctx_print
from synth_ai.core.apps.common import (
    build_fastapi_route_index,
    extract_routes_from_app,
    get_asgi_app,
    load_module,
    validate_py_file_compiles,
)
from synth_ai.core.paths import is_hidden_path, validate_file_type
from synth_ai.sdk.task.contracts import TaskInfo
from synth_ai.sdk.task.datasets import TaskDatasetRegistry
from synth_ai.sdk.task.server import ProxyConfig, RubricBundle, TaskAppConfig


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
        "response_validator": lambda payload, _: _validate_rollout_payload(payload),
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

def _validate_rollout_payload(payload: Any) -> None:
    """Validate that /rollout returns a proper RolloutResponse schema.

    This catches the common error: "Pattern validation failed: Failed to fetch
    baseline messages: No trajectories in response" that occurs when manual
    FastAPI implementations return simplified response formats instead of the
    complete RolloutResponse schema required by training.
    """
    data = _ensure_mapping(payload, "/rollout")

    # Check required top-level fields
    required_fields = ["run_id", "trajectories", "metrics"]
    for field in required_fields:
        if field not in data:
            raise ValueError(
                f"`/rollout` response missing required field '{field}'. "
                f"The response must include: run_id, trajectories, and metrics. "
                f"This error often occurs with manual FastAPI implementations. "
                f"Use create_task_app(build_config()) instead."
            )

    # Validate trajectories is a list
    trajectories = data.get("trajectories")
    if not isinstance(trajectories, list):
        raise ValueError(
            f"`/rollout` response field 'trajectories' must be a list, got {type(trajectories).__name__}. "
            f"Make sure your rollout executor returns a proper RolloutResponse with a list of RolloutTrajectory objects."
        )

    # Ensure trajectories list is not empty (training will fail if it's empty)
    if len(trajectories) == 0:
        raise ValueError(
            "`/rollout` response field 'trajectories' is an empty list. "
            "Training will fail with 'No trajectories in response'. "
            "Your rollout executor must return at least one trajectory with steps."
        )

    # Validate first trajectory has required fields
    first_traj = trajectories[0]
    if not isinstance(first_traj, Mapping):
        raise ValueError(
            f"`/rollout` trajectories must contain objects, got {type(first_traj).__name__}"
        )

    required_traj_fields = ["env_id", "policy_id", "steps", "length", "inference_url"]
    for field in required_traj_fields:
        if field not in first_traj:
            raise ValueError(
                f"`/rollout` trajectory missing required field '{field}'. "
                f"Each trajectory must include: env_id, policy_id, steps, length, and inference_url."
            )

    # STRICT: Validate inference_url contains correlation ID for trace correlation
    # This catches: "Rollout response is missing trace correlation IDs in inference_url entries"
    inference_url = first_traj.get("inference_url")
    if inference_url is None:
        raise ValueError(
            "`/rollout` trajectory.inference_url is None. "
            "Each trajectory must include a valid inference_url with ?cid=trace_xxxxx parameter for trace correlation. "
            "Example: 'http://example.com/v1/chat/completions?cid=trace_abc123'"
        )

    if not isinstance(inference_url, str):
        raise ValueError(
            f"`/rollout` trajectory.inference_url must be a string, got {type(inference_url).__name__}"
        )

    parsed_inference_url = urlparse(inference_url)
    if not parsed_inference_url.scheme or not parsed_inference_url.netloc:
        raise ValueError(
            "`/rollout` trajectory.inference_url must include a scheme and host. "
            "Example: 'http://example.com?cid=trace_abc123'"
        )

    if parsed_inference_url.path not in ("", "/"):
        raise ValueError(
            f"`/rollout` trajectory.inference_url must be a base URL only (scheme + host). "
            f"Found path: '{parsed_inference_url.path}'. "
            f"Remove the path component - the backend will append it automatically. "
            f"Expected format: 'http://example.com?cid=trace_xxx' (no '/v1' or '/v1/chat/completions')."
        )

    cid_values = parse_qs(parsed_inference_url.query).get("cid", [])
    if not cid_values or not cid_values[0]:
        raise ValueError(
            "`/rollout` trajectory.inference_url missing correlation ID parameter. "
            "URL must include ?cid=trace_xxxxx for trace correlation. "
            "Example: 'http://example.com?cid=trace_abc123' (note: no path, just base URL + query)."
        )

    # Validate steps is a list and not empty
    steps = first_traj.get("steps")
    if not isinstance(steps, list):
        raise ValueError(
            f"`/rollout` trajectory 'steps' must be a list, got {type(steps).__name__}"
        )

    # For prompt learning (MIPRO, etc), we need messages in step.info
    # This catches: "Could not extract messages from rollout response - ensure task app stores messages in step.info"
    if len(steps) > 0:
        first_step = steps[0]
        if not isinstance(first_step, Mapping):
            raise ValueError(
                f"`/rollout` steps must contain objects, got {type(first_step).__name__}"
            )

        # STRICT: Require step.info with messages for prompt learning compatibility
        step_info = first_step.get("info")
        if step_info is None:
            raise ValueError(
                "`/rollout` step.info is missing. "
                "For prompt learning (MIPRO), each step must include an 'info' field with 'messages'. "
                "Use create_task_app(build_config()) with a proper rollout executor."
            )

        if not isinstance(step_info, Mapping):
            raise ValueError(
                f"`/rollout` step.info must be an object, got {type(step_info).__name__}"
            )

        messages = step_info.get("messages")
        if messages is None:
            raise ValueError(
                "`/rollout` step.info['messages'] is missing. "
                "Prompt learning requires conversation history in step.info['messages']. "
                "The SDK's rollout executor handles this automatically."
            )

        if not isinstance(messages, list):
            raise ValueError(
                f"`/rollout` step.info['messages'] must be a list, got {type(messages).__name__}"
            )

        if len(messages) == 0:
            raise ValueError(
                "`/rollout` step.info['messages'] is an empty list. "
                "Prompt learning requires at least one message in the conversation history."
            )

    # Validate metrics structure
    metrics = data.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError(
            f"`/rollout` response field 'metrics' must be an object, got {type(metrics).__name__}"
        )

    # Metrics can be either:
    # 1. Full RolloutMetrics with episode_returns (list), mean_return, num_steps
    # 2. Simple dict with scalar values (episode_returns as float, mean_return, num_steps)
    required_metrics_fields = ["episode_returns", "mean_return", "num_steps"]
    for field in required_metrics_fields:
        if field not in metrics:
            raise ValueError(
                f"`/rollout` metrics missing required field '{field}'. "
                f"Metrics must include: episode_returns, mean_return, and num_steps."
            )

    # Validate types - episode_returns can be either a list or a scalar
    episode_returns = metrics.get("episode_returns")
    if not isinstance(episode_returns, list | int | float):
        raise ValueError(
            f"`/rollout` metrics.episode_returns must be a list or number, got {type(episode_returns).__name__}"
        )

    mean_return = metrics.get("mean_return")
    if not isinstance(mean_return, int | float):
        raise ValueError(
            f"`/rollout` metrics.mean_return must be a number, got {type(mean_return).__name__}"
        )

    num_steps = metrics.get("num_steps")
    if not isinstance(num_steps, int):
        raise ValueError(
            f"`/rollout` metrics.num_steps must be an integer, got {type(num_steps).__name__}"
        )


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

    def _base_url(url: str) -> str:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError("Inference URLs must include a scheme and host")
        return f"{parsed.scheme}://{parsed.netloc}"

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
    rollout_interceptor_url = "http://0.0.0.0:54444/v1/chat/completions"
    rollout_interceptor_base = _base_url(rollout_interceptor_url)
    rollout_trace_id = "trace_contract_validation"

    try:
        with TestClient(cast(ASGIApp, app)) as client:  # type: ignore[redundant-cast]
            for path, spec in ROUTE_CONTRACTS.items():
                headers = auth_headers if spec.get("require_auth") else {}
                params = None
                json_payload = None
                if path == "/task_info":
                    params = {"seed": 0}
                if path == "/rollout":
                    # Send the actual RolloutRequest format used by prompt learning backend
                    # This matches the payload from evaluation.py:_execute_rollout_request()
                    json_payload = {
                        "run_id": "validate",
                        "env": {
                            "env_name": "validation",
                            "config": {"index": 0},
                            "seed": 0,
                        },
                        "policy": {
                            "policy_name": "validation",
                            "config": {
                                "model": "gpt-4o-mini",
                                "provider": "openai",
                                "temperature": 0.7,
                                "inference_url": rollout_interceptor_url,
                                "trace_correlation_id": rollout_trace_id,
                            },
                            "assert_proxy": True,   # Backend always sets this for prompt learning
                            "proxy_only": True,     # Backend always sets this for prompt learning
                        },
                        "ops": ["agent", "env"],  # Critical: training sends this
                        "record": {"trajectories": True},
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

                if path == "/rollout":
                    trajectories = payload.get("trajectories") or []
                    first_traj = trajectories[0]
                    inference_url = first_traj.get("inference_url")
                    parsed_inference_url = urlparse(inference_url)
                    returned_base = f"{parsed_inference_url.scheme}://{parsed_inference_url.netloc}"
                    if returned_base != rollout_interceptor_base:
                        raise ValueError(
                            "`/rollout` trajectory.inference_url must use the interceptor base URL "
                            "provided in policy.config.inference_url."
                        )

                    cid_values = parse_qs(parsed_inference_url.query).get("cid", [])
                    if rollout_trace_id not in cid_values:
                        raise ValueError(
                            "`/rollout` trajectory.inference_url must include the trace correlation "
                            "ID from policy.config.trace_correlation_id."
                        )

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

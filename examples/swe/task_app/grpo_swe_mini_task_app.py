"""Compatibility wrapper for the mini-SWE task app."""

from __future__ import annotations

import argparse
from pathlib import Path

from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.requests import Request
from synth_ai.task.apps import ModalDeploymentConfig, registry
from synth_ai.task.auth import is_api_key_header_authorized, normalize_environment_api_key
from synth_ai.task.server import TaskAppConfig, create_task_app, run_task_app

from .grpo_swe_mini import build_config

APP_ID = "swe-mini"


def _build_base_config() -> TaskAppConfig:
    return build_config()


try:
    _REGISTERED_ENTRY = registry.get(APP_ID)
except Exception:  # pragma: no cover - registry unavailable in some contexts
    MODAL_DEPLOYMENT: ModalDeploymentConfig | None = None
    ENV_FILES: tuple[str, ...] = ()
else:
    MODAL_DEPLOYMENT = _REGISTERED_ENTRY.modal
    ENV_FILES = tuple(_REGISTERED_ENTRY.env_files)


def build_task_app_config() -> TaskAppConfig:
    base = _build_base_config()
    return base.clone()


def fastapi_app():
    app = create_task_app(build_task_app_config())

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

    @app.exception_handler(RequestValidationError)
    async def _on_validation_error(request: Request, exc: RequestValidationError):
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
    parser = argparse.ArgumentParser(description="Run the mini-SWE task app locally")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8020)
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn autoreload")
    parser.add_argument(
        "--env-file",
        action="append",
        default=[],
        help="Additional .env files to load before startup",
    )
    args = parser.parse_args()

    default_env = Path(__file__).resolve().parents[4] / "backend" / ".env.dev"
    env_files = [str(default_env)] if default_env.exists() else []
    env_files.extend(args.env_file or [])

    run_task_app(
        build_task_app_config,
        host=args.host,
        port=args.port,
        reload=args.reload,
        env_files=env_files,
    )


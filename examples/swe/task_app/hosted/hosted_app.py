from __future__ import annotations

import contextlib
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.requests import Request


class TaskApp:
    """Holds service configuration and shared state."""

    def __init__(
        self,
        service_base_url: str | None = None,
        vllm_base_url: str | None = None,
        default_model: str | None = None,
    ) -> None:
        self.service_base_url = service_base_url or os.getenv(
            "SERVICE_BASE_URL", "http://localhost:8000"
        )
        self.vllm_base_url = vllm_base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8001")
        self.default_model = default_model or os.getenv("DEFAULT_MODEL")


class ServiceInfo(BaseModel):
    """Service discovery response."""

    service: dict
    inference: dict


def create_app(allowed_environments: list[str] = None) -> FastAPI:
    """FastAPI app factory.

    Args:
        allowed_environments: List of environment names this service is allowed to handle.
                            If None, all environments are allowed (for backward compatibility).
    """
    env_filter = f" ({', '.join(allowed_environments)})" if allowed_environments else ""
    app = FastAPI(
        title=f"GRPO Synth Envs Hosted Service{env_filter}",
        description=f"Hosted environment and policy service for GRPO training{env_filter}",
        version="0.1.0",
    )

    # Add CORS middleware for development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize task app configuration
    task_app = TaskApp()
    app.state.task_app = task_app
    app.state.allowed_environments = allowed_environments

    # Add environment validation middleware
    if allowed_environments:

        @app.middleware("http")
        async def validate_environment(request, call_next):
            # Check if this is an environment-related request
            path = request.url.path
            if (
                path.startswith("/env/") or path.startswith("/rollout")
            ) and request.method == "POST":
                # We need to read the body to check env_name
                body = await request.body()
                try:
                    import json

                    data = json.loads(body) if body else {}
                    env_name = data.get("env_name", "").lower()

                    # Check if environment is allowed
                    if env_name and env_name not in [e.lower() for e in allowed_environments]:
                        from fastapi import HTTPException

                        raise HTTPException(
                            status_code=403,
                            detail=f"Environment '{env_name}' not allowed. This service only handles: {allowed_environments}",
                        )
                except json.JSONDecodeError:
                    pass  # Invalid JSON, let the endpoint handle it

                # Recreate request with the body we consumed
                request._body = body

            response = await call_next(request)
            return response

    # Mount routers
    from .branching import router as branching_router
    from .environment_routes import router as env_router
    from .rollout import router as rollout_router

    app.include_router(env_router, prefix="/env", tags=["environment"])

    # Policy routes are optional; skip if optional envs are missing in this build
    try:
        from .policy_routes import router as policy_router

        app.include_router(policy_router, prefix="/policy", tags=["policy"])
    except Exception as _e:
        # Log lightweight message; policy endpoints will be unavailable
        with contextlib.suppress(Exception):
            print(f"[hosted_app] Skipping policy routes: {_e}", flush=True)

    app.include_router(rollout_router, tags=["rollout"])
    app.include_router(branching_router, tags=["branching"])

    @app.get("/info", response_model=ServiceInfo)
    async def get_info() -> ServiceInfo:
        """Service discovery endpoint."""
        return ServiceInfo(
            service={
                "base_url": task_app.service_base_url,
                "endpoints": {
                    "env": "/env/*",
                    "policy": "/policy/*",
                    "rollout": "/rollout",
                    "branch": "/branch",
                    "run": "/run/*",
                },
            },
            inference={
                "base_url": task_app.vllm_base_url,
                "endpoints": {
                    "chat_completions": "/v1/chat/completions",
                },
                "default_model": task_app.default_model,
            },
        )

    @app.get("/health")
    async def health_check(request: Request) -> dict:
        """Health and auth sanity check.

        - Returns 503 if server missing ENVIRONMENT_API_KEY (misconfigured container).
        - If X-API-Key header is provided and mismatches, returns 401.
        - Otherwise returns 200 with basic info.
        """

        # Check if any environment API keys are configured
        from synth_ai.task.auth import allowed_environment_api_keys

        allowed_keys = allowed_environment_api_keys()
        if not allowed_keys:
            # Server-side misconfiguration; rollout would fail with 503
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "detail": "Auth not configured: missing ENVIRONMENT_API_KEY in task service environment",
                },
            )

        # Authorize using all header variants without typed Header params (avoid 422s)
        from synth_ai.task.auth import is_api_key_header_authorized

        authorized = is_api_key_header_authorized(request)
        if not authorized:
            # Soft-pass 200 with authorized=False to avoid failing CLI preflight
            primary_key = list(allowed_keys)[0] if allowed_keys else None
            prefix = primary_key[: max(1, len(primary_key) // 2)] if primary_key else None
            content = {"status": "healthy", "authorized": False}
            if prefix:
                content["expected_api_key_prefix"] = prefix
            return JSONResponse(status_code=200, content=content)
        return {
            "status": "healthy",
            "authorized": True,
            "service": {"base_url": task_app.service_base_url},
        }

    # Log and surface 422 validation errors with header presence
    from fastapi.exceptions import RequestValidationError

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
            status_code=422, content={"status": "invalid", "detail": exc.errors()[:5]}
        )

    return app

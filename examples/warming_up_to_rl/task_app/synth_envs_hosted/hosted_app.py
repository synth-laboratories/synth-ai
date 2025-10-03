from __future__ import annotations

import os
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel


class TaskApp:
    """Holds service configuration and shared state."""

    def __init__(
        self,
        service_base_url: Optional[str] = None,
        vllm_base_url: Optional[str] = None,
        default_model: Optional[str] = None,
    ) -> None:
        self.service_base_url = service_base_url or os.getenv(
            "SERVICE_BASE_URL", "http://localhost:8000"
        )
        self.vllm_base_url = vllm_base_url or os.getenv(
            "VLLM_BASE_URL", "http://localhost:8001"
        )
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
            if request.url.path.startswith("/env/") or request.url.path.startswith(
                "/rollout"
            ):
                # Extract environment name from request body for POST requests
                if request.method == "POST":
                    # We need to read the body to check env_name
                    body = await request.body()
                    try:
                        import json

                        data = json.loads(body) if body else {}
                        env_name = data.get("env_name", "").lower()

                        # Check if environment is allowed
                        if env_name and env_name not in [
                            e.lower() for e in allowed_environments
                        ]:
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
    from .environment_routes import router as env_router
    from .policy_routes import router as policy_router
    from .rollout import router as rollout_router
    from .branching import router as branching_router

    app.include_router(env_router, prefix="/env", tags=["environment"])
    app.include_router(policy_router, prefix="/policy", tags=["policy"])
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
        import os as _os

        env_key = _os.getenv("ENVIRONMENT_API_KEY") or _os.getenv(
            "dev_environment_api_key"
        )
        if not env_key:
            # Server-side misconfiguration; rollout would fail with 503
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "detail": "Auth not configured: missing ENVIRONMENT_API_KEY in task service environment",
                },
            )
        header_key = request.headers.get("x-api-key")
        keys_header = request.headers.get("x-api-keys")
        # Accept either exact match on single key, or any match from a CSV list in X-API-Keys
        keys_to_check = []
        if isinstance(keys_header, str) and keys_header.strip():
            keys_to_check = [k.strip() for k in keys_header.split(",") if k.strip()]
        if header_key:
            keys_to_check.insert(0, header_key)
        if keys_to_check and env_key not in keys_to_check:
            def _mask(v: str) -> dict:
                return {
                    "prefix": (v[:6] + "…") if len(v) >= 6 else v,
                    "suffix": ("…" if len(v) > 4 else "") + v[-4:],
                    "len": len(v),
                }
            got = {"first": _mask(keys_to_check[0]), "others": max(0, len(keys_to_check) - 1)}
            expected = {"prefix": (env_key[:7] + "…") if len(env_key) >= 7 else env_key, "len": len(env_key)}
            detail = {
                "status": "unauthorized",
                "detail": "Invalid API key(s) for health check",
                "got": got,
                "expected": expected,
            }
            return JSONResponse(status_code=401, content=detail)
        return {
            "status": "healthy",
            "service": {
                "base_url": task_app.service_base_url,
            },
        }

    return app

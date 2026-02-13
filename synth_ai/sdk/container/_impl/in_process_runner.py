"""SDK helper for running prompt-learning jobs against a tunneled container.

This module keeps everything in-process:
1) Spins up a FastAPI container via InProcessContainer
2) Opens a tunnel (Cloudflare by default, or uses preconfigured URL)
3) Applies dot-notation overrides (container_url, budgets, seeds, models)
4) Submits jobs to the remote backend using SDK clients
5) Optionally polls until completion and returns a structured result

Note: RL job support has been moved to the research repo.

Tunnel Modes:
- "quick" (default): Creates Cloudflare quick tunnel - works for local development
- "named": Uses Cloudflare managed tunnel (requires setup)
- "local": No tunnel, uses localhost URL directly
- "preconfigured": Uses externally-provided URL - for container environments
  (ngrok, etc.) where Cloudflare tunnels don't work

Environment Variables:
- SYNTH_CONTAINER_URL: If set, auto-enables preconfigured mode with this URL
- SYNTH_TUNNEL_MODE: Override tunnel mode (e.g., "preconfigured", "local")
"""

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Mapping, MutableMapping
from urllib.parse import urlparse

from synth_ai.core.tunnels import TunnelBackend
from synth_ai.core.utils.dict import deep_update as _deep_update
from synth_ai.core.utils.urls import BACKEND_URL_BASE
from synth_ai.sdk.container._impl.in_process import InProcessContainer
from synth_ai.sdk.optimization.internal.container_api import ContainerHealth, check_container_health
from synth_ai.sdk.optimization.internal.prompt_learning import PromptLearningJob
from synth_ai.sdk.optimization.internal.utils import ensure_api_base

BackendMode = Literal["prompt_learning"]


@dataclass
class InProcessJobResult:
    """Result of an in-process job submission."""

    job_id: str
    status: Dict[str, Any]
    container_url: str
    backend_url: str
    container_health: ContainerHealth | None = None


def _normalize_base_url(url: str) -> str:
    """Strip trailing slashes and /api suffix for consistent handling."""
    base = url.strip().rstrip("/")
    if base.endswith("/api"):
        base = base[: -len("/api")]
    return base


def resolve_backend_api_base(override: str | None = None) -> str:
    """Resolve backend base URL using the documented priority order.

    Priority:
    1. Explicit override argument
    2. TARGET_BACKEND_BASE_URL
    3. BACKEND_OVERRIDE
    4. SYNTH_BACKEND_URL
    5. BACKEND_BASE_URL
    6. NEXT_PUBLIC_API_URL
    7. Fallback to BACKEND_URL_BASE
    """

    env_order = [
        "TARGET_BACKEND_BASE_URL",
        "BACKEND_OVERRIDE",
        "SYNTH_BACKEND_URL",
        "BACKEND_BASE_URL",
        "NEXT_PUBLIC_API_URL",
    ]

    if override and override.strip():
        candidate = override.strip()
    else:
        candidate = ""
        for key in env_order:
            value = os.environ.get(key, "").strip()
            if value:
                candidate = value
                break
        if not candidate:
            candidate = BACKEND_URL_BASE

    normalized = _normalize_base_url(candidate)
    return ensure_api_base(normalized)


def _is_local_backend_api_base(url: str) -> bool:
    try:
        host = (urlparse(url).hostname or "").strip().lower()
    except Exception:
        return False
    return host in {"localhost", "127.0.0.1", "host.docker.internal"}


def _require_env(key: str, *, friendly_name: str | None = None) -> str:
    value = os.environ.get(key, "").strip()
    if not value:
        label = friendly_name or key
        raise ValueError(
            f"{label} is required. Set {key} in your environment or pass it explicitly."
        )
    return value


def merge_dot_overrides(
    base: Mapping[str, Any] | None, extra: Mapping[str, Any] | None
) -> Dict[str, Any]:
    """Merge two override dictionaries with dot-notation support."""
    merged: MutableMapping[str, Any] = {}
    if base:
        _deep_update(merged, dict(base))
    if extra:
        _deep_update(merged, dict(extra))
    return dict(merged)


async def run_in_process_job(
    *,
    job_type: BackendMode,
    config_path: str | Path,
    backend_url: str | None = None,
    api_key: str | None = None,
    container_api_key: str | None = None,
    allow_experimental: bool | None = None,
    overrides: Mapping[str, Any] | None = None,
    poll: bool = True,
    poll_interval: float = 5.0,
    timeout: float = 3600.0,
    on_status: Callable[[Dict[str, Any]], None] | None = None,
    # InProcessContainer args (exactly one of app/config/config_factory/container_path required)
    app: Any | None = None,
    config: Any | None = None,
    config_factory: Callable[[], Any] | None = None,
    container_path: str | Path | None = None,
    tunnel_mode: str = "synthtunnel",
    tunnel_backend: TunnelBackend | str | None = None,
    preconfigured_url: str | None = None,
    preconfigured_auth_header: str | None = None,
    preconfigured_auth_token: str | None = None,
    skip_tunnel_verification: bool = True,  # Default True - verification is unreliable
    force_new_tunnel: bool = True,  # Default True - always get fresh tunnel
    host: str = "127.0.0.1",
    port: int = 8114,
    auto_find_port: bool = True,
    health_check_timeout: float = 30.0,
) -> InProcessJobResult:
    """Run a prompt-learning or RL job with a tunneled container."""
    """Run a prompt-learning or RL job with a tunneled container.
    
    Args:
        job_type: Type of job - "prompt_learning" or "rl"
        config_path: Path to the TOML config file
        backend_url: Optional backend URL override
        api_key: Synth API key for backend auth
        container_api_key: API key for container auth
        allow_experimental: Allow experimental features
        overrides: Config overrides (dot-notation supported)
        poll: Whether to poll for completion
        poll_interval: Polling interval in seconds
        timeout: Maximum time to wait for job completion
        on_status: Callback for status updates
        app: FastAPI app instance
        config: ContainerConfig object
        config_factory: Callable that returns ContainerConfig
        container_path: Path to container .py file
        tunnel_mode: Tunnel mode - "synthtunnel", "quick", "named", "local", or "preconfigured"
        tunnel_backend: Explicit tunnel backend (overrides tunnel_mode when set)
        preconfigured_url: External tunnel URL when tunnel_mode="preconfigured"
        preconfigured_auth_header: Auth header name for preconfigured URL
        preconfigured_auth_token: Auth token for preconfigured URL
        skip_tunnel_verification: Skip HTTP verification of tunnel
        host: Local host to bind to
        port: Local port to bind to
        auto_find_port: Auto-find available port if busy
        health_check_timeout: Max time to wait for health check
        
    Returns:
        InProcessJobResult with job_id, status, and URLs
    """
    backend_api_base = resolve_backend_api_base(backend_url)

    # Set SYNTH_BACKEND_URL so that tunnel operations (like rotate_tunnel) use the correct backend
    os.environ["SYNTH_BACKEND_URL"] = backend_api_base

    # Local prompt-learning runs don't need a tunnel. Default to direct localhost unless:
    # - the caller explicitly chose a tunnel mode, or
    # - an explicit tunnel backend is set, or
    # - SYNTH_TUNNEL_MODE is set (CLI override).
    if (
        tunnel_backend is None
        and tunnel_mode == "synthtunnel"
        and not (os.environ.get("SYNTH_TUNNEL_MODE") or "").strip()
        and _is_local_backend_api_base(backend_api_base)
    ):
        tunnel_mode = "local"

    resolved_api_key = api_key or _require_env("SYNTH_API_KEY", friendly_name="Backend API key")
    resolved_container_key = container_api_key or _require_env(
        "ENVIRONMENT_API_KEY", friendly_name="Container API key"
    )

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Launch the container with tunnel (Cloudflare by default, or preconfigured URL)
    async with InProcessContainer(
        app=app,
        config=config,
        config_factory=config_factory,
        container_path=container_path,
        host=host,
        port=port,
        tunnel_mode=tunnel_mode,
        tunnel_backend=tunnel_backend,
        preconfigured_url=preconfigured_url,
        preconfigured_auth_header=preconfigured_auth_header,
        preconfigured_auth_token=preconfigured_auth_token,
        skip_tunnel_verification=skip_tunnel_verification,
        force_new_tunnel=force_new_tunnel,
        api_key=resolved_container_key,
        auto_find_port=auto_find_port,
        health_check_timeout=health_check_timeout,
    ) as container:
        task_url = container.url or f"http://{host}:{container.port}"

        # Check if backend verified DNS propagation (so we can skip local health checks)
        dns_verified_by_backend = getattr(container, "_dns_verified_by_backend", False)

        # Skip health check when:
        # 1. tunnel verification is skipped (DNS may not have propagated locally)
        # 2. OR backend verified DNS propagation (safe to skip redundant local check)
        should_skip_health_check = skip_tunnel_verification or dns_verified_by_backend
        worker_token = getattr(container, "container_worker_token", None)
        if should_skip_health_check:
            reason = (
                "tunnel verification disabled"
                if skip_tunnel_verification
                else "backend verified DNS"
            )
            health = ContainerHealth(
                ok=True,
                health_status=200,
                task_info_status=200,
                detail=f"Skipped ({reason})",
            )
        else:
            health = check_container_health(
                task_url,
                resolved_container_key,
                worker_token=worker_token,
            )
            if not health.ok:
                raise RuntimeError(f"Container health check failed for {task_url}: {health.detail}")

        # Common overrides: task URL + API key injected in both dot and flat forms
        task_overrides = {
            "task_url": task_url,
            "prompt_learning.container_url": task_url,
        }
        merged_overrides = merge_dot_overrides(overrides, task_overrides)

        if job_type == "prompt_learning":
            job = PromptLearningJob.from_config(
                config_path=config_path,
                backend_url=backend_api_base,
                api_key=resolved_api_key,
                container_api_key=resolved_container_key,
                container_worker_token=worker_token,
                allow_experimental=allow_experimental,
                overrides=merged_overrides,
            )
        else:
            raise ValueError(
                f"Unknown job_type: {job_type}. Note: RL support has been moved to the research repo."
            )

        # Skip job's health check if we already skipped above (DNS verified by backend or tunnel verification disabled)
        if should_skip_health_check:
            job._skip_health_check = True

        job_id = job.submit()
        if not poll:
            status = {"status": "submitted", "job_id": job_id}
        else:
            status = job.poll_until_complete(
                timeout=timeout,
                interval=poll_interval,
                on_status=on_status,
            )

    return InProcessJobResult(
        job_id=job_id,
        status=status,
        container_url=task_url,
        backend_url=backend_api_base,
        container_health=health,
    )


def run_in_process_job_sync(**kwargs: Any) -> InProcessJobResult:
    """Synchronous wrapper for run_in_process_job()."""

    return asyncio.run(run_in_process_job(**kwargs))


__all__ = [
    "InProcessJobResult",
    "merge_dot_overrides",
    "resolve_backend_api_base",
    "run_in_process_job",
    "run_in_process_job_sync",
]

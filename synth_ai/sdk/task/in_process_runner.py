"""SDK helper for running prompt-learning and RL jobs against a tunneled LocalAPI.

This module keeps everything in-process:
1) Spins up a FastAPI LocalAPI via InProcessTaskApp
2) Opens a tunnel (Cloudflare by default, or uses preconfigured URL)
3) Applies dot-notation overrides (localapi_url, budgets, seeds, models)
4) Submits jobs to the remote backend using SDK clients
5) Optionally polls until completion and returns a structured result

Tunnel Modes:
- "quick" (default): Creates Cloudflare quick tunnel - works for local development
- "named": Uses Cloudflare managed tunnel (requires setup)
- "local": No tunnel, uses localhost URL directly
- "preconfigured": Uses externally-provided URL - for container environments
  (ngrok, etc.) where Cloudflare tunnels don't work

Environment Variables:
- SYNTH_LOCALAPI_URL: If set, auto-enables preconfigured mode with this URL
- SYNTH_TUNNEL_MODE: Override tunnel mode (e.g., "preconfigured", "local")
"""

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Mapping, MutableMapping

from synth_ai.core.dict_utils import deep_update as _deep_update
from synth_ai.core.telemetry import log_info
from synth_ai.core.urls import synth_api_url
from synth_ai.sdk.api.train.local_api import LocalAPIHealth, check_local_api_health
from synth_ai.sdk.api.train.prompt_learning import PromptLearningJob
from synth_ai.sdk.api.train.rl import RLJob
from synth_ai.sdk.task.in_process import InProcessTaskApp

BackendMode = Literal["prompt_learning", "rl"]


@dataclass
class InProcessJobResult:
    """Result of an in-process job submission."""

    job_id: str
    status: Dict[str, Any]
    localapi_url: str
    synth_base_url: str | None
    task_app_health: LocalAPIHealth | None = None


def resolve_backend_api_base(synth_base_url: str | None = None) -> str:
    """Resolve backend base URL using the documented priority order.

    Priority:
    1. Explicit override argument
    2. SYNTH_BACKEND_URL
    3. Fallback to default backend base
    """
    return synth_api_url("", synth_base_url)


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
    synth_user_key: str | None = None,
    localapi_key: str | None = None,
    allow_experimental: bool | None = None,
    overrides: Mapping[str, Any] | None = None,
    poll: bool = True,
    poll_interval: float = 5.0,
    timeout: float = 3600.0,
    on_status: Callable[[Dict[str, Any]], None] | None = None,
    # InProcessTaskApp args (exactly one of app/config/config_factory/task_app_path required)
    app: Any | None = None,
    config: Any | None = None,
    config_factory: Callable[[], Any] | None = None,
    task_app_path: str | Path | None = None,
    tunnel_mode: str = "quick",
    preconfigured_url: str | None = None,
    preconfigured_auth_header: str | None = None,
    preconfigured_auth_token: str | None = None,
    skip_tunnel_verification: bool = True,  # Default True - verification is unreliable
    force_new_tunnel: bool = True,  # Default True - always get fresh tunnel
    host: str = "127.0.0.1",
    port: int = 8114,
    auto_find_port: bool = True,
    health_check_timeout: float = 30.0,
    synth_base_url: str | None = None,
) -> InProcessJobResult:
    """Run a prompt-learning or RL job with a tunneled LocalAPI."""
    ctx: Dict[str, Any] = {
        "job_type": job_type,
        "config_path": str(config_path),
        "poll": poll,
        "tunnel_mode": tunnel_mode,
    }
    log_info("run_in_process_job invoked", ctx=ctx)
    """Run a prompt-learning or RL job with a tunneled LocalAPI.
    
    Args:
        job_type: Type of job - "prompt_learning" or "rl"
        config_path: Path to the TOML config file
        synth_base_url: Optional Synth base URL override
        synth_user_key: Synth API key for backend auth
        localapi_key: API key for LocalAPI auth
        allow_experimental: Allow experimental features
        overrides: Config overrides (dot-notation supported)
        poll: Whether to poll for completion
        poll_interval: Polling interval in seconds
        timeout: Maximum time to wait for job completion
        on_status: Callback for status updates
        app: FastAPI app instance
        config: TaskAppConfig object
        config_factory: Callable that returns TaskAppConfig
        task_app_path: Path to LocalAPI .py file
        tunnel_mode: Tunnel mode - "quick", "named", "local", or "preconfigured"
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
    resolved_synth_base_url = synth_base_url

    # Set SYNTH_BACKEND_URL so that tunnel operations (like rotate_tunnel) use the correct backend
    if resolved_synth_base_url:
        os.environ["SYNTH_BACKEND_URL"] = resolved_synth_base_url

    resolved_synth_user_key = synth_user_key or _require_env(
        "SYNTH_API_KEY", friendly_name="Backend API key"
    )
    resolved_localapi_key = localapi_key or _require_env(
        "ENVIRONMENT_API_KEY", friendly_name="LocalAPI API key"
    )

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Launch the LocalAPI with tunnel (Cloudflare by default, or preconfigured URL)
    async with InProcessTaskApp(
        app=app,
        config=config,
        config_factory=config_factory,
        task_app_path=task_app_path,
        host=host,
        port=port,
        tunnel_mode=tunnel_mode,
        preconfigured_url=preconfigured_url,
        preconfigured_auth_header=preconfigured_auth_header,
        preconfigured_auth_token=preconfigured_auth_token,
        skip_tunnel_verification=skip_tunnel_verification,
        force_new_tunnel=force_new_tunnel,
        localapi_key=resolved_localapi_key,
        auto_find_port=auto_find_port,
        health_check_timeout=health_check_timeout,
    ) as task_app:
        localapi_url = task_app.url or f"http://{host}:{task_app.port}"

        # Check if backend verified DNS propagation (so we can skip local health checks)
        dns_verified_by_backend = getattr(task_app, "_dns_verified_by_backend", False)

        # Skip health check when:
        # 1. tunnel verification is skipped (DNS may not have propagated locally)
        # 2. OR backend verified DNS propagation (safe to skip redundant local check)
        should_skip_health_check = skip_tunnel_verification or dns_verified_by_backend
        if should_skip_health_check:
            reason = (
                "tunnel verification disabled"
                if skip_tunnel_verification
                else "backend verified DNS"
            )
            health = LocalAPIHealth(
                ok=True,
                health_status=200,
                task_info_status=200,
                detail=f"Skipped ({reason})",
            )
        else:
            health = check_local_api_health(localapi_url, resolved_localapi_key)
            if not health.ok:
                raise RuntimeError(
                    f"LocalAPI health check failed for {localapi_url}: {health.detail}"
                )

        # Common overrides: LocalAPI URL + API key injected in both dot and flat forms
        localapi_overrides = {
            "localapi_url": localapi_url,
            "localapi_key": resolved_localapi_key,
            "prompt_learning.localapi_url": localapi_url,
            "prompt_learning.localapi_key": resolved_localapi_key,
        }
        merged_overrides = merge_dot_overrides(overrides, localapi_overrides)

        if job_type == "prompt_learning":
            job = PromptLearningJob.from_config(
                config_path=config_path,
                synth_base_url=resolved_synth_base_url,
                synth_user_key=resolved_synth_user_key,
                localapi_key=resolved_localapi_key,
                allow_experimental=allow_experimental,
                overrides=merged_overrides,
            )
        elif job_type == "rl":
            job = RLJob.from_config(
                config_path=config_path,
                synth_base_url=resolved_synth_base_url,
                synth_user_key=resolved_synth_user_key,
                localapi_url=localapi_url,
                localapi_key=resolved_localapi_key,
                allow_experimental=allow_experimental,
                overrides=merged_overrides,
            )
        else:
            raise ValueError(f"Unknown job_type: {job_type}")

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
        localapi_url=localapi_url,
        synth_base_url=resolved_synth_base_url,
        task_app_health=health,
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

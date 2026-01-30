"""LocalAPI deploy helpers (cloud via Harbor)."""

from __future__ import annotations

import os
from dataclasses import dataclass

from synth_ai.sdk.harbor import HarborBuildSpec, HarborLimits, upload_harbor_deployment


@dataclass(slots=True)
class LocalAPIDeployResult:
    """Result of a LocalAPI deployment."""

    deployment_id: str
    deployment_name: str | None
    task_app_url: str
    status: str
    build_id: str | None = None
    snapshot_id: str | None = None


def deploy_localapi(
    *,
    name: str,
    dockerfile_path: str,
    context_dir: str,
    entrypoint: str,
    entrypoint_mode: str = "stdio",
    description: str | None = None,
    env_vars: dict[str, str] | None = None,
    limits: HarborLimits | dict[str, int] | None = None,
    backend_url: str | None = None,
    api_key: str | None = None,
    wait_for_ready: bool = False,
    build_timeout_s: float = 600.0,
) -> LocalAPIDeployResult:
    """Deploy a LocalAPI via Harbor and return the task_app_url.

    Args:
        name: Deployment name (org-unique).
        dockerfile_path: Path to Dockerfile.
        context_dir: Build context directory.
        entrypoint: Command to start the LocalAPI server.
        entrypoint_mode: Harbor entrypoint mode ("command" or "file").
        description: Optional deployment description.
        env_vars: Optional environment variables (no LLM API keys).
        limits: HarborLimits or dict with resource limits.
        backend_url: Synth backend URL (defaults to SYNTH_BACKEND_URL).
        api_key: Synth API key (defaults to SYNTH_API_KEY).
        wait_for_ready: Whether to wait for build completion.
        build_timeout_s: Max wait time for build completion.

    Returns:
        LocalAPIDeployResult with deployment details and task_app_url.
    """
    spec = HarborBuildSpec(
        name=name,
        dockerfile_path=dockerfile_path,
        context_dir=context_dir,
        entrypoint=entrypoint,
        entrypoint_mode=entrypoint_mode,
        description=description,
        env_vars=env_vars or {},
        limits=limits or HarborLimits(),
        metadata={"localapi": True},
    )

    result = upload_harbor_deployment(
        spec,
        api_key=api_key,
        backend_url=backend_url,
        auto_build=True,
        wait_for_ready=wait_for_ready,
        build_timeout_s=build_timeout_s,
    )

    resolved_backend = backend_url or os.getenv("SYNTH_BACKEND_URL", "https://api.usesynth.ai")
    deployment_key = result.deployment_name or result.deployment_id
    task_app_url = f"{resolved_backend.rstrip('/')}/api/harbor/deployments/{deployment_key}"

    return LocalAPIDeployResult(
        deployment_id=result.deployment_id,
        deployment_name=deployment_key,
        task_app_url=task_app_url,
        status=result.status,
        build_id=result.build_id,
        snapshot_id=result.snapshot_id,
    )

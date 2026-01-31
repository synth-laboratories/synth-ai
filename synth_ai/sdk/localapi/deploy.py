"""LocalAPI deploy helpers (managed cloud or Harbor)."""

from __future__ import annotations

import io
import json
import os
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

try:
    import synth_ai_py as _rust_core  # type: ignore
except Exception:
    _rust_core = None

from synth_ai.sdk.harbor import HarborBuildSpec, HarborLimits, upload_harbor_deployment


@dataclass(slots=True)
class LocalAPIDeployResult:
    """Result of a LocalAPI deployment."""

    deployment_id: str
    deployment_name: str | None
    task_app_url: str
    status: str
    provider: str
    build_id: str | None = None
    snapshot_id: str | None = None


def _package_context(context_dir: str) -> bytes:
    context_path = Path(context_dir).resolve()
    if not context_path.exists():
        raise FileNotFoundError(f"Context dir not found: {context_dir}")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for path in context_path.rglob("*"):
            if path.is_file():
                tar.add(path, arcname=str(path.relative_to(context_path)))
    return buffer.getvalue()


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
    provider: str = "harbor",
    port: int = 8000,
    metadata: dict[str, object] | None = None,
) -> LocalAPIDeployResult:
    """Deploy a LocalAPI via the managed backend or Harbor and return task_app_url.

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
        provider: "harbor" (default) or "cloud" for managed deploy.
        port: Port your task app listens on (default 8000).
        metadata: Optional metadata dictionary stored with the deployment.

    Returns:
        LocalAPIDeployResult with deployment details and task_app_url. For managed deploys
        ("cloud"), task_app_url is the Synth backend proxy URL to use for eval/optimization
        jobs and health checks.
    """
    if provider == "harbor":
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
            provider="harbor",
            build_id=result.build_id,
            snapshot_id=result.snapshot_id,
        )

    if provider != "cloud":
        raise ValueError(f"Unknown provider: {provider}")

    if isinstance(limits, HarborLimits):
        limits_dict = limits.to_dict()
    elif limits is None:
        limits_dict = HarborLimits().to_dict()
    else:
        limits_dict = limits

    if _rust_core is not None and hasattr(_rust_core, "localapi_deploy_from_dir"):
        payload = _rust_core.localapi_deploy_from_dir(
            name=name,
            dockerfile_path=dockerfile_path,
            context_dir=context_dir,
            entrypoint=entrypoint,
            entrypoint_mode=entrypoint_mode,
            description=description,
            env_vars=env_vars or {},
            limits=limits_dict,
            backend_url=backend_url,
            api_key=api_key,
            wait_for_ready=wait_for_ready,
            build_timeout_s=build_timeout_s,
            port=port,
            metadata=metadata or {},
        )
    else:
        if not api_key:
            api_key = os.getenv("SYNTH_API_KEY")
        if not api_key:
            raise ValueError("SYNTH_API_KEY is required for deploy")

        resolved_backend = backend_url or os.getenv("SYNTH_BACKEND_URL", "https://api.usesynth.ai")

        spec_body = {
            "name": name,
            "dockerfile_path": dockerfile_path,
            "entrypoint": entrypoint,
            "entrypoint_mode": entrypoint_mode,
            "port": port,
            "description": description,
            "env_vars": env_vars or {},
            "limits": limits_dict,
            "metadata": metadata or {},
        }

        archive_bytes = _package_context(context_dir)
        files = {
            "spec_json": (None, json.dumps(spec_body), "application/json"),
            "context": ("context.tar.gz", archive_bytes, "application/gzip"),
        }
        headers = {"Authorization": f"Bearer {api_key}"}

        with httpx.Client(timeout=build_timeout_s) as client:
            response = client.post(
                f"{resolved_backend.rstrip('/')}/api/localapi/deployments",
                files=files,
                headers=headers,
            )
            response.raise_for_status()
            payload = response.json()

            if wait_for_ready:
                deployment_id = payload["deployment_id"]
                status_url = f"{resolved_backend.rstrip('/')}/api/localapi/deployments/{deployment_id}/status"
                deadline = time.time() + build_timeout_s
                while time.time() < deadline:
                    status_resp = client.get(status_url, headers=headers)
                    status_resp.raise_for_status()
                    status_body = status_resp.json()
                    status = status_body.get("status", payload.get("status", "building"))
                    payload["status"] = status
                    if status in ("ready", "failed"):
                        break
                    time.sleep(5)

    return LocalAPIDeployResult(
        deployment_id=payload["deployment_id"],
        deployment_name=None,
        task_app_url=payload["task_app_url"],
        status=payload.get("status", "building"),
        provider="cloud",
    )

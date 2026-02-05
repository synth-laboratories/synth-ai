from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from synth_ai.core.rust_core.http import RustCoreHttpClient
from synth_ai.core.rust_core.urls import ensure_api_base
from synth_ai.core.utils.env import get_api_key
from synth_ai.core.utils.urls import BACKEND_URL_BASE, normalize_backend_base


@dataclass(frozen=True)
class InferenceArtifactSpec:
    """Artifact selection for inference jobs.

    Use this to request a specific artifact type/format when the job completes.
    """

    type: str = "final_artifact"
    format: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"type": self.type}
        if self.format:
            payload["format"] = self.format
        return payload


@dataclass
class InferenceJobRequest:
    """Convenience container for inference job submissions.

    Provide either ``bundle_bytes`` or ``bundle_path``. Harbor/browser settings
    are passed through in their respective fields.
    """

    environment_type: str
    bundle_bytes: bytes | None = None
    bundle_path: str | Path | None = None
    filename: str | None = None
    content_type: str | None = None
    archive_type: str | None = None
    harbor: dict[str, Any] | None = None
    browser: dict[str, Any] | None = None
    policy: dict[str, Any] | None = None
    limits: dict[str, Any] | None = None
    container_overrides: dict[str, Any] | None = None
    artifact: InferenceArtifactSpec | dict[str, Any] | None = None

    def resolve_bundle(self) -> tuple[bytes, str | None]:
        if self.bundle_bytes is not None:
            return self.bundle_bytes, self.filename
        if self.bundle_path is not None:
            path = Path(self.bundle_path)
            data = path.read_bytes()
            return data, self.filename or path.name
        raise ValueError("bundle_bytes or bundle_path is required for inference jobs")


class InferenceJobsClient:
    """Client for creating and polling inference jobs.

    This targets the Rust backend inference job endpoints.
    """

    def __init__(self, base_url: str, api_key: str, *, timeout: float = 60.0) -> None:
        self._base_url = ensure_api_base(base_url).rstrip("/")
        self._api_key = api_key
        self._timeout = timeout

    async def create_job(
        self,
        *,
        environment_type: str,
        bundle_bytes: bytes,
        filename: str | None = None,
        content_type: str | None = None,
        archive_type: str | None = None,
        harbor: dict[str, Any] | None = None,
        browser: dict[str, Any] | None = None,
        policy: dict[str, Any] | None = None,
        limits: dict[str, Any] | None = None,
        container_overrides: dict[str, Any] | None = None,
        artifact: InferenceArtifactSpec | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create an inference job from an in-memory bundle.

        Provide ``environment_type`` (e.g., ``harbor`` or ``browser``) plus any
        environment-specific overrides in ``harbor`` or ``browser``.
        """
        body: dict[str, Any] = {
            "environment_type": environment_type,
            "upload": {
                "bytes_base64": base64.b64encode(bundle_bytes).decode("ascii"),
                "filename": filename,
                "content_type": content_type,
                "archive_type": archive_type,
            },
        }
        if harbor is not None:
            body["harbor"] = harbor
        if browser is not None:
            body["browser"] = browser
        if policy is not None:
            body["policy"] = policy
        if limits is not None:
            body["limits"] = limits
        if container_overrides is not None:
            body["container_overrides"] = container_overrides
        if artifact is not None:
            if isinstance(artifact, InferenceArtifactSpec):
                body["artifact"] = artifact.to_dict()
            else:
                body["artifact"] = dict(artifact)

        async with RustCoreHttpClient(self._base_url, self._api_key, timeout=self._timeout) as http:
            return await http.post_json("/api/inference/jobs", json=body)

    async def create_job_from_request(self, request: InferenceJobRequest) -> dict[str, Any]:
        """Create an inference job from an ``InferenceJobRequest``."""
        bundle_bytes, filename = request.resolve_bundle()
        return await self.create_job(
            environment_type=request.environment_type,
            bundle_bytes=bundle_bytes,
            filename=filename,
            content_type=request.content_type,
            archive_type=request.archive_type,
            harbor=request.harbor,
            browser=request.browser,
            policy=request.policy,
            limits=request.limits,
            container_overrides=request.container_overrides,
            artifact=request.artifact,
        )

    async def create_job_from_path(
        self,
        *,
        environment_type: str,
        bundle_path: str | Path,
        filename: str | None = None,
        content_type: str | None = None,
        archive_type: str | None = None,
        harbor: dict[str, Any] | None = None,
        browser: dict[str, Any] | None = None,
        policy: dict[str, Any] | None = None,
        limits: dict[str, Any] | None = None,
        container_overrides: dict[str, Any] | None = None,
        artifact: InferenceArtifactSpec | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create an inference job by reading a bundle from disk."""
        path = Path(bundle_path)
        bundle_bytes = path.read_bytes()
        return await self.create_job(
            environment_type=environment_type,
            bundle_bytes=bundle_bytes,
            filename=filename or path.name,
            content_type=content_type,
            archive_type=archive_type,
            harbor=harbor,
            browser=browser,
            policy=policy,
            limits=limits,
            container_overrides=container_overrides,
            artifact=artifact,
        )

    async def list_artifacts(
        self, job_id: str, *, artifact_type: str | None = None
    ) -> dict[str, Any]:
        """List artifacts produced by an inference job."""
        params = {"artifact_type": artifact_type} if artifact_type else None
        async with RustCoreHttpClient(self._base_url, self._api_key, timeout=self._timeout) as http:
            return await http.get(f"/api/jobs/{job_id}/artifacts", params=params)

    async def download_artifact(
        self, job_id: str, artifact_id: str, *, timeout: float | None = None
    ) -> bytes:
        """Download a single artifact produced by an inference job."""
        import httpx

        url = f"{self._base_url}/inference/jobs/{job_id}/artifacts/{artifact_id}"
        headers = {"Authorization": f"Bearer {self._api_key}"}
        async with httpx.AsyncClient(timeout=timeout or self._timeout) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            return response.content

    async def get_job(self, job_id: str) -> dict[str, Any]:
        """Fetch the current status of an inference job."""
        async with RustCoreHttpClient(self._base_url, self._api_key, timeout=self._timeout) as http:
            return await http.get(f"/api/inference/jobs/{job_id}")


def _resolve_base_url(base_url: str | None) -> str:
    if base_url and base_url.strip():
        return normalize_backend_base(base_url)
    return normalize_backend_base(BACKEND_URL_BASE)


def _resolve_api_key(api_key: str | None) -> str:
    if api_key and api_key.strip():
        return api_key
    try:
        resolved = get_api_key("SYNTH_API_KEY", required=True)
    except Exception:
        resolved = os.environ.get("SYNTH_API_KEY", "").strip()
    if not resolved:
        raise ValueError("api_key is required (provide or set SYNTH_API_KEY)")
    return resolved


async def create_inference_job(
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: float = 60.0,
    environment_type: str,
    bundle_bytes: bytes,
    filename: str | None = None,
    content_type: str | None = None,
    archive_type: str | None = None,
    harbor: dict[str, Any] | None = None,
    browser: dict[str, Any] | None = None,
    policy: dict[str, Any] | None = None,
    limits: dict[str, Any] | None = None,
    container_overrides: dict[str, Any] | None = None,
    artifact: InferenceArtifactSpec | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create an inference job with an in-memory bundle."""
    client = InferenceJobsClient(
        _resolve_base_url(base_url), _resolve_api_key(api_key), timeout=timeout
    )
    return await client.create_job(
        environment_type=environment_type,
        bundle_bytes=bundle_bytes,
        filename=filename,
        content_type=content_type,
        archive_type=archive_type,
        harbor=harbor,
        browser=browser,
        policy=policy,
        limits=limits,
        container_overrides=container_overrides,
        artifact=artifact,
    )


async def create_inference_job_from_path(
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: float = 60.0,
    environment_type: str,
    bundle_path: str | Path,
    filename: str | None = None,
    content_type: str | None = None,
    archive_type: str | None = None,
    harbor: dict[str, Any] | None = None,
    browser: dict[str, Any] | None = None,
    policy: dict[str, Any] | None = None,
    limits: dict[str, Any] | None = None,
    container_overrides: dict[str, Any] | None = None,
    artifact: InferenceArtifactSpec | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create an inference job with a bundle loaded from disk."""
    client = InferenceJobsClient(
        _resolve_base_url(base_url), _resolve_api_key(api_key), timeout=timeout
    )
    return await client.create_job_from_path(
        environment_type=environment_type,
        bundle_path=bundle_path,
        filename=filename,
        content_type=content_type,
        archive_type=archive_type,
        harbor=harbor,
        browser=browser,
        policy=policy,
        limits=limits,
        container_overrides=container_overrides,
        artifact=artifact,
    )


async def get_inference_job(
    job_id: str,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: float = 60.0,
) -> dict[str, Any]:
    client = InferenceJobsClient(
        _resolve_base_url(base_url), _resolve_api_key(api_key), timeout=timeout
    )
    return await client.get_job(job_id)


async def download_inference_artifact(
    job_id: str,
    artifact_id: str,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: float = 60.0,
) -> bytes:
    client = InferenceJobsClient(
        _resolve_base_url(base_url), _resolve_api_key(api_key), timeout=timeout
    )
    return await client.download_artifact(job_id, artifact_id)


__all__ = [
    "InferenceArtifactSpec",
    "InferenceJobRequest",
    "InferenceJobsClient",
    "create_inference_job",
    "create_inference_job_from_path",
    "get_inference_job",
    "download_inference_artifact",
]

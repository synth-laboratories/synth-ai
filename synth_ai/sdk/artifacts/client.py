"""HTTP client for artifacts API endpoints."""

import logging
from typing import Any, Dict, List, Optional

from synth_ai.core.http import AsyncHttpClient
from synth_ai.core.urls import (
    synth_api_v1_base,
    synth_artifact_model_url,
    synth_artifact_prompt_url,
    synth_artifacts_url,
    synth_learning_exports_hf_url,
    synth_learning_models_on_wasabi_url,
    synth_prompt_learning_artifacts_url,
    synth_prompt_learning_snapshot_url,
)

logger = logging.getLogger(__name__)


class ArtifactsClient:
    """Client for artifacts API endpoints.

    Provides methods to interact with the Synth AI artifacts API, including
    listing artifacts, retrieving model and prompt details, exporting models
    to HuggingFace, and accessing prompt snapshots.
    """

    def __init__(
        self,
        synth_user_key: str | None = None,
        *,
        timeout: float = 30.0,
        synth_base_url: str | None = None,
    ) -> None:
        """Initialize the artifacts client.

        Args:
            synth_user_key: Synth API key for authentication
            timeout: Request timeout in seconds (default: 30.0)
            synth_base_url: Backend URL override (defaults to SYNTH_BACKEND_URL or production)
        """
        self._synth_base_url = synth_base_url
        if synth_user_key is None:
            raise ValueError("synth_user_key is required")
        self._synth_user_key = synth_user_key
        self._timeout = timeout

    async def list_artifacts(
        self,
        *,
        artifact_type: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """List all artifacts (models and prompts)."""
        params: Dict[str, Any] = {"limit": limit}
        if artifact_type:
            params["type"] = artifact_type
        if status:
            params["status"] = status

        async with AsyncHttpClient(
            synth_api_v1_base(self._synth_base_url), self._synth_user_key, timeout=self._timeout
        ) as http:
            return await http.get(synth_artifacts_url(self._synth_base_url), params=params)

    async def get_model(self, model_id: str) -> Dict[str, Any]:
        """Get detailed information about a model artifact."""
        async with AsyncHttpClient(
            synth_api_v1_base(self._synth_base_url), self._synth_user_key, timeout=self._timeout
        ) as http:
            return await http.get(synth_artifact_model_url(model_id, self._synth_base_url))

    async def get_prompt(self, job_id: str) -> Dict[str, Any]:
        """Get detailed information about a prompt optimization job."""
        async with AsyncHttpClient(
            synth_api_v1_base(self._synth_base_url), self._synth_user_key, timeout=self._timeout
        ) as http:
            return await http.get(synth_artifact_prompt_url(job_id, self._synth_base_url))

    async def export_to_huggingface(
        self,
        *,
        wasabi_key: str,
        repo_id: str,
        repo_type: str = "model",
        artifact_kind: str = "lora",
        base_model: Optional[str] = None,
        visibility: str = "private",
        tags: Optional[List[str]] = None,
        bucket: Optional[str] = None,
        folder_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Export a model artifact to HuggingFace Hub."""
        body: Dict[str, Any] = {
            "repo_id": repo_id,
            "repo_type": repo_type,
            "artifact_kind": artifact_kind,
            "key": wasabi_key,
            "visibility": visibility,
        }
        if base_model:
            body["base_model"] = base_model
        if tags:
            body["tags"] = tags
        if bucket:
            body["bucket"] = bucket
        if folder_name:
            body["folder_name"] = folder_name

        async with AsyncHttpClient(
            synth_api_v1_base(self._synth_base_url), self._synth_user_key, timeout=self._timeout
        ) as http:
            return await http.post_json(
                synth_learning_exports_hf_url(self._synth_base_url), json=body
            )

    async def get_prompt_snapshot(
        self,
        job_id: str,
        snapshot_id: str,
    ) -> Dict[str, Any]:
        """Get a specific prompt snapshot."""
        async with AsyncHttpClient(
            synth_api_v1_base(self._synth_base_url), self._synth_user_key, timeout=self._timeout
        ) as http:
            return await http.get(
                synth_prompt_learning_snapshot_url(job_id, snapshot_id, self._synth_base_url)
            )

    async def list_prompt_snapshots(
        self,
        job_id: str,
    ) -> List[Dict[str, Any]]:
        """List all artifacts (snapshots) for a prompt job."""
        async with AsyncHttpClient(
            synth_api_v1_base(self._synth_base_url), self._synth_user_key, timeout=self._timeout
        ) as http:
            artifacts = await http.get(
                synth_prompt_learning_artifacts_url(job_id, self._synth_base_url)
            )
            if isinstance(artifacts, list):
                return artifacts
            if isinstance(artifacts, dict) and isinstance(artifacts.get("artifacts"), list):
                return artifacts["artifacts"]
            return []

    async def get_models_on_wasabi(self) -> Dict[str, Any]:
        """Get models available on Wasabi."""
        async with AsyncHttpClient(
            synth_api_v1_base(self._synth_base_url), self._synth_user_key, timeout=self._timeout
        ) as http:
            return await http.get(synth_learning_models_on_wasabi_url(self._synth_base_url))
